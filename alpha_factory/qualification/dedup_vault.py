"""Local PnL correlation deduplication vault and cluster reconciliation."""

from __future__ import annotations

import json
import logging
import math
import threading
from typing import Any, Mapping, Optional

import numpy as np

from alpha_factory.brain.client import BrainClient
from alpha_factory.storage.db import AlphaDB
from alpha_factory.storage.models import DEFAULT_SETTINGS, SimulationRecord, VaultEntry
from alpha_factory.storage.pnl_store import PnlStore, extract_daily_returns_from_recordset

logger = logging.getLogger(__name__)

_FLOAT_EPS = 1e-12


def _opt_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _has_pyramid_matches(sim: SimulationRecord) -> bool:
    raw = getattr(sim, "pyramid_matches_json", "[]")
    if not raw or raw == "[]":
        return False
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return False
    if isinstance(parsed, (list, tuple, set, dict)):
        return len(parsed) > 0
    return bool(parsed)


def compute_composite_score(
    sim: SimulationRecord,
    has_pyramid_match: Optional[bool] = None,
) -> float:
    """Compute composite ranking score for a qualified simulation record."""
    fitness = 0.0 if sim.fitness is None else float(sim.fitness)
    sharpe = 0.0 if sim.sharpe is None else float(sim.sharpe)
    prod_corr = 0.0 if sim.prod_corr_max is None else float(sim.prod_corr_max)
    turnover = 0.0 if sim.turnover is None else float(sim.turnover)
    pyramid_match = (
        bool(has_pyramid_match)
        if has_pyramid_match is not None
        else _has_pyramid_matches(sim)
    )
    return (
        1.0 * fitness
        + 0.5 * sharpe
        - 1.5 * prod_corr
        - 0.5 * turnover
        + 0.5 * (1.0 if pyramid_match else 0.0)
    )


def _abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Compute ``|Pearson(x, y)|`` safely, returning ``0.0`` when variance is zero."""
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return 0.0
    dx = x - np.mean(x)
    dy = y - np.mean(y)
    norm_x = float(np.sqrt(np.dot(dx, dx)))
    norm_y = float(np.sqrt(np.dot(dy, dy)))
    if (
        norm_x == 0.0
        or norm_y == 0.0
        or not math.isfinite(norm_x)
        or not math.isfinite(norm_y)
    ):
        return 0.0
    corr = float(np.dot(dx, dy) / (norm_x * norm_y))
    if not math.isfinite(corr):
        return 0.0
    return min(1.0, abs(corr))


def pairwise_effective_corr(
    dates_a: np.ndarray,
    ret_a: np.ndarray,
    dates_b: np.ndarray,
    ret_b: np.ndarray,
    min_overlap: int = 120,
    trailing_days: int = 504,
) -> float:
    """Return ``max(|corr_full|, |corr_trailing|)`` over common dates.

    Returns ``0.0`` when the overlapping finite window is shorter than ``min_overlap``
    or when either series has zero variance on the window.
    """
    d_a = np.asarray(dates_a)
    d_b = np.asarray(dates_b)
    r_a = np.asarray(ret_a, dtype=np.float64)
    r_b = np.asarray(ret_b, dtype=np.float64)

    if d_a.ndim != 1 or r_a.ndim != 1 or d_a.shape != r_a.shape:
        raise ValueError("dates_a and ret_a must be 1D arrays of equal length")
    if d_b.ndim != 1 or r_b.ndim != 1 or d_b.shape != r_b.shape:
        raise ValueError("dates_b and ret_b must be 1D arrays of equal length")

    if d_a.size == 0 or d_b.size == 0:
        return 0.0

    _, idx_a, idx_b = np.intersect1d(d_a, d_b, return_indices=True)
    aligned_a = r_a[idx_a]
    aligned_b = r_b[idx_b]

    finite_mask = np.isfinite(aligned_a) & np.isfinite(aligned_b)
    if not np.all(finite_mask):
        aligned_a = aligned_a[finite_mask]
        aligned_b = aligned_b[finite_mask]

    effective_min = max(2, int(min_overlap))
    if aligned_a.size < effective_min:
        return 0.0

    corr_full = _abs_pearson(aligned_a, aligned_b)
    corr_tail = 0.0
    if trailing_days > 0:
        tail_a = aligned_a[-int(trailing_days) :]
        tail_b = aligned_b[-int(trailing_days) :]
        if tail_a.size >= effective_min:
            corr_tail = _abs_pearson(tail_a, tail_b)

    return max(corr_full, corr_tail)


class CorrelationDedupVault:
    """Thread-safe local PnL correlation vault enforcing per-(region, delay) dedup."""

    def __init__(
        self,
        db: AlphaDB,
        pnl_store: PnlStore,
        client: Optional[BrainClient] = None,
        dedup_cfg: Optional[Mapping[str, Any]] = None,
        tag_on_brain: bool = True,
    ) -> None:
        self.db = db
        self.pnl_store = pnl_store
        self.client = client
        self.tag_on_brain = bool(tag_on_brain)
        self._lock = threading.RLock()

        defaults = DEFAULT_SETTINGS["dedup"]
        raw_cfg = dict(dedup_cfg or {})
        if isinstance(raw_cfg.get("dedup"), Mapping):
            cfg = dict(raw_cfg["dedup"])
        else:
            cfg = raw_cfg

        self.corr_threshold = float(
            cfg.get(
                "local_corr_threshold",
                cfg.get("corr_threshold", defaults["local_corr_threshold"]),
            )
        )
        self.min_overlap_days = int(
            cfg.get("min_overlap_days", cfg.get("min_overlap", defaults["min_overlap_days"]))
        )
        self.trailing_window_days = int(
            cfg.get(
                "trailing_window_days",
                cfg.get("trailing_days", defaults["trailing_window_days"]),
            )
        )
        self.replacement_hysteresis = float(
            cfg.get("replacement_hysteresis", defaults["replacement_hysteresis"])
        )

    def _tag_alpha(self, alpha_id: str, color: str) -> None:
        if not self.tag_on_brain or self.client is None:
            return
        try:
            self.client.patch_alpha_properties(alpha_id, color=color)
        except Exception as exc:
            logger.warning(
                "Failed tagging alpha %s with color=%s on BRAIN: %s",
                alpha_id,
                color,
                exc,
            )

    def _fetch_and_store_pnl(self, alpha_id: str, region: str) -> bool:
        if self.client is None:
            return False
        raw_recordset: Optional[dict[str, Any]] = None
        used_rec_name = "pnl"
        for rec_name in ("pnl", "daily-pnl"):
            try:
                raw_recordset = self.client.fetch_recordset(alpha_id, rec_name)
                if raw_recordset:
                    used_rec_name = rec_name
                    break
            except Exception as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if rec_name == "pnl" and status_code == 404:
                    continue
                logger.warning(
                    "Failed fetching %s recordset for submitted alpha %s: %s",
                    rec_name,
                    alpha_id,
                    exc,
                )
                return False

        if not raw_recordset:
            return False

        rs_payload: dict[str, Any] = dict(raw_recordset)
        if (
            used_rec_name == "daily-pnl"
            and rs_payload.get("name") is None
            and not (isinstance(rs_payload.get("schema"), Mapping) and rs_payload["schema"].get("name"))
        ):
            rs_payload["recordset_name"] = "daily-pnl"

        try:
            dates, daily_returns = extract_daily_returns_from_recordset(rs_payload)
        except Exception as exc:
            logger.warning("Failed parsing PnL recordset for alpha %s: %s", alpha_id, exc)
            return False

        if dates.size == 0:
            return False
        self.pnl_store.save(alpha_id, region, dates, daily_returns)
        return True

    def sync_submitted_alphas(
        self,
        region: Optional[str] = None,
        page_limit: int = 50,
        force_refresh_pnl: bool = False,
    ) -> int:
        """Sync live ``ACTIVE`` alphas from BRAIN into ``pnl_store`` and ``submittable_vault``."""
        if self.client is None:
            logger.warning("Cannot sync submitted alphas without a BrainClient instance.")
            return 0

        limit = max(1, int(page_limit))
        offset = 0
        synced = 0
        seen_ids: set[str] = set()

        with self._lock:
            while True:
                list_kwargs: dict[str, Any] = {
                    "status": "ACTIVE",
                    "limit": limit,
                    "offset": offset,
                }
                if region is not None:
                    list_kwargs["region"] = region
                page = self.client.list_user_alphas(**list_kwargs)
                if not isinstance(page, Mapping):
                    break

                results = page.get("results") or []
                if not results:
                    break

                page_ids: list[str] = []
                for item in results:
                    if not isinstance(item, Mapping):
                        continue
                    alpha_id = str(item.get("id") or item.get("alpha_id") or "").strip()
                    if not alpha_id:
                        continue
                    page_ids.append(alpha_id)
                    if alpha_id in seen_ids:
                        continue
                    seen_ids.add(alpha_id)

                    settings = item.get("settings")
                    if not isinstance(settings, Mapping):
                        settings = {}
                    alpha_region = str(settings.get("region") or item.get("region") or region or "USA")
                    raw_delay = settings.get("delay", item.get("delay", 1))
                    alpha_delay = int(1 if raw_delay is None else raw_delay)

                    regular = item.get("regular")
                    if isinstance(regular, Mapping):
                        expression = str(regular.get("code") or "")
                    elif isinstance(regular, str):
                        expression = regular
                    else:
                        expression = str(item.get("expression") or "")

                    is_data = item.get("is")
                    if not isinstance(is_data, Mapping):
                        is_data = {}

                    existing_sim = self.db.get_simulation(alpha_id)
                    if not expression and existing_sim is not None:
                        expression = existing_sim.expression

                    sharpe = _opt_float(is_data.get("sharpe", item.get("sharpe")))
                    fitness = _opt_float(is_data.get("fitness", item.get("fitness")))
                    turnover = _opt_float(is_data.get("turnover", item.get("turnover")))
                    prod_corr_max = _opt_float(
                        item.get("prod_corr_max", is_data.get("prodCorrMax"))
                    )
                    if prod_corr_max is None and existing_sim is not None:
                        prod_corr_max = existing_sim.prod_corr_max

                    pyramids = item.get("pyramids")
                    has_pyr: Optional[bool]
                    if isinstance(pyramids, list):
                        has_pyr = len(pyramids) > 0
                    elif existing_sim is not None:
                        has_pyr = None
                    else:
                        has_pyr = False

                    sim_rec = SimulationRecord(
                        alpha_id=alpha_id,
                        expression=expression,
                        region=alpha_region,
                        delay=alpha_delay,
                        sharpe=sharpe,
                        fitness=fitness,
                        turnover=turnover,
                        prod_corr_max=prod_corr_max,
                        pyramid_matches_json=(
                            existing_sim.pyramid_matches_json if existing_sim is not None else "[]"
                        ),
                    )
                    score = compute_composite_score(sim_rec, has_pyramid_match=has_pyr)

                    if force_refresh_pnl or not self.pnl_store.has(alpha_id, alpha_region):
                        self._fetch_and_store_pnl(alpha_id, alpha_region)

                    self.db.upsert_vault_entry(
                        VaultEntry(
                            alpha_id=alpha_id,
                            region=alpha_region,
                            delay=alpha_delay,
                            expression=expression,
                            composite_score=score,
                            vault_status="SUBMITTED",
                            duplicate_of=None,
                            max_local_corr=None,
                            max_local_corr_peer=None,
                            sharpe=sharpe,
                            fitness=fitness,
                            turnover=turnover,
                            prod_corr_max=prod_corr_max,
                        )
                    )
                    synced += 1

                if not page_ids:
                    break
                offset += len(results)
                total_count = page.get("count")
                if total_count is not None:
                    if offset >= int(total_count):
                        break
                elif "next" in page:
                    if not page.get("next"):
                        break
                elif len(results) < limit:
                    break

        return synced

    @staticmethod
    def _build_entry(
        sim: SimulationRecord,
        score: float,
        vault_status: str,
        duplicate_of: Optional[str],
        max_local_corr: float,
        max_local_corr_peer: Optional[str],
    ) -> VaultEntry:
        return VaultEntry(
            alpha_id=sim.alpha_id,
            region=sim.region,
            delay=int(sim.delay),
            expression=sim.expression,
            composite_score=float(score),
            vault_status=vault_status,
            duplicate_of=duplicate_of,
            max_local_corr=float(max_local_corr),
            max_local_corr_peer=max_local_corr_peer,
            sharpe=sim.sharpe,
            fitness=sim.fitness,
            turnover=sim.turnover,
            prod_corr_max=sim.prod_corr_max,
        )

    def _peer_correlation(
        self,
        dates: np.ndarray,
        daily_returns: np.ndarray,
        peer_id: str,
        region: str,
    ) -> float:
        peer_pnl = self.pnl_store.load(peer_id, region)
        if peer_pnl is None:
            return 0.0
        peer_dates, peer_returns = peer_pnl
        return pairwise_effective_corr(
            dates,
            daily_returns,
            peer_dates,
            peer_returns,
            min_overlap=self.min_overlap_days,
            trailing_days=self.trailing_window_days,
        )

    def admit(
        self,
        sim: SimulationRecord,
        dates: np.ndarray,
        daily_returns: np.ndarray,
        has_pyramid_match: Optional[bool] = None,
    ) -> VaultEntry:
        """Admit a qualified alpha into the vault using local PnL dedup & replacement rules."""
        with self._lock:
            self.pnl_store.save(sim.alpha_id, sim.region, dates, daily_returns)
            new_score = compute_composite_score(sim, has_pyramid_match=has_pyramid_match)

            peers = self.db.get_vault_entries(
                region=sim.region,
                delay=int(sim.delay),
                statuses=("SUBMITTED", "KEPT"),
            )

            max_corr = 0.0
            max_corr_peer: Optional[str] = None
            corr_submitted: list[tuple[VaultEntry, float]] = []
            corr_kept: list[tuple[VaultEntry, float]] = []

            for peer in peers:
                if peer.alpha_id == sim.alpha_id:
                    continue
                corr = self._peer_correlation(dates, daily_returns, peer.alpha_id, sim.region)
                if corr > max_corr:
                    max_corr = corr
                    max_corr_peer = peer.alpha_id

                if corr + _FLOAT_EPS >= self.corr_threshold:
                    if peer.vault_status == "SUBMITTED":
                        corr_submitted.append((peer, corr))
                    else:
                        corr_kept.append((peer, corr))

            # Rule 1: Correlated with any SUBMITTED alpha -> permanent CORR_DUP
            if corr_submitted:
                best_submitted, _ = max(
                    corr_submitted,
                    key=lambda item: (float(item[0].composite_score), item[1]),
                )
                entry = self._build_entry(
                    sim=sim,
                    score=new_score,
                    vault_status="CORR_DUP",
                    duplicate_of=best_submitted.alpha_id,
                    max_local_corr=max_corr,
                    max_local_corr_peer=max_corr_peer,
                )
                self.db.upsert_vault_entry(entry)
                self._tag_alpha(sim.alpha_id, "YELLOW")
                return entry

            # Rule 2: Correlated with KEPT peer(s) -> replace iff score beats max + hysteresis
            if corr_kept:
                best_kept, _ = max(
                    corr_kept,
                    key=lambda item: (float(item[0].composite_score), item[1]),
                )
                best_kept_score = float(best_kept.composite_score)

                if new_score + _FLOAT_EPS >= best_kept_score + self.replacement_hysteresis:
                    entry = self._build_entry(
                        sim=sim,
                        score=new_score,
                        vault_status="KEPT",
                        duplicate_of=None,
                        max_local_corr=max_corr,
                        max_local_corr_peer=max_corr_peer,
                    )
                    self.db.upsert_vault_entry(entry)
                    self._tag_alpha(sim.alpha_id, "GREEN")

                    for peer, peer_corr in corr_kept:
                        self.db.demote_vault_entry(
                            peer.alpha_id,
                            duplicate_of=sim.alpha_id,
                            max_local_corr=peer_corr,
                            peer_id=sim.alpha_id,
                        )
                        self._tag_alpha(peer.alpha_id, "YELLOW")
                    return entry

                entry = self._build_entry(
                    sim=sim,
                    score=new_score,
                    vault_status="CORR_DUP",
                    duplicate_of=best_kept.alpha_id,
                    max_local_corr=max_corr,
                    max_local_corr_peer=max_corr_peer,
                )
                self.db.upsert_vault_entry(entry)
                self._tag_alpha(sim.alpha_id, "YELLOW")
                return entry

            # Rule 3: Uncorrelated with all SUBMITTED and KEPT peers -> KEPT
            entry = self._build_entry(
                sim=sim,
                score=new_score,
                vault_status="KEPT",
                duplicate_of=None,
                max_local_corr=max_corr,
                max_local_corr_peer=max_corr_peer,
            )
            self.db.upsert_vault_entry(entry)
            self._tag_alpha(sim.alpha_id, "GREEN")
            return entry

    def reconcile_clusters(self, region: str, delay: int) -> dict[str, int]:
        """Re-cluster all non-rejected vault entries for ``(region, delay)`` greedily by score."""
        with self._lock:
            entries = self.db.get_vault_entries(
                region=region,
                delay=int(delay),
                statuses=("SUBMITTED", "KEPT", "CORR_DUP", "DUPLICATE"),
            )
            if not entries:
                return {"kept": 0, "demoted": 0, "promoted": 0}

            submitted_entries = [e for e in entries if e.vault_status == "SUBMITTED"]
            candidates = sorted(
                (e for e in entries if e.vault_status != "SUBMITTED"),
                key=lambda e: float(e.composite_score),
                reverse=True,
            )

            pnl_cache: dict[str, Optional[tuple[np.ndarray, np.ndarray]]] = {
                entry.alpha_id: self.pnl_store.load(entry.alpha_id, region)
                for entry in entries
            }

            kept_winners: list[VaultEntry] = []
            kept_count = 0
            demoted_count = 0
            promoted_count = 0

            for cand in candidates:
                cand_pnl = pnl_cache.get(cand.alpha_id)
                max_corr = 0.0
                max_corr_peer: Optional[str] = None
                corr_submitted: list[tuple[VaultEntry, float]] = []
                corr_kept: list[tuple[VaultEntry, float]] = []

                if cand_pnl is not None:
                    c_dates, c_rets = cand_pnl
                    for sub in submitted_entries:
                        sub_pnl = pnl_cache.get(sub.alpha_id)
                        if sub_pnl is None:
                            continue
                        corr = pairwise_effective_corr(
                            c_dates,
                            c_rets,
                            sub_pnl[0],
                            sub_pnl[1],
                            min_overlap=self.min_overlap_days,
                            trailing_days=self.trailing_window_days,
                        )
                        if corr > max_corr:
                            max_corr = corr
                            max_corr_peer = sub.alpha_id
                        if corr + _FLOAT_EPS >= self.corr_threshold:
                            corr_submitted.append((sub, corr))

                    for winner in kept_winners:
                        w_pnl = pnl_cache.get(winner.alpha_id)
                        if w_pnl is None:
                            continue
                        corr = pairwise_effective_corr(
                            c_dates,
                            c_rets,
                            w_pnl[0],
                            w_pnl[1],
                            min_overlap=self.min_overlap_days,
                            trailing_days=self.trailing_window_days,
                        )
                        if corr > max_corr:
                            max_corr = corr
                            max_corr_peer = winner.alpha_id
                        if corr + _FLOAT_EPS >= self.corr_threshold:
                            corr_kept.append((winner, corr))

                if corr_submitted:
                    best_sub, _ = max(
                        corr_submitted,
                        key=lambda item: (float(item[0].composite_score), item[1]),
                    )
                    target_status = "CORR_DUP"
                    target_dup: Optional[str] = best_sub.alpha_id
                elif corr_kept:
                    best_winner, _ = max(
                        corr_kept,
                        key=lambda item: (float(item[0].composite_score), item[1]),
                    )
                    target_status = "CORR_DUP"
                    target_dup = best_winner.alpha_id
                else:
                    target_status = "KEPT"
                    target_dup = None
                    kept_winners.append(cand)

                prev_status = cand.vault_status
                if target_status == "KEPT":
                    kept_count += 1
                    if prev_status in ("CORR_DUP", "DUPLICATE"):
                        promoted_count += 1
                else:
                    if prev_status == "KEPT":
                        demoted_count += 1

                prev_corr = 0.0 if cand.max_local_corr is None else float(cand.max_local_corr)
                changed = (
                    prev_status != target_status
                    or (cand.duplicate_of or None) != target_dup
                    or abs(prev_corr - max_corr) > 1e-12
                    or (cand.max_local_corr_peer or None) != max_corr_peer
                )
                if changed:
                    updated = VaultEntry(
                        alpha_id=cand.alpha_id,
                        region=cand.region,
                        delay=int(cand.delay),
                        expression=cand.expression,
                        composite_score=float(cand.composite_score),
                        vault_status=target_status,
                        duplicate_of=target_dup,
                        max_local_corr=float(max_corr),
                        max_local_corr_peer=max_corr_peer,
                        sharpe=cand.sharpe,
                        fitness=cand.fitness,
                        turnover=cand.turnover,
                        prod_corr_max=cand.prod_corr_max,
                    )
                    self.db.upsert_vault_entry(updated)

                if target_status == "KEPT" and prev_status != "KEPT":
                    self._tag_alpha(cand.alpha_id, "GREEN")
                elif target_status == "CORR_DUP" and prev_status == "KEPT":
                    self._tag_alpha(cand.alpha_id, "YELLOW")

            return {
                "kept": kept_count,
                "demoted": demoted_count,
                "promoted": promoted_count,
            }


__all__ = [
    "CorrelationDedupVault",
    "compute_composite_score",
    "pairwise_effective_corr",
]
