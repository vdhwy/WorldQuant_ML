"""Thompson Sampling multi-armed bandit scheduler for alpha mining allocation."""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Mapping, Optional, Sequence

from alpha_factory.storage.db import AlphaDB

logger = logging.getLogger(__name__)

_DEFAULT_CATEGORIES: tuple[str, ...] = (
    "fundamental",
    "pv",
    "analyst",
    "sentiment",
)
_PREFERRED_NEUTRALIZATIONS: tuple[str, ...] = (
    "INDUSTRY",
    "SUBINDUSTRY",
    "SECTOR",
    "MARKET",
    "REVERSION_AND_MOMENTUM",
)
_DEFAULT_MIX: dict[str, float] = {
    "template": 0.40,
    "algorithmic_llm": 0.35,
    "evolution": 0.25,
}


def format_arm_key(
    region: str,
    universe: str,
    category: str,
    generator_type: str,
    neutralization: str,
    decay: int,
) -> str:
    """Return canonical string key for a 6-tuple bandit arm."""
    return f"{region}|{universe}|{category}|{generator_type}|{neutralization}|{int(decay)}"


def parse_arm_key(arm_key: str) -> dict[str, Any]:
    """Parse canonical arm_key string back into arm dimension fields."""
    parts = arm_key.split("|")
    if len(parts) != 6:
        raise ValueError(f"Expected 6-part arm_key, got {arm_key!r}")
    return {
        "arm_key": arm_key,
        "region": parts[0],
        "universe": parts[1],
        "category": parts[2],
        "generator_type": parts[3],
        "neutralization": parts[4],
        "decay": int(parts[5]),
    }


def _select_neutralizations(mkt_cfg: Mapping[str, Any], override: Optional[Sequence[str]]) -> list[str]:
    available = [str(n) for n in (mkt_cfg.get("neutralizations") or []) if str(n).strip()]
    default_neut = str(mkt_cfg.get("default_neutralization") or "INDUSTRY")
    if override:
        filtered = [str(n) for n in override if not available or str(n) in available]
        if filtered:
            return list(dict.fromkeys(filtered))

    if not available:
        return [default_neut]

    selected: list[str] = []
    if default_neut in available:
        selected.append(default_neut)
    for pref in _PREFERRED_NEUTRALIZATIONS:
        if pref in available and pref not in selected:
            selected.append(pref)
        if len(selected) >= 3:
            break
    if len(selected) < 3:
        for neut in available:
            if neut not in selected:
                selected.append(neut)
            if len(selected) >= 3:
                break
    return selected


def _select_decays(settings: Mapping[str, Any]) -> list[int]:
    gen_cfg = settings.get("generation") or {}
    raw_decays = gen_cfg.get("decays")
    if isinstance(raw_decays, Sequence) and not isinstance(raw_decays, (str, bytes)) and raw_decays:
        return list(dict.fromkeys(int(d) for d in raw_decays))

    exec_defaults = (settings.get("execution") or {}).get("default_settings") or {}
    default_decay = int(exec_defaults.get("decay", 4))
    candidates = [default_decay, 0] if default_decay != 0 else [0, 4]
    return list(dict.fromkeys(candidates))


class ThompsonBanditScheduler:
    """Beta-Bernoulli Thompson Sampling scheduler over generation & market arms."""

    def __init__(
        self,
        db: AlphaDB,
        settings: Mapping[str, Any],
        exploration_floor: float = 0.20,
        seed: Optional[int] = None,
    ) -> None:
        self.db = db
        self.settings = dict(settings)
        self.exploration_floor = max(0.0, min(1.0, float(exploration_floor)))
        self._rng = random.Random(seed)
        self._lock = threading.RLock()
        self._arms: dict[str, dict[str, Any]] = self._build_arm_catalog()

    def _build_arm_catalog(self) -> dict[str, dict[str, Any]]:
        markets = self.settings.get("markets") or {}
        gen_cfg = self.settings.get("generation") or {}

        raw_mix = gen_cfg.get("mix") or _DEFAULT_MIX
        generator_types = [str(k) for k, v in raw_mix.items() if float(v) > 0.0]
        if not generator_types:
            generator_types = ["template"]

        raw_cats = gen_cfg.get("categories")
        if isinstance(raw_cats, Sequence) and not isinstance(raw_cats, (str, bytes)) and raw_cats:
            categories = [str(c) for c in raw_cats]
        else:
            categories = list(_DEFAULT_CATEGORIES)

        neut_override = gen_cfg.get("neutralizations")
        decays = _select_decays(self.settings)

        arms: dict[str, dict[str, Any]] = {}
        for region, mkt_cfg in markets.items():
            if not isinstance(mkt_cfg, Mapping):
                continue
            delay = int(mkt_cfg.get("delay", 1))
            default_univ = str(mkt_cfg.get("default_universe") or "TOP3000")
            raw_univs = mkt_cfg.get("universes") or [default_univ]
            universes = list(dict.fromkeys(str(u) for u in raw_univs))
            if default_univ and default_univ not in universes:
                universes.insert(0, default_univ)
            neutralizations = _select_neutralizations(mkt_cfg, neut_override)

            for universe in universes:
                for category in categories:
                    for gen_type in generator_types:
                        for neutralization in neutralizations:
                            for decay in decays:
                                key = format_arm_key(
                                    region=str(region),
                                    universe=universe,
                                    category=category,
                                    generator_type=gen_type,
                                    neutralization=neutralization,
                                    decay=decay,
                                )
                                arms[key] = {
                                    "arm_key": key,
                                    "region": str(region),
                                    "universe": universe,
                                    "delay": delay,
                                    "category": category,
                                    "generator_type": gen_type,
                                    "neutralization": neutralization,
                                    "decay": int(decay),
                                    "alpha_param": 1.0,
                                    "beta_param": 1.0,
                                    "pulls": 0,
                                    "total_reward": 0.0,
                                    "cooldown_until": 0.0,
                                    "mix_weight": float(raw_mix.get(gen_type, 1.0)),
                                }
        return arms

    def _load_merged_arms(self) -> list[dict[str, Any]]:
        persisted = {row["arm_key"]: row for row in self.db.get_bandit_arms()}
        merged: list[dict[str, Any]] = []
        for arm_key, base in self._arms.items():
            item = dict(base)
            row = persisted.get(arm_key)
            if row is not None:
                item["alpha_param"] = float(row.get("alpha_param", 1.0))
                item["beta_param"] = float(row.get("beta_param", 1.0))
                item["pulls"] = int(row.get("pulls", 0))
                item["total_reward"] = float(row.get("total_reward", 0.0))
                item["cooldown_until"] = float(row.get("cooldown_until", 0.0))
            merged.append(item)
        return merged

    def list_arms(self) -> list[dict[str, Any]]:
        """Return all configured bandit arms merged with persisted SQLite posteriors."""
        with self._lock:
            return self._load_merged_arms()

    def get_arm(self, arm_key: str) -> Optional[dict[str, Any]]:
        """Return a single arm snapshot by arm_key, or None if not defined."""
        with self._lock:
            if arm_key not in self._arms:
                return None
            persisted = {row["arm_key"]: row for row in self.db.get_bandit_arms()}
            item = dict(self._arms[arm_key])
            row = persisted.get(arm_key)
            if row is not None:
                item["alpha_param"] = float(row.get("alpha_param", 1.0))
                item["beta_param"] = float(row.get("beta_param", 1.0))
                item["pulls"] = int(row.get("pulls", 0))
                item["total_reward"] = float(row.get("total_reward", 0.0))
                item["cooldown_until"] = float(row.get("cooldown_until", 0.0))
            return item

    def select_arm(
        self,
        region_cooldowns: Optional[Mapping[str, float]] = None,
        region: Optional[str] = None,
    ) -> dict[str, Any]:
        """Select a generation arm via Thompson Sampling with exploration floor & cooldown skipping."""
        with self._lock:
            arms = self._load_merged_arms()
            if not arms:
                raise RuntimeError("No bandit arms configured from settings['markets']")

            if region is not None:
                arms = [a for a in arms if a["region"] == region]
                if not arms:
                    raise ValueError(f"No bandit arms configured for region {region!r}")

            now = time.time()
            cooldowns = region_cooldowns or {}

            def _effective_cooldown(arm: Mapping[str, Any]) -> float:
                reg_cd = float(cooldowns.get(arm["region"], 0.0) or 0.0)
                arm_cd = float(arm.get("cooldown_until", 0.0) or 0.0)
                return max(reg_cd, arm_cd)

            active = [a for a in arms if _effective_cooldown(a) <= now]
            if not active:
                # All matching regions are cooled down; pick soonest-expiring cooldown pool.
                soonest = min(_effective_cooldown(a) for a in arms)
                active = [a for a in arms if _effective_cooldown(a) <= soonest + 1e-6]

            if self._rng.random() < self.exploration_floor:
                min_pulls = min(int(a["pulls"]) for a in active)
                undersampled = [a for a in active if int(a["pulls"]) == min_pulls]
                chosen = dict(self._rng.choice(undersampled))
                chosen["selection_mode"] = "explore"
                return chosen

            best_arm: Optional[dict[str, Any]] = None
            best_theta = -1.0
            for arm in active:
                alpha = max(1e-6, float(arm["alpha_param"]))
                beta = max(1e-6, float(arm["beta_param"]))
                theta = self._rng.betavariate(alpha, beta)
                if theta > best_theta:
                    best_theta = theta
                    best_arm = arm

            assert best_arm is not None
            chosen = dict(best_arm)
            chosen["sampled_theta"] = best_theta
            chosen["selection_mode"] = "thompson"
            return chosen

    def record_outcome(
        self,
        arm_key: str,
        reward: float,
        cooldown_until: float = 0.0,
    ) -> dict[str, Any]:
        """Persist a Beta posterior update and optional cooldown for ``arm_key``."""
        bounded_reward = min(1.0, max(0.0, float(reward)))
        with self._lock:
            return self.db.update_bandit_arm(
                arm_key=arm_key,
                reward=bounded_reward,
                cooldown_until=float(cooldown_until),
            )


__all__ = [
    "ThompsonBanditScheduler",
    "format_arm_key",
    "parse_arm_key",
]
