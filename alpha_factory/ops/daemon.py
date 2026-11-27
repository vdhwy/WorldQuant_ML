"""24/7 headless alpha mining daemon orchestrating generation, execution, qualification, and ops."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import signal
import threading
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Mapping, Optional

import requests

from alpha_factory.brain.auth import AuthenticationError, BrainAPIError, BrainAuthManager
from alpha_factory.brain.catalog import CatalogSync
from alpha_factory.brain.client import BrainClient
from alpha_factory.execution.dispatcher import SimulationDispatcher
from alpha_factory.execution.quota import NY_TZ, DailyQuotaManager
from alpha_factory.execution.slot_pool import SLOT_IDLE, SlotPool
from alpha_factory.generation.algorithmic_llm import AlgorithmicLLMGenerator
from alpha_factory.generation.ast_mutator import SafeASTMutator
from alpha_factory.generation.bandit import ThompsonBanditScheduler
from alpha_factory.generation.pre_dedup import DEFAULT_KNOWN_OPERATORS, PreSimDeduplicator
from alpha_factory.generation.producer import CandidateProducer
from alpha_factory.generation.validator import ExpressionValidator
from alpha_factory.ops.alerts import AlertManager
from alpha_factory.qualification.checks import evaluate_gate1, evaluate_gate2
from alpha_factory.qualification.dedup_vault import CorrelationDedupVault
from alpha_factory.qualification.submitter import SequentialSubmitter
from alpha_factory.storage.db import AlphaDB
from alpha_factory.storage.models import DEFAULT_SETTINGS, SimulationRecord, load_dotenv_file
from alpha_factory.storage.pnl_store import PnlStore

logger = logging.getLogger(__name__)

_CB_REGION_RE = re.compile(r"\bregion\s+([A-Za-z0-9_]+)\b", re.IGNORECASE)
_HOURLY_INTERVAL_S = 3600.0
_MAX_BACKUP_FILES = 14


def setup_logging(log_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Configure midnight-rotating file logging and stderr stream logging for ``alpha_factory``."""
    log_path = Path(log_dir).expanduser()
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "alpha_factory.log"

    target_logger = logging.getLogger("alpha_factory")
    target_logger.setLevel(level)

    for existing in list(target_logger.handlers):
        target_logger.removeHandler(existing)
        try:
            existing.close()
        except Exception:
            pass

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(threadName)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=_MAX_BACKUP_FILES,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    target_logger.addHandler(file_handler)
    target_logger.addHandler(stream_handler)
    return target_logger


def resolve_storage_paths(settings: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    """Resolve ``(db_path, pnl_dir, backup_dir)`` keeping test ``db_path`` isolated."""
    storage_cfg = settings.get("storage")
    if not isinstance(storage_cfg, Mapping):
        storage_cfg = {}

    default_storage = DEFAULT_SETTINGS["storage"]
    raw_db = str(storage_cfg.get("db_path") or default_storage["db_path"])
    db_path = Path(raw_db).expanduser()
    custom_db = raw_db != default_storage["db_path"] and raw_db != ":memory:"

    raw_pnl = storage_cfg.get("pnl_dir")
    if raw_pnl:
        pnl_dir = Path(str(raw_pnl)).expanduser()
    elif custom_db:
        pnl_dir = db_path.parent / "pnl"
    else:
        pnl_dir = Path(default_storage["pnl_dir"]).expanduser()

    raw_backup = storage_cfg.get("backup_dir")
    if raw_backup:
        backup_dir = Path(str(raw_backup)).expanduser()
    elif custom_db:
        backup_dir = db_path.parent / "backups"
    else:
        backup_dir = Path(default_storage["backup_dir"]).expanduser()

    return db_path, pnl_dir, backup_dir


def iter_configured_region_delays(settings: Mapping[str, Any]) -> list[tuple[str, int]]:
    """Return unique configured ``(region, delay)`` tuples from ``settings['markets']``."""
    markets_cfg = settings.get("markets")
    if not isinstance(markets_cfg, Mapping) or not markets_cfg:
        markets_cfg = DEFAULT_SETTINGS["markets"]

    seen: set[tuple[str, int]] = set()
    pairs: list[tuple[str, int]] = []
    for raw_region, mkt_cfg in markets_cfg.items():
        if not isinstance(mkt_cfg, Mapping):
            continue
        if mkt_cfg.get("enabled") is False:
            continue
        region = str(raw_region).strip()
        if not region:
            continue
        raw_delays = mkt_cfg.get("delays")
        if isinstance(raw_delays, (list, tuple)) and raw_delays:
            delays = [int(d) for d in raw_delays]
        else:
            delays = [int(mkt_cfg.get("delay", 1))]
        for delay in delays:
            pair = (region, delay)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def _simulation_to_alpha_payload(sim: SimulationRecord) -> dict[str, Any]:
    """Reconstruct a Gate-1 compatible alpha payload dictionary from a ``SimulationRecord``."""
    try:
        checks = json.loads(sim.checks_json) if sim.checks_json else []
    except (TypeError, ValueError):
        checks = []
    if not isinstance(checks, list):
        checks = []
    return {
        "id": sim.alpha_id,
        "is": {
            "sharpe": sim.sharpe,
            "fitness": sim.fitness,
            "turnover": sim.turnover,
            "margin": sim.margin,
            "returns": sim.returns,
            "longCount": sim.long_count,
            "shortCount": sim.short_count,
            "checks": checks,
        },
    }


def _build_digest_stats(dash: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten ``AlphaDB.get_dashboard_stats`` into keys formatted by ``AlertManager.send_daily_digest``."""
    quota = dash.get("quota") or {}
    sims = dash.get("simulations") or {}
    vault = dash.get("vault") or {}
    queue_stats = dash.get("queue") or {}
    kept_cnt = int(vault.get("KEPT", 0))
    submitted_brain_cnt = int(vault.get("SUBMITTED", 0))
    return {
        "date_ny": dash.get("date_ny", ""),
        "submitted": int(quota.get("submitted_count", 0)),
        "completed": int(quota.get("completed_count", 0)),
        "failed": int(quota.get("failed_count", 0)),
        "gate1_pass": int(sims.get("gate1_pass", 0)),
        "gate2_pass": int(sims.get("gate2_pass", 0)),
        "kept": kept_cnt,
        "submitted_to_brain": submitted_brain_cnt,
        "vault_active": kept_cnt + submitted_brain_cnt,
        "corr_dup": int(vault.get("CORR_DUP", 0)),
        "queue_pending": int(queue_stats.get("PENDING", 0)),
    }


class AlphaMiningDaemon:
    """Orchestrates 24/7 candidate generation, simulation dispatch, qualification, and ops."""

    def __init__(
        self,
        settings: dict[str, Any],
        client: Optional[BrainClient] = None,
        db: Optional[AlphaDB] = None,
        pnl_store: Optional[PnlStore] = None,
        alerts: Optional[AlertManager] = None,
    ) -> None:
        self.settings: dict[str, Any] = dict(settings)
        db_path, pnl_dir, backup_dir = resolve_storage_paths(self.settings)
        self._backup_dir = backup_dir

        self.db: AlphaDB = db if db is not None else AlphaDB(db_path)
        self.pnl_store: PnlStore = pnl_store if pnl_store is not None else PnlStore(pnl_dir)

        self._alerts_injected = alerts is not None
        self.alerts: AlertManager = (
            alerts if alerts is not None else AlertManager.from_settings(self.settings)
        )

        self.client: Optional[BrainClient] = client
        existing_auth = getattr(client, "auth_manager", None) if client is not None else None
        if existing_auth is not None:
            self.auth: BrainAuthManager = existing_auth
            if getattr(self.auth, "alert_callback", None) is None:
                self.auth.alert_callback = self._on_biometric_alert
        else:
            self.auth = BrainAuthManager.from_settings(
                self.settings,
                alert_callback=self._on_biometric_alert,
            )

        self.quota: DailyQuotaManager = DailyQuotaManager.from_settings(self.db, self.settings)
        self.catalog: CatalogSync = CatalogSync(db=self.db, client=self.client)

        ExpressionValidator.seed_compiler_knowledge(self.db)
        operators = self.db.get_operators()
        compiler_rules = self.db.get_compiler_rules()

        gen_cfg = self.settings.get("generation") or {}
        if not isinstance(gen_cfg, Mapping):
            gen_cfg = {}
        qual_cfg = self.settings.get("qualification") or {}
        if not isinstance(qual_cfg, Mapping):
            qual_cfg = {}
        dedup_cfg = self.settings.get("dedup") or {}
        if not isinstance(dedup_cfg, Mapping):
            dedup_cfg = {}
        sub_cfg = self.settings.get("submission") or {}
        if not isinstance(sub_cfg, Mapping):
            sub_cfg = {}
        ops_cfg = self.settings.get("ops") or {}
        if not isinstance(ops_cfg, Mapping):
            ops_cfg = {}

        self.validator: ExpressionValidator = ExpressionValidator(
            operators=operators,
            compiler_rules=compiler_rules,
        )
        lookbacks = tuple(int(x) for x in (gen_cfg.get("lookback_days") or (5, 10, 20, 60, 120)))
        self.mutator: SafeASTMutator = SafeASTMutator(
            validator=self.validator,
            lookback_grid=lookbacks,
        )
        self.pre_dedup: PreSimDeduplicator = PreSimDeduplicator(
            db=self.db,
            max_per_structure=int(gen_cfg.get("max_per_structure", 3)),
            max_per_dataset_structure=int(gen_cfg.get("max_per_dataset_structure", 2)),
        )
        self.algo_llm: AlgorithmicLLMGenerator = AlgorithmicLLMGenerator(
            llm_cfg=gen_cfg.get("llm") or {},
            operators=operators,
            datafields=[],
        )
        self.bandit: ThompsonBanditScheduler = ThompsonBanditScheduler(
            db=self.db,
            settings=self.settings,
        )

        self.slot_pool: SlotPool = SlotPool.from_settings(self.settings)
        self.qualification_queue: queue.Queue[Optional[str]] = queue.Queue()

        self.dispatcher: SimulationDispatcher = SimulationDispatcher(
            db=self.db,
            client=self.client,  # type: ignore[arg-type]
            quota=self.quota,
            slot_pool=self.slot_pool,
            settings=self.settings,
            qualification_queue=self.qualification_queue,  # type: ignore[arg-type]
            validator=self.validator,
            alert_callback=self._on_dispatcher_alert,
        )
        self.producer: CandidateProducer = CandidateProducer(
            db=self.db,
            settings=self.settings,
            bandit=self.bandit,
            validator=self.validator,
            pre_dedup=self.pre_dedup,
            algo_llm=self.algo_llm,
            mutator=self.mutator,
            region_cooldowns_fn=self.dispatcher.get_region_cooldowns,
        )
        self.vault: CorrelationDedupVault = CorrelationDedupVault(
            db=self.db,
            pnl_store=self.pnl_store,
            client=self.client,
            dedup_cfg=dedup_cfg,
            tag_on_brain=bool(qual_cfg.get("tag_on_brain", True)),
        )
        self.submitter: SequentialSubmitter = SequentialSubmitter(
            db=self.db,
            client=self.client,  # type: ignore[arg-type]
            vault=self.vault,
            qual_cfg=qual_cfg,
            max_submissions_per_day=int(sub_cfg.get("max_submissions_per_day", 2)),
        )

        self._shutdown_drain_timeout_s = max(
            0.0, float(ops_cfg.get("shutdown_drain_timeout_s", 300.0))
        )
        self._num_qual_workers = max(1, int(ops_cfg.get("qualification_workers", 2)))
        self._housekeeping_interval_s = max(
            1.0, float(ops_cfg.get("housekeeping_interval_s", 60.0))
        )

        self._stop_event = threading.Event()
        self._qual_stop_event = threading.Event()
        self._housekeeping_stop_event = threading.Event()
        self._qual_cond = threading.Condition()
        self._qual_threads: list[threading.Thread] = []
        self._housekeeping_thread: Optional[threading.Thread] = None

        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, int] = {
            "candidates_queued": 0,
            "dispatched": 0,
            "qualified": 0,
            "kept": 0,
            "corr_dup": 0,
            "gate1_failed": 0,
            "gate2_failed": 0,
            "submitted": 0,
            "errors": 0,
        }
        self._quota_warned_dates: set[str] = set()
        self._current_ny_date: str = self.quota.get_ny_date()
        self._last_backup_date_ny: Optional[str] = self.db.get_runtime_state("last_backup_date_ny")
        self._last_hourly_log_mono: float = 0.0

    # ------------------------------------------------------------------
    # Internal wiring, callbacks, and refresh helpers
    # ------------------------------------------------------------------

    def _on_biometric_alert(self, verify_url: str) -> None:
        try:
            self.alerts.alert_biometric_required(verify_url)
        except Exception:
            logger.exception("Failed sending biometric verification alert")

    def _on_dispatcher_alert(self, message: str) -> None:
        try:
            if "circuit breaker" in message.lower():
                match = _CB_REGION_RE.search(message)
                region = match.group(1) if match else "UNKNOWN"
                self.alerts.alert_circuit_breaker(region, message)
            else:
                self.alerts.send(message, key="dispatcher_alert")
        except Exception:
            logger.exception("Failed sending dispatcher alert: %s", message)

    def _refresh_alert_env(self) -> None:
        if self._alerts_injected:
            return
        if not self.alerts.token:
            env_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
            if env_token:
                self.alerts.token = env_token
        if not self.alerts.chat_id:
            env_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
            if env_chat_id:
                self.alerts.chat_id = env_chat_id

    def _ensure_client(self) -> BrainClient:
        if self.client is None:
            brain_cfg = self.settings.get("brain") or {}
            polling_cfg = self.settings.get("polling") or {}
            self.client = BrainClient(
                auth_manager=self.auth,
                api_url=str(brain_cfg.get("api_url", DEFAULT_SETTINGS["brain"]["api_url"])),
                connect_timeout_s=float(brain_cfg.get("connect_timeout_s", 10)),
                read_timeout_s=float(brain_cfg.get("read_timeout_s", 60)),
                max_wait_s=float(polling_cfg.get("max_wait_s", 3600)),
                recordset_max_wait_s=float(polling_cfg.get("recordset_max_wait_s", 600)),
                default_interval_s=float(polling_cfg.get("default_interval_s", 5)),
                max_http_retries=int(polling_cfg.get("max_http_retries", 3)),
            )
        self.catalog.client = self.client
        self.dispatcher.client = self.client
        self.vault.client = self.client
        self.submitter.client = self.client
        return self.client

    def _authenticate(self, client: BrainClient) -> None:
        if "ensure_authenticated" in getattr(client, "__dict__", {}) or self.auth is None:
            client.ensure_authenticated()
        elif hasattr(client, "session"):
            self.auth.ensure_authenticated(client.session)
        else:
            client.ensure_authenticated()

    def _has_any_datafields(self) -> bool:
        markets_cfg = self.settings.get("markets") or DEFAULT_SETTINGS["markets"]
        if not isinstance(markets_cfg, Mapping):
            return False
        for region, mkt_cfg in markets_cfg.items():
            if not isinstance(mkt_cfg, Mapping):
                continue
            universe = str(mkt_cfg.get("default_universe") or "TOP3000")
            delay = int(mkt_cfg.get("delay", 1))
            if self.db.get_datafields(str(region), universe, delay):
                return True
        return False

    def _refresh_generators_from_db(self) -> None:
        ops = self.db.get_operators()
        rules = self.db.get_compiler_rules()
        if ops:
            self.validator.operators = {op.name: op for op in ops if op.name}
            self.pre_dedup.known_operators = DEFAULT_KNOWN_OPERATORS | frozenset(
                op.name for op in ops if op.name
            )
            if type(self.algo_llm) is AlgorithmicLLMGenerator and not self.algo_llm.operators:
                refreshed_llm = AlgorithmicLLMGenerator(
                    llm_cfg=self.algo_llm.llm_cfg,
                    operators=ops,
                    datafields=self.algo_llm.datafields,
                )
                self.algo_llm = refreshed_llm
                self.producer.algo_llm = refreshed_llm
        self.validator.load_compiler_rules(rules)

    def _startup_sync(self, client: BrainClient) -> None:
        self._authenticate(client)

        ops_cfg = self.settings.get("ops") or {}
        workspace_root = str(ops_cfg.get("workspace_root", ".")) if isinstance(ops_cfg, Mapping) else "."
        bootstrapped = False

        try:
            self.catalog.sync_operators()
        except AuthenticationError:
            raise
        except Exception as exc:
            logger.warning("Operator catalog sync failed (%s); falling back to local cache.", exc)
            if not self.db.get_operators():
                self.catalog.bootstrap_from_local_cache(workspace_root=workspace_root)
                bootstrapped = True

        if (
            not bootstrapped
            and not self._has_any_datafields()
            and self.db.count_queue(status="PENDING") == 0
        ):
            self.catalog.bootstrap_from_local_cache(workspace_root=workspace_root)

        self._refresh_generators_from_db()

        try:
            synced = self.vault.sync_submitted_alphas()
            logger.info("Synced %d active submitted alpha(s) from BRAIN.", synced)
        except AuthenticationError:
            raise
        except (BrainAPIError, requests.RequestException) as exc:
            logger.warning("Vault active-alpha sync failed (%s); continuing.", exc)

        recovered = self.dispatcher.recover_in_flight()
        if recovered:
            logger.info("Recovered %d in-flight simulation batch(es) at startup.", recovered)

    # ------------------------------------------------------------------
    # Qualification worker
    # ------------------------------------------------------------------

    def _tag_brain_alpha(self, alpha_id: str, color: str) -> None:
        if self.client is None:
            return
        try:
            self.client.patch_alpha_properties(alpha_id, color=color)
        except Exception as exc:
            logger.warning("Failed tagging alpha %s with color=%s: %s", alpha_id, color, exc)

    def _resolve_arm_key_for_sim(self, sim: SimulationRecord) -> Optional[str]:
        if not sim.config_hash:
            return None
        try:
            with self.db._read_conn() as conn:
                row = conn.execute(
                    "SELECT source FROM candidate_queue WHERE config_hash = ? ORDER BY id DESC LIMIT 1",
                    (sim.config_hash,),
                ).fetchone()
            if row is not None and row["source"]:
                return str(row["source"])
        except Exception:
            logger.debug("Unable to look up bandit arm for config_hash=%s", sim.config_hash, exc_info=True)
        return None

    def _qualify_alpha(self, alpha_id: str) -> None:
        sim = self.db.get_simulation(alpha_id)
        if sim is None:
            logger.warning("Qualification skipped: simulation %s not found in DB.", alpha_id)
            return

        qual_cfg = self.settings.get("qualification")
        if not isinstance(qual_cfg, Mapping):
            qual_cfg = {}
        tag_on_brain = bool(qual_cfg.get("tag_on_brain", True))
        arm_key = self._resolve_arm_key_for_sim(sim)

        alpha_payload = _simulation_to_alpha_payload(sim)
        g1 = evaluate_gate1(alpha_payload, qual_cfg)
        if not g1.passed:
            self.db.update_simulation_qualification(
                alpha_id,
                gate1_pass=False,
                gate2_pass=False,
                pyramid_matches_json=g1.pyramid_matches,
            )
            if tag_on_brain:
                self._tag_brain_alpha(alpha_id, "RED")
            if arm_key:
                self.bandit.record_outcome(arm_key, 0.0)
            with self._metrics_lock:
                self._metrics["qualified"] += 1
                self._metrics["gate1_failed"] += 1
            logger.info("Alpha %s failed Gate 1: %s", alpha_id, "; ".join(g1.fail_reasons))
            return

        if self.client is None:
            raise RuntimeError("BrainClient is required for Gate 2 qualification")

        try:
            g2, pnl_data = evaluate_gate2(self.client, alpha_id, qual_cfg)
        except AuthenticationError as exc:
            logger.error("Authentication error during Gate 2 for %s: %s", alpha_id, exc)
            self.alerts.send(
                f"Authentication error qualifying alpha {alpha_id}: {exc}",
                key="auth_error",
            )
            with self._metrics_lock:
                self._metrics["errors"] += 1
            return
        except (BrainAPIError, requests.RequestException) as exc:
            logger.warning("API/transport error during Gate 2 for %s: %s", alpha_id, exc)
            with self._metrics_lock:
                self._metrics["errors"] += 1
            return

        merged_pyramids = list(dict.fromkeys([*g1.pyramid_matches, *g2.pyramid_matches]))
        self.db.update_simulation_qualification(
            alpha_id,
            gate1_pass=True,
            gate2_pass=bool(g2.passed),
            self_corr_max=g2.self_corr_max,
            prod_corr_max=g2.prod_corr_max,
            power_pool_corr_max=g2.power_pool_corr_max,
            pyramid_matches_json=merged_pyramids,
        )

        if not g2.passed or pnl_data is None:
            if tag_on_brain:
                self._tag_brain_alpha(alpha_id, "RED")
            if arm_key:
                self.bandit.record_outcome(arm_key, 0.25)
            with self._metrics_lock:
                self._metrics["qualified"] += 1
                self._metrics["gate2_failed"] += 1
            logger.info("Alpha %s failed Gate 2: %s", alpha_id, "; ".join(g2.fail_reasons))
            return

        updated_sim = self.db.get_simulation(alpha_id) or sim
        dates, daily_returns = pnl_data
        entry = self.vault.admit(
            updated_sim,
            dates,
            daily_returns,
            has_pyramid_match=bool(merged_pyramids),
        )
        if entry.vault_status == "KEPT":
            reward = 1.0
            with self._metrics_lock:
                self._metrics["qualified"] += 1
                self._metrics["kept"] += 1
            try:
                self.alerts.alert_new_kept_alpha(entry)
            except Exception:
                logger.exception("Failed sending new kept alpha alert for %s", alpha_id)
            logger.info(
                "Alpha %s admitted to vault as KEPT (score=%.4f, sharpe=%s, fitness=%s).",
                alpha_id,
                entry.composite_score,
                entry.sharpe,
                entry.fitness,
            )
        else:
            reward = 0.6
            with self._metrics_lock:
                self._metrics["qualified"] += 1
                self._metrics["corr_dup"] += 1
            logger.info(
                "Alpha %s marked CORR_DUP of %s (max_local_corr=%s).",
                alpha_id,
                entry.duplicate_of,
                entry.max_local_corr,
            )

        if arm_key:
            self.bandit.record_outcome(arm_key, reward)

    def _qualification_worker_loop(self) -> None:
        while True:
            if self._qual_stop_event.is_set() and self.qualification_queue.empty():
                break
            try:
                alpha_id = self.qualification_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if alpha_id is not None:
                    self._qualify_alpha(str(alpha_id))
            except Exception:
                logger.exception("Unexpected failure qualifying alpha %r", alpha_id)
                with self._metrics_lock:
                    self._metrics["errors"] += 1
            finally:
                self.qualification_queue.task_done()
                with self._qual_cond:
                    self._qual_cond.notify_all()

    def _start_qualification_workers(self) -> None:
        self._qual_stop_event.clear()
        self._qual_threads = []
        for idx in range(self._num_qual_workers):
            worker = threading.Thread(
                target=self._qualification_worker_loop,
                name=f"alpha-factory-qualifier-{idx}",
                daemon=True,
            )
            worker.start()
            self._qual_threads.append(worker)

    def _wait_for_qualification_drain(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._qual_cond:
            while self.qualification_queue.unfinished_tasks > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._qual_cond.wait(timeout=min(0.1, remaining))
            return True

    def _stop_qualification_workers(self, timeout_s: float = 5.0) -> None:
        self._qual_stop_event.set()
        deadline = time.monotonic() + max(0.0, timeout_s)
        for worker in self._qual_threads:
            if worker.is_alive() and worker is not threading.current_thread():
                remaining = max(0.0, deadline - time.monotonic())
                worker.join(timeout=remaining)
        self._qual_threads = [t for t in self._qual_threads if t.is_alive()]

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def _check_quota_warning(self) -> None:
        status = self.quota.get_status()
        if status.warning_triggered and status.date_ny not in self._quota_warned_dates:
            self._quota_warned_dates.add(status.date_ny)
            try:
                self.alerts.alert_quota_warning(status)
            except Exception:
                logger.exception("Failed sending quota warning alert")

    def _check_slot_watchdog(self) -> None:
        polling_cfg = self.settings.get("polling") or {}
        ops_cfg = self.settings.get("ops") or {}
        threshold_s = float(polling_cfg.get("max_wait_s", 3600)) + float(
            ops_cfg.get("watchdog_extra_s", 60)
        )
        now = time.time()
        for slot in self.slot_pool.get_snapshot():
            if slot.get("status") == SLOT_IDLE or slot.get("acquired_at") is None:
                continue
            held_s = now - float(slot["acquired_at"])
            if held_s > threshold_s:
                msg = (
                    f"Slot watchdog warning: slot {slot['slot_id']} "
                    f"(region={slot.get('region')}, job_id={slot.get('job_id')}) "
                    f"held for {held_s:.1f}s (> {threshold_s:.1f}s)"
                )
                logger.warning(msg)
                try:
                    self.alerts.send(msg, key=f"slot_watchdog:{slot['slot_id']}")
                except Exception:
                    logger.exception("Failed sending slot watchdog alert")

    def _log_hourly_stats(self, force: bool = False) -> None:
        now_mono = time.monotonic()
        if not force and self._last_hourly_log_mono > 0.0 and (now_mono - self._last_hourly_log_mono) < _HOURLY_INTERVAL_S:
            return
        self._last_hourly_log_mono = now_mono
        dash = self.db.get_dashboard_stats(date_ny=self.quota.get_ny_date())
        quota_stat = dash["quota"]
        q_counts = dash["queue"]
        sim_counts = dash["simulations"]
        vault_counts = dash["vault"]
        logger.info(
            "Hourly status [NY=%s] quota=%d/%d queue(P=%d,F=%d,D=%d) sims(total=%d,g1=%d,g2=%d) "
            "vault(KEPT=%d,CORR_DUP=%d,SUBMITTED=%d) slots=%d/%d",
            dash["date_ny"],
            quota_stat["submitted_count"],
            self.quota.daily_sim_limit,
            q_counts.get("PENDING", 0),
            q_counts.get("IN_FLIGHT", 0),
            q_counts.get("DONE", 0),
            sim_counts.get("total", 0),
            sim_counts.get("gate1_pass", 0),
            sim_counts.get("gate2_pass", 0),
            vault_counts.get("KEPT", 0),
            vault_counts.get("CORR_DUP", 0),
            vault_counts.get("SUBMITTED", 0),
            self.slot_pool.active_count(),
            self.slot_pool.max_slots,
        )

    def _run_daily_backup(self, ny_date: str) -> None:
        if self._last_backup_date_ny == ny_date or str(self.db.db_path) == ":memory:":
            return
        dest = self._backup_dir / f"alpha_factory_{ny_date}.db"
        try:
            self.db.backup_to(dest)
            self._last_backup_date_ny = ny_date
            self.db.set_runtime_state("last_backup_date_ny", ny_date)
            logger.info("Completed daily SQLite backup to %s", dest)
        except Exception:
            logger.exception("Daily SQLite backup to %s failed", dest)

    def reconcile_all_clusters(self) -> dict[tuple[str, int], dict[str, int]]:
        """Run ``CorrelationDedupVault.reconcile_clusters`` for every configured ``(region, delay)``."""
        results: dict[tuple[str, int], dict[str, int]] = {}
        for region, delay in iter_configured_region_delays(self.settings):
            try:
                results[(region, delay)] = self.vault.reconcile_clusters(region, delay)
            except Exception:
                logger.exception("Cluster reconciliation failed for (%s, delay=%d)", region, delay)
        return results

    def _run_auto_submit(self) -> None:
        sub_cfg = self.settings.get("submission") or {}
        if not isinstance(sub_cfg, Mapping) or not bool(sub_cfg.get("auto_submit", False)):
            return
        if self.client is None:
            return
        if not self.db.get_vault_entries(statuses="KEPT", limit=1):
            return
        try:
            results = self.submitter.submit_top_kept()
            submitted_now = sum(1 for r in results if r.get("submitted"))
            if submitted_now:
                with self._metrics_lock:
                    self._metrics["submitted"] += submitted_now
                logger.info("Auto-submitted %d KEPT alpha(s) to BRAIN.", submitted_now)
        except Exception:
            logger.exception("Auto-submission failed during housekeeping")

    def run_housekeeping_once(self, now_ny: Optional[datetime] = None) -> None:
        """Execute one housekeeping pass: quota alert, watchdog, backup, NY rollover, auto-submit."""
        ny_date = (
            now_ny.astimezone(NY_TZ).strftime("%Y-%m-%d")
            if now_ny is not None
            else self.quota.get_ny_date()
        )

        self._check_quota_warning()
        self._check_slot_watchdog()
        self._log_hourly_stats()
        self._run_daily_backup(ny_date)

        if ny_date != self._current_ny_date:
            prev_date = self._current_ny_date
            dash = self.db.get_dashboard_stats(date_ny=prev_date)
            digest_payload = _build_digest_stats(dash)
            try:
                self.alerts.send_daily_digest(digest_payload)
            except Exception:
                logger.exception("Failed sending NY-midnight daily digest for %s", prev_date)
            self.reconcile_all_clusters()
            self._current_ny_date = ny_date
            self.db.set_runtime_state("last_digest_date_ny", ny_date)

        self._run_auto_submit()

    def _housekeeping_loop(self) -> None:
        while not self._housekeeping_stop_event.is_set():
            try:
                self.run_housekeeping_once()
            except Exception:
                logger.exception("Error in Housekeeping loop")
            self._housekeeping_stop_event.wait(self._housekeeping_interval_s)

    def _start_housekeeping(self) -> None:
        self._housekeeping_stop_event.clear()
        thread = threading.Thread(
            target=self._housekeeping_loop,
            name="alpha-factory-housekeeping",
            daemon=True,
        )
        self._housekeeping_thread = thread
        thread.start()

    def _stop_housekeeping(self, timeout_s: float = 5.0) -> None:
        self._housekeeping_stop_event.set()
        thread = self._housekeeping_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)
        if thread is None or not thread.is_alive():
            self._housekeeping_thread = None

    # ------------------------------------------------------------------
    # Lifecycle & run entrypoint
    # ------------------------------------------------------------------

    def _wait_for_dispatcher_workers(self, timeout_s: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        current = threading.current_thread()
        while True:
            with self.dispatcher._workers_lock:
                workers = [
                    t
                    for t in self.dispatcher._active_workers
                    if t.is_alive() and t is not current
                ]
            if not workers:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            for worker in workers:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    break
                worker.join(timeout=rem)

    def _install_signal_handlers(self) -> tuple[bool, Any, Any]:
        if threading.current_thread() is not threading.main_thread():
            return False, None, None

        def _handle_signal(signum: int, _frame: Any) -> None:
            logger.info("Received signal %s; initiating graceful shutdown.", signum)
            self.request_stop()

        prev_int = signal.getsignal(signal.SIGINT)
        prev_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        return True, prev_int, prev_term

    def _restore_signal_handlers(self, prev_int: Any, prev_term: Any) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        if prev_int is not None:
            signal.signal(signal.SIGINT, prev_int)
        if prev_term is not None:
            signal.signal(signal.SIGTERM, prev_term)

    def _build_summary(self, *, dry_run: bool, cycles_completed: int) -> dict[str, Any]:
        with self._metrics_lock:
            snapshot = dict(self._metrics)
        return {
            "dry_run": dry_run,
            "cycles_completed": cycles_completed,
            **snapshot,
            "stopped": self._stop_event.is_set(),
            "stats": self.db.get_dashboard_stats(date_ny=self.quota.get_ny_date()),
        }

    def request_stop(self) -> None:
        """Signal the daemon to stop accepting new work and drain cleanly."""
        self._stop_event.set()

    def run(
        self,
        dry_run: bool = False,
        max_cycles: Optional[int] = None,
        install_signals: bool = True,
    ) -> dict[str, Any]:
        """Run the alpha mining pipeline in dry-run, bounded, or 24/7 daemon mode."""
        load_dotenv_file()
        self._refresh_alert_env()

        if dry_run:
            cycles_target = 1 if max_cycles is None else max(0, int(max_cycles))
            cycles_completed = 0
            llm_cfg = self.algo_llm.llm_cfg
            had_backend = "backend" in llm_cfg
            prev_backend = llm_cfg.get("backend")
            llm_cfg["backend"] = "none"
            try:
                for _ in range(cycles_target):
                    if self._stop_event.is_set():
                        break
                    inserted = self.producer.produce_once()
                    with self._metrics_lock:
                        self._metrics["candidates_queued"] += int(inserted)
                    cycles_completed += 1
            finally:
                if had_backend:
                    llm_cfg["backend"] = prev_backend
                else:
                    llm_cfg.pop("backend", None)

            pending = self.db.count_queue(status="PENDING")
            logger.info(
                "Dry-run complete: cycles=%d candidates_queued=%d queue_pending=%d",
                cycles_completed,
                self._metrics["candidates_queued"],
                pending,
            )
            return self._build_summary(dry_run=True, cycles_completed=cycles_completed)

        client = self._ensure_client()
        self._startup_sync(client)

        installed_signals = False
        prev_int = None
        prev_term = None
        if install_signals:
            installed_signals, prev_int, prev_term = self._install_signal_handlers()

        cycles_completed = 0
        try:
            self._start_qualification_workers()
            if max_cycles is not None:
                target_cycles = max(0, int(max_cycles))
                # Drain any simulations recovered during startup before bounded cycles.
                self._wait_for_qualification_drain(self._shutdown_drain_timeout_s)

                for _ in range(target_cycles):
                    if self._stop_event.is_set():
                        break
                    if self.db.count_queue(status="PENDING") < self.producer.queue_low_watermark:
                        produced = self.producer.produce_once()
                        with self._metrics_lock:
                            self._metrics["candidates_queued"] += int(produced)
                    if self._stop_event.is_set():
                        break
                    dispatched = self.dispatcher.dispatch_once()
                    with self._metrics_lock:
                        self._metrics["dispatched"] += int(dispatched)
                    self._wait_for_dispatcher_workers(self._shutdown_drain_timeout_s)
                    self._wait_for_qualification_drain(self._shutdown_drain_timeout_s)
                    cycles_completed += 1

                if not self._stop_event.is_set():
                    self.run_housekeeping_once()
            else:
                if not self._stop_event.is_set():
                    self.producer.start()
                    self.dispatcher.start()
                    self._start_housekeeping()
                    while not self._stop_event.is_set():
                        self._stop_event.wait(0.5)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received; initiating graceful shutdown.")
            self.request_stop()
        finally:
            drain_deadline = time.monotonic() + self._shutdown_drain_timeout_s
            self.producer.stop(timeout=5.0)
            self.dispatcher.stop(
                drain_timeout_s=max(0.0, drain_deadline - time.monotonic())
            )
            self._wait_for_qualification_drain(max(0.0, drain_deadline - time.monotonic()))
            self._stop_qualification_workers(timeout_s=5.0)
            self._stop_housekeeping(timeout_s=5.0)
            if installed_signals:
                self._restore_signal_handlers(prev_int, prev_term)

        return self._build_summary(dry_run=False, cycles_completed=cycles_completed)


__all__ = [
    "AlphaMiningDaemon",
    "iter_configured_region_delays",
    "resolve_storage_paths",
    "setup_logging",
]
