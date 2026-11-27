"""Thread-safe SQLite WAL persistence repository for alpha_factory."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from alpha_factory.storage.models import (
    CandidateItem,
    DailyQuotaStatus,
    DataFieldRecord,
    OperatorRecord,
    SimulationRecord,
    VaultEntry,
)

NY_TZ = ZoneInfo("America/New_York")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT NOT NULL,
    region TEXT NOT NULL,
    universe TEXT NOT NULL,
    delay INTEGER NOT NULL,
    instrument_type TEXT NOT NULL DEFAULT 'EQUITY',
    name TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    subcategory TEXT NOT NULL DEFAULT '',
    coverage REAL NOT NULL DEFAULT 0.0,
    field_count INTEGER NOT NULL DEFAULT 0,
    user_count INTEGER NOT NULL DEFAULT 0,
    value_score REAL NOT NULL DEFAULT 0.0,
    alpha_count INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    PRIMARY KEY (id, region, universe, delay)
);

CREATE TABLE IF NOT EXISTS datafields (
    field_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL DEFAULT '',
    region TEXT NOT NULL,
    universe TEXT NOT NULL,
    delay INTEGER NOT NULL,
    field_type TEXT NOT NULL DEFAULT 'MATRIX',
    category TEXT NOT NULL DEFAULT '',
    subcategory TEXT NOT NULL DEFAULT '',
    coverage REAL NOT NULL DEFAULT 0.0,
    user_count INTEGER NOT NULL DEFAULT 0,
    alpha_count INTEGER NOT NULL DEFAULT 0,
    description TEXT NOT NULL DEFAULT '',
    is_event_input INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    PRIMARY KEY (field_id, region, universe, delay)
);

CREATE TABLE IF NOT EXISTS operators (
    name TEXT PRIMARY KEY,
    category TEXT NOT NULL DEFAULT '',
    scope_json TEXT NOT NULL DEFAULT '[]',
    definition TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    num_inputs INTEGER NOT NULL DEFAULT 1,
    min_inputs INTEGER NOT NULL DEFAULT 1,
    uses_lookback INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS candidate_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_hash TEXT UNIQUE NOT NULL,
    ast_fingerprint TEXT NOT NULL,
    expression TEXT NOT NULL,
    region TEXT NOT NULL,
    universe TEXT NOT NULL,
    delay INTEGER NOT NULL,
    settings_json TEXT NOT NULL,
    source TEXT NOT NULL,
    dataset_id TEXT DEFAULT '',
    priority REAL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'PENDING',
    progress_url TEXT DEFAULT '',
    batch_id TEXT DEFAULT '',
    error_message TEXT DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_candidate_queue_status_mkt
    ON candidate_queue (status, region, universe, delay);

CREATE INDEX IF NOT EXISTS idx_candidate_queue_ast_fp
    ON candidate_queue (ast_fingerprint);

CREATE TABLE IF NOT EXISTS simulations (
    alpha_id TEXT PRIMARY KEY,
    config_hash TEXT NOT NULL,
    ast_fingerprint TEXT NOT NULL,
    expression TEXT NOT NULL,
    region TEXT NOT NULL,
    universe TEXT NOT NULL,
    delay INTEGER NOT NULL,
    settings_json TEXT NOT NULL,
    status TEXT NOT NULL,
    sharpe REAL DEFAULT 0.0,
    fitness REAL DEFAULT 0.0,
    turnover REAL DEFAULT 0.0,
    returns REAL DEFAULT 0.0,
    drawdown REAL DEFAULT 0.0,
    margin REAL DEFAULT 0.0,
    long_count INTEGER DEFAULT 0,
    short_count INTEGER DEFAULT 0,
    checks_json TEXT DEFAULT '[]',
    fail_count INTEGER DEFAULT 0,
    warn_count INTEGER DEFAULT 0,
    gate1_pass INTEGER DEFAULT 0,
    gate2_pass INTEGER DEFAULT 0,
    self_corr_max REAL,
    prod_corr_max REAL,
    power_pool_corr_max REAL,
    pyramid_matches_json TEXT DEFAULT '[]',
    error_message TEXT DEFAULT '',
    simulated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_simulations_region_delay_g2
    ON simulations (region, delay, gate2_pass);

CREATE INDEX IF NOT EXISTS idx_simulations_config_hash
    ON simulations (config_hash);

CREATE TABLE IF NOT EXISTS submittable_vault (
    alpha_id TEXT PRIMARY KEY,
    region TEXT NOT NULL,
    delay INTEGER NOT NULL,
    expression TEXT NOT NULL,
    composite_score REAL NOT NULL,
    vault_status TEXT NOT NULL,
    duplicate_of TEXT DEFAULT '',
    max_local_corr REAL DEFAULT 0.0,
    max_local_corr_peer TEXT DEFAULT '',
    sharpe REAL DEFAULT 0.0,
    fitness REAL DEFAULT 0.0,
    turnover REAL DEFAULT 0.0,
    prod_corr_max REAL DEFAULT 0.0,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vault_region_delay_status
    ON submittable_vault (region, delay, vault_status);

CREATE TABLE IF NOT EXISTS daily_quota (
    date_ny TEXT PRIMARY KEY,
    submitted_count INTEGER DEFAULT 0,
    completed_count INTEGER DEFAULT 0,
    failed_count INTEGER DEFAULT 0,
    warning_sent INTEGER DEFAULT 0,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS compiler_knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type TEXT NOT NULL,
    operator_name TEXT DEFAULT '',
    field_id TEXT DEFAULT '',
    error_pattern TEXT DEFAULT '',
    replacement TEXT DEFAULT '',
    occurrences INTEGER DEFAULT 1,
    updated_at REAL NOT NULL,
    UNIQUE(rule_type, operator_name, field_id, error_pattern)
);

CREATE TABLE IF NOT EXISTS bandit_arms (
    arm_key TEXT PRIMARY KEY,
    alpha_param REAL DEFAULT 1.0,
    beta_param REAL DEFAULT 1.0,
    pulls INTEGER DEFAULT 0,
    total_reward REAL DEFAULT 0.0,
    cooldown_until REAL DEFAULT 0.0,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runtime_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


def _extract_label(value: Any) -> str:
    """Normalize a string or BRAIN category/subcategory dict to a plain string."""
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return str(value.get("name") or value.get("id") or "")
    return str(value)


def _coerce_timestamp(value: Any, default: Optional[float] = None) -> float:
    """Convert numeric, numeric-string, or ISO timestamp values to epoch seconds."""
    if value is None or value == "":
        return time.time() if default is None else float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            try:
                return datetime.fromisoformat(value).timestamp()
            except ValueError:
                return time.time() if default is None else float(default)
    return time.time() if default is None else float(default)


def _coerce_json(value: Any, default_str: str) -> str:
    """Serialize list/dict payloads to JSON text while passing strings through."""
    if value is None or value == "":
        return default_str
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _optional_bool_int(value: Any) -> Optional[int]:
    return None if value is None else int(bool(value))


def _field_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Fetch first present non-None key/attribute from a Mapping or dataclass/object."""
    if isinstance(obj, Mapping):
        for key in keys:
            if key in obj and obj[key] is not None:
                return obj[key]
        return default
    for key in keys:
        if hasattr(obj, key):
            val = getattr(obj, key)
            if val is not None:
                return val
    return default


def _row_to_candidate(row: sqlite3.Row) -> CandidateItem:
    return CandidateItem(
        id=int(row["id"]),
        config_hash=str(row["config_hash"]),
        ast_fingerprint=str(row["ast_fingerprint"]),
        expression=str(row["expression"]),
        region=str(row["region"]),
        universe=str(row["universe"]),
        delay=int(row["delay"]),
        settings_json=str(row["settings_json"]),
        source=str(row["source"]),
        dataset_id=str(row["dataset_id"] or ""),
        priority=float(row["priority"] if row["priority"] is not None else 0.0),
        status=str(row["status"]),
        progress_url=row["progress_url"] or None,
        batch_id=row["batch_id"] or None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_simulation(row: sqlite3.Row) -> SimulationRecord:
    return SimulationRecord(
        alpha_id=str(row["alpha_id"]),
        config_hash=str(row["config_hash"]),
        ast_fingerprint=str(row["ast_fingerprint"]),
        expression=str(row["expression"]),
        region=str(row["region"]),
        universe=str(row["universe"]),
        delay=int(row["delay"]),
        settings_json=str(row["settings_json"]),
        status=str(row["status"]),
        sharpe=_optional_float(row["sharpe"]),
        fitness=_optional_float(row["fitness"]),
        turnover=_optional_float(row["turnover"]),
        returns=_optional_float(row["returns"]),
        drawdown=_optional_float(row["drawdown"]),
        margin=_optional_float(row["margin"]),
        long_count=_optional_int(row["long_count"]),
        short_count=_optional_int(row["short_count"]),
        checks_json=str(row["checks_json"] or "[]"),
        fail_count=int(row["fail_count"] if row["fail_count"] is not None else 0),
        warn_count=int(row["warn_count"] if row["warn_count"] is not None else 0),
        gate1_pass=None if row["gate1_pass"] is None else bool(row["gate1_pass"]),
        gate2_pass=None if row["gate2_pass"] is None else bool(row["gate2_pass"]),
        self_corr_max=_optional_float(row["self_corr_max"]),
        prod_corr_max=_optional_float(row["prod_corr_max"]),
        power_pool_corr_max=_optional_float(row["power_pool_corr_max"]),
        pyramid_matches_json=str(row["pyramid_matches_json"] or "[]"),
        error_message=row["error_message"] or None,
        simulated_at=str(row["simulated_at"]),
    )


def _row_to_vault_entry(row: sqlite3.Row) -> VaultEntry:
    return VaultEntry(
        alpha_id=str(row["alpha_id"]),
        region=str(row["region"]),
        delay=int(row["delay"]),
        expression=str(row["expression"]),
        composite_score=float(row["composite_score"]),
        vault_status=str(row["vault_status"]),
        duplicate_of=row["duplicate_of"] or None,
        max_local_corr=_optional_float(row["max_local_corr"]),
        max_local_corr_peer=row["max_local_corr_peer"] or None,
        sharpe=_optional_float(row["sharpe"]),
        fitness=_optional_float(row["fitness"]),
        turnover=_optional_float(row["turnover"]),
        prod_corr_max=_optional_float(row["prod_corr_max"]),
        updated_at=str(row["updated_at"]),
    )


class AlphaDB:
    """Thread-safe SQLite WAL repository for alpha_factory state and catalogs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection configured with WAL mode and safety PRAGMAs."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @contextmanager
    def _write_conn(self) -> Iterator[sqlite3.Connection]:
        """Acquire write lock, wrap transaction, and close connection on exit."""
        with self._write_lock, contextlib.closing(self._connect()) as conn:
            with conn:
                yield conn

    @contextmanager
    def _read_conn(self) -> Iterator[sqlite3.Connection]:
        """Open a read connection and close it cleanly on exit."""
        with contextlib.closing(self._connect()) as conn:
            yield conn

    def _init_schema(self) -> None:
        with self._write_conn() as conn:
            conn.executescript(_SCHEMA_SQL)

    # ------------------------------------------------------------------
    # Catalog: datasets, datafields, operators
    # ------------------------------------------------------------------

    def upsert_datasets(
        self,
        datasets: Sequence[Mapping[str, Any] | Any],
        region: Optional[str] = None,
        universe: Optional[str] = None,
        delay: Optional[int] = None,
        instrument_type: Optional[str] = None,
    ) -> int:
        """Batch-upsert dataset metadata records in a single transaction."""
        if not datasets:
            return 0
        now = time.time()
        rows = []
        for item in datasets:
            ds_id = str(_field_get(item, "id", "dataset_id", default=""))
            ds_region = str(_field_get(item, "region", default=region or "USA"))
            ds_universe = str(_field_get(item, "universe", default=universe or "TOP3000"))
            ds_delay = int(_field_get(item, "delay", default=1 if delay is None else delay))
            ds_inst = str(
                _field_get(
                    item,
                    "instrument_type",
                    "instrumentType",
                    default=instrument_type or "EQUITY",
                )
            )
            rows.append(
                (
                    ds_id,
                    ds_region,
                    ds_universe,
                    ds_delay,
                    ds_inst,
                    str(_field_get(item, "name", default="")),
                    _extract_label(_field_get(item, "category", default="")),
                    _extract_label(_field_get(item, "subcategory", default="")),
                    float(_field_get(item, "coverage", default=0.0)),
                    int(_field_get(item, "field_count", "fieldCount", default=0)),
                    int(_field_get(item, "user_count", "userCount", default=0)),
                    float(_field_get(item, "value_score", "valueScore", default=0.0)),
                    int(_field_get(item, "alpha_count", "alphaCount", default=0)),
                    now,
                )
            )
        sql = """
            INSERT INTO datasets (
                id, region, universe, delay, instrument_type, name,
                category, subcategory, coverage, field_count,
                user_count, value_score, alpha_count, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id, region, universe, delay) DO UPDATE SET
                instrument_type = excluded.instrument_type,
                name = excluded.name,
                category = excluded.category,
                subcategory = excluded.subcategory,
                coverage = excluded.coverage,
                field_count = excluded.field_count,
                user_count = excluded.user_count,
                value_score = excluded.value_score,
                alpha_count = excluded.alpha_count,
                updated_at = excluded.updated_at
        """
        with self._write_conn() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def get_datasets(self, region: str, universe: str, delay: int) -> list[dict[str, Any]]:
        """Fetch datasets for a market tuple ordered by value_score descending."""
        with self._read_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM datasets
                WHERE region = ? AND universe = ? AND delay = ?
                ORDER BY value_score DESC, field_count DESC, id ASC
                """,
                (region, universe, int(delay)),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_datafields(
        self,
        datafields: Sequence[DataFieldRecord | Mapping[str, Any]],
        region: Optional[str] = None,
        universe: Optional[str] = None,
        delay: Optional[int] = None,
        dataset_id: Optional[str] = None,
    ) -> int:
        """Batch-upsert DataFieldRecord entries in a single transaction."""
        if not datafields:
            return 0
        now = time.time()
        rows = []
        for item in datafields:
            raw_dataset = _field_get(item, "dataset_id", "dataset", default=dataset_id or "")
            if isinstance(raw_dataset, Mapping):
                ds_id = str(raw_dataset.get("id") or dataset_id or "")
            else:
                ds_id = str(raw_dataset)

            rows.append(
                (
                    str(_field_get(item, "field_id", "id", default="")),
                    ds_id,
                    str(_field_get(item, "region", default=region or "USA")),
                    str(_field_get(item, "universe", default=universe or "TOP3000")),
                    int(_field_get(item, "delay", default=1 if delay is None else delay)),
                    str(_field_get(item, "field_type", "type", default="MATRIX")),
                    _extract_label(_field_get(item, "category", default="")),
                    _extract_label(_field_get(item, "subcategory", default="")),
                    float(_field_get(item, "coverage", default=0.0)),
                    int(_field_get(item, "user_count", "userCount", default=0)),
                    int(_field_get(item, "alpha_count", "alphaCount", default=0)),
                    str(_field_get(item, "description", default="")),
                    int(bool(_field_get(item, "is_event_input", "isEventInput", default=False))),
                    now,
                )
            )
        sql = """
            INSERT INTO datafields (
                field_id, dataset_id, region, universe, delay, field_type,
                category, subcategory, coverage, user_count, alpha_count,
                description, is_event_input, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(field_id, region, universe, delay) DO UPDATE SET
                dataset_id = excluded.dataset_id,
                field_type = excluded.field_type,
                category = excluded.category,
                subcategory = excluded.subcategory,
                coverage = excluded.coverage,
                user_count = excluded.user_count,
                alpha_count = excluded.alpha_count,
                description = excluded.description,
                is_event_input = excluded.is_event_input,
                updated_at = excluded.updated_at
        """
        with self._write_conn() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def get_datafields(
        self,
        region: str,
        universe: str,
        delay: int,
        category: Optional[str] = None,
    ) -> list[DataFieldRecord]:
        """Query data fields for a market tuple, optionally filtered by category."""
        sql = """
            SELECT * FROM datafields
            WHERE region = ? AND universe = ? AND delay = ?
        """
        params: list[Any] = [region, universe, int(delay)]
        if category is not None:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY coverage DESC, alpha_count DESC, field_id ASC"
        with self._read_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            DataFieldRecord(
                field_id=str(row["field_id"]),
                dataset_id=str(row["dataset_id"]),
                region=str(row["region"]),
                universe=str(row["universe"]),
                delay=int(row["delay"]),
                field_type=str(row["field_type"]),
                category=str(row["category"]),
                subcategory=str(row["subcategory"]),
                coverage=float(row["coverage"]),
                user_count=int(row["user_count"]),
                alpha_count=int(row["alpha_count"]),
                description=str(row["description"]),
                is_event_input=bool(row["is_event_input"]),
            )
            for row in rows
        ]

    def upsert_operators(self, operators: Sequence[OperatorRecord | Mapping[str, Any]]) -> int:
        """Batch-upsert OperatorRecord metadata in a single transaction."""
        if not operators:
            return 0
        now = time.time()
        rows = []
        for item in operators:
            raw_scope = _field_get(item, "scope_json", "scope", default="[]")
            scope_json = _coerce_json(raw_scope, "[]")
            rows.append(
                (
                    str(_field_get(item, "name", default="")),
                    _extract_label(_field_get(item, "category", default="")),
                    scope_json,
                    str(_field_get(item, "definition", default="")),
                    str(_field_get(item, "description", default="")),
                    int(_field_get(item, "num_inputs", "numInputs", default=1)),
                    int(_field_get(item, "min_inputs", "minInputs", default=1)),
                    int(bool(_field_get(item, "uses_lookback", "usesLookback", default=False))),
                    now,
                )
            )
        sql = """
            INSERT INTO operators (
                name, category, scope_json, definition, description,
                num_inputs, min_inputs, uses_lookback, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                category = excluded.category,
                scope_json = excluded.scope_json,
                definition = excluded.definition,
                description = excluded.description,
                num_inputs = excluded.num_inputs,
                min_inputs = excluded.min_inputs,
                uses_lookback = excluded.uses_lookback,
                updated_at = excluded.updated_at
        """
        with self._write_conn() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def get_operators(self) -> list[OperatorRecord]:
        """Return all stored FastExpr operators ordered by name."""
        with self._read_conn() as conn:
            rows = conn.execute("SELECT * FROM operators ORDER BY name ASC").fetchall()
        return [
            OperatorRecord(
                name=str(row["name"]),
                category=str(row["category"]),
                scope_json=str(row["scope_json"]),
                definition=str(row["definition"]),
                description=str(row["description"]),
                num_inputs=int(row["num_inputs"]),
                min_inputs=int(row["min_inputs"]),
                uses_lookback=bool(row["uses_lookback"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Candidate queue
    # ------------------------------------------------------------------

    def enqueue_candidates(self, items: Sequence[CandidateItem | Mapping[str, Any]]) -> int:
        """Insert candidate items ignoring duplicate config_hash rows; return inserted count."""
        if not items:
            return 0
        now = time.time()
        rows = []
        for item in items:
            created_at = _coerce_timestamp(_field_get(item, "created_at"), default=now)
            updated_at = _coerce_timestamp(_field_get(item, "updated_at"), default=now)
            settings_json = _coerce_json(_field_get(item, "settings_json", default="{}"), "{}")
            rows.append(
                (
                    str(_field_get(item, "config_hash", default="")),
                    str(_field_get(item, "ast_fingerprint", default="")),
                    str(_field_get(item, "expression", default="")),
                    str(_field_get(item, "region", default="USA")),
                    str(_field_get(item, "universe", default="TOP3000")),
                    int(_field_get(item, "delay", default=1)),
                    settings_json,
                    str(_field_get(item, "source", default="template")),
                    str(_field_get(item, "dataset_id", default="")),
                    float(_field_get(item, "priority", default=0.0)),
                    str(_field_get(item, "status", default="PENDING") or "PENDING"),
                    str(_field_get(item, "progress_url", default="")),
                    str(_field_get(item, "batch_id", default="")),
                    str(_field_get(item, "error_message", default="")),
                    created_at,
                    updated_at,
                )
            )
        sql = """
            INSERT OR IGNORE INTO candidate_queue (
                config_hash, ast_fingerprint, expression, region, universe, delay,
                settings_json, source, dataset_id, priority, status,
                progress_url, batch_id, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._write_conn() as conn:
            before = conn.total_changes
            conn.executemany(sql, rows)
            return conn.total_changes - before

    def count_queue(
        self,
        status: Optional[str] = "PENDING",
        region: Optional[str] = None,
    ) -> int:
        """Count queued candidates filtered by optional status and region."""
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if region is not None:
            clauses.append("region = ?")
            params.append(region)
        sql = "SELECT COUNT(*) FROM candidate_queue"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._read_conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row is not None else 0

    def has_config_hash(self, config_hash: str) -> bool:
        """Return True if config_hash exists in either candidate_queue or simulations."""
        with self._read_conn() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM candidate_queue WHERE config_hash = ?
                UNION ALL
                SELECT 1 FROM simulations WHERE config_hash = ?
                LIMIT 1
                """,
                (config_hash, config_hash),
            ).fetchone()
        return row is not None

    def count_fingerprint(self, ast_fingerprint: str, dataset_id: Optional[str] = None) -> int:
        """Count queued candidates sharing an AST fingerprint (optionally per dataset)."""
        if dataset_id is not None:
            sql = "SELECT COUNT(*) FROM candidate_queue WHERE ast_fingerprint = ? AND dataset_id = ?"
            params: tuple[Any, ...] = (ast_fingerprint, dataset_id)
        else:
            sql = "SELECT COUNT(*) FROM candidate_queue WHERE ast_fingerprint = ?"
            params = (ast_fingerprint,)
        with self._read_conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row is not None else 0

    def pop_pending_batch(
        self,
        region: str,
        universe: str,
        delay: int,
        limit: int = 10,
    ) -> list[CandidateItem]:
        """Atomically select highest-priority PENDING rows and mark them DISPATCHING."""
        if limit <= 0:
            return []
        with self._write_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM candidate_queue
                WHERE status = 'PENDING' AND region = ? AND universe = ? AND delay = ?
                ORDER BY priority DESC, id ASC
                LIMIT ?
                """,
                (region, universe, int(delay), int(limit)),
            ).fetchall()
            if not rows:
                return []
            ids = [int(row["id"]) for row in rows]
            now = time.time()
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE candidate_queue SET status = 'DISPATCHING', updated_at = ? "
                f"WHERE id IN ({placeholders})",
                [now, *ids],
            )
        items: list[CandidateItem] = []
        for row in rows:
            item = _row_to_candidate(row)
            item.status = "DISPATCHING"
            item.updated_at = str(now)
            items.append(item)
        return items

    def mark_in_flight(self, ids: Sequence[int], progress_url: str, batch_id: str) -> None:
        """Mark candidate queue rows as IN_FLIGHT with their polling URL and batch ID."""
        id_list = [int(i) for i in ids]
        if not id_list:
            return
        now = time.time()
        placeholders = ",".join("?" for _ in id_list)
        with self._write_conn() as conn:
            conn.execute(
                f"UPDATE candidate_queue "
                f"SET status = 'IN_FLIGHT', progress_url = ?, batch_id = ?, updated_at = ? "
                f"WHERE id IN ({placeholders})",
                [progress_url, batch_id, now, *id_list],
            )

    def get_in_flight_jobs(self) -> list[CandidateItem]:
        """Return all IN_FLIGHT candidate queue rows ordered by id ascending."""
        with self._read_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM candidate_queue WHERE status = 'IN_FLIGHT' ORDER BY id ASC"
            ).fetchall()
        return [_row_to_candidate(row) for row in rows]

    def mark_queue_done(self, ids: Sequence[int]) -> None:
        """Mark candidate queue rows as DONE."""
        id_list = [int(i) for i in ids]
        if not id_list:
            return
        now = time.time()
        placeholders = ",".join("?" for _ in id_list)
        with self._write_conn() as conn:
            conn.execute(
                f"UPDATE candidate_queue SET status = 'DONE', updated_at = ? "
                f"WHERE id IN ({placeholders})",
                [now, *id_list],
            )

    def mark_queue_failed(self, ids: Sequence[int], error_message: str = "") -> None:
        """Mark candidate queue rows as FAILED and store an error summary."""
        id_list = [int(i) for i in ids]
        if not id_list:
            return
        now = time.time()
        placeholders = ",".join("?" for _ in id_list)
        with self._write_conn() as conn:
            conn.execute(
                f"UPDATE candidate_queue SET status = 'FAILED', error_message = ?, updated_at = ? "
                f"WHERE id IN ({placeholders})",
                [str(error_message or ""), now, *id_list],
            )

    def requeue_items(self, ids: Sequence[int]) -> None:
        """Reset candidate queue rows back to PENDING with cleared execution metadata."""
        id_list = [int(i) for i in ids]
        if not id_list:
            return
        now = time.time()
        placeholders = ",".join("?" for _ in id_list)
        with self._write_conn() as conn:
            conn.execute(
                f"UPDATE candidate_queue "
                f"SET status = 'PENDING', progress_url = '', batch_id = '', "
                f"error_message = '', updated_at = ? "
                f"WHERE id IN ({placeholders})",
                [now, *id_list],
            )

    # ------------------------------------------------------------------
    # Simulations & submittable vault
    # ------------------------------------------------------------------

    def save_simulation(self, record: SimulationRecord | Mapping[str, Any]) -> None:
        """Upsert a SimulationRecord into simulations."""
        now = time.time()
        simulated_at = _coerce_timestamp(_field_get(record, "simulated_at"), default=now)
        params = (
            str(_field_get(record, "alpha_id", default="")),
            str(_field_get(record, "config_hash", default="")),
            str(_field_get(record, "ast_fingerprint", default="")),
            str(_field_get(record, "expression", default="")),
            str(_field_get(record, "region", default="USA")),
            str(_field_get(record, "universe", default="TOP3000")),
            int(_field_get(record, "delay", default=1)),
            _coerce_json(_field_get(record, "settings_json", default="{}"), "{}"),
            str(_field_get(record, "status", default="COMPLETE")),
            _optional_float(_field_get(record, "sharpe")),
            _optional_float(_field_get(record, "fitness")),
            _optional_float(_field_get(record, "turnover")),
            _optional_float(_field_get(record, "returns")),
            _optional_float(_field_get(record, "drawdown")),
            _optional_float(_field_get(record, "margin")),
            _optional_int(_field_get(record, "long_count")),
            _optional_int(_field_get(record, "short_count")),
            _coerce_json(_field_get(record, "checks_json", default="[]"), "[]"),
            int(_field_get(record, "fail_count", default=0)),
            int(_field_get(record, "warn_count", default=0)),
            _optional_bool_int(_field_get(record, "gate1_pass")),
            _optional_bool_int(_field_get(record, "gate2_pass")),
            _optional_float(_field_get(record, "self_corr_max")),
            _optional_float(_field_get(record, "prod_corr_max")),
            _optional_float(_field_get(record, "power_pool_corr_max")),
            _coerce_json(_field_get(record, "pyramid_matches_json", default="[]"), "[]"),
            str(_field_get(record, "error_message", default="") or ""),
            simulated_at,
        )
        sql = """
            INSERT INTO simulations (
                alpha_id, config_hash, ast_fingerprint, expression, region, universe,
                delay, settings_json, status, sharpe, fitness, turnover, returns,
                drawdown, margin, long_count, short_count, checks_json, fail_count,
                warn_count, gate1_pass, gate2_pass, self_corr_max, prod_corr_max,
                power_pool_corr_max, pyramid_matches_json, error_message, simulated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(alpha_id) DO UPDATE SET
                config_hash = excluded.config_hash,
                ast_fingerprint = excluded.ast_fingerprint,
                expression = excluded.expression,
                region = excluded.region,
                universe = excluded.universe,
                delay = excluded.delay,
                settings_json = excluded.settings_json,
                status = excluded.status,
                sharpe = excluded.sharpe,
                fitness = excluded.fitness,
                turnover = excluded.turnover,
                returns = excluded.returns,
                drawdown = excluded.drawdown,
                margin = excluded.margin,
                long_count = excluded.long_count,
                short_count = excluded.short_count,
                checks_json = excluded.checks_json,
                fail_count = excluded.fail_count,
                warn_count = excluded.warn_count,
                gate1_pass = excluded.gate1_pass,
                gate2_pass = excluded.gate2_pass,
                self_corr_max = excluded.self_corr_max,
                prod_corr_max = excluded.prod_corr_max,
                power_pool_corr_max = excluded.power_pool_corr_max,
                pyramid_matches_json = excluded.pyramid_matches_json,
                error_message = excluded.error_message,
                simulated_at = excluded.simulated_at
        """
        with self._write_conn() as conn:
            conn.execute(sql, params)

    def get_simulation(self, alpha_id: str) -> Optional[SimulationRecord]:
        """Fetch a single SimulationRecord by alpha_id, or None if missing."""
        with self._read_conn() as conn:
            row = conn.execute(
                "SELECT * FROM simulations WHERE alpha_id = ?",
                (alpha_id,),
            ).fetchone()
        return None if row is None else _row_to_simulation(row)

    def get_simulations(
        self,
        region: Optional[str] = None,
        gate1_pass: Optional[bool] = None,
        gate2_pass: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> list[SimulationRecord]:
        """Query simulations filtered by optional region and gate pass flags."""
        clauses: list[str] = []
        params: list[Any] = []
        if region is not None:
            clauses.append("region = ?")
            params.append(region)
        if gate1_pass is not None:
            clauses.append("gate1_pass = ?")
            params.append(int(bool(gate1_pass)))
        if gate2_pass is not None:
            clauses.append("gate2_pass = ?")
            params.append(int(bool(gate2_pass)))

        sql = "SELECT * FROM simulations"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY simulated_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        with self._read_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_simulation(row) for row in rows]

    def update_simulation_qualification(
        self,
        alpha_id: str,
        gate1_pass: Optional[bool],
        gate2_pass: Optional[bool],
        self_corr_max: Optional[float] = None,
        prod_corr_max: Optional[float] = None,
        power_pool_corr_max: Optional[float] = None,
        pyramid_matches_json: str | Sequence[Any] = "[]",
    ) -> None:
        """Update Gate 1/Gate 2 qualification metrics for an existing simulation."""
        pyramid_str = _coerce_json(pyramid_matches_json, "[]")
        with self._write_conn() as conn:
            conn.execute(
                """
                UPDATE simulations
                SET gate1_pass = ?,
                    gate2_pass = ?,
                    self_corr_max = ?,
                    prod_corr_max = ?,
                    power_pool_corr_max = ?,
                    pyramid_matches_json = ?
                WHERE alpha_id = ?
                """,
                (
                    _optional_bool_int(gate1_pass),
                    _optional_bool_int(gate2_pass),
                    _optional_float(self_corr_max),
                    _optional_float(prod_corr_max),
                    _optional_float(power_pool_corr_max),
                    pyramid_str,
                    alpha_id,
                ),
            )

    def upsert_vault_entry(self, entry: VaultEntry | Mapping[str, Any]) -> None:
        """Insert or update a qualified alpha in submittable_vault."""
        now = time.time()
        updated_at = _coerce_timestamp(_field_get(entry, "updated_at"), default=now)
        params = (
            str(_field_get(entry, "alpha_id", default="")),
            str(_field_get(entry, "region", default="USA")),
            int(_field_get(entry, "delay", default=1)),
            str(_field_get(entry, "expression", default="")),
            float(_field_get(entry, "composite_score", default=0.0)),
            str(_field_get(entry, "vault_status", default="ACTIVE")),
            str(_field_get(entry, "duplicate_of", default="") or ""),
            _optional_float(_field_get(entry, "max_local_corr")),
            str(_field_get(entry, "max_local_corr_peer", default="") or ""),
            _optional_float(_field_get(entry, "sharpe")),
            _optional_float(_field_get(entry, "fitness")),
            _optional_float(_field_get(entry, "turnover")),
            _optional_float(_field_get(entry, "prod_corr_max")),
            updated_at,
        )
        sql = """
            INSERT INTO submittable_vault (
                alpha_id, region, delay, expression, composite_score, vault_status,
                duplicate_of, max_local_corr, max_local_corr_peer,
                sharpe, fitness, turnover, prod_corr_max, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(alpha_id) DO UPDATE SET
                region = excluded.region,
                delay = excluded.delay,
                expression = excluded.expression,
                composite_score = excluded.composite_score,
                vault_status = excluded.vault_status,
                duplicate_of = excluded.duplicate_of,
                max_local_corr = excluded.max_local_corr,
                max_local_corr_peer = excluded.max_local_corr_peer,
                sharpe = excluded.sharpe,
                fitness = excluded.fitness,
                turnover = excluded.turnover,
                prod_corr_max = excluded.prod_corr_max,
                updated_at = excluded.updated_at
        """
        with self._write_conn() as conn:
            conn.execute(sql, params)

    def get_vault_entries(
        self,
        region: Optional[str] = None,
        delay: Optional[int] = None,
        statuses: Optional[Sequence[str] | str] = None,
        limit: Optional[int] = None,
    ) -> list[VaultEntry]:
        """Fetch vault entries ordered by composite_score descending."""
        clauses: list[str] = []
        params: list[Any] = []
        if region is not None:
            clauses.append("region = ?")
            params.append(region)
        if delay is not None:
            clauses.append("delay = ?")
            params.append(int(delay))
        if statuses is not None:
            status_list = [statuses] if isinstance(statuses, str) else list(statuses)
            if not status_list:
                return []
            placeholders = ",".join("?" for _ in status_list)
            clauses.append(f"vault_status IN ({placeholders})")
            params.extend(status_list)

        sql = "SELECT * FROM submittable_vault"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY composite_score DESC, updated_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        with self._read_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_vault_entry(row) for row in rows]

    def demote_vault_entry(
        self,
        alpha_id: str,
        duplicate_of: str,
        max_local_corr: float,
        peer_id: str,
    ) -> None:
        """Mark an existing vault entry as DUPLICATE with correlation attribution."""
        now = time.time()
        with self._write_conn() as conn:
            conn.execute(
                """
                UPDATE submittable_vault
                SET vault_status = 'DUPLICATE',
                    duplicate_of = ?,
                    max_local_corr = ?,
                    max_local_corr_peer = ?,
                    updated_at = ?
                WHERE alpha_id = ?
                """,
                (
                    str(duplicate_of),
                    float(max_local_corr),
                    str(peer_id),
                    now,
                    alpha_id,
                ),
            )

    # ------------------------------------------------------------------
    # Daily quota
    # ------------------------------------------------------------------

    def get_daily_quota(self, date_ny: str) -> dict[str, Any]:
        """Return New York-date quota counts as a dictionary."""
        with self._read_conn() as conn:
            row = conn.execute(
                "SELECT * FROM daily_quota WHERE date_ny = ?",
                (date_ny,),
            ).fetchone()
        if row is None:
            return {
                "date_ny": date_ny,
                "submitted_count": 0,
                "completed_count": 0,
                "failed_count": 0,
                "warning_sent": False,
                "warning_triggered": False,
                "updated_at": 0.0,
            }
        warning_sent = bool(row["warning_sent"])
        return {
            "date_ny": str(row["date_ny"]),
            "submitted_count": int(row["submitted_count"]),
            "completed_count": int(row["completed_count"]),
            "failed_count": int(row["failed_count"]),
            "warning_sent": warning_sent,
            "warning_triggered": warning_sent,
            "updated_at": float(row["updated_at"]),
        }

    def increment_daily_quota_submitted(
        self,
        date_ny: str,
        count: int,
        warning_threshold: int = 4000,
        daily_limit: int = 5000,
    ) -> DailyQuotaStatus:
        """Atomically increment daily submitted count and return a DailyQuotaStatus snapshot."""
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        now = time.time()
        with self._write_conn() as conn:
            if count > 0:
                conn.execute(
                    """
                    INSERT INTO daily_quota (
                        date_ny, submitted_count, completed_count, failed_count, warning_sent, updated_at
                    ) VALUES (?, ?, 0, 0, CASE WHEN ? >= ? THEN 1 ELSE 0 END, ?)
                    ON CONFLICT(date_ny) DO UPDATE SET
                        submitted_count = daily_quota.submitted_count + excluded.submitted_count,
                        warning_sent = CASE
                            WHEN daily_quota.warning_sent = 1
                                 OR (daily_quota.submitted_count + excluded.submitted_count) >= ?
                            THEN 1 ELSE 0
                        END,
                        updated_at = excluded.updated_at
                    """,
                    (
                        date_ny,
                        int(count),
                        int(count),
                        int(warning_threshold),
                        now,
                        int(warning_threshold),
                    ),
                )
            row = conn.execute(
                "SELECT * FROM daily_quota WHERE date_ny = ?",
                (date_ny,),
            ).fetchone()

        if row is None:
            submitted = 0
            completed = 0
            failed = 0
            stored_warning = False
        else:
            submitted = int(row["submitted_count"])
            completed = int(row["completed_count"])
            failed = int(row["failed_count"])
            stored_warning = bool(row["warning_sent"])

        limit_int = int(daily_limit)
        return DailyQuotaStatus(
            date_ny=date_ny,
            submitted_count=submitted,
            completed_count=completed,
            failed_count=failed,
            limit=limit_int,
            remaining=max(0, limit_int - submitted),
            warning_triggered=stored_warning or (submitted >= int(warning_threshold)),
            can_submit=submitted < limit_int,
        )

    def increment_daily_quota_outcome(
        self,
        date_ny: str,
        completed: int = 0,
        failed: int = 0,
    ) -> None:
        """Atomically increment terminal completed/failed counts for a NY date."""
        if completed < 0 or failed < 0:
            raise ValueError(
                f"completed and failed must be non-negative, got completed={completed}, failed={failed}"
            )
        if completed == 0 and failed == 0:
            return
        now = time.time()
        with self._write_conn() as conn:
            conn.execute(
                """
                INSERT INTO daily_quota (
                    date_ny, submitted_count, completed_count, failed_count, warning_sent, updated_at
                ) VALUES (?, 0, ?, ?, 0, ?)
                ON CONFLICT(date_ny) DO UPDATE SET
                    completed_count = daily_quota.completed_count + excluded.completed_count,
                    failed_count = daily_quota.failed_count + excluded.failed_count,
                    updated_at = excluded.updated_at
                """,
                (date_ny, int(completed), int(failed), now),
            )

    # ------------------------------------------------------------------
    # Compiler knowledge & Thompson Sampling bandit arms
    # ------------------------------------------------------------------

    def record_compiler_rule(
        self,
        rule_type: str,
        operator_name: str = "",
        field_id: str = "",
        error_pattern: str = "",
        replacement: str = "",
        increment: int = 1,
    ) -> None:
        """Insert or increment an observed compiler normalization/error rule."""
        now = time.time()
        with self._write_conn() as conn:
            conn.execute(
                """
                INSERT INTO compiler_knowledge (
                    rule_type, operator_name, field_id, error_pattern,
                    replacement, occurrences, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rule_type, operator_name, field_id, error_pattern) DO UPDATE SET
                    occurrences = compiler_knowledge.occurrences + excluded.occurrences,
                    replacement = CASE
                        WHEN excluded.replacement != '' THEN excluded.replacement
                        ELSE compiler_knowledge.replacement
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    str(rule_type),
                    str(operator_name or ""),
                    str(field_id or ""),
                    str(error_pattern or ""),
                    str(replacement or ""),
                    int(increment),
                    now,
                ),
            )

    def get_compiler_rules(
        self,
        rule_type: Optional[str] = None,
        operator_name: Optional[str] = None,
        field_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Query compiler knowledge rules ordered by occurrences descending."""
        clauses: list[str] = []
        params: list[Any] = []
        if rule_type is not None:
            clauses.append("rule_type = ?")
            params.append(rule_type)
        if operator_name is not None:
            clauses.append("operator_name = ?")
            params.append(operator_name)
        if field_id is not None:
            clauses.append("field_id = ?")
            params.append(field_id)

        sql = "SELECT * FROM compiler_knowledge"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY occurrences DESC, updated_at DESC"
        with self._read_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_bandit_arms(self) -> list[dict[str, Any]]:
        """Return all persisted Thompson Sampling bandit arms."""
        with self._read_conn() as conn:
            rows = conn.execute("SELECT * FROM bandit_arms ORDER BY arm_key ASC").fetchall()
        return [dict(row) for row in rows]

    def update_bandit_arm(
        self,
        arm_key: str,
        reward: float,
        cooldown_until: float = 0.0,
    ) -> dict[str, Any]:
        """Update a Beta-Bernoulli bandit arm with an observed reward and cooldown."""
        raw_reward = float(reward)
        clamped = min(1.0, max(0.0, raw_reward))
        alpha_inc = clamped
        beta_inc = 1.0 - clamped
        cooldown = float(cooldown_until)
        now = time.time()

        with self._write_conn() as conn:
            conn.execute(
                """
                INSERT INTO bandit_arms (
                    arm_key, alpha_param, beta_param, pulls, total_reward, cooldown_until, updated_at
                ) VALUES (?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(arm_key) DO UPDATE SET
                    alpha_param = bandit_arms.alpha_param + ?,
                    beta_param = bandit_arms.beta_param + ?,
                    pulls = bandit_arms.pulls + 1,
                    total_reward = bandit_arms.total_reward + ?,
                    cooldown_until = ?,
                    updated_at = ?
                """,
                (
                    arm_key,
                    1.0 + alpha_inc,
                    1.0 + beta_inc,
                    raw_reward,
                    cooldown,
                    now,
                    alpha_inc,
                    beta_inc,
                    raw_reward,
                    cooldown,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM bandit_arms WHERE arm_key = ?",
                (arm_key,),
            ).fetchone()
        return dict(row) if row is not None else {}

    # ------------------------------------------------------------------
    # Ops: backup, dashboard stats, runtime key-value state
    # ------------------------------------------------------------------

    def backup_to(self, dest_path: str | Path) -> Path:
        """Create an atomic online SQLite backup at ``dest_path`` using ``Connection.backup``."""
        dest = Path(dest_path).expanduser()
        if str(self.db_path) != ":memory:" and dest.resolve() == self.db_path.resolve():
            raise ValueError("Backup destination must differ from the active database path")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_dest = dest.with_name(f".{dest.name}.tmp.{os.getpid()}.{threading.get_ident()}")
        try:
            with self._write_lock, contextlib.closing(self._connect()) as src_conn:
                with contextlib.closing(sqlite3.connect(str(tmp_dest))) as dst_conn:
                    src_conn.backup(dst_conn)
            os.replace(tmp_dest, dest)
        except Exception:
            tmp_dest.unlink(missing_ok=True)
            raise
        return dest

    def get_dashboard_stats(self, date_ny: Optional[str] = None) -> dict[str, Any]:
        """Return aggregated queue, simulation, vault, catalog, and quota metrics."""
        target_date = date_ny or datetime.now(NY_TZ).strftime("%Y-%m-%d")
        queue_counts = {
            "PENDING": 0,
            "DISPATCHING": 0,
            "IN_FLIGHT": 0,
            "DONE": 0,
            "FAILED": 0,
            "total": 0,
        }
        with self._read_conn() as conn:
            for row in conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM candidate_queue GROUP BY status"
            ).fetchall():
                status_name = str(row["status"])
                cnt = int(row["cnt"])
                queue_counts[status_name] = cnt
                queue_counts["total"] += cnt

            sim_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN status = 'COMPLETE' THEN 1 ELSE 0 END) AS complete,
                    SUM(CASE WHEN status != 'COMPLETE' THEN 1 ELSE 0 END) AS failed,
                    SUM(CASE WHEN gate1_pass = 1 THEN 1 ELSE 0 END) AS gate1_pass,
                    SUM(CASE WHEN gate2_pass = 1 THEN 1 ELSE 0 END) AS gate2_pass
                FROM simulations
                """
            ).fetchone()

            vault_counts: dict[str, int] = {"ACTIVE": 0, "DUPLICATE": 0, "total": 0}
            for row in conn.execute(
                "SELECT vault_status, COUNT(*) AS cnt FROM submittable_vault GROUP BY vault_status"
            ).fetchall():
                v_status = str(row["vault_status"])
                cnt = int(row["cnt"])
                vault_counts[v_status] = cnt
                vault_counts["total"] += cnt

            ds_cnt = int(conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0])
            df_cnt = int(conn.execute("SELECT COUNT(*) FROM datafields").fetchone()[0])
            op_cnt = int(conn.execute("SELECT COUNT(*) FROM operators").fetchone()[0])

            quota_row = conn.execute(
                "SELECT * FROM daily_quota WHERE date_ny = ?",
                (target_date,),
            ).fetchone()

        if quota_row is None:
            quota_dict: dict[str, Any] = {
                "date_ny": target_date,
                "submitted_count": 0,
                "completed_count": 0,
                "failed_count": 0,
                "warning_sent": False,
                "warning_triggered": False,
                "updated_at": 0.0,
            }
        else:
            warn_sent = bool(quota_row["warning_sent"])
            quota_dict = {
                "date_ny": str(quota_row["date_ny"]),
                "submitted_count": int(quota_row["submitted_count"]),
                "completed_count": int(quota_row["completed_count"]),
                "failed_count": int(quota_row["failed_count"]),
                "warning_sent": warn_sent,
                "warning_triggered": warn_sent,
                "updated_at": float(quota_row["updated_at"]),
            }

        sim_counts = {
            "total": int(sim_row["total"] or 0) if sim_row is not None else 0,
            "complete": int(sim_row["complete"] or 0) if sim_row is not None else 0,
            "failed": int(sim_row["failed"] or 0) if sim_row is not None else 0,
            "gate1_pass": int(sim_row["gate1_pass"] or 0) if sim_row is not None else 0,
            "gate2_pass": int(sim_row["gate2_pass"] or 0) if sim_row is not None else 0,
        }

        return {
            "date_ny": target_date,
            "quota": quota_dict,
            "queue": queue_counts,
            "simulations": sim_counts,
            "vault": vault_counts,
            "catalog": {
                "datasets": ds_cnt,
                "datafields": df_cnt,
                "operators": op_cnt,
            },
        }

    def set_runtime_state(self, key: str, value: Any) -> None:
        """Persist a key-value entry in runtime_state."""
        if isinstance(value, str):
            serialized = value
        elif isinstance(value, (dict, list)):
            serialized = json.dumps(value, sort_keys=True)
        else:
            serialized = str(value)
        now = time.time()
        with self._write_conn() as conn:
            conn.execute(
                """
                INSERT INTO runtime_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (str(key), serialized, now),
            )

    def get_runtime_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Read a runtime_state value by key, returning ``default`` if absent."""
        with self._read_conn() as conn:
            row = conn.execute(
                "SELECT value FROM runtime_state WHERE key = ?",
                (str(key),),
            ).fetchone()
        return default if row is None else str(row["value"])


AlphaDatabase = AlphaDB

__all__ = [
    "AlphaDB",
    "AlphaDatabase",
]
