"""
session_history.py
==================
SQLite-backed session history for the WorldQuant ACE alpha simulator.

Tracks which alpha expressions have already been simulated so that
restarting a session with the same batch of alphas automatically
resumes from where it left off.
"""

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def hash_alpha(alpha_dict: dict) -> str:
    """
    Produce a deterministic SHA-256 hex digest for an alpha configuration.

    Sorts keys so that insertion order doesn't affect the hash.
    """
    canonical = json.dumps(alpha_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS simulation_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    expression      TEXT    NOT NULL,
    alpha_hash      TEXT    NOT NULL,
    alpha_id        TEXT,
    status          TEXT    NOT NULL,
    region          TEXT    NOT NULL,
    universe        TEXT    NOT NULL,
    settings_json   TEXT,
    error_message   TEXT,
    created_at      TEXT    NOT NULL,
    UNIQUE(alpha_hash, region, universe)
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_hash_region_universe
    ON simulation_history (alpha_hash, region, universe);
"""


class SessionHistoryDB:
    """
    SQLite-backed store that records every simulation attempt so that
    subsequent sessions can skip already-simulated alphas.

    Usage::

        db = SessionHistoryDB()                       # uses default path
        remaining = db.filter_unsimulated(alpha_list, region, universe)
        # ... simulate remaining ...
        db.record_simulation(expression, alpha_hash, alpha_id, 'SUCCESS',
                             region, universe)
    """

    def __init__(self, db_path: str = "simulation_history.db"):
        self.db_path = db_path
        self._init_db()

    # -- internal helpers ---------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
        logger.info("Session history DB ready at %s", self.db_path)

    # -- public API ---------------------------------------------------------

    def record_simulation(
        self,
        expression: str,
        alpha_hash: str,
        alpha_id: Optional[str],
        status: str,
        region: str,
        universe: str,
        settings_json: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Insert a simulation result.  Silently ignores duplicates
        (same hash + region + universe).
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO simulation_history
                    (expression, alpha_hash, alpha_id, status,
                     region, universe, settings_json, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    expression,
                    alpha_hash,
                    alpha_id,
                    status,
                    region,
                    universe,
                    settings_json,
                    error_message,
                    now,
                ),
            )

    def get_simulated_hashes(self, region: str, universe: str) -> set:
        """Return all alpha hashes already recorded for this region/universe."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT alpha_hash FROM simulation_history WHERE region = ? AND universe = ?",
                (region, universe),
            ).fetchall()
        return {r[0] for r in rows}

    def filter_unsimulated(
        self,
        alpha_list: list[dict],
        region: str,
        universe: str,
    ) -> list[dict]:
        """
        Given a full ``alpha_list`` (each element is an alpha dict from
        ``ace.generate_alpha``), return only those that have **not** yet
        been recorded in the DB for the given region/universe.
        """
        already = self.get_simulated_hashes(region, universe)
        remaining = []
        skipped = 0
        for alpha_dict in alpha_list:
            h = hash_alpha(alpha_dict)
            if h in already:
                skipped += 1
            else:
                remaining.append(alpha_dict)
        if skipped:
            logger.info(
                "Skipping %d already-simulated alphas (%d remaining)",
                skipped,
                len(remaining),
            )
        return remaining

    def record_batch_results(
        self,
        batch_alphas: list[dict],
        results: list[dict],
        region: str,
        universe: str,
    ) -> None:
        """
        Record simulation results for an entire batch.

        ``batch_alphas`` is the list of alpha dicts that were submitted.
        ``results``      is the list returned by
                         ``ace.simulate_alpha_list_multi()``.

        We pair them up by matching the ``simulate_data`` field in each
        result dict back to the original alpha dict.
        """
        # Build a lookup from hash -> result
        result_by_hash: dict[str, dict] = {}
        for r in results:
            if isinstance(r, dict) and r.get("simulate_data"):
                h = hash_alpha(r["simulate_data"])
                result_by_hash[h] = r

        for alpha_dict in batch_alphas:
            h = hash_alpha(alpha_dict)
            expression = (
                alpha_dict.get("regular")
                or alpha_dict.get("selection", "")
            )
            settings_json = json.dumps(
                alpha_dict.get("settings", {}), sort_keys=True
            )

            r = result_by_hash.get(h)
            if r and r.get("alpha_id"):
                self.record_simulation(
                    expression=expression,
                    alpha_hash=h,
                    alpha_id=r["alpha_id"],
                    status="SUCCESS",
                    region=region,
                    universe=universe,
                    settings_json=settings_json,
                )
            elif r:
                # result exists but no alpha_id → error / cancelled
                err = r.get("error_message") or r.get("message") or ""
                self.record_simulation(
                    expression=expression,
                    alpha_hash=h,
                    alpha_id=None,
                    status="ERROR",
                    region=region,
                    universe=universe,
                    settings_json=settings_json,
                    error_message=str(err),
                )
            else:
                # no matching result at all → mark as error
                self.record_simulation(
                    expression=expression,
                    alpha_hash=h,
                    alpha_id=None,
                    status="ERROR",
                    region=region,
                    universe=universe,
                    settings_json=settings_json,
                    error_message="No result returned from simulation",
                )

    def get_session_summary(self, region: str, universe: str) -> dict:
        """
        Return a summary dict:
        ``{ total, success, error, cancelled }``
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) FROM simulation_history
                WHERE region = ? AND universe = ?
                GROUP BY status
                """,
                (region, universe),
            ).fetchall()
        counts = {r[0]: r[1] for r in rows}
        return {
            "total": sum(counts.values()),
            "success": counts.get("SUCCESS", 0),
            "error": counts.get("ERROR", 0),
            "cancelled": counts.get("CANCELLED", 0),
        }

    def get_batch_history(self, region: str, universe: str) -> pd.DataFrame:
        """Return all records for this region/universe as a DataFrame."""
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM simulation_history WHERE region = ? AND universe = ?",
                conn,
                params=(region, universe),
            )
        return df
