"""Phase 1 tests for AlphaDB, DailyQuotaManager, and PnlStore."""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from alpha_factory.brain.client import parse_recordset
from alpha_factory.execution.quota import DailyQuotaManager, NY_TZ
from alpha_factory.storage.db import AlphaDB, AlphaDatabase
from alpha_factory.storage.models import CandidateItem, SimulationRecord
from alpha_factory.storage.pnl_store import PnlStore, extract_daily_returns_from_recordset


# ---------------------------------------------------------------------------
# AlphaDB: WAL mode, concurrent writes, atomic pop_pending_batch
# ---------------------------------------------------------------------------


def test_alphadb_wal_mode_and_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "wal_test.db"
    db = AlphaDB(db_path)

    assert AlphaDatabase is AlphaDB
    assert db_path.is_file()

    with db._read_conn() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        sync_mode = conn.execute("PRAGMA synchronous;").fetchone()[0]
        fk_mode = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    assert str(journal_mode).lower() == "wal"
    assert int(sync_mode) == 1  # NORMAL
    assert int(fk_mode) == 1

    expected_tables = {
        "datasets",
        "datafields",
        "operators",
        "candidate_queue",
        "simulations",
        "submittable_vault",
        "daily_quota",
        "compiler_knowledge",
        "bandit_arms",
        "runtime_state",
    }
    assert expected_tables.issubset(tables)

    # Confirm an independent SQLite connection also observes WAL mode on the file.
    raw_conn = sqlite3.connect(str(db_path))
    try:
        assert str(raw_conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower() == "wal"
    finally:
        raw_conn.close()


def test_alphadb_concurrent_multithreaded_writes(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "concurrent_writes.db")
    num_threads = 8
    candidates_per_thread = 15
    sims_per_thread = 10
    quota_inc_per_thread = 25
    bandit_updates_per_thread = 5

    barrier = threading.Barrier(num_threads)

    def _worker(worker_id: int) -> None:
        barrier.wait(timeout=5.0)
        batch = [
            CandidateItem(
                config_hash=f"cfg-{worker_id}-{i}",
                ast_fingerprint=f"fp-{i % 3}",
                expression=f"rank(close) + {worker_id * 100 + i}",
                region="USA",
                universe="TOP3000",
                delay=1,
                source="template",
                priority=float(i),
            )
            for i in range(candidates_per_thread)
        ]
        inserted = db.enqueue_candidates(batch)
        assert inserted == candidates_per_thread

        for j in range(sims_per_thread):
            db.save_simulation(
                SimulationRecord(
                    alpha_id=f"alpha-{worker_id}-{j}",
                    config_hash=f"simcfg-{worker_id}-{j}",
                    ast_fingerprint="fp-sim",
                    expression=f"rank(volume) + {j}",
                    region="USA",
                    universe="TOP3000",
                    delay=1,
                    status="COMPLETE",
                    sharpe=2.1,
                    fitness=1.3,
                )
            )

        for _ in range(quota_inc_per_thread):
            db.increment_daily_quota_submitted("2026-06-15", 1, warning_threshold=4000)

        for _ in range(bandit_updates_per_thread):
            db.update_bandit_arm("USA:fundamental:template", reward=0.75)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(_worker, idx) for idx in range(num_threads)]
        for fut in futures:
            fut.result()

    assert db.count_queue(status="PENDING") == num_threads * candidates_per_thread
    assert len(db.get_simulations()) == num_threads * sims_per_thread
    quota = db.get_daily_quota("2026-06-15")
    assert quota["submitted_count"] == num_threads * quota_inc_per_thread
    arms = db.get_bandit_arms()
    assert len(arms) == 1
    assert arms[0]["pulls"] == num_threads * bandit_updates_per_thread


def test_alphadb_atomic_pop_pending_batch_no_duplicates(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "atomic_pop.db")
    total_usa = 64
    usa_candidates = [
        CandidateItem(
            config_hash=f"usa-hash-{i}",
            ast_fingerprint=f"fp-{i % 4}",
            expression=f"rank(close) * {i + 1}",
            region="USA",
            universe="TOP3000",
            delay=1,
            priority=float(i),
        )
        for i in range(total_usa)
    ]
    eur_candidates = [
        CandidateItem(
            config_hash=f"eur-hash-{i}",
            ast_fingerprint="fp-eur",
            expression=f"rank(open) + {i}",
            region="EUR",
            universe="TOP2500",
            delay=1,
            priority=float(i),
        )
        for i in range(5)
    ]
    assert db.enqueue_candidates(usa_candidates + eur_candidates) == total_usa + 5

    num_threads = 8
    barrier = threading.Barrier(num_threads)

    def _pop_loop(_tid: int) -> list[CandidateItem]:
        barrier.wait(timeout=5.0)
        popped: list[CandidateItem] = []
        while True:
            batch = db.pop_pending_batch("USA", "TOP3000", delay=1, limit=3)
            if not batch:
                return popped
            popped.extend(batch)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        thread_results = list(pool.map(_pop_loop, range(num_threads)))

    all_popped = [item for sub in thread_results for item in sub]
    popped_ids = [item.id for item in all_popped]
    popped_hashes = [item.config_hash for item in all_popped]

    assert len(all_popped) == total_usa
    assert len(set(popped_ids)) == total_usa
    assert len(set(popped_hashes)) == total_usa
    assert all(item.status == "DISPATCHING" for item in all_popped)
    assert db.count_queue(status="PENDING", region="USA") == 0
    assert db.count_queue(status="DISPATCHING", region="USA") == total_usa
    assert db.count_queue(status="PENDING", region="EUR") == 5


def test_alphadb_pop_priority_and_queue_lifecycle(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "lifecycle.db")
    inserted = db.enqueue_candidates(
        [
            CandidateItem(config_hash="h1", ast_fingerprint="f1", expression="e1", priority=1.0),
            CandidateItem(config_hash="h2", ast_fingerprint="f2", expression="e2", priority=10.0),
            CandidateItem(config_hash="h3", ast_fingerprint="f3", expression="e3", priority=5.0),
            # Duplicate config_hash should be ignored
            CandidateItem(config_hash="h2", ast_fingerprint="f2", expression="e2", priority=99.0),
        ]
    )
    assert inserted == 3
    assert db.has_config_hash("h2") is True

    popped = db.pop_pending_batch("USA", "TOP3000", delay=1, limit=2)
    assert [item.config_hash for item in popped] == ["h2", "h3"]

    ids = [int(item.id) for item in popped if item.id is not None]
    db.mark_in_flight(ids, progress_url="https://api/sim/b1", batch_id="b1")
    in_flight = db.get_in_flight_jobs()
    assert [item.config_hash for item in in_flight] == ["h2", "h3"]

    db.mark_queue_done([ids[0]])
    db.mark_queue_failed([ids[1]], error_message="timeout")
    assert db.count_queue(status="DONE") == 1
    assert db.count_queue(status="FAILED") == 1

    db.requeue_items([ids[1]])
    assert db.count_queue(status="PENDING") == 2


# ---------------------------------------------------------------------------
# DailyQuotaManager: NY timezone, DST-safe midnight, 4k warn, 5k stop, pacing
# ---------------------------------------------------------------------------


def test_quota_ny_timezone_boundaries(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "quota_tz.db")
    now_ref = [dt.datetime(2026, 6, 15, 3, 59, 59, tzinfo=dt.timezone.utc)]
    qm = DailyQuotaManager(db=db, now_fn=lambda: now_ref[0])

    # 03:59:59 UTC in June (EDT = UTC-4) is still 23:59:59 on 2026-06-14 in NY
    assert qm.get_ny_date() == "2026-06-14"

    # Two seconds later it is 2026-06-15 00:00:01 EDT
    now_ref[0] = dt.datetime(2026, 6, 15, 4, 0, 1, tzinfo=dt.timezone.utc)
    assert qm.get_ny_date() == "2026-06-15"

    # Winter EST (UTC-5): 04:59:59 UTC is 23:59:59 previous NY day
    now_ref[0] = dt.datetime(2026, 1, 15, 4, 59, 59, tzinfo=dt.timezone.utc)
    assert qm.get_ny_date() == "2026-01-14"

    now_ref[0] = dt.datetime(2026, 1, 15, 5, 0, 0, tzinfo=dt.timezone.utc)
    assert qm.get_ny_date() == "2026-01-15"


def test_quota_seconds_until_ny_midnight_dst_safe(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "quota_dst.db")
    now_ref = [dt.datetime(2026, 3, 8, 0, 0, 0, tzinfo=NY_TZ)]
    qm = DailyQuotaManager(db=db, now_fn=lambda: now_ref[0])

    # 2026-03-08 is US Spring Forward (2:00 AM -> 3:00 AM): 23-hour day in NY
    assert qm.seconds_until_ny_midnight() == pytest.approx(23 * 3600.0)

    now_ref[0] = dt.datetime(2026, 3, 8, 1, 30, 0, tzinfo=NY_TZ)
    assert qm.seconds_until_ny_midnight() == pytest.approx(21.5 * 3600.0)

    # 2026-11-01 is US Fall Back (2:00 AM -> 1:00 AM): 25-hour day in NY
    now_ref[0] = dt.datetime(2026, 11, 1, 0, 0, 0, tzinfo=NY_TZ)
    assert qm.seconds_until_ny_midnight() == pytest.approx(25 * 3600.0)

    # Disambiguate fold=0 (EDT, 01:30 before fallback) vs fold=1 (EST, 01:30 after fallback)
    now_ref[0] = dt.datetime(2026, 11, 1, 1, 30, 0, fold=0, tzinfo=NY_TZ)
    assert qm.seconds_until_ny_midnight() == pytest.approx(23.5 * 3600.0)

    now_ref[0] = dt.datetime(2026, 11, 1, 1, 30, 0, fold=1, tzinfo=NY_TZ)
    assert qm.seconds_until_ny_midnight() == pytest.approx(22.5 * 3600.0)

    # Normal non-DST day
    now_ref[0] = dt.datetime(2026, 6, 15, 18, 30, 0, tzinfo=NY_TZ)
    assert qm.seconds_until_ny_midnight() == pytest.approx(5.5 * 3600.0)


def test_quota_4000_warning_and_5000_hard_stop(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    db = AlphaDB(tmp_path / "quota_limits.db")
    now_ref = [dt.datetime(2026, 6, 15, 10, 0, 0, tzinfo=NY_TZ)]
    qm = DailyQuotaManager(
        db=db,
        daily_sim_limit=5000,
        warning_threshold=4000,
        now_fn=lambda: now_ref[0],
    )

    with caplog.at_level(logging.WARNING):
        s_3999 = qm.record_submitted(3999)
        assert s_3999.submitted_count == 3999
        assert s_3999.remaining == 1001
        assert s_3999.warning_triggered is False
        assert s_3999.can_submit is True

        s_4000 = qm.record_submitted(1)
        assert s_4000.submitted_count == 4000
        assert s_4000.remaining == 1000
        assert s_4000.warning_triggered is True
        assert s_4000.can_submit is True
        assert db.get_daily_quota("2026-06-15")["warning_sent"] is True

        # Recording more should not emit duplicate warning logs on the same NY date
        s_4995 = qm.record_submitted(995)
        assert s_4995.submitted_count == 4995
        assert qm.can_submit(5) is True
        assert qm.can_submit(6) is False

        s_5000 = qm.record_submitted(5)
        assert s_5000.submitted_count == 5000
        assert s_5000.remaining == 0
        assert s_5000.can_submit is False
        assert qm.can_submit(1) is False
        assert qm.can_submit(0) is False

    warn_msgs = [rec for rec in caplog.records if "Daily BRAIN simulation quota warning" in rec.message]
    assert len(warn_msgs) == 1

    qm.record_outcome(completed=4, failed=2)
    refreshed = qm.get_status()
    assert refreshed.completed_count == 4
    assert refreshed.failed_count == 2

    # Rolling past NY midnight resets quota for the new day
    now_ref[0] = dt.datetime(2026, 6, 16, 0, 5, 0, tzinfo=NY_TZ)
    new_day = qm.get_status()
    assert new_day.date_ny == "2026-06-16"
    assert new_day.submitted_count == 0
    assert new_day.warning_triggered is False
    assert new_day.can_submit is True


def test_quota_should_pace(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "quota_pace.db")
    # 04:00:00 EDT => 20.0 hours remaining until NY midnight
    now_ref = [dt.datetime(2026, 6, 15, 4, 0, 0, tzinfo=NY_TZ)]
    qm = DailyQuotaManager(
        db=db,
        daily_sim_limit=5000,
        warning_threshold=4000,
        hourly_pacing=True,
        pace_buffer=1.25,
        now_fn=lambda: now_ref[0],
    )

    # 200 submitted in hour 4: remaining=4800, pace=(4800/20)*1.25=300.0 => 200 <= 300.0 (no pace)
    qm.record_submitted(200)
    assert qm.get_current_hour_submitted() == 200
    assert qm.should_pace() is False

    # +100 => 300 in hour 4: remaining=4700, pace=(4700/20)*1.25=293.75 => 300 > 293.75 (pace!)
    qm.record_submitted(100)
    assert qm.get_current_hour_submitted() == 300
    assert qm.should_pace() is True

    # Moving to next hour resets hourly counter
    now_ref[0] = dt.datetime(2026, 6, 15, 5, 0, 0, tzinfo=NY_TZ)
    assert qm.get_current_hour_submitted() == 0
    assert qm.should_pace() is False

    # When hourly_pacing=False, should_pace() is always False
    qm_unpaced = DailyQuotaManager(
        db=db,
        daily_sim_limit=5000,
        hourly_pacing=False,
        now_fn=lambda: now_ref[0],
    )
    qm_unpaced.record_submitted(4500)
    assert qm_unpaced.should_pace() is False


# ---------------------------------------------------------------------------
# PnlStore & extract_daily_returns_from_recordset
# ---------------------------------------------------------------------------


def test_pnl_store_save_load_roundtrip_and_delete(tmp_path: Path) -> None:
    pnl_dir = tmp_path / "pnl_store"
    store = PnlStore(pnl_dir=pnl_dir, max_cache_entries=4)

    dates = np.array([739000, 739001, 739002], dtype=np.int64)
    returns = np.array([0.012, -0.004, 0.019], dtype=np.float64)

    saved_path = store.save("alpha_001", "USA", dates, returns)
    assert saved_path == pnl_dir / "USA" / "alpha_001.npz"
    assert saved_path.is_file()
    assert store.has("alpha_001", "USA") is True
    assert len(store) == 1

    # Load from in-memory LRU cache
    cached_loaded = store.load("alpha_001", "USA")
    assert cached_loaded is not None
    c_dates, c_rets = cached_loaded
    assert c_dates.dtype == np.int64
    assert c_rets.dtype == np.float64
    assert np.array_equal(c_dates, dates)
    assert np.allclose(c_rets, returns)

    # Clear cache and verify round-trip reload from disk
    store.clear_cache()
    assert len(store) == 0
    assert store.has("alpha_001", "USA") is True

    disk_loaded = store.load("alpha_001", "USA")
    assert disk_loaded is not None
    d_dates, d_rets = disk_loaded
    assert d_dates.dtype == np.int64
    assert d_rets.dtype == np.float64
    assert np.array_equal(d_dates, dates)
    assert np.allclose(d_rets, returns)
    assert len(store) == 1

    # Missing alpha returns None; delete removes both cache and file
    assert store.load("missing_alpha", "USA") is None
    assert store.delete("alpha_001", "USA") is True
    assert store.delete("alpha_001", "USA") is False
    assert store.has("alpha_001", "USA") is False

    with pytest.raises(ValueError):
        store.save("../escape", "USA", dates, returns)
    with pytest.raises(ValueError):
        store.save("bad_shape", "USA", dates[:2], returns)


def test_pnl_store_lru_eviction(tmp_path: Path) -> None:
    store = PnlStore(pnl_dir=tmp_path / "pnl_lru", max_cache_entries=2)
    dates = np.array([739000, 739001], dtype=np.int64)

    store.save("a1", "USA", dates, np.array([0.01, 0.02]))
    store.save("a2", "USA", dates, np.array([0.03, 0.04]))
    assert len(store) == 2

    # Touch a1 so a2 becomes the least-recently-used entry
    assert store.load("a1", "USA") is not None

    # Saving a3 must evict a2 from in-memory cache while keeping a1 & a3
    store.save("a3", "USA", dates, np.array([0.05, 0.06]))
    assert len(store) == 2
    assert ("USA", "a1") in store._cache
    assert ("USA", "a3") in store._cache
    assert ("USA", "a2") not in store._cache

    # a2 remains persisted on disk and reloads transparently (evicting a1)
    assert store.has("a2", "USA") is True
    reloaded = store.load("a2", "USA")
    assert reloaded is not None
    assert np.allclose(reloaded[1], np.array([0.03, 0.04]))
    assert ("USA", "a2") in store._cache
    assert ("USA", "a3") in store._cache
    assert ("USA", "a1") not in store._cache


def test_extract_daily_returns_from_recordset_cumulative_pnl() -> None:
    raw_cumulative = {
        "schema": {
            "name": "pnl",
            "properties": [
                {"name": "date", "type": "date"},
                {"name": "pnl", "type": "decimal"},
            ],
        },
        "records": [
            ["2026-01-03", 15.0],
            ["2026-01-01", 10.0],
            ["2026-01-02", 8.0],
        ],
    }

    expected_dates = np.array(
        [
            dt.date(2026, 1, 1).toordinal(),
            dt.date(2026, 1, 2).toordinal(),
            dt.date(2026, 1, 3).toordinal(),
        ],
        dtype=np.int64,
    )
    expected_daily = np.array([10.0, -2.0, 7.0], dtype=np.float64)

    dates_raw, daily_raw = extract_daily_returns_from_recordset(raw_cumulative)
    assert dates_raw.dtype == np.int64
    assert daily_raw.dtype == np.float64
    assert np.array_equal(dates_raw, expected_dates)
    assert np.allclose(daily_raw, expected_daily)

    parsed = parse_recordset(raw_cumulative)
    dates_parsed, daily_parsed = extract_daily_returns_from_recordset(parsed)
    assert np.array_equal(dates_parsed, expected_dates)
    assert np.allclose(daily_parsed, expected_daily)


def test_extract_daily_returns_from_recordset_daily_pnl() -> None:
    daily_pnl_named = {
        "name": "daily-pnl",
        "schema": {
            "name": "daily-pnl",
            "properties": [
                {"name": "date", "type": "date"},
                {"name": "pnl", "type": "decimal"},
            ],
        },
        "records": [
            ["2026-01-02", -0.02],
            ["2026-01-01", 0.01],
            ["2026-01-03", 0.03],
        ],
    }
    expected_dates = np.array(
        [
            dt.date(2026, 1, 1).toordinal(),
            dt.date(2026, 1, 2).toordinal(),
            dt.date(2026, 1, 3).toordinal(),
        ],
        dtype=np.int64,
    )
    expected_daily = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    dates_a, rets_a = extract_daily_returns_from_recordset(daily_pnl_named)
    assert np.array_equal(dates_a, expected_dates)
    assert np.allclose(rets_a, expected_daily)

    returns_col_payload = parse_recordset(
        {
            "schema": {
                "properties": [
                    {"name": "date", "type": "date"},
                    {"name": "returns", "type": "decimal"},
                ]
            },
            "records": [
                ["2026-01-01", 0.01],
                ["2026-01-02", -0.02],
                ["2026-01-03", 0.03],
            ],
        }
    )
    dates_b, rets_b = extract_daily_returns_from_recordset(returns_col_payload)
    assert np.array_equal(dates_b, expected_dates)
    assert np.allclose(rets_b, expected_daily)

    empty_dates, empty_rets = extract_daily_returns_from_recordset({})
    assert empty_dates.shape == (0,)
    assert empty_rets.shape == (0,)
