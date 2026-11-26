#!/usr/bin/env python3
"""
crawl_brain_data.py — Crawl all WorldQuant BRAIN datasets & datafields → MySQL
═══════════════════════════════════════════════════════════════════════════════

Authenticates with the BRAIN API, then iterates over every
(instrument_type, region, delay, universe) combination to fetch:

  1. All datasets
  2. All datafields for each dataset

Everything is stored in a local MySQL database (`worldquant_brain`).

Usage
-----
    # Full crawl (all regions, all universes)
    python crawl_brain_data.py

    # Crawl a specific region only
    python crawl_brain_data.py --region USA

    # Crawl specific region + universe
    python crawl_brain_data.py --region TWN --universe TOP500

    # Resume a previous crawl (skips already-crawled combos)
    python crawl_brain_data.py --resume
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

import mysql.connector
import pandas as pd
from dotenv import load_dotenv

import ace_lib as ace

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crawl_brain")

# ── MySQL Config ─────────────────────────────────────────────────────────────
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DATABASE = "worldquant_brain"


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

CREATE_DB_SQL = f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}`"

CREATE_DATASETS_TABLE = """
CREATE TABLE IF NOT EXISTS datasets (
    id               VARCHAR(255)   NOT NULL,
    name             VARCHAR(500),
    description      TEXT,
    category         VARCHAR(255),
    subcategory      VARCHAR(255),
    coverage         DOUBLE,
    delay            INT,
    field_count      INT,
    user_count       INT,
    instrument_type  VARCHAR(50),
    region           VARCHAR(50),
    universe         VARCHAR(50),
    value_score      DOUBLE,
    alpha_count      INT,
    created_at       DATETIME,
    updated_at       DATETIME,
    crawled_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, region, universe, delay),
    INDEX idx_region_universe (region, universe),
    INDEX idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

CREATE_DATAFIELDS_TABLE = """
CREATE TABLE IF NOT EXISTS datafields (
    id               VARCHAR(500)   NOT NULL,
    dataset_id       VARCHAR(255),
    name             VARCHAR(500),
    description      TEXT,
    type             VARCHAR(50),
    category         VARCHAR(255),
    subcategory      VARCHAR(255),
    coverage         DOUBLE,
    delay            INT,
    instrument_type  VARCHAR(50),
    region           VARCHAR(50),
    universe         VARCHAR(50),
    crawled_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, region, universe, delay),
    INDEX idx_dataset (dataset_id),
    INDEX idx_type (type),
    INDEX idx_region_universe (region, universe)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

CREATE_OPERATORS_TABLE = """
CREATE TABLE IF NOT EXISTS operators (
    name             VARCHAR(255)   NOT NULL,
    description      TEXT,
    definition       TEXT,
    category         VARCHAR(100),
    scope            VARCHAR(50),
    crawled_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (name, scope),
    INDEX idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

CREATE_CRAWL_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS crawl_log (
    id               INT            AUTO_INCREMENT PRIMARY KEY,
    instrument_type  VARCHAR(50)    NOT NULL,
    region           VARCHAR(50)    NOT NULL,
    universe         VARCHAR(50)    NOT NULL,
    delay            INT            NOT NULL,
    datasets_found   INT            DEFAULT 0,
    datafields_found INT            DEFAULT 0,
    status           VARCHAR(20)    DEFAULT 'COMPLETED',
    started_at       DATETIME,
    finished_at      DATETIME,
    UNIQUE KEY uq_combo (instrument_type, region, universe, delay)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def get_connection(database: str | None = None) -> mysql.connector.MySQLConnection:
    """Create a MySQL connection."""
    kwargs = {
        "host": MYSQL_HOST,
        "user": MYSQL_USER,
    }
    if MYSQL_PASSWORD:
        kwargs["password"] = MYSQL_PASSWORD
    if database:
        kwargs["database"] = database
    return mysql.connector.connect(**kwargs)


def init_database() -> None:
    """Create the database and tables if they don't exist."""
    logger.info("Initializing MySQL database: %s", MYSQL_DATABASE)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(CREATE_DB_SQL)
    conn.commit()
    cursor.close()
    conn.close()

    conn = get_connection(MYSQL_DATABASE)
    cursor = conn.cursor()
    cursor.execute(CREATE_DATASETS_TABLE)
    cursor.execute(CREATE_DATAFIELDS_TABLE)
    cursor.execute(CREATE_OPERATORS_TABLE)
    cursor.execute(CREATE_CRAWL_LOG_TABLE)
    conn.commit()
    cursor.close()
    conn.close()
    logger.info("Database and tables ready")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA INSERTION
# ══════════════════════════════════════════════════════════════════════════════


def upsert_datasets(
    conn: mysql.connector.MySQLConnection,
    datasets_df: pd.DataFrame,
    region: str,
    universe: str,
    delay: int,
    instrument_type: str,
) -> int:
    """Insert or update datasets into MySQL. Returns count inserted."""
    if datasets_df.empty:
        return 0

    cursor = conn.cursor()
    count = 0

    for _, row in datasets_df.iterrows():
        try:
            cursor.execute(
                """
                INSERT INTO datasets
                    (id, name, description, category, subcategory, coverage,
                     delay, field_count, user_count, instrument_type,
                     region, universe, value_score, alpha_count, crawled_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    name = VALUES(name),
                    description = VALUES(description),
                    category = VALUES(category),
                    subcategory = VALUES(subcategory),
                    coverage = VALUES(coverage),
                    field_count = VALUES(field_count),
                    user_count = VALUES(user_count),
                    value_score = VALUES(value_score),
                    alpha_count = VALUES(alpha_count),
                    crawled_at = VALUES(crawled_at)
                """,
                (
                    str(row.get("id", "")),
                    str(row.get("name", ""))[:500] if row.get("name") else None,
                    str(row.get("description", "")) if row.get("description") else None,
                    str(row.get("category", "")) if row.get("category") else None,
                    str(row.get("subcategory", "")) if row.get("subcategory") else None,
                    float(row["coverage"]) if pd.notna(row.get("coverage")) else None,
                    delay,
                    int(row["fieldCount"]) if pd.notna(row.get("fieldCount")) else None,
                    int(row["userCount"]) if pd.notna(row.get("userCount")) else None,
                    instrument_type,
                    region,
                    universe,
                    float(row["valueScore"]) if pd.notna(row.get("valueScore")) else None,
                    int(row["alphaCount"]) if pd.notna(row.get("alphaCount")) else None,
                    datetime.now(timezone.utc),
                ),
            )
            count += 1
        except Exception as e:
            logger.warning("Failed to insert dataset %s: %s", row.get("id"), e)

    conn.commit()
    cursor.close()
    return count


def upsert_datafields(
    conn: mysql.connector.MySQLConnection,
    datafields_df: pd.DataFrame,
    dataset_id: str,
    region: str,
    universe: str,
    delay: int,
    instrument_type: str,
) -> int:
    """Insert or update datafields into MySQL. Returns count inserted."""
    if datafields_df.empty:
        return 0

    cursor = conn.cursor()
    count = 0

    for _, row in datafields_df.iterrows():
        try:
            cursor.execute(
                """
                INSERT INTO datafields
                    (id, dataset_id, name, description, type, category,
                     subcategory, coverage, delay, instrument_type,
                     region, universe, crawled_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    dataset_id = VALUES(dataset_id),
                    name = VALUES(name),
                    description = VALUES(description),
                    type = VALUES(type),
                    category = VALUES(category),
                    subcategory = VALUES(subcategory),
                    coverage = VALUES(coverage),
                    crawled_at = VALUES(crawled_at)
                """,
                (
                    str(row.get("id", ""))[:500],
                    dataset_id,
                    str(row.get("name", ""))[:500] if row.get("name") else None,
                    str(row.get("description", "")) if row.get("description") else None,
                    str(row.get("type", "")) if row.get("type") else None,
                    str(row.get("category", "")) if row.get("category") else None,
                    str(row.get("subcategory", "")) if row.get("subcategory") else None,
                    float(row["coverage"]) if pd.notna(row.get("coverage")) else None,
                    delay,
                    instrument_type,
                    region,
                    universe,
                    datetime.now(timezone.utc),
                ),
            )
            count += 1
        except Exception as e:
            logger.warning("Failed to insert datafield %s: %s", row.get("id"), e)

    conn.commit()
    cursor.close()
    return count


def upsert_operators(
    conn: mysql.connector.MySQLConnection,
    operators_df: pd.DataFrame,
) -> int:
    """Insert or update operators into MySQL."""
    if operators_df.empty:
        return 0

    cursor = conn.cursor()
    count = 0

    for _, row in operators_df.iterrows():
        try:
            cursor.execute(
                """
                INSERT INTO operators
                    (name, description, definition, category, scope, crawled_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    description = VALUES(description),
                    definition = VALUES(definition),
                    category = VALUES(category),
                    crawled_at = VALUES(crawled_at)
                """,
                (
                    str(row.get("name", "")),
                    str(row.get("description", "")) if row.get("description") else None,
                    str(row.get("definition", "")) if row.get("definition") else None,
                    str(row.get("category", "")) if row.get("category") else None,
                    str(row.get("scope", "")) if row.get("scope") else None,
                    datetime.now(timezone.utc),
                ),
            )
            count += 1
        except Exception as e:
            logger.warning("Failed to insert operator %s: %s", row.get("name"), e)

    conn.commit()
    cursor.close()
    return count


# ══════════════════════════════════════════════════════════════════════════════
#  CRAWL LOGIC
# ══════════════════════════════════════════════════════════════════════════════


def get_completed_combos(conn: mysql.connector.MySQLConnection) -> set[tuple]:
    """Return the set of (instrument_type, region, universe, delay) already crawled."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT instrument_type, region, universe, delay FROM crawl_log WHERE status = 'COMPLETED'"
    )
    result = {tuple(row) for row in cursor.fetchall()}
    cursor.close()
    return result


def log_crawl(
    conn: mysql.connector.MySQLConnection,
    instrument_type: str,
    region: str,
    universe: str,
    delay: int,
    datasets_found: int,
    datafields_found: int,
    started_at: datetime,
) -> None:
    """Record a completed crawl combo."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO crawl_log
            (instrument_type, region, universe, delay,
             datasets_found, datafields_found, status, started_at, finished_at)
        VALUES (%s, %s, %s, %s, %s, %s, 'COMPLETED', %s, %s)
        ON DUPLICATE KEY UPDATE
            datasets_found = VALUES(datasets_found),
            datafields_found = VALUES(datafields_found),
            status = 'COMPLETED',
            finished_at = VALUES(finished_at)
        """,
        (
            instrument_type, region, universe, delay,
            datasets_found, datafields_found,
            started_at, datetime.now(timezone.utc),
        ),
    )
    conn.commit()
    cursor.close()


def crawl_combo(
    s: ace.SingleSession,
    conn: mysql.connector.MySQLConnection,
    instrument_type: str,
    region: str,
    universe: str,
    delay: int,
) -> tuple[int, int]:
    """
    Crawl datasets and their datafields for a single
    (instrument_type, region, universe, delay) combination.

    Returns (datasets_count, datafields_count).
    """
    started_at = datetime.now(timezone.utc)

    # Fetch datasets
    try:
        datasets_df = ace.get_datasets(
            s,
            instrument_type=instrument_type,
            region=region,
            delay=delay,
            universe=universe,
        )
    except Exception as e:
        logger.error("Failed to fetch datasets for %s/%s/%s/d%d: %s",
                     instrument_type, region, universe, delay, e)
        return 0, 0

    ds_count = upsert_datasets(conn, datasets_df, region, universe, delay, instrument_type)
    logger.info("    Datasets inserted/updated: %d", ds_count)

    if datasets_df.empty:
        log_crawl(conn, instrument_type, region, universe, delay, 0, 0, started_at)
        return 0, 0

    # Fetch datafields for each dataset
    dataset_ids = datasets_df["id"].values.tolist()
    total_fields = 0

    for i, ds_id in enumerate(dataset_ids):
        try:
            s = ace.check_session_and_relogin(s)
            df = ace.get_datafields(
                s,
                instrument_type=instrument_type,
                region=region,
                delay=delay,
                universe=universe,
                search=str(ds_id),
            )
            if not df.empty:
                count = upsert_datafields(conn, df, str(ds_id), region, universe, delay, instrument_type)
                total_fields += count

            if (i + 1) % 10 == 0:
                logger.info("      [%d/%d] datasets processed, %d fields so far",
                            i + 1, len(dataset_ids), total_fields)

        except Exception as e:
            logger.warning("    ⚠ Failed datafields for dataset %s: %s", ds_id, e)

        time.sleep(0.5)  # rate-limit courtesy

    logger.info("    Datafields inserted/updated: %d", total_fields)

    log_crawl(conn, instrument_type, region, universe, delay, ds_count, total_fields, started_at)
    return ds_count, total_fields


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Crawl WorldQuant BRAIN datasets & datafields into MySQL",
    )
    parser.add_argument("--region", help="Crawl only this region (e.g. USA, TWN, CHN)")
    parser.add_argument("--universe", help="Crawl only this universe (e.g. TOP3000, TOP500)")
    parser.add_argument("--delay", type=int, help="Crawl only this delay (e.g. 0, 1)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (instrument_type, region, universe, delay) combos that were already crawled",
    )
    args = parser.parse_args()

    # ── Init DB ───────────────────────────────────────────────────────────
    init_database()
    conn = get_connection(MYSQL_DATABASE)

    # ── Authenticate ──────────────────────────────────────────────────────
    logger.info("━━━ Authenticating with BRAIN API ━━━")
    s = ace.start_session()
    ttl = ace.check_session_timeout(s)
    logger.info("Session active — TTL %d seconds", ttl)

    # ── Fetch all (instrument_type, region, delay, universe) combos ──────
    logger.info("━━━ Fetching available instrument/region/delay/universe combos ━━━")
    combos_df = ace.get_instrument_type_region_delay(s)
    logger.info("Total combos found: %d", len(combos_df))

    # ── Crawl operators (global, done once) ──────────────────────────────
    logger.info("━━━ Crawling operators ━━━")
    operators_df = ace.get_operators(s)
    op_count = upsert_operators(conn, operators_df)
    logger.info("Operators inserted/updated: %d", op_count)

    # ── Build the work list ──────────────────────────────────────────────
    completed = get_completed_combos(conn) if args.resume else set()
    if args.resume and completed:
        logger.info("Resume mode: %d combos already completed, will be skipped", len(completed))

    total_datasets = 0
    total_datafields = 0
    combos_processed = 0

    for _, row in combos_df.iterrows():
        instrument_type = row["InstrumentType"]
        region = row["Region"]
        delay = row["Delay"]
        universes = row["Universe"]  # This is a list

        # Apply CLI filters
        if args.region and region != args.region:
            continue
        if args.delay is not None and delay != args.delay:
            continue

        for universe in universes:
            if args.universe and universe != args.universe:
                continue

            combo_key = (instrument_type, region, universe, delay)

            if args.resume and combo_key in completed:
                logger.debug("Skipping already-crawled combo: %s", combo_key)
                continue

            combos_processed += 1
            logger.info(
                "═══ [%d] Crawling: %s | %s | %s | delay=%d ═══",
                combos_processed, instrument_type, region, universe, delay,
            )

            # Re-authenticate if needed
            s = ace.check_session_and_relogin(s)

            ds_count, df_count = crawl_combo(
                s, conn, instrument_type, region, universe, delay
            )
            total_datasets += ds_count
            total_datafields += df_count

    conn.close()

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("══════════════════════════════════════════════════════════════")
    logger.info("Crawl complete!")
    logger.info("  Combos processed:    %d", combos_processed)
    logger.info("  Datasets inserted:   %d", total_datasets)
    logger.info("  Datafields inserted: %d", total_datafields)
    logger.info("══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
