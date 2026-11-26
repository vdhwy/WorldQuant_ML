#!/usr/bin/env python3
"""
pipeline.py — Automated WorldQuant BRAIN Simulation Pipeline
═══════════════════════════════════════════════════════════════

Replaces the interactive simulator.ipynb with a fully automated,
non-interactive pipeline that:

  1. Authenticates with WorldQuant BRAIN
  2. Discovers datasets matching configurable filters
  3. Fetches datafields for each dataset
  4. Loads operators (Vector & Time-Series)
  5. Generates alpha expressions from built-in + custom templates
  6. Wraps expressions into alpha dicts
  7. Simulates in batches with safe polling (Retry-After fix)
  8. Exports results to CSV / Parquet

Usage
-----
    # Full run
    python pipeline.py --config pipeline_config.yaml

    # Dry run (generates expressions only, no simulation)
    python pipeline.py --config pipeline_config.yaml --dry-run

    # Override region/universe from CLI
    python pipeline.py --region CHN --universe TOP2000

    # Interactive mode — prompt for custom expression templates
    python pipeline.py --interactive

    # Pass templates directly from CLI
    python pipeline.py --template "ts_delta({field}, {lookback})" --template "rank({ts_op}({vector_field}, {lookback}))"
"""

from __future__ import annotations

import argparse
import itertools
import re
import logging
import os
import sys
import time
from datetime import datetime, timezone
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ── project imports ───────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from alpha_mining.storage.backtest_storage import BacktestStorage
    from alpha_mining.core.template_similarity import TemplateSimilarityChecker
except ImportError as e:
    logger.warning("alpha_mining module not found. Similarity skipping will be disabled.")
    BacktestStorage = None
    TemplateSimilarityChecker = None

import ace_lib as ace
import helpful_functions as hf
from pipeline_utils import (
    PollingTimeoutError,
    SimulationCache,
    SimulationError,
    deduplicate_similar_expressions,
    fmt_duration,
    hash_alpha,
    load_config,
    safe_poll,
    safe_poll_children,
    send_telegram,
)

load_dotenv()

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — Authentication
# ══════════════════════════════════════════════════════════════════════════════


def stage_authenticate() -> ace.SingleSession:
    """Start an authenticated BRAIN session."""
    logger.info("━━━ Stage 1: Authentication ━━━")
    s = ace.start_session()
    ttl = ace.check_session_timeout(s)
    logger.info("Session active — TTL %s seconds", ttl)
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — Dataset Discovery
# ══════════════════════════════════════════════════════════════════════════════


def stage_discover_datasets(
    s: ace.SingleSession, cfg: dict
) -> pd.DataFrame:
    """Fetch and filter datasets based on config."""
    logger.info("━━━ Stage 2: Dataset Discovery ━━━")

    mkt = cfg["market"]
    flt = cfg["dataset_filters"]

    datasets_df = ace.get_datasets(
        s,
        instrument_type=mkt["instrument_type"],
        region=mkt["region"],
        delay=mkt["delay"],
        universe=mkt["universe"],
    )
    logger.info("Raw datasets fetched: %d", len(datasets_df))

    # Apply filters — mirrors the notebook's selection logic
    mask = (
        (datasets_df["delay"] == mkt["delay"])
        & (datasets_df["coverage"] > flt["coverage_min"])
        & (datasets_df["coverage"] <= flt["coverage_max"])
        & (datasets_df["fieldCount"] > flt["field_count_min"])
        & (datasets_df["fieldCount"] < flt["field_count_max"])
        & (datasets_df["region"] == mkt["region"])
        & (datasets_df["universe"] == mkt["universe"])
        & (datasets_df["userCount"] > flt["user_count_min"])
        & (datasets_df["userCount"] < flt["user_count_max"])
        & (datasets_df["valueScore"] > flt["value_score_min"])
        & (datasets_df["valueScore"] < flt["value_score_max"])
    )
    selected = datasets_df[mask].sort_values(
        by=[flt["sort_by"]], ascending=flt["sort_ascending"]
    )

    # Optional category filter
    category = flt.get("category", "")
    if category:
        selected = selected[selected["category_id"] == category]

    logger.info(
        "Filtered datasets: %d (category=%s)",
        len(selected),
        category or "ALL",
    )
    if selected.empty:
        logger.warning("No datasets matched the filters — pipeline will stop.")
    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — Datafield Fetching
# ══════════════════════════════════════════════════════════════════════════════


def stage_fetch_datafields(
    s: ace.SingleSession, datasets: pd.DataFrame, cfg: dict
) -> pd.DataFrame:
    """Fetch datafields for every selected dataset, with rate-limit awareness."""
    logger.info("━━━ Stage 3: Datafield Fetching ━━━")

    mkt = cfg["market"]
    dataset_ids = datasets["id"].values.tolist()
    all_fields: list[pd.DataFrame] = []

    for i, ds_id in enumerate(dataset_ids):
        logger.info(
            "  [%d/%d] Fetching fields for dataset %s …",
            i + 1,
            len(dataset_ids),
            ds_id,
        )
        try:
            df = ace.get_datafields(
                s,
                instrument_type=mkt["instrument_type"],
                region=mkt["region"],
                delay=mkt["delay"],
                universe=mkt["universe"],
                search=str(ds_id),
            )
            if not df.empty:
                all_fields.append(df)
        except Exception as exc:
            logger.warning("  ⚠ Failed for %s: %s", ds_id, exc)
        time.sleep(0.5)  # rate-limit courtesy

    if not all_fields:
        logger.warning("No datafields retrieved — pipeline will stop.")
        return pd.DataFrame()

    datafields_df = pd.concat(all_fields, ignore_index=True).drop_duplicates(
        subset=["id"]
    )
    logger.info(
        "Total unique datafields: %d  (VECTOR=%d, MATRIX=%d, other=%d)",
        len(datafields_df),
        (datafields_df["type"] == "VECTOR").sum(),
        (datafields_df["type"] == "MATRIX").sum(),
        (
            ~datafields_df["type"].isin(["VECTOR", "MATRIX"])
        ).sum(),
    )
    return datafields_df


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — Operator Loading
# ══════════════════════════════════════════════════════════════════════════════


def stage_load_operators(s: ace.SingleSession) -> tuple[list[str], list[str]]:
    """
    Fetch operators and filter for single-argument Vector & Time-Series ops.

    Returns (vector_ops, time_series_ops).
    """
    logger.info("━━━ Stage 4: Operator Loading ━━━")
    operators = ace.get_operators(s)

    # Vector operators — single-argument (definition doesn't contain 'y')
    vec_mask = (
        (operators["scope"] == "REGULAR")
        & (operators["category"] == "Vector")
    )
    vec_ops = operators[vec_mask]
    vec_ops = vec_ops[~vec_ops["definition"].str.contains("y", na=False)]
    vector_ops = vec_ops["name"].tolist()

    # Time-series operators — single-argument
    ts_mask = (
        (operators["scope"] == "REGULAR")
        & (operators["category"] == "Time Series")
    )
    ts_ops = operators[ts_mask]
    ts_ops = ts_ops[~ts_ops["definition"].str.contains("y", na=False)]
    time_series_ops = ts_ops["name"].tolist()

    logger.info(
        "Vector ops: %d  |  Time-series ops: %d", len(vector_ops), len(time_series_ops)
    )
    logger.debug("Vector: %s", vector_ops)
    logger.debug("TS:     %s", time_series_ops)
    return vector_ops, time_series_ops


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 — Expression Generation
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPLATE ENGINE — Custom expression templates with placeholder expansion
# ══════════════════════════════════════════════════════════════════════════════

# Supported placeholders:
#   {ts_op}         → each Time-Series operator
#   {vec_op}        → each Vector operator
#   {field}         → every datafield (any type)
#   {vector_field}  → VECTOR-type datafields only
#   {matrix_field}  → MATRIX-type datafields only
#   {pos}           → datafields whose id contains "pos" (sentiment)
#   {neg}           → datafields whose id contains "neg" (sentiment)
#   {lookback}      → each lookback window from config

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _extract_placeholders(template: str) -> list[str]:
    """Return the list of unique placeholder names in a template string."""
    return list(dict.fromkeys(_PLACEHOLDER_RE.findall(template)))


def _build_value_lists(
    placeholders: list[str],
    datafields_df: pd.DataFrame,
    vector_ops: list[str],
    time_series_ops: list[str],
    lookbacks: list[int],
) -> dict[str, list[str]]:
    """
    Map each placeholder name to the list of concrete values it expands to.
    """
    ids = datafields_df["id"].tolist() if "id" in datafields_df.columns else []
    has_type = "type" in datafields_df.columns

    mapping: dict[str, list[str]] = {}
    for ph in placeholders:
        if ph == "ts_op":
            mapping[ph] = time_series_ops
        elif ph == "vec_op":
            mapping[ph] = vector_ops
        elif ph == "field":
            mapping[ph] = ids
        elif ph == "vector_field":
            mapping[ph] = (
                datafields_df[datafields_df["type"] == "VECTOR"]["id"].tolist()
                if has_type else ids
            )
        elif ph == "matrix_field":
            mapping[ph] = (
                datafields_df[datafields_df["type"] == "MATRIX"]["id"].tolist()
                if has_type else ids
            )
        elif ph == "risk_matrix":
            mapping[ph] = (
                [fid for fid in datafields_df[datafields_df["type"] == "MATRIX"]["id"].tolist() if "risk" in str(fid).lower() or "rsk" in str(fid).lower()]
                if has_type else [fid for fid in ids if "risk" in str(fid).lower() or "rsk" in str(fid).lower()]
            )
        elif ph == "pos":
            mapping[ph] = [fid for fid in ids if "pos" in str(fid).lower()]
        elif ph == "neg":
            mapping[ph] = [fid for fid in ids if "neg" in str(fid).lower()]
        elif ph in ("lookback", "a"):
            mapping[ph] = [str(lb) for lb in lookbacks]
        else:
            logger.warning(
                "Unknown placeholder {%s} — treating as literal", ph
            )
            mapping[ph] = [f"{{{ph}}}"]
    return mapping


def expand_template(
    template: str,
    datafields_df: pd.DataFrame,
    vector_ops: list[str],
    time_series_ops: list[str],
    lookbacks: list[int],
) -> list[str]:
    """
    Expand a single template string into all concrete expressions by
    taking the Cartesian product of every placeholder's value list.

    Example
    -------
    >>> expand_template(
    ...     "ts_delta({vec_op}({vector_field}), {lookback})",
    ...     datafields_df, vector_ops, ts_ops, [5, 10]
    ... )
    ['ts_delta(rank(field1), 5)', 'ts_delta(rank(field1), 10)', ...]
    """
    placeholders = _extract_placeholders(template)
    if not placeholders:
        # No placeholders — return template as-is (literal expression)
        return [template]

    value_lists = _build_value_lists(
        placeholders, datafields_df, vector_ops, time_series_ops, lookbacks
    )

    # Check for empty value lists
    for ph, vals in value_lists.items():
        if not vals:
            logger.warning(
                "Placeholder {%s} has 0 values — template '%s' will produce 0 expressions",
                ph,
                template,
            )
            return []

    # Cartesian product of all value lists (order matches placeholder order)
    keys = placeholders
    combos = itertools.product(*(value_lists[k] for k in keys))

    results = []
    for combo in combos:
        expr = template
        for key, val in zip(keys, combo):
            expr = expr.replace(f"{{{key}}}", str(val))
        results.append(expr)
    return results


def prompt_templates_interactive(
    datafields_df: pd.DataFrame,
    vector_ops: list[str],
    time_series_ops: list[str],
    lookbacks: list[int],
) -> list[str]:
    """
    Interactive prompt that lets the user define custom expression templates.

    Shows available placeholders and sample values, then collects templates
    one at a time until the user enters an empty line.
    """
    ids = datafields_df["id"].tolist() if "id" in datafields_df.columns else []
    has_type = "type" in datafields_df.columns
    vec_fields = (
        datafields_df[datafields_df["type"] == "VECTOR"]["id"].tolist()
        if has_type else []
    )
    mat_fields = (
        datafields_df[datafields_df["type"] == "MATRIX"]["id"].tolist()
        if has_type else []
    )
    risk_mat_fields = [f for f in mat_fields if "risk" in str(f).lower() or "rsk" in str(f).lower()] if has_type else [f for f in ids if "risk" in str(f).lower() or "rsk" in str(f).lower()]
    pos_fields = [f for f in ids if "pos" in str(f).lower()]
    neg_fields = [f for f in ids if "neg" in str(f).lower()]

    def _sample(lst: list, n: int = 3) -> str:
        shown = lst[:n]
        suffix = f" … (+{len(lst)-n} more)" if len(lst) > n else ""
        return ", ".join(str(x) for x in shown) + suffix

    print("\n" + "═" * 66)
    print("  CUSTOM EXPRESSION TEMPLATE BUILDER")
    print("═" * 66)
    print("\n  Available placeholders and current values:\n")
    print(f"    {{ts_op}}         → {len(time_series_ops)} ops   : {_sample(time_series_ops)}")
    print(f"    {{vec_op}}        → {len(vector_ops)} ops   : {_sample(vector_ops)}")
    print(f"    {{field}}         → {len(ids)} fields : {_sample(ids)}")
    print(f"    {{vector_field}}  → {len(vec_fields)} fields : {_sample(vec_fields)}")
    print(f"    {{matrix_field}}  → {len(mat_fields)} fields : {_sample(mat_fields)}")
    print(f"    {{risk_matrix}}   → {len(risk_mat_fields)} fields : {_sample(risk_mat_fields)}")
    print(f"    {{pos}}           → {len(pos_fields)} fields : {_sample(pos_fields)}")
    print(f"    {{neg}}           → {len(neg_fields)} fields : {_sample(neg_fields)}")
    print(f"    {{lookback}}      → {lookbacks}")
    print("\n  Examples:")
    print("    ts_delta({field}, {lookback})")
    print("    rank({ts_op}({vector_field}, {lookback}))")
    print("    {ts_op}({neg} - {pos}, {lookback})")
    print("    group_neutralize({ts_op}({field}, {lookback}), sector)")
    print("\n  Enter templates one per line. Empty line to finish.\n")

    user_templates: list[str] = []
    while True:
        try:
            line = input(f"  template[{len(user_templates)+1}] ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        # Quick validation: check that all placeholders are known
        phs = _extract_placeholders(line)
        known = {"ts_op", "vec_op", "field", "vector_field", "matrix_field",
                 "risk_matrix", "pos", "neg", "lookback", "a"}
        unknown = set(phs) - known
        if unknown:
            print(f"    ⚠  Unknown placeholders: {unknown} — will be kept as literals")
        # Preview count
        expanded = expand_template(
            line, datafields_df, vector_ops, time_series_ops, lookbacks
        )
        print(f"    → will generate {len(expanded)} expressions")
        if expanded:
            print(f"    → sample: {expanded[0]}")
        user_templates.append(line)

    if not user_templates:
        print("  No custom templates entered.")
        return []

    # Expand all
    all_expanded: list[str] = []
    for tmpl in user_templates:
        all_expanded.extend(
            expand_template(
                tmpl, datafields_df, vector_ops, time_series_ops, lookbacks
            )
        )
    print(f"\n  ✓ Custom templates produced {len(all_expanded)} expressions")
    print("═" * 66 + "\n")
    return all_expanded


def stage_generate_expressions(
    datafields_df: pd.DataFrame,
    vector_ops: list[str],
    time_series_ops: list[str],
    cfg: dict,
    custom_templates: list[str] | None = None,
) -> list[str]:
    """
    Generate alpha expressions from built-in templates + custom templates.

        Built-in templates (controllable via config):
            1. Sentiment comparison:  ts_op(vec_op(neg) - vec_op(pos), lookback)
      2. Vector expressions:    ts_op(vec_op(field), lookback)
      3. Matrix expressions:    ts_op(field, lookback)

    Custom templates use placeholders like {ts_op}, {field}, {lookback}
    that expand via Cartesian product.
    """
    logger.info("━━━ Stage 5: Expression Generation ━━━")

    expr_cfg = cfg["expressions"]
    lookbacks = expr_cfg["lookback_days"]
    templates = expr_cfg["templates"]
    all_expressions: list[str] = []

    ids = datafields_df["id"] if "id" in datafields_df.columns else pd.Series(dtype=str)

    # ── 5a. Sentiment comparison ──────────────────────────────────────────
    if templates.get("sentiment_comparison", True):
        pos_fields = [fid for fid in ids if "pos" in str(fid).lower()]
        neg_fields = [fid for fid in ids if "neg" in str(fid).lower()]
        sentiment_count = 0
        for a in lookbacks:
            for op, neg, pos in itertools.product(time_series_ops, neg_fields, pos_fields):
                for v_op in vector_ops:
                    expr = f"{op}({v_op}({neg}) - {v_op}({pos}), {a})"
                    all_expressions.append(expr)
                    sentiment_count += 1
        logger.info("  Sentiment expressions: %d", sentiment_count)
    else:
        logger.info("  Sentiment template disabled — skipping")

    # ── 5b. Vector expressions ────────────────────────────────────────────
    if templates.get("vector_ops", True):
        vector_fields = datafields_df[datafields_df["type"] == "VECTOR"]["id"].tolist() if "type" in datafields_df.columns else []
        vec_count = 0
        for a in lookbacks:
            for v_op in vector_ops:
                for ts_op in time_series_ops:
                    for fid in vector_fields:
                        expr = f"{ts_op}({v_op}({fid}),{a})"
                        all_expressions.append(expr)
                        vec_count += 1
        logger.info("  Vector expressions: %d", vec_count)
    else:
        logger.info("  Vector template disabled — skipping")

    # ── 5c. Matrix expressions ────────────────────────────────────────────
    if templates.get("matrix_ops", True):
        matrix_fields = datafields_df[datafields_df["type"] == "MATRIX"]["id"].tolist() if "type" in datafields_df.columns else []
        mat_count = 0
        for a in lookbacks:
            for ts_op in time_series_ops:
                for fid in matrix_fields:
                    expr = f"{ts_op}({fid},{a})"
                    all_expressions.append(expr)
                    mat_count += 1
        logger.info("  Matrix expressions: %d", mat_count)
    else:
        logger.info("  Matrix template disabled — skipping")

    # ── 5d. Custom templates ──────────────────────────────────────────────
    if custom_templates:
        custom_count = 0
        for tmpl in custom_templates:
            expanded = expand_template(
                tmpl, datafields_df, vector_ops, time_series_ops, lookbacks
            )
            all_expressions.extend(expanded)
            custom_count += len(expanded)
        logger.info("  Custom template expressions: %d", custom_count)

    # ── 5e. Config-file custom_templates ──────────────────────────────────
    cfg_templates = expr_cfg.get("custom_templates", [])
    if cfg_templates:
        cfg_tmpl_count = 0
        for tmpl in cfg_templates:
            expanded = expand_template(
                tmpl, datafields_df, vector_ops, time_series_ops, lookbacks
            )
            all_expressions.extend(expanded)
            cfg_tmpl_count += len(expanded)
        logger.info("  Config custom template expressions: %d", cfg_tmpl_count)

    # Deduplicate exact matches
    before = len(all_expressions)
    all_expressions = list(dict.fromkeys(all_expressions))
    logger.info(
        "Total expressions: %d (deduped from %d)", len(all_expressions), before
    )

    # ── Read similarity settings from config ──────────────────────────────
    sim_filter = cfg.get("similarity", {})
    skip_historical = sim_filter.get("skip_historical", False)
    skip_intra = sim_filter.get("skip_intra", False)
    max_per_structure = sim_filter.get("max_per_structure", 1)

    # ── Similarity filtering against simulated alphas ─────────────────────
    if skip_historical:
        logger.info("  Historical similarity check DISABLED (--no-historical-similarity)")
    elif BacktestStorage and TemplateSimilarityChecker:
        db_path = Path(__file__).resolve().parent.parent / "alpha_mining" / "generation_two_backtests.db"
        if db_path.exists():
            logger.info("  Checking similarity against historical simulations (this may take a moment)...")
            try:
                storage = BacktestStorage(db_path=str(db_path))
                region = cfg["market"]["region"]
                # Fetch recent successful simulated templates
                simulated = storage.get_all_templates(region=region, limit=1000)
                
                if simulated:
                    sim_checker = TemplateSimilarityChecker(similarity_threshold=0.85)
                    filtered_expressions = []
                    skipped_count = 0
                    
                    for expr in all_expressions:
                        similar = sim_checker.find_similar_templates(expr, simulated)
                        if similar:
                            skipped_count += 1
                        else:
                            filtered_expressions.append(expr)
                            
                    all_expressions = filtered_expressions
                    logger.info("  Historical Similarity Filter: Skipped %d expressions, %d remaining.", skipped_count, len(all_expressions))
                else:
                    logger.info("  No historical simulations found for region %s to compare against.", region)
            except Exception as e:
                logger.error("  Failed during similarity check: %s", e)
        else:
            logger.warning("  alpha_mining DB not found at %s. Skipping similarity check.", db_path)

    # ── Intra-batch similarity filtering (O(n) structural fingerprint) ────
    # Expressions generated via Cartesian product often share the exact same
    # operator structure and differ only in which datafield they reference.
    # Instead of O(n²) pairwise comparison, we replace field-name tokens with
    # canonical placeholders to build a "structural fingerprint", then keep
    # at most `max_per_structure` expressions per unique fingerprint.
    if skip_intra:
        logger.info("  Intra-batch similarity check DISABLED (--no-intra-similarity)")
    elif len(all_expressions) > 1:
        logger.info("  Intra-batch similarity dedup (%d expressions, max_per_structure=%d)…", len(all_expressions), max_per_structure)
        all_expressions, intra_skipped = deduplicate_similar_expressions(
            all_expressions, max_per_structure=max_per_structure
        )
        logger.info(
            "  Intra-batch Similarity Filter: Skipped %d structurally similar expressions, %d remaining.",
            intra_skipped,
            len(all_expressions),
        )

    return all_expressions


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 6 — Alpha Generation
# ══════════════════════════════════════════════════════════════════════════════


def stage_generate_alphas(
    expressions: list[str], cfg: dict
) -> list[dict]:
    """Convert each expression string into an alpha dict for simulation."""
    logger.info("━━━ Stage 6: Alpha Generation ━━━")

    mkt = cfg["market"]
    sim = cfg["simulation"]

    alpha_list = [
        ace.generate_alpha(
            x,
            alpha_type=sim["alpha_type"],
            region=mkt["region"],
            universe=mkt["universe"],
            delay=mkt["delay"],
            decay=sim["decay"],
            neutralization=sim["neutralization"],
            truncation=sim["truncation"],
            pasteurization=sim["pasteurization"],
            test_period=sim["test_period"],
            unit_handling=sim["unit_handling"],
            nan_handling=sim["nan_handling"],
            max_trade=sim["max_trade"],
            visualization=sim["visualization"],
        )
        for x in expressions
    ]
    logger.info("Alpha dicts generated: %d", len(alpha_list))
    return alpha_list


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 7 — Simulation (with safe polling & caching)
# ══════════════════════════════════════════════════════════════════════════════


def _safe_simulate_multi(
    s: ace.SingleSession,
    batch: list[dict],
    poll_cfg: dict,
) -> list[dict]:
    """
    Submit a multi-simulation batch and poll children with safe_poll().

    This replaces the buggy multisimulation_progress() in ace_lib.py.
    """
    s = ace.check_session_and_relogin(s)

    if len(batch) == 1:
        return [_safe_simulate_single(s, batch[0], poll_cfg)]

    # Submit
    resp = ace.start_simulation(s, batch)
    if resp.status_code // 100 != 2:
        # Log the first expression so we can see what was rejected
        sample_expr = batch[0].get("regular", "?") if batch else "?"
        try:
            err_body = resp.json()
        except Exception:
            err_body = resp.text[:500]
        logger.error(
            "Multi-sim submit failed (%d) for %d alphas (first expr: %s). "
            "Response: %s",
            resp.status_code,
            len(batch),
            sample_expr,
            err_body,
        )
        return [{"alpha_id": None, "simulate_data": x} for x in batch]

    # Poll parent
    parent_url = resp.headers.get("Location", "")
    if not parent_url:
        logger.error("No Location header on multi-sim response")
        return [{"alpha_id": None, "simulate_data": x} for x in batch]

    try:
        parent_resp = safe_poll(
            s,
            parent_url,
            max_wait_s=poll_cfg["max_wait_s"],
            default_interval_s=poll_cfg["default_interval_s"],
            max_http_retries=poll_cfg["max_http_retries"],
        )
    except (PollingTimeoutError, SimulationError) as exc:
        sample_expr = batch[0].get("regular", "?") if batch else "?"
        logger.error(
            "Parent poll failed for %d alphas (first expr: %s): %s",
            len(batch),
            sample_expr,
            exc,
        )
        return [{"alpha_id": None, "simulate_data": x} for x in batch]

    parent_body = parent_resp.json()
    children = parent_body.get("children", [])
    if not children:
        logger.warning("Multi-sim returned 0 children: %s", parent_body)
        return [{"alpha_id": None, "simulate_data": x} for x in batch]

    # Poll each child individually (changelog fix #5)
    child_bodies = safe_poll_children(
        s,
        children,
        brain_api_url,
        max_wait_s=poll_cfg["max_wait_s"],
        default_interval_s=poll_cfg["default_interval_s"],
        max_http_retries=poll_cfg["max_http_retries"],
    )

    results = []
    for child_body in child_bodies:
        alpha_id = child_body.get("alpha", None)
        if alpha_id:
            # Fetch full alpha result
            full = ace.get_simulation_result_json(s, alpha_id)
            results.append({
                "alpha_id": full.get("id", alpha_id),
                "simulate_data": {
                    "type": full.get("type", "REGULAR"),
                    "settings": full.get("settings", {}),
                    "regular": full.get("regular", {}).get("code", "")
                    if isinstance(full.get("regular"), dict)
                    else full.get("regular", ""),
                },
            })
        else:
            child_status = child_body.get("status", "UNKNOWN")
            logger.warning(
                "Child sim %s: no alpha (status=%s)",
                child_body.get("id", "?"),
                child_status,
            )
            results.append({"alpha_id": None, "simulate_data": {}})

    return results


def _safe_simulate_single(
    s: ace.SingleSession,
    alpha_data: dict,
    poll_cfg: dict,
) -> dict:
    """Submit and poll a single simulation with safe_poll()."""
    s = ace.check_session_and_relogin(s)
    resp = ace.start_simulation(s, alpha_data)

    if resp.status_code // 100 != 2:
        logger.warning(
            "Single-sim submit failed (%d): %s",
            resp.status_code,
            resp.text[:300],
        )
        return {"alpha_id": None, "simulate_data": alpha_data}

    location = resp.headers.get("Location", "")
    if not location:
        logger.error("No Location header on single-sim response")
        return {"alpha_id": None, "simulate_data": alpha_data}

    try:
        poll_resp = safe_poll(
            s,
            location,
            max_wait_s=poll_cfg["max_wait_s"],
            default_interval_s=poll_cfg["default_interval_s"],
            max_http_retries=poll_cfg["max_http_retries"],
        )
    except (PollingTimeoutError, SimulationError) as exc:
        logger.error("Single poll failed: %s", exc)
        return {"alpha_id": None, "simulate_data": alpha_data}

    body = poll_resp.json()
    alpha_id = body.get("alpha")
    if not alpha_id:
        logger.warning("Single sim finished but no alpha: %s", body)
        return {"alpha_id": None, "simulate_data": alpha_data}

    return {"alpha_id": alpha_id, "simulate_data": alpha_data}


def stage_simulate(
    s: ace.SingleSession,
    alpha_list: list[dict],
    cfg: dict,
) -> list[dict]:
    """
    Simulate alphas in batches using multi-simulation with safe polling.

    Supports:
      • Parquet-backed dedup cache (skip already-simulated)
      • Auto session re-login between batches
      • Safe polling with Retry-After fix
      • Telegram notifications
    """
    logger.info("━━━ Stage 7: Simulation ━━━")

    sim_cfg = cfg["simulation"]
    poll_cfg = cfg["polling"]
    out_cfg = cfg["output"]
    tg_cfg = cfg["telegram"]

    batch_size = sim_cfg["batch_size"]
    multi_limit = sim_cfg["multi_sim_limit"]
    concurrent = sim_cfg["concurrent_limit"]

    # ── Cache: skip already-simulated alphas ──────────────────────────────
    cache = SimulationCache(out_cfg["cache_file"])
    cached_hashes = cache.already_simulated_hashes()
    before = len(alpha_list)
    alpha_list = [a for a in alpha_list if hash_alpha(a) not in cached_hashes]
    skipped = before - len(alpha_list)
    if skipped:
        logger.info("Skipped %d already-cached alphas", skipped)

    if not alpha_list:
        logger.info("All alphas already in cache — nothing to simulate.")
        return []

    logger.info(
        "Simulating %d alphas in batches of %d (multi=%d, concurrent=%d)",
        len(alpha_list),
        batch_size,
        multi_limit,
        concurrent,
    )

    all_results: list[dict] = []
    total_batches = (len(alpha_list) + batch_size - 1) // batch_size
    pipeline_start = time.monotonic()
    consecutive_full_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3  # circuit breaker threshold

    for batch_idx in range(total_batches):
        batch_start = time.monotonic()
        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, len(alpha_list))
        batch = alpha_list[start_i:end_i]

        logger.info(
            "┌─ Batch %d/%d  (%d alphas) ─────────────────────────────",
            batch_idx + 1,
            total_batches,
            len(batch),
        )

        # Re-login before each batch
        s = ace.check_session_and_relogin(s)

        # Split batch into multi-sim chunks
        chunks = [
            batch[i : i + multi_limit]
            for i in range(0, len(batch), multi_limit)
        ]

        # Run chunks concurrently
        batch_results: list[dict] = []
        worker = partial(_safe_simulate_multi, s, poll_cfg=poll_cfg)

        with ThreadPool(concurrent) as pool:
            for chunk_result in pool.imap_unordered(worker, chunks):
                batch_results.extend(chunk_result)

        # Cache results
        for r in batch_results:
            alpha_id = r.get("alpha_id")
            sim_data = r.get("simulate_data", {})
            expr = sim_data.get("regular", "") if isinstance(sim_data, dict) else ""
            cache.add(sim_data, alpha_id, expression=expr)

        all_results.extend(batch_results)

        elapsed = time.monotonic() - batch_start
        succeeded = sum(1 for r in batch_results if r.get("alpha_id"))
        failed = len(batch_results) - succeeded

        logger.info(
            "└─ Batch %d done in %s — ✓ %d  ✗ %d",
            batch_idx + 1,
            fmt_duration(elapsed),
            succeeded,
            failed,
        )

        # ── Circuit breaker: stop if too many consecutive full failures ──
        if succeeded == 0:
            consecutive_full_failures += 1
            if consecutive_full_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(
                    "⛔ Circuit breaker: %d consecutive batches with 0%% success. "
                    "The expressions are likely invalid for this region/universe. "
                    "Stopping early to avoid wasting time. "
                    "Check the error messages above for the API's rejection reason.",
                    consecutive_full_failures,
                )
                if tg_cfg.get("enabled"):
                    send_telegram(
                        f"⛔ Pipeline stopped — {consecutive_full_failures} consecutive "
                        f"batches failed for {cfg['market']['region']}/{cfg['market']['universe']}"
                    )
                break
        else:
            consecutive_full_failures = 0  # reset on any success

        if tg_cfg.get("enabled"):
            send_telegram(
                f"🧠 Batch {batch_idx+1}/{total_batches} done\n"
                f"✓ {succeeded} | ✗ {failed} | ⏱ {fmt_duration(elapsed)}"
            )

    total_elapsed = time.monotonic() - pipeline_start
    total_ok = sum(1 for r in all_results if r.get("alpha_id"))
    logger.info(
        "Simulation complete — %d/%d succeeded in %s",
        total_ok,
        len(all_results),
        fmt_duration(total_elapsed),
    )
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 8 — Results Collection & Export
# ══════════════════════════════════════════════════════════════════════════════


def stage_collect_results(
    s: ace.SingleSession,
    simulation_results: list[dict],
    cfg: dict,
) -> pd.DataFrame | None:
    """Fetch full stats for each successful alpha and export."""
    logger.info("━━━ Stage 8: Results Collection ━━━")

    out_cfg = cfg["output"]
    mkt = cfg["market"]

    succeeded = [r for r in simulation_results if r.get("alpha_id")]
    if not succeeded:
        logger.warning("No successful simulations — nothing to export.")
        return None

    logger.info("Fetching stats for %d successful alphas…", len(succeeded))

    sim_config = {
        "get_pnl": out_cfg["save_pnl"],
        "get_stats": out_cfg["save_stats"],
        "save_pnl_file": out_cfg["save_pnl"],
        "save_stats_file": out_cfg["save_stats"],
        "save_result_file": out_cfg["save_result_file"],
        "check_submission": out_cfg["check_submission"],
        "check_self_corr": out_cfg["check_self_corr"],
        "check_prod_corr": out_cfg["check_prod_corr"],
    }

    stats_results = []

    def fetch_stats(r: dict) -> dict:
        return ace.get_specified_alpha_stats(
            s, r["alpha_id"], r["simulate_data"], **sim_config
        )

    with ThreadPool(3) as pool:
        for result in pool.imap_unordered(fetch_stats, succeeded):
            stats_results.append(result)

    # Build summary DataFrame
    try:
        pretty = hf.prettify_result(stats_results)
    except Exception as exc:
        logger.warning("prettify_result failed: %s — building minimal table", exc)
        rows = []
        for r in stats_results:
            row = {"alpha_id": r.get("alpha_id")}
            if isinstance(r.get("is_stats"), pd.DataFrame) and not r["is_stats"].empty:
                for col in r["is_stats"].columns:
                    if col != "alpha_id":
                        row[col] = r["is_stats"].iloc[0][col]
            sd = r.get("simulate_data", {})
            row["expression"] = sd.get("regular", str(sd))
            rows.append(row)
        pretty = pd.DataFrame(rows)

    # Save
    os.makedirs(out_cfg["results_dir"], exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    region = mkt["region"]

    csv_path = os.path.join(out_cfg["results_dir"], f"results_{region}_{ts}.csv")
    pretty.to_csv(csv_path, index=False)
    logger.info("Results saved → %s  (%d rows)", csv_path, len(pretty))

    # Print top alphas
    if "fitness" in pretty.columns:
        top = pretty.nlargest(10, "fitness")
        logger.info("Top 10 by fitness:\n%s", top.to_string(index=False))

    if cfg["telegram"].get("enabled"):
        send_telegram(
            f"✅ Pipeline complete — {region}\n"
            f"Alphas simulated: {len(succeeded)}\n"
            f"Results: {csv_path}"
        )

    return pretty


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Automated WorldQuant BRAIN Simulation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Template placeholders:
  {ts_op}         Time-Series operators
  {vec_op}        Vector operators
  {field}         All datafields
  {vector_field}  VECTOR-type datafields
  {matrix_field}  MATRIX-type datafields
  {pos}           Positive-sentiment fields
  {neg}           Negative-sentiment fields
  {lookback}      Lookback windows from config

Examples:
  python pipeline.py --template "ts_delta({field}, {lookback})"
  python pipeline.py --template "{ts_op}(rank({vector_field}), {lookback})" --dry-run
  python pipeline.py --interactive
""",
    )
    parser.add_argument(
        "--config",
        default="pipeline_config.yaml",
        help="Path to YAML config file (default: pipeline_config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate expressions only — do not simulate",
    )
    parser.add_argument("--region", help="Override market.region from config")
    parser.add_argument("--universe", help="Override market.universe from config")
    parser.add_argument("--category", help="Override dataset_filters.category")
    parser.add_argument(
        "--lookback",
        type=int,
        nargs="+",
        help="Override expressions.lookback_days (space-separated ints)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively prompt for custom expression templates",
    )
    parser.add_argument(
        "--template",
        action="append",
        default=[],
        metavar="EXPR",
        help="Custom expression template with {placeholders} (repeatable)",
    )
    parser.add_argument(
        "--only-custom",
        action="store_true",
        help="Disable built-in templates, run ONLY custom/interactive templates",
    )
    parser.add_argument(
        "--template-file",
        help="Path to a text file containing custom expression templates (one per line)",
    )

    # ── Similarity exclusion options ──────────────────────────────────────
    sim_group = parser.add_argument_group(
        "similarity filtering",
        "Control which similarity filters are applied to the generated expressions.",
    )
    sim_group.add_argument(
        "--no-similarity",
        action="store_true",
        help="Disable ALL similarity filtering (both historical and intra-batch)",
    )
    sim_group.add_argument(
        "--no-historical-similarity",
        action="store_true",
        help="Skip similarity check against previously simulated alphas in the DB",
    )
    sim_group.add_argument(
        "--no-intra-similarity",
        action="store_true",
        help="Skip intra-batch structural fingerprint dedup (keep all field variants)",
    )
    sim_group.add_argument(
        "--max-per-structure",
        type=int,
        default=1,
        metavar="N",
        help="Keep N expressions per unique operator structure (default: 1)",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    cfg = load_config(args.config)

    # Similarity filter settings (CLI → config)
    cfg["similarity"] = {
        "skip_historical": args.no_similarity or args.no_historical_similarity,
        "skip_intra": args.no_similarity or args.no_intra_similarity,
        "max_per_structure": args.max_per_structure,
    }

    # CLI overrides
    if args.region:
        cfg["market"]["region"] = args.region
    if args.universe:
        cfg["market"]["universe"] = args.universe
    if args.category:
        cfg["dataset_filters"]["category"] = args.category
    if args.lookback:
        cfg["expressions"]["lookback_days"] = args.lookback

    # If --only-custom, disable built-in templates
    if args.only_custom:
        cfg["expressions"]["templates"]["sentiment_comparison"] = False
        cfg["expressions"]["templates"]["vector_ops"] = False
        cfg["expressions"]["templates"]["matrix_ops"] = False

    region = cfg["market"]["region"]
    universe = cfg["market"]["universe"]
    mode = (
        "interactive" if args.interactive
        else f"{len(args.template)} CLI templates" if args.template
        else "auto"
    )
    logger.info(
        "╔══════════════════════════════════════════════════════════════╗"
    )
    logger.info(
        "║  WorldQuant BRAIN — Automated Simulation Pipeline          ║"
    )
    logger.info(
        "╚══════════════════════════════════════════════════════════════╝"
    )
    logger.info(
        "Region: %s  |  Universe: %s  |  Mode: %s  |  Dry-run: %s",
        region, universe, mode, args.dry_run,
    )

    pipeline_start = time.monotonic()

    # ── Stage 1 ───────────────────────────────────────────────────────────
    s = stage_authenticate()

    # ── Stage 2 ───────────────────────────────────────────────────────────
    datasets = stage_discover_datasets(s, cfg)
    if datasets.empty:
        logger.error("No datasets found — exiting.")
        sys.exit(1)

    # ── Stage 3 ───────────────────────────────────────────────────────────
    datafields = stage_fetch_datafields(s, datasets, cfg)
    if datafields.empty:
        logger.error("No datafields found — exiting.")
        sys.exit(1)

    # ── Stage 4 ───────────────────────────────────────────────────────────
    vector_ops, ts_ops = stage_load_operators(s)

    # ── Collect custom templates ──────────────────────────────────────────
    custom_templates: list[str] = list(args.template)  # from --template flags

    if args.template_file:
        try:
            with open(args.template_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        custom_templates.append(line)
            logger.info("Loaded %d templates from %s", len(custom_templates) - len(args.template), args.template_file)
        except Exception as e:
            logger.error("Failed to read template file %s: %s", args.template_file, e)
            sys.exit(1)

    if args.interactive:
        lookbacks = cfg["expressions"]["lookback_days"]
        interactive_exprs = prompt_templates_interactive(
            datafields, vector_ops, ts_ops, lookbacks
        )
        # For interactive mode, the expressions are already expanded.
        # We inject them as literal expressions (no placeholders left).
        # We'll pass the raw templates to Stage 5 for expansion instead.
        # Actually, prompt_templates_interactive already expands them,
        # so we need a different approach: collect raw templates.
        # Let's re-prompt but just collect the raw template strings.
        # -- The interactive prompt already shows a preview and collects
        #    raw templates. Let's refactor to collect raw strings.
        pass  # handled below via a cleaner approach

    # If interactive, re-prompt to collect raw template strings
    if args.interactive:
        print("\n" + "═" * 66)
        print("  CUSTOM EXPRESSION TEMPLATE BUILDER")
        print("═" * 66)

        ids = datafields["id"].tolist() if "id" in datafields.columns else []
        has_type = "type" in datafields.columns
        vec_fields = (
            datafields[datafields["type"] == "VECTOR"]["id"].tolist()
            if has_type else []
        )
        mat_fields = (
            datafields[datafields["type"] == "MATRIX"]["id"].tolist()
            if has_type else []
        )
        risk_mat_fields = [f for f in mat_fields if "risk" in str(f).lower() or "rsk" in str(f).lower()] if has_type else [f for f in ids if "risk" in str(f).lower() or "rsk" in str(f).lower()]
        pos_fields = [f for f in ids if "pos" in str(f).lower()]
        neg_fields = [f for f in ids if "neg" in str(f).lower()]
        lookbacks = cfg["expressions"]["lookback_days"]

        def _sample(lst, n=3):
            shown = lst[:n]
            suffix = f" … (+{len(lst)-n} more)" if len(lst) > n else ""
            return ", ".join(str(x) for x in shown) + suffix

        print("\n  Available placeholders and current values:\n")
        print(f"    {{ts_op}}         → {len(ts_ops)} ops   : {_sample(ts_ops)}")
        print(f"    {{vec_op}}        → {len(vector_ops)} ops   : {_sample(vector_ops)}")
        print(f"    {{field}}         → {len(ids)} fields : {_sample(ids)}")
        print(f"    {{vector_field}}  → {len(vec_fields)} fields : {_sample(vec_fields)}")
        print(f"    {{matrix_field}}  → {len(mat_fields)} fields : {_sample(mat_fields)}")
        print(f"    {{risk_matrix}}   → {len(risk_mat_fields)} fields : {_sample(risk_mat_fields)}")
        print(f"    {{pos}}           → {len(pos_fields)} fields : {_sample(pos_fields)}")
        print(f"    {{neg}}           → {len(neg_fields)} fields : {_sample(neg_fields)}")
        print(f"    {{lookback}}      → {lookbacks}")
        print("\n  Examples:")
        print("    ts_delta({field}, {lookback})")
        print("    rank({ts_op}({vector_field}, {lookback}))")
        print("    {ts_op}({vec_op}({neg}) - {vec_op}({pos}), {lookback})")
        print("    group_neutralize({ts_op}({field}, {lookback}), sector)")
        print("\n  Enter templates one per line. Empty line to finish.\n")

        while True:
            try:
                line = input(f"  template[{len(custom_templates)+1}] ▸ ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                break
            phs = _extract_placeholders(line)
            known = {"ts_op", "vec_op", "field", "vector_field", "matrix_field",
                     "risk_matrix", "pos", "neg", "lookback", "a"}
            unknown = set(phs) - known
            if unknown:
                print(f"    ⚠  Unknown placeholders: {unknown} — will be kept as literals")
            preview = expand_template(line, datafields, vector_ops, ts_ops, lookbacks)
            print(f"    → will generate {len(preview)} expressions")
            if preview:
                print(f"    → sample: {preview[0]}")
            custom_templates.append(line)

        if custom_templates:
            print(f"\n  ✓ {len(custom_templates)} custom template(s) queued")
        print("═" * 66 + "\n")

    # ── Stage 5 ───────────────────────────────────────────────────────────
    expressions = stage_generate_expressions(
        datafields, vector_ops, ts_ops, cfg,
        custom_templates=custom_templates if custom_templates else None,
    )
    if not expressions:
        logger.error("No expressions generated — exiting.")
        sys.exit(1)

    # ── Stage 6 ───────────────────────────────────────────────────────────
    alpha_list = stage_generate_alphas(expressions, cfg)

    if args.dry_run:
        logger.info("═══ DRY RUN — Skipping simulation ═══")
        logger.info("Expressions generated: %d", len(expressions))
        logger.info("Alpha dicts ready:     %d", len(alpha_list))
        logger.info("Sample expressions:")
        for expr in expressions[:10]:
            logger.info("  • %s", expr)
        total = time.monotonic() - pipeline_start
        logger.info("Dry run completed in %s", fmt_duration(total))
        sys.exit(0)

    # ── Stage 7 ───────────────────────────────────────────────────────────
    results = stage_simulate(s, alpha_list, cfg)

    # ── Stage 8 ───────────────────────────────────────────────────────────
    stage_collect_results(s, results, cfg)

    total = time.monotonic() - pipeline_start
    logger.info(
        "══════════════════════════════════════════════════════════════"
    )
    logger.info("Pipeline finished in %s", fmt_duration(total))
    logger.info(
        "══════════════════════════════════════════════════════════════"
    )


if __name__ == "__main__":
    main()
