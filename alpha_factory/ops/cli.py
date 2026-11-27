"""Command-line interface for operating and inspecting alpha_factory."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from alpha_factory.brain.catalog import CatalogSync
from alpha_factory.brain.client import BrainClient
from alpha_factory.execution.quota import NY_TZ, DailyQuotaManager
from alpha_factory.ops.daemon import (
    AlphaMiningDaemon,
    iter_configured_region_delays,
    resolve_storage_paths,
    setup_logging,
)
from alpha_factory.qualification.dedup_vault import CorrelationDedupVault
from alpha_factory.qualification.submitter import SequentialSubmitter
from alpha_factory.storage.db import AlphaDB
from alpha_factory.storage.models import DEFAULT_SETTINGS, load_dotenv_file, load_settings
from alpha_factory.storage.pnl_store import PnlStore

DEFAULT_CONFIG_PATH = "alpha_factory/config/settings.yaml"

_CSV_COLUMNS: tuple[str, ...] = (
    "alpha_id",
    "region",
    "delay",
    "vault_status",
    "composite_score",
    "sharpe",
    "fitness",
    "turnover",
    "prod_corr_max",
    "max_local_corr",
    "max_local_corr_peer",
    "duplicate_of",
    "expression",
    "updated_at",
)


def _load_cli_settings(config_path: Optional[str]) -> dict[str, Any]:
    if config_path is None:
        return load_settings(None)
    expanded = Path(config_path).expanduser()
    if config_path == DEFAULT_CONFIG_PATH and not expanded.is_file():
        return load_settings(None)
    if not expanded.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_settings(str(expanded))


def _fmt_opt(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{precision}f}"


def _open_db(settings: Mapping[str, Any]) -> AlphaDB:
    db_path, _, _ = resolve_storage_paths(settings)
    return AlphaDB(db_path)


def _iter_sync_targets(
    settings: Mapping[str, Any],
    region_filter: Optional[str] = None,
) -> list[tuple[str, str, int]]:
    markets_cfg = settings.get("markets")
    if not isinstance(markets_cfg, Mapping) or not markets_cfg:
        markets_cfg = DEFAULT_SETTINGS["markets"]

    target_region = region_filter.strip().upper() if region_filter else None
    targets: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str, int]] = set()

    for raw_region, mkt_cfg in markets_cfg.items():
        if not isinstance(mkt_cfg, Mapping):
            continue
        region = str(raw_region).strip().upper()
        if not region or (target_region is not None and region != target_region):
            continue
        if target_region is None and mkt_cfg.get("enabled") is False:
            continue

        default_univ = str(mkt_cfg.get("default_universe") or "TOP3000").strip()
        raw_univs = mkt_cfg.get("universes")
        universes: list[str] = []
        if default_univ:
            universes.append(default_univ)
        if isinstance(raw_univs, (list, tuple)):
            for u in raw_univs:
                u_str = str(u).strip()
                if u_str and u_str not in universes:
                    universes.append(u_str)
        if not universes:
            universes = ["TOP3000"]

        raw_delays = mkt_cfg.get("delays")
        if isinstance(raw_delays, (list, tuple)) and raw_delays:
            delays = list(dict.fromkeys(int(d) for d in raw_delays))
        else:
            delays = [int(mkt_cfg.get("delay", 1))]

        for delay in delays:
            for universe in universes:
                key = (region, universe, delay)
                if key not in seen:
                    seen.add(key)
                    targets.append(key)

    if target_region is not None and not targets:
        targets.append((target_region, "TOP3000", 1))
    return targets


# ----------------------------------------------------------------------
# Subcommand handlers
# ----------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args.config)
    ops_cfg = settings.get("ops") or {}
    log_dir = str(ops_cfg.get("log_dir") or DEFAULT_SETTINGS["ops"]["log_dir"])
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    setup_logging(log_dir=log_dir, level=level)

    daemon = AlphaMiningDaemon(settings=settings)
    summary = daemon.run(
        dry_run=bool(args.dry_run),
        max_cycles=args.max_cycles,
        install_signals=True,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_sync_catalog(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args.config)
    db = _open_db(settings)

    if args.offline:
        catalog = CatalogSync(db=db, client=None)
        counts = catalog.bootstrap_from_local_cache()
        print(
            f"Offline catalog bootstrap complete: "
            f"operators={counts.get('operators', 0)} "
            f"datasets={counts.get('datasets', 0)} "
            f"datafields={counts.get('datafields', 0)}"
        )
        return 0

    client = BrainClient.from_settings(settings)
    client.ensure_authenticated()
    catalog = CatalogSync(db=db, client=client)

    op_count = catalog.sync_operators()
    total_ds = 0
    total_df = 0
    targets = _iter_sync_targets(settings, region_filter=args.region)
    for region, universe, delay in targets:
        ds_cnt, df_cnt = catalog.sync_region(
            instrument_type="EQUITY",
            region=region,
            delay=delay,
            universe=universe,
            max_datasets=args.max_datasets,
        )
        total_ds += ds_cnt
        total_df += df_cnt
        print(
            f"Synced ({region}, {universe}, delay={delay}): "
            f"datasets={ds_cnt} datafields={df_cnt}"
        )

    print(
        f"Online catalog sync complete: "
        f"operators={op_count} datasets={total_ds} datafields={total_df}"
    )
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args.config)
    db = _open_db(settings)
    quota = DailyQuotaManager.from_settings(db, settings)
    q_status = quota.get_status()
    stats = db.get_dashboard_stats(date_ny=q_status.date_ny)

    q_counts = stats["queue"]
    sim_counts = stats["simulations"]
    vault_counts = stats["vault"]
    cat_counts = stats["catalog"]

    sim_total = int(sim_counts.get("total", 0))
    sim_complete = int(sim_counts.get("complete", 0))
    g1_pass = int(sim_counts.get("gate1_pass", 0))
    g2_pass = int(sim_counts.get("gate2_pass", 0))

    denom = sim_complete if sim_complete > 0 else sim_total
    g1_rate = (100.0 * g1_pass / denom) if denom > 0 else 0.0
    g2_rate = (100.0 * g2_pass / denom) if denom > 0 else 0.0
    g2_cond_rate = (100.0 * g2_pass / g1_pass) if g1_pass > 0 else 0.0

    kept_cnt = int(vault_counts.get("KEPT", 0))
    corr_dup_cnt = int(vault_counts.get("CORR_DUP", 0))
    submitted_cnt = int(vault_counts.get("SUBMITTED", 0))
    dup_cnt = int(vault_counts.get("DUPLICATE", 0))
    rejected_cnt = int(vault_counts.get("REJECTED", 0))

    lines = [
        f"Alpha Factory Status (NY Date: {q_status.date_ny})",
        (
            f"NY Quota:     {q_status.submitted_count}/{q_status.limit} "
            f"(remaining={q_status.remaining}, completed={q_status.completed_count}, failed={q_status.failed_count})"
        ),
        (
            f"Queue:        PENDING={q_counts.get('PENDING', 0)} "
            f"DISPATCHING={q_counts.get('DISPATCHING', 0)} "
            f"IN_FLIGHT={q_counts.get('IN_FLIGHT', 0)} "
            f"DONE={q_counts.get('DONE', 0)} "
            f"FAILED={q_counts.get('FAILED', 0)} "
            f"(total={q_counts.get('total', 0)})"
        ),
        (
            f"Simulations:  total={sim_total} complete={sim_complete} "
            f"failed={sim_counts.get('failed', 0)} "
            f"gate1_pass={g1_pass} gate2_pass={g2_pass}"
        ),
        f"Gate-1 Rate:  {g1_rate:.1f}% ({g1_pass}/{denom})",
        f"Gate-2 Rate:  {g2_rate:.1f}% ({g2_pass}/{denom}, {g2_cond_rate:.1f}% of Gate-1)",
        (
            f"Vault:        KEPT={kept_cnt} CORR_DUP={corr_dup_cnt} SUBMITTED={submitted_cnt} "
            f"DUPLICATE={dup_cnt} REJECTED={rejected_cnt} (total={vault_counts.get('total', 0)})"
        ),
        (
            f"Catalog:      datasets={cat_counts.get('datasets', 0)} "
            f"datafields={cat_counts.get('datafields', 0)} "
            f"operators={cat_counts.get('operators', 0)}"
        ),
    ]
    print("\n".join(lines))
    return 0


def _cmd_vault_list(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args.config)
    db = _open_db(settings)

    region = args.region.strip().upper() if args.region else None
    status_filter = None if not args.status or args.status.strip().upper() == "ALL" else args.status.strip().upper()
    entries = db.get_vault_entries(
        region=region,
        statuses=status_filter,
        limit=max(1, int(args.limit)),
    )
    if not entries:
        print("No vault entries found.")
        return 0

    header = (
        f"{'ALPHA_ID':<12} {'REGION':<6} {'D':<2} {'STATUS':<10} "
        f"{'SCORE':>8} {'SHARPE':>7} {'FIT':>7} {'TVR':>7} {'PCORR':>7} {'EXPRESSION'}"
    )
    print(header)
    print("-" * len(header))
    for entry in entries:
        expr = entry.expression.replace("\n", " ")
        if len(expr) > 70:
            expr = expr[:67] + "..."
        print(
            f"{entry.alpha_id:<12} {entry.region:<6} {entry.delay:<2} {entry.vault_status:<10} "
            f"{entry.composite_score:>8.4f} {_fmt_opt(entry.sharpe, 2):>7} "
            f"{_fmt_opt(entry.fitness, 2):>7} {_fmt_opt(entry.turnover, 3):>7} "
            f"{_fmt_opt(entry.prod_corr_max, 3):>7} {expr}"
        )
    return 0


def _cmd_export_submittable(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args.config)
    db = _open_db(settings)

    status_filter = (
        None
        if not args.status or args.status.strip().upper() == "ALL"
        else args.status.strip().upper()
    )
    region = args.region.strip().upper() if args.region else None
    entries = db.get_vault_entries(region=region, statuses=status_filter)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_CSV_COLUMNS))
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "alpha_id": entry.alpha_id,
                    "region": entry.region,
                    "delay": entry.delay,
                    "vault_status": entry.vault_status,
                    "composite_score": entry.composite_score,
                    "sharpe": entry.sharpe,
                    "fitness": entry.fitness,
                    "turnover": entry.turnover,
                    "prod_corr_max": entry.prod_corr_max,
                    "max_local_corr": entry.max_local_corr,
                    "max_local_corr_peer": entry.max_local_corr_peer or "",
                    "duplicate_of": entry.duplicate_of or "",
                    "expression": entry.expression,
                    "updated_at": entry.updated_at or "",
                }
            )

    print(f"Exported {len(entries)} vault alpha(s) to {out_path}")
    return 0


def _cmd_reconcile(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args.config)
    db_path, pnl_dir, _ = resolve_storage_paths(settings)
    db = AlphaDB(db_path)
    pnl_store = PnlStore(pnl_dir)
    vault = CorrelationDedupVault(
        db=db,
        pnl_store=pnl_store,
        client=None,
        dedup_cfg=settings.get("dedup"),
        tag_on_brain=False,
    )

    region_filter = args.region.strip().upper() if args.region else None
    pairs = iter_configured_region_delays(settings)
    if region_filter is not None:
        pairs = [(r, d) for r, d in pairs if r.upper() == region_filter]
        if not pairs:
            pairs = [(region_filter, 1)]

    total_kept = 0
    total_demoted = 0
    total_promoted = 0
    for region, delay in pairs:
        summary = vault.reconcile_clusters(region, delay)
        total_kept += summary["kept"]
        total_demoted += summary["demoted"]
        total_promoted += summary["promoted"]
        print(
            f"Reconciled ({region}, delay={delay}): "
            f"kept={summary['kept']} demoted={summary['demoted']} promoted={summary['promoted']}"
        )

    print(
        f"Total reconciliation: kept={total_kept} demoted={total_demoted} promoted={total_promoted}"
    )
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args.config)
    db_path, pnl_dir, _ = resolve_storage_paths(settings)
    db = AlphaDB(db_path)
    pnl_store = PnlStore(pnl_dir)

    sub_cfg = settings.get("submission") or {}
    if not isinstance(sub_cfg, Mapping):
        sub_cfg = {}
    qual_cfg = settings.get("qualification") or {}
    if not isinstance(qual_cfg, Mapping):
        qual_cfg = {}
    max_per_day = int(sub_cfg.get("max_submissions_per_day", 2))
    limit = max(0, int(args.limit))

    if not args.yes:
        ny_date = datetime.now(NY_TZ).strftime("%Y-%m-%d")
        raw_today = db.get_runtime_state(f"submissions:{ny_date}", "0")
        try:
            submitted_today = max(0, int(raw_today or 0))
        except (TypeError, ValueError):
            submitted_today = 0
        remaining_today = max(0, max_per_day - submitted_today)
        preview_count = min(limit, remaining_today)
        entries = db.get_vault_entries(statuses="KEPT", limit=preview_count) if preview_count > 0 else []

        print(
            f"DRY-RUN PREVIEW (NY Date: {ny_date}, submitted today: {submitted_today}/{max_per_day}, "
            f"requested limit: {limit})"
        )
        if not entries:
            print("No eligible KEPT alphas to submit (or daily submission cap reached).")
        else:
            for idx, entry in enumerate(entries, start=1):
                print(
                    f"  {idx}. {entry.alpha_id} [{entry.region} d={entry.delay}] "
                    f"score={entry.composite_score:.4f} sharpe={_fmt_opt(entry.sharpe, 2)} "
                    f"fitness={_fmt_opt(entry.fitness, 2)} expr={entry.expression}"
                )
        print("Re-run with --yes to run live BRAIN checks and submit.")
        return 0

    client = BrainClient.from_settings(settings)
    client.ensure_authenticated()
    vault = CorrelationDedupVault(
        db=db,
        pnl_store=pnl_store,
        client=client,
        dedup_cfg=settings.get("dedup"),
        tag_on_brain=bool(qual_cfg.get("tag_on_brain", True)),
    )
    submitter = SequentialSubmitter(
        db=db,
        client=client,
        vault=vault,
        qual_cfg=qual_cfg,
        max_submissions_per_day=max_per_day,
    )
    results = submitter.submit_top_kept(limit=limit, dry_run=False)
    if not results:
        print("No alphas submitted (vault empty or daily submission cap reached).")
        return 0

    for item in results:
        print(
            f"{item['alpha_id']} [{item['region']} d={item['delay']}] "
            f"status={item['status']} submitted={item['submitted']} "
            f"self_corr={_fmt_opt(item.get('self_corr'), 4)} "
            f"prod_corr={_fmt_opt(item.get('prod_corr'), 4)} "
            f"reason={item.get('reason') or '-'}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``alpha_factory`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m alpha_factory",
        description="WorldQuant BRAIN 24/7 Alpha Factory CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run the alpha mining daemon")
    p_run.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to settings YAML")
    p_run.add_argument("--dry-run", action="store_true", help="Run offline generation only")
    p_run.add_argument("--max-cycles", type=int, default=None, help="Maximum cycles before clean exit")
    p_run.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, WARNING)")
    p_run.set_defaults(handler=_cmd_run)

    # sync-catalog
    p_sync = subparsers.add_parser("sync-catalog", help="Sync BRAIN operators, datasets, and datafields")
    p_sync.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to settings YAML")
    p_sync.add_argument("--offline", action="store_true", help="Bootstrap from local JSON caches")
    p_sync.add_argument("--region", default=None, help="Optional region filter (e.g. USA)")
    p_sync.add_argument("--max-datasets", type=int, default=None, help="Optional dataset cap per market")
    p_sync.set_defaults(handler=_cmd_sync_catalog)

    # status
    p_status = subparsers.add_parser("status", help="Show NY quota, queue, simulation, and vault stats")
    p_status.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to settings YAML")
    p_status.set_defaults(handler=_cmd_status)

    # vault-list
    p_vault = subparsers.add_parser("vault-list", help="List qualified alphas in the vault")
    p_vault.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to settings YAML")
    p_vault.add_argument("--region", default=None, help="Filter by region (e.g. USA)")
    p_vault.add_argument("--status", default=None, help="Filter by vault_status (e.g. KEPT, SUBMITTED, ALL)")
    p_vault.add_argument("--limit", type=int, default=50, help="Maximum rows to display")
    p_vault.set_defaults(handler=_cmd_vault_list)

    # export-submittable
    p_export = subparsers.add_parser("export-submittable", help="Export vault alphas to CSV")
    p_export.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to settings YAML")
    p_export.add_argument("--out", default="submittable_alphas.csv", help="Output CSV path")
    p_export.add_argument("--status", default="KEPT", help="Vault status filter (default: KEPT)")
    p_export.add_argument("--region", default=None, help="Optional region filter")
    p_export.set_defaults(handler=_cmd_export_submittable)

    # reconcile
    p_reconcile = subparsers.add_parser("reconcile", help="Re-cluster vault entries across configured regions")
    p_reconcile.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to settings YAML")
    p_reconcile.add_argument("--region", default=None, help="Optional region filter")
    p_reconcile.set_defaults(handler=_cmd_reconcile)

    # submit
    p_submit = subparsers.add_parser("submit", help="Preview or sequentially submit top KEPT alphas")
    p_submit.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to settings YAML")
    p_submit.add_argument("--limit", type=int, default=1, help="Max alphas to preview/submit")
    p_submit.add_argument("--yes", action="store_true", help="Confirm live re-check and submission")
    p_submit.set_defaults(handler=_cmd_submit)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint returning process exit status."""
    load_dotenv_file()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


__all__ = ["DEFAULT_CONFIG_PATH", "build_parser", "main"]
