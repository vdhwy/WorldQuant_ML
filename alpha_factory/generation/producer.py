"""Async queue-refilling candidate producer composing bandit, generators, validator, and pre-dedup."""

from __future__ import annotations

import hashlib
import inspect
import logging
import random
import re
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from alpha_factory.generation.algorithmic_llm import AlgorithmicLLMGenerator
from alpha_factory.generation.ast_mutator import SafeASTMutator
from alpha_factory.generation.bandit import ThompsonBanditScheduler, format_arm_key
from alpha_factory.generation.pre_dedup import PreSimDeduplicator
from alpha_factory.generation.template_expander import expand_template, load_templates_yaml
from alpha_factory.generation.validator import ExpressionValidator
from alpha_factory.storage.db import AlphaDB
from alpha_factory.storage.models import CandidateItem, DataFieldRecord, SimulationSettings

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "config" / "templates.yaml"
_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_DEFAULT_VECTOR_OPS: tuple[str, ...] = ("vec_avg", "vec_sum", "vec_max", "vec_min")
_DEFAULT_TS_OPS: tuple[str, ...] = (
    "ts_rank",
    "ts_zscore",
    "ts_mean",
    "ts_delta",
    "ts_decay_linear",
    "ts_std_dev",
    "ts_backfill",
)

_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "pv": ("pv", "price", "volume", "price_volume", "price-volume"),
    "fundamental": ("fundamental", "fnd", "financial"),
    "analyst": ("analyst", "anl", "estimate"),
    "sentiment": ("sentiment", "sent", "news"),
    "news": ("news", "sentiment"),
}


def _load_default_templates() -> list[str]:
    if not _DEFAULT_TEMPLATES_PATH.is_file():
        return []
    return load_templates_yaml(_DEFAULT_TEMPLATES_PATH)


def _normalize_templates(templates: Optional[Sequence[str | Mapping[str, Any]]]) -> list[str]:
    if templates is None:
        return _load_default_templates()
    patterns: list[str] = []
    for item in templates:
        if isinstance(item, Mapping):
            pat = item.get("pattern") or item.get("template")
            if pat:
                patterns.append(str(pat))
        elif isinstance(item, str) and item.strip():
            patterns.append(item.strip())
    return patterns


def _fallback_config_hash(expression: str, settings_json: str) -> str:
    payload = f"{expression.strip()}|{settings_json}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fallback_ast_fingerprint(expression: str) -> str:
    normalized = re.sub(r"\s+", "", expression.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _infer_dataset_id(expression: str, field_map: Mapping[str, str], default_dataset: str = "") -> str:
    for token in _TOKEN_RE.findall(expression):
        if token in field_map and field_map[token]:
            return field_map[token]
    return default_dataset


def _call_with_accepted_kwargs(fn: Callable[..., Any], *pos_args: Any, **kwargs: Any) -> Any:
    """Invoke ``fn`` passing only keyword arguments accepted by its signature."""
    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return fn(*pos_args, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(*pos_args, **filtered)


def _is_valid_result(result: Any) -> bool:
    if isinstance(result, bool):
        return result
    if isinstance(result, tuple):
        return bool(result[0]) if result else False
    for attr in ("valid", "is_valid", "ok"):
        if hasattr(result, attr):
            return bool(getattr(result, attr))
    return bool(result)


class CandidateProducer:
    """Composes bandit scheduling, expression generation, validation, and pre-sim dedup."""

    def __init__(
        self,
        db: AlphaDB,
        settings: Mapping[str, Any],
        bandit: Optional[ThompsonBanditScheduler] = None,
        validator: Optional[ExpressionValidator] = None,
        pre_dedup: Optional[PreSimDeduplicator] = None,
        algo_llm: Optional[AlgorithmicLLMGenerator] = None,
        mutator: Optional[SafeASTMutator] = None,
        templates: Optional[Sequence[str | Mapping[str, Any]]] = None,
        region_cooldowns_fn: Optional[Callable[[], Mapping[str, float]]] = None,
        poll_interval_s: float = 1.0,
    ) -> None:
        self.db = db
        self.settings: dict[str, Any] = dict(settings)
        gen_cfg = self.settings.get("generation") or {}
        llm_cfg = gen_cfg.get("llm") or {}
        self._region_cooldowns_fn = region_cooldowns_fn
        self._poll_interval_s = max(0.05, float(poll_interval_s))
        self.queue_low_watermark = int(gen_cfg.get("queue_low_watermark", 100))
        self.queue_high_watermark = int(gen_cfg.get("queue_high_watermark", 500))
        self.lookbacks: tuple[int, ...] = tuple(
            int(x) for x in (gen_cfg.get("lookback_days") or (5, 10, 20, 60, 120))
        )

        self.bandit = bandit or ThompsonBanditScheduler(db=db, settings=self.settings)

        operators = db.get_operators()
        self.validator = validator or ExpressionValidator(
            operators=operators,
            compiler_rules=db.get_compiler_rules(),
        )
        self.pre_dedup = pre_dedup or self._build_default_pre_dedup()
        self.algo_llm = algo_llm or AlgorithmicLLMGenerator(
            llm_cfg=llm_cfg,
            operators=operators,
            datafields=[],
        )
        self.mutator = mutator or self._build_default_mutator()
        self.templates: list[str] = _normalize_templates(templates)

        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._rng = random.Random()
        self._thread: Optional[threading.Thread] = None

    def _build_default_pre_dedup(self) -> PreSimDeduplicator:
        gen_cfg = self.settings.get("generation") or {}
        max_per_structure = int(gen_cfg.get("max_per_structure", 3))
        max_per_dataset_structure = int(gen_cfg.get("max_per_dataset_structure", 2))
        return _call_with_accepted_kwargs(
            PreSimDeduplicator,
            db=self.db,
            max_per_structure=max_per_structure,
            max_per_dataset_structure=max_per_dataset_structure,
            settings=self.settings,
        )

    def _build_default_mutator(self) -> SafeASTMutator:
        return _call_with_accepted_kwargs(
            SafeASTMutator,
            validator=self.validator,
            lookback_grid=self.lookbacks,
            operators=self.db.get_operators(),
            settings=self.settings,
        )

    def _select_arm(self, region: Optional[str]) -> dict[str, Any]:
        cooldowns = dict(self._region_cooldowns_fn()) if self._region_cooldowns_fn else None
        sig = inspect.signature(self.bandit.select_arm)
        params = sig.parameters
        kwargs: dict[str, Any] = {}
        if "region_cooldowns" in params:
            kwargs["region_cooldowns"] = cooldowns
        if region is not None and "region" in params:
            kwargs["region"] = region
        if kwargs:
            arm = dict(self.bandit.select_arm(**kwargs))
        elif cooldowns is not None and len(params) >= 1:
            arm = dict(self.bandit.select_arm(cooldowns))
        else:
            arm = dict(self.bandit.select_arm())

        if region is not None and arm.get("region") != region:
            mkt_cfg = (self.settings.get("markets") or {}).get(region, {})
            exec_defaults = (self.settings.get("execution") or {}).get("default_settings") or {}
            universe = str(mkt_cfg.get("default_universe") or arm.get("universe") or "TOP3000")
            delay = int(mkt_cfg.get("delay", arm.get("delay", 1)))
            neutralization = str(
                mkt_cfg.get("default_neutralization") or arm.get("neutralization") or "INDUSTRY"
            )
            decay = int(arm.get("decay", exec_defaults.get("decay", 4)))
            category = str(arm.get("category") or "fundamental")
            generator_type = str(arm.get("generator_type") or "template")
            arm.update(
                {
                    "region": region,
                    "universe": universe,
                    "delay": delay,
                    "neutralization": neutralization,
                    "decay": decay,
                    "arm_key": format_arm_key(
                        region=region,
                        universe=universe,
                        category=category,
                        generator_type=generator_type,
                        neutralization=neutralization,
                        decay=decay,
                    ),
                }
            )
        return arm

    def _resolve_datafields(
        self,
        region: str,
        universe: str,
        delay: int,
        category: str,
    ) -> list[DataFieldRecord]:
        fields = self.db.get_datafields(region, universe, delay)
        if not fields:
            mkt_cfg = (self.settings.get("markets") or {}).get(region, {})
            default_univ = str(mkt_cfg.get("default_universe") or "")
            if default_univ and default_univ != universe:
                fields = self.db.get_datafields(region, default_univ, delay)
        if not fields and self.algo_llm.datafields:
            regional = [f for f in self.algo_llm.datafields if f.region == region]
            fields = regional if regional else list(self.algo_llm.datafields)
        if not fields or not category:
            return fields

        cat_lower = category.lower()
        aliases = _CATEGORY_ALIASES.get(cat_lower, (cat_lower,))
        filtered = [
            f
            for f in fields
            if any(alias in (f.category or "").lower() for alias in aliases)
        ]
        return filtered if filtered else fields

    def _normalize_generated_pairs(
        self,
        raw_items: Any,
        field_map: Mapping[str, str],
        default_dataset: str = "",
    ) -> list[tuple[str, str]]:
        if not raw_items:
            return []
        if isinstance(raw_items, str):
            raw_items = [raw_items]

        pairs: list[tuple[str, str]] = []
        for item in raw_items:
            if isinstance(item, tuple) and len(item) >= 2:
                expr, ds_id = str(item[0]), str(item[1] or "")
            elif isinstance(item, Mapping):
                expr = str(item.get("expression") or "")
                ds_id = str(item.get("dataset_id") or "")
            elif hasattr(item, "expression"):
                expr = str(getattr(item, "expression") or "")
                ds_id = str(getattr(item, "dataset_id", "") or "")
            elif isinstance(item, str):
                expr = item
                ds_id = ""
            else:
                continue
            expr = expr.strip()
            if not expr:
                continue
            if not ds_id:
                ds_id = _infer_dataset_id(expr, field_map, default_dataset)
            pairs.append((expr, ds_id))
        return pairs

    def _operator_pools_for_templates(self) -> tuple[list[str], list[str]]:
        operators = self.db.get_operators()
        vector_ops: list[str] = []
        ts_ops: list[str] = []
        for op in operators:
            name = (op.name or "").strip()
            if not name:
                continue
            cat = (op.category or "").lower()
            if (name.startswith("vec_") or "vector" in cat) and int(op.min_inputs or 1) <= 1 and not op.uses_lookback:
                vector_ops.append(name)
            elif (
                op.uses_lookback
                and int(op.min_inputs or 1) == 1
                and not name.startswith("group_")
                and "group" not in cat
            ):
                ts_ops.append(name)

        if not vector_ops:
            vector_ops = list(_DEFAULT_VECTOR_OPS)
        elif "vec_avg" in vector_ops:
            vector_ops = ["vec_avg"] + [op for op in vector_ops if op != "vec_avg"]

        if not ts_ops:
            ts_ops = list(_DEFAULT_TS_OPS)
        else:
            preferred = [op for op in _DEFAULT_TS_OPS if op in ts_ops]
            rest = [op for op in ts_ops if op not in preferred]
            ts_ops = preferred + rest

        return vector_ops, ts_ops

    def _generate_from_templates(
        self,
        region: str,
        universe: str,
        delay: int,
        fields: Sequence[DataFieldRecord],
        count: int,
        lookbacks: Sequence[int],
    ) -> list[tuple[str, str]]:
        field_map = {f.field_id: f.dataset_id for f in fields}
        default_ds = fields[0].dataset_id if fields else ""
        templates = list(self.templates or _load_default_templates())
        if not templates:
            return []

        vector_ops, ts_ops = self._operator_pools_for_templates()
        groups: tuple[str, ...] = ("sector", "industry", "subindustry")
        if region.upper() in {"GLB", "EUR", "ASI"}:
            groups = ("sector", "industry", "subindustry", "country")

        self._rng.shuffle(templates)
        per_tmpl_limit = max(4, (count + len(templates) - 1) // len(templates) + 2)
        collected: list[tuple[str, str]] = []
        seen: set[str] = set()

        for tmpl in templates:
            if len(collected) >= count:
                break
            raw = _call_with_accepted_kwargs(
                expand_template,
                template=tmpl,
                datafields=list(fields),
                vector_ops=vector_ops,
                time_series_ops=ts_ops,
                lookbacks=list(lookbacks),
                groups=groups,
                max_per_template=per_tmpl_limit,
                region=region,
                universe=universe,
                delay=delay,
            )
            for expr, ds_id in self._normalize_generated_pairs(raw, field_map, default_ds):
                if expr not in seen:
                    seen.add(expr)
                    collected.append((expr, ds_id))
                    if len(collected) >= count:
                        break

        return collected

    def _generate_from_evolution(
        self,
        region: str,
        fields: Sequence[DataFieldRecord],
        count: int,
        lookbacks: Sequence[int],
    ) -> Optional[list[tuple[str, str]]]:
        sims = self.db.get_simulations(region=region, limit=200)
        parents = [
            s
            for s in sims
            if (s.expression or "").strip() and str(s.status).upper() in {"COMPLETE", "WARNING"}
        ]
        if len(parents) < 5:
            return None

        parents.sort(
            key=lambda s: (
                1 if s.gate2_pass else 0,
                1 if s.gate1_pass else 0,
                float(s.sharpe if s.sharpe is not None else 0.0),
                float(s.fitness if s.fitness is not None else 0.0),
            ),
            reverse=True,
        )
        top_parents = parents[: min(20, len(parents))]
        parent_exprs = {p.expression.strip() for p in top_parents}
        field_map = {f.field_id: f.dataset_id for f in fields}
        fields_by_type: dict[str, list[str]] = {}
        for f in fields:
            ftype = str(f.field_type or "MATRIX").upper()
            fields_by_type.setdefault(ftype, []).append(f.field_id)
        default_ds = fields[0].dataset_id if fields else ""

        collected: list[tuple[str, str]] = []
        seen: set[str] = set(parent_exprs)

        mutate_batch = getattr(self.mutator, "mutate_batch", None)
        if callable(mutate_batch):
            raw = _call_with_accepted_kwargs(
                mutate_batch,
                expressions=[p.expression for p in top_parents],
                parents=[p.expression for p in top_parents],
                datafields=list(fields),
                fields=list(fields),
                fields_by_type=fields_by_type,
                operators=self.db.get_operators(),
                lookbacks=list(lookbacks),
                count=count,
                n=count,
            )
            for expr, ds_id in self._normalize_generated_pairs(raw, field_map, default_ds):
                if expr not in seen:
                    seen.add(expr)
                    collected.append((expr, ds_id))

        if len(collected) < count:
            mutate_fn = None
            for name in ("mutate", "mutate_expression", "evolve"):
                fn = getattr(self.mutator, name, None)
                if callable(fn):
                    mutate_fn = fn
                    break
            if mutate_fn is not None:
                sig = inspect.signature(mutate_fn)
                param_names = list(sig.parameters.keys())
                per_parent = max(1, (count + len(top_parents) - 1) // len(top_parents))
                for parent in top_parents:
                    if len(collected) >= count:
                        break
                    for _ in range(per_parent):
                        if len(collected) >= count:
                            break
                        kwargs = {
                            "expression": parent.expression,
                            "expr": parent.expression,
                            "parent": parent.expression,
                            "datafields": list(fields),
                            "fields": list(fields),
                            "fields_by_type": fields_by_type,
                            "operators": self.db.get_operators(),
                            "lookbacks": list(lookbacks),
                            "count": 1,
                            "n": 1,
                        }
                        if param_names and param_names[0] not in kwargs:
                            raw = _call_with_accepted_kwargs(mutate_fn, parent.expression, **kwargs)
                        else:
                            raw = _call_with_accepted_kwargs(mutate_fn, **kwargs)
                        for expr, ds_id in self._normalize_generated_pairs(raw, field_map, default_ds):
                            if expr not in seen:
                                seen.add(expr)
                                collected.append((expr, ds_id))
                                if len(collected) >= count:
                                    break

        if len(collected) < count and len(top_parents) >= 2:
            crossover_fn = getattr(self.mutator, "crossover", None)
            if callable(crossover_fn):
                for idx in range(len(top_parents) - 1):
                    if len(collected) >= count:
                        break
                    raw = _call_with_accepted_kwargs(
                        crossover_fn,
                        top_parents[idx].expression,
                        top_parents[idx + 1].expression,
                    )
                    for expr, ds_id in self._normalize_generated_pairs(raw, field_map, default_ds):
                        if expr not in seen:
                            seen.add(expr)
                            collected.append((expr, ds_id))
                            if len(collected) >= count:
                                break

        return collected[:count] if collected else None

    def _build_simulation_settings(
        self,
        region: str,
        universe: str,
        delay: int,
        neutralization: str,
        decay: int,
    ) -> SimulationSettings:
        exec_defaults = (self.settings.get("execution") or {}).get("default_settings") or {}
        return SimulationSettings(
            instrument_type="EQUITY",
            region=region,
            universe=universe,
            delay=int(delay),
            decay=int(decay),
            neutralization=neutralization,
            truncation=float(exec_defaults.get("truncation", 0.08)),
            pasteurization=str(exec_defaults.get("pasteurization", "ON")),
            test_period=str(exec_defaults.get("test_period", "P2Y0M0D")),
            unit_handling=str(exec_defaults.get("unit_handling", "VERIFY")),
            nan_handling=str(exec_defaults.get("nan_handling", "OFF")),
            max_trade=str(exec_defaults.get("max_trade", "OFF")),
            language="FASTEXPR",
            visualization=bool(exec_defaults.get("visualization", False)),
        )

    def _dedup_and_prepare(
        self,
        valid_pairs: Sequence[tuple[str, str]],
        region: str,
        universe: str,
        delay: int,
        sim_settings: SimulationSettings,
        arm_key: str,
        priority: float,
        target: int,
    ) -> list[CandidateItem]:
        settings_json = sim_settings.to_json()
        raw_candidates = [
            CandidateItem(
                expression=expr,
                region=region,
                universe=universe,
                delay=delay,
                settings_json=settings_json,
                source=arm_key,
                dataset_id=ds_id,
                priority=priority,
                status="PENDING",
            )
            for expr, ds_id in valid_pairs
        ]

        filter_fn = self.pre_dedup.filter_candidates
        sig = inspect.signature(filter_fn)
        params = list(sig.parameters.values())
        param_names = {p.name for p in params}

        if "settings_dict" in param_names:
            filtered = _call_with_accepted_kwargs(
                filter_fn,
                list(valid_pairs),
                settings_dict=sim_settings,
                region=region,
                universe=universe,
                delay=delay,
                source=arm_key,
            )
        else:
            first_param = params[0] if params else None
            first_name = first_param.name if first_param else "candidates"
            first_ann = str(first_param.annotation or "") if first_param else ""
            expects_pairs = (
                first_name in {"expressions", "exprs", "pairs"}
                or ("tuple" in first_ann.lower() and "candidateitem" not in first_ann.lower())
            )
            input_arg: Any = list(valid_pairs) if expects_pairs else raw_candidates
            kwargs: dict[str, Any] = {
                "region": region,
                "universe": universe,
                "delay": delay,
                "settings_json": settings_json,
                "settings_dict": sim_settings,
                "source": arm_key,
                "priority": priority,
                "max_items": target,
                "limit": target,
            }
            try:
                filtered = _call_with_accepted_kwargs(filter_fn, input_arg, **kwargs)
            except (TypeError, KeyError, AttributeError):
                dict_candidates = [
                    {
                        "expression": c.expression,
                        "region": c.region,
                        "universe": c.universe,
                        "delay": c.delay,
                        "settings_json": c.settings_json,
                        "source": c.source,
                        "dataset_id": c.dataset_id,
                        "priority": c.priority,
                        "status": c.status,
                    }
                    for c in raw_candidates
                ]
                filtered = _call_with_accepted_kwargs(filter_fn, dict_candidates, **kwargs)

        if not filtered:
            return []

        prepared: list[CandidateItem] = []
        for entry in filtered:
            if len(prepared) >= target:
                break
            if isinstance(entry, CandidateItem):
                item = entry
                item.source = arm_key
                item.priority = priority
                item.region = item.region or region
                item.universe = item.universe or universe
                item.delay = int(item.delay or delay)
                if not item.settings_json or item.settings_json == "{}":
                    item.settings_json = settings_json
                item.status = "PENDING"
            elif isinstance(entry, Mapping):
                expr = str(entry.get("expression") or "")
                if not expr:
                    continue
                item = CandidateItem(
                    config_hash=str(entry.get("config_hash") or ""),
                    ast_fingerprint=str(entry.get("ast_fingerprint") or ""),
                    expression=expr,
                    region=str(entry.get("region") or region),
                    universe=str(entry.get("universe") or universe),
                    delay=int(entry.get("delay", delay)),
                    settings_json=str(entry.get("settings_json") or settings_json),
                    source=arm_key,
                    dataset_id=str(entry.get("dataset_id") or ""),
                    priority=priority,
                    status="PENDING",
                )
            elif isinstance(entry, tuple) and len(entry) >= 2:
                item = CandidateItem(
                    expression=str(entry[0]),
                    region=region,
                    universe=universe,
                    delay=delay,
                    settings_json=settings_json,
                    source=arm_key,
                    dataset_id=str(entry[1] or ""),
                    priority=priority,
                    status="PENDING",
                )
            else:
                continue

            if not item.config_hash:
                item.config_hash = _fallback_config_hash(item.expression, item.settings_json)
            if not item.ast_fingerprint:
                item.ast_fingerprint = _fallback_ast_fingerprint(item.expression)
            prepared.append(item)

        return prepared

    def _validate_and_dedup(
        self,
        candidates: Sequence[tuple[str, str]],
        arm_region: str,
        universe: str,
        delay: int,
        sim_settings: SimulationSettings,
        arm_key: str,
        priority: float,
        target: int,
    ) -> list[CandidateItem]:
        if not candidates:
            return []
        valid_pairs: list[tuple[str, str]] = []
        seen_exprs: set[str] = set()
        for expr, ds_id in candidates:
            if expr in seen_exprs:
                continue
            if _is_valid_result(self.validator.validate(expr)):
                seen_exprs.add(expr)
                valid_pairs.append((expr, ds_id))
        if not valid_pairs:
            return []
        return self._dedup_and_prepare(
            valid_pairs=valid_pairs,
            region=arm_region,
            universe=universe,
            delay=delay,
            sim_settings=sim_settings,
            arm_key=arm_key,
            priority=priority,
            target=target,
        )

    def produce_once(self, region: Optional[str] = None, batch_target: int = 30) -> int:
        """Run one candidate generation, validation, dedup, and enqueue cycle."""
        if batch_target <= 0:
            return 0
        pending = self.db.count_queue(status="PENDING")
        if pending >= self.queue_high_watermark:
            return 0

        target = min(int(batch_target), max(0, self.queue_high_watermark - pending))
        if target <= 0:
            return 0

        arm = self._select_arm(region=region)
        arm_region = str(arm["region"])
        universe = str(arm["universe"])
        delay = int(arm.get("delay", 1))
        category = str(arm.get("category") or "")
        generator_type = str(arm.get("generator_type") or "template")
        neutralization = str(arm.get("neutralization") or "INDUSTRY")
        decay = int(arm.get("decay", 4))
        arm_key = str(arm["arm_key"])

        alpha = float(arm.get("alpha_param", 1.0))
        beta = float(arm.get("beta_param", 1.0))
        posterior_mean = alpha / max(1e-6, alpha + beta)
        if "sampled_theta" in arm:
            priority = round(0.5 * float(arm["sampled_theta"]) + 0.5 * posterior_mean, 6)
        else:
            priority = round(posterior_mean, 6)

        cat_fields = self._resolve_datafields(arm_region, universe, delay, category)
        all_fields = self._resolve_datafields(arm_region, universe, delay, "")
        if not cat_fields and not all_fields:
            logger.debug("No datafields available for (%s, %s, delay=%s)", arm_region, universe, delay)
            return 0

        sim_settings = self._build_simulation_settings(
            region=arm_region,
            universe=universe,
            delay=delay,
            neutralization=neutralization,
            decay=decay,
        )
        raw_target = max(target * 2, 20)

        if generator_type == "evolution":
            strategy_order = ("evolution", "template", "algorithmic_llm")
        elif generator_type == "template":
            strategy_order = ("template", "algorithmic_llm")
        else:
            strategy_order = ("algorithmic_llm", "template")

        field_pools: list[list[DataFieldRecord]] = [cat_fields] if cat_fields else []
        if all_fields and len(all_fields) > len(cat_fields):
            field_pools.append(all_fields)

        for field_pool in field_pools:
            for strat in strategy_order:
                if strat == "evolution":
                    generated = self._generate_from_evolution(
                        arm_region, field_pool, raw_target, self.lookbacks
                    ) or []
                elif strat == "template":
                    generated = self._generate_from_templates(
                        arm_region, universe, delay, field_pool, raw_target, self.lookbacks
                    )
                else:
                    generated = self.algo_llm.generate_batch(
                        region=arm_region,
                        universe=universe,
                        delay=delay,
                        datafields=field_pool,
                        count=raw_target,
                        lookbacks=self.lookbacks,
                    )
                items = self._validate_and_dedup(
                    candidates=generated,
                    arm_region=arm_region,
                    universe=universe,
                    delay=delay,
                    sim_settings=sim_settings,
                    arm_key=arm_key,
                    priority=priority,
                    target=target,
                )
                if items:
                    return int(self.db.enqueue_candidates(items))

        return 0

    @property
    def is_running(self) -> bool:
        """Return True while the background producer thread is alive."""
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                pending = self.db.count_queue(status="PENDING")
                if pending < self.queue_low_watermark:
                    zero_streak = 0
                    while (
                        not self._stop_event.is_set()
                        and pending < self.queue_high_watermark
                    ):
                        batch_target = min(30, self.queue_high_watermark - pending)
                        inserted = self.produce_once(batch_target=batch_target)
                        pending = self.db.count_queue(status="PENDING")
                        if inserted <= 0:
                            zero_streak += 1
                            if zero_streak >= 3:
                                break
                        else:
                            zero_streak = 0
            except Exception:
                logger.exception("Error in CandidateProducer background loop")
            self._stop_event.wait(self._poll_interval_s)

    def start(self) -> None:
        """Start the background daemon thread that keeps the candidate queue topped up."""
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run_loop,
                name="alpha-factory-candidate-producer",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the background producer thread to stop and join up to ``timeout`` seconds."""
        self._stop_event.set()
        with self._state_lock:
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._state_lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None


__all__ = ["CandidateProducer"]
