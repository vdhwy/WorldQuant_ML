"""Data models, configuration loading, and environment helpers for alpha_factory."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

_COMMON_NEUTRALIZATIONS: list[str] = [
    "INDUSTRY",
    "SUBINDUSTRY",
    "SECTOR",
    "COUNTRY",
    "REVERSION_AND_MOMENTUM",
    "STATISTICAL",
    "CROWDING",
    "FAST",
    "SLOW",
    "MARKET",
    "SLOW_AND_FAST",
]

DEFAULT_SETTINGS: dict[str, Any] = {
    "brain": {
        "api_url": "https://api.worldquantbrain.com",
        "connect_timeout_s": 10,
        "read_timeout_s": 60,
        "session_cache_path": "~/secrets/alpha-factory-session.json",
        "credentials_path": "~/secrets/platform-brain.json",
        "relogin_ttl_threshold_s": 1800,
        "biometric_poll_interval_s": 15,
        "biometric_max_wait_s": 1800,
    },
    "polling": {
        "max_wait_s": 3600,
        "recordset_max_wait_s": 600,
        "default_interval_s": 5,
        "max_http_retries": 3,
    },
    "storage": {
        "db_path": "alpha_factory/data/alpha_factory.db",
        "pnl_dir": "alpha_factory/data/pnl",
        "backup_dir": "alpha_factory/data/backups",
    },
    "markets": {
        "USA": {
            "default_universe": "TOP3000",
            "universes": ["TOP3000", "TOP1000", "TOP500", "TOP200", "ILLIQUID_MINVOL1M"],
            "neutralizations": list(_COMMON_NEUTRALIZATIONS),
            "default_neutralization": "INDUSTRY",
            "slots_required": 1,
            "delay": 1,
        },
        "EUR": {
            "default_universe": "TOP2500",
            "universes": ["TOP2500", "TOP1200", "TOP800", "TOP400", "ILLIQUID_MINVOL1M"],
            "neutralizations": list(_COMMON_NEUTRALIZATIONS),
            "default_neutralization": "INDUSTRY",
            "slots_required": 1,
            "delay": 1,
        },
        "ASI": {
            "default_universe": "MINVOL1M",
            "universes": ["MINVOL1M", "ILLIQUID_MINVOL1M"],
            "neutralizations": list(_COMMON_NEUTRALIZATIONS),
            "default_neutralization": "INDUSTRY",
            "slots_required": 1,
            "delay": 1,
        },
        "CHN": {
            "default_universe": "TOP2000U",
            "universes": ["TOP2000U"],
            "neutralizations": [
                "INDUSTRY",
                "SUBINDUSTRY",
                "SECTOR",
                "REVERSION_AND_MOMENTUM",
                "CROWDING",
                "FAST",
                "SLOW",
                "MARKET",
                "SLOW_AND_FAST",
            ],
            "default_neutralization": "INDUSTRY",
            "slots_required": 1,
            "delay": 1,
        },
        "IND": {
            "default_universe": "TOP500",
            "universes": ["TOP500"],
            "neutralizations": list(_COMMON_NEUTRALIZATIONS),
            "default_neutralization": "INDUSTRY",
            "slots_required": 1,
            "delay": 1,
        },
        "GLB": {
            "default_universe": "TOP3000",
            "universes": ["TOP3000", "MINVOL1M", "TOPDIV3000"],
            "neutralizations": list(_COMMON_NEUTRALIZATIONS),
            "default_neutralization": "INDUSTRY",
            "slots_required": 2,
            "delay": 1,
        },
    },
    "execution": {
        "max_slots": 3,
        "multi_sim_limit": 10,
        "daily_sim_limit": 5000,
        "warning_threshold": 4000,
        "hourly_pacing": True,
        "circuit_breaker_threshold": 3,
        "region_cooldown_s": 1800,
        "default_settings": {
            "alpha_type": "REGULAR",
            "delay": 1,
            "decay": 4,
            "neutralization": "INDUSTRY",
            "truncation": 0.08,
            "pasteurization": "ON",
            "unit_handling": "VERIFY",
            "nan_handling": "OFF",
            "max_trade": "OFF",
            "test_period": "P2Y0M0D",
            "visualization": False,
        },
    },
    "generation": {
        "queue_low_watermark": 100,
        "queue_high_watermark": 500,
        "mix": {
            "template": 0.40,
            "algorithmic_llm": 0.35,
            "evolution": 0.25,
        },
        "max_per_structure": 3,
        "max_per_dataset_structure": 2,
        "lookback_days": [5, 10, 20, 60, 120],
        "llm": {
            "backend": "ollama",
            "base_url": "http://localhost:11434",
            "models": [
                "deepseek-r1:8b",
                "qwen2.5-coder:7b",
                "deepseek-r1:1.5b",
                "llama3:3b",
                "phi3:mini",
            ],
            "api_base_url": "",
            "api_model": "",
            "timeout_s": 45,
        },
    },
    "qualification": {
        "min_sharpe": 1.58,
        "min_fitness": 1.0,
        "min_turnover": 0.01,
        "max_turnover": 0.70,
        "min_margin": 0.0008,
        "max_sanity_sharpe": 6.0,
        "max_sanity_returns": 0.50,
        "max_self_corr": 0.70,
        "max_prod_corr": 0.70,
        "check_power_pool": False,
        "max_power_pool_corr": 0.50,
        "flatline_zero_ratio_max": 0.30,
        "flatline_tail_days": 252,
        "tag_on_brain": True,
    },
    "dedup": {
        "local_corr_threshold": 0.65,
        "min_overlap_days": 120,
        "trailing_window_days": 504,
        "replacement_hysteresis": 0.05,
    },
    "submission": {
        "auto_submit": False,
        "max_submissions_per_day": 2,
    },
    "ops": {
        "log_dir": "alpha_factory/data/logs",
        "shutdown_drain_timeout_s": 300,
        "watchdog_extra_s": 60,
        "telegram_enabled": False,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override onto a copy of base."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _strip_inline_comment(raw_value: str) -> str:
    """Strip unquoted inline comments and surrounding quotes from a dotenv value."""
    raw_value = raw_value.strip()
    if not raw_value:
        return ""
    if raw_value[0] in ("'", '"'):
        quote = raw_value[0]
        end_idx = raw_value.find(quote, 1)
        if end_idx != -1:
            return raw_value[1:end_idx]
        return raw_value[1:]
    comment_idx = raw_value.find(" #")
    if comment_idx != -1:
        raw_value = raw_value[:comment_idx].rstrip()
    return raw_value


def load_dotenv_file(path: Optional[str] = None) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into os.environ without external dependencies.

    Existing environment variables take precedence (uses ``os.environ.setdefault``).
    Returns a dictionary of the parsed key-value pairs.
    """
    if path is not None:
        env_path = Path(path).expanduser()
    else:
        cwd_env = Path(".env")
        repo_env = Path(__file__).resolve().parent.parent.parent / ".env"
        env_path = cwd_env if cwd_env.is_file() else repo_env

    if not env_path.is_file():
        return {}

    parsed: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _strip_inline_comment(raw_value)
        os.environ.setdefault(key, value)
        parsed[key] = value
    return parsed


def load_settings(path: Optional[str] = None) -> dict[str, Any]:
    """Load settings YAML and deep-merge it over ``DEFAULT_SETTINGS``."""
    if path is not None:
        cfg_path = Path(path).expanduser()
    else:
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

    if not cfg_path.is_file():
        return copy.deepcopy(DEFAULT_SETTINGS)

    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return copy.deepcopy(DEFAULT_SETTINGS)
    return _deep_merge(DEFAULT_SETTINGS, loaded)


@dataclass
class SimulationSettings:
    """BRAIN simulation settings payload for /simulations requests."""

    instrument_type: str = "EQUITY"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 4
    neutralization: str = "INDUSTRY"
    truncation: float = 0.08
    pasteurization: str = "ON"
    test_period: str = "P2Y0M0D"
    unit_handling: str = "VERIFY"
    nan_handling: str = "OFF"
    max_trade: str = "OFF"
    language: str = "FASTEXPR"
    visualization: bool = False

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize settings to exact WorldQuant BRAIN /simulations JSON keys."""
        return {
            "instrumentType": self.instrument_type,
            "region": self.region,
            "universe": self.universe,
            "delay": self.delay,
            "decay": self.decay,
            "neutralization": self.neutralization,
            "truncation": self.truncation,
            "pasteurization": self.pasteurization,
            "testPeriod": self.test_period,
            "unitHandling": self.unit_handling,
            "nanHandling": self.nan_handling,
            "maxTrade": self.max_trade,
            "language": self.language,
            "visualization": self.visualization,
        }

    def to_json(self) -> str:
        """Return canonical JSON representation of API simulation settings."""
        return json.dumps(self.to_api_dict(), sort_keys=True)


@dataclass
class CandidateItem:
    """Queued alpha candidate awaiting or undergoing BRAIN simulation."""

    id: Optional[int] = None
    config_hash: str = ""
    ast_fingerprint: str = ""
    expression: str = ""
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    settings_json: str = "{}"
    source: str = "template"
    dataset_id: str = ""
    priority: float = 0.0
    status: str = "PENDING"
    progress_url: Optional[str] = None
    batch_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class SimulationRecord:
    """Completed or failed BRAIN simulation result persisted in SQLite."""

    alpha_id: str = ""
    config_hash: str = ""
    ast_fingerprint: str = ""
    expression: str = ""
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    settings_json: str = "{}"
    status: str = "COMPLETE"
    sharpe: Optional[float] = None
    fitness: Optional[float] = None
    turnover: Optional[float] = None
    returns: Optional[float] = None
    drawdown: Optional[float] = None
    margin: Optional[float] = None
    long_count: Optional[int] = None
    short_count: Optional[int] = None
    checks_json: str = "[]"
    fail_count: int = 0
    warn_count: int = 0
    gate1_pass: Optional[bool] = None
    gate2_pass: Optional[bool] = None
    self_corr_max: Optional[float] = None
    prod_corr_max: Optional[float] = None
    power_pool_corr_max: Optional[float] = None
    pyramid_matches_json: str = "[]"
    error_message: Optional[str] = None
    simulated_at: Optional[str] = None


@dataclass
class CheckItem:
    """Single BRAIN alpha check entry."""

    name: str
    result: str
    limit: Optional[float] = None
    value: Optional[float] = None


@dataclass
class QualificationResult:
    """Aggregated Gate 1 and Gate 2 qualification outcome for an alpha."""

    alpha_id: str
    passed: bool = False
    needs_review: bool = False
    fail_reasons: list[str] = field(default_factory=list)
    warn_reasons: list[str] = field(default_factory=list)
    checks: list[CheckItem] = field(default_factory=list)
    self_corr_max: Optional[float] = None
    prod_corr_max: Optional[float] = None
    prod_risk_tier: str = "UNKNOWN"
    power_pool_corr_max: Optional[float] = None
    pyramid_matches: list[str] = field(default_factory=list)
    theme_matches: list[str] = field(default_factory=list)
    competition_matches: list[str] = field(default_factory=list)


@dataclass
class VaultEntry:
    """Qualified alpha tracked in the deduplicated submission vault."""

    alpha_id: str
    region: str = "USA"
    delay: int = 1
    expression: str = ""
    composite_score: float = 0.0
    vault_status: str = "ACTIVE"
    duplicate_of: Optional[str] = None
    max_local_corr: Optional[float] = None
    max_local_corr_peer: Optional[str] = None
    sharpe: Optional[float] = None
    fitness: Optional[float] = None
    turnover: Optional[float] = None
    prod_corr_max: Optional[float] = None
    updated_at: Optional[str] = None


@dataclass
class DailyQuotaStatus:
    """New York-date simulation quota accounting snapshot."""

    date_ny: str
    submitted_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    limit: int = 5000
    remaining: int = 5000
    warning_triggered: bool = False
    can_submit: bool = True


@dataclass
class DataFieldRecord:
    """Metadata for a BRAIN data field in a specific region/universe/delay."""

    field_id: str
    dataset_id: str
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    field_type: str = "MATRIX"
    category: str = ""
    subcategory: str = ""
    coverage: float = 0.0
    user_count: int = 0
    alpha_count: int = 0
    description: str = ""
    is_event_input: bool = False


@dataclass
class OperatorRecord:
    """Metadata for a FastExpr operator supported by BRAIN."""

    name: str
    category: str = ""
    scope_json: str = "[]"
    definition: str = ""
    description: str = ""
    num_inputs: int = 1
    min_inputs: int = 1
    uses_lookback: bool = False
