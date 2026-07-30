"""
DynamicXLAMOrchestrator with external tools (db_tool, search, llm_api, simulation).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ml.core.external_tools import (
    db_reference_default_limit,
    db_reference_default_schema,
    db_reference_default_sqlite_table,
    db_reference_default_table_name,
    db_reference_series_limit,
    execute_db_tool,
    execute_digital_twin_tool,
    execute_llm_api_tool,
    execute_search_tool,
    execute_simulation_tool,
    is_target_like_column,
    is_external_output_success,
    is_structured_db_query,
)
from ml.core.feature_extraction import extract_nl_features, route_models
from ml.core.forecast_horizon import (
    ForecastHorizon,
    HorizonModelScore,
    describe_runtime_horizon,
    extract_forecast_horizon,
    forecast_selection_target_for_query,
    rank_model_scores_for_horizon,
    strip_horizon_routing_features,
)
from ml.core.feature_store import (
    resolve_component_prediction_features,
    resolve_prediction_features,
)
from ml.core.hybrid_stockout import (
    build_hybrid_stockout_prediction,
    find_hybrid_stockout_regression_runtime,
    hybrid_stockout_enabled,
    stockout_orchestrator_routing_config,
)
from ml.core.model_metadata import infer_model_task, runtime_prediction_defaults
from ml.core.model_loading import (
    run_prediction_batch_with_fallback,
    run_prediction_with_fallback,
)
from ml.core.registry import ModelRegistry, ModelRuntime
from ml.core.simulation_tool import (
    DEFAULT_COMPARE_POLICIES,
    DEFAULT_SIMULATION_POLICY,
    is_simulation_query,
    simulation_default_base_url,
    simulation_default_compare_on_recommend,
    simulation_default_timeout_s,
)
from ml.core.utils import as_float, as_int, safe_slug

logger = logging.getLogger(__name__)

# ── Project paths for local-first defaults ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # backendMulti/
BACKEND_ML_DIR = PROJECT_ROOT / "ml"
WORKSPACE_ROOT = PROJECT_ROOT.parent
ML_BACKEND_DIR = WORKSPACE_ROOT / "ml-backend"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    rows: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        rows.append(path)
    return rows


def _clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _first_existing_path(candidates: list[Path]) -> Path:
    resolved = _dedupe_paths(
        [candidate.expanduser().resolve() for candidate in candidates]
    )
    if not resolved:
        raise RuntimeError("Expected at least one fallback path candidate.")
    for candidate in resolved:
        if candidate.exists():
            return candidate
    return resolved[0]


def _resolve_env_or_path(
    *,
    env_names: tuple[str, ...],
    default_path: Path,
) -> Path:
    for env_name in env_names:
        raw = os.getenv(env_name, "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    return default_path


DEFAULT_XLAM_MODEL_DIR = _first_existing_path(
    [
        BACKEND_ML_DIR / "models" / "gguf",
        ML_BACKEND_DIR / "models" / "gguf",
    ]
)
DEFAULT_EXTERNAL_TOOLS_CONFIG_PATH = _first_existing_path(
    [
        BACKEND_ML_DIR / "config" / "config.json",
        ML_BACKEND_DIR / "notebooks" / "ml_models" / "config.json",
    ]
)
DEFAULT_DB_SUPPLY_CSV_PATH = _first_existing_path(
    [
        ML_BACKEND_DIR / "datasets_featues_labels_seperated" / "hospital_supply_features.parquet",
        BACKEND_ML_DIR / "datasets" / "synthetic_bloodbank_daily.csv",
        ML_BACKEND_DIR / "datasets" / "synthetic_bloodbank_daily.csv",
    ]
)
DEFAULT_DB_DONOR_CSV_PATH = _first_existing_path(
    [
        BACKEND_ML_DIR
        / "datasets"
        / "blood_registry_sythentic"
        / "data"
        / "blood_donation_registry_ml_ready.csv",
        ML_BACKEND_DIR
        / "datasets"
        / "blood_registry_sythentic"
        / "data"
        / "blood_donation_registry_ml_ready.csv",
        ML_BACKEND_DIR / "datasets" / "synthetic_donor_candidates_clustered.csv",
        ML_BACKEND_DIR / "datasets" / "donor_snapshots_with_targets.csv",
    ]
)


@dataclass
class XLAMVariant:
    model_id: str
    name: str
    description: str
    repo_id: str
    filename: str
    size_mb: float | None
    size_bytes: int | None
    min_ram_gb: float
    min_vram_mb: int
    n_ctx: int
    n_batch: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    reasoning: str = ""


@dataclass
class ExecutionPlan:
    steps: list[ToolCall]
    reasoning: str
    is_multi_step: bool


@dataclass
class ExecutionResult:
    step: int
    tool_name: str
    inputs: dict[str, Any]
    output: dict[str, Any]
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class ExternalToolDefinition:
    id: str
    name: str
    adapter: str
    enabled: bool
    config: dict[str, Any]


class PlanPromptTooLargeError(RuntimeError):
    """Raised when no compaction can fit the planning prompt into n_ctx.

    Signals the caller to route to a clean fallback without invoking the LLM,
    so llama.cpp never raises a raw "exceed context window" error.
    """


LOCAL_ORCHESTRATOR_REPO_ID = "local"
MIN_LOCAL_GGUF_SIZE_BYTES = 32 * 1024 * 1024
CAPABILITY_CACHE_VERSION = 1
CAPABILITY_CACHE_FILENAME = ".orchestrator_capabilities.json"
CAPABILITY_SIDECAR_FILENAMES = (
    "{filename}.capabilities.json",
    "{stem}.capabilities.json",
)
HANDSHAKE_JSON_PROMPT = 'Return only this exact JSON object and nothing else: {"ok":true}'
HANDSHAKE_MAX_TOKENS = 48
# Context window for the planner LLM. The xLAM-7b-fc-r model (Mistral-7B base)
# natively supports far more than this; 4096 leaves comfortable room for the
# planning prompt (template + tool descriptions + workflow examples). But the KV
# cache for that window costs memory ON TOP OF the model weights, so the safe
# window depends on how much RAM is left after the model loads: a ~4GB quant on
# an 8GB Mac must stay at 2048 (4096 exhausts unified memory and llama.cpp fails
# to decode), while a ~2.6GB quant on the same machine comfortably reaches 4096.
# `_auto_planner_n_ctx` computes the largest window that fits; these bound it.
# The budgeted prompt builder then trims the tool list to whatever window is
# active. All overridable per-machine via PIOS_XLAM_N_CTX.
DEFAULT_PLANNER_N_CTX = 4096
LOW_RAM_PLANNER_N_CTX = 2048
# Supported context windows, largest first — the auto-sizer picks the biggest one
# whose KV cache fits in the RAM left after the model weights.
PLANNER_N_CTX_CANDIDATES = (4096, 3072, 2048, 1024)
# KV-cache cost per token for xLAM-7b-fc-r (Mistral-7B, GQA 8 KV heads × 128 dim ×
# 32 layers × 2 tensors × 2 bytes f16 ≈ 128 KB/token).
PLANNER_KV_MB_PER_TOKEN = 0.125
# RAM that must stay free for the OS, the Python/Django backend, and llama.cpp
# compute buffers — i.e. everything that is NOT model weights or KV cache.
# Calibrated against an 8GB Apple Silicon host: a ~4GB quant keeps 2048 (works)
# but is denied 4096 (observed llama_decode failure), while a ~2.6GB quant reaches
# 4096. Overridable via PIOS_XLAM_RESERVED_MB.
PLANNER_RESERVED_NON_KV_MB = 3900
ARGUMENT_REFERENCE_PATTERN = re.compile(
    r"\$(?:request|last|steps)(?:\.[A-Za-z0-9_]+)*"
)


def _model_filename_key(value: str) -> str:
    stem = Path(str(value)).stem.lower()
    return re.sub(r"[^a-z0-9]+", "", stem)


def _format_model_token(token: str) -> str:
    lowered = token.lower()
    if re.fullmatch(r"q\d+(?:_[a-z0-9]+)*", lowered):
        return lowered.upper()
    if re.fullmatch(r"\d+b", lowered):
        return f"{lowered[:-1]}B"
    if token != token.lower() and token != token.upper():
        return token
    if lowered in {"fc", "gguf", "llm"} or (token.isalpha() and len(token) <= 3):
        return lowered.upper()
    return token[:1].upper() + token[1:]


def _humanize_model_label(value: str) -> str:
    text = str(value or "").strip()
    stem = Path(text).stem if text else ""
    tokens = [token for token in re.split(r"[.\-_\s]+", stem) if token]
    if not tokens:
        return stem or text
    return " ".join(_format_model_token(token) for token in tokens)


def _humanize_model_filename(filename: str) -> str:
    return _humanize_model_label(Path(filename).stem)


def _extract_quant_label(value: str) -> str | None:
    stem = Path(str(value)).stem
    match = re.search(r"(q\d+(?:_[a-z0-9]+)*)", stem, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def _display_name_from_metadata(metadata: dict[str, Any], model_path: Path) -> str:
    general_name = str(metadata.get("general.name") or "").strip()
    quant_label = _extract_quant_label(model_path.name)
    if general_name:
        base = _humanize_model_label(general_name)
        if quant_label and quant_label.lower() not in base.lower():
            return f"{base} {quant_label}"
        return base
    return _humanize_model_filename(model_path.name)


def _extract_context_length(metadata: dict[str, Any]) -> int | None:
    preferred_keys = (
        "context_length",
        "llama.context_length",
    )
    for key in preferred_keys:
        if key in metadata:
            return as_int(metadata.get(key), 0) or None
    for key, value in metadata.items():
        if str(key).endswith(".context_length"):
            return as_int(value, 0) or None
    return None


def _normalize_prompt_mode(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"chat", "completion"}:
        return normalized
    return None


def _merge_capabilities(
    *payloads: dict[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if value is None:
                continue
            merged[key] = value
    return merged


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _estimate_min_ram_gb(size_bytes: int) -> float:
    size_gb = max(0.0, float(size_bytes) / (1024**3))
    return round(max(2.0, size_gb * 1.35), 1)


def _default_xlam_catalog_payload(
    *,
    repo_id: str,
    custom_filename: str | None = None,
) -> dict[str, Any]:
    if custom_filename:
        filename = str(custom_filename).strip()
        return {
            "models": [
                {
                    "id": safe_slug(filename),
                    "name": _humanize_model_filename(filename),
                    "description": "Configured from PIOS_XLAM_FILENAME.",
                    "repo_id": repo_id,
                    "filename": filename,
                    "size_mb": None,
                    "min_ram_gb": 0.0,
                    "min_vram_mb": 0,
                    "n_ctx": 4096,
                    "n_batch": 128,
                    "source": "configured",
                    "auto_detected": False,
                    "local_only": False,
                }
            ],
            "selected_model_id": safe_slug(filename),
        }

    def _row(
        model_id, name, filename, size_mb, min_ram_gb, min_vram_mb, n_ctx, n_batch
    ):
        return {
            "id": model_id,
            "name": name,
            "description": f"{name} quantization for xLAM orchestration.",
            "repo_id": repo_id,
            "filename": filename,
            "size_mb": size_mb,
            "size_bytes": int(size_mb * 1024 * 1024),
            "min_ram_gb": min_ram_gb,
            "min_vram_mb": min_vram_mb,
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "source": "catalog",
            "auto_detected": False,
            "local_only": False,
            "prompt_mode": "completion",
        }

    return {
        "models": [
            _row(
                "xlam_7b_q6_k",
                "xLAM 7B Q6_K",
                "xLAM-7b-fc-r-Q6_K.gguf",
                6200.0,
                14.0,
                11000,
                4096,
                512,
            ),
            _row(
                "xlam_7b_q5_k_m",
                "xLAM 7B Q5_K_M",
                "xLAM-7b-fc-r-Q5_K_M.gguf",
                5100.0,
                10.0,
                8000,
                3072,
                384,
            ),
            _row(
                "xlam_7b_q4_k_m",
                "xLAM 7B Q4_K_M",
                "xLAM-7b-fc-r-Q4_K_M.gguf",
                4200.0,
                8.0,
                5000,
                2048,
                256,
            ),
            _row(
                "xlam_7b_q3_k_m",
                "xLAM 7B Q3_K_M",
                "xLAM-7b-fc-r-Q3_K_M.gguf",
                3300.0,
                6.0,
                2500,
                4096,
                128,
            ),
            _row(
                "xlam_7b_q2_k",
                "xLAM 7B Q2_K",
                "xLAM-7b-fc-r-Q2_K.gguf",
                2600.0,
                4.0,
                0,
                1024,
                64,
            ),
        ],
        "selected_model_id": None,
    }


class DynamicXLAMOrchestrator:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.repo_id = os.getenv("PIOS_XLAM_REPO_ID", "bartowski/xLAM-7b-fc-r-GGUF")
        self.preferred_model_id = os.getenv("PIOS_XLAM_MODEL_ID", "").strip() or None
        self.local_model_path = os.getenv("PIOS_XLAM_MODEL_PATH")
        self.model_dir = (
            Path(os.getenv("PIOS_XLAM_MODEL_DIR", str(DEFAULT_XLAM_MODEL_DIR)))
            .expanduser()
            .resolve()
        )
        self.disable_llm = os.getenv("PIOS_XLAM_DISABLE_LLM", "0") == "1"
        self.allow_fallback = os.getenv("PIOS_XLAM_ALLOW_FALLBACK", "1") == "1"
        self.init_mode = os.getenv("PIOS_XLAM_INIT_MODE", "background").strip().lower()
        if self.init_mode not in {"background", "blocking"}:
            self.init_mode = "background"
        self.disable_progress = os.getenv("PIOS_XLAM_DISABLE_PROGRESS", "0") == "1"
        self.planner_max_tools = max(1, int(os.getenv("PIOS_XLAM_PLANNER_TOOLS", "4")))
        self.planner_max_features = max(
            1, int(os.getenv("PIOS_XLAM_PLANNER_FEATURES", "8"))
        )
        self.model_scope_min_score = float(
            os.getenv("PIOS_ORCH_MODEL_SCOPE_MIN_SCORE", "2.0")
        )
        self.model_scope_min_input_overlap = max(
            1, int(os.getenv("PIOS_ORCH_MODEL_SCOPE_MIN_INPUT_OVERLAP", "2"))
        )
        stockout_routing_defaults = stockout_orchestrator_routing_config()
        self.forecast_selection_target = (
            os.getenv(
                "PIOS_FORECAST_SELECTION_TARGET",
                str(stockout_routing_defaults.get("forecast_selection_default_target") or ""),
            )
            .strip()
            .lower()
        )
        self.forecast_selection_metric = (
            os.getenv(
                "PIOS_FORECAST_SELECTION_METRIC",
                str(stockout_routing_defaults.get("forecast_selection_default_metric") or ""),
            )
            .strip()
            .lower()
        )
        self._forecast_metrics_cache: dict[str, dict[str, Any] | None] = {}

        # ── External tools config ────────────────────────
        self.external_tools_config_path = (
            Path(
                os.getenv(
                    "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG",
                    os.getenv(
                        "PIOS_MODEL_CONFIG",
                        str(DEFAULT_EXTERNAL_TOOLS_CONFIG_PATH),
                    ),
                )
            )
            .expanduser()
            .resolve()
        )
        self._apply_external_tools_runtime_state(
            self._build_external_tools_runtime_state()
        )
        self.simulation_base_url = (
            os.getenv("PIOS_ORCH_SIMULATION_BASE_URL", simulation_default_base_url())
            .strip()
            .rstrip("/")
        )
        self.simulation_timeout_s = max(
            1.0,
            as_float(
                os.getenv("PIOS_ORCH_SIMULATION_TIMEOUT_S"),
                simulation_default_timeout_s(),
            ),
        )
        self.simulation_default_policy = safe_slug(
            os.getenv("PIOS_ORCH_SIMULATION_DEFAULT_POLICY", DEFAULT_SIMULATION_POLICY)
        )
        raw_compare_on_recommend = os.getenv("PIOS_ORCH_SIMULATION_COMPARE_ON_RECOMMEND")
        self.simulation_compare_on_recommend = (
            simulation_default_compare_on_recommend()
            if raw_compare_on_recommend is None
            else raw_compare_on_recommend.strip().lower() in {"1", "true", "yes", "on"}
        )
        self._simulation_pre_routing_enabled = (
            os.getenv("PIOS_ORCH_SIMULATION_PRE_ROUTING", "1") == "1"
        )
        raw_sim_compare = os.getenv("PIOS_ORCH_SIMULATION_COMPARE_POLICIES")
        self.simulation_default_compare_policies = (
            [
                safe_slug(name)
                for name in raw_sim_compare.split(",")
                if safe_slug(name)
            ]
            if raw_sim_compare is not None
            else list(DEFAULT_COMPARE_POLICIES)
        )
        self.db_supply_csv_path = _resolve_env_or_path(
            env_names=("PIOS_ORCH_DB_SUPPLY_CSV_PATH", "PIOS_ORCH_DB_BOOTSTRAP_CSV"),
            default_path=DEFAULT_DB_SUPPLY_CSV_PATH,
        )
        self.db_donor_csv_path = _resolve_env_or_path(
            env_names=("PIOS_ORCH_DB_DONOR_CSV_PATH", "PIOS_ORCH_DB_BOOTSTRAP_CSV"),
            default_path=DEFAULT_DB_DONOR_CSV_PATH,
        )
        self.db_sqlite_path = os.getenv("PIOS_ORCH_DB_SQLITE_PATH", "").strip()
        self.db_sqlite_table = safe_slug(
            os.getenv("PIOS_ORCH_DB_SQLITE_TABLE", db_reference_default_sqlite_table())
        )
        self.db_series_limit = max(
            1, as_int(os.getenv("PIOS_ORCH_DB_SERIES_LIMIT"), db_reference_series_limit())
        )
        self.db_schema = safe_slug(
            os.getenv("PIOS_ORCH_DB_SCHEMA", db_reference_default_schema())
        )
        self.db_donors_table = safe_slug(
            os.getenv(
                "PIOS_ORCH_DB_DONORS_TABLE",
                db_reference_default_table_name("donors"),
            )
        )
        self.db_hospitals_table = safe_slug(
            os.getenv(
                "PIOS_ORCH_DB_HOSPITALS_TABLE",
                db_reference_default_table_name("hospitals"),
            )
        )
        self.db_result_limit = max(
            1, as_int(os.getenv("PIOS_ORCH_DB_RESULT_LIMIT"), db_reference_default_limit())
        )
        self.search_api_url = os.getenv("PIOS_ORCH_SEARCH_API_URL", "").strip()
        self.search_timeout_s = max(
            1.0, as_float(os.getenv("PIOS_ORCH_SEARCH_TIMEOUT_S"), 10.0)
        )
        self.search_result_limit = max(
            1, as_int(os.getenv("PIOS_ORCH_SEARCH_RESULT_LIMIT"), 5)
        )
        self.llm_api_url = os.getenv("PIOS_ORCH_LLM_API_URL", "").strip()
        self.llm_api_key = os.getenv("PIOS_ORCH_LLM_API_KEY", "").strip()
        self.llm_api_model = os.getenv("PIOS_ORCH_LLM_API_MODEL", "").strip()
        self.llm_api_timeout_s = max(
            1.0, as_float(os.getenv("PIOS_ORCH_LLM_API_TIMEOUT_S"), 20.0)
        )

        # ── LLM state ───────────────────────────────────
        self._llm: Any = None
        self._grammar: Any = None
        self._llm_n_ctx: int | None = None
        self._llm_attempted = False
        self._llm_error: str | None = None
        self._selected_variant: XLAMVariant | None = None
        self._selected_model_path: str | None = None
        self._active_model_id: str | None = None
        self._active_capabilities: dict[str, Any] = {}
        self._init_thread: threading.Thread | None = None
        self._init_in_progress = False
        self._init_started_at: float | None = None
        self._pending_warmup = False
        self._lock = threading.Lock()
        self._resource_profile = self._detect_resources()
        self._catalog_signature: str | None = None
        self._catalog_generation = 0
        self._available_variants: list[XLAMVariant] = []
        self._variants_by_id: dict[str, XLAMVariant] = {}
        self._capability_cache_path = (self.model_dir / CAPABILITY_CACHE_FILENAME).resolve()
        self._capability_cache = self._load_capability_cache()

        self.update_model_catalog(
            _default_xlam_catalog_payload(
                repo_id=self.repo_id,
                custom_filename=os.getenv("PIOS_XLAM_FILENAME"),
            ).get("models", []),
            selected_model_id=self.preferred_model_id,
            force=True,
            warmup=False,
            wait=False,
        )

        self._plan_schema = {
            "type": "object",
            "properties": {
                "is_multi_step": {"type": "boolean"},
                "reasoning": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "arguments": {"type": "object"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["tool", "arguments", "reasoning"],
                    },
                },
            },
            "required": ["is_multi_step", "reasoning", "steps"],
        }

    # ══════════════════════════════════════════════════════
    #  HARDWARE DETECTION
    # ══════════════════════════════════════════════════════

    def _env_int_override(self, name: str) -> int | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _total_ram_mb(self) -> float:
        """Best-effort total physical RAM in MB (0.0 if it cannot be determined)."""
        if platform.system() == "Darwin":
            try:
                mem_bytes = int(
                    subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
                )
                return mem_bytes / (1024**2)
            except Exception:
                pass
        try:
            meminfo = Path("/proc/meminfo")
            if meminfo.exists():
                for line in meminfo.read_text(encoding="utf-8").splitlines():
                    if line.startswith("MemTotal:"):
                        return float(line.split()[1]) / 1024  # kB -> MB
        except (OSError, ValueError):
            pass
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return (pages * page_size) / (1024**2)
        except (AttributeError, ValueError, OSError):
            pass
        return 0.0

    def _auto_planner_n_ctx(self, total_ram_mb: float, model_mb: float | None) -> int:
        """Largest supported context window whose KV cache fits in the RAM left
        after the model weights and fixed runtime overhead.

        This is what lets a small quant (e.g. ~2.6GB) use 4096 on an 8GB host
        while a large quant (e.g. ~4GB) is held to 2048 on the same host. Falls
        back to the conservative low-RAM window when sizes are unknown.
        """
        if not model_mb or model_mb <= 0 or total_ram_mb <= 0:
            return LOW_RAM_PLANNER_N_CTX
        reserved = self._env_int_override("PIOS_XLAM_RESERVED_MB")
        reserved_mb = float(reserved) if reserved is not None else PLANNER_RESERVED_NON_KV_MB
        kv_budget_mb = total_ram_mb - model_mb - reserved_mb
        smallest = min(PLANNER_N_CTX_CANDIDATES)
        if kv_budget_mb <= 0:
            return smallest
        affordable_tokens = kv_budget_mb / PLANNER_KV_MB_PER_TOKEN
        for candidate in PLANNER_N_CTX_CANDIDATES:  # largest first
            if affordable_tokens >= candidate:
                return candidate
        return smallest

    def _get_hardware_config(self, model_mb: float | None = None) -> dict[str, Any]:
        system = platform.system()
        machine = platform.machine()
        cpu_cores = os.cpu_count() or 4
        env_threads = self._env_int_override("PIOS_XLAM_CPU_THREADS")
        env_ctx = self._env_int_override("PIOS_XLAM_N_CTX")
        env_batch = self._env_int_override("PIOS_XLAM_N_BATCH")
        env_gpu_layers = self._env_int_override("PIOS_XLAM_N_GPU_LAYERS")
        total_ram_mb = self._total_ram_mb()
        # RAM-aware default; the explicit env override always wins.
        auto_ctx = self._auto_planner_n_ctx(total_ram_mb, model_mb)

        if system == "Darwin" and machine == "arm64":
            total_ram_gb = (total_ram_mb / 1024) if total_ram_mb > 0 else 8.0
            logger.info(
                "Apple Silicon detected (RAM: ~%.1f GB, model: ~%s MB, ctx: %s)",
                total_ram_gb,
                round(model_mb) if model_mb else "?",
                env_ctx if env_ctx is not None else auto_ctx,
            )
            return {
                "type": "Metal (Low RAM)" if total_ram_gb < 12 else "Metal (High RAM)",
                "n_gpu_layers": env_gpu_layers if env_gpu_layers is not None else -1,
                "n_threads": max(
                    1, env_threads if env_threads is not None else (4 if total_ram_gb < 12 else 6)
                ),
                "n_ctx": max(512, env_ctx if env_ctx is not None else auto_ctx),
                "n_batch": max(32, env_batch if env_batch is not None else 512),
                "use_mmap": True,
                "use_mlock": False,
            }

        try:
            subprocess.check_output(["nvidia-smi"])
            logger.info("NVIDIA GPU detected.")
            return {
                "type": "CUDA",
                "n_gpu_layers": env_gpu_layers if env_gpu_layers is not None else -1,
                "n_threads": max(1, env_threads if env_threads is not None else 8),
                # KV cache lives in VRAM on CUDA, not system RAM, so use the full
                # default window here rather than the RAM-budgeted one.
                "n_ctx": max(512, env_ctx if env_ctx is not None else DEFAULT_PLANNER_N_CTX),
                "n_batch": max(32, env_batch if env_batch is not None else 1024),
                "use_mmap": True,
                "use_mlock": False,
            }
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        if system == "Linux" and machine in {"arm64", "aarch64"} and Path(
            "/.dockerenv"
        ).exists():
            logger.info(
                "[xLAM] Apple Silicon GPU passthrough is not available inside "
                "Docker Desktop Linux containers. Run the backend natively on "
                "macOS to use Metal; Docker on this machine will use CPU."
            )
        logger.info(
            "No GPU detected. Using CPU mode (RAM: ~%s MB, model: ~%s MB, ctx: %s).",
            round(total_ram_mb) if total_ram_mb else "?",
            round(model_mb) if model_mb else "?",
            env_ctx if env_ctx is not None else auto_ctx,
        )
        return {
            "type": "CPU",
            "n_gpu_layers": env_gpu_layers if env_gpu_layers is not None else 0,
            "n_threads": max(1, env_threads if env_threads is not None else cpu_cores),
            "n_ctx": max(512, env_ctx if env_ctx is not None else auto_ctx),
            "n_batch": max(32, env_batch if env_batch is not None else 256),
            "use_mmap": True,
            "use_mlock": False,
        }

    def _available_ram_gb(self) -> float:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return (pages * page_size) / (1024**3)
        except (AttributeError, ValueError, OSError):
            pass
        try:
            meminfo = Path("/proc/meminfo")
            if meminfo.exists():
                for line in meminfo.read_text(encoding="utf-8").splitlines():
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return float(parts[1]) / (1024**2)
        except (OSError, ValueError):
            pass
        return 4.0

    def _free_vram_mb(self) -> int | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            values = [int(float(row)) for row in rows]
            return max(values) if values else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def _detect_resources(self) -> dict[str, Any]:
        ram = round(self._available_ram_gb(), 2)
        vram = self._free_vram_mb()
        return {
            "ram_available_gb": ram,
            "vram_free_mb": vram,
            "gpu_available": vram is not None and vram > 0,
        }

    # ══════════════════════════════════════════════════════
    #  CATALOG MANAGEMENT
    # ══════════════════════════════════════════════════════

    def default_catalog_payload(self) -> dict[str, Any]:
        return _default_xlam_catalog_payload(
            repo_id=self.repo_id,
            custom_filename=os.getenv("PIOS_XLAM_FILENAME"),
        )

    def _candidate_model_dirs(self) -> list[Path]:
        candidates = [self.model_dir]
        if self.local_model_path:
            raw = Path(self.local_model_path).expanduser()
            if raw.suffix.lower() == ".gguf":
                resolved = (
                    raw.resolve()
                    if raw.is_absolute()
                    else (self.model_dir / raw).resolve()
                )
                candidates.append(resolved.parent)
            else:
                candidates.append(
                    raw.resolve()
                    if raw.is_absolute()
                    else (self.model_dir / raw).resolve()
                )
        return _dedupe_paths(candidates)

    def _iter_local_model_files(self) -> list[Path]:
        rows: list[Path] = []
        for directory in self._candidate_model_dirs():
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.gguf")):
                try:
                    if path.stat().st_size < MIN_LOCAL_GGUF_SIZE_BYTES:
                        continue
                except OSError:
                    continue
                rows.append(path.resolve())
        return _dedupe_paths(rows)

    def _build_discovered_variant(self, model_path: Path) -> XLAMVariant:
        size_bytes = model_path.stat().st_size
        filename = model_path.name
        capabilities = self._resolved_capabilities(model_path)
        prompt_mode = _normalize_prompt_mode(capabilities.get("prompt_mode"))
        context_length = as_int(capabilities.get("context_length"), 4096)
        architecture = str(capabilities.get("architecture") or "").strip() or None
        return XLAMVariant(
            model_id=safe_slug(model_path.stem),
            name=str(capabilities.get("display_name") or _humanize_model_filename(filename)),
            description="Auto-detected local GGUF orchestrator model.",
            repo_id=LOCAL_ORCHESTRATOR_REPO_ID,
            filename=filename,
            size_mb=round(size_bytes / (1024**2), 2),
            size_bytes=size_bytes,
            min_ram_gb=_estimate_min_ram_gb(size_bytes),
            min_vram_mb=0,
            n_ctx=max(256, context_length),
            n_batch=128,
            metadata={
                "source": "local",
                "auto_detected": True,
                "local_only": True,
                "model_family": architecture,
                "prompt_mode": prompt_mode,
            },
        )

    def _merge_discovered_variants(
        self, variants: list[XLAMVariant]
    ) -> list[XLAMVariant]:
        merged = list(variants)
        seen_ids = {variant.model_id for variant in merged}
        seen_keys = {_model_filename_key(variant.filename) for variant in merged}
        for local_path in self._iter_local_model_files():
            discovered_id = safe_slug(local_path.stem)
            filename_key = _model_filename_key(local_path.name)
            if discovered_id in seen_ids or filename_key in seen_keys:
                continue
            merged.append(self._build_discovered_variant(local_path))
            seen_ids.add(discovered_id)
            seen_keys.add(filename_key)
        return merged

    def _find_local_model_path(self, variant: XLAMVariant | None) -> Path | None:
        if variant is None:
            return None
        direct = self._compute_target_path(variant)
        try:
            if direct.exists() and direct.stat().st_size >= MIN_LOCAL_GGUF_SIZE_BYTES:
                return direct
        except OSError:
            pass

        wanted_key = _model_filename_key(variant.filename)
        for local_path in self._iter_local_model_files():
            if _model_filename_key(local_path.name) == wanted_key:
                return local_path
        return None

    def _is_local_only_variant(self, variant: XLAMVariant | None) -> bool:
        if variant is None:
            return False
        if variant.repo_id == LOCAL_ORCHESTRATOR_REPO_ID:
            return True
        return bool((variant.metadata or {}).get("local_only"))

    def _load_capability_cache(self) -> dict[str, Any]:
        path = self._capability_cache_path
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        if as_int(payload.get("version"), 0) != CAPABILITY_CACHE_VERSION:
            return {}
        models = payload.get("models", {})
        return models if isinstance(models, dict) else {}

    def _save_capability_cache(self) -> None:
        path = self._capability_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CAPABILITY_CACHE_VERSION,
            "models": self._capability_cache,
        }
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            tmp_path.write_text(
                json.dumps(payload, sort_keys=True, indent=2, default=str),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        except OSError:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    def _model_signature(self, model_path: Path) -> str | None:
        try:
            stat = model_path.stat()
        except OSError:
            return None
        return f"{int(stat.st_size)}:{int(stat.st_mtime_ns)}"

    def _cached_capabilities(self, model_path: Path) -> dict[str, Any] | None:
        signature = self._model_signature(model_path)
        if not signature:
            return None
        entry = self._capability_cache.get(str(model_path))
        if not isinstance(entry, dict):
            return None
        if str(entry.get("signature") or "") != signature:
            return None
        payload = entry.get("capabilities")
        if not isinstance(payload, dict):
            return None
        return dict(payload)

    def _update_capability_cache(
        self, model_path: Path, capabilities: dict[str, Any]
    ) -> None:
        signature = self._model_signature(model_path)
        if not signature:
            return
        self._capability_cache[str(model_path)] = {
            "signature": signature,
            "updated_at": time.time(),
            "capabilities": capabilities,
        }
        self._save_capability_cache()

    def _capability_sidecar_paths(self, model_path: Path) -> list[Path]:
        return _dedupe_paths(
            [
                (model_path.parent / pattern.format(filename=model_path.name, stem=model_path.stem)).resolve()
                for pattern in CAPABILITY_SIDECAR_FILENAMES
            ]
        )

    def _read_sidecar_capabilities(self, model_path: Path) -> dict[str, Any]:
        for sidecar_path in self._capability_sidecar_paths(model_path):
            if not sidecar_path.exists():
                continue
            try:
                payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            nested = payload.get("capabilities")
            if isinstance(nested, dict):
                return dict(nested)
            return dict(payload)
        return {}

    def _inspect_model_metadata(self, model_path: Path) -> dict[str, Any]:
        try:
            with model_path.open("rb") as handle:
                if handle.read(4) != b"GGUF":
                    return {}
        except OSError:
            return {}

        try:
            import llama_cpp
            from llama_cpp import llama_chat_format
            from llama_cpp._internals import LlamaModel
        except Exception:
            return {}

        try:
            params = llama_cpp.llama_model_default_params()
            params.n_gpu_layers = 0
            params.vocab_only = True
            model = LlamaModel(
                path_model=str(model_path),
                params=params,
                verbose=False,
            )
        except Exception:
            return {}

        try:
            metadata = model.metadata()
        except Exception:
            metadata = {}
        finally:
            try:
                model.close()
            except Exception:
                pass

        if not isinstance(metadata, dict):
            metadata = {}

        has_chat_template = "tokenizer.chat_template" in metadata or any(
            str(key).startswith("tokenizer.chat_template.") for key in metadata
        )
        guessed_chat_format = None
        if has_chat_template:
            try:
                guessed_chat_format = llama_chat_format.guess_chat_format_from_gguf_metadata(metadata)
            except Exception:
                guessed_chat_format = None

        context_length = _extract_context_length(metadata)
        capabilities = {
            "metadata_available": bool(metadata),
            "display_name": _display_name_from_metadata(metadata, model_path),
            "architecture": str(metadata.get("general.architecture") or "").strip()
            or None,
            "has_chat_template": has_chat_template,
            "supports_chat_completion": has_chat_template,
            "prompt_mode": "chat" if has_chat_template else "completion",
            "prompt_mode_source": "metadata" if has_chat_template else "default",
            "chat_format": str(guessed_chat_format or "").strip() or None,
            "context_length": context_length,
            "general_name": str(metadata.get("general.name") or "").strip() or None,
            "metadata_checked": True,
        }
        return capabilities

    def _probe_json_text(self, raw: Any) -> dict[str, Any] | None:
        if isinstance(raw, dict):
            return raw
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _probe_prompt_mode(
        self,
        llm: Any,
        *,
        prompt_mode: str,
        max_tokens: int,
        chat_format: str | None = None,
    ) -> dict[str, Any]:
        prompt = HANDSHAKE_JSON_PROMPT
        try:
            if prompt_mode == "chat" and hasattr(llm, "create_chat_completion"):
                kwargs: dict[str, Any] = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                }
                if chat_format:
                    kwargs["model"] = chat_format
                output = llm.create_chat_completion(**kwargs)
                choices = output.get("choices", []) if isinstance(output, dict) else []
                message = choices[0].get("message", {}) if choices else {}
                raw = self._extract_chat_message_text(message.get("content"))
            else:
                output = llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    echo=False,
                    stop=["\n\n"],
                )
                raw = str(output["choices"][0]["text"]).strip()
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        payload = self._probe_json_text(raw)
        return {
            "success": bool(isinstance(payload, dict) and payload.get("ok") is True),
            "response_preview": str(raw)[:160],
        }

    def _run_capability_handshake(
        self,
        llm: Any,
        model_path: Path,
        capabilities: dict[str, Any],
    ) -> dict[str, Any]:
        preferred_prompt_mode = _normalize_prompt_mode(capabilities.get("prompt_mode"))
        chat_supported = bool(capabilities.get("supports_chat_completion")) and hasattr(
            llm, "create_chat_completion"
        )

        candidates: list[str] = []
        if preferred_prompt_mode:
            candidates.append(preferred_prompt_mode)
        if chat_supported and "chat" not in candidates:
            candidates.append("chat")
        if "completion" not in candidates:
            candidates.append("completion")

        results: dict[str, Any] = {}
        chosen_mode: str | None = None
        for candidate in candidates:
            result = self._probe_prompt_mode(
                llm,
                prompt_mode=candidate,
                max_tokens=HANDSHAKE_MAX_TOKENS,
                chat_format=str(capabilities.get("chat_format") or "").strip() or None,
            )
            results[candidate] = result
            if result.get("success") and chosen_mode is None:
                chosen_mode = candidate
            if (
                result.get("success")
                and preferred_prompt_mode
                and candidate == preferred_prompt_mode
            ):
                chosen_mode = candidate
                break

        payload = {
            "handshake_complete": True,
            "handshake_model_path": str(model_path),
            "handshake_prompt_modes": results,
            "handshake_prompt_mode": chosen_mode,
            "handshake_at": time.time(),
        }
        if chosen_mode:
            payload["probed_prompt_mode"] = chosen_mode
        return payload

    def _variant_capabilities(self, variant: XLAMVariant | None) -> dict[str, Any]:
        if variant is None:
            return {}
        payload = {
            "display_name": variant.name,
            "prompt_mode": _normalize_prompt_mode((variant.metadata or {}).get("prompt_mode")),
            "context_length": variant.n_ctx if variant.n_ctx > 0 else None,
            "model_family": str((variant.metadata or {}).get("model_family") or "").strip()
            or None,
        }
        return {key: value for key, value in payload.items() if value is not None}

    def _resolved_capabilities(
        self,
        model_path: Path | None,
        *,
        variant: XLAMVariant | None = None,
        llm: Any = None,
        require_handshake: bool = False,
    ) -> dict[str, Any]:
        capabilities = self._variant_capabilities(variant)
        if model_path is None or not model_path.exists():
            return _merge_capabilities(
                capabilities,
                {"prompt_mode": capabilities.get("prompt_mode") or "completion"},
            )

        cached = self._cached_capabilities(model_path)
        if cached is None:
            cached = self._inspect_model_metadata(model_path)
            if cached:
                self._update_capability_cache(model_path, cached)
        sidecar = self._read_sidecar_capabilities(model_path)
        base = _merge_capabilities(cached, capabilities)
        merged = _merge_capabilities(base, sidecar)

        explicit_prompt_mode = _normalize_prompt_mode(sidecar.get("prompt_mode"))
        if explicit_prompt_mode is None:
            explicit_prompt_mode = _normalize_prompt_mode(capabilities.get("prompt_mode"))

        if require_handshake and llm is not None:
            handshake = self._run_capability_handshake(llm, model_path, merged)
            base = _merge_capabilities(base, handshake)
            merged = _merge_capabilities(base, sidecar)
            probed_prompt_mode = _normalize_prompt_mode(handshake.get("probed_prompt_mode"))
            prompt_mode = explicit_prompt_mode
            if prompt_mode and handshake.get("handshake_prompt_modes", {}).get(prompt_mode, {}).get("success"):
                merged["prompt_mode"] = prompt_mode
            elif probed_prompt_mode:
                merged["prompt_mode"] = probed_prompt_mode
            elif prompt_mode:
                merged["prompt_mode"] = prompt_mode
            else:
                merged["prompt_mode"] = _normalize_prompt_mode(merged.get("prompt_mode")) or "completion"
            self._update_capability_cache(model_path, base)
        else:
            merged["prompt_mode"] = (
                explicit_prompt_mode
                or _normalize_prompt_mode(merged.get("probed_prompt_mode"))
                or _normalize_prompt_mode(merged.get("prompt_mode"))
                or "completion"
            )

        if variant is None and not merged.get("display_name"):
            merged["display_name"] = _humanize_model_filename(model_path.name)
        return merged

    def _normalize_variant(self, payload: dict[str, Any]) -> XLAMVariant | None:
        raw_id = payload.get("id", payload.get("model_id"))
        model_id = safe_slug(str(raw_id or "").strip())
        filename = str(payload.get("filename", "")).strip()
        if not model_id or not filename:
            return None
        return XLAMVariant(
            model_id=model_id,
            name=str(payload.get("name", model_id)).strip() or model_id,
            description=str(payload.get("description", filename)).strip() or filename,
            repo_id=str(payload.get("repo_id", self.repo_id)).strip() or self.repo_id,
            filename=filename,
            size_mb=as_float(payload.get("size_mb"), 0.0)
            if payload.get("size_mb") is not None
            else None,
            size_bytes=as_int(payload.get("size_bytes"), 0)
            if payload.get("size_bytes") is not None
            else None,
            min_ram_gb=as_float(payload.get("min_ram_gb"), 0.0),
            min_vram_mb=as_int(payload.get("min_vram_mb"), 0),
            n_ctx=max(256, as_int(payload.get("n_ctx"), 4096)),
            n_batch=max(1, as_int(payload.get("n_batch"), 128)),
            metadata={
                k: v
                for k, v in payload.items()
                if k
                not in {
                    "id",
                    "model_id",
                    "name",
                    "description",
                    "repo_id",
                    "filename",
                    "size_mb",
                    "size_bytes",
                    "min_ram_gb",
                    "min_vram_mb",
                    "n_ctx",
                    "n_batch",
                }
            },
        )

    def _catalog_digest(self, variants: list[XLAMVariant], selected: str | None) -> str:
        payload = {
            "selected_model_id": selected,
            "models": [
                {
                    "id": v.model_id,
                    "name": v.name,
                    "filename": v.filename,
                    "size_mb": v.size_mb,
                }
                for v in variants
            ],
        }
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        ).hexdigest()

    def update_model_catalog(
        self, models, *, selected_model_id, force=False, warmup=True, wait=False
    ) -> bool:
        variants = self._merge_discovered_variants(
            [
                v
                for row in models
                if isinstance(row, dict)
                for v in [self._normalize_variant(row)]
                if v
            ]
        )
        if not variants:
            fallback = self.default_catalog_payload()
            variants = self._merge_discovered_variants(
                [
                    v
                    for row in fallback.get("models", [])
                    if isinstance(row, dict)
                    for v in [self._normalize_variant(row)]
                    if v
                ]
            )
        selected = (
            safe_slug(str(selected_model_id).strip()) if selected_model_id else None
        )
        allowed_ids = {v.model_id for v in variants}
        if selected and selected not in allowed_ids:
            selected = None
        digest = self._catalog_digest(variants, selected)
        queue_warmup = False
        with self._lock:
            changed = force or digest != self._catalog_signature
            if not changed:
                return False
            self._available_variants = variants
            self._variants_by_id = {v.model_id: v for v in variants}
            self.preferred_model_id = selected
            self._catalog_signature = digest
            self._catalog_generation += 1
            if not self._init_in_progress:
                self._llm = None
                self._grammar = None
                self._llm_n_ctx = None
                self._llm_attempted = False
                self._llm_error = None
                self._active_model_id = None
                self._active_capabilities = {}

            self._selected_variant = self._pick_variant()
            selected_path = self._find_local_model_path(self._selected_variant)
            if selected_path is None:
                selected_path = self._compute_target_path(self._selected_variant)
            self._selected_model_path = str(selected_path)
            queue_warmup = warmup and self._init_in_progress
            if queue_warmup:
                self._pending_warmup = True
        if warmup and not queue_warmup:
            self._ensure_llm(blocking=wait)
        return True

    def list_available_models(self) -> list[dict[str, Any]]:
        rows = []
        for v in self._variant_candidates():
            local_path = self._find_local_model_path(v)
            resolved_path = local_path or self._compute_target_path(v)
            local_exists = local_path is not None
            local_size_mb = None
            if local_exists and local_path:
                try:
                    local_size_mb = round(local_path.stat().st_size / (1024**2), 2)
                except OSError:
                    pass
            capabilities = self._resolved_capabilities(
                resolved_path if local_exists else None,
                variant=v,
            )
            prompt_mode = _normalize_prompt_mode(capabilities.get("prompt_mode"))
            context_length = as_int(capabilities.get("context_length"), v.n_ctx)
            row = {
                "id": v.model_id,
                "name": str(capabilities.get("display_name") or v.name),
                "description": v.description,
                "repo_id": v.repo_id,
                "filename": v.filename,
                "size_mb": v.size_mb,
                "size_bytes": v.size_bytes,
                "min_ram_gb": v.min_ram_gb,
                "min_vram_mb": v.min_vram_mb,
                "n_ctx": max(256, context_length),
                "n_batch": v.n_batch,
                "local_path": str(resolved_path),
                "local_exists": local_exists,
                "local_size_mb": local_size_mb,
                **v.metadata,
            }
            if prompt_mode:
                row["prompt_mode"] = prompt_mode
            architecture = str(capabilities.get("architecture") or "").strip()
            if architecture:
                row["model_family"] = architecture
            if capabilities.get("supports_chat_completion") is not None:
                row["supports_chat_completion"] = bool(
                    capabilities.get("supports_chat_completion")
                )
            if capabilities.get("has_chat_template") is not None:
                row["has_chat_template"] = bool(capabilities.get("has_chat_template"))
            rows.append(
                row
            )
        return rows

    def _compute_target_path(self, variant: XLAMVariant) -> Path:
        if self.local_model_path:
            raw = Path(self.local_model_path).expanduser()
            if raw.suffix.lower() == ".gguf":
                return (
                    raw.resolve()
                    if raw.is_absolute()
                    else (self.model_dir / raw).resolve()
                )
            target_dir = (
                raw.resolve() if raw.is_absolute() else (self.model_dir / raw).resolve()
            )
            return (target_dir / variant.filename).resolve()
        return (self.model_dir / variant.filename).resolve()

    def _variant_candidates(self) -> list[XLAMVariant]:
        if self._available_variants:
            return list(self._available_variants)
        fallback = self.default_catalog_payload()
        variants = self._merge_discovered_variants(
            v
            for row in fallback.get("models", [])
            if isinstance(row, dict)
            for v in [self._normalize_variant(row)]
            if v
        )
        self._available_variants = variants
        self._variants_by_id = {v.model_id: v for v in variants}
        return variants

    def _pick_variant(self) -> XLAMVariant:
        candidates = self._variant_candidates()
        if not candidates:
            raise RuntimeError("No orchestrator model variants configured.")
        if self.preferred_model_id:
            preferred = self._variants_by_id.get(self.preferred_model_id)
            if preferred:
                return preferred
        for variant in candidates:
            local_path = self._find_local_model_path(variant)
            if local_path is not None:
                logger.info("Found existing local model: %s. Using it.", local_path.name)
                return variant
        local_files = self._iter_local_model_files()
        if local_files:
            chosen = max(local_files, key=lambda p: p.stat().st_size)
            logger.info("Found existing local model: %s. Using it.", chosen.name)
            return self._build_discovered_variant(chosen)
        profile = self._resource_profile
        ram = float(profile["ram_available_gb"])
        vram = profile.get("vram_free_mb")
        has_gpu = bool(profile["gpu_available"])
        for v in candidates:
            ram_ok = ram >= v.min_ram_gb
            vram_ok = (
                ((vram or 0) >= v.min_vram_mb) if has_gpu else (v.min_vram_mb == 0)
            )
            if ram_ok and vram_ok:
                return v
        return candidates[-1]

    # ══════════════════════════════════════════════════════
    #  LLM LOADING
    # ══════════════════════════════════════════════════════

    def _download_to_dir(
        self, *, repo_id, filename, target_dir, hf_hub_download
    ) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        expected = target_dir / filename
        if expected.exists():
            if expected.stat().st_size / (1024 * 1024) < 100:
                logger.warning(
                    "Found corrupted GGUF file (%.2f MB). Deleting: %s",
                    expected.stat().st_size / (1024 * 1024),
                    expected,
                )
                expected.unlink()
            else:
                return expected
        logger.info("Downloading %s to %s.", filename, target_dir)
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_dir),
                resume_download=False,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            if expected.exists() and expected.stat().st_size < 1024:
                expected.unlink()
            raise e
        return Path(downloaded).resolve()

    def _resolve_or_download(self, variant, hf_hub_download) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        local_path = self._find_local_model_path(variant)
        if local_path is not None:
            return local_path
        if self._is_local_only_variant(variant):
            raise FileNotFoundError(
                f"Local-only GGUF not found for orchestrator model '{variant.filename}'."
            )
        if self.local_model_path:
            raw = Path(self.local_model_path).expanduser()
            if raw.suffix.lower() == ".gguf":
                target = (
                    raw.resolve()
                    if raw.is_absolute()
                    else (self.model_dir / raw).resolve()
                )
                if target.exists():
                    return target
                downloaded = self._download_to_dir(
                    repo_id=variant.repo_id,
                    filename=target.name,
                    target_dir=target.parent,
                    hf_hub_download=hf_hub_download,
                )
                return target if target.exists() else downloaded
            target_dir = (
                raw.resolve() if raw.is_absolute() else (self.model_dir / raw).resolve()
            )
            target = target_dir / variant.filename
            if target.exists():
                return target.resolve()
            downloaded = self._download_to_dir(
                repo_id=variant.repo_id,
                filename=variant.filename,
                target_dir=target_dir,
                hf_hub_download=hf_hub_download,
            )
            return target.resolve() if target.exists() else downloaded
        target = (self.model_dir / variant.filename).resolve()
        if target.exists():
            return target
        downloaded = self._download_to_dir(
            repo_id=variant.repo_id,
            filename=variant.filename,
            target_dir=self.model_dir,
            hf_hub_download=hf_hub_download,
        )
        return target if target.exists() else downloaded

    def _load_llm_impl(self) -> None:
        with self._lock:
            start_gen = self._catalog_generation
            start_variant_id = (
                self._selected_variant.model_id if self._selected_variant else None
            )

        if self.disable_llm:
            self._llm_error = "LLM loading disabled."
            logger.info("%s", self._llm_error)
            return
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            self._llm_error = f"Import failed: {exc}"
            logger.warning("%s", self._llm_error)
            return
        variant = self._pick_variant()
        self._selected_variant = variant
        selected_path = self._find_local_model_path(variant)
        if selected_path is None:
            selected_path = self._compute_target_path(variant)
        self._selected_model_path = str(selected_path)
        logger.info("Selected variant: %s (%s)", variant.model_id, variant.filename)
        try:
            model_path = self._resolve_or_download(variant, hf_hub_download)
            self._selected_model_path = str(model_path)
            logger.info("Model ready at: %s", model_path)
        except Exception as exc:
            self._llm_error = f"Download failed: {exc}"
            logger.warning("%s", self._llm_error)
            return
        try:
            from llama_cpp import Llama
        except Exception as exc:
            self._llm_error = f"llama_cpp import failed: {exc}"
            return
        try:
            # Size the context window against this specific quant's footprint:
            # the actual GGUF file size (mmap-resident) is the most accurate, with
            # the catalog estimate as a fallback.
            model_mb: float | None = None
            try:
                model_mb = Path(model_path).stat().st_size / (1024**2)
            except OSError:
                model_mb = float(variant.size_mb) if variant.size_mb else None
            hw = self._get_hardware_config(model_mb=model_mb)
            logger.info("Auto config: %s", hw["type"])
            logger.info(
                "LLM settings: threads=%s, layers=%s, context=%s",
                hw["n_threads"],
                hw["n_gpu_layers"],
                hw["n_ctx"],
            )
            llm = Llama(
                model_path=str(model_path),
                n_ctx=hw["n_ctx"],
                n_threads=hw["n_threads"],
                n_gpu_layers=hw["n_gpu_layers"],
                n_batch=hw["n_batch"],
                use_mlock=hw["use_mlock"],
                use_mmap=hw["use_mmap"],
                verbose=False,
            )
            active_capabilities = self._resolved_capabilities(
                model_path,
                variant=variant,
                llm=llm,
                require_handshake=True,
            )
            # with self._lock:
            #     if start_gen != self._catalog_generation:
            #         self._llm_error = "Stale LLM load discarded."
            #         return
            with self._lock:
                current_variant_id = (
                    self._selected_variant.model_id if self._selected_variant else None
                )
                if start_variant_id != current_variant_id:
                    self._llm_error = "Stale LLM load discarded."
                    return
                self._llm_n_ctx = hw["n_ctx"]
                self._llm = llm
                self._grammar = None
                self._llm_error = None
                self._active_model_id = variant.model_id
                self._active_capabilities = active_capabilities
            logger.info("Model loaded successfully.")
        except Exception as exc:
            self._llm_error = f"Model load failed: {exc}"
            logger.warning("%s", self._llm_error)

    def _background_init_worker(self) -> None:
        should_retry = False
        try:
            self._load_llm_impl()
        finally:
            with self._lock:
                should_retry = bool(self._pending_warmup)
                self._pending_warmup = False
                self._init_in_progress = False
                self._init_thread = None
        if should_retry:
            self._ensure_llm(blocking=False)

    def _ensure_llm(self, blocking=False) -> None:
        if self._llm is not None:
            return
        if blocking:
            while True:
                join_thread = None
                run_inline = False
                with self._lock:
                    if self._llm is not None:
                        return
                    self._llm_attempted = True
                    if self._init_in_progress and self._init_thread:
                        join_thread = self._init_thread
                    elif not self._init_in_progress:
                        self._init_in_progress = True
                        self._init_started_at = time.time()
                        run_inline = True
                if join_thread:
                    join_thread.join()
                    continue
                if run_inline:
                    try:
                        self._load_llm_impl()
                    finally:
                        with self._lock:
                            self._init_in_progress = False
                            self._init_thread = None
                return
        with self._lock:
            if self._llm is not None or self._init_in_progress:
                return
            self._llm_attempted = True
            if not self._selected_variant:
                self._selected_variant = self._pick_variant()
            self._init_in_progress = True
            self._init_started_at = time.time()
            self._init_thread = threading.Thread(
                target=self._background_init_worker, name="xlam-init", daemon=True
            )
            self._init_thread.start()

    def initialize_on_startup(self) -> None:
        logger.info("Startup warmup begin mode=%s", self.init_mode)
        self._ensure_llm(blocking=self.init_mode == "blocking")

    def _current_variant(self) -> XLAMVariant | None:
        if self._active_model_id:
            active = self._variants_by_id.get(self._active_model_id)
            if active is not None:
                return active
        return self._selected_variant

    def _current_capabilities(self) -> dict[str, Any]:
        if self._active_capabilities:
            return dict(self._active_capabilities)
        variant = self._current_variant()
        model_path = (
            Path(self._selected_model_path).expanduser().resolve()
            if self._selected_model_path
            else None
        )
        if model_path is not None and model_path.exists():
            return self._resolved_capabilities(model_path, variant=variant)
        return self._variant_capabilities(variant)

    def _current_prompt_mode(self) -> str:
        capabilities = self._current_capabilities()
        return _normalize_prompt_mode(capabilities.get("prompt_mode")) or "completion"

    def _extract_chat_message_text(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, list):
            parts: list[str] = []
            for item in payload:
                if isinstance(item, dict):
                    text = str(item.get("text", "")).strip()
                    if text:
                        parts.append(text)
                else:
                    text = str(item).strip()
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(payload or "").strip()

    def _generate_llm_text(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.1,
        system_prompt: str | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        if self._llm is None:
            raise RuntimeError("Orchestrator LLM unavailable.")

        if self._current_prompt_mode() == "chat" and hasattr(
            self._llm, "create_chat_completion"
        ):
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            try:
                if event_callback:
                    chunks = self._llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )
                    accumulated = []
                    for chunk in chunks:
                        choices = chunk.get("choices", []) if isinstance(chunk, dict) else []
                        if choices and isinstance(choices[0], dict):
                            delta = choices[0].get("delta", {}) if isinstance(choices[0].get("delta"), dict) else {}
                            text_token = delta.get("content") or ""
                            if text_token:
                                accumulated.append(text_token)
                                current_text = "".join(accumulated)
                                self._emit_run_event(
                                    event_callback,
                                    {
                                        "type": "token",
                                        "phase": "summarizing",
                                        "token": text_token,
                                        "text": current_text,
                                    },
                                )
                    full_text = "".join(accumulated).strip()
                    if full_text:
                        return full_text

                output = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                choices = output.get("choices", []) if isinstance(output, dict) else []
                if choices:
                    message = (
                        choices[0].get("message", {})
                        if isinstance(choices[0], dict)
                        else {}
                    )
                    text = self._extract_chat_message_text(message.get("content"))
                    if text:
                        return text
            except Exception:
                pass

        if event_callback:
            try:
                chunks = self._llm(
                    prompt, max_tokens=max_tokens, temperature=temperature, echo=False, stream=True
                )
                accumulated = []
                for chunk in chunks:
                    choices = chunk.get("choices", []) if isinstance(chunk, dict) else []
                    if choices and isinstance(choices[0], dict):
                        text_token = choices[0].get("text") or ""
                        if text_token:
                            accumulated.append(text_token)
                            current_text = "".join(accumulated)
                            self._emit_run_event(
                                event_callback,
                                {
                                    "type": "token",
                                    "phase": "summarizing",
                                    "token": text_token,
                                    "text": current_text,
                                },
                            )
                full_text = "".join(accumulated).strip()
                if full_text:
                    return full_text
            except Exception:
                pass

        output = self._llm(
            prompt, max_tokens=max_tokens, temperature=temperature, echo=False
        )
        return str(output["choices"][0]["text"]).strip()

    # ══════════════════════════════════════════════════════
    #  SCOPE / ANALYTICS DETECTION
    # ══════════════════════════════════════════════════════

    def _looks_like_analytics_query(self, query: str) -> bool:
        lowered = (query or "").lower()
        tokens = {
            "prevalence",
            "percentage",
            "percent",
            "breakdown",
            "distribution",
            "over time",
            "trend",
            "changed",
            "by age",
            "by race",
            "by ethnicity",
            "population",
            "u.s.",
            "united states",
        }
        return any(t in lowered for t in tokens)

    def _tokenize_query(self, text: str) -> set[str]:
        return {t for t in re.split(r"[^a-z0-9+-]+", (text or "").lower()) if t}

    def _token_overlap(self, query_tokens: set[str], text: str) -> int:
        tokens = self._tokenize_query(text)
        return len(query_tokens.intersection(tokens)) if query_tokens and tokens else 0

    def _model_scope_score(self, query: str, runtime: ModelRuntime) -> float:
        query_tokens = self._tokenize_query(query)
        if not query_tokens:
            return 0.0
        good_examples = [
            str(r.get("user_query", ""))
            for r in runtime.examples
            if isinstance(r, dict) and str(r.get("kind", "")).strip().lower() == "good"
        ]
        bad_examples = [
            str(r.get("user_query", ""))
            for r in runtime.examples
            if isinstance(r, dict) and str(r.get("kind", "")).strip().lower() == "bad"
        ]
        desc_text = " ".join(
            [
                runtime.description or "",
                " ".join(runtime.feature_names or []),
                " ".join(runtime.aliases or []),
            ]
        )
        desc_overlap = self._token_overlap(query_tokens, desc_text)
        max_good = max(
            [self._token_overlap(query_tokens, t) for t in good_examples], default=0
        )
        max_bad = max(
            [self._token_overlap(query_tokens, t) for t in bad_examples], default=0
        )
        return float(desc_overlap + max_good - (max_bad * 1.25))

    def _is_query_in_model_scope(self, query: str, runtime: ModelRuntime) -> bool:
        if self._model_step_has_complete_inputs(
            query,
            ToolCall(name=runtime.model_id, arguments={}),
            runtime,
        ):
            return True
        score = self._model_scope_score(query, runtime)
        input_overlap = self._model_step_input_overlap_count(
            query,
            ToolCall(name=runtime.model_id, arguments={}),
            runtime,
        )
        request_inputs = self._drop_empty_model_inputs(extract_nl_features(query))
        if self._has_explicit_model_inputs(request_inputs) and input_overlap == 0:
            return False
        return (
            input_overlap >= self.model_scope_min_input_overlap
            or (input_overlap > 0 and score > 0.0)
            or score >= self.model_scope_min_score
        )

    def _has_model_input_value(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and math.isnan(value):
            return False
        if isinstance(value, str):
            text = value.strip()
            return bool(text) and text.lower() not in {"null", "none", "nan"}
        return True

    def _drop_empty_model_inputs(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in dict(payload or {}).items()
            if self._has_model_input_value(value)
        }

    def _normalize_model_input_name(self, name: str) -> str:
        normalized = str(name or "").strip().lower()
        if "__" in normalized:
            normalized = normalized.split("__", 1)[1]
        if ":" in normalized:
            normalized = normalized.split(":", 1)[1]
        return normalized

    def _runtime_entity_keys(self, runtime: ModelRuntime) -> list[str]:
        defaults = self._planner_runtime_defaults(runtime)
        feast_config = defaults.get("feast", {})
        if not isinstance(feast_config, dict):
            feast_config = {}

        raw_entity_keys = feast_config.get("entity_keys")
        if isinstance(raw_entity_keys, str) and raw_entity_keys.strip():
            return [raw_entity_keys.strip()]
        if isinstance(raw_entity_keys, list):
            return [
                str(item).strip() for item in raw_entity_keys if str(item).strip()
            ]

        feature_service = str(feast_config.get("feature_service") or "").strip()
        if feature_service.endswith("_service"):
            inferred = feature_service[: -len("_service")].strip()
            if inferred:
                return [inferred]
        return []

    def _model_step_input_overlap_count(
        self,
        query: str,
        step: ToolCall,
        runtime: ModelRuntime,
    ) -> int:
        request_inputs = extract_nl_features(query)
        if isinstance(step.arguments, dict):
            request_inputs.update(self._drop_empty_model_inputs(step.arguments))
        if not request_inputs:
            return 0

        provided_names: set[str] = set()
        for key, value in request_inputs.items():
            if not self._has_model_input_value(value):
                continue
            raw_key = str(key or "").strip().lower()
            if raw_key:
                provided_names.add(raw_key)
                provided_names.add(self._normalize_model_input_name(raw_key))

        expected_names = {
            self._normalize_model_input_name(name)
            for name in [*self._runtime_entity_keys(runtime), *(runtime.feature_names or [])]
            if str(name or "").strip()
        }
        return len(provided_names.intersection(expected_names))

    def _has_explicit_model_inputs(self, payload: dict[str, Any]) -> bool:
        return any(
            self._has_model_input_value(value)
            and self._normalize_model_input_name(key)
            not in {"forecast_horizon", "forecast_timeframe", "timeframe", "days"}
            for key, value in (payload or {}).items()
        )

    def _model_step_can_use_request_inputs(
        self,
        query: str,
        step: ToolCall,
        runtime: ModelRuntime,
        request_inputs: dict[str, Any],
        *,
        horizon_routed: bool,
        forced_model_ids: list[str] | None,
    ) -> bool:
        if forced_model_ids or horizon_routed:
            return True
        if self._model_step_has_complete_inputs(query, step, runtime):
            return True
        if not self._has_explicit_model_inputs(request_inputs):
            return False
        return self._model_step_input_overlap_count(query, step, runtime) > 0

    def _model_step_has_complete_inputs(
        self,
        query: str,
        step: ToolCall,
        runtime: ModelRuntime,
    ) -> bool:
        request_inputs = extract_nl_features(query)
        if isinstance(step.arguments, dict):
            request_inputs.update(self._drop_empty_model_inputs(step.arguments))
        if not request_inputs:
            return False

        entity_keys = self._runtime_entity_keys(runtime)
        if entity_keys and all(
            self._has_model_input_value(request_inputs.get(key)) for key in entity_keys
        ):
            return True

        expected_features = [str(name) for name in runtime.feature_names or [] if name]
        if not expected_features:
            return False

        provided_names: set[str] = set()
        for key, value in request_inputs.items():
            if not self._has_model_input_value(value):
                continue
            raw_key = str(key or "").strip().lower()
            if raw_key:
                provided_names.add(raw_key)
                provided_names.add(self._normalize_model_input_name(raw_key))

        return all(
            str(feature).strip().lower() in provided_names
            or self._normalize_model_input_name(feature) in provided_names
            for feature in expected_features
        )

    # ══════════════════════════════════════════════════════
    #  EXTERNAL TOOL IDENTIFICATION
    # ══════════════════════════════════════════════════════

    def _is_db_tool(self, tool_name: str) -> bool:
        return self._tool_adapter(tool_name) == "db_query"

    def _is_stockout_hybrid_tool(self, tool_name: str) -> bool:
        return self._tool_adapter(tool_name) == "stockout_hybrid"

    def _is_stockout_prediction_runtime(self, runtime: ModelRuntime) -> bool:
        defaults = self._planner_runtime_defaults(runtime)
        output = defaults.get("output")
        output_name = ""
        if isinstance(output, dict):
            output_name = str(output.get("name") or "").strip().lower()
        text = " ".join(
            [
                str(getattr(runtime, "model_id", "") or ""),
                str(getattr(runtime, "description", "") or ""),
                output_name,
            ]
        ).lower()
        routing = stockout_orchestrator_routing_config()
        output_names = {
            value.lower()
            for value in _clean_list(routing.get("prediction_output_names"))
        }
        text_terms = [value.lower() for value in _clean_list(routing.get("prediction_text_terms"))]
        return bool(
            (output_name and output_name in output_names)
            or any(term and term in text for term in text_terms)
        )

    def _hide_stockout_prediction_models(self) -> bool:
        return bool(self.stockout_hybrid_tool_name) and (
            find_hybrid_stockout_regression_runtime(self.registry) is not None
        )

    def _is_stockout_hybrid_query(
        self,
        query: str,
        features: dict[str, Any] | None = None,
    ) -> bool:
        lowered = (query or "").lower()
        request_features = {
            self._normalize_model_input_name(key): value
            for key, value in (features or {}).items()
            if self._has_model_input_value(value)
        }
        routing = stockout_orchestrator_routing_config()
        has_stockout_intent = any(
            token.lower() in lowered
            for token in _clean_list(routing.get("query_intent_terms"))
        )
        has_operational_stock_features = bool(
            {
                self._normalize_model_input_name(value)
                for value in _clean_list(routing.get("operational_feature_names"))
            }.intersection(request_features)
        )
        has_prediction_intent = any(
            token.lower() in lowered
            for token in _clean_list(routing.get("prediction_intent_terms"))
        )
        return has_stockout_intent or (
            has_operational_stock_features and has_prediction_intent
        )

    def _select_stockout_hybrid_runtime(
        self,
        query: str = "",
        features: dict[str, Any] | None = None,
        allowed_models: list[ModelRuntime] | None = None,
        preferred_model_id: str | None = None,
    ) -> ModelRuntime | None:
        preferred = str(preferred_model_id or "").strip()
        if preferred:
            runtime = find_hybrid_stockout_regression_runtime(
                self.registry,
                preferred_model_id=preferred,
            )
            if runtime is not None:
                return runtime

        candidates: list[ModelRuntime] = []
        for runtime in allowed_models or []:
            if not hybrid_stockout_enabled(runtime):
                continue
            resolved = find_hybrid_stockout_regression_runtime(
                self.registry,
                preferred_model_id=getattr(runtime, "model_id", ""),
            )
            if resolved is not None and resolved.model_id == runtime.model_id:
                candidates.append(runtime)

        request_features = {
            self._normalize_model_input_name(key): value
            for key, value in (features or {}).items()
            if self._has_model_input_value(value)
        }
        routing = stockout_orchestrator_routing_config()
        scoped_features = {
            self._normalize_model_input_name(value)
            for value in _clean_list(routing.get("scoped_feature_names"))
        }
        hospital_scoped = bool(
            scoped_features.intersection(request_features)
        ) or any(
            term.lower() in (query or "").lower()
            for term in _clean_list(routing.get("scoped_query_terms"))
        )

        if candidates:
            forecast_horizon = extract_forecast_horizon(query, features or {})
            primary = find_hybrid_stockout_regression_runtime(self.registry)
            indexed = list(enumerate(candidates))
            indexed.sort(
                key=lambda item: self._stockout_hybrid_runtime_rank_key(
                    item[1],
                    item[0],
                    forecast_horizon=forecast_horizon,
                    hospital_scoped=hospital_scoped,
                    primary_model_id=getattr(primary, "model_id", "") if primary else "",
                )
            )
            return indexed[0][1]

        return find_hybrid_stockout_regression_runtime(self.registry)

    def _stockout_compare_requested(self, arguments: dict[str, Any]) -> bool:
        compare_mode = str(arguments.get("compare_mode") or "").strip().lower()
        if compare_mode in {"compare", "rank", "batch"}:
            return True
        hospital_rows = arguments.get("hospital_rows")
        if isinstance(hospital_rows, list) and len(hospital_rows) > 1:
            return True
        hospital_ids = arguments.get("hospital_ids")
        if isinstance(hospital_ids, list) and len(hospital_ids) > 1:
            return True
        return False

    def _stockout_compare_config(self) -> dict[str, Any]:
        definition = self._tool_definition(self.stockout_hybrid_tool_name)
        if definition is None:
            return {}
        config = definition.config.get("batch_compare")
        return dict(config) if isinstance(config, dict) else {}

    def _stockout_compare_bool(
        self,
        arguments: dict[str, Any],
        config: dict[str, Any],
        key: str,
    ) -> bool:
        value = arguments.get(key)
        if value is None:
            value = config.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _stockout_compare_entity_key(
        self,
        arguments: dict[str, Any],
        config: dict[str, Any],
    ) -> str:
        return str(
            arguments.get("entity_key")
            or config.get("entity_key")
            or "hospital_id"
        ).strip()

    def _stockout_compare_control_keys(
        self,
        config: dict[str, Any],
    ) -> set[str]:
        keys = set(_clean_list(config.get("control_argument_keys")))
        if not keys:
            keys = {
                "hospital_ids",
                "hospital_rows",
                "compare_mode",
                "max_rows",
                "limit",
                "raw_row_limit",
                "entity_key",
                "group_rows",
            }
        return keys

    def _stockout_compare_specs(self, value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        specs: list[dict[str, str]] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            field = str(row.get("field") or "").strip()
            if not field:
                continue
            direction = str(row.get("direction") or "asc").strip().lower()
            specs.append(
                {
                    "field": field,
                    "direction": "desc" if direction.startswith("desc") else "asc",
                }
            )
        return specs

    def _stockout_compare_numeric_value(
        self,
        value: Any,
        *,
        risk_order: dict[str, Any] | None = None,
    ) -> float | None:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        if risk_order and isinstance(value, str):
            risk_value = risk_order.get(value.strip().lower())
            if isinstance(risk_value, (int, float)):
                return float(risk_value)
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    def _stockout_compare_sort_key(
        self,
        row: dict[str, Any],
        specs: list[dict[str, str]],
        *,
        risk_order: dict[str, Any] | None = None,
        label_key: str = "",
    ) -> tuple[Any, ...]:
        parts: list[Any] = []
        for spec in specs:
            field = spec["field"]
            value = row.get(field)
            numeric = self._stockout_compare_numeric_value(
                value,
                risk_order=risk_order if field == "risk_level" else None,
            )
            missing = numeric is None and not self._has_model_input_value(value)
            if numeric is not None:
                ordered = -numeric if spec["direction"] == "desc" else numeric
                parts.append((1 if missing else 0, 0, ordered))
            else:
                text = str(value or "").strip().lower()
                parts.append((1 if missing else 0, 1, text))
        label = str(row.get(label_key) or row.get("location") or "").strip()
        parts.append(label)
        return tuple(parts)

    def _stockout_compare_row_key(
        self,
        row: dict[str, Any],
        config: dict[str, Any] | None = None,
        entity_key: str = "",
    ) -> tuple[Any, ...]:
        config = dict(config or {})
        sort_specs = self._stockout_compare_specs(config.get("sort"))
        risk_order = config.get("risk_order")
        risk_order = dict(risk_order) if isinstance(risk_order, dict) else {}
        if sort_specs:
            return self._stockout_compare_sort_key(
                row,
                sort_specs,
                risk_order=risk_order,
                label_key=entity_key,
            )

        probability = row.get("stockout_probability")
        probability_value = (
            float(probability)
            if isinstance(probability, (int, float))
            else -1.0
        )
        days = row.get("estimated_days_until_stockout")
        days_value = (
            float(days)
            if isinstance(days, (int, float))
            else math.inf
        )
        risk_level = str(row.get("risk_level") or "").strip().lower()
        risk_value = risk_order.get(risk_level, -1)
        label = str(row.get(entity_key) or row.get("location") or "").strip()
        return (-probability_value, days_value, -risk_value, label)

    def _stockout_select_compare_source_row(
        self,
        rows: list[dict[str, Any]],
        config: dict[str, Any],
        entity_key: str,
    ) -> dict[str, Any]:
        if not rows:
            return {}
        sort_specs = self._stockout_compare_specs(config.get("row_selection"))
        if not sort_specs:
            return dict(rows[0])
        ranked = sorted(
            rows,
            key=lambda row: self._stockout_compare_sort_key(
                row,
                sort_specs,
                label_key=entity_key,
            ),
        )
        return dict(ranked[0])

    def _stockout_prepare_compare_source_rows(
        self,
        raw_rows: list[Any],
        *,
        arguments: dict[str, Any],
        config: dict[str, Any],
        max_rows: int,
        entity_key: str,
    ) -> list[dict[str, Any]]:
        raw_row_limit = as_int(arguments.get("raw_row_limit"), 0)
        if raw_row_limit <= 0:
            raw_row_limit = as_int(config.get("raw_row_limit"), 0)
        if raw_row_limit > 0 and len(raw_rows) > raw_row_limit:
            raise RuntimeError(
                f"Stockout compare requires at most {raw_row_limit} source row(s)."
            )

        source_rows: list[dict[str, Any]] = []
        for index, raw_row in enumerate(raw_rows, 1):
            if isinstance(raw_row, dict):
                source_rows.append(dict(raw_row))
            elif str(raw_row or "").strip():
                source_rows.append({entity_key: str(raw_row).strip()})
            else:
                continue

        if self._stockout_compare_bool(arguments, config, "group_rows") and entity_key:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for index, row in enumerate(source_rows, 1):
                entity_value = row.get(entity_key)
                key = (
                    str(entity_value).strip()
                    if entity_value not in {None, ""}
                    else f"row_{index}"
                )
                grouped.setdefault(key, []).append(row)
            source_rows = [
                self._stockout_select_compare_source_row(rows, config, entity_key)
                for rows in grouped.values()
            ]

        if len(source_rows) > max_rows:
            raise RuntimeError(
                f"Stockout compare requires at most {max_rows} grouped row(s)."
            )
        return source_rows

    def _stockout_compare_rows(
        self,
        *,
        runtime: ModelRuntime,
        query: str,
        request_features: dict[str, Any],
        arguments: dict[str, Any],
        preferred_model_id: str,
    ) -> dict[str, Any]:
        compare_config = self._stockout_compare_config()
        entity_key = self._stockout_compare_entity_key(arguments, compare_config)
        max_rows = as_int(
            arguments.get("max_rows", arguments.get("limit")),
            0,
        )
        if max_rows <= 0:
            max_rows = as_int(compare_config.get("max_rows"), 0)
        compare_mode = str(arguments.get("compare_mode") or "").strip().lower()
        raw_rows = arguments.get("hospital_rows")
        if not isinstance(raw_rows, list) or not raw_rows:
            raw_ids = arguments.get("hospital_ids")
            if isinstance(raw_ids, list) and raw_ids:
                raw_rows = [
                    {entity_key: hospital_id}
                    for hospital_id in raw_ids
                    if str(hospital_id or "").strip()
                ]
        if not isinstance(raw_rows, list) or not raw_rows:
            raise RuntimeError(
                "Stockout compare requires bounded hospital_rows or hospital_ids."
            )
        if max_rows <= 0:
            max_rows = len(raw_rows)

        source_rows = self._stockout_prepare_compare_source_rows(
            raw_rows,
            arguments=arguments,
            config=compare_config,
            max_rows=max_rows,
            entity_key=entity_key,
        )

        compare_rows: list[dict[str, Any]] = []
        warnings: list[str] = []
        control_keys = self._stockout_compare_control_keys(compare_config)
        for index, raw_row in enumerate(source_rows, 1):
            row_features = dict(request_features)
            row_features.update(raw_row)
            for key in control_keys:
                row_features.pop(key, None)
            row_features = self._drop_empty_model_inputs(row_features)
            if compare_mode == "compare" and entity_key not in row_features:
                row_features[entity_key] = str(index)
            try:
                output = build_hybrid_stockout_prediction(
                    registry=self.registry,
                    regression_runtime=runtime,
                    request_features=row_features,
                    query=query,
                )
            except Exception as exc:
                warnings.append(str(exc))
                continue
            if not isinstance(output, dict) or not output:
                warnings.append(
                    f"{row_features.get(entity_key) or index}: no stockout output"
                )
                continue
            entity_id = str(
                row_features.get(entity_key)
                or output.get("location")
                or output.get(entity_key)
                or index
            ).strip()
            compare_row: dict[str, Any] = {
                "rank": 0,
                entity_key: entity_id,
                "location": output.get("location") or entity_id,
                "model_id": runtime.model_id,
                "model_score": output.get("stockout_probability"),
                "stockout_probability": output.get("stockout_probability"),
                "estimated_days_until_stockout": output.get("estimated_days_until_stockout"),
                "risk_level": output.get("risk_level"),
                "recommended_action": output.get("recommended_action"),
                "worst_blood_group": output.get("worst_blood_group"),
                "worst_component": output.get("worst_component"),
                "regression_model_id": output.get("regression_model_id", preferred_model_id),
                "model_output": output,
            }
            if output.get("confidence") is not None:
                compare_row["confidence"] = output.get("confidence")
            if output.get("horizon") is not None:
                compare_row["horizon"] = output.get("horizon")
            if output.get("feature_resolution") is not None:
                compare_row["feature_resolution"] = output.get("feature_resolution")
            compare_rows.append(compare_row)

        if not compare_rows:
            raise RuntimeError("No comparable hospital rows could be scored.")

        compare_rows.sort(
            key=lambda row: self._stockout_compare_row_key(
                row,
                compare_config,
                entity_key,
            )
        )
        for index, row in enumerate(compare_rows, 1):
            row["rank"] = index

        preview_parts = []
        for row in compare_rows[:3]:
            label = str(row.get(entity_key) or row.get("location") or row.get("rank"))
            score = row.get("model_score")
            if isinstance(score, (int, float)):
                label = f"{label} score {float(score):.4f}"
            preview_parts.append(label)
        preview = ", ".join(preview_parts)
        payload: dict[str, Any] = {
            "query_kind": "batch_model_ranking",
            "comparison_mode": compare_mode or "compare",
            "entity_key": entity_key,
            "model_id": runtime.model_id,
            "row_count": len(compare_rows),
            "source_row_count": len(raw_rows),
            "rows": compare_rows,
            "answer": (
                f"Ranked {len(compare_rows)} candidate(s) with {runtime.model_id}."
                + (f" Top candidates: {preview}." if preview else "")
            ),
        }
        if warnings:
            payload["warnings"] = sorted(set(warnings))
        return payload

    def _stockout_hybrid_runtime_rank_key(
        self,
        runtime: ModelRuntime,
        index: int,
        *,
        forecast_horizon: ForecastHorizon | None,
        hospital_scoped: bool,
        primary_model_id: str,
    ) -> tuple[float, int]:
        defaults = self._planner_runtime_defaults(runtime)
        routing = defaults.get("horizon_routing")
        if not isinstance(routing, dict):
            routing = {}
        raw_timeframes = routing.get("timeframes") or []
        if isinstance(raw_timeframes, str):
            raw_timeframes = [raw_timeframes]
        timeframes = {
            str(value or "").strip().lower()
            for value in raw_timeframes
            if str(value or "").strip()
        }
        min_days = (
            as_float(routing.get("min_days"), 0.0)
            if routing.get("min_days") is not None
            else None
        )
        max_days = (
            as_float(routing.get("max_days"), 0.0)
            if routing.get("max_days") is not None
            else None
        )
        model_id = str(getattr(runtime, "model_id", "") or "").lower()
        description = str(getattr(runtime, "description", "") or "").lower()
        haystack = f"{model_id} {description}"
        routing = stockout_orchestrator_routing_config()
        scores = routing.get("scores")
        scores = scores if isinstance(scores, dict) else {}
        timeframe_scores = routing.get("default_timeframe_scores")
        timeframe_scores = timeframe_scores if isinstance(timeframe_scores, dict) else {}
        priority_terms = routing.get("model_priority_terms")
        priority_terms = priority_terms if isinstance(priority_terms, dict) else {}
        penalty_terms = routing.get("model_penalty_terms")
        penalty_terms = penalty_terms if isinstance(penalty_terms, dict) else {}
        scoped_model_terms = _clean_list(
            routing.get("scoped_model_terms")
            or routing.get("scoped_query_terms")
        )

        score = 0.0
        if forecast_horizon is not None:
            if forecast_horizon.timeframe in timeframes:
                score += as_float(scores.get("horizon_timeframe_match"), 0.0)
            if forecast_horizon.days is not None:
                days = float(forecast_horizon.days)
                min_ok = min_days is None or days >= min_days
                max_ok = max_days is None or days <= max_days
                score += as_float(
                    scores.get("horizon_days_match" if min_ok and max_ok else "horizon_days_mismatch"),
                    0.0,
                )
        else:
            for timeframe in timeframes:
                score += as_float(timeframe_scores.get(timeframe), 0.0)
            for term, value in penalty_terms.items():
                if str(term or "").lower() in model_id:
                    score += as_float(value, 0.0)

        if hospital_scoped and any(
            str(term or "").lower() in haystack for term in scoped_model_terms
        ):
            score += as_float(scores.get("scoped_text_match"), 0.0)
        for term, value in priority_terms.items():
            if str(term or "").lower() in model_id:
                score += as_float(value, 0.0)
        if str(primary_model_id or "") and runtime.model_id == primary_model_id:
            score += as_float(scores.get("primary_model"), 0.0)

        return (-score, index)

    def _is_search_tool(self, tool_name: str) -> bool:
        return self._tool_adapter(tool_name) == "search"

    def _is_llm_api_tool(self, tool_name: str) -> bool:
        return self._tool_adapter(tool_name) == "llm_api"

    def _is_simulation_tool(self, tool_name: str) -> bool:
        return self._tool_adapter(tool_name) == "simulation_query"

    def _is_digital_twin_tool(self, tool_name: str) -> bool:
        return self._tool_adapter(tool_name) == "digital_twin"

    def _is_external_fallback_step(self, tool_name: str) -> bool:
        normalized = safe_slug(tool_name)
        return bool(self._tool_adapter(normalized) or normalized in self.fallback_aliases)

    def _is_external_query_tool(self, tool_name: str) -> bool:
        return self._is_external_fallback_step(tool_name)

    def _plan_needs_external_fallback(
        self,
        query: str,
        plan: ExecutionPlan,
        allowed_models: dict[str, ModelRuntime],
        forced_model_ids: list[str] | None,
    ) -> bool:
        if not self.enable_external_fallback:
            return False
        if forced_model_ids:
            return False
        model_steps = [
            allowed_models[s.name] for s in plan.steps if s.name in allowed_models
        ]
        if not model_steps:
            return False
        has_external = any(self._is_external_query_tool(s.name) for s in plan.steps)
        if has_external:
            return False
        for step in plan.steps:
            rt = allowed_models.get(step.name)
            if rt is not None and self._model_step_has_complete_inputs(query, step, rt):
                return False
        return all(not self._is_query_in_model_scope(query, rt) for rt in model_steps)

    def _plan_contains_runtime_step(
        self,
        plan: ExecutionPlan,
        runtime: ModelRuntime,
    ) -> bool:
        valid_names = {
            safe_slug(name)
            for name in [runtime.model_id, *list(runtime.aliases or [])]
            if str(name or "").strip()
        }
        return any(safe_slug(step.name) in valid_names for step in plan.steps)

    def _horizon_routed_plan(
        self,
        query: str,
        runtime: ModelRuntime,
        prediction_features: dict[str, Any],
    ) -> ExecutionPlan:
        arguments = extract_nl_features(query)
        arguments.update(prediction_features)
        return ExecutionPlan(
            steps=[
                ToolCall(
                    name=runtime.model_id,
                    arguments=arguments,
                    reasoning=(
                        "Deterministic horizon routing selected this model as the "
                        "best validation match for the requested forecast horizon."
                    ),
                )
            ],
            reasoning=(
                "Horizon routing selected the best stockout model for the requested "
                "forecast horizon, so model inference is required."
            ),
            is_multi_step=False,
        )

    def _repair_plan_for_simulation_query(
        self,
        query: str,
        plan: ExecutionPlan,
    ) -> ExecutionPlan:
        if not self.simulation_tool_enabled or not is_simulation_query(query):
            return plan
        if len(plan.steps) == 1 and self._is_simulation_tool(plan.steps[0].name):
            step = plan.steps[0]
            arguments = dict(step.arguments) if isinstance(step.arguments, dict) else {}
            arguments["query"] = query
            return ExecutionPlan(
                steps=[
                    ToolCall(
                        name=self.simulation_tool_name,
                        arguments=arguments,
                        reasoning=step.reasoning,
                    )
                ],
                reasoning=plan.reasoning,
                is_multi_step=False,
            )
        return ExecutionPlan(
            steps=[
                ToolCall(
                    name=self.simulation_tool_name,
                    arguments={"query": query},
                    reasoning="Repair: simulation-intent query must use the simulation tool.",
                )
            ],
            reasoning="Repaired plan: simulation-intent query routed to the simulation tool.",
            is_multi_step=False,
        )

    # ══════════════════════════════════════════════════════
    #  EXTERNAL FALLBACK CHAIN
    # ══════════════════════════════════════════════════════

    def _execute_db_fallback(
        self,
        query: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.db_fallback_enabled:
            raise RuntimeError("DB fallback is disabled.")
        db_arguments = dict(arguments or {})
        db_arguments.pop("query", None)
        return execute_db_tool(
            query,
            db_supply_csv_path=self.db_supply_csv_path,
            db_sqlite_path=self.db_sqlite_path,
            db_sqlite_table=self.db_sqlite_table,
            db_series_limit=self.db_series_limit,
            db_schema=self.db_schema,
            donors_table_name=self.db_donors_table,
            hospitals_table_name=self.db_hospitals_table,
            csv_candidates=self._db_csv_candidates(),
            **db_arguments,
        )

    def _db_csv_candidates(self) -> list[Path]:
        candidates = [
            self.db_supply_csv_path,
            self.db_donor_csv_path,
            BACKEND_ML_DIR / "datasets" / "synthetic_bloodbank_daily.csv",
            BACKEND_ML_DIR / "datasets" / "synthetic_blood_transfusion.csv",
            BACKEND_ML_DIR
            / "datasets"
            / "synthetic_blood_transfusion_with_features.csv",
            BACKEND_ML_DIR
            / "datasets"
            / "blood_registry_sythentic"
            / "data"
            / "blood_donation_registry_ml_ready.csv",
            BACKEND_ML_DIR / "datasets" / "synthetic_donor_candidates_clustered.csv",
            BACKEND_ML_DIR / "datasets" / "donor_snapshots_with_targets.csv",
            ML_BACKEND_DIR / "datasets" / "synthetic_bloodbank_daily.csv",
            ML_BACKEND_DIR / "datasets" / "synthetic_blood_transfusion.csv",
            ML_BACKEND_DIR
            / "datasets"
            / "synthetic_blood_transfusion_with_features.csv",
            ML_BACKEND_DIR
            / "datasets"
            / "blood_registry_sythentic"
            / "data"
            / "blood_donation_registry_ml_ready.csv",
            ML_BACKEND_DIR / "datasets" / "synthetic_donor_candidates_clustered.csv",
            ML_BACKEND_DIR / "datasets" / "donor_snapshots_with_targets.csv",
        ]
        return _dedupe_paths(
            [candidate.expanduser().resolve() for candidate in candidates]
        )

    def _execute_search_tool(self, query: str) -> dict[str, Any]:
        if not self.search_fallback_enabled:
            raise RuntimeError("Search fallback is disabled.")
        return execute_search_tool(
            query,
            search_api_url=self.search_api_url,
            search_timeout_s=self.search_timeout_s,
            search_result_limit=self.search_result_limit,
        )

    def _execute_llm_api_fallback(self, query: str) -> dict[str, Any]:
        return execute_llm_api_tool(
            query,
            llm_api_url=self.llm_api_url,
            llm_api_key=self.llm_api_key,
            llm_api_model=self.llm_api_model,
            llm_api_timeout_s=self.llm_api_timeout_s,
            llm_api_system_prompt=self.llm_api_system_prompt,
            local_llm=self._llm,
            local_llm_active_model_id=self._active_model_id,
            local_llm_prompt_mode=self._current_prompt_mode(),
            safe_generation_tokens_fn=self._safe_generation_tokens,
        )

    def _execute_simulation_tool(self, query: str) -> dict[str, Any]:
        if not self.simulation_tool_enabled:
            raise RuntimeError("Simulation tool is disabled.")
        return execute_simulation_tool(
            query,
            simulation_base_url=self.simulation_base_url,
            simulation_timeout_s=self.simulation_timeout_s,
            simulation_default_policy=self.simulation_default_policy,
            simulation_compare_on_recommend=self.simulation_compare_on_recommend,
            simulation_default_compare_policies=self.simulation_default_compare_policies,
        )

    def _execute_digital_twin_tool(
        self,
        tool_name: str,
        query: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        definition = self.external_tool_aliases.get(safe_slug(tool_name))
        if definition is None:
            raise RuntimeError(f"Unknown digital twin tool: {tool_name}")
        operation = str(definition.config.get("operation") or definition.id).strip()
        tool_arguments = dict(arguments or {})
        tool_arguments.setdefault("tool_id", definition.id)
        return execute_digital_twin_tool(
            operation=operation,
            query=query,
            arguments=tool_arguments,
        )

    def _execute_stockout_hybrid_tool(
        self,
        query: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        arguments = dict(arguments or {})
        preferred_model_id = str(
            arguments.pop("regression_model_id", "")
            or arguments.pop("model_id", "")
            or ""
        ).strip()
        compare_requested = self._stockout_compare_requested(arguments)
        request_features = {
            **self._drop_empty_model_inputs(extract_nl_features(query)),
            **self._drop_empty_model_inputs(arguments),
        }
        runtime = self._select_stockout_hybrid_runtime(
            query,
            request_features,
            preferred_model_id=preferred_model_id,
        )
        if runtime is None:
            raise RuntimeError("No configured hybrid stockout regression model is loaded.")
        if compare_requested:
            return self._stockout_compare_rows(
                runtime=runtime,
                query=query,
                request_features=request_features,
                arguments=arguments,
                preferred_model_id=preferred_model_id or runtime.model_id,
            )
        output = build_hybrid_stockout_prediction(
            registry=self.registry,
            regression_runtime=runtime,
            request_features=request_features,
            query=query,
        )
        if not output:
            raise RuntimeError(
                f"{runtime.model_id} is not configured for hybrid stockout."
            )
        return output

    def _execute_external_tool(
        self,
        tool_name: str,
        query: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._is_stockout_hybrid_tool(tool_name):
            return self._execute_stockout_hybrid_tool(query, arguments)
        if self._is_db_tool(tool_name):
            return self._execute_db_fallback(query, arguments)
        if self._is_search_tool(tool_name):
            return self._execute_search_tool(query)
        if self._is_llm_api_tool(tool_name):
            return self._execute_llm_api_fallback(query)
        if self._is_simulation_tool(tool_name):
            return self._execute_simulation_tool(query)
        if self._is_digital_twin_tool(tool_name):
            return self._execute_digital_twin_tool(tool_name, query, arguments)
        raise RuntimeError(f"Unknown external tool: {tool_name}")

    def list_configured_tools(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen_tool_ids: set[str] = set()
        for definition in self.external_tool_definitions:
            config = definition.config
            aliases = _clean_list(config.get("aliases"))
            rows.append(
                {
                    "id": definition.id,
                    "name": definition.name,
                    "adapter": definition.adapter,
                    "enabled": definition.enabled,
                    "available": self._direct_tool_available(definition),
                    "description": str(config.get("description") or "").strip(),
                    "aliases": aliases,
                    "input_schema": config.get("input_schema"),
                    "output_schema": config.get("output_schema"),
                    "run_endpoint": f"/api/ml/tools/{definition.id}/run/",
                }
            )
            seen_tool_ids.add(definition.id)
        for runtime in self.registry.loaded():
            if runtime.model_id in seen_tool_ids:
                continue
            output_schema = {}
            defaults = getattr(runtime, "defaults", {})
            if isinstance(defaults, dict):
                output_schema = defaults.get("output") if isinstance(defaults.get("output"), dict) else {}
            feature_info = getattr(runtime, "feature_info", {}) or {}
            properties = {}
            for feature in getattr(runtime, "feature_names", []) or []:
                dtype = str(feature_info.get(feature) or "").lower()
                properties[feature] = {
                    "type": "string"
                    if dtype in {"string", "category", "categorical", "date", "datetime", "timestamp"}
                    else "number"
                }
            rows.append(
                {
                    "id": runtime.model_id,
                    "name": runtime.model_id.replace("_", " ").title(),
                    "adapter": "model",
                    "enabled": True,
                    "available": runtime.status == "loaded",
                    "description": str(runtime.description or "").strip(),
                    "aliases": list(dict.fromkeys(getattr(runtime, "aliases", []) or [])),
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                    },
                    "output_schema": output_schema,
                    "run_endpoint": f"/api/ml/tools/{runtime.model_id}/run/",
                }
            )
        return rows

    def execute_configured_tool(
        self,
        tool_name: str,
        *,
        query: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        definition = self._tool_definition(tool_name)
        if definition is None:
            runtime = self._model_tool_runtime(tool_name)
            if runtime is None:
                raise RuntimeError(f"Unknown configured tool: {tool_name}")
            return self._execute_model_tool(runtime, dict(arguments or {}))
        if not definition.enabled:
            raise RuntimeError(f"Configured tool is disabled: {definition.id}")
        if not self._direct_tool_available(definition):
            raise RuntimeError(f"Configured tool is unavailable: {definition.id}")
        return self._execute_external_tool(
            definition.id,
            query,
            dict(arguments or {}),
        )

    def configured_tool_id(self, tool_name: str) -> str:
        definition = self._tool_definition(tool_name)
        if definition is not None:
            return definition.id
        runtime = self._model_tool_runtime(tool_name)
        return runtime.model_id if runtime is not None else ""

    def _model_tool_runtime(self, tool_name: str) -> ModelRuntime | None:
        runtime = self.registry.get(tool_name)
        if runtime is None or runtime.status != "loaded":
            return None
        return runtime

    def _execute_model_tool(
        self,
        runtime: ModelRuntime,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        feature_rows = arguments.get("input_rows")
        if not isinstance(feature_rows, list):
            feature_rows = arguments.get("inputs")
        if isinstance(feature_rows, list) and feature_rows:
            resolved_rows = [
                resolve_prediction_features(runtime, dict(row)).features
                for row in feature_rows
                if isinstance(row, dict)
            ]
            outputs = run_prediction_batch_with_fallback(
                runtime,
                resolved_rows,
                registry=self.registry,
            )
            return {
                "query_kind": "batch_predictions",
                "model_id": runtime.model_id,
                "row_count": len(outputs),
                "rows": outputs,
            }

        features = arguments.get("features") if isinstance(arguments.get("features"), dict) else arguments
        resolved = resolve_prediction_features(runtime, dict(features or {}))
        return run_prediction_with_fallback(
            runtime,
            resolved.features,
            registry=self.registry,
        )

    def _run_external_fallback_chain(
        self,
        query: str,
        start_step: int,
        preferred_first: str | None = None,
        preferred_arguments: dict[str, Any] | None = None,
    ) -> list[ExecutionResult]:
        if not self.enable_external_fallback:
            return []
        order = list(self.external_fallback_order)
        if preferred_first and self._is_external_fallback_step(preferred_first):
            normalized = safe_slug(preferred_first)
            order = [normalized] + [n for n in order if n != normalized]
        strict_db_only = bool(
            preferred_first
            and self._is_db_tool(preferred_first)
            and isinstance(preferred_arguments, dict)
            and any(
                key in preferred_arguments
                for key in (
                    "table",
                    "field",
                    "fields",
                    "aggregate",
                    "group_by",
                    "filters",
                    "latest_only",
                )
            )
        )

        rows: list[ExecutionResult] = []
        next_step = start_step
        for tool_name in order:
            definition = self._tool_definition(tool_name)
            if definition is None or not definition.enabled:
                continue
            if strict_db_only and definition.adapter != "db_query":
                break
            is_preferred = bool(
                preferred_first and self._tool_definition(preferred_first) == definition
            )
            tool_inputs = {"query": query}
            if is_preferred and preferred_arguments:
                tool_inputs.update(preferred_arguments)
            try:
                output = self._execute_external_tool(
                    definition.id,
                    query,
                    preferred_arguments if is_preferred else None,
                )
                success = is_external_output_success(output)
                rows.append(
                    ExecutionResult(
                        next_step,
                        definition.id,
                        tool_inputs,
                        output,
                        success,
                        None if success else f"{definition.name} returned no usable answer.",
                    )
                )
            except Exception as exc:
                rows.append(
                    ExecutionResult(
                        next_step,
                        definition.id,
                        tool_inputs,
                        {},
                        False,
                        str(exc),
                    )
                )
            next_step += 1
            if rows and (rows[-1].success or strict_db_only):
                break
        return rows

    # ══════════════════════════════════════════════════════
    #  PLANNER
    # ══════════════════════════════════════════════════════

    def _configured_tool_id_from(
        self,
        *,
        adapter: str,
        env_name: str,
        aliases: dict[str, ExternalToolDefinition],
        by_adapter: dict[str, list[ExternalToolDefinition]],
        required: bool = True,
    ) -> str:
        raw_override = os.getenv(env_name, "").strip()
        if raw_override:
            definition = aliases.get(safe_slug(raw_override))
            if definition is None:
                raise RuntimeError(
                    f"{env_name} references an unknown configured external tool: {raw_override}"
                )
            if definition.adapter != adapter:
                raise RuntimeError(
                    f"{env_name} references adapter '{definition.adapter}', expected '{adapter}'."
                )
            if not definition.enabled:
                if required:
                    raise RuntimeError(
                        f"{env_name} references a disabled external tool: {raw_override}"
                    )
                return ""
            return definition.id
        candidates = [row for row in by_adapter.get(adapter, []) if row.enabled]
        if candidates:
            return candidates[0].id
        if required:
            raise RuntimeError(f"No enabled external tool configured for adapter '{adapter}'.")
        return ""

    def _build_external_tools_runtime_state(self) -> dict[str, Any]:
        payload = self._load_external_tools_config()
        orchestrator_config = dict(payload.get("orchestrator") or {})
        definitions = self._load_external_tool_definitions(payload)
        aliases = self._external_tool_aliases(definitions)
        by_adapter = self._external_tools_by_adapter(definitions)
        fallback_aliases = {
            safe_slug(value)
            for value in orchestrator_config.get("fallback_aliases", [])
            if safe_slug(value)
        }
        prompt_templates = dict(orchestrator_config.get("prompts") or {})
        model_execution_config = dict(orchestrator_config.get("model_execution") or {})
        model_control_argument_keys = set(
            _clean_list(model_execution_config.get("control_argument_keys"))
        ) or {"batch"}
        planning_limits_config = dict(orchestrator_config.get("planning_limits") or {})
        dynamic_planning_config = dict(orchestrator_config.get("dynamic_planning") or {})
        direct_response_config = dict(dynamic_planning_config.get("direct_response") or {})

        planner_example_limit = max(
            0,
            as_int(
                os.getenv(
                    "PIOS_ORCH_PLANNER_EXAMPLE_LIMIT",
                    planning_limits_config.get(
                        "example_limit",
                        orchestrator_config.get("planner_example_limit"),
                    ),
                ),
                4,
            ),
        )
        planner_schema_field_limit = max(
            0,
            as_int(
                os.getenv(
                    "PIOS_ORCH_PLANNER_SCHEMA_FIELD_LIMIT",
                    planning_limits_config.get("schema_field_limit"),
                ),
                18,
            ),
        )
        planner_tool_description_chars = max(
            80,
            as_int(
                os.getenv(
                    "PIOS_ORCH_PLANNER_TOOL_DESCRIPTION_CHARS",
                    planning_limits_config.get("tool_description_chars"),
                ),
                420,
            ),
        )
        planner_model_description_chars = max(
            80,
            as_int(
                os.getenv(
                    "PIOS_ORCH_PLANNER_MODEL_DESCRIPTION_CHARS",
                    planning_limits_config.get("model_description_chars"),
                ),
                140,
            ),
        )
        planner_completion_tokens = max(
            64,
            as_int(
                os.getenv(
                    "PIOS_ORCH_PLANNER_COMPLETION_TOKENS",
                    planning_limits_config.get("completion_tokens"),
                ),
                256,
            ),
        )
        planner_context_reserve_tokens = max(
            16,
            as_int(
                os.getenv(
                    "PIOS_ORCH_PLANNER_CONTEXT_RESERVE_TOKENS",
                    planning_limits_config.get("context_reserve_tokens"),
                ),
                128,
            ),
        )
        planner_workflow_example_limit = max(
            0,
            as_int(
                os.getenv(
                    "PIOS_ORCH_PLANNER_WORKFLOW_EXAMPLE_LIMIT",
                    planning_limits_config.get(
                        "workflow_example_limit",
                        dynamic_planning_config.get("workflow_example_limit"),
                    ),
                ),
                0,
            ),
        )
        fallback_model_step_limit = max(
            0,
            as_int(
                os.getenv(
                    "PIOS_ORCH_FALLBACK_MODEL_STEP_LIMIT",
                    planning_limits_config.get("fallback_model_step_limit"),
                ),
                0,
            ),
        )

        enable_external_fallback = (
            os.getenv("PIOS_ORCH_ENABLE_EXTERNAL_FALLBACK", "1") == "1"
        )
        db_fallback_enabled = (
            os.getenv(
                "PIOS_ORCH_ENABLE_DB_FALLBACK",
                os.getenv("PIOS_ORCH_ENABLE_DB_TOOL", "1"),
            )
            == "1"
        )
        db_tool_name = self._configured_tool_id_from(
            adapter="db_query",
            env_name="PIOS_ORCH_DB_TOOL_NAME",
            aliases=aliases,
            by_adapter=by_adapter,
        )
        stockout_hybrid_tool_name = self._configured_tool_id_from(
            adapter="stockout_hybrid",
            env_name="PIOS_ORCH_STOCKOUT_HYBRID_TOOL_NAME",
            aliases=aliases,
            by_adapter=by_adapter,
            required=False,
        )
        search_tool_name = self._configured_tool_id_from(
            adapter="search",
            env_name="PIOS_ORCH_SEARCH_TOOL_NAME",
            aliases=aliases,
            by_adapter=by_adapter,
            required=False,
        )
        simulation_tool_name = self._configured_tool_id_from(
            adapter="simulation_query",
            env_name="PIOS_ORCH_SIMULATION_TOOL_NAME",
            aliases=aliases,
            by_adapter=by_adapter,
            required=False,
        )
        search_fallback_enabled = (
            os.getenv(
                "PIOS_ORCH_ENABLE_SEARCH_FALLBACK",
                os.getenv("PIOS_ORCH_ENABLE_SEARCH_TOOL", "1"),
            )
            == "1"
        ) and bool(search_tool_name)
        simulation_tool_enabled = (
            os.getenv(
                "PIOS_ORCH_ENABLE_SIMULATION_TOOL",
                os.getenv("PIOS_ORCH_ENABLE_SIMULATION_FALLBACK", "1"),
            )
            == "1"
        ) and bool(simulation_tool_name)
        llm_api_system_prompt = (
            os.getenv(
                "PIOS_ORCH_LLM_API_SYSTEM_PROMPT",
                str(prompt_templates.get("llm_api_system") or ""),
            ).strip()
            or str(prompt_templates.get("llm_api_system") or "").strip()
        )
        raw_fallback = os.getenv(
            "PIOS_ORCH_EXTERNAL_FALLBACK_ORDER",
            os.getenv(
                "PIOS_ORCH_DEFAULT_FALLBACK_ORDER",
                ",".join(orchestrator_config.get("fallback_order") or []),
            ),
        )
        external_fallback_order = [
            safe_slug(name) for name in raw_fallback.split(",") if safe_slug(name)
        ]
        if not external_fallback_order:
            external_fallback_order = [
                tool.id
                for tool in definitions
                if tool.adapter not in {"stockout_hybrid", "search"}
            ]

        return {
            "external_tools_config": payload,
            "orchestrator_config": orchestrator_config,
            "external_tool_definitions": definitions,
            "external_tool_aliases": aliases,
            "external_tools_by_adapter": by_adapter,
            "fallback_aliases": fallback_aliases,
            "prompt_templates": prompt_templates,
            "model_execution_config": model_execution_config,
            "model_control_argument_keys": model_control_argument_keys,
            "planning_limits_config": planning_limits_config,
            "dynamic_planning_config": dynamic_planning_config,
            "direct_response_config": direct_response_config,
            "planner_example_limit": planner_example_limit,
            "planner_schema_field_limit": planner_schema_field_limit,
            "planner_tool_description_chars": planner_tool_description_chars,
            "planner_model_description_chars": planner_model_description_chars,
            "planner_completion_tokens": planner_completion_tokens,
            "planner_context_reserve_tokens": planner_context_reserve_tokens,
            "planner_workflow_example_limit": planner_workflow_example_limit,
            "fallback_model_step_limit": fallback_model_step_limit,
            "enable_external_fallback": enable_external_fallback,
            "db_fallback_enabled": db_fallback_enabled,
            "search_fallback_enabled": search_fallback_enabled,
            "simulation_tool_enabled": simulation_tool_enabled,
            "db_tool_name": db_tool_name,
            "stockout_hybrid_tool_name": stockout_hybrid_tool_name,
            "search_tool_name": search_tool_name,
            "simulation_tool_name": simulation_tool_name,
            "llm_api_system_prompt": llm_api_system_prompt,
            "external_fallback_order": external_fallback_order,
        }

    def _apply_external_tools_runtime_state(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    def _external_tools_runtime_signature(
        self,
        state: dict[str, Any] | None = None,
    ) -> str:
        if state is None:
            definitions = getattr(self, "external_tool_definitions", [])
            payload = {
                "external_tools": [definition.config for definition in definitions],
                "orchestrator": getattr(self, "orchestrator_config", {}),
                "fallback_aliases": sorted(getattr(self, "fallback_aliases", set())),
                "prompt_templates": getattr(self, "prompt_templates", {}),
                "planning_limits": getattr(self, "planning_limits_config", {}),
                "dynamic_planning": getattr(self, "dynamic_planning_config", {}),
                "external_fallback_order": getattr(self, "external_fallback_order", []),
            }
        else:
            definitions = state["external_tool_definitions"]
            payload = {
                "external_tools": [definition.config for definition in definitions],
                "orchestrator": state["orchestrator_config"],
                "fallback_aliases": sorted(state["fallback_aliases"]),
                "prompt_templates": state["prompt_templates"],
                "planning_limits": state["planning_limits_config"],
                "dynamic_planning": state["dynamic_planning_config"],
                "external_fallback_order": state["external_fallback_order"],
            }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def reload_external_tools_config(self, *, reason: str) -> dict[str, Any]:
        before_signature = self._external_tools_runtime_signature()
        state = self._build_external_tools_runtime_state()
        after_signature = self._external_tools_runtime_signature(state)
        changed = before_signature != after_signature
        with self._lock:
            self._apply_external_tools_runtime_state(state)
        definitions = state["external_tool_definitions"]
        return {
            "available": True,
            "reason": reason,
            "changed": changed,
            "config_path": str(self.external_tools_config_path),
            "external_tools_total": len(definitions),
            "enabled_external_tools_total": sum(1 for item in definitions if item.enabled),
            "prompt_keys": sorted(state["prompt_templates"].keys()),
            "planning_limits": dict(state["planning_limits_config"]),
            "reloaded_at": time.time(),
        }

    def _load_external_tools_config(self) -> dict[str, Any]:
        path = self.external_tools_config_path
        if not path.exists():
            raise RuntimeError(f"External tools config not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid external tools config JSON: {path}") from exc
        except OSError as exc:
            raise RuntimeError(f"Unable to read external tools config: {path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("External tools config must be a JSON object.")
        return payload

    def _load_external_tool_definitions(
        self,
        payload: dict[str, Any],
    ) -> list[ExternalToolDefinition]:
        rows = payload.get("external_tools", [])
        if not isinstance(rows, list):
            raise RuntimeError("external_tools config must be a list.")
        required = {
            "id",
            "name",
            "type",
            "enabled",
            "description",
            "input_schema",
            "output_schema",
            "adapter",
            "entrypoint",
            "permissions",
            "timeout_seconds",
        }
        supported_types = {"external"}
        supported_adapters = {
            "db_query",
            "search",
            "llm_api",
            "simulation_query",
            "digital_twin",
            "stockout_hybrid",
        }
        definitions: list[ExternalToolDefinition] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("Each external tool config must be an object.")
            missing = required - set(row)
            if missing:
                raw_name = row.get("id") or row.get("name") or "<unknown>"
                raise RuntimeError(
                    f"External tool '{raw_name}' missing required fields: "
                    f"{', '.join(sorted(missing))}"
                )
            tool_id = safe_slug(row.get("id"))
            tool_name = safe_slug(row.get("name"))
            adapter = safe_slug(row.get("adapter"))
            if not tool_id or not tool_name:
                raise RuntimeError("External tool id and name must be non-empty.")
            if tool_id in seen:
                raise RuntimeError(f"Duplicate external tool id: {tool_id}")
            seen.add(tool_id)
            if str(row.get("type")).strip() not in supported_types:
                raise RuntimeError(f"Unsupported external tool type for {tool_id}.")
            if adapter not in supported_adapters:
                raise RuntimeError(f"Unsupported external tool adapter for {tool_id}.")
            if not isinstance(row.get("enabled"), bool):
                raise RuntimeError(f"External tool '{tool_id}' enabled must be boolean.")
            if not isinstance(row.get("input_schema"), dict):
                raise RuntimeError(f"External tool '{tool_id}' input_schema must be an object.")
            if not isinstance(row.get("output_schema"), dict):
                raise RuntimeError(f"External tool '{tool_id}' output_schema must be an object.")
            if not isinstance(row.get("permissions"), list):
                raise RuntimeError(f"External tool '{tool_id}' permissions must be a list.")
            if as_float(row.get("timeout_seconds"), 0.0) <= 0:
                raise RuntimeError(f"External tool '{tool_id}' timeout_seconds must be positive.")
            batch_compare = row.get("batch_compare")
            if batch_compare is not None:
                if not isinstance(batch_compare, dict):
                    raise RuntimeError(
                        f"External tool '{tool_id}' batch_compare must be an object."
                    )
                if "entity_key" in batch_compare and not str(
                    batch_compare.get("entity_key") or ""
                ).strip():
                    raise RuntimeError(
                        f"External tool '{tool_id}' batch_compare.entity_key must be non-empty."
                    )
                if "max_rows" in batch_compare and as_int(
                    batch_compare.get("max_rows"),
                    0,
                ) <= 0:
                    raise RuntimeError(
                        f"External tool '{tool_id}' batch_compare.max_rows must be positive."
                    )
            definitions.append(
                ExternalToolDefinition(
                    id=tool_id,
                    name=tool_name,
                    adapter=adapter,
                    enabled=bool(row.get("enabled")),
                    config=row,
                )
            )
        return definitions

    def _external_tool_aliases(
        self,
        definitions: list[ExternalToolDefinition],
    ) -> dict[str, ExternalToolDefinition]:
        aliases: dict[str, ExternalToolDefinition] = {}
        for definition in definitions:
            names = [definition.id, definition.name]
            names.extend(_clean_list(definition.config.get("aliases")))
            for name in names:
                normalized = safe_slug(name)
                if normalized:
                    aliases[normalized] = definition
        return aliases

    def _external_tools_by_adapter(
        self,
        definitions: list[ExternalToolDefinition],
    ) -> dict[str, list[ExternalToolDefinition]]:
        rows: dict[str, list[ExternalToolDefinition]] = {}
        for definition in definitions:
            rows.setdefault(definition.adapter, []).append(definition)
        return rows

    def _configured_tool_id(
        self,
        *,
        adapter: str,
        env_name: str,
        required: bool = True,
    ) -> str:
        raw_override = os.getenv(env_name, "").strip()
        if raw_override:
            definition = self.external_tool_aliases.get(safe_slug(raw_override))
            if definition is None:
                raise RuntimeError(
                    f"{env_name} references an unknown configured external tool: {raw_override}"
                )
            if definition.adapter != adapter:
                raise RuntimeError(
                    f"{env_name} references adapter '{definition.adapter}', expected '{adapter}'."
                )
            return definition.id
        candidates = [
            row
            for row in self.external_tools_by_adapter.get(adapter, [])
            if row.enabled
        ]
        if candidates:
            return candidates[0].id
        if required:
            raise RuntimeError(f"No enabled external tool configured for adapter '{adapter}'.")
        return ""

    def _available_tool_id_for_adapter(self, adapter: str) -> str:
        for definition in self.external_tools_by_adapter.get(adapter, []):
            if self._external_tool_available(definition):
                return definition.id
        return ""

    def _tool_definition(self, tool_name: str) -> ExternalToolDefinition | None:
        return self.external_tool_aliases.get(safe_slug(tool_name))

    def _tool_adapter(self, tool_name: str) -> str:
        definition = self._tool_definition(tool_name)
        return definition.adapter if definition is not None and definition.enabled else ""

    def _load_external_tool_overrides(self) -> dict[str, dict[str, Any]]:
        return {definition.id: definition.config for definition in self.external_tool_definitions}

    def _truncate_text(self, value: Any, limit: int) -> str:
        text = str(value or "").strip().replace("\n", " ")
        if limit <= 0 or len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _configured_term_present(self, text: str, term: str) -> bool:
        clean = str(term or "").strip().lower()
        if not clean:
            return False
        lowered = (text or "").lower()
        if re.search(r"[^a-z0-9_+-]", clean):
            return clean in lowered
        return bool(re.search(rf"(?<![a-z0-9_+-]){re.escape(clean)}(?![a-z0-9_+-])", lowered))

    def _prefer_dynamic_planner(self, query: str) -> bool:
        terms = _clean_list(self.dynamic_planning_config.get("prefer_planner_terms"))
        if not terms:
            return False
        return any(self._configured_term_present(query, term) for term in terms)

    def _tool_schema_summary(self, schema_payload: Any, query: str = "") -> str:
        if not isinstance(schema_payload, dict):
            return ""
        parts: list[str] = []
        query_tokens = self._tokenize_query(query)
        for table_name, table_payload in schema_payload.items():
            if not isinstance(table_payload, dict):
                continue
            fields = table_payload.get("fields", [])
            note = str(table_payload.get("notes", "")).strip()
            field_list = ""
            if isinstance(fields, list):
                clean_fields = [
                    str(item).strip()
                    for item in fields
                    if str(item).strip() and not is_target_like_column(str(item))
                ]
                original_clean_field_count = len(clean_fields)
                if self.planner_schema_field_limit > 0:
                    scored = sorted(
                        enumerate(clean_fields),
                        key=lambda row: (
                            -self._token_overlap(query_tokens, row[1]),
                            row[0],
                        ),
                    )
                    selected_indexes = sorted(
                        index
                        for index, _field in scored[
                            : self.planner_schema_field_limit
                        ]
                    )
                    clean_fields = [clean_fields[index] for index in selected_indexes]
                    if original_clean_field_count > len(clean_fields):
                        clean_fields.append("...")
                field_list = ", ".join(clean_fields)
            summary = f"{table_name}({field_list})" if field_list else str(table_name)
            if note:
                summary += f" note: {note}"
            parts.append(summary)
        return " | ".join(parts)

    def _tool_examples_block(self, examples: Any, query: str = "") -> str:
        if not isinstance(examples, list):
            return ""
        if self.planner_example_limit <= 0:
            return ""
        blocks: list[str] = []
        emitted = 0
        query_tokens = self._tokenize_query(query)
        ranked_examples = sorted(
            enumerate(examples),
            key=lambda row: (
                -self._token_overlap(
                    query_tokens,
                    str(row[1].get("user_query", "")) if isinstance(row[1], dict) else "",
                ),
                row[0],
            ),
        )
        for _index, ex in ranked_examples:
            if emitted >= self.planner_example_limit:
                break
            if not isinstance(ex, dict):
                continue
            kind = str(ex.get("kind", "")).strip().lower()
            q = str(ex.get("user_query", "")).strip()
            if kind not in {"good", "bad"} or not q:
                continue
            label = "Good" if kind == "good" else "Bad"
            blocks.append(f"\n{label} example: {q}")
            arguments = ex.get("arguments")
            if isinstance(arguments, dict) and arguments:
                blocks.append(
                    f"\n{label} example args: "
                    f"{json.dumps(arguments, sort_keys=True, ensure_ascii=True)}"
                )
            why = str(ex.get("why", "")).strip()
            if why:
                blocks.append(f"\n{label} example why: {why}")
            emitted += 1
        return "".join(blocks)

    def _ranked_workflow_examples(self, query: str) -> list[tuple[int, dict[str, Any]]]:
        examples = self.dynamic_planning_config.get("workflow_examples")
        if not isinstance(examples, list):
            return []
        query_tokens = self._tokenize_query(query)

        def rank_key(row: tuple[int, Any]) -> tuple[int, int]:
            index, example = row
            if not isinstance(example, dict):
                return (0, index)
            texts = [
                str(example.get("user_query") or ""),
                str(example.get("description") or ""),
                " ".join(_clean_list(example.get("trigger_terms"))),
            ]
            return (-self._token_overlap(query_tokens, " ".join(texts)), index)

        ranked = [
            (index, example)
            for index, example in sorted(enumerate(examples), key=rank_key)
            if isinstance(example, dict)
        ]
        return ranked

    def _workflow_example_matches(
        self,
        query: str,
        query_tokens: set[str],
        example: dict[str, Any],
    ) -> bool:
        required_terms = _clean_list(example.get("required_terms"))
        if required_terms and not all(
            self._configured_term_present(query, term) for term in required_terms
        ):
            return False

        required_any_terms = _clean_list(example.get("required_any_terms"))
        if required_any_terms and not any(
            self._configured_term_present(query, term) for term in required_any_terms
        ):
            return False

        groups = example.get("required_term_groups")
        if isinstance(groups, list):
            for group in groups:
                terms = _clean_list(group)
                if terms and not any(
                    self._configured_term_present(query, term) for term in terms
                ):
                    return False

        texts = [
            str(example.get("user_query") or ""),
            str(example.get("description") or ""),
            " ".join(_clean_list(example.get("trigger_terms"))),
        ]
        return self._token_overlap(query_tokens, " ".join(texts)) > 0

    def _workflow_example_plan(self, query: str) -> ExecutionPlan | None:
        ranked_examples = self._ranked_workflow_examples(query)
        if not ranked_examples:
            return None
        query_tokens = self._tokenize_query(query)
        extracted = self._drop_empty_model_inputs(extract_nl_features(query))
        for _index, example in ranked_examples:
            if example.get("fallback_enabled") is False:
                continue
            steps = example.get("tool_calls") or example.get("steps")
            if not isinstance(steps, list) or not steps:
                continue
            if not self._workflow_example_tools_available(steps):
                continue
            if not self._workflow_example_matches(query, query_tokens, example):
                continue
            tool_calls: list[ToolCall] = []
            for row in steps:
                if not isinstance(row, dict):
                    continue
                tool_name = str(row.get("tool") or row.get("name") or "").strip()
                arguments = row.get("arguments", {})
                if isinstance(arguments, dict) and extracted:
                    arguments = {**arguments, **extracted}
                reasoning = str(row.get("reasoning") or "").strip()
                if tool_name:
                    tool_calls.append(
                        ToolCall(
                            name=tool_name,
                            arguments=arguments if isinstance(arguments, dict) else {},
                            reasoning=reasoning,
                        )
                    )
            if not tool_calls:
                continue
            reasoning = str(
                example.get("reasoning")
                or example.get("description")
                or "Configured workflow example."
            ).strip()
            return ExecutionPlan(
                steps=tool_calls,
                reasoning=reasoning,
                is_multi_step=len(tool_calls) > 1,
            )
        return None

    def _workflow_example_tools_available(self, steps: list[Any]) -> bool:
        """Return whether every configured workflow step can be executed now."""
        for row in steps:
            if not isinstance(row, dict):
                return False
            tool_name = str(row.get("tool") or row.get("name") or "").strip()
            if not tool_name:
                return False
            if self.registry.get(tool_name) is not None:
                continue
            definition = self._tool_definition(tool_name)
            if definition is None or not self._external_tool_available(definition):
                return False
        return True

    def _direct_response_rules(self) -> list[dict[str, Any]]:
        rules = self.direct_response_config.get("rules")
        if not isinstance(rules, list):
            rules = self.dynamic_planning_config.get("direct_responses")
        if not isinstance(rules, list):
            return []
        return [rule for rule in rules if isinstance(rule, dict)]

    def _direct_response_rule_matches(
        self,
        query: str,
        query_tokens: set[str],
        rule: dict[str, Any],
    ) -> bool:
        if rule.get("enabled") is False:
            return False
        for term in _clean_list(rule.get("blocked_terms")):
            if self._configured_term_present(query, term):
                return False
        trigger_terms = _clean_list(rule.get("trigger_terms"))
        if trigger_terms and not any(
            self._configured_term_present(query, term) for term in trigger_terms
        ):
            return False
        required_terms = _clean_list(rule.get("required_terms"))
        if any(
            not self._configured_term_present(query, term)
            for term in required_terms
        ):
            return False
        required_any_terms = _clean_list(rule.get("required_any_terms"))
        if required_any_terms and not any(
            self._configured_term_present(query, term) for term in required_any_terms
        ):
            return False
        groups = rule.get("required_term_groups")
        if isinstance(groups, list):
            for group in groups:
                terms = _clean_list(group)
                if terms and not any(
                    self._configured_term_present(query, term) for term in terms
                ):
                    return False
        if trigger_terms or required_terms or required_any_terms or isinstance(groups, list):
            return True
        return self._token_overlap(
            query_tokens,
            " ".join(
                [
                    str(rule.get("id") or ""),
                    str(rule.get("description") or ""),
                    " ".join(_clean_list(rule.get("terms"))),
                ]
            ),
        ) > 0

    def _direct_response_payload(self, query: str) -> dict[str, Any] | None:
        if self.direct_response_config.get("enabled") is False:
            return None
        query_tokens = self._tokenize_query(query)
        for rule in self._direct_response_rules():
            if not self._direct_response_rule_matches(query, query_tokens, rule):
                continue
            response = str(
                rule.get("response_markdown")
                or rule.get("response_template")
                or rule.get("answer")
                or ""
            ).strip()
            if not response:
                continue
            try:
                response = response.format(query=query)
            except Exception:
                pass
            return {
                "id": str(rule.get("id") or "").strip(),
                "reasoning": str(
                    rule.get("reasoning")
                    or rule.get("description")
                    or "Configured direct response."
                ).strip(),
                "markdown": response,
            }
        return None

    def _workflow_examples_block(self, query: str = "") -> str:
        examples = self.dynamic_planning_config.get("workflow_examples")
        if not isinstance(examples, list) or self.planner_workflow_example_limit <= 0:
            return ""

        blocks: list[str] = []
        emitted = 0
        query_tokens = self._tokenize_query(query)
        for _index, example in self._ranked_workflow_examples(query):
            if emitted >= self.planner_workflow_example_limit:
                break
            steps = example.get("tool_calls") or example.get("steps")
            if not isinstance(steps, list) or not steps:
                continue
            if not self._workflow_example_tools_available(steps):
                continue
            if not self._workflow_example_matches(query, query_tokens, example):
                continue
            compact: dict[str, Any] = {}
            for key in ("id", "user_query", "description", "reasoning"):
                value = example.get(key)
                if self._has_model_input_value(value):
                    compact[key] = value
            compact["tool_calls"] = steps
            blocks.append(json.dumps(compact, ensure_ascii=True, default=str))
            emitted += 1

        if not blocks:
            return ""
        return "\n\nWorkflow examples from config:\n" + "\n".join(blocks)

    def _planner_runtime_defaults(self, runtime: ModelRuntime) -> dict[str, Any]:
        defaults = runtime_prediction_defaults(runtime.model_id)
        if not isinstance(defaults, dict):
            defaults = {}
        runtime_defaults = runtime.defaults if isinstance(runtime.defaults, dict) else {}
        return _merge_dicts(defaults, runtime_defaults)

    def _planner_model_input_contract(self, runtime: ModelRuntime) -> str:
        defaults = self._planner_runtime_defaults(runtime)
        source = str(
            defaults.get("prediction_source")
            or defaults.get("feature_source")
            or defaults.get("inference_source")
            or ""
        ).strip().lower()
        if source != "feast_online":
            return ""

        feast_config = defaults.get("feast", {})
        if not isinstance(feast_config, dict):
            feast_config = {}

        raw_entity_keys = feast_config.get("entity_keys")
        entity_keys: list[str] = []
        if isinstance(raw_entity_keys, str) and raw_entity_keys.strip():
            entity_keys = [raw_entity_keys.strip()]
        elif isinstance(raw_entity_keys, list):
            entity_keys = [
                str(item).strip() for item in raw_entity_keys if str(item).strip()
            ]

        if not entity_keys:
            feature_service = str(feast_config.get("feature_service") or "").strip()
            if feature_service.endswith("_service"):
                inferred = feature_service[: -len("_service")].strip()
                if inferred:
                    entity_keys = [inferred]

        entity_summary = ", ".join(entity_keys) if entity_keys else "entity_id"
        feature_count = len(runtime.feature_names or [])
        if feature_count:
            return (
                f"{entity_summary} is an optional lookup key, not a model input; "
                f"provide any known model features and leave unknown values absent. "
                f"The resolver completes missing values from the feature store/database "
                f"or model stats across {feature_count} model features; do not pad "
                "missing features with null/NaN"
            )
        return (
            f"{entity_summary} is an optional lookup key, not a model input; "
            "do not invent missing features"
        )

    def _planner_model_horizon_contract(self, runtime: ModelRuntime) -> str:
        summary = describe_runtime_horizon(runtime)
        if not summary:
            return ""
        return f"Forecast horizon: {summary}"

    def _external_tool_available(self, definition: ExternalToolDefinition) -> bool:
        if not definition.enabled:
            return False
        if definition.adapter == "stockout_hybrid":
            return find_hybrid_stockout_regression_runtime(self.registry) is not None
        if not self.enable_external_fallback:
            return False
        if definition.adapter == "db_query":
            return self.db_fallback_enabled
        if definition.adapter == "simulation_query":
            return self.simulation_tool_enabled
        if definition.adapter == "digital_twin":
            return True
        if definition.adapter == "search":
            return self.search_fallback_enabled
        if definition.adapter == "llm_api":
            return bool(self.llm_api_url or self._llm is not None)
        return False

    def _direct_tool_available(self, definition: ExternalToolDefinition) -> bool:
        if not definition.enabled:
            return False
        if definition.adapter == "stockout_hybrid":
            return find_hybrid_stockout_regression_runtime(self.registry) is not None
        if definition.adapter == "db_query":
            return self.db_fallback_enabled
        if definition.adapter == "simulation_query":
            return self.simulation_tool_enabled
        if definition.adapter == "digital_twin":
            return True
        if definition.adapter == "search":
            return self.search_fallback_enabled
        if definition.adapter == "llm_api":
            return bool(self.llm_api_url or self._llm is not None)
        return False

    def _planner_external_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for definition in self.external_tool_definitions:
            if not self._external_tool_available(definition):
                continue
            config = definition.config
            tools.append(
                {
                    "name": definition.name or definition.id,
                    "description": self._truncate_text(
                        config.get("description"),
                        self.planner_tool_description_chars,
                    ),
                    "input_hints": str(config.get("input_hints") or "").strip(),
                    "examples": config.get("examples", []),
                    "schema": config.get("schema"),
                }
            )
        return tools

    def _ranked_planner_external_tools(self, query: str) -> list[dict[str, Any]]:
        """External tools ordered by relevance to the query (most relevant first).

        Used by the budgeter so that when the prompt must be trimmed to fit
        n_ctx, the query-relevant tools survive truncation. Mirrors the ranking
        in `_ranked_workflow_examples`; ties keep the original config order.
        """
        tools = self._planner_external_tools()
        query_tokens = self._tokenize_query(query)
        if not query_tokens or len(tools) <= 1:
            return tools

        def rank_key(row: tuple[int, dict[str, Any]]) -> tuple[int, int]:
            index, tool = row
            example_queries = " ".join(
                str(ex.get("user_query") or "")
                for ex in (tool.get("examples") or [])
                if isinstance(ex, dict)
            )
            texts = [
                str(tool.get("name") or ""),
                str(tool.get("description") or ""),
                str(tool.get("input_hints") or ""),
                example_queries,
            ]
            return (-self._token_overlap(query_tokens, " ".join(texts)), index)

        return [tool for _index, tool in sorted(enumerate(tools), key=rank_key)]

    def _build_tools_description(
        self,
        allowed_models: list[ModelRuntime],
        *,
        query: str = "",
        max_external_tools: int | None = None,
    ) -> str:
        rows: list[str] = []
        external_tools = self._ranked_planner_external_tools(query)
        if max_external_tools is not None:
            external_tools = external_tools[: max(0, max_external_tools)]
        for tool in external_tools:
            block = "Tool: {name}\nDescription: {description}\nInput hints: {input_hints}".format(
                **tool
            )
            schema_summary = self._tool_schema_summary(tool.get("schema"), query=query)
            if schema_summary:
                block += f"\nSchema: {schema_summary}"
            block += self._tool_examples_block(tool.get("examples"), query=query)
            rows.append(block)
        for rt in allowed_models:
            if self._hide_stockout_prediction_models() and self._is_stockout_prediction_runtime(rt):
                continue
            features = rt.feature_names[: self.planner_max_features]
            feature_hint = ", ".join(features)
            if len(rt.feature_names) > self.planner_max_features:
                feature_hint += ", ..."
            desc = self._truncate_text(
                rt.description,
                self.planner_model_description_chars,
            )
            example_block = self._tool_examples_block(rt.examples, query=query)
            block = (
                f"Tool: {rt.model_id}\nDescription: {desc}\nInput hints: {feature_hint}"
            )
            contract = self._planner_model_input_contract(rt)
            if contract:
                block += f"\nInput contract: {contract}"
            horizon_contract = self._planner_model_horizon_contract(rt)
            if horizon_contract:
                block += f"\n{horizon_contract}"
            block += example_block
            rows.append(block)
        return "\n\n".join(rows)

    def _select_planning_models(
        self, query: str, allowed_models: list[ModelRuntime], top_k: int
    ) -> list[ModelRuntime]:
        if self.simulation_tool_enabled and is_simulation_query(query):
            return []
        if self._hide_stockout_prediction_models():
            allowed_models = [
                runtime
                for runtime in allowed_models
                if not self._is_stockout_prediction_runtime(runtime)
            ]
        ranked = route_models(query, allowed_models, top_k=len(allowed_models))
        desired = max(1, min(self.planner_max_tools, max(1, top_k)))
        return ranked[:desired]

    def _row_scoring_config(self) -> dict[str, Any]:
        value = self.dynamic_planning_config.get("row_scoring")
        return dict(value) if isinstance(value, dict) else {}

    def _row_source_config(self) -> dict[str, Any]:
        value = self._row_scoring_config().get("row_source")
        return dict(value) if isinstance(value, dict) else {}

    def _query_requests_row_scoring(self, query: str) -> bool:
        config = self._row_scoring_config()
        terms = _clean_list(config.get("trigger_terms"))
        return bool(terms) and any(
            self._configured_term_present(query, term) for term in terms
        )

    def _plan_has_model_step(
        self,
        plan: ExecutionPlan,
        allowed_models: list[ModelRuntime],
    ) -> bool:
        allowed_ids = {runtime.model_id for runtime in allowed_models}
        return any(step.name in allowed_ids for step in plan.steps)

    def _step_row_output_key(self, step: ToolCall) -> str:
        config = self._row_scoring_config()
        source_keys = _clean_list(config.get("source_output_keys"))
        if not source_keys:
            return ""
        definition = self._tool_definition(step.name)
        if definition is None:
            return ""
        output_schema = definition.config.get("output_schema")
        properties = (
            output_schema.get("properties")
            if isinstance(output_schema, dict)
            else {}
        )
        if not isinstance(properties, dict):
            properties = {}
        for key in source_keys:
            if key in properties:
                return key
        return ""

    def _plan_has_row_output_step(self, plan: ExecutionPlan) -> bool:
        return any(self._step_row_output_key(step) for step in plan.steps)

    def _configured_row_source_tool_id(self) -> str:
        config = self._row_source_config()
        for adapter in _clean_list(config.get("tool_adapters")):
            tool_id = self._available_tool_id_for_adapter(adapter)
            if tool_id:
                return tool_id
        return ""

    def _configured_row_source_arguments(self) -> dict[str, Any]:
        config = self._row_source_config()
        arguments: dict[str, Any] = {}
        argument_defaults = config.get("argument_defaults")
        if isinstance(argument_defaults, dict):
            arguments.update(argument_defaults)

        filters = arguments.get("filters")
        filters = dict(filters) if isinstance(filters, dict) else {}
        filter_defaults = config.get("filter_defaults")
        if isinstance(filter_defaults, dict):
            for key, value in filter_defaults.items():
                if not self._has_model_input_value(filters.get(key)):
                    filters[key] = value
        if filters:
            arguments["filters"] = filters
        return arguments

    def _append_configured_row_source_step(
        self,
        query: str,
        plan: ExecutionPlan,
        allowed_models: list[ModelRuntime],
    ) -> ExecutionPlan:
        if not self._row_source_config() or not self._query_requests_row_scoring(query):
            return plan
        if self._plan_has_row_output_step(plan):
            return plan
        if self._plan_has_model_step(plan, allowed_models):
            return plan

        tool_id = self._configured_row_source_tool_id()
        if not tool_id:
            return plan

        reasoning = str(self._row_source_config().get("reasoning") or "").strip()
        steps = [
            *plan.steps,
            ToolCall(
                name=tool_id,
                arguments=self._configured_row_source_arguments(),
                reasoning=reasoning,
            ),
        ]
        return ExecutionPlan(
            steps=steps,
            reasoning=plan.reasoning,
            is_multi_step=len(steps) > 1,
        )

    def _runtime_matches_row_scoring(
        self,
        runtime: ModelRuntime,
        config: dict[str, Any],
    ) -> bool:
        tasks = set(_clean_list(config.get("model_tasks")))
        if tasks:
            task = infer_model_task(
                runtime.model_id,
                getattr(runtime, "model_type", ""),
                runtime.feature_names or [],
            )
            if task not in tasks:
                return False
        required_entity_keys = set(_clean_list(config.get("entity_keys")))
        if required_entity_keys:
            runtime_entity_keys = set(self._runtime_entity_keys(runtime))
            if not required_entity_keys.intersection(runtime_entity_keys):
                return False
        return True

    def _append_row_scoring_step(
        self,
        query: str,
        plan: ExecutionPlan,
        allowed_models: list[ModelRuntime],
    ) -> ExecutionPlan:
        config = self._row_scoring_config()
        if not config or not self._query_requests_row_scoring(query):
            return plan
        if self._plan_has_model_step(plan, allowed_models):
            return plan

        row_step_index = 0
        row_output_key = ""
        for index, step in enumerate(plan.steps, 1):
            output_key = self._step_row_output_key(step)
            if output_key:
                row_step_index = index
                row_output_key = output_key

        if not row_step_index or not row_output_key:
            return plan

        candidates = [
            runtime
            for runtime in allowed_models
            if self._runtime_matches_row_scoring(runtime, config)
        ]
        if not candidates:
            return plan
        runtime = route_models(query, candidates, top_k=1)[0]

        control_key = str(config.get("control_argument_key") or "").strip()
        if not control_key:
            control_key = sorted(self.model_control_argument_keys)[0]
        entity_keys = _clean_list(config.get("entity_keys")) or self._runtime_entity_keys(
            runtime
        )
        batch_config: dict[str, Any] = {
            "rows": f"$steps.{row_step_index}.output.{row_output_key}",
        }
        if entity_keys:
            batch_config["entity_key"] = entity_keys[0]
        for key in ("max_rows", "include_fields", "score_keys", "sort"):
            if key in config:
                batch_config[key] = config[key]

        steps = [
            *plan.steps,
            ToolCall(
                name=runtime.model_id,
                arguments={control_key: batch_config},
                reasoning="Dynamic row scoring selected a configured model for bounded rows.",
            ),
        ]
        return ExecutionPlan(
            steps=steps,
            reasoning=(
                f"{plan.reasoning} Added row scoring because the query requested ranked/best candidates."
            ).strip(),
            is_multi_step=len(steps) > 1,
        )

    def _row_scoring_batch_config(
        self,
        *,
        row_step_index: int,
        row_output_key: str,
        runtime: ModelRuntime,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        control_payload: dict[str, Any] = {
            "rows": f"$steps.{row_step_index}.output.{row_output_key}",
        }
        entity_keys = _clean_list(config.get("entity_keys")) or self._runtime_entity_keys(
            runtime
        )
        if entity_keys:
            control_payload["entity_key"] = entity_keys[0]
        for key in ("max_rows", "include_fields", "score_keys", "sort"):
            if key in config:
                control_payload[key] = config[key]
        return control_payload

    def _rewrite_model_only_row_scoring_plan(
        self,
        query: str,
        plan: ExecutionPlan,
        allowed_models: list[ModelRuntime],
    ) -> ExecutionPlan:
        config = self._row_scoring_config()
        if not config or not self._query_requests_row_scoring(query):
            return plan
        if self._plan_has_row_output_step(plan):
            return plan

        source_tool_id = self._configured_row_source_tool_id()
        source_keys = _clean_list(config.get("source_output_keys"))
        if not source_tool_id or not source_keys:
            return plan

        allowed_by_id = {runtime.model_id: runtime for runtime in allowed_models}
        rewritten_steps: list[ToolCall] = []
        inserted_source = False
        changed = False
        for step in plan.steps:
            runtime = allowed_by_id.get(step.name)
            if (
                runtime is None
                or not self._runtime_matches_row_scoring(runtime, config)
                or self._model_batch_config(step.arguments) is not None
            ):
                rewritten_steps.append(step)
                continue

            if not inserted_source:
                rewritten_steps.append(
                    ToolCall(
                        name=source_tool_id,
                        arguments=self._configured_row_source_arguments(),
                        reasoning=str(
                            self._row_source_config().get("reasoning") or ""
                        ).strip(),
                    )
                )
                inserted_source = True

            row_step_index = next(
                (
                    index
                    for index, candidate in enumerate(rewritten_steps, 1)
                    if candidate.name == source_tool_id
                ),
                len(rewritten_steps),
            )
            control_key = str(config.get("control_argument_key") or "").strip()
            if not control_key:
                control_key = sorted(self.model_control_argument_keys)[0]
            rewritten_steps.append(
                ToolCall(
                    name=runtime.model_id,
                    arguments={
                        control_key: self._row_scoring_batch_config(
                            row_step_index=row_step_index,
                            row_output_key=source_keys[0],
                            runtime=runtime,
                            config=config,
                        )
                    },
                    reasoning="Configured row scoring converted this model call to bounded batch scoring.",
                )
            )
            changed = True

        if not changed:
            return plan
        return ExecutionPlan(
            steps=rewritten_steps,
            reasoning=(
                f"{plan.reasoning} Added configured row lookup before batch scoring."
            ).strip(),
            is_multi_step=len(rewritten_steps) > 1,
        )

    def _repair_plan_for_row_scoring(
        self,
        query: str,
        plan: ExecutionPlan,
        allowed_models: list[ModelRuntime],
    ) -> ExecutionPlan:
        rewritten = self._rewrite_model_only_row_scoring_plan(
            query,
            plan,
            allowed_models,
        )
        if rewritten is not plan:
            return rewritten
        with_source = self._append_configured_row_source_step(
            query,
            plan,
            allowed_models,
        )
        return self._append_row_scoring_step(query, with_source, allowed_models)

    def _result_output_value_from_paths(
        self,
        results: list[ExecutionResult],
        *,
        source_tool_adapters: list[str],
        output_paths: list[str],
    ) -> Any:
        source_adapters = {safe_slug(adapter) for adapter in source_tool_adapters}
        for result in reversed(results):
            if not result.success or not isinstance(result.output, dict):
                continue
            if source_adapters and safe_slug(
                self._tool_adapter(result.tool_name)
            ) not in source_adapters:
                continue
            for path in output_paths:
                value = self._lookup_path(
                    result.output,
                    [part for part in str(path or "").split(".") if part],
                )
                if self._has_model_input_value(value):
                    return value
        return None

    def _repair_row_source_arguments(
        self,
        query: str,
        tool_name: str,
        arguments: dict[str, Any],
        results: list[ExecutionResult],
    ) -> dict[str, Any]:
        config = self._row_source_config()
        if not config or not self._query_requests_row_scoring(query):
            return arguments
        definition = self._tool_definition(tool_name)
        if definition is None:
            return arguments
        tool_adapters = {safe_slug(adapter) for adapter in _clean_list(config.get("tool_adapters"))}
        if tool_adapters and safe_slug(definition.adapter) not in tool_adapters:
            return arguments

        repaired = dict(arguments or {})
        argument_defaults = config.get("argument_defaults")
        if isinstance(argument_defaults, dict):
            for key, value in argument_defaults.items():
                if not self._has_model_input_value(repaired.get(key)):
                    repaired[key] = value

        filters = repaired.get("filters")
        filters = dict(filters) if isinstance(filters, dict) else {}
        remove_filters = {safe_slug(field) for field in _clean_list(config.get("filter_remove"))}
        for field in list(filters):
            if safe_slug(field) in remove_filters:
                filters.pop(field, None)

        filter_defaults = config.get("filter_defaults")
        if isinstance(filter_defaults, dict):
            for key, value in filter_defaults.items():
                if not self._has_model_input_value(filters.get(key)):
                    filters[key] = value

        dependencies = config.get("filter_dependencies")
        if isinstance(dependencies, list):
            for dependency in dependencies:
                if not isinstance(dependency, dict):
                    continue
                field = str(dependency.get("field") or "").strip()
                if not field:
                    continue
                overwrite = bool(dependency.get("overwrite"))
                if not overwrite and self._has_model_input_value(filters.get(field)):
                    continue
                value = self._result_output_value_from_paths(
                    results,
                    source_tool_adapters=_clean_list(
                        dependency.get("source_tool_adapters")
                    ),
                    output_paths=_clean_list(dependency.get("output_paths")),
                )
                if self._has_model_input_value(value):
                    filters[field] = value

        if filters:
            repaired["filters"] = filters
        return repaired

    def _rank_forecast_horizon_models(
        self,
        query: str,
        allowed_models: list[ModelRuntime],
        horizon: ForecastHorizon | None,
        features: dict[str, Any] | None = None,
    ) -> list[HorizonModelScore]:
        if horizon is None:
            return []
        compatible_models = [
            runtime
            for runtime in allowed_models
            if self._runtime_entity_compatible_for_horizon(query, runtime, features)
        ]
        return rank_model_scores_for_horizon(
            query,
            compatible_models,
            horizon,
            features=features,
            target=forecast_selection_target_for_query(
                query,
                features,
                default_target=self.forecast_selection_target,
            ),
            primary_metric=self.forecast_selection_metric,
            metric_payloads_fn=self._runtime_forecast_metric_payloads,
            scope_score_fn=lambda runtime: self._model_scope_score(query, runtime),
        )

    def _runtime_entity_compatible_for_horizon(
        self,
        query: str,
        runtime: ModelRuntime,
        features: dict[str, Any] | None,
    ) -> bool:
        defaults = runtime_prediction_defaults(runtime.model_id)
        base_defaults = defaults
        if isinstance(runtime.defaults, dict):
            defaults = self._deep_merge_dict(defaults, runtime.defaults)
        base_feast = base_defaults.get("feast")
        merged_feast = defaults.get("feast")
        if isinstance(base_feast, dict) and isinstance(merged_feast, dict):
            base_entity_keys = [
                str(key).strip()
                for key in base_feast.get("entity_keys", [])
                if str(key).strip()
            ]
            merged_entity_keys = [
                str(key).strip()
                for key in merged_feast.get("entity_keys", [])
                if str(key).strip()
            ]
            if base_entity_keys:
                merged_feast["entity_keys"] = [
                    *merged_entity_keys,
                    *[key for key in base_entity_keys if key not in merged_entity_keys],
                ]
        feast_config = defaults.get("feast")
        if not isinstance(feast_config, dict):
            return True
        required_keys = {
            str(key).strip()
            for key in feast_config.get("entity_keys", [])
            if str(key).strip()
        }
        optional_keys = {
            str(key).strip()
            for key in feast_config.get("optional_entity_keys", [])
            if str(key).strip()
        }
        required_keys = required_keys - optional_keys
        if not required_keys:
            return True
        extracted = extract_nl_features(query)
        available = {
            **{
                self._canonical_entity_key(key): value
                for key, value in extracted.items()
            },
            **{
                self._canonical_entity_key(key): value
                for key, value in (features or {}).items()
            },
        }
        if all(
            key in available and self._has_model_input_value(available.get(key))
            for key in required_keys
        ):
            return True
        missing_keys = {
            key
            for key in required_keys
            if key not in available
            or not self._has_model_input_value(available.get(key))
        }
        if missing_keys == {"blood_type"} and self._has_model_input_value(
            available.get("hospital_id")
        ):
            return True
        available_entity_keys = {
            key
            for key, value in available.items()
            if self._has_model_input_value(value)
            and (key.endswith("_id") or key in {"entity_id"})
        }
        return not available_entity_keys

    def _canonical_entity_key(self, key: Any) -> str:
        clean = str(key or "").strip()
        return {
            "hospital": "hospital_id",
            "donor": "donor_id",
            "supply": "supply_id",
            "component": "blood_type",
            "blood_component": "blood_type",
            "blood_product_type": "blood_type",
        }.get(clean, clean)

    def _deep_merge_dict(self, base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge_dict(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _clean_metric_model_id(self, runtime: ModelRuntime) -> str:
        model_id = str(runtime.model_id or "").strip()
        return model_id.split("_champion")[0].split("_@")[0]

    def _metric_payload_with_source(
        self,
        payload: Any,
        source: str,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict) or not payload:
            return None
        row = dict(payload)
        row.setdefault("__metric_source", source)
        return row

    def _runtime_metric_artifact_payload(self, runtime: ModelRuntime) -> dict[str, Any] | None:
        clean_id = self._clean_metric_model_id(runtime)
        if clean_id in self._forecast_metrics_cache:
            return self._forecast_metrics_cache[clean_id]

        candidates = [
            ML_BACKEND_DIR / "artifacts" / "models" / clean_id / "metrics.json",
            BACKEND_ML_DIR / "artifacts" / "models" / clean_id / "metrics.json",
        ]
        for path in candidates:
            try:
                if not path.exists():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            row = self._metric_payload_with_source(payload, str(path))
            self._forecast_metrics_cache[clean_id] = row
            return row

        self._forecast_metrics_cache[clean_id] = None
        return None

    def _runtime_model_stats_metric_payload(
        self, runtime: ModelRuntime
    ) -> dict[str, Any] | None:
        clean_id = self._clean_metric_model_id(runtime)
        try:
            from ml.models import ModelStats

            stats_row = ModelStats.objects.filter(model_id=clean_id).first()
        except Exception:
            return None
        if stats_row is None:
            return None
        return self._metric_payload_with_source(
            getattr(stats_row, "stats", None),
            f"model_stats:{clean_id}",
        )

    def _runtime_forecast_metric_payloads(
        self, runtime: ModelRuntime
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        defaults = runtime.defaults if isinstance(runtime.defaults, dict) else {}

        for key in (
            "horizon_metrics",
            "validation_metrics",
            "selection_metrics",
            "metrics",
        ):
            row = self._metric_payload_with_source(
                defaults.get(key),
                f"runtime.defaults.{key}",
            )
            if row is not None:
                payloads.append(row)

        selection = defaults.get("forecast_selection")
        if isinstance(selection, dict):
            for key in ("horizon_metrics", "validation_metrics", "metrics"):
                row = self._metric_payload_with_source(
                    selection.get(key),
                    f"runtime.defaults.forecast_selection.{key}",
                )
                if row is not None:
                    payloads.append(row)

        routing = defaults.get("horizon_routing")
        if isinstance(routing, dict):
            for key in ("horizon_metrics", "validation_metrics", "metrics"):
                row = self._metric_payload_with_source(
                    routing.get(key),
                    f"runtime.defaults.horizon_routing.{key}",
                )
                if row is not None:
                    payloads.append(row)

        for row in (
            self._runtime_model_stats_metric_payload(runtime),
            self._runtime_metric_artifact_payload(runtime),
        ):
            if row is not None:
                payloads.append(row)

        return payloads

    def _safe_generation_tokens(
        self, prompt: str, requested: int, reserve: int = 64
    ) -> int:
        n_ctx = int(self._llm_n_ctx or DEFAULT_PLANNER_N_CTX)
        prompt_tokens = self._prompt_token_count(prompt)
        available = max(1, n_ctx - prompt_tokens - reserve)
        return max(1, min(requested, available))

    def _prompt_token_count(self, prompt: str) -> int:
        if self._llm:
            try:
                return len(self._llm.tokenize(prompt.encode("utf-8")))
            except Exception:
                pass
        return max(1, len(prompt) // 2)

    def _build_plan_prompt(
        self,
        query: str,
        allowed_models: list[ModelRuntime],
        top_k: int,
        *,
        max_external_tools: int | None = None,
    ) -> str:
        tools_description = self._build_tools_description(
            allowed_models, query=query, max_external_tools=max_external_tools
        )
        workflow_examples = self._workflow_examples_block(query)
        logger.debug("Available tools: %s", tools_description)
        template = str(self.prompt_templates.get("planning") or "").strip()
        if not template:
            raise RuntimeError("Missing orchestrator planning prompt template in config.")
        return template.format(
            tools_description=tools_description,
            workflow_examples=workflow_examples,
            query=query,
        )

    def _plan_prompt_fits(self, prompt: str, n_ctx: int) -> bool:
        return (
            self._prompt_token_count(prompt)
            + self.planner_completion_tokens
            + self.planner_context_reserve_tokens
        ) <= n_ctx

    def _build_budgeted_plan_prompt(
        self,
        query: str,
        allowed_models: list[ModelRuntime],
        top_k: int,
    ) -> tuple[str, int]:
        original_example_limit = self.planner_example_limit
        original_schema_limit = self.planner_schema_field_limit
        original_feature_limit = self.planner_max_features
        original_tool_description_chars = self.planner_tool_description_chars
        original_model_description_chars = self.planner_model_description_chars
        original_workflow_example_limit = self.planner_workflow_example_limit
        n_ctx = int(self._llm_n_ctx or DEFAULT_PLANNER_N_CTX)

        def _accept(prompt: str) -> tuple[str, int]:
            return prompt, self._safe_generation_tokens(
                prompt,
                requested=self.planner_completion_tokens,
                reserve=self.planner_context_reserve_tokens,
            )

        try:
            # Stage 1 — shrink per-item detail (examples, schema, descriptions).
            compact_levels = [
                (
                    original_example_limit,
                    original_schema_limit,
                    original_feature_limit,
                    original_workflow_example_limit,
                ),
                (
                    min(original_example_limit, 2),
                    max(8, original_schema_limit // 2) if original_schema_limit else 0,
                    max(4, original_feature_limit // 2),
                    min(original_workflow_example_limit, 1),
                ),
                (
                    1,
                    max(6, original_schema_limit // 3) if original_schema_limit else 0,
                    4,
                    min(original_workflow_example_limit, 1),
                ),
                (0, 0, 3, 0),
            ]
            for example_limit, schema_limit, feature_limit, workflow_limit in compact_levels:
                self.planner_example_limit = example_limit
                self.planner_schema_field_limit = schema_limit
                self.planner_max_features = feature_limit
                self.planner_workflow_example_limit = workflow_limit
                self.planner_tool_description_chars = min(
                    original_tool_description_chars,
                    260,
                )
                self.planner_model_description_chars = min(
                    original_model_description_chars,
                    120,
                )
                prompt = self._build_plan_prompt(query, allowed_models, top_k)
                if self._plan_prompt_fits(prompt, n_ctx):
                    return _accept(prompt)

            # Stage 2 — per-item detail is already maximally compacted; now reduce
            # the NUMBER of tool/model blocks, dropping least-relevant first so the
            # query-relevant tools survive. Drop models before external tools, since
            # the external tools (e.g. stockout_hybrid, db_tool) are the primary
            # planner targets. Slicing creates new lists — never mutates the caller.
            for n_models in range(len(allowed_models) - 1, -1, -1):
                prompt = self._build_plan_prompt(
                    query, allowed_models[:n_models], top_k
                )
                if self._plan_prompt_fits(prompt, n_ctx):
                    return _accept(prompt)

            total_tools = len(self._planner_external_tools())
            for n_tools in range(total_tools - 1, 0, -1):
                prompt = self._build_plan_prompt(
                    query, [], top_k, max_external_tools=n_tools
                )
                if self._plan_prompt_fits(prompt, n_ctx):
                    return _accept(prompt)

            # Stage 3 — even a single tool with no models will not fit. Signal a
            # clean fallback rather than returning an over-budget prompt that would
            # make llama.cpp raise a raw "exceed context window" error.
            raise PlanPromptTooLargeError(
                "planning prompt cannot fit n_ctx after maximal compaction"
            )
        finally:
            self.planner_example_limit = original_example_limit
            self.planner_schema_field_limit = original_schema_limit
            self.planner_max_features = original_feature_limit
            self.planner_tool_description_chars = original_tool_description_chars
            self.planner_model_description_chars = original_model_description_chars
            self.planner_workflow_example_limit = original_workflow_example_limit

    def _parse_json_object(self, raw: str) -> dict[str, Any]:
        candidate = raw.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", candidate)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}

    def _fallback_plan(
        self,
        query: str,
        allowed_models: list[ModelRuntime],
        top_k: int,
        *,
        forced_model_ids: list[str] | None = None,
        provided_features: dict[str, Any] | None = None,
        cause: str = "orchestrator LLM unavailable",
        allow_stockout_shortcut_when_prefer_dynamic: bool = False,
    ) -> ExecutionPlan:
        reason_prefix = f"Fallback: {cause}"
        extracted = {
            **extract_nl_features(query),
            **self._drop_empty_model_inputs(provided_features or {}),
        }
        selected = route_models(
            query, allowed_models, top_k=max(1, min(top_k, len(allowed_models)))
        )
        in_scope_models = [
            rt for rt in selected if self._is_query_in_model_scope(query, rt)
        ]
        if self.fallback_model_step_limit:
            in_scope_models = in_scope_models[: self.fallback_model_step_limit]
        prefer_dynamic_planner = self._prefer_dynamic_planner(query)

        if forced_model_ids and selected:
            steps = [
                ToolCall(
                    name=rt.model_id,
                    arguments=dict(extracted),
                    reasoning="Heuristic fallback: explicit model selection routed directly.",
                )
                for rt in selected
            ]
            return ExecutionPlan(
                steps=steps,
                reasoning=f"{reason_prefix}, routed to the explicitly selected model.",
                is_multi_step=len(steps) > 1,
            )

        if self.enable_external_fallback:
            workflow_plan = self._workflow_example_plan(query)
            if workflow_plan is not None:
                return ExecutionPlan(
                    steps=workflow_plan.steps,
                    reasoning=f"{reason_prefix}, routed by configured workflow example.",
                    is_multi_step=workflow_plan.is_multi_step,
                )
            if self.simulation_tool_enabled and is_simulation_query(query):
                return ExecutionPlan(
                    steps=[
                        ToolCall(
                            name=self.simulation_tool_name,
                            arguments={"query": query},
                            reasoning="Heuristic fallback: simulation scenario request routed to the simulation tool.",
                        )
                    ],
                    reasoning=f"{reason_prefix}, routed to simulation tool.",
                    is_multi_step=False,
                )
            if (
                (not prefer_dynamic_planner or allow_stockout_shortcut_when_prefer_dynamic)
                and self._is_stockout_hybrid_query(query, extracted)
            ):
                hybrid_runtime = self._select_stockout_hybrid_runtime(
                    query,
                    extracted,
                    in_scope_models or allowed_models,
                )
                if hybrid_runtime is not None and self.stockout_hybrid_tool_name:
                    return ExecutionPlan(
                        steps=[
                            ToolCall(
                                name=self.stockout_hybrid_tool_name,
                                arguments={
                                    **dict(extracted),
                                    "regression_model_id": hybrid_runtime.model_id,
                                },
                                reasoning="Heuristic fallback: stockout-risk request routed to hybrid stockout function.",
                            )
                        ],
                        reasoning=f"{reason_prefix}, routed to hybrid stockout function.",
                        is_multi_step=False,
                    )
                # Hybrid tool disabled: route the stockout-intent query to the
                # stockout prediction model directly rather than letting generic
                # ranking pick an adjacent forecast model. Re-select over the full
                # candidate set so a stockout model is found even when it fell
                # outside the generic in-scope ranking for this phrasing.
                stockout_runtime = hybrid_runtime or self._select_stockout_hybrid_runtime(
                    query,
                    extracted,
                    allowed_models,
                )
                if stockout_runtime is not None:
                    return ExecutionPlan(
                        steps=[
                            ToolCall(
                                name=stockout_runtime.model_id,
                                arguments=dict(extracted),
                                reasoning="Heuristic fallback: stockout-risk request routed to the stockout prediction model.",
                            )
                        ],
                        reasoning=f"{reason_prefix}, routed to the stockout prediction model.",
                        is_multi_step=False,
                    )
            if self.db_fallback_enabled and is_structured_db_query(query):
                return ExecutionPlan(
                    steps=[
                        ToolCall(
                            name=self.db_tool_name,
                            arguments={"query": query},
                            reasoning="Heuristic fallback: structured local-data query mapped to DB tool.",
                        )
                    ],
                    reasoning=f"{reason_prefix}, routed to DB tool.",
                    is_multi_step=False,
                )
            if in_scope_models:
                steps = [
                    ToolCall(
                        name=rt.model_id,
                        arguments=dict(extracted),
                        reasoning="Heuristic fallback: query matched model metadata/examples, so route to model inference.",
                    )
                    for rt in in_scope_models
                ]
                return ExecutionPlan(
                    steps=steps,
                    reasoning=f"{reason_prefix}, routed to the best in-scope model.",
                    is_multi_step=len(steps) > 1,
                )
            llm_tool_id = self._available_tool_id_for_adapter("llm_api")
            if llm_tool_id:
                return ExecutionPlan(
                    steps=[
                        ToolCall(
                            name=llm_tool_id,
                            arguments={"query": query},
                            reasoning="Heuristic fallback: external LLM answer.",
                        )
                    ],
                    reasoning=f"{reason_prefix}, routed to external LLM API.",
                    is_multi_step=False,
                )
        steps = [
            ToolCall(
                name=rt.model_id,
                arguments=dict(extracted),
                reasoning="Fallback selection from matching tool metadata.",
            )
            for rt in in_scope_models
        ]
        return ExecutionPlan(
            steps=steps,
            reasoning=(
                f"{reason_prefix}, no matching model metadata was found."
                if not steps
                else f"{reason_prefix}, routed by matching tool metadata."
            ),
            is_multi_step=len(steps) > 1,
        )

    def _plan_execution(
        self,
        query: str,
        allowed_models: list[ModelRuntime],
        top_k: int,
        forced_model_ids: list[str] | None = None,
        provided_features: dict[str, Any] | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[ExecutionPlan, str]:
        logger.debug("Planning steps for query: %s", query)
        extracted = {
            **extract_nl_features(query),
            **self._drop_empty_model_inputs(provided_features or {}),
        }
        prefer_dynamic_planner = (
            self._llm is not None
            and not forced_model_ids
            and self._prefer_dynamic_planner(query)
        )
        if (
            not prefer_dynamic_planner
            and not forced_model_ids
            and bool(self.stockout_hybrid_tool_name)
            and self._is_stockout_hybrid_query(query, extracted)
        ):
            hybrid_runtime = self._select_stockout_hybrid_runtime(
                query,
                extracted,
                allowed_models,
            )
            if hybrid_runtime is None:
                hybrid_runtime = find_hybrid_stockout_regression_runtime(self.registry)
        else:
            hybrid_runtime = None
        if hybrid_runtime is not None:
            plan = self._fallback_plan(
                query,
                allowed_models,
                top_k,
                forced_model_ids=forced_model_ids,
                provided_features=provided_features,
                cause="stockout request matched configured hybrid routing",
                allow_stockout_shortcut_when_prefer_dynamic=True,
            )
            plan = self._repair_plan_for_row_scoring(query, plan, allowed_models)
            planner_mode = "fallback"
            if plan.steps and self._is_stockout_hybrid_tool(plan.steps[0].name):
                planner_mode = "function_tool"
            elif any(self._is_external_fallback_step(step.name) for step in plan.steps):
                planner_mode = "external_fallback"
            return (
                plan,
                planner_mode,
            )
        if self._llm is None:
            if not self.allow_fallback:
                raise RuntimeError("Orchestrator LLM unavailable.")
            plan = self._fallback_plan(
                query,
                allowed_models,
                top_k,
                forced_model_ids=forced_model_ids,
                provided_features=provided_features,
                cause="orchestrator LLM unavailable",
                allow_stockout_shortcut_when_prefer_dynamic=True,
            )
            repaired = self._repair_plan_for_simulation_query(query, plan)
            repaired = self._repair_plan_for_row_scoring(query, repaired, allowed_models)
            return repaired, "fallback"
        # Pre-check: route simulation queries before LLM planner
        if (
            not forced_model_ids
            and self.enable_external_fallback
            and self.simulation_tool_enabled
            and self._simulation_pre_routing_enabled
            and is_simulation_query(query)
        ):
            return ExecutionPlan(
                steps=[
                    ToolCall(
                        name=self.simulation_tool_name,
                        arguments={"query": query},
                        reasoning="Pre-planner routing: simulation scenario query.",
                    )
                ],
                reasoning="Routed to simulation tool (what-if/scenario query detected).",
                is_multi_step=False,
            ), "function_tool"
        planning_models = self._select_planning_models(
            query, allowed_models, top_k=top_k
        )
        planner_candidates = [
            {
                "index": index,
                "tool": str(tool.get("id") or tool.get("name") or ""),
                "reasoning": str(tool.get("adapter") or "configured tool"),
            }
            for index, tool in enumerate(self._planner_external_tools(), 1)
        ]
        offset = len(planner_candidates)
        planner_candidates.extend(
            {
                "index": offset + index,
                "tool": runtime.model_id,
                "reasoning": str(getattr(runtime, "model_type", "") or "model"),
            }
            for index, runtime in enumerate(planning_models, 1)
        )
        self._emit_run_event(
            event_callback,
            {
                "type": "planner_context",
                "phase": "planning",
                "reasoning": "LLM is selecting tools and arguments from the configured registry.",
                "steps": planner_candidates,
            },
        )
        try:
            prompt, max_tokens = self._build_budgeted_plan_prompt(
                query, planning_models, top_k=min(top_k, len(planning_models))
            )
        except PlanPromptTooLargeError as exc:
            logger.warning("Planner prompt too large for context window: %s", exc)
            if not self.allow_fallback:
                raise RuntimeError(f"Orchestrator LLM planning failed: {exc}") from exc
            plan = self._fallback_plan(
                query,
                allowed_models,
                top_k,
                forced_model_ids=forced_model_ids,
                provided_features=provided_features,
                cause="planning prompt exceeded model context after compaction",
            )
            repaired = self._repair_plan_for_simulation_query(query, plan)
            repaired = self._repair_plan_for_row_scoring(query, repaired, allowed_models)
            planner_mode = (
                "external_fallback"
                if any(
                    self._is_external_fallback_step(step.name)
                    for step in repaired.steps
                )
                else "fallback"
            )
            return repaired, planner_mode
        logger.debug("Generating LLM plan.")
        try:
            text = self._generate_llm_text(
                prompt,
                max_tokens=max_tokens,
                temperature=0.1,
            )
        except Exception as exc:
            logger.warning("Planner generation failed: %s", exc)
            if not self.allow_fallback:
                raise RuntimeError(f"Orchestrator LLM planning failed: {exc}") from exc
            plan = self._fallback_plan(
                query,
                allowed_models,
                top_k,
                forced_model_ids=forced_model_ids,
                provided_features=provided_features,
                cause=f"orchestrator LLM planning failed: {exc}",
            )
            repaired = self._repair_plan_for_simulation_query(query, plan)
            repaired = self._repair_plan_for_row_scoring(query, repaired, allowed_models)
            planner_mode = (
                "external_fallback"
                if any(
                    self._is_external_fallback_step(step.name)
                    for step in repaired.steps
                )
                else "fallback"
            )
            return repaired, planner_mode
        logger.debug("Raw planner LLM output: %s", text)
        payload = self._parse_json_object(text)
        steps_payload = payload.get("tool_calls") or payload.get("steps")
        global_reasoning = str(
            payload.get("reasoning")
            or payload.get("thought")
            or "Generated by orchestrator LLM."
        )
        steps: list[ToolCall] = []
        if isinstance(steps_payload, list):
            for row in steps_payload:
                if not isinstance(row, dict):
                    continue
                tool_name = str(row.get("tool") or row.get("name") or "").strip()
                arguments = row.get("arguments", {})
                reasoning = str(row.get("reasoning") or global_reasoning).strip()
                if tool_name:
                    steps.append(
                        ToolCall(
                            name=tool_name, arguments=arguments, reasoning=reasoning
                        )
                    )
        if not steps:
            logger.debug("Empty plan. Using fallback.")
            if not self.allow_fallback:
                raise RuntimeError("Empty plan.")
            plan = self._fallback_plan(
                query,
                allowed_models,
                top_k,
                forced_model_ids=forced_model_ids,
                provided_features=provided_features,
                cause="orchestrator LLM returned no usable plan",
            )
            repaired = self._repair_plan_for_simulation_query(query, plan)
            repaired = self._repair_plan_for_row_scoring(query, repaired, allowed_models)
            planner_mode = (
                "external_fallback"
                if any(self._is_external_fallback_step(step.name) for step in repaired.steps)
                else "fallback"
            )
            return repaired, planner_mode
        logger.debug("Plan accepted: %s step(s).", len(steps))
        execution_plan = ExecutionPlan(
            steps=steps, reasoning=global_reasoning, is_multi_step=len(steps) > 1
        )
        repaired = self._repair_plan_for_simulation_query(query, execution_plan)
        repaired = self._repair_plan_for_row_scoring(query, repaired, allowed_models)
        return repaired, "llm"

    # ══════════════════════════════════════════════════════
    #  SUMMARIZER
    # ══════════════════════════════════════════════════════

    def _prediction_name_for_summary(self, output: dict[str, Any]) -> str:
        return str(output.get("prediction_name") or "").strip()

    def _prediction_label_for_summary(self, output: dict[str, Any]) -> str:
        name = self._prediction_name_for_summary(output)
        return name.replace("_", " ") if name else "prediction"

    def _prediction_value_for_summary(self, output: dict[str, Any]) -> Any:
        name = self._prediction_name_for_summary(output)
        if name and name in output:
            return output.get(name)
        return output.get("prediction")

    def _compact_stockout_output(self, output: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(output, dict):
            return {}
        keys = (
            "location",
            "blood_component",
            "blood_group",
            "risk_1_7_days",
            "risk_8_30_days",
            "risk_30_plus_days",
            "estimated_days_until_stockout",
            "stockout_probability",
            "horizon",
            "risk_level",
            "recommended_action",
            "confidence",
            "worst_component",
            "worst_blood_group",
            "regression_model_id",
        )
        compact = {key: output[key] for key in keys if key in output}
        rows = output.get("component_predictions")
        if isinstance(rows, list):
            compact["component_count"] = len([row for row in rows if isinstance(row, dict)])
            worst_component = compact.get("worst_component")
            worst_blood_group = compact.get("worst_blood_group")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if (
                    worst_component
                    and row.get("blood_component") != worst_component
                ):
                    continue
                if worst_blood_group and row.get("blood_group") != worst_blood_group:
                    continue
                compact["worst_component_prediction"] = {
                    key: row[key]
                    for key in (
                        "blood_component",
                        "blood_group",
                        "estimated_days_until_stockout",
                        "stockout_probability",
                        "horizon",
                        "risk_level",
                        "recommended_action",
                    )
                    if key in row
                }
                break
        return compact

    def _execution_result_payload(self, result: ExecutionResult) -> dict[str, Any]:
        row: dict[str, Any] = {"tool": result.tool_name, "success": result.success}
        if result.inputs:
            row["inputs"] = result.inputs
        if result.error:
            row["error"] = result.error
        if (
            result.success
            and self._is_stockout_hybrid_tool(result.tool_name)
            and isinstance(result.output, dict)
            and result.output.get("hybrid_prediction") is True
        ):
            row["output"] = self._compact_stockout_output(result.output)
            row["summary"] = self._result_value_for_summary(result.output)
            return row
        row["output"] = result.output
        return row

    def _result_value_for_summary(self, output: dict[str, Any]) -> str:
        if not isinstance(output, dict):
            return str(output)
        if output.get("hybrid_prediction") is True:
            probability = output.get("stockout_probability")
            days = output.get("estimated_days_until_stockout")
            horizon = output.get("horizon")
            risk_level = output.get("risk_level")
            action = output.get("recommended_action")
            parts = []
            if days is not None:
                parts.append(f"estimated days until stockout: {days}")
            if isinstance(probability, (int, float)):
                parts.append(
                    f"stockout probability: {float(probability):.4f} "
                    f"({float(probability) * 100:.1f}%)"
                )
            elif probability is not None:
                parts.append(f"stockout probability: {probability}")
            if horizon:
                parts.append(f"horizon: {horizon}")
            if risk_level:
                parts.append(f"risk level: {risk_level}")
            if action:
                parts.append(f"recommended action: {action}")
            worst_component = output.get("worst_component")
            worst_blood_group = output.get("worst_blood_group")
            if worst_component:
                label = str(worst_component)
                if worst_blood_group:
                    label = f"{label} {worst_blood_group}"
                parts.append(f"highest-risk component: {label}")
            return "; ".join(parts) or str(output)
        component_forecasts = output.get("component_forecasts")
        if isinstance(component_forecasts, list) and component_forecasts:
            parts = []
            label = self._prediction_label_for_summary(output)
            for row in component_forecasts:
                if not isinstance(row, dict):
                    continue
                component = str(row.get("component") or row.get("blood_type") or "").strip()
                value = self._prediction_value_for_summary(row)
                unit = str(row.get("prediction_unit") or output.get("prediction_unit") or "days").strip()
                if component and isinstance(value, (int, float)):
                    parts.append(f"{component}: {float(value):.2f} {unit}".strip())
                elif component and value not in {None, ""}:
                    parts.append(f"{component}: {value} {unit}".strip())
            return f"component forecasts ({label}): " + ", ".join(parts)
        if "probability" in output:
            prob = output.get("probability")
            if isinstance(prob, (int, float)):
                return f"{float(prob):.4f} ({float(prob) * 100:.1f}%)"
            return str(prob)
        if "prediction" in output:
            prediction = output.get("prediction")
            unit = str(output.get("prediction_unit") or "").strip()
            name = str(output.get("prediction_name") or "").strip()
            if isinstance(prediction, (int, float)) and unit:
                label = name.replace("_", " ") if name else "prediction"
                component = str(output.get("component") or output.get("blood_type") or "").strip()
                prefix = f"{component}: " if component else ""
                return f"{prefix}{float(prediction):.2f} {unit} ({label})"
            return str(prediction)
        answer = output.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        rows = output.get("results")
        if isinstance(rows, list):
            if not rows:
                return "0 results"
            preview = json.dumps(rows, ensure_ascii=True, default=str)
            return f"{len(rows)} result(s): {preview}"
        rows = output.get("rows")
        if isinstance(rows, list):
            if not rows:
                return "0 rows"
            preview = json.dumps(rows, ensure_ascii=True, default=str)
            return f"{len(rows)} row(s): {preview}"
        data = output.get("data")
        if data is not None:
            rendered = json.dumps(data, ensure_ascii=True, default=str)
            return rendered
        return str(output)

    def _markdown_table_cell(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, (dict, list)):
            return ""
        return str(value).replace("|", "\\|")

    def _markdown_for_ranked_rows(self, output: dict[str, Any]) -> str | None:
        if output.get("query_kind") != "batch_model_ranking":
            return None
        rows = output.get("rows")
        if not isinstance(rows, list) or not rows:
            return None
        dict_rows = [row for row in rows if isinstance(row, dict)]
        if not dict_rows:
            return None

        column_keys: list[str] = []
        for row in dict_rows:
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    continue
                if value in {None, ""}:
                    continue
                if key not in column_keys:
                    column_keys.append(key)

        if not column_keys:
            return None

        header = [key.replace("_", " ").title() for key in column_keys]
        lines = [
            "## Ranked Results",
            "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(column_keys)) + " |",
        ]
        for row in dict_rows:
            cells = [self._markdown_table_cell(row.get(key)) for key in column_keys]
            lines.append("| " + " | ".join(cells) + " |")

        answer = output.get("answer")
        if isinstance(answer, str) and answer.strip():
            lines.extend(["", answer.strip()])
        return "\n".join(lines)

    def _summarize(
        self,
        query: str,
        execution_results: list[ExecutionResult],
        plan: ExecutionPlan,
        planner_mode: str,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        success_rows = [r for r in execution_results if r.success]
        if not success_rows:
            return "No tool call succeeded."
        if (
            not planner_mode.startswith("llm")
            or self._llm is None
        ):
            for result in reversed(success_rows):
                if isinstance(result.output, dict):
                    markdown = self._markdown_for_ranked_rows(result.output)
                    if markdown:
                        return markdown
            return " | ".join(
                [
                    f"{r.tool_name}: {self._result_value_for_summary(r.output)}"
                    for r in success_rows
                ]
            )

        data_block_lines = []
        for row in success_rows:
            human_readable = self._result_value_for_summary(row.output)
            data_block_lines.append(
                f"- Step {row.step} using tool '{row.tool_name}' returned: {human_readable}"
            )
        data_block = "\n".join(data_block_lines)

        template = str(self.prompt_templates.get("summarizer") or "").strip()
        if not template:
            raise RuntimeError("Missing orchestrator summarizer prompt template in config.")
        prompt = template.format(query=query, data_block=data_block)

        logger.debug("Summarizer data context: %s", data_block)
        max_tokens = self._safe_generation_tokens(prompt, requested=256)
        return self._generate_llm_text(
            prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            event_callback=event_callback,
        )

    def _extract_response(self, results) -> str:
        for r in results:
            if not r.success:
                continue
            out = r.output

        # Search tool
        if isinstance(out, dict):
            if out.get("answer"):
                return out["answer"]
            if out.get("results") and isinstance(out["results"], list):
                snippets = [
                    res.get("snippet", "")
                    for res in out["results"][:3]
                    if res.get("snippet")
                ]
                if snippets:
                    return " | ".join(snippets)

        # DB tool — adjust key to whatever your db_tool returns
        if isinstance(out, dict) and out.get("data"):
            return str(out["data"])

        # Fallback: stringify whatever is there
        if out:
            return str(out)[:500]

        return "No results found."

    def _emit_run_event(
        self,
        event_callback: Callable[[dict[str, Any]], None] | None,
        payload: dict[str, Any],
    ) -> None:
        if event_callback is None:
            return
        try:
            event_callback(dict(payload))
        except Exception:
            logger.debug("Orchestrator progress callback failed.", exc_info=True)

    def _tool_command_text(self, tool_name: str, arguments: dict[str, Any]) -> str:
        return json.dumps(
            {"tool": tool_name, "arguments": arguments},
            ensure_ascii=True,
            default=str,
        )

    def _plan_progress_payload(self, plan: ExecutionPlan) -> dict[str, Any]:
        return {
            "reasoning": plan.reasoning,
            "steps": [
                {
                    "index": index,
                    "tool": step.name,
                    "arguments": step.arguments,
                    "command": self._tool_command_text(step.name, step.arguments),
                    "reasoning": step.reasoning,
                }
                for index, step in enumerate(plan.steps, 1)
            ],
        }

    def _execution_progress_payload(self, result: ExecutionResult) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step": result.step,
            "tool": result.tool_name,
            "success": result.success,
        }
        if result.inputs:
            payload["arguments"] = result.inputs
            payload["command"] = self._tool_command_text(
                result.tool_name,
                result.inputs,
            )
        if result.success:
            payload["summary"] = self._result_value_for_summary(result.output)
        if result.error:
            payload["error"] = result.error
        return payload

    # ══════════════════════════════════════════════════════
    #  STATUS
    # ══════════════════════════════════════════════════════

    def _active_incomplete_download(self) -> tuple[str | None, float | None]:
        download_dir = self.model_dir / ".cache" / "huggingface" / "download"
        if not download_dir.exists():
            return None, None
        try:
            candidates = sorted(
                download_dir.glob("*.incomplete"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return None, None
        if not candidates:
            return None, None
        candidate = candidates[0]
        try:
            size_mb = round(candidate.stat().st_size / (1024**2), 2)
        except OSError:
            size_mb = None
        return str(candidate.resolve()), size_mb

    def status(self) -> dict[str, Any]:
        init_elapsed_s = None
        if self._init_in_progress and self._init_started_at is not None:
            init_elapsed_s = round(max(0.0, time.time() - self._init_started_at), 2)
        selected_path = (
            Path(self._selected_model_path).expanduser().resolve()
            if self._selected_model_path
            else None
        )
        selected_exists = bool(selected_path and selected_path.exists())
        selected_size_mb = None
        if selected_exists and selected_path:
            try:
                selected_size_mb = round(selected_path.stat().st_size / (1024**2), 2)
            except OSError:
                pass
        incomplete_path, incomplete_size_mb = self._active_incomplete_download()
        return {
            "llm_ready": self._llm is not None,
            "llm_n_ctx": self._llm_n_ctx,
            "llm_attempted": self._llm_attempted,
            "llm_error": self._llm_error,
            "planner_mode": "llm" if self._llm is not None else "fallback",
            "init_mode": self.init_mode,
            "init_in_progress": self._init_in_progress,
            "init_elapsed_s": init_elapsed_s,
            "resource_profile": self._resource_profile,
            "selected_variant": (
                {
                    "id": self._selected_variant.model_id,
                    "name": self._selected_variant.name,
                    "description": self._selected_variant.description,
                    "repo_id": self._selected_variant.repo_id,
                    "filename": self._selected_variant.filename,
                    "n_ctx": self._selected_variant.n_ctx,
                    "n_batch": self._selected_variant.n_batch,
                }
                if self._selected_variant
                else None
            ),
            "selected_model_id": self.preferred_model_id,
            "active_model_id": self._active_model_id,
            "selected_model_path": self._selected_model_path,
            "selected_model_exists": selected_exists,
            "selected_model_size_mb": selected_size_mb,
            "active_download_path": incomplete_path,
            "active_download_size_mb": incomplete_size_mb,
            "model_dir": str(self.model_dir),
            "available_models_total": len(self._variant_candidates()),
            "external_fallback": {
                "enabled": self.enable_external_fallback,
                "order": list(self.external_fallback_order),
                "db_enabled": self.db_fallback_enabled,
                "db_tool_name": self.db_tool_name,
                "db_schema": self.db_schema,
                "db_donors_table": self.db_donors_table,
                "db_hospitals_table": self.db_hospitals_table,
                "db_csv_path": str(self.db_supply_csv_path),
                "db_csv_exists": self.db_supply_csv_path.exists(),
                "db_sqlite_configured": bool(self.db_sqlite_path),
                "stockout_hybrid_tool_name": self.stockout_hybrid_tool_name,
                "stockout_hybrid_available": (
                    find_hybrid_stockout_regression_runtime(self.registry) is not None
                ),
                "simulation_enabled": self.simulation_tool_enabled,
                "simulation_tool_name": self.simulation_tool_name,
                "simulation_base_url": self.simulation_base_url,
                "simulation_default_policy": self.simulation_default_policy,
                "simulation_compare_on_recommend": self.simulation_compare_on_recommend,
                "search_enabled": self.search_fallback_enabled,
                "search_api_configured": bool(self.search_api_url),
                "llm_api_configured": bool(self.llm_api_url),
            },
        }

    def _lookup_path(self, value: Any, path: list[str]) -> Any:
        current = value
        for part in path:
            if isinstance(current, dict):
                current = current.get(part)
                continue
            if isinstance(current, list) and part.isdigit():
                index = int(part)
                if index < 0 or index >= len(current):
                    return None
                current = current[index]
                continue
            return None
        return current

    def _result_for_step(
        self,
        results: list[ExecutionResult],
        step_number: int,
    ) -> ExecutionResult | None:
        for result in results:
            if result.step == step_number and result.success:
                return result
        for result in results:
            if result.step == step_number:
                return result
        return None

    def _resolve_argument_reference(
        self,
        value: str,
        *,
        query: str,
        request_features: dict[str, Any],
        results: list[ExecutionResult],
    ) -> Any:
        stripped = value.strip()
        full_reference = ARGUMENT_REFERENCE_PATTERN.fullmatch(stripped)
        if not full_reference:
            if "$" not in value:
                return value

            def replace_reference(match: re.Match[str]) -> str:
                resolved = self._resolve_argument_reference(
                    match.group(0),
                    query=query,
                    request_features=request_features,
                    results=results,
                )
                if resolved is None:
                    return ""
                if isinstance(resolved, (dict, list)):
                    return json.dumps(resolved, ensure_ascii=True, default=str)
                return str(resolved)

            return ARGUMENT_REFERENCE_PATTERN.sub(replace_reference, value)

        parts = stripped[1:].split(".")
        if not parts:
            return value
        if parts[0] == "request":
            if len(parts) == 2 and parts[1] == "query":
                return query
            if len(parts) >= 3 and parts[1] == "features":
                return self._lookup_path(request_features, parts[2:])
            return None
        if parts[0] == "last":
            result = next((row for row in reversed(results) if row.success), None)
            if result is None:
                return None
            target = result.output if len(parts) > 1 and parts[1] == "output" else result.inputs
            return self._lookup_path(target, parts[2:])
        if parts[0] == "steps" and len(parts) >= 3 and parts[1].isdigit():
            result = self._result_for_step(results, int(parts[1]))
            if result is None:
                return None
            target = result.output if parts[2] == "output" else result.inputs
            return self._lookup_path(target, parts[3:])
        return value

    def _drop_empty_argument_values(self, value: Any) -> Any:
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for key, child in value.items():
                resolved = self._drop_empty_argument_values(child)
                if self._has_model_input_value(resolved):
                    cleaned[key] = resolved
            return cleaned
        if isinstance(value, list):
            return [
                resolved
                for child in value
                for resolved in [self._drop_empty_argument_values(child)]
                if self._has_model_input_value(resolved)
            ]
        return value

    def _resolve_step_arguments(
        self,
        arguments: dict[str, Any],
        *,
        query: str,
        request_features: dict[str, Any],
        results: list[ExecutionResult],
    ) -> dict[str, Any]:
        def resolve(value: Any) -> Any:
            if isinstance(value, str):
                return self._resolve_argument_reference(
                    value,
                    query=query,
                    request_features=request_features,
                    results=results,
                )
            if isinstance(value, dict):
                return {key: resolve(child) for key, child in value.items()}
            if isinstance(value, list):
                return [resolve(child) for child in value]
            return value

        resolved = resolve(dict(arguments or {}))
        cleaned = self._drop_empty_argument_values(resolved)
        return cleaned if isinstance(cleaned, dict) else {}

    def _model_data_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in dict(arguments or {}).items()
            if safe_slug(key) not in self.model_control_argument_keys
        }

    def _model_batch_config(self, arguments: dict[str, Any]) -> dict[str, Any] | None:
        for key in self.model_control_argument_keys:
            value = arguments.get(key)
            if not isinstance(value, dict):
                continue
            rows = value.get("rows")
            if isinstance(rows, list):
                return dict(value)
        return None

    def _model_output_score(
        self,
        output: dict[str, Any],
        score_keys: list[str],
    ) -> float | None:
        for key in score_keys:
            value = output.get(key)
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            if isinstance(value, (int, float)):
                return float(value)
        prediction = output.get("prediction")
        if isinstance(prediction, bool):
            return 1.0 if prediction else 0.0
        if isinstance(prediction, (int, float)):
            return float(prediction)
        return None

    def _execute_model_batch(
        self,
        runtime: ModelRuntime,
        base_inputs: dict[str, Any],
        batch_config: dict[str, Any],
        *,
        forecast_horizon: ForecastHorizon | None,
    ) -> dict[str, Any]:
        rows = [row for row in batch_config.get("rows", []) if isinstance(row, dict)]
        max_rows = max(1, as_int(batch_config.get("max_rows"), len(rows) or 1))
        entity_key = str(batch_config.get("entity_key") or "").strip()
        include_fields = _clean_list(batch_config.get("include_fields"))
        score_keys = _clean_list(batch_config.get("score_keys")) or [
            "probability",
            "prediction",
        ]
        sort_direction = str(batch_config.get("sort") or "desc").strip().lower()

        output_rows: list[dict[str, Any]] = []
        errors: list[str] = []
        prepared_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for source_row in rows[:max_rows]:
            row_inputs = dict(base_inputs)
            row_inputs.update(source_row)
            if entity_key and source_row.get(entity_key) not in {None, ""}:
                row_inputs.setdefault(entity_key, source_row.get(entity_key))
            try:
                resolved = resolve_prediction_features(
                    runtime,
                    row_inputs,
                    forecast_horizon=(
                        forecast_horizon.as_dict() if forecast_horizon else None
                    ),
                )
                prepared_rows.append((source_row, resolved.features, resolved.metadata))
            except Exception as exc:
                label = source_row.get(entity_key) if entity_key else len(output_rows) + 1
                errors.append(f"{label}: {exc}")
                continue

        try:
            prediction_outputs = run_prediction_batch_with_fallback(
                runtime,
                [features for _, features, _ in prepared_rows],
                registry=self.registry,
            )
        except Exception as exc:
            prediction_outputs = []
            errors.append(f"batch: {exc}")

        for (source_row, _features, metadata), prediction_output in zip(
            prepared_rows,
            prediction_outputs,
        ):
            if include_fields:
                public_row = {
                    field: source_row.get(field)
                    for field in include_fields
                    if field in source_row
                }
            else:
                public_row = dict(source_row)
            score = self._model_output_score(prediction_output, score_keys)
            public_row.update(
                {
                    "model_id": runtime.model_id,
                    "model_score": score,
                    "model_output": prediction_output,
                    "feature_resolution": metadata,
                }
            )
            output_rows.append(public_row)

        output_rows.sort(
            key=lambda row: (
                row.get("model_score") is None,
                row.get("model_score") if row.get("model_score") is not None else 0.0,
            ),
            reverse=sort_direction != "asc",
        )
        for index, row in enumerate(output_rows, 1):
            row["rank"] = index

        answer_target = entity_key or "row"
        preview_parts = []
        for row in output_rows[:3]:
            label = str(row.get(entity_key) or row.get("rank"))
            score = row.get("model_score")
            if isinstance(score, (int, float)):
                label = f"{label} score {float(score):.4f}"
            preview_parts.append(label)
        preview = ", ".join(preview_parts)
        payload: dict[str, Any] = {
            "query_kind": "batch_model_ranking",
            "model_id": runtime.model_id,
            "entity_key": entity_key,
            "row_count": len(output_rows),
            "rows": output_rows,
            "answer": (
                f"Ranked {len(output_rows)} {answer_target} candidate(s) with "
                f"{runtime.model_id}."
                + (f" Top candidates: {preview}." if preview else "")
            ),
        }
        if errors:
            payload["warnings"] = errors
        return payload

    # ══════════════════════════════════════════════════════
    #  MAIN RUN
    # ══════════════════════════════════════════════════════

    def run(
        self,
        query: str,
        provided_features: dict[str, Any] | None = None,
        forced_model_ids: list[str] | None = None,
        top_k: int | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        prepared_query = (query or "").strip() or "Run prediction."
        logger.debug("New orchestrator request: %s", prepared_query)
        provided_features = dict(provided_features or {})
        self._emit_run_event(
            event_callback,
            {"type": "progress", "phase": "received"},
        )
        if not forced_model_ids and not self._drop_empty_model_inputs(provided_features):
            direct_response = self._direct_response_payload(prepared_query)
            if direct_response is not None:
                self._emit_run_event(
                    event_callback,
                    {"type": "progress", "phase": "planning"},
                )
                plan = ExecutionPlan(
                    steps=[],
                    reasoning=direct_response["reasoning"],
                    is_multi_step=False,
                )
                self._emit_run_event(
                    event_callback,
                    {
                        "type": "plan",
                        "phase": "planned",
                        **self._plan_progress_payload(plan),
                    },
                )
                self._emit_run_event(
                    event_callback,
                    {
                        "type": "progress",
                        "phase": "summarizing",
                        "successful_tool_count": 0,
                    },
                )
                self._emit_run_event(
                    event_callback,
                    {"type": "progress", "phase": "summary_ready"},
                )
                return {
                    "success": True,
                    "query": prepared_query,
                    "planner_mode": "direct_response",
                    "plan": {"reasoning": plan.reasoning, "steps": []},
                    "execution_results": [],
                    "tool_summaries": [],
                    "natural_language_response": direct_response["markdown"],
                    "verification_data": [direct_response["markdown"]],
                    "orchestrator_status": self.status(),
                }
        prediction_features = strip_horizon_routing_features(provided_features)
        forecast_horizon = extract_forecast_horizon(prepared_query, provided_features)

        if self._llm is None and not self.disable_llm:
            if not self._llm_attempted:
                self._ensure_llm(blocking=False)
            if self._init_in_progress:
                self._ensure_llm(blocking=True)

        loaded = self.registry.loaded()
        if forced_model_ids:
            allowed = [r for r in loaded if r.model_id in forced_model_ids]
        else:
            allowed = loaded

        if not allowed and not self.enable_external_fallback:
            self._emit_run_event(
                event_callback,
                {"type": "error", "phase": "unavailable"},
            )
            return {
                "success": False,
                "error": "No loaded models available and external fallback is disabled.",
            }

        horizon_scores: list[HorizonModelScore] = []
        horizon_candidates: list[ModelRuntime] = []
        horizon_routing_applied = False
        horizon_compatible_allowed = allowed
        if forecast_horizon is not None:
            horizon_compatible_allowed = [
                runtime
                for runtime in allowed
                if self._runtime_entity_compatible_for_horizon(
                    prepared_query,
                    runtime,
                    provided_features,
                )
            ]
        if not forced_model_ids and allowed:
            horizon_scores = self._rank_forecast_horizon_models(
                prepared_query,
                allowed,
                forecast_horizon,
                provided_features,
            )
            horizon_candidates = [score.runtime for score in horizon_scores]
            if horizon_scores:
                allowed = [horizon_scores[0].runtime]
                horizon_routing_applied = True
            elif horizon_compatible_allowed:
                allowed = horizon_compatible_allowed

        allowed_ids = {r.model_id for r in allowed}
        allowed_by_id = {r.model_id: r for r in allowed}
        self._emit_run_event(
            event_callback,
            {"type": "progress", "phase": "planning"},
        )
        try:
            plan, planner_mode = self._plan_execution(
                prepared_query,
                allowed,
                top_k=top_k or len(allowed) or 1,
                forced_model_ids=forced_model_ids,
                provided_features=provided_features,
                event_callback=event_callback,
            )
        except Exception as e:
            logger.exception("Planning crashed.")
            raise e

        if horizon_routing_applied and horizon_scores:
            selected_runtime = horizon_scores[0].runtime
            if not self._plan_contains_runtime_step(plan, selected_runtime):
                plan = self._horizon_routed_plan(
                    prepared_query,
                    selected_runtime,
                    prediction_features,
                )
                if planner_mode == "llm":
                    planner_mode = "llm_horizon_routed"
                elif planner_mode.startswith("llm"):
                    planner_mode = f"{planner_mode}_horizon_routed"
                else:
                    planner_mode = "horizon_routed"

        force_external = False
        if not horizon_routing_applied:
            force_external = self._plan_needs_external_fallback(
                prepared_query, plan, allowed_by_id, forced_model_ids
            )
        if force_external:
            logger.debug(
                "Planned model step appears out of scope. Using external fallback chain."
            )

        self._emit_run_event(
            event_callback,
            {
                "type": "plan",
                "phase": "planned",
                **self._plan_progress_payload(plan),
            },
        )

        extracted = extract_nl_features(prepared_query)
        results: list[ExecutionResult] = []
        fallback_applied = False
        strict_db_requested = False
        strict_model_requested = False

        def append_result(result: ExecutionResult) -> None:
            results.append(result)
            self._emit_run_event(
                event_callback,
                {
                    "type": "tool_finished",
                    "phase": "executing",
                    **self._execution_progress_payload(result),
                },
            )

        logger.debug("Executing %s step(s).", len(plan.steps))
        self._emit_run_event(
            event_callback,
            {
                "type": "progress",
                "phase": "executing",
                "step_count": len(plan.steps),
            },
        )
        for i, step in enumerate(plan.steps, 1):
            step_arguments = self._resolve_step_arguments(
                step.arguments,
                query=prepared_query,
                request_features={**extracted, **provided_features},
                results=results,
            )
            logger.debug("Step %s tool=%s args=%s", i, step.name, step_arguments)
            self._emit_run_event(
                event_callback,
                {
                    "type": "tool_started",
                    "phase": "executing",
                    "step": i,
                    "tool": step.name,
                    "arguments": step_arguments,
                    "command": self._tool_command_text(step.name, step_arguments),
                },
            )
            normalized = safe_slug(step.name)

            if self._is_stockout_hybrid_tool(normalized):
                strict_model_requested = True
                tool_inputs = {
                    **self._drop_empty_model_inputs(extracted),
                    **self._drop_empty_model_inputs(step_arguments),
                    **self._drop_empty_model_inputs(prediction_features),
                }
                try:
                    out = self._execute_stockout_hybrid_tool(
                        prepared_query,
                        tool_inputs,
                    )
                    logger.debug(
                        "Step %s succeeded. Stockout probability: %s",
                        i,
                        out.get("stockout_probability"),
                    )
                    append_result(
                        ExecutionResult(
                            i,
                            self.stockout_hybrid_tool_name,
                            tool_inputs,
                            out,
                            True,
                        )
                    )
                except Exception as exc:
                    logger.warning("Step %s failed: %s", i, exc)
                    append_result(
                        ExecutionResult(
                            i,
                            self.stockout_hybrid_tool_name,
                            tool_inputs,
                            {},
                            False,
                            str(exc),
                        )
                    )
                continue

            if self.enable_external_fallback and self._is_external_fallback_step(
                normalized
            ):
                preferred = (
                    normalized if normalized not in self.fallback_aliases else None
                )
                if preferred:
                    step_arguments = self._repair_row_source_arguments(
                        prepared_query,
                        preferred,
                        step_arguments,
                        results,
                    )
                strict_db_requested = strict_db_requested or bool(
                    preferred
                    and self._is_db_tool(preferred)
                    and isinstance(step_arguments, dict)
                    and any(
                        key in step_arguments
                        for key in (
                            "table",
                            "field",
                            "fields",
                            "aggregate",
                            "group_by",
                            "filters",
                            "latest_only",
                        )
                    )
                )
                fallback_rows = self._run_external_fallback_chain(
                    prepared_query,
                    i,
                    preferred_first=preferred,
                    preferred_arguments=step_arguments if preferred else None,
                )
                used_fallback = False
                if preferred is None:
                    used_fallback = bool(fallback_rows)
                elif fallback_rows:
                    first_tool = safe_slug(fallback_rows[0].tool_name)
                    used_fallback = first_tool != preferred or len(fallback_rows) > 1
                fallback_applied = fallback_applied or used_fallback
                for fallback_row in fallback_rows:
                    append_result(fallback_row)
                if (
                    fallback_rows
                    and any(r.success for r in fallback_rows)
                    and (preferred is None or i == len(plan.steps))
                ):
                    break
                continue

            rt = self.registry.get(step.name)
            if not rt or rt.model_id not in allowed_ids:
                append_result(
                    ExecutionResult(
                        i, step.name, step_arguments, {}, False, "Invalid tool"
                    )
                )
                continue

            model_arguments = self._model_data_arguments(step_arguments)
            request_inputs = {
                **self._drop_empty_model_inputs(extracted),
                **self._drop_empty_model_inputs(model_arguments),
                **self._drop_empty_model_inputs(prediction_features),
            }
            batch_config = self._model_batch_config(step_arguments)
            if (
                self._hide_stockout_prediction_models()
                and self._is_stockout_prediction_runtime(rt)
            ):
                if not self._model_step_can_use_request_inputs(
                    prepared_query,
                    step,
                    rt,
                    request_inputs,
                    horizon_routed=horizon_routing_applied,
                    forced_model_ids=forced_model_ids,
                ):
                    append_result(
                        ExecutionResult(
                            i,
                            rt.model_id,
                            step_arguments,
                            {},
                            False,
                            "Skipped: request inputs do not match this model's declared inputs.",
                        )
                    )
                    continue
                strict_model_requested = True
                tool_inputs = {
                    **request_inputs,
                    "regression_model_id": rt.model_id,
                }
                try:
                    out = self._execute_stockout_hybrid_tool(
                        prepared_query,
                        tool_inputs,
                    )
                    logger.debug(
                        "Step %s succeeded. Stockout probability: %s",
                        i,
                        out.get("stockout_probability"),
                    )
                    append_result(
                        ExecutionResult(
                            i,
                            self.stockout_hybrid_tool_name,
                            tool_inputs,
                            out,
                            True,
                        )
                    )
                except Exception as exc:
                    logger.warning("Step %s failed: %s", i, exc)
                    append_result(
                        ExecutionResult(
                            i,
                            self.stockout_hybrid_tool_name,
                            tool_inputs,
                            {},
                            False,
                            str(exc),
                        )
                    )
                continue

            if force_external:
                append_result(
                    ExecutionResult(
                        i,
                        rt.model_id,
                        step_arguments,
                        {},
                        False,
                        "Query is out-of-scope for this model based on metadata/examples; external fallback required.",
                    )
                )
                continue

            if batch_config is None and not self._model_step_can_use_request_inputs(
                prepared_query,
                step,
                rt,
                request_inputs,
                horizon_routed=horizon_routing_applied,
                forced_model_ids=forced_model_ids,
            ):
                append_result(
                    ExecutionResult(
                        i,
                        rt.model_id,
                        step_arguments,
                        {},
                        False,
                        "Skipped: request inputs do not match this model's declared inputs.",
                    )
                )
                continue
            strict_model_requested = True
            try:
                if batch_config is not None:
                    out = self._execute_model_batch(
                        rt,
                        request_inputs,
                        batch_config,
                        forecast_horizon=forecast_horizon,
                    )
                    append_result(
                        ExecutionResult(i, rt.model_id, request_inputs, out, True)
                    )
                    continue
                component_payloads = resolve_component_prediction_features(
                    rt,
                    request_inputs,
                    forecast_horizon=(
                        forecast_horizon.as_dict() if forecast_horizon else None
                    ),
                )
                if len(component_payloads) > 1:
                    component_rows: list[dict[str, Any]] = []
                    component_metadata: list[dict[str, Any]] = []
                    output_name = "prediction"
                    output_unit = ""
                    output_task_type = ""
                    for component, component_resolved in component_payloads:
                        component_output = run_prediction_with_fallback(
                            rt,
                            component_resolved.features,
                            registry=self.registry,
                        )
                        prediction_name = str(
                            component_output.get("prediction_name") or "prediction"
                        ).strip()
                        prediction_unit = str(
                            component_output.get("prediction_unit") or ""
                        ).strip()
                        prediction_task_type = str(
                            component_output.get("prediction_task_type") or ""
                        ).strip()
                        if prediction_name:
                            output_name = prediction_name
                        if prediction_unit:
                            output_unit = prediction_unit
                        if prediction_task_type:
                            output_task_type = prediction_task_type
                        component_value = component_output.get(
                            prediction_name,
                            component_output.get("prediction"),
                        )
                        row = {
                            "component": component,
                            "blood_type": component,
                            "prediction": component_value,
                            "prediction_name": prediction_name,
                            "prediction_unit": prediction_unit,
                        }
                        if prediction_task_type:
                            row["prediction_task_type"] = prediction_task_type
                        if prediction_name and prediction_name not in row:
                            row[prediction_name] = component_value
                        if "raw_prediction" in component_output:
                            row["raw_prediction"] = component_output["raw_prediction"]
                        if "prediction_clipped" in component_output:
                            row["prediction_clipped"] = component_output[
                                "prediction_clipped"
                            ]
                        component_rows.append(row)
                        component_metadata.append(component_resolved.metadata)
                    out = {
                        "model_id": rt.model_id,
                        "aliases": getattr(rt, "aliases", []),
                        "model_type": getattr(rt, "model_type", "unknown"),
                        "file_path": str(getattr(rt, "file_path", "")),
                        "prediction_name": output_name,
                        "prediction_unit": output_unit,
                        "prediction_task_type": output_task_type,
                        "component_forecasts": component_rows,
                        "predictions_by_component": {
                            row["component"]: row["prediction"]
                            for row in component_rows
                        },
                        "feature_resolution": {
                            "source": "component_fanout",
                            "component_key": "blood_type",
                            "components": [row["component"] for row in component_rows],
                            "component_feature_resolutions": component_metadata,
                        },
                    }
                    logger.debug(
                        "Step %s succeeded. Component forecasts: %s",
                        i,
                        out["predictions_by_component"],
                    )
                    append_result(
                        ExecutionResult(i, rt.model_id, request_inputs, out, True)
                    )
                    continue
                resolved = resolve_prediction_features(
                    rt,
                    request_inputs,
                    forecast_horizon=(
                        forecast_horizon.as_dict() if forecast_horizon else None
                    ),
                )
                out = run_prediction_with_fallback(
                    rt,
                    resolved.features,
                    registry=self.registry,
                )
                if isinstance(out, dict):
                    runtime_defaults = getattr(rt, "defaults", {}) or {}
                    if (
                        isinstance(runtime_defaults, dict)
                        and "hybrid_stockout" in runtime_defaults
                    ):
                        try:
                            hybrid_output = build_hybrid_stockout_prediction(
                                registry=self.registry,
                                regression_runtime=rt,
                                request_features=request_inputs,
                                resolved_regression_features=resolved.features,
                                regression_output=out,
                                query=prepared_query,
                            )
                            if hybrid_output:
                                out = {**out, **hybrid_output}
                        except Exception as exc:
                            out.setdefault("hybrid_stockout_error", str(exc))
                    entity_row = resolved.metadata.get("entity_row")
                    if isinstance(entity_row, dict):
                        component_value = entity_row.get("blood_type")
                        if component_value not in {None, ""}:
                            out.setdefault("component", component_value)
                            out.setdefault("blood_type", component_value)
                    out["feature_resolution"] = resolved.metadata
                val = out.get("probability", out.get("prediction"))
                logger.debug("Step %s succeeded. Result: %s", i, val)
                append_result(
                    ExecutionResult(i, rt.model_id, resolved.features, out, True)
                )
            except Exception as e:
                logger.warning("Step %s failed: %s", i, e)
                append_result(
                    ExecutionResult(i, rt.model_id, request_inputs, {}, False, str(e))
                )

        if self.enable_external_fallback and (
            force_external or not any(r.success for r in results)
        ):
            if not strict_db_requested and not strict_model_requested:
                fallback_rows = self._run_external_fallback_chain(
                    prepared_query, len(results) + 1
                )
                fallback_applied = fallback_applied or bool(fallback_rows)
                for fallback_row in fallback_rows:
                    append_result(fallback_row)
                if fallback_rows and any(r.success for r in fallback_rows):
                    planner_mode = (
                        "external_fallback"
                        if planner_mode != "llm"
                        else "llm_with_external_fallback"
                    )

        if fallback_applied and planner_mode == "llm":
            planner_mode = "llm_with_external_fallback"

        logger.debug("Summarizing results.")
        self._emit_run_event(
            event_callback,
            {
                "type": "progress",
                "phase": "summarizing",
                "successful_tool_count": len([r for r in results if r.success]),
            },
        )
        summary = self._summarize(
            prepared_query, results, plan, planner_mode, event_callback=event_callback
        )
        logger.debug("Finished request. Summary preview: %s", summary[:80])
        self._emit_run_event(
            event_callback,
            {"type": "progress", "phase": "summary_ready"},
        )
        stockout_output = next(
            (
                r.output
                for r in results
                if r.success
                and self._is_stockout_hybrid_tool(r.tool_name)
                and isinstance(r.output, dict)
                and r.output.get("hybrid_prediction") is True
            ),
            None,
        )

        response = {
            "success": any(r.success for r in results),
            "query": prepared_query,
            "planner_mode": planner_mode,
            "plan": {
                "reasoning": plan.reasoning,
                "steps": [
                    {"tool": s.name, "arguments": s.arguments} for s in plan.steps
                ],
            },
            "execution_results": [
                self._execution_result_payload(r) for r in results
            ],
            "tool_summaries": [
                {
                    "step": r.step,
                    "tool": r.tool_name,
                    "success": r.success,
                    "summary": (
                        self._result_value_for_summary(r.output)
                        if r.success
                        else r.error
                    ),
                }
                for r in results
            ],
            "natural_language_response": summary,
            "verification_data": [
                f"{r.tool_name}: {self._result_value_for_summary(r.output)}"
                for r in results
                if r.success
            ],
            "forecast_horizon": (
                forecast_horizon.as_dict() if forecast_horizon is not None else None
            ),
            "horizon_routing": {
                "applied": horizon_routing_applied,
                "candidate_model_ids": [rt.model_id for rt in horizon_candidates],
                "selected_model_id": horizon_scores[0].model_id if horizon_scores else "",
                "scores": [score.as_dict() for score in horizon_scores],
            },
            "orchestrator_status": self.status(),
        }
        if isinstance(stockout_output, dict):
            response["stockout_prediction"] = stockout_output
        return response
