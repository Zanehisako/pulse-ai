from __future__ import annotations

import logging
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ml.core.feature_extraction import extract_nl_features
from ml.core.http_client import http_json_request
from ml.core.utils import as_float, as_int, safe_slug

logger = logging.getLogger(__name__)

BACKEND_ML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SIMULATION_TOOL_CONFIG_PATH = BACKEND_ML_DIR / "config" / "simulation_tool.json"
_SIMULATION_CONFIG_CACHE: tuple[Path, float, dict[str, Any]] | None = None


def _clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _simulation_config_path() -> Path:
    raw = os.getenv("PIOS_SIMULATION_TOOL_CONFIG", "").strip()
    return Path(raw).expanduser().resolve() if raw else DEFAULT_SIMULATION_TOOL_CONFIG_PATH


def _simulation_config() -> dict[str, Any]:
    global _SIMULATION_CONFIG_CACHE
    path = _simulation_config_path()
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:
        raise RuntimeError(f"Simulation tool config not found: {path}") from exc
    if (
        _SIMULATION_CONFIG_CACHE is not None
        and _SIMULATION_CONFIG_CACHE[0] == path
        and _SIMULATION_CONFIG_CACHE[1] == mtime
    ):
        return _SIMULATION_CONFIG_CACHE[2]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid simulation tool config JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Simulation tool config must be a JSON object.")
    _SIMULATION_CONFIG_CACHE = (path, mtime, payload)
    return payload


def _config_dict(key: str) -> dict[str, Any]:
    value = _simulation_config().get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"Simulation tool config '{key}' must be an object.")
    return value


def _config_patterns(key: str) -> list[str]:
    values = _clean_list(_simulation_config().get(key))
    if not values:
        raise RuntimeError(f"Simulation tool config '{key}' must be a non-empty list.")
    return values


def _policy_aliases() -> dict[str, str]:
    return {
        safe_slug(key): safe_slug(value)
        for key, value in _config_dict("policy_aliases").items()
        if safe_slug(key) and safe_slug(value)
    }


def _predefined_scenario_aliases() -> dict[str, str]:
    return {
        safe_slug(key): safe_slug(value)
        for key, value in _config_dict("predefined_scenario_aliases").items()
        if safe_slug(key) and safe_slug(value)
    }


def _default_simulation_policy() -> str:
    return safe_slug(_simulation_config().get("default_policy"))


def _default_compare_policies_config() -> list[str]:
    return [safe_slug(value) for value in _clean_list(_simulation_config().get("default_compare_policies")) if safe_slug(value)]


def _custom_scenario_field_aliases() -> dict[str, list[str]]:
    return {
        str(key).strip(): _clean_list(value)
        for key, value in _config_dict("custom_scenario_field_aliases").items()
        if str(key).strip()
    }


def _unsupported_scope_warning() -> str:
    return str(_simulation_config().get("unsupported_scope_warning") or "").strip()


def _config_bool(key: str, default: bool = False) -> bool:
    value = _simulation_config().get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _config_section(key: str) -> dict[str, Any]:
    value = _simulation_config().get(key)
    return dict(value) if isinstance(value, dict) else {}


def _backend_config() -> dict[str, Any]:
    return _config_section("backend")


def _backend_bool(key: str, default: bool = False) -> bool:
    value = _backend_config().get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _default_scenario_key() -> str:
    value = safe_slug(_backend_config().get("default_scenario_key"))
    if not value:
        raise RuntimeError("Simulation backend.default_scenario_key must be configured.")
    return value


def _fallback_policy_key() -> str:
    value = safe_slug(_backend_config().get("fallback_policy_key"))
    if not value:
        raise RuntimeError("Simulation backend.fallback_policy_key must be configured.")
    return value


def _run_key_policy() -> str:
    return safe_slug(_backend_config().get("run_key_policy"))


def _include_timeline() -> bool:
    return _backend_bool("include_timeline", True)


def _default_hours() -> int:
    value = as_int(_backend_config().get("default_hours"), 0)
    if value <= 0:
        raise RuntimeError("Simulation backend.default_hours must be positive.")
    return value


def _weather_options() -> list[str]:
    return _clean_list(_backend_config().get("weather_options"))


def _inventory_keys() -> list[str]:
    values = _clean_list(_backend_config().get("inventory_keys"))
    if not values:
        raise RuntimeError("Simulation backend.inventory_keys must be configured.")
    return values


def simulation_default_base_url() -> str:
    value = str(_simulation_config().get("base_url") or "").strip().rstrip("/")
    if not value:
        raise RuntimeError("Simulation tool config base_url must be configured.")
    return value


def simulation_default_timeout_s() -> float:
    value = as_float(_simulation_config().get("timeout_seconds"), 0.0)
    if value <= 0:
        raise RuntimeError("Simulation tool config timeout_seconds must be positive.")
    return value


def simulation_default_compare_on_recommend() -> bool:
    return _config_bool("compare_on_recommend", False)


DEFAULT_SIMULATION_POLICY = _default_simulation_policy()
DEFAULT_COMPARE_POLICIES = _default_compare_policies_config()
POLICY_ALIASES = _policy_aliases()
PREDEFINED_SCENARIO_ALIASES = _predefined_scenario_aliases()
SIMULATION_INTENT_PATTERNS = _config_patterns("intent_patterns")
RECOMMEND_PATTERNS = _config_patterns("recommend_patterns")
COMPARE_PATTERNS = _config_patterns("compare_patterns")
SHOCK_PATTERNS = _config_patterns("shock_patterns")
NEGATIVE_ADJUSTMENT_VERBS = _config_patterns("negative_adjustment_verbs")
POSITIVE_ADJUSTMENT_VERBS = _config_patterns("positive_adjustment_verbs")
UNSUPPORTED_SCOPE_WARNING = _unsupported_scope_warning()
CUSTOM_SCENARIO_FIELD_ALIASES = _custom_scenario_field_aliases()
ML_PREDICTION_EXCLUSION_TERMS = _clean_list(_simulation_config().get("ml_prediction_exclusion_terms"))
SCENARIO_OVERRIDE_TERMS = _clean_list(_simulation_config().get("scenario_override_terms"))


@dataclass
class SimulationToolConfig:
    base_url: str
    timeout_s: float
    default_policy: str = DEFAULT_SIMULATION_POLICY
    compare_on_recommend: bool = False
    default_compare_policies: list[str] = field(
        default_factory=lambda: list(DEFAULT_COMPARE_POLICIES)
    )


@dataclass
class ParsedShock:
    kind: str
    magnitude_pct: float
    parameter_overrides: dict[str, Any]
    summary: str


@dataclass
class SimulationRequest:
    original_query: str
    mode: str
    scenario_kind: str
    scenario_key: str | None
    hours_override: int | None
    custom_scenario_payload: dict[str, Any] | None
    selected_policy_key: str | None
    compare_policy_keys: list[str] = field(default_factory=list)
    requested_region: str | None = None
    requested_blood_type: str | None = None
    requested_policy_tokens: list[str] = field(default_factory=list)
    unknown_policy_tokens: list[str] = field(default_factory=list)
    shocks: list[ParsedShock] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SimulationBackendClient:
    def __init__(self, *, base_url: str, timeout_s: float) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")
        self.timeout_s = max(1.0, float(timeout_s))
        if not self.base_url:
            raise RuntimeError("Simulation backend base URL is not configured.")

    def _request(
        self,
        *,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return http_json_request(
            method=method,
            url=f"{self.base_url}{path}",
            body=body,
            timeout_s=self.timeout_s,
        )

    def get_metadata(self) -> dict[str, Any]:
        response = self._request(method="GET", path="/api/meta")
        payload = response.get("data")
        return payload if isinstance(payload, dict) else {}

    def create_custom_scenario(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._request(
            method="POST",
            path="/api/custom-scenarios",
            body=payload,
        )
        data = response.get("data")
        return data if isinstance(data, dict) else {}

    def run_predefined_scenario(
        self,
        *,
        scenario_key: str,
        strategy_key: str,
        hours_override: int | None,
        include_timeline: bool,
        seed: int = 100,
        dreamerv3_run_key: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "scenario_key": scenario_key,
            "strategy_key": strategy_key,
            "seed": int(seed),
            "include_timeline": bool(include_timeline),
        }
        if hours_override is not None:
            body["hours_override"] = int(hours_override)
        if dreamerv3_run_key:
            body["dreamerv3_run_key"] = dreamerv3_run_key
        response = self._request(method="POST", path="/api/evaluations/run", body=body)
        data = response.get("data")
        return data if isinstance(data, dict) else {}

    def run_custom_scenario(
        self,
        *,
        custom_payload: dict[str, Any],
        strategy_key: str,
        hours_override: int | None,
        include_timeline: bool,
        seed: int = 100,
        dreamerv3_run_key: str | None = None,
    ) -> dict[str, Any]:
        created = self.create_custom_scenario(custom_payload)
        scenario = created.get("scenario")
        if not isinstance(scenario, dict):
            raise RuntimeError(
                "Simulation backend did not return a custom scenario payload."
            )
        scenario_key = str(scenario.get("key", "")).strip()
        if not scenario_key:
            raise RuntimeError(
                "Simulation backend did not return a custom scenario key."
            )
        return self.run_predefined_scenario(
            scenario_key=scenario_key,
            strategy_key=strategy_key,
            hours_override=hours_override,
            include_timeline=include_timeline,
            seed=seed,
            dreamerv3_run_key=dreamerv3_run_key,
        )

    def compare_policies(
        self,
        *,
        scenario_key: str,
        strategy_keys: list[str],
        hours_override: int | None,
        runs: int = 1,
        seed_start: int = 100,
        dreamerv3_run_key: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "scenario_keys": [scenario_key],
            "strategy_keys": strategy_keys,
            "runs": int(runs),
            "seed_start": int(seed_start),
        }
        if hours_override is not None:
            body["hours_override"] = int(hours_override)
        if dreamerv3_run_key:
            body["dreamerv3_run_key"] = dreamerv3_run_key
        response = self._request(
            method="POST",
            path="/api/evaluations/compare",
            body=body,
        )
        data = response.get("data")
        return data if isinstance(data, dict) else {}

    def compare_custom_policies(
        self,
        *,
        custom_payload: dict[str, Any],
        strategy_keys: list[str],
        hours_override: int | None,
        runs: int = 1,
        seed_start: int = 100,
        dreamerv3_run_key: str | None = None,
    ) -> dict[str, Any]:
        created = self.create_custom_scenario(custom_payload)
        scenario = created.get("scenario")
        if not isinstance(scenario, dict):
            raise RuntimeError(
                "Simulation backend did not return a custom scenario payload."
            )
        scenario_key = str(scenario.get("key", "")).strip()
        if not scenario_key:
            raise RuntimeError(
                "Simulation backend did not return a custom scenario key."
            )
        return self.compare_policies(
            scenario_key=scenario_key,
            strategy_keys=strategy_keys,
            hours_override=hours_override,
            runs=runs,
            seed_start=seed_start,
            dreamerv3_run_key=dreamerv3_run_key,
        )


def is_simulation_query(query: str) -> bool:
    lowered = (query or "").lower()
    # Policy comparison queries always go to simulation
    if any(re.search(pattern, lowered) for pattern in COMPARE_PATTERNS) and any(
        alias in safe_slug(lowered) for alias in POLICY_ALIASES
    ):
        return True
    # Exclude single-point ML prediction queries unless scenario language present
    if ML_PREDICTION_EXCLUSION_TERMS and any(term in lowered for term in ML_PREDICTION_EXCLUSION_TERMS):
        if not SCENARIO_OVERRIDE_TERMS or not any(ind in lowered for ind in SCENARIO_OVERRIDE_TERMS):
            return False
    if any(re.search(pattern, lowered) for pattern in SIMULATION_INTENT_PATTERNS):
        return True
    if "forecast" in lowered and any(
        re.search(pattern, lowered) for pattern in SHOCK_PATTERNS
    ):
        return True
    if any(re.search(pattern, lowered) for pattern in RECOMMEND_PATTERNS) and any(
        re.search(pattern, lowered) for pattern in SHOCK_PATTERNS
    ):
        return True
    return False


def execute_simulation_query(
    query: str,
    *,
    config: SimulationToolConfig,
) -> dict[str, Any]:
    prepared_query = (query or "").strip()
    if not is_simulation_query(prepared_query):
        raise RuntimeError("Simulation tool does not match this query.")

    started_at = time.perf_counter()
    logger.debug("simulation_tool.intent_detected query=%r", prepared_query)

    client = SimulationBackendClient(
        base_url=config.base_url,
        timeout_s=config.timeout_s,
    )

    metadata: dict[str, Any] = {}
    metadata_error: str | None = None
    try:
        metadata = client.get_metadata()
    except Exception as exc:
        metadata_error = str(exc)
        logger.warning("simulation_tool.metadata_failed error=%s", exc)

    request = build_simulation_request(
        prepared_query,
        metadata=metadata,
        config=config,
    )
    logger.debug(
        "simulation_tool.request mode=%s scenario_kind=%s policy=%s compare=%s",
        request.mode,
        request.scenario_kind,
        request.selected_policy_key,
        request.compare_policy_keys,
    )

    dreamerv3_run_key = (
        str(metadata.get("default_dreamerv3_run_key") or "").strip() or None
    )

    try:
        backend_started = time.perf_counter()
        if request.mode == "compare":
            if len(request.compare_policy_keys) <= 1:
                request.warnings.append(
                    "The compare request resolved to one policy, so a single-policy run was used."
                )
                raw = _run_single_from_request(
                    client=client,
                    request=request,
                    dreamerv3_run_key=dreamerv3_run_key,
                )
            else:
                raw = _run_compare_from_request(
                    client=client,
                    request=request,
                    dreamerv3_run_key=dreamerv3_run_key,
                )
        elif (
            request.mode == "recommend"
            and config.compare_on_recommend
            and request.compare_policy_keys
        ):
            comparison_snapshot = None
            try:
                compare_raw = _run_compare_from_request(
                    client=client,
                    request=request,
                    dreamerv3_run_key=dreamerv3_run_key,
                )
                comparison_snapshot = _normalize_compare_result(
                    request=request,
                    raw=compare_raw,
                    metadata=metadata,
                )
                best_policy = comparison_snapshot.get("best_policy")
                if isinstance(best_policy, dict) and best_policy.get("policy_key"):
                    request.selected_policy_key = str(best_policy["policy_key"])
                else:
                    request.warnings.append(
                        "The optional policy comparison did not produce a clear winner, so the default recommendation policy was used."
                    )
            except Exception as exc:
                logger.warning("simulation_tool.recommend_compare_failed error=%s", exc)
                request.warnings.append(
                    "Optional policy comparison failed, so the recommendation fell back to a single-policy simulation."
                )
            raw = _run_single_from_request(
                client=client,
                request=request,
                dreamerv3_run_key=dreamerv3_run_key,
            )
            if comparison_snapshot is not None and isinstance(raw, dict):
                raw = dict(raw)
                raw["_recommendation_comparison"] = comparison_snapshot
        else:
            raw = _run_single_from_request(
                client=client,
                request=request,
                dreamerv3_run_key=dreamerv3_run_key,
            )
        backend_latency_ms = round((time.perf_counter() - backend_started) * 1000, 2)
        logger.debug("simulation_tool.backend_latency_ms=%s", backend_latency_ms)
    except Exception as exc:
        logger.exception("simulation_tool.backend_failed")
        answer = _format_simulation_failure(
            request=request,
            metadata_error=metadata_error,
            backend_error=str(exc),
        )
        total_latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return {
            "tool_type": "simulation",
            "provider": "simulation_backend",
            "mode": request.mode,
            "request": simulation_request_payload(request),
            "backend_success": False,
            "latency_ms": total_latency_ms,
            "warnings": list(request.warnings),
            "answer": answer,
        }

    normalized = normalize_simulation_result(
        request=request,
        raw=raw,
        metadata=metadata,
    )
    total_latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    answer = synthesize_simulation_response(normalized)
    logger.debug(
        "simulation_tool.response_generated mode=%s latency_ms=%s",
        request.mode,
        total_latency_ms,
    )
    return {
        "tool_type": "simulation",
        "provider": "simulation_backend",
        "mode": request.mode,
        "request": simulation_request_payload(request),
        "backend_success": True,
        "latency_ms": total_latency_ms,
        "normalized_result": normalized,
        "warnings": normalized.get("warnings", []),
        "answer": answer,
    }


def simulation_request_payload(request: SimulationRequest) -> dict[str, Any]:
    payload = asdict(request)
    payload["shocks"] = [asdict(shock) for shock in request.shocks]
    return payload


def build_simulation_request(
    query: str,
    *,
    metadata: dict[str, Any],
    config: SimulationToolConfig,
) -> SimulationRequest:
    mode = _infer_mode(query)
    hours_override = _parse_hours_override(query)
    requested_region = _extract_region(query)
    requested_blood_type = _extract_blood_type(query)
    policies, unknown_tokens, requested_policy_tokens = _resolve_policies(
        query,
        mode=mode,
        metadata=metadata,
        config=config,
    )
    scenario_key = _resolve_predefined_scenario(query, metadata=metadata)
    shocks = _extract_shocks(
        query,
        metadata=metadata,
        base_scenario_key=scenario_key or _default_scenario_key(),
    )
    warnings: list[str] = []
    assumptions: list[str] = []

    if requested_region or requested_blood_type:
        warnings.append(UNSUPPORTED_SCOPE_WARNING)

    if unknown_tokens:
        warnings.append(
            "Unknown policy names were ignored: "
            + ", ".join(sorted(set(unknown_tokens)))
            + "."
        )

    if not scenario_key and not shocks:
        scenario_key = _default_scenario_key()
        assumptions.append(
            f"No explicit scenario shock was supplied, so a {scenario_key} scenario was used."
        )

    if hours_override is None:
        hours_override = _default_hours()
        days = int(hours_override / 24) if hours_override % 24 == 0 else None
        window = f"{days}-day" if days else f"{hours_override}-hour"
        assumptions.append(f"No time horizon was supplied, so a {window} window was used.")

    if scenario_key and not shocks:
        scenario_kind = "predefined"
        custom_payload = None
    else:
        scenario_kind = "custom"
        custom_payload = _build_custom_scenario_payload(
            query=query,
            base_scenario_key=scenario_key or _default_scenario_key(),
            hours_override=hours_override,
            selected_policy_key=policies[0] if policies else DEFAULT_SIMULATION_POLICY,
            shocks=shocks,
        )
        if shocks:
            assumptions.extend(
                [
                    "Custom scenario overrides were derived directly from the natural-language shock description.",
                ]
            )

    selected_policy_key = None
    compare_policy_keys: list[str] = []
    if mode == "compare":
        compare_policy_keys = list(dict.fromkeys(policies))
        if not compare_policy_keys:
            compare_policy_keys = list(_default_compare_policies(metadata, config))
            assumptions.append(
                "No explicit policy list was supplied, so the default policy comparison set was used."
            )
    else:
        selected_policy_key = _select_single_policy(
            resolved_policies=policies,
            metadata=metadata,
            config=config,
            default_policy_key=(
                _fallback_policy_key() if mode == "forecast" else config.default_policy
            ),
            warnings=warnings,
        )
        if mode == "recommend" and config.compare_on_recommend:
            compare_policy_keys = list(
                dict.fromkeys(
                    [
                        selected_policy_key,
                        *_default_compare_policies(metadata, config),
                    ]
                )
            )

    return SimulationRequest(
        original_query=query,
        mode=mode,
        scenario_kind=scenario_kind,
        scenario_key=scenario_key if scenario_kind == "predefined" else None,
        hours_override=hours_override,
        custom_scenario_payload=custom_payload,
        selected_policy_key=selected_policy_key,
        compare_policy_keys=compare_policy_keys,
        requested_region=requested_region,
        requested_blood_type=requested_blood_type,
        requested_policy_tokens=requested_policy_tokens,
        unknown_policy_tokens=unknown_tokens,
        shocks=shocks,
        assumptions=assumptions,
        warnings=warnings,
    )


def _infer_mode(query: str) -> str:
    lowered = (query or "").lower()
    if any(re.search(pattern, lowered) for pattern in COMPARE_PATTERNS):
        return "compare"
    if any(re.search(pattern, lowered) for pattern in RECOMMEND_PATTERNS):
        return "recommend"
    return "forecast"


def _parse_hours_override(query: str) -> int | None:
    lowered = (query or "").lower()
    week_match = re.search(r"\bnext\s+(\d+)\s+weeks?\b", lowered)
    if week_match:
        return max(24, int(week_match.group(1)) * 7 * 24)
    if "next week" in lowered:
        return 7 * 24
    if "next month" in lowered:
        return 30 * 24
    day_match = re.search(r"\b(?:next\s+)?(\d+)\s+days?\b", lowered)
    if day_match:
        return max(24, int(day_match.group(1)) * 24)
    hour_match = re.search(r"\b(\d+)\s+hours?\b", lowered)
    if hour_match:
        return max(24, int(hour_match.group(1)))
    return None


def _extract_region(query: str) -> str | None:
    patterns = (
        r"\b(?:in|across|within)\s+the\s+([A-Za-z][A-Za-z\s-]+?)\s+region\b",
        r"\b([A-Za-z][A-Za-z\s-]+?)\s+region\b",
        r"\b(?:in|across|within)\s+([A-Za-z][A-Za-z\s-]+?)\s+(?:next|over|under|for)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" ,.")
            if value and len(value) <= 80:
                return value
    return None


def _extract_blood_type(query: str) -> str | None:
    extracted = extract_nl_features(query)
    blood_type = extracted.get("blood_type")
    if isinstance(blood_type, str) and blood_type.strip():
        return blood_type.strip().upper()
    return None


def _resolve_policies(
    query: str,
    *,
    mode: str,
    metadata: dict[str, Any],
    config: SimulationToolConfig,
) -> tuple[list[str], list[str], list[str]]:
    catalog = _policy_catalog(metadata)
    resolved: list[str] = []
    unknown: list[str] = []
    requested_tokens: list[str] = []

    compare_clause = (
        _extract_compare_policy_clause(query) if mode == "compare" else None
    )
    if compare_clause:
        requested_tokens = _split_policy_tokens(compare_clause)
    else:
        requested_tokens = _scan_policy_mentions(query)

    for token in requested_tokens:
        normalized = _resolve_policy_token(token, catalog)
        if normalized:
            resolved.append(normalized)
        else:
            unknown.append(token)

    if mode == "compare":
        return list(dict.fromkeys(resolved)), unknown, requested_tokens

    return list(dict.fromkeys(resolved)), unknown, requested_tokens


def _policy_catalog(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = metadata.get("strategies")
    catalog: dict[str, dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = safe_slug(str(row.get("key", "")).strip())
            if not key:
                continue
            catalog[key] = row
            aliases = {
                key,
                safe_slug(str(row.get("name", "")).strip()),
                safe_slug(str(row.get("description", "")).strip().split(".")[0]),
            }
            for alias in aliases:
                if alias:
                    POLICY_ALIASES.setdefault(alias, key)
    return catalog


def _extract_compare_policy_clause(query: str) -> str | None:
    match = re.search(
        r"\bcompare\s+(.+?)(?:\s+(?:for|under|on|in|with)\b|$)",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip()


def _split_policy_tokens(raw: str) -> list[str]:
    normalized = re.sub(r"\bversus\b|\bvs\.?\b", ",", raw, flags=re.IGNORECASE)
    normalized = re.sub(r"\band\b", ",", normalized, flags=re.IGNORECASE)
    return [token.strip(" .") for token in normalized.split(",") if token.strip(" .")]


def _scan_policy_mentions(query: str) -> list[str]:
    lowered_slug = safe_slug(query)
    found: list[str] = []
    for alias in sorted(POLICY_ALIASES, key=len, reverse=True):
        if not alias:
            continue
        if alias in lowered_slug:
            found.append(alias)
    return found


def _resolve_policy_token(token: str, catalog: dict[str, dict[str, Any]]) -> str | None:
    normalized = safe_slug(token)
    if not normalized:
        return None
    direct = POLICY_ALIASES.get(normalized)
    if direct:
        return direct
    if normalized in catalog:
        return normalized
    for key, row in catalog.items():
        name = safe_slug(str(row.get("name", "")).strip())
        if normalized == name:
            return key
    return None


def _default_compare_policies(
    metadata: dict[str, Any],
    config: SimulationToolConfig,
) -> list[str]:
    defaults = metadata.get("defaults", {})
    metadata_defaults = (
        defaults.get("comparison_strategies") if isinstance(defaults, dict) else None
    )
    requested = (
        config.default_compare_policies
        if config.default_compare_policies
        else metadata_defaults
    )
    catalog = _policy_catalog(metadata)
    rows: list[str] = []
    for key in requested:
        normalized = _resolve_policy_token(str(key), catalog) or safe_slug(str(key))
        if not normalized:
            continue
        if catalog:
            policy_row = catalog.get(normalized)
            if isinstance(policy_row, dict) and policy_row.get("available") is False:
                continue
        rows.append(normalized)
    if rows:
        return list(dict.fromkeys(rows))
    fallback = _first_available_policy(catalog, DEFAULT_COMPARE_POLICIES)
    return [fallback] if fallback else [_fallback_policy_key()]


def _first_available_policy(
    catalog: dict[str, dict[str, Any]],
    preferred_keys: list[str],
) -> str | None:
    if catalog:
        for key in preferred_keys:
            normalized = safe_slug(key)
            row = catalog.get(normalized)
            if isinstance(row, dict) and row.get("available") is not False:
                return normalized
        for key, row in catalog.items():
            if row.get("available") is not False:
                return key
        return None
    for key in preferred_keys:
        normalized = safe_slug(key)
        if normalized:
            return normalized
    return None


def _select_single_policy(
    *,
    resolved_policies: list[str],
    metadata: dict[str, Any],
    config: SimulationToolConfig,
    default_policy_key: str,
    warnings: list[str],
) -> str:
    catalog = _policy_catalog(metadata)
    if resolved_policies:
        requested = resolved_policies[0]
        if catalog:
            row = catalog.get(requested)
            if isinstance(row, dict) and row.get("available") is False:
                fallback = _first_available_policy(
                    catalog, [default_policy_key, *config.default_compare_policies]
                )
                warnings.append(
                    f"Requested policy '{requested}' was unavailable, so '{fallback}' was used instead."
                )
                return fallback or _fallback_policy_key()
        return requested

    preferred = _resolve_policy_token(
        default_policy_key, catalog
    ) or _first_available_policy(
        catalog,
        [default_policy_key, *config.default_compare_policies],
    )
    if preferred != safe_slug(default_policy_key):
        warnings.append(
            f"Default policy '{safe_slug(default_policy_key)}' was unavailable, so '{preferred}' was used instead."
        )
    return preferred or _fallback_policy_key()


def _resolve_predefined_scenario(query: str, *, metadata: dict[str, Any]) -> str | None:
    normalized_query = safe_slug(query)
    catalog = metadata.get("scenarios")
    if isinstance(catalog, list):
        candidates: list[tuple[int, str]] = []
        for row in catalog:
            if not isinstance(row, dict):
                continue
            key = safe_slug(str(row.get("key", "")).strip())
            name = safe_slug(str(row.get("name", "")).strip())
            description = safe_slug(str(row.get("description", "")).strip())
            for alias in {key, name}:
                if alias:
                    PREDEFINED_SCENARIO_ALIASES.setdefault(alias, key)
            if key and key in normalized_query:
                candidates.append((len(key), key))
            if name and name in normalized_query:
                candidates.append((len(name), key))
            if description and description and description in normalized_query:
                candidates.append((len(description), key))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]

    for alias, key in sorted(
        PREDEFINED_SCENARIO_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if alias and alias in normalized_query:
            return key
    return None


def _scenario_params_lookup(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = metadata.get("scenarios")
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return lookup
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = safe_slug(str(row.get("key", "")).strip())
        params = row.get("params")
        if key and isinstance(params, dict):
            lookup[key] = params
    return lookup


def _resolve_base_scenario_params(
    metadata: dict[str, Any],
    base_scenario_key: str,
) -> dict[str, Any]:
    catalog = _scenario_params_lookup(metadata)
    normalized_key = safe_slug(base_scenario_key)
    if normalized_key and normalized_key in catalog:
        return catalog[normalized_key]
    return catalog.get(_default_scenario_key(), {})


def _custom_scenario_field_specs(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    editor = metadata.get("custom_scenario_editor")
    if not isinstance(editor, dict):
        return []
    fields = editor.get("fields")
    if not isinstance(fields, list):
        return []
    return [row for row in fields if isinstance(row, dict)]


def _field_aliases(spec: dict[str, Any]) -> list[str]:
    key = str(spec.get("key", "")).strip()
    normalized_key = key.replace("_", " ").replace("-", " ").strip().lower()
    label = str(spec.get("label", "")).strip().lower()
    aliases = {
        alias.strip().lower()
        for alias in CUSTOM_SCENARIO_FIELD_ALIASES.get(key, ())
        if alias.strip()
    }
    if normalized_key:
        aliases.add(normalized_key)
    if label:
        aliases.add(label)
    return sorted(aliases, key=len, reverse=True)


def _alias_pattern(aliases: list[str]) -> str:
    parts: list[str] = []
    for alias in aliases:
        escaped = re.escape(alias)
        escaped = escaped.replace(r"\ ", r"\s+").replace(r"\-", r"[-\s]?")
        parts.append(escaped)
    return "|".join(parts)


def _is_negative_adjustment(verb: str) -> bool:
    normalized = safe_slug(verb).replace("-", " ")
    return normalized in NEGATIVE_ADJUSTMENT_VERBS


def _default_field_base_value(spec: dict[str, Any]) -> float:
    key = str(spec.get("key", "")).strip()
    if key in {
        "donor_show_factor",
        "eligible_rate",
        "demand_surge_factor",
        "transport_penalty",
        "lab_time_factor",
        "proc_time_factor",
        "regional_replenishment_rate",
    }:
        return 1.0
    return 0.0


def _coerce_numeric_override(spec: dict[str, Any], value: float) -> int | float:
    numeric = float(value)
    minimum = spec.get("min")
    maximum = spec.get("max")
    if minimum is not None:
        numeric = max(numeric, float(minimum))
    if maximum is not None:
        numeric = min(numeric, float(maximum))
    if str(spec.get("input", "")).strip() == "integer":
        return int(round(numeric))
    return round(numeric, 4)


def _extract_numeric_field_override(
    query: str,
    *,
    spec: dict[str, Any],
    base_value: Any,
) -> int | float | None:
    aliases = _field_aliases(spec)
    if not aliases:
        return None
    alias_pattern = _alias_pattern(aliases)
    if not alias_pattern:
        return None

    base_numeric = (
        as_float(base_value, _default_field_base_value(spec))
        if base_value is not None
        else _default_field_base_value(spec)
    )

    relative_patterns = (
        rf"\b(?P<verb>{'|'.join(POSITIVE_ADJUSTMENT_VERBS + NEGATIVE_ADJUSTMENT_VERBS)})\b"
        rf"(?:\s+\w+){{0,4}}?\s+(?:the\s+)?(?:{alias_pattern})\b"
        rf"(?:\s+\w+){{0,3}}?\s+(?:of\s+|by\s+)(?P<value>\d+(?:\.\d+)?)\s*(?P<pct>%|percent)?",
        rf"\b(?:{alias_pattern})\b(?:\s+\w+){{0,4}}?\s+"
        rf"(?P<verb>{'|'.join(POSITIVE_ADJUSTMENT_VERBS + NEGATIVE_ADJUSTMENT_VERBS)})\b"
        rf"(?:\s+of|\s+by)?\s+(?P<value>\d+(?:\.\d+)?)\s*(?P<pct>%|percent)?",
    )
    for pattern in relative_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        amount = float(match.group("value"))
        direction = -1.0 if _is_negative_adjustment(match.group("verb")) else 1.0
        if match.group("pct"):
            candidate = base_numeric * (1.0 + (direction * amount / 100.0))
        else:
            candidate = base_numeric + (direction * amount)
        return _coerce_numeric_override(spec, candidate)

    absolute_patterns = (
        rf"\bset\s+(?:the\s+)?(?:{alias_pattern})\b(?:\s+(?:to|at|as))?\s+(?P<value>\d+(?:\.\d+)?)",
        rf"\b(?:{alias_pattern})\b\s*(?:=|:)\s*(?P<value>\d+(?:\.\d+)?)",
        rf"\b(?:{alias_pattern})\b(?:\s+(?:to|at|of|is))\s+(?P<value>\d+(?:\.\d+)?)",
        rf"\bwith\s+(?:an?\s+)?(?:{alias_pattern})\b(?:\s+(?:of|at))?\s+(?P<value>\d+(?:\.\d+)?)",
    )
    for pattern in absolute_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return _coerce_numeric_override(spec, float(match.group("value")))
    return None


def _field_override_summary(spec: dict[str, Any], value: Any) -> str:
    label = str(spec.get("label") or spec.get("key") or "override").strip()
    if str(spec.get("input", "")).strip() == "select":
        selected = value
        options = spec.get("options")
        if isinstance(options, list):
            for option in options:
                if not isinstance(option, dict):
                    continue
                if str(option.get("value")) == str(value):
                    selected = option.get("label") or option.get("value") or value
                    break
        return f"{label}={selected}"
    return f"{label}={value}"


def _extract_explicit_field_overrides(
    query: str,
    *,
    metadata: dict[str, Any],
    base_scenario_key: str,
) -> list[ParsedShock]:
    field_specs = _custom_scenario_field_specs(metadata)
    if not field_specs:
        return []

    base_params = _resolve_base_scenario_params(metadata, base_scenario_key)
    rows: list[ParsedShock] = []
    for spec in field_specs:
        key = str(spec.get("key", "")).strip()
        if not key or key in {"sim_hours", "forced_weather"}:
            continue
        input_kind = str(spec.get("input", "")).strip()
        if input_kind not in {"number", "integer"}:
            continue
        value = _extract_numeric_field_override(
            query,
            spec=spec,
            base_value=base_params.get(key),
        )
        if value is None:
            continue
        rows.append(
            ParsedShock(
                kind=f"field_override:{key}",
                magnitude_pct=0.0,
                parameter_overrides={key: value},
                summary=_field_override_summary(spec, value),
            )
        )
    return rows


def _extract_generic_percentage_change(
    query: str,
    *,
    subject_pattern: str,
    subject_verbs: tuple[str, ...],
) -> float | None:
    verb_pattern = "|".join(subject_verbs)
    patterns = (
        rf"(?:{subject_pattern})(?:\s+\w+){{0,4}}?\s+(?:{verb_pattern})"
        rf"(?:\s+of\s+|\s+by\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)?",
        rf"(?:{verb_pattern})(?:\s+\w+){{0,4}}?\s+(?:{subject_pattern})"
        rf"(?:\s+\w+){{0,3}}?\s+(?:of\s+|by\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)?",
        rf"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+(?:{subject_pattern})"
        rf"(?:\s+\w+){{0,4}}?\s+(?:{verb_pattern})",
    )
    return _extract_percentage(query, patterns=patterns)


def _extract_shocks(
    query: str,
    *,
    metadata: dict[str, Any],
    base_scenario_key: str,
) -> list[ParsedShock]:
    base_params = _resolve_base_scenario_params(metadata, base_scenario_key)
    rows: list[ParsedShock] = []

    donor_pct = _extract_generic_percentage_change(
        query,
        subject_pattern=(
            r"donor(?:s| supply| turnout| show[\s-]?up)?"
            r"(?!\s+(?:inter[\s-]?arrival|arrival interval|eligibility|show factor))"
        ),
        subject_verbs=NEGATIVE_ADJUSTMENT_VERBS,
    )
    if donor_pct is not None:
        base_show_factor = as_float(base_params.get("donor_show_factor"), 1.0)
        factor = max(0.1, base_show_factor * (1.0 - (donor_pct / 100.0)))
        rows.append(
            ParsedShock(
                kind="donor_supply_drop",
                magnitude_pct=donor_pct,
                parameter_overrides={"donor_show_factor": round(factor, 4)},
                summary=f"{donor_pct:.1f}% donor supply drop",
            )
        )

    demand_pct = _extract_generic_percentage_change(
        query,
        subject_pattern=(
            r"demand"
            r"(?!\s+(?:forecast|shock|inter[\s-]?arrival|rate|interval|noise))"
        ),
        subject_verbs=(
            "surge",
            "surged",
            "spike",
            "spiked",
            "increase",
            "increased",
            "rise",
            "rose",
            "up",
        ),
    )
    if demand_pct is not None:
        base_demand_factor = as_float(base_params.get("demand_surge_factor"), 1.0)
        rows.append(
            ParsedShock(
                kind="demand_spike",
                magnitude_pct=demand_pct,
                parameter_overrides={
                    "demand_surge_factor": round(
                        base_demand_factor * (1.0 + (demand_pct / 100.0)),
                        4,
                    )
                },
                summary=f"{demand_pct:.1f}% demand spike",
            )
        )

    transport_pct = _extract_generic_percentage_change(
        query,
        subject_pattern=(
            r"(?:transport|shipment|delivery|supply chain)"
            r"(?!\s+(?:lead[\s-]?time noise|noise))"
        ),
        subject_verbs=(
            "delay",
            "delayed",
            "disruption",
            "disrupted",
            "slowdown",
            "slow",
            "slowed",
            "increase",
            "increased",
        ),
    )
    if transport_pct is not None:
        base_transport_penalty = as_float(base_params.get("transport_penalty"), 1.0)
        rows.append(
            ParsedShock(
                kind="transport_disruption",
                magnitude_pct=transport_pct,
                parameter_overrides={
                    "transport_penalty": round(
                        base_transport_penalty * (1.0 + (transport_pct / 100.0)),
                        4,
                    )
                },
                summary=f"{transport_pct:.1f}% transport disruption",
            )
        )

    weather = _extract_weather(query)
    if weather:
        rows.append(
            ParsedShock(
                kind="forced_weather",
                magnitude_pct=0.0,
                parameter_overrides={"forced_weather": weather},
                summary=f"forced weather={weather}",
            )
        )

    rows.extend(
        _extract_explicit_field_overrides(
            query,
            metadata=metadata,
            base_scenario_key=base_scenario_key,
        )
    )
    return rows


def _extract_percentage(query: str, *, patterns: tuple[str, ...]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _extract_weather(query: str) -> str | None:
    lowered = (query or "").lower()
    for option in _weather_options():
        needle = option.replace("_", " ")
        if needle in lowered or option in lowered:
            return option
    return None


def _build_custom_scenario_payload(
    *,
    query: str,
    base_scenario_key: str,
    hours_override: int | None,
    selected_policy_key: str,
    shocks: list[ParsedShock],
) -> dict[str, Any]:
    name_parts = []
    if shocks:
        name_parts.extend(shock.summary for shock in shocks)
    else:
        name_parts.append("custom scenario")
    if hours_override:
        name_parts.append(f"{int(hours_override / 24)}d")
    name = "Orchestrator " + " ".join(name_parts)
    description = "Generated from a natural-language simulation request."
    narrative = f"User query: {query}"
    payload: dict[str, Any] = {
        "name": name[:80],
        "description": description[:280],
        "narrative": narrative[:2000],
        "base_scenario_key": base_scenario_key,
        "recommended_strategy_key": selected_policy_key or _fallback_policy_key(),
    }
    if hours_override is not None:
        payload["sim_hours"] = int(hours_override)
    for shock in shocks:
        payload.update(shock.parameter_overrides)

    # Pre-fill from ML model predictions
    try:
        from ml.core.simulation_ml_bridge import prefill_scenario_params

        blood_type = _extract_blood_type(query)
        ml_overrides = prefill_scenario_params(
            query,
            payload,
            blood_type=blood_type,
            component_type=None,
        )
        # ML overrides are defaults — explicit shocks take priority
        for key, value in ml_overrides.items():
            if key not in payload and not key.startswith("_"):
                payload[key] = value
    except Exception as exc:
        logger.debug("ML bridge prefill skipped: %s", exc)

    return payload


def _run_single_from_request(
    *,
    client: SimulationBackendClient,
    request: SimulationRequest,
    dreamerv3_run_key: str | None,
) -> dict[str, Any]:
    policy_key = request.selected_policy_key or DEFAULT_SIMULATION_POLICY
    include_timeline = _include_timeline()
    logger.debug(
        "simulation_tool.run_single scenario_kind=%s policy=%s",
        request.scenario_kind,
        policy_key,
    )
    if request.scenario_kind == "predefined":
        return client.run_predefined_scenario(
            scenario_key=request.scenario_key or _default_scenario_key(),
            strategy_key=policy_key,
            hours_override=request.hours_override,
            include_timeline=include_timeline,
            dreamerv3_run_key=dreamerv3_run_key
            if _run_key_policy() and policy_key == _run_key_policy()
            else None,
        )
    if not isinstance(request.custom_scenario_payload, dict):
        raise RuntimeError("Missing custom scenario payload.")
    return client.run_custom_scenario(
        custom_payload=request.custom_scenario_payload,
        strategy_key=policy_key,
        hours_override=request.hours_override,
        include_timeline=include_timeline,
        dreamerv3_run_key=dreamerv3_run_key
        if _run_key_policy() and policy_key == _run_key_policy()
        else None,
    )


def _run_compare_from_request(
    *,
    client: SimulationBackendClient,
    request: SimulationRequest,
    dreamerv3_run_key: str | None,
) -> dict[str, Any]:
    compare_keys = list(dict.fromkeys(request.compare_policy_keys))
    if request.scenario_kind == "predefined":
        return client.compare_policies(
            scenario_key=request.scenario_key or _default_scenario_key(),
            strategy_keys=compare_keys,
            hours_override=request.hours_override,
            dreamerv3_run_key=dreamerv3_run_key
            if _run_key_policy() and _run_key_policy() in compare_keys
            else None,
        )
    if not isinstance(request.custom_scenario_payload, dict):
        raise RuntimeError("Missing custom scenario payload.")
    return client.compare_custom_policies(
        custom_payload=request.custom_scenario_payload,
        strategy_keys=compare_keys,
        hours_override=request.hours_override,
        dreamerv3_run_key=dreamerv3_run_key
        if _run_key_policy() and _run_key_policy() in compare_keys
        else None,
    )


def normalize_simulation_result(
    *,
    request: SimulationRequest,
    raw: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if (
        request.mode == "compare"
        and request.compare_policy_keys
        and "summary_rows" in raw
    ):
        return _normalize_compare_result(request=request, raw=raw, metadata=metadata)
    return _normalize_single_result(request=request, raw=raw, metadata=metadata)


def _normalize_single_result(
    *,
    request: SimulationRequest,
    raw: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    summary = raw.get("summary") if isinstance(raw.get("summary"), dict) else {}
    scenario = raw.get("scenario") if isinstance(raw.get("scenario"), dict) else {}
    strategy = raw.get("strategy") if isinstance(raw.get("strategy"), dict) else {}
    shortages = _normalize_shortages(summary)
    inventory = _normalize_inventory(raw.get("inventory_timeline"))
    actions = _normalize_actions(summary, metadata)
    policy_key = safe_slug(
        str(strategy.get("key") or request.selected_policy_key or "")
    )
    metrics = {
        "shortage_rate": _metric(summary, "shortage_rate"),
        "service_rate": _metric(summary, "service_rate"),
        "total_shortage": _metric(summary, "total_shortage"),
        "total_net_requested": _metric(summary, "total_net_requested"),
        "total_transfused": _metric(summary, "total_transfused"),
        "budget_spent": _metric(summary, "budget_spent"),
        "budget_remaining": _metric(summary, "budget_remaining"),
        "episode_score": _metric(summary, "episode_score"),
        "reward_total": _metric(summary, "reward_total"),
        "exact_match_rate": _metric(summary, "exact_match_rate"),
        "compatible_substitution_rate": _metric(
            summary, "compatible_substitution_rate"
        ),
    }
    warnings = list(dict.fromkeys(request.warnings))
    if not shortages["impacted_blood_types"]:
        warnings.append(
            "The backend response did not include blood-type-specific shortage results."
        )
    return {
        "mode": request.mode,
        "scenario": {
            "kind": request.scenario_kind,
            "key": scenario.get("key") or request.scenario_key,
            "name": scenario.get("name") or request.scenario_key or "Custom scenario",
            "description": scenario.get("description") or "",
            "hours": raw.get("hours") or request.hours_override,
            "requested_region": request.requested_region,
            "requested_blood_type": request.requested_blood_type,
            "shocks": [asdict(shock) for shock in request.shocks],
        },
        "policy": {
            "key": policy_key or request.selected_policy_key,
            "name": strategy.get("name") or policy_key or request.selected_policy_key,
            "description": strategy.get("description") or "",
            "dreamerv3_run": raw.get("dreamerv3_run"),
        },
        "metrics": metrics,
        "shortages": shortages,
        "inventory": inventory,
        "recommended_actions": actions,
        "policy_comparison": raw.get("_recommendation_comparison"),
        "assumptions": list(dict.fromkeys(request.assumptions)),
        "warnings": warnings,
    }


def _normalize_compare_result(
    *,
    request: SimulationRequest,
    raw: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    summary_rows = raw.get("summary_rows")
    rows = summary_rows if isinstance(summary_rows, list) else []
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        comparisons.append(
            {
                "policy_key": safe_slug(str(row.get("strategy", "")).strip()),
                "policy_name": str(
                    row.get("strategy_name") or row.get("strategy") or ""
                ).strip(),
                "shortage_rate": _metric(row, "shortage_rate_mean"),
                "service_rate": _metric(row, "service_rate_mean"),
                "total_shortage": _metric(row, "total_shortage_mean"),
                "budget_spent": _metric(row, "budget_spent_mean"),
                "episode_score": _metric(row, "episode_score_mean"),
                "reward_total": _metric(row, "reward_total_mean"),
                "exact_match_rate": _metric(row, "exact_match_rate_mean"),
                "compatible_substitution_rate": _metric(
                    row, "compatible_substitution_rate_mean"
                ),
            }
        )
    comparisons.sort(
        key=lambda row: (
            float(row.get("shortage_rate") or 0.0),
            -float(row.get("service_rate") or 0.0),
            -float(row.get("episode_score") or 0.0),
        )
    )
    best = comparisons[0] if comparisons else None
    warnings = list(dict.fromkeys(request.warnings))
    if request.unknown_policy_tokens:
        warnings.append(
            "Unknown policy names were ignored: "
            + ", ".join(sorted(set(request.unknown_policy_tokens)))
            + "."
        )
    return {
        "mode": "compare",
        "scenario": {
            "kind": request.scenario_kind,
            "key": (
                rows[0].get("scenario")
                if rows and isinstance(rows[0], dict)
                else request.scenario_key
            ),
            "name": (
                rows[0].get("scenario_name")
                if rows and isinstance(rows[0], dict)
                else request.scenario_key or "Custom scenario"
            ),
            "hours": request.hours_override,
            "requested_region": request.requested_region,
            "requested_blood_type": request.requested_blood_type,
            "shocks": [asdict(shock) for shock in request.shocks],
        },
        "policies": comparisons,
        "best_policy": best,
        "assumptions": list(dict.fromkeys(request.assumptions)),
        "warnings": warnings,
        "dreamerv3_run": raw.get("dreamerv3_run"),
    }


def _metric(payload: dict[str, Any], key: str) -> float | int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 4)
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _normalize_shortages(summary: dict[str, Any]) -> dict[str, Any]:
    by_component = summary.get("shortage_by_component")
    component_rows = []
    if isinstance(by_component, dict):
        for component, value in by_component.items():
            amount = as_float(value, 0.0)
            if amount <= 0:
                continue
            component_rows.append(
                {"component": str(component), "units": round(amount, 2)}
            )
    component_rows.sort(key=lambda row: row["units"], reverse=True)

    by_hospital = summary.get("shortage_by_hospital")
    hospital_rows = []
    if isinstance(by_hospital, dict):
        for hospital, value in by_hospital.items():
            amount = as_float(value, 0.0)
            if amount <= 0:
                continue
            hospital_rows.append({"location": str(hospital), "units": round(amount, 2)})
    hospital_rows.sort(key=lambda row: row["units"], reverse=True)

    return {
        "total_units": _metric(summary, "total_shortage"),
        "shortage_rate": _metric(summary, "shortage_rate"),
        "by_component": component_rows,
        "by_location": hospital_rows[:5],
        "impacted_regions": [row["location"] for row in hospital_rows[:5]],
        "impacted_blood_types": [],
    }


def _normalize_inventory(timeline: Any) -> dict[str, Any]:
    if not isinstance(timeline, list) or not timeline:
        return {}
    rows = [row for row in timeline if isinstance(row, dict)]
    if not rows:
        return {}
    first_shortage_hour = None
    for row in rows:
        shortages = as_float(row.get("shortages"), 0.0)
        if shortages > 0:
            first_shortage_hour = as_float(row.get("hour"), 0.0)
            break
    final = rows[-1]
    final_inventory = {
        key: int(as_float(final.get(key), 0.0))
        for key in _inventory_keys()
    }
    total_final_inventory = sum(final_inventory.values())
    return {
        "timeline_points": len(rows),
        "days_until_shortage_onset": round(first_shortage_hour / 24.0, 2)
        if first_shortage_hour is not None
        else None,
        "final_inventory": final_inventory,
        "final_inventory_total_units": total_final_inventory,
    }


def _normalize_actions(
    summary: dict[str, Any], metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    action_lookup = _action_catalog(metadata)
    keys = summary.get("active_action_keys")
    if not isinstance(keys, list):
        return []
    rows: list[dict[str, Any]] = []
    for action_key in keys:
        normalized = safe_slug(str(action_key))
        action = action_lookup.get(normalized, {})
        rows.append(
            {
                "key": normalized,
                "name": str(action.get("name") or normalized).strip(),
                "description": str(action.get("description") or "").strip(),
                "operational_cost": _metric(action, "operational_cost"),
            }
        )
    return rows


def _action_catalog(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = metadata.get("actions")
    catalog: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return catalog
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = safe_slug(str(row.get("key", "")).strip())
        if key:
            catalog[key] = row
    return catalog


def synthesize_simulation_response(normalized: dict[str, Any]) -> str:
    mode = str(normalized.get("mode") or "forecast")
    if mode == "compare":
        return _synthesize_compare_response(normalized)
    if mode == "recommend":
        return _synthesize_recommend_response(normalized)
    return _synthesize_forecast_response(normalized)


def _synthesize_forecast_response(normalized: dict[str, Any]) -> str:
    scenario = normalized.get("scenario", {})
    policy = normalized.get("policy", {})
    metrics = normalized.get("metrics", {})
    shortages = normalized.get("shortages", {})
    inventory = normalized.get("inventory", {})

    lines = [
        (
            f"Scenario simulated: {_scenario_label(scenario)} using "
            f"{policy.get('name') or policy.get('key') or 'the backend default policy'}."
        ),
        _single_run_impact_line(metrics),
    ]

    shortage_line = _shortage_line(shortages)
    if shortage_line:
        lines.append(shortage_line)

    shortage_onset = inventory.get("days_until_shortage_onset")
    if isinstance(shortage_onset, (int, float)):
        lines.append(
            f"First recorded shortages appeared around day {float(shortage_onset):.1f}."
        )

    lines.extend(_assumption_lines(normalized))
    return " ".join(line for line in lines if line)


def _synthesize_recommend_response(normalized: dict[str, Any]) -> str:
    scenario = normalized.get("scenario", {})
    policy = normalized.get("policy", {})
    metrics = normalized.get("metrics", {})
    actions = normalized.get("recommended_actions", [])
    shortages = normalized.get("shortages", {})
    comparison = normalized.get("policy_comparison")

    lines = [
        (
            f"Scenario simulated: {_scenario_label(scenario)} using "
            f"{policy.get('name') or policy.get('key') or DEFAULT_SIMULATION_POLICY}."
        ),
        _single_run_impact_line(metrics),
    ]

    if actions:
        action_names = ", ".join(
            str(action.get("name") or action.get("key")) for action in actions[:5]
        )
        lines.append(f"Recommended actions from the policy: {action_names}.")
    else:
        lines.append(
            "The selected policy did not surface explicit intervention actions in this run."
        )

    shortage_line = _shortage_line(shortages)
    if shortage_line:
        lines.append(shortage_line)

    if isinstance(comparison, dict):
        best = comparison.get("best_policy")
        if isinstance(best, dict):
            lines.append(
                f"In the optional policy comparison, {best.get('policy_name') or best.get('policy_key')} "
                f"performed best on shortage rate before the detailed run above."
            )

    lines.extend(_assumption_lines(normalized))
    return " ".join(line for line in lines if line)


def _synthesize_compare_response(normalized: dict[str, Any]) -> str:
    scenario = normalized.get("scenario", {})
    policies = normalized.get("policies", [])
    best = normalized.get("best_policy") or {}

    lines = [f"Scenario simulated: {_scenario_label(scenario)}."]
    if policies:
        rows = []
        for row in policies[:5]:
            rows.append(
                (
                    f"{row.get('policy_name') or row.get('policy_key')}: shortage "
                    f"{_fmt_pct(row.get('shortage_rate'))}, service {_fmt_pct(row.get('service_rate'))}, "
                    f"unmet {_fmt_num(row.get('total_shortage'))}, budget {_fmt_num(row.get('budget_spent'))}"
                )
            )
        lines.append("Policy comparison: " + " | ".join(rows) + ".")
    if best:
        lines.append(
            f"Best performer: {best.get('policy_name') or best.get('policy_key')} "
            f"because it had the lowest shortage rate in this comparison."
        )
    lines.extend(_assumption_lines(normalized))
    return " ".join(line for line in lines if line)


def _scenario_label(scenario: dict[str, Any]) -> str:
    name = str(scenario.get("name") or scenario.get("key") or "scenario").strip()
    hours = scenario.get("hours")
    if isinstance(hours, (int, float)) and float(hours) > 0:
        return f"{name} over {float(hours) / 24.0:.1f} days"
    return name


def _single_run_impact_line(metrics: dict[str, Any]) -> str:
    shortage = _fmt_pct(metrics.get("shortage_rate"))
    service = _fmt_pct(metrics.get("service_rate"))
    unmet = _fmt_num(metrics.get("total_shortage"))
    requested = _fmt_num(metrics.get("total_net_requested"))
    budget_spent = _fmt_num(metrics.get("budget_spent"))
    return (
        f"Projected impact: shortage rate {shortage}, service rate {service}, "
        f"and {unmet} unmet units out of {requested} net requested; budget spent {budget_spent}."
    )


def _shortage_line(shortages: dict[str, Any]) -> str:
    components = shortages.get("by_component")
    locations = shortages.get("by_location")
    parts: list[str] = []
    if isinstance(components, list) and components:
        summary = ", ".join(
            f"{row['component']} {int(as_float(row['units'], 0.0))}"
            for row in components[:3]
        )
        parts.append(f"Biggest component shortages: {summary}")
    if isinstance(locations, list) and locations:
        summary = ", ".join(
            f"{row['location']} {int(as_float(row['units'], 0.0))}"
            for row in locations[:3]
        )
        parts.append(f"Top impacted locations: {summary}")
    return ". ".join(parts) + ("." if parts else "")


def _assumption_lines(normalized: dict[str, Any]) -> list[str]:
    assumptions = normalized.get("assumptions")
    warnings = normalized.get("warnings")
    lines: list[str] = []
    if isinstance(assumptions, list) and assumptions:
        lines.append(
            "Assumptions: " + "; ".join(str(item) for item in assumptions[:3]) + "."
        )
    if isinstance(warnings, list) and warnings:
        lines.append("Caveats: " + "; ".join(str(item) for item in warnings[:3]) + ".")
    return lines


def _format_simulation_failure(
    *,
    request: SimulationRequest,
    metadata_error: str | None,
    backend_error: str,
) -> str:
    parts = [
        "I recognized this as a simulation request, but the simulation backend could not complete the run.",
        f"Parsed request: mode={request.mode}, scenario={request.scenario_key or 'custom'}, horizon={request.hours_override or 'default'}h.",
    ]
    if request.selected_policy_key:
        parts.append(f"Requested policy: {request.selected_policy_key}.")
    if metadata_error:
        parts.append(f"Metadata lookup failed: {metadata_error}.")
    parts.append(f"Backend error: {backend_error}.")
    return " ".join(parts)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.1f}"
