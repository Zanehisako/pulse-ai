from __future__ import annotations

import re
import json
import os
from pathlib import Path
from typing import Any


BACKEND_ML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DOMAIN_CONFIG_PATH = BACKEND_ML_DIR / "config" / "feature_store_dynamic.json"
_DOMAIN_CONFIG_CACHE: tuple[Path, float, dict[str, Any]] | None = None


def _domain_config() -> dict[str, Any]:
    global _DOMAIN_CONFIG_CACHE
    raw = os.getenv("PIOS_FEATURE_DOMAIN_CONFIG", "").strip()
    path = Path(raw).expanduser().resolve() if raw else DEFAULT_FEATURE_DOMAIN_CONFIG_PATH
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if (
        _DOMAIN_CONFIG_CACHE is not None
        and _DOMAIN_CONFIG_CACHE[0] == path
        and _DOMAIN_CONFIG_CACHE[1] == mtime
    ):
        return _DOMAIN_CONFIG_CACHE[2]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    _DOMAIN_CONFIG_CACHE = (path, mtime, payload)
    return payload


def _component_config() -> dict[str, Any]:
    value = _domain_config().get("component")
    return dict(value) if isinstance(value, dict) else {}


def _coerce_inline_feature_value(raw_value: str) -> Any:
    value = str(raw_value or "").strip().strip("\"'").rstrip(".!?")
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "nan"}:
        return None
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+[eE][-+]?\d+|\d+\.\d*[eE][-+]?\d+)", value):
        return float(value)
    return value


def _extract_inline_kv_features(query: str) -> dict[str, Any]:
    features: dict[str, Any] = {}
    pattern = re.compile(
        r"([A-Za-z][A-Za-z0-9_]*(?:__[A-Za-z0-9_]+)*)\s*=\s*([^,;\n]+)"
    )
    for match in pattern.finditer(query or ""):
        key = match.group(1).strip().lower()
        if key:
            features[key] = _coerce_inline_feature_value(match.group(2))
    return features


def _normalize_blood_component(value: str) -> str:
    normalized = str(value or "").strip().upper().replace(" ", "_")
    aliases = {
        str(key).strip().upper().replace(" ", "_"): str(value).strip().upper()
        for key, value in dict(_component_config().get("aliases") or {}).items()
        if str(key).strip() and str(value).strip()
    }
    return aliases.get(normalized, normalized)


def _component_terms_regex() -> str:
    config = _component_config()
    terms = []
    for value in list(config.get("values") or []) + list(
        dict(config.get("aliases") or {}).keys()
    ):
        normalized = str(value or "").strip()
        if normalized:
            terms.append(re.escape(normalized.lower().replace("_", " ")))
    terms = sorted(set(terms), key=len, reverse=True)
    return "|".join(terms)


def _normalize_blood_group(value: str) -> str:
    return str(value or "").strip().upper()


def _nl_extraction_config() -> dict[str, Any]:
    value = _domain_config().get("natural_language_extraction")
    return dict(value) if isinstance(value, dict) else {}


def _configured_patterns(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _first_configured_match(patterns: Any, text: str) -> str | None:
    matches = _configured_match_values(patterns, text, max_matches=1)
    return matches[0] if matches else None


def _configured_match_values(
    patterns: Any,
    text: str,
    *,
    unique: bool = True,
    max_matches: int | None = None,
) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for pattern in _configured_patterns(patterns):
        for match in re.finditer(pattern, text):
            groups = [group for group in match.groups() if group not in {None, ""}]
            raw_value = "".join(groups) if groups else match.group(0)
            value = str(raw_value or "").strip()
            if not value:
                continue
            if unique:
                normalized = value.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
            values.append(value)
            if max_matches is not None and max_matches > 0 and len(values) >= max_matches:
                return values
    return values


def _coerce_configured_feature_value(value: str, value_type: str) -> Any:
    if value_type == "int":
        return int(value)
    if value_type == "float":
        return float(value)
    if value_type == "upper":
        return value.upper()
    if value_type == "capitalize":
        return value.capitalize()
    return value


def _apply_transform(value: str, transform: str) -> str:
    if transform == "upper":
        return value.upper()
    if transform == "lower":
        return value.lower()
    if transform == "capitalize":
        return value.capitalize()
    return value


def _extract_configured_numeric_features(
    *,
    text: str,
    extracted: dict[str, Any],
    config: dict[str, Any],
) -> None:
    rows = config.get("numeric_fields")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        field = str(row.get("field") or "").strip()
        if not field or field in extracted:
            continue
        value = _first_configured_match(row.get("patterns"), text)
        if value is None:
            continue
        extracted[field] = _coerce_configured_feature_value(
            value,
            str(row.get("type") or "").strip().lower(),
        )


def _extract_configured_entity_features(
    *,
    text: str,
    extracted: dict[str, Any],
    config: dict[str, Any],
) -> None:
    rows = config.get("entity_patterns")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        field = str(row.get("field") or "").strip()
        if not field or field in extracted:
            continue

        many = bool(row.get("many"))
        max_matches_raw = row.get("max_matches")
        max_matches = (
            max_matches_raw
            if isinstance(max_matches_raw, int) and max_matches_raw > 0
            else None
        )
        unique = bool(row.get("unique", True))
        patterns = row.get("pattern")
        if many:
            values = _configured_match_values(
                patterns,
                text,
                unique=unique,
                max_matches=max_matches,
            )
            if not values:
                continue
            transformed = [
                _apply_transform(
                    value,
                    str(row.get("transform") or "").strip().lower(),
                )
                for value in values
            ]
            extracted[field] = transformed
            id_field = str(row.get("id_field") or "").strip()
            if id_field and len(transformed) == 1 and id_field not in extracted:
                extracted[id_field] = transformed[0]
            continue

        value = _first_configured_match(patterns, text)
        if value is None:
            continue
        transformed = _apply_transform(
            value,
            str(row.get("transform") or "").strip().lower(),
        )
        extracted[field] = transformed
        id_field = str(row.get("id_field") or "").strip()
        if id_field:
            extracted.setdefault(id_field, transformed)


def extract_nl_features(query: str) -> dict[str, Any]:
    """
    Parse common feature values from a natural language query.

    Examples:
        "35 year old male"           → {"age": 35, "sex": "M"}
        "BMI 24.5"                   → {"bmi": 24.5}
        "3 donations in last 12 months" → {"donation_count_last_12m": 3}
        "hospital East blood type A-"  → {"hospital": "East", "blood_type": "A-"}
    """
    query = query or ""
    text = query.lower()
    extracted: dict[str, Any] = _extract_inline_kv_features(query)
    extraction_config = _nl_extraction_config()

    _extract_configured_numeric_features(
        text=text,
        extracted=extracted,
        config=extraction_config,
    )

    # Blood group and component. Keep blood_type for backward compatibility,
    # but stockout code treats blood_group and blood_component separately.
    if "blood_group" not in extracted:
        blood_group = _first_configured_match(
            extraction_config.get("blood_group_patterns"),
            text,
        )
        if blood_group:
            extracted["blood_group"] = _normalize_blood_group(blood_group)
            extracted.setdefault("blood_type", extracted["blood_group"])

    component_terms = _component_terms_regex()
    if component_terms:
        component = None
        context_terms = _configured_patterns(extraction_config.get("component_context_terms"))
        if context_terms:
            context_pattern = "|".join(
                re.escape(term.lower().replace("_", " "))
                for term in sorted(context_terms, key=len, reverse=True)
            )
            component = re.search(
                rf"\b(?:{context_pattern})\s*[:=]?\s*({component_terms})\b",
                text,
            )
        if not component:
            component = re.search(rf"\b({component_terms})\b", text)
        if component:
            extracted["blood_component"] = _normalize_blood_component(component.group(1))

    sex_values = extraction_config.get("sex_values")
    if isinstance(sex_values, dict) and "sex" not in extracted:
        terms = sorted(
            [str(key).strip().lower() for key in sex_values if str(key).strip()],
            key=len,
            reverse=True,
        )
        if terms:
            sex = re.search(rf"\b({'|'.join(re.escape(term) for term in terms)})\b", text)
            if sex:
                extracted["sex"] = str(sex_values.get(sex.group(1), "")).strip()

    _extract_configured_entity_features(
        text=text,
        extracted=extracted,
        config=extraction_config,
    )

    flag_terms = extraction_config.get("flag_terms")
    if isinstance(flag_terms, dict):
        for field, terms in flag_terms.items():
            clean_field = str(field or "").strip()
            if not clean_field or clean_field in extracted:
                continue
            if any(str(term).strip().lower() in text for term in _configured_patterns(terms)):
                extracted[clean_field] = 1

    return extracted


def _normalize_name(name: Any) -> str:
    value = str(name or "").strip().lower()
    if "__" in value:
        value = value.split("__", 1)[1]
    if ":" in value:
        value = value.split(":", 1)[1]
    return value


def _tokenize(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9+-]+", (text or "").lower()) if t}


def _example_texts(examples: Any, kind: str) -> list[str]:
    if not isinstance(examples, list):
        return []
    return [
        " ".join(str(row.get(key, "")) for key in ("user_query", "why"))
        for row in examples
        if isinstance(row, dict) and str(row.get("kind", "")).lower() == kind
    ]


def route_models(query: str, candidates: list, top_k: int) -> list:
    
    # Rank ML models from their own metadata: description, examples, aliases, and inputs.
    
    lowered = query.lower()
    tokens = _tokenize(lowered)
    extracted_features = {
        _normalize_name(key)
        for key, value in extract_nl_features(query).items()
        if value is not None and _normalize_name(key)
    }

    scored_rows: list[tuple[float, object]] = []
    for rt in candidates:
        description = str(getattr(rt, "description", "") or "").lower()
        feature_names = [
            str(feature).lower() for feature in getattr(rt, "feature_names", []) or []
        ]
        aliases = [
            str(alias).lower() for alias in getattr(rt, "aliases", []) or [] if alias
        ]
        examples = getattr(rt, "examples", []) or []
        good_examples = " ".join(_example_texts(examples, "good"))
        bad_examples = " ".join(_example_texts(examples, "bad"))
        metadata_text = " ".join(
            [
                str(getattr(rt, "model_id", "") or ""),
                description,
                " ".join(feature_names),
                " ".join(aliases),
            ]
        )

        model_tokens = _tokenize(metadata_text)
        good_tokens = _tokenize(good_examples)
        bad_tokens = _tokenize(bad_examples)
        normalized_features = {_normalize_name(name) for name in feature_names}
        input_overlap = len(extracted_features.intersection(normalized_features))
        text_overlap = len(tokens.intersection(model_tokens))
        good_overlap = len(tokens.intersection(good_tokens))
        bad_overlap = len(tokens.intersection(bad_tokens))
        score = (
            float(text_overlap)
            + (float(good_overlap) * 2.0)
            + (float(input_overlap) * 5.0)
            - (float(bad_overlap) * 2.0)
        )

        scored_rows.append((score, rt))

    ranked = [rt for _, rt in sorted(scored_rows, key=lambda r: r[0], reverse=True)]
    return ranked[:top_k]
