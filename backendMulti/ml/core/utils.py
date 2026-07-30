"""
Pure utility functions extracted from the legacy ML orchestration layer.
ZERO framework dependency. Only stdlib + numpy.

These helpers were originally private functions in the ML runtime code
(prefixed with `_`) and were moved here as reusable standalone utilities.
"""

import math
import re
from typing import Any

import numpy as np


# Was: _safe_slug() in api.py
def safe_slug(value: str) -> str:
    """Convert any string to a URL-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "model"


# Was: _sigmoid() in api.py
def sigmoid(value: float) -> float:
    """Numerically stable sigmoid function."""
    clipped = max(-50.0, min(50.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


# Was: _numeric() in api.py
def numeric(value: Any) -> float:
    """Convert any value to a float for model input."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0
    return 0.0


# Was: _as_float() in api.py
def as_float(value: Any, default: float) -> float:
    try:
        return float(default) if value is None else float(value)
    except (TypeError, ValueError):
        return float(default)


# Was: _as_int() in api.py
def as_int(value: Any, default: int) -> int:
    try:
        return int(default) if value is None else int(float(value))
    except (TypeError, ValueError):
        return int(default)


# Was: _encode_with_classes() in api.py
def encode_with_classes(value: Any, classes: list[Any]) -> float:
    """Map a categorical value to its index in a class list."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    lowered = text.lower()
    for idx, cls_name in enumerate(classes):
        cls_text = str(cls_name).strip()
        if text == cls_text or lowered == cls_text.lower():
            return float(idx)
    return 0.0


# Was: _is_placeholder_text() in api.py
def is_placeholder_text(value: str | None) -> bool:
    """Detect Swagger 'Try it out' placeholder values."""
    if value is None:
        return True
    return value.strip().lower() in {"", "string", "none", "null", "undefined"}


# Was: _sanitize_features() in api.py
def sanitize_features(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove Swagger junk keys and non-scalar values."""
    return {
        k: v
        for k, v in payload.items()
        if not k.startswith("additionalProp")
        and not isinstance(v, (dict, list, tuple, set))
    }