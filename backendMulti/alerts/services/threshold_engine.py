from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum number of historical rows required before we trust a dynamic
# threshold computation. Below this, we fall back to the static value.
MIN_HISTORY_FOR_DYNAMIC = 14
# Maximum window we look back for dynamic threshold computations.
MAX_HISTORY_WINDOW = 90


@dataclass
class ThresholdResult:
    """
    Rich threshold computation result for observability.

    - value: the resolved threshold (dynamic when possible, fallback otherwise)
    - mode: which dynamic mode was requested
    - n_samples: how many history rows were considered
    - fallback_used: True when the static fallback was used
    - reason: human-readable explanation
    """
    value: Optional[float]
    mode: str
    n_samples: int
    fallback_used: bool
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


class ThresholdEngine:
    def compute_with_metadata(
        self,
        entity_id: str,
        blood_type: str,
        mode: str,
        percentile: int = 10,
        fallback: Optional[float] = None,
    ) -> ThresholdResult:
        """
        Compute the threshold and return rich metadata about how it was
        derived, so callers can stamp observability fields on the alert event.
        """
        import numpy as np
        from inventory.models import PredictionResult

        try:
            history = list(
                PredictionResult.objects.filter(
                    hospital_id=entity_id,
                    blood_type=blood_type,
                )
                .order_by("-predicted_for_date")
                .values_list("predicted_value", flat=True)[:MAX_HISTORY_WINDOW]
            )
        except Exception as exc:
            logger.error("ThresholdEngine.compute_with_metadata failed: %s — using fallback", exc)
            return ThresholdResult(
                value=fallback,
                mode=mode,
                n_samples=0,
                fallback_used=True,
                reason=f"history_query_failed: {exc.__class__.__name__}",
            )

        n = len(history)

        if n < MIN_HISTORY_FOR_DYNAMIC:
            logger.debug(
                "Insufficient history for %s/%s (%d rows < %d) — using fallback %s",
                entity_id,
                blood_type,
                n,
                MIN_HISTORY_FOR_DYNAMIC,
                fallback,
            )
            return ThresholdResult(
                value=fallback,
                mode=mode,
                n_samples=n,
                fallback_used=True,
                reason=f"insufficient_history (n={n}<{MIN_HISTORY_FOR_DYNAMIC})",
            )

        values = np.array(history, dtype=float)

        if mode == "zscore":
            value = float(values.mean() - 2 * values.std())
        elif mode == "percentile":
            value = float(np.percentile(values, percentile))
        elif mode == "mad":
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            value = float(median - 3 * mad)
        else:
            logger.warning("Unknown threshold mode '%s' — using fallback", mode)
            return ThresholdResult(
                value=fallback,
                mode=mode,
                n_samples=n,
                fallback_used=True,
                reason=f"unknown_mode: {mode!r}",
            )

        logger.info(
            "Dynamic threshold [%s] for %s/%s = %.4f (n=%d)",
            mode,
            entity_id,
            blood_type,
            value,
            n,
        )
        return ThresholdResult(
            value=value,
            mode=mode,
            n_samples=n,
            fallback_used=False,
            reason="ok",
        )

    def compute(
        self,
        entity_id: str,
        blood_type: str,
        mode: str,
        percentile: int = 10,
        fallback: Optional[float] = None,
    ) -> Optional[float]:
        """Backward-compatible scalar API. Prefer compute_with_metadata."""
        return self.compute_with_metadata(
            entity_id=entity_id,
            blood_type=blood_type,
            mode=mode,
            percentile=percentile,
            fallback=fallback,
        ).value
