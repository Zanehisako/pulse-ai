from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TrendDetector:
    def detect_declining_trend(
        self,
        entity_id: str,
        blood_type: str,
        window: int = 7,
    ) -> tuple[bool, float, float]:
        import numpy as np
        from inventory.models import PredictionResult

        try:
            latest_window = list(
                PredictionResult.objects.filter(
                    hospital_id=entity_id,
                    blood_type=blood_type,
                )
                .order_by("-predicted_for_date")
                .values_list("predicted_for_date", "predicted_value")[:window]
            )
            recent = list(reversed(latest_window))

            if len(recent) < 4:
                return False, 0.0, 0.0

            x = np.arange(len(recent), dtype=float)
            y = np.array([r[1] for r in recent], dtype=float)

            slope, _ = np.polyfit(x, y, 1)
            if np.std(y) == 0.0:
                return False, 0.0, 0.0
            correlation = float(np.corrcoef(x, y)[0, 1])
            confidence = abs(correlation)

            is_declining = slope < 0 and confidence > 0.7

            logger.debug(
                "Trend for %s/%s: slope=%.4f confidence=%.2f declining=%s",
                entity_id,
                blood_type,
                slope,
                confidence,
                is_declining,
            )

            return is_declining, float(slope), confidence

        except Exception as exc:
            logger.error("TrendDetector failed: %s", exc)
            return False, 0.0, 0.0