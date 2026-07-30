from __future__ import annotations

import json
import logging
from pathlib import Path

from django.conf import settings

from alerts.models import (
    AlertRule,
    AlertScopeType,
    AlertSeverity,
    AlertTriggerType,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(settings.ML_CONFIG_DIR) / "alert_rule_templates.json"

_SEVERITY_MAP = {v.lower(): v for v in AlertSeverity.values}
_TRIGGER_MAP = {v.lower(): v for v in AlertTriggerType.values}
_SCOPE_MAP = {v.lower(): v for v in AlertScopeType.values}


def _load_templates() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to load alert rule templates from %s: %s", _CONFIG_PATH, exc)
        return {"blood_types": [], "templates": []}


class AlertRuleFactory:
    def generate_for_hospital(
        self,
        hospital_id: str,
        blood_types: list[str] | None = None,
    ) -> list[AlertRule]:
        config = _load_templates()
        types = blood_types or config.get("blood_types", [])
        templates = config.get("templates", [])
        created_rules: list[AlertRule] = []

        for blood_type in types:
            for template in templates:
                payload = self._build_payload(template, hospital_id=hospital_id, blood_type=blood_type)
                if payload is None:
                    continue
                rule, created = AlertRule.objects.get_or_create(
                    name=payload["name"],
                    defaults=payload,
                )
                if created:
                    created_rules.append(rule)

        logger.info(
            "Generated %s alert rule(s) for hospital=%s",
            len(created_rules),
            hospital_id,
        )
        return created_rules

    def _build_payload(self, template: dict, *, hospital_id: str, blood_type: str) -> dict | None:
        try:
            fmt = {"hospital_id": hospital_id, "blood_type": blood_type}
            return {
                "name": template["name_pattern"].format(**fmt),
                "description": template.get("description_pattern", "").format(**fmt),
                "severity_base": _SEVERITY_MAP.get(template.get("severity_base", "warning"), AlertSeverity.WARNING),
                "trigger_type": _TRIGGER_MAP.get(template.get("trigger_type", "threshold_gap"), AlertTriggerType.THRESHOLD_GAP),
                "scope_type": _SCOPE_MAP.get(template.get("scope_type", "hospital"), AlertScopeType.HOSPITAL),
                "scope_ref": hospital_id,
                "conditions": template["conditions"],
                "channels": template.get("channels", ["notification"]),
                "dedup_window_minutes": template.get("dedup_window_minutes", 60),
                "escalation_after_minutes": template.get("escalation_after_minutes", 120),
                "created_by": "rule-factory",
            }
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping invalid template %s: %s", template.get("id"), exc)
            return None
