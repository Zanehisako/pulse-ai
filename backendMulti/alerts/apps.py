import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class AlertsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "alerts"
    verbose_name = "Alerts"

    def ready(self):
        import alerts.signals  # noqa: F401
        from django.db.models.signals import post_migrate

        post_migrate.connect(_auto_seed_rules, sender=self)


def _auto_seed_rules(sender, **kwargs):
    """Auto-seed alert rules after migrations if the table is empty."""
    try:
        from alerts.models import AlertRule

        if AlertRule.objects.exists():
            return

        from alerts.services.rule_factory import AlertRuleFactory
        from inventory.models import Hospital

        hospital_ids = list(
            Hospital.objects.order_by("hospital_id").values_list("hospital_id", flat=True)
        )
        if not hospital_ids:
            return

        factory = AlertRuleFactory()
        total = sum(len(factory.generate_for_hospital(hospital_id=hid)) for hid in hospital_ids)
        logger.info("Auto-seeded %d alert rule(s) for %d hospital(s).", total, len(hospital_ids))
    except Exception as exc:
        logger.debug("Alert rule auto-seed skipped: %s", exc)
