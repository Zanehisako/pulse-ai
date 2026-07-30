from __future__ import annotations

from django.db.models.signals import post_save
from django.dispatch import receiver

from inventory.models import Hospital


@receiver(post_save, sender=Hospital)
def create_rules_for_new_hospital(sender, instance, created, **kwargs):
    if created:
        from alerts.services.rule_factory import AlertRuleFactory

        AlertRuleFactory().generate_for_hospital(hospital_id=instance.hospital_id)
