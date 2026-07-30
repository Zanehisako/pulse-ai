from __future__ import annotations

from django.core.management.base import BaseCommand

from alerts.services.rule_factory import AlertRuleFactory
from inventory.models import Hospital


class Command(BaseCommand):
    help = "Generate alert rules for one hospital or all hospitals."

    def add_arguments(self, parser):
        parser.add_argument("--hospital", dest="hospital_id")

    def handle(self, *args, **options):
        hospital_id = options.get("hospital_id")
        factory = AlertRuleFactory()
        created_count = 0

        if hospital_id:
            created_count = len(factory.generate_for_hospital(hospital_id=hospital_id))
            self.stdout.write(
                self.style.SUCCESS(
                    f"Generated {created_count} alert rule(s) for hospital {hospital_id}."
                )
            )
            return

        for current_hospital_id in Hospital.objects.order_by("hospital_id").values_list(
            "hospital_id",
            flat=True,
        ):
            created_count += len(factory.generate_for_hospital(hospital_id=current_hospital_id))

        self.stdout.write(self.style.SUCCESS(f"Generated {created_count} alert rule(s)."))
