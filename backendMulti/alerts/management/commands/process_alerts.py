from django.core.management.base import BaseCommand

from alerts.services.rule_engine import AlertEngine


class Command(BaseCommand):
    help = "Process stale alert rules and escalate matching open alerts."

    def handle(self, *args, **options):
        count = AlertEngine().process_stale_alerts()
        self.stdout.write(self.style.SUCCESS(f"Processed {count} alert(s)."))
