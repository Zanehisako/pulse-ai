from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Preload configured scheduled prediction models from MLflow"

    def add_arguments(self, parser):
        parser.add_argument(
            "--job",
            action="append",
            default=None,
            help="Preload only the configured scheduled prediction job id. Can be repeated.",
        )

    def handle(self, *args, **options):
        from inventory.tasks import preload_configured_scheduled_models

        job_ids = set(options["job"]) if options["job"] else None
        self.stdout.write("Preloading configured scheduled prediction models...")
        summary = preload_configured_scheduled_models(job_ids=job_ids)

        failures = 0
        for row in summary.get("models", []):
            status = row.get("status", "unknown")
            model_id = row.get("model_id", "")
            version = row.get("model_version", "")
            if status == "ok":
                self.stdout.write(self.style.SUCCESS(f"  {model_id}@{version}: ready"))
            else:
                failures += 1
                detail = row.get("error", "")
                suffix = f" ({detail})" if detail else ""
                self.stdout.write(
                    self.style.WARNING(f"  {model_id}@{version}: {status}{suffix}")
                )

        if failures:
            raise SystemExit(1)
