from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Run configured scheduled ML predictions, then sync dashboard snapshots"

    def add_arguments(self, parser):
        parser.add_argument(
            "--job",
            action="append",
            default=None,
            help="Run only the configured scheduled prediction job id. Can be repeated.",
        )
        parser.add_argument(
            "--skip-dashboard",
            action="store_true",
            help="Skip dashboard snapshot sync after prediction jobs.",
        )

    def handle(self, *args, **options):
        from inventory.tasks import run_configured_scheduled_predictions

        job_ids = set(options["job"]) if options["job"] else None
        self.stdout.write("Running configured scheduled predictions...")
        summary = run_configured_scheduled_predictions(job_ids=job_ids)

        for row in summary.get("jobs", []):
            status_text = row.get("status", "unknown")
            saved = row.get("saved", 0)
            errors = row.get("errors", 0)
            job_id = row.get("id", "")
            if status_text == "ok":
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  {job_id}: saved={saved}, errors={errors}"
                    )
                )
            elif status_text == "disabled":
                self.stdout.write(self.style.WARNING(f"  {job_id}: disabled"))
            else:
                detail = row.get("error", "")
                suffix = f" ({detail})" if detail else ""
                self.stdout.write(
                    self.style.WARNING(f"  {job_id}: {status_text}{suffix}")
                )

        if not options["skip_dashboard"]:
            self.stdout.write("Syncing dashboard...")
            call_command("seed_dashboard_from_inventory", force=True, verbosity=1)
            self.stdout.write(self.style.SUCCESS("Done"))
