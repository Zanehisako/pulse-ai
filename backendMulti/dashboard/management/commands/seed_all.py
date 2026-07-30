"""
Run the full deterministic Quebec demo seed pipeline: hospitals and supply,
donors, historical prediction rows, fresh prediction upserts, then dashboard
snapshots so one command rebuilds the end-to-end demo state.
"""
import time

from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Full seed pipeline: supply -> rules -> donors -> prediction history -> predictions -> dashboard"

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true", help="Reset seed-owned tables before reseeding.")
        parser.add_argument("--force", action="store_true", help="Wipe and reseed all data even if rows exist.")
        parser.add_argument("--skip-supply", action="store_true", help="Skip seed_blood_supply")
        parser.add_argument("--skip-donors", action="store_true", help="Skip seed_donors")
        parser.add_argument("--skip-rules", action="store_true", help="Skip generate_alert_rules")
        parser.add_argument(
            "--skip-predict-history",
            action="store_true",
            help="Skip the synthetic prediction history seed (no MLflow needed).",
        )
        parser.add_argument(
            "--skip-predict-live",
            action="store_true",
            help="Skip live ML prediction calls (requires MLflow / loaded models).",
        )
        parser.add_argument(
            "--skip-predict",
            action="store_true",
            help="(deprecated) Skip both prediction history and live predictions. "
                 "Prefer --skip-predict-live so synthetic history can still seed.",
        )
        parser.add_argument("--wait", type=int, default=5, help="Seconds to wait after predictions (default 5)")

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("\nPIOS Full Seed Pipeline\n"))
        force = options.get("force", False)
        reset = options["reset"] or force

        # --skip-predict (legacy) implies both new flags
        skip_history = options["skip_predict_history"] or options["skip_predict"]
        skip_live = options["skip_predict_live"] or options["skip_predict"]

        # -- Idempotency guard -------------------------------------------------
        from inventory.models import BloodSupply
        data_exists = BloodSupply.objects.exists()

        if data_exists and not force:
            self.stdout.write(
                self.style.WARNING(
                    "Data already exists -- skipping seed. "
                    "Run with --force to reseed from scratch."
                )
            )
        else:
            # -- 1. Blood supply -----------------------------------------------
            if not options["skip_supply"]:
                self.stdout.write("1/6 Seeding BloodSupply...")
                call_command("seed_blood_supply", reset=reset, verbosity=0)
                self.stdout.write(self.style.SUCCESS("    v BloodSupply done"))
            else:
                self.stdout.write("1/6 BloodSupply -- skipped")

            # -- 2. Alert rules (need Hospitals to exist) ----------------------
            # We do this explicitly rather than relying solely on the
            # post_save signal so that rule generation runs even when
            # hospitals are inserted via bulk_create (which bypasses signals).
            if not options["skip_rules"]:
                self.stdout.write("2/6 Generating alert rules...")
                try:
                    call_command("generate_alert_rules", verbosity=0)
                    self.stdout.write(self.style.SUCCESS("    v Alert rules done"))
                except Exception as exc:
                    self.stdout.write(
                        self.style.WARNING(f"    ! Alert rule generation failed: {exc}")
                    )
            else:
                self.stdout.write("2/6 Alert rules -- skipped")

            # -- 3. Donors -----------------------------------------------------
            if not options["skip_donors"]:
                self.stdout.write("3/6 Seeding Donors...")
                call_command("seed_donors", reset=reset, verbosity=0)
                self.stdout.write(self.style.SUCCESS("    v Donors done"))
            else:
                self.stdout.write("3/6 Donors -- skipped")

            # -- 4. Synthetic prediction history (NO MLflow dependency) --------
            # Trend detection (>=4 rows) and dynamic thresholds (>=14 rows)
            # rely on this table. Keeping it independent of MLflow guarantees
            # the alert engine is alive on first run even if MLflow is down.
            if not skip_history:
                self.stdout.write("4/6 Seeding prediction history (synthetic)...")
                call_command("seed_prediction_history", reset=reset, verbosity=0)
                self.stdout.write(self.style.SUCCESS("    v Prediction history done"))
            else:
                self.stdout.write("4/6 Prediction history -- skipped")

            # -- 5. Live ML predictions (requires MLflow / loaded models) ------
            if not skip_live:
                self.stdout.write("5/6 Running dashboard predictions (ML)...")
                try:
                    from inventory.tasks import run_dashboard_predictions
                    run_dashboard_predictions()
                    self.stdout.write(self.style.SUCCESS("    v Predictions done"))
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f"    ! Predictions failed: {e}"))
                    self.stdout.write(
                        "    Dashboard will show without fresh stockout and forecast predictions -- re-run with --skip-supply --skip-donors later"
                    )

                if options["wait"] > 0:
                    self.stdout.write(f"    Waiting {options['wait']}s...")
                    time.sleep(options["wait"])
            else:
                self.stdout.write("5/6 Predictions -- skipped")

        # -- 6. Dashboard snapshot (always runs so every restart gets a fresh
        #        sync regardless of whether seeding was skipped) ---------------
        self.stdout.write("6/6 Syncing dashboard...")
        call_command("seed_dashboard_from_inventory", force=True, verbosity=1)
        self.stdout.write(self.style.SUCCESS("    v Dashboard done"))

        self.stdout.write(self.style.SUCCESS("\nAll done -- refresh the dashboard\n"))

        # -- Backfill BloodSupplySnapshot (always runs, even when seed skipped) -
        from django.utils.timezone import now as tz_now
        from datetime import timedelta
        import random
        from inventory.models import BloodSupplySnapshot

        if not BloodSupplySnapshot.objects.exists():
            self.stdout.write("Backfilling BloodSupplySnapshot history (24h)...")
            supplies = list(BloodSupply.objects.select_related("hospital").all())
            base_time = tz_now()
            snapshots = []
            for supply in supplies:
                for hours_ago in range(24, 0, -1):
                    drift = random.uniform(0.92, 1.08)
                    snapshots.append(BloodSupplySnapshot(
                        hospital=supply.hospital,
                        blood_product_type=supply.blood_product_type,
                        current_stock_units=round(supply.current_stock_units * drift, 2),
                        usage_today=round(supply.usage_today * drift, 2),
                        days_since_last_restock=supply.days_since_last_restock,
                        recorded_at=base_time - timedelta(hours=hours_ago),
                    ))
            BloodSupplySnapshot.objects.bulk_create(snapshots)
            self.stdout.write(self.style.SUCCESS(
                f"Backfilled {len(snapshots)} snapshot rows."
            ))
        else:
            self.stdout.write("BloodSupplySnapshot already has rows -- skipping backfill.")
