"""
Seed deterministic Quebec hospital inventory snapshots using a fixed catalog of
real hospitals, size-tiered stock formulas, and one HospitalSupplyFeature row
per hospital so demo predictions have realistic operational context.
"""

from __future__ import annotations

from datetime import datetime, timezone
import random

from django.core.management.base import BaseCommand, CommandError

from inventory.models import BloodSupply, Hospital, HospitalSupplyFeature
from inventory.seed_data import (
    BLOOD_TYPE_PROFILES,
    BLOOD_TYPES,
    HOSPITAL_SIZE_PROFILES,
    QUEBEC_HOSPITALS,
    SEED,
    seeded_random,
)


class Command(BaseCommand):
    help = "Seed Quebec-realistic BloodSupply, Hospital, and HospitalSupplyFeature rows."

    def add_arguments(self, parser):
        parser.add_argument("--hospitals", type=int, default=len(QUEBEC_HOSPITALS))
        parser.add_argument("--reset", action="store_true")

    def handle(self, *args, **options):
        random.seed(SEED)

        num_hospitals = options["hospitals"]
        reset = options["reset"]

        if num_hospitals < 1 or num_hospitals > len(QUEBEC_HOSPITALS):
            raise CommandError(
                f"--hospitals must be between 1 and {len(QUEBEC_HOSPITALS)}."
            )

        selected_hospitals = QUEBEC_HOSPITALS[:num_hospitals]
        expected_supply_count = num_hospitals * len(BLOOD_TYPES)

        if reset:
            from alerts.models import AlertEvent, AlertEventHistory
            from inventory.models import PredictionResult

            self.stdout.write(
                "Resetting BloodSupply, HospitalSupplyFeature, Hospital, alert state, and predictions..."
            )
            # PredictionResult has no FK to Hospital so it must be deleted
            # explicitly — otherwise old rows with stale entity_ids accumulate
            # across reseeds and corrupt the dashboard stockout/forecast cards.
            PredictionResult.objects.all().delete()
            AlertEventHistory.objects.all().delete()
            AlertEvent.objects.all().delete()
            HospitalSupplyFeature.objects.all().delete()
            BloodSupply.objects.all().delete()
            Hospital.objects.all().delete()

        if (
            Hospital.objects.count() >= num_hospitals
            and BloodSupply.objects.count() >= expected_supply_count
            and HospitalSupplyFeature.objects.count() >= num_hospitals
            and not reset
        ):
            self.stdout.write(
                self.style.WARNING("Quebec blood supply seed data already exists. Skipping...")
            )
            return

        seed_timestamp = datetime.now(timezone.utc).replace(microsecond=0)

        hospitals: list[Hospital] = []
        created_hospitals = 0
        updated_hospitals = 0

        for hospital_seed in selected_hospitals:
            hospital, was_created = Hospital.objects.get_or_create(
                hospital_id=hospital_seed.hospital_id,
                defaults={
                    "name": hospital_seed.name,
                    "wilaya": hospital_seed.wilaya,
                },
            )
            if was_created:
                created_hospitals += 1
            elif (
                hospital.name != hospital_seed.name
                or hospital.wilaya != hospital_seed.wilaya
            ):
                hospital.name = hospital_seed.name
                hospital.wilaya = hospital_seed.wilaya
                hospital.save(update_fields=["name", "wilaya"])
                updated_hospitals += 1

            hospitals.append(hospital)

        created_supplies = 0
        total_supplies = 0

        for hospital in hospitals:
            hospital_seed = next(
                row for row in selected_hospitals if row.hospital_id == hospital.hospital_id
            )
            size_profile = HOSPITAL_SIZE_PROFILES[hospital_seed.size_tier]

            for blood_type in BLOOD_TYPES:
                profile = BLOOD_TYPE_PROFILES[blood_type]
                rng = seeded_random("blood-supply", hospital.hospital_id, blood_type)
                volatility = size_profile["volatility"]

                stock_units = round(
                    profile["stock_base"]
                    * size_profile["stock_scale"]
                    * rng.uniform(1.0 - volatility, 1.07 + volatility),
                    2,
                )
                usage_today = round(
                    max(
                        0.8,
                        profile["usage_base"]
                        * size_profile["usage_scale"]
                        * rng.uniform(0.92 - (volatility / 2), 1.1 + volatility),
                    ),
                    2,
                )
                stock_pressure = usage_today / max(stock_units, 1.0)
                stockout_count_90d = min(
                    18,
                    int(
                        round(
                            (profile["scarcity"] * 4.2)
                            + size_profile["shortage_penalty"]
                            + (stock_pressure * 35)
                            + rng.uniform(0.0, 2.2)
                        )
                    ),
                )
                scheduled_surgeries = int(
                    round(
                        size_profile["surgery_base"]
                        * rng.uniform(0.85, 1.18)
                    )
                )
                days_since_last_restock = rng.randint(
                    1,
                    5 if hospital_seed.size_tier == "large" else 12,
                )
                lead_time_days = size_profile["lead_time_days"] + rng.randint(0, 2)

                supply_id = (
                    f"S{hospital.hospital_id[1:]}"
                    f"{blood_type.replace('+', 'P').replace('-', 'N')}"
                )

                _obj, was_created = BloodSupply.objects.get_or_create(
                    supply_id=supply_id,
                    defaults={
                        "hospital": hospital,
                        "blood_product_type": blood_type,
                        "current_stock_units": stock_units,
                        "usage_today": usage_today,
                        "lead_time_days": lead_time_days,
                        "days_since_last_restock": days_since_last_restock,
                        "stockout_count_90d": stockout_count_90d,
                        "scheduled_surgeries_next7d": scheduled_surgeries,
                        "event_timestamp": seed_timestamp,
                    },
                )
                created_supplies += int(was_created)
                total_supplies += 1

        hsf_created = 0
        hsf_updated = 0
        for hospital in hospitals:
            hospital_seed = next(
                row for row in selected_hospitals if row.hospital_id == hospital.hospital_id
            )
            size_profile = HOSPITAL_SIZE_PROFILES[hospital_seed.size_tier]
            hospital_supplies = list(
                BloodSupply.objects.filter(hospital=hospital).only(
                    "current_stock_units",
                    "scheduled_surgeries_next7d",
                    "usage_today",
                )
            )
            total_inventory = sum(row.current_stock_units for row in hospital_supplies)
            total_surgeries = sum(
                row.scheduled_surgeries_next7d for row in hospital_supplies
            )
            total_usage = sum(row.usage_today for row in hospital_supplies)
            rng = seeded_random("hospital-feature", hospital.hospital_id)

            feature_defaults = {
                "temperature": round(
                    15.5 - abs(hospital_seed.latitude - 45.0) * 2.4 + rng.uniform(-3.0, 3.0),
                    1,
                ),
                "rain_mm": round(
                    max(0.0, 4.0 + abs(hospital_seed.longitude + 73.0) * 0.4 + rng.uniform(0.0, 7.5)),
                    1,
                ),
                "holiday": int(hospital_seed.wilaya in {"Montréal", "Québec"} and rng.random() < 0.18),
                "disaster": int(hospital_seed.size_tier == "remote" and rng.random() < 0.12),
                "scheduled_surgeries": max(
                    int(round(total_surgeries / max(len(BLOOD_TYPES), 1))),
                    size_profile["surgery_base"],
                ),
                "trauma_cases": max(
                    1,
                    int(
                        round(
                            size_profile["trauma_base"]
                            + (total_usage / 18.0)
                            + rng.uniform(0.0, 3.0)
                        )
                    ),
                ),
                "current_inventory": int(round(total_inventory)),
            }

            _feature, was_created = HospitalSupplyFeature.objects.update_or_create(
                hospital_id=hospital.hospital_id,
                event_timestamp=seed_timestamp,
                defaults=feature_defaults,
            )
            if was_created:
                hsf_created += 1
            else:
                hsf_updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                "Seeded Quebec hospital inventory: "
                f"{created_hospitals} hospital(s) created, "
                f"{updated_hospitals} updated, "
                f"{created_supplies}/{total_supplies} BloodSupply row(s) created, "
                f"{hsf_created} HospitalSupplyFeature row(s) created, "
                f"{hsf_updated} updated."
            )
        )
