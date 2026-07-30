"""
Seed deterministic Quebec donor profiles by weighting donors to regional
population centres, assigning French-Canadian names, and generating consistent
city, cluster, recency, and eligibility patterns for dashboard demos.
"""

from __future__ import annotations

from datetime import datetime, timezone
import random

from django.core.management.base import BaseCommand

from inventory.models import Donor
from inventory.seed_data import (
    BLOOD_TYPE_WEIGHTS,
    CITIES_BY_ID,
    FRENCH_QUEBEC_GIVEN_NAMES,
    FRENCH_QUEBEC_SURNAMES,
    QUEBEC_CITIES,
    SEED,
    allocate_counts,
    seeded_random,
)


def _build_city_plan(total: int) -> list[str]:
    quotas = allocate_counts(
        total,
        [(city.city_id, city.population_weight) for city in QUEBEC_CITIES],
    )
    plan: list[str] = []
    for city in QUEBEC_CITIES:
        plan.extend([city.city_id] * quotas[city.city_id])
    return plan


class Command(BaseCommand):
    help = "Seed Quebec-realistic donor features safely (idempotent)."

    def add_arguments(self, parser):
        parser.add_argument("--count", type=int, default=200)
        parser.add_argument("--reset", action="store_true")

    def handle(self, *args, **options):
        random.seed(SEED)

        count = options["count"]
        reset = options["reset"]

        if reset:
            self.stdout.write("Resetting donors...")
            Donor.objects.all().delete()

        existing_count = Donor.objects.count()
        updated_existing = 0

        for donor in Donor.objects.all().only("id", "name", "wilaya", "city_id", "cluster_id"):
            city = CITIES_BY_ID.get(donor.city_id)
            should_update = False

            if not donor.name or not donor.name.strip():
                rng = seeded_random("donor-name-backfill", donor.donor_id)
                donor.name = (
                    f"{rng.choice(FRENCH_QUEBEC_GIVEN_NAMES)} "
                    f"{rng.choice(FRENCH_QUEBEC_SURNAMES)}"
                )
                should_update = True

            if city is not None and donor.wilaya != city.wilaya:
                donor.wilaya = city.wilaya
                should_update = True

            if city is not None and donor.cluster_id != city.cluster_id:
                donor.cluster_id = city.cluster_id
                should_update = True

            if should_update:
                donor.save(update_fields=["name", "wilaya", "cluster_id"])
                updated_existing += 1

        if existing_count >= count and not reset:
            self.stdout.write(
                self.style.WARNING(
                    f"{existing_count} donors already exist (>= {count}). Skipping..."
                )
            )
            if updated_existing:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Backfilled {updated_existing} existing donor(s) with Quebec metadata."
                    )
                )
            return

        city_plan = _build_city_plan(count)
        blood_types = list(BLOOD_TYPE_WEIGHTS.keys())
        blood_weights = list(BLOOD_TYPE_WEIGHTS.values())
        now_utc = datetime.now(timezone.utc).replace(microsecond=0)

        seeded = 0

        for idx in range(existing_count + 1, count + 1):
            donor_id = f"D{idx:07d}"
            city = CITIES_BY_ID[city_plan[idx - 1]]
            rng = seeded_random("donor", donor_id)

            needs_recovery_time = (idx % 5 == 0)
            if needs_recovery_time:
                # Split recent donors across three recency bands so that D+7,
                # D+30, and D+90 dashboard windows all show realistic activity.
                # bucket 0 → 1-6 days ago  (visible in D+7)
                # bucket 1 → 7-29 days ago (visible in D+30)
                # bucket 2 → 30-84 days ago (visible in D+90 only)
                bucket = (idx // 5) % 3
                if bucket == 0:
                    recency_days = rng.randint(1, 6)
                elif bucket == 1:
                    recency_days = rng.randint(7, 29)
                else:
                    recency_days = rng.randint(30, 84)
                days_until_eligible = max(1, min(56, 56 - recency_days + rng.randint(0, 6)))
                frequency_365 = rng.randint(2, 5)
                availability = 0
            else:
                recency_days = rng.randint(90, 540)
                frequency_365 = (
                    rng.randint(1, 4)
                    if recency_days <= 220
                    else rng.randint(0, 2)
                )
                days_until_eligible = 0
                availability = int(rng.random() < 0.74)

            lat = round(
                city.latitude + rng.uniform(-city.latitude_jitter, city.latitude_jitter),
                6,
            )
            lon = round(
                city.longitude + rng.uniform(-city.longitude_jitter, city.longitude_jitter),
                6,
            )

            Donor.objects.create(
                donor_id=donor_id,
                name=(
                    f"{rng.choice(FRENCH_QUEBEC_GIVEN_NAMES)} "
                    f"{rng.choice(FRENCH_QUEBEC_SURNAMES)}"
                ),
                wilaya=city.wilaya,
                event_timestamp=now_utc,
                city_id=city.city_id,
                lat=lat,
                lon=lon,
                availability=availability,
                blood_group=rng.choices(blood_types, weights=blood_weights, k=1)[0],
                recency_days=recency_days,
                frequency_365=frequency_365,
                days_until_eligible=days_until_eligible,
                cluster_id=city.cluster_id,
            )
            seeded += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Seeded {seeded} new Quebec donor row(s) (total now: {Donor.objects.count()})."
            )
        )
