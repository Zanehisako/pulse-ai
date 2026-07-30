from __future__ import annotations

from datetime import date
from unittest.mock import patch

from django.core.management import call_command
from django.test import SimpleTestCase, TestCase

from alerts.models import AlertEvent, AlertEventHistory
from inventory.models import BloodSupply, Donor, Hospital, HospitalSupplyFeature, PredictionResult
from inventory.seed_data import (
    BLOOD_TYPES,
    CITIES_BY_ID,
    QUEBEC_CITIES,
    QUEBEC_HOSPITALS,
    quebec_seasonality_modifier,
)


class SeedBloodSupplyCommandTests(TestCase):
    def test_reset_clears_alert_state_and_populates_hospitals_supply_and_features(self):
        event = AlertEvent.objects.create(
            event_key="stale-seed-event",
            title="Stale seed event",
            message="Should be removed by reset seeding.",
        )
        AlertEventHistory.objects.create(event=event, action="created")

        call_command("seed_blood_supply", reset=True, verbosity=0)

        self.assertEqual(AlertEvent.objects.count(), 0)
        self.assertEqual(AlertEventHistory.objects.count(), 0)
        hospital_count = len(QUEBEC_HOSPITALS)
        self.assertEqual(Hospital.objects.count(), hospital_count)
        self.assertEqual(BloodSupply.objects.count(), hospital_count * len(BLOOD_TYPES))
        self.assertEqual(HospitalSupplyFeature.objects.count(), hospital_count)
        self.assertEqual(BloodSupply.objects.filter(hospital__hospital_id="H003").count(), 8)
        self.assertEqual(
            Hospital.objects.get(hospital_id="H003").name,
            "Centre hospitalier de l'Université de Montréal (CHUM)",
        )


class SeedDonorsCommandTests(TestCase):
    def test_generates_consistent_quebec_donor_distribution(self):
        call_command("seed_donors", reset=True, count=200, verbosity=0)

        self.assertEqual(Donor.objects.count(), 200)
        self.assertEqual(Donor.objects.filter(days_until_eligible__gt=0).count(), 40)
        self.assertEqual(Donor.objects.filter(recency_days__gte=90).count(), 160)
        city_weights = {city.wilaya: city.population_weight for city in QUEBEC_CITIES}
        larger_wilaya, smaller_wilaya = sorted(
            city_weights,
            key=city_weights.get,
            reverse=True,
        )[:2]
        self.assertGreater(
            Donor.objects.filter(wilaya=larger_wilaya).count(),
            Donor.objects.filter(wilaya=smaller_wilaya).count(),
        )

        for city_id, cluster_id in Donor.objects.values_list("city_id", "cluster_id").distinct():
            self.assertEqual(cluster_id, CITIES_BY_ID[city_id].cluster_id)
            self.assertEqual(
                Donor.objects.filter(city_id=city_id).values_list("wilaya", flat=True).first(),
                CITIES_BY_ID[city_id].wilaya,
            )


class _FixedSummerDate(date):
    @classmethod
    def today(cls) -> "_FixedSummerDate":
        return cls(2026, 8, 15)


class SeedPredictionHistoryCommandTests(TestCase):
    def test_reset_builds_deterministic_history_with_quebec_holiday_dips(self):
        call_command("seed_blood_supply", reset=True, verbosity=0)

        with patch("inventory.management.commands.seed_prediction_history.date", _FixedSummerDate):
            call_command("seed_prediction_history", reset=True, verbosity=0)

        self.assertEqual(
            PredictionResult.objects.count(),
            len(QUEBEC_HOSPITALS) * len(BLOOD_TYPES) * 60,
        )

        june_17 = PredictionResult.objects.get(
            entity_id="S001OP",
            model_name="stockout_days_predictor",
            predicted_for_date=date(2026, 6, 17),
        )
        june_24 = PredictionResult.objects.get(
            entity_id="S001OP",
            model_name="stockout_days_predictor",
            predicted_for_date=date(2026, 6, 24),
        )

        self.assertLess(june_24.predicted_value, june_17.predicted_value)


class QuebecSeasonalityModifierTests(SimpleTestCase):
    def test_campus_drive_season_offsets_more_than_summer_dip(self):
        self.assertGreater(
            quebec_seasonality_modifier(date(2026, 9, 15)),
            quebec_seasonality_modifier(date(2026, 7, 15)),
        )

    def test_holiday_dip_is_lower_than_nearby_summer_day(self):
        self.assertLess(
            quebec_seasonality_modifier(date(2026, 6, 24)),
            quebec_seasonality_modifier(date(2026, 6, 17)),
        )
