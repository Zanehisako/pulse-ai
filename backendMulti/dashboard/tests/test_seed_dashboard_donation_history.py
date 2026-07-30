from datetime import date

from django.test import SimpleTestCase

from dashboard.management.commands.seed_dashboard_from_inventory import (
    build_donation_history,
)


class BuildDonationHistoryTests(SimpleTestCase):
    def test_builds_complete_daily_series(self):
        result = build_donation_history(
            [
                {"recency_days": 0, "frequency_365": 1},
                {"recency_days": 2, "frequency_365": 1},
            ],
            reference_date=date(2026, 5, 2),
            days=3,
        )

        self.assertEqual(
            result,
            {
                "data": [
                    {"date": "2026-04-30", "value": 1},
                    {"date": "2026-05-01", "value": 0},
                    {"date": "2026-05-02", "value": 1},
                ]
            },
        )

    def test_expands_frequency_into_recent_history(self):
        result = build_donation_history(
            [{"recency_days": 1, "frequency_365": 4}],
            reference_date=date(2026, 5, 2),
            days=100,
        )

        values_by_date = {
            point["date"]: point["value"]
            for point in result["data"]
            if point["value"] > 0
        }

        self.assertEqual(
            values_by_date,
            {
                "2026-05-01": 1,
                "2026-01-30": 1,
            },
        )

    def test_ignores_invalid_rows_without_breaking_shape(self):
        result = build_donation_history(
            [
                {"recency_days": None, "frequency_365": 2},
                {"recency_days": "bad", "frequency_365": 2},
            ],
            reference_date=date(2026, 5, 2),
            days=2,
        )

        self.assertEqual(
            result,
            {
                "data": [
                    {"date": "2026-05-01", "value": 0},
                    {"date": "2026-05-02", "value": 0},
                ]
            },
        )
