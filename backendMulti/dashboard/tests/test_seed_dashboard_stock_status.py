from django.test import SimpleTestCase

from dashboard.management.commands.seed_dashboard_from_inventory import (
    build_stock_status,
    derive_stock_status,
)


class BuildStockStatusTests(SimpleTestCase):
    def test_derives_stock_status_from_percentage_bands(self):
        self.assertEqual(derive_stock_status(299), "low")
        self.assertEqual(derive_stock_status(300), "medium")
        self.assertEqual(derive_stock_status(699), "medium")
        self.assertEqual(derive_stock_status(700), "high")
        self.assertEqual(derive_stock_status(1000), "high")

    def test_builds_status_for_blood_types_only(self):
        result = build_stock_status(
            {
                "O+": 250,
                "A+": 450,
                "B+": 750,
                "stockout_days": {"O+": 3},
            }
        )

        self.assertEqual(
            result,
            {
                "O+": "low",
                "A+": "medium",
                "B+": "high",
            },
        )
