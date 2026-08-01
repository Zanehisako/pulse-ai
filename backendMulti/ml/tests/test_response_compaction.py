import unittest

from ml.core.response_compaction import (
    DEFAULT_COMPACT_KEYS,
    compact_prediction_output,
    compact_prediction_row,
    configured_compact_keys,
)


class ResponseCompactionTests(unittest.TestCase):
    def test_schema_payload_survives_configured_compaction(self):
        output = {
            "answer": "DB schema contains tables: donors, hospitals.",
            "tables": {
                "donors": {
                    "logical_table": "donors",
                    "default_table_name": "donors",
                    "column_count": 2,
                    "columns": [
                        {"name": "donor_id", "type": "varchar"},
                        {"name": "blood_type", "type": "varchar"},
                    ],
                    "field_aliases": {"blood_type": "blood group"},
                    "identity_fields": ["donor_id"],
                }
            },
            "table_names": ["donors"],
            "internal_secret": "must not leak",
        }
        keys = configured_compact_keys(
            {"api_output_compact_keys": sorted(DEFAULT_COMPACT_KEYS)}
        )
        compact = compact_prediction_output(output, compact_keys=keys)

        self.assertIn("tables", compact)
        self.assertIn("table_names", compact)
        self.assertIn("answer", compact)
        self.assertIn("columns", compact["tables"]["donors"])
        self.assertNotIn("internal_secret", compact)

    def test_configured_keys_override_defaults(self):
        output = {"answer": "ok", "tables": {"donors": {}}, "rows": []}
        keys = configured_compact_keys({"api_output_compact_keys": ["answer"]})
        compact = compact_prediction_output(output, compact_keys=keys)

        self.assertEqual(compact, {"answer": "ok", "rows": []})

    def test_missing_config_key_falls_back_to_defaults(self):
        output = {
            "answer": "ok",
            "tables": {"donors": {}},
            "unknown_field": "x",
        }
        compact = compact_prediction_output(output, compact_keys=None)

        self.assertIn("answer", compact)
        self.assertIn("tables", compact)
        self.assertNotIn("unknown_field", compact)

    def test_non_dict_output_passes_through(self):
        self.assertEqual(compact_prediction_output("plain text"), "plain text")
        self.assertIsNone(compact_prediction_output(None))

    def test_risk_prefixed_keys_always_survive(self):
        output = {"risk_1_7_days": 0.4, "answer": "ok"}
        compact = compact_prediction_output(output, compact_keys=configured_compact_keys(None))

        self.assertIn("risk_1_7_days", compact)

    def test_rows_are_compacted_with_row_whitelist(self):
        output = {
            "rows": [
                {
                    "donor_id": "D001",
                    "blood_type": "O+",
                    "model_output": {"prediction": 1, "secret": "x"},
                    "feature_resolution": {"source": "feast"},
                    "hidden": True,
                }
            ]
        }
        compact = compact_prediction_output(output, compact_keys=None)

        row = compact["rows"][0]
        self.assertEqual(row["donor_id"], "D001")
        self.assertEqual(row["blood_type"], "O+")
        self.assertEqual(row["prediction"], 1)
        self.assertNotIn("secret", row)
        self.assertNotIn("feature_resolution", row)
        self.assertNotIn("hidden", row)

    def test_compact_row_passthrough_for_non_dict(self):
        self.assertEqual(compact_prediction_row([1, 2]), [1, 2])


if __name__ == "__main__":
    unittest.main()
