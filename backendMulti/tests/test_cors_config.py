import os
from unittest.mock import patch
from django.test import SimpleTestCase
from backendMulti.settings import _parse_origins


class CorsConfigTests(SimpleTestCase):
    def test_parse_origins_returns_default_when_empty(self):
        default = ["http://localhost:5173", "http://127.0.0.1:5173"]
        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": ""}):
            result = _parse_origins("CORS_ALLOWED_ORIGINS", default)
            self.assertEqual(result, default)

    def test_parse_origins_splits_comma_separated_domains(self):
        default = ["http://localhost:5173"]
        custom = "https://pulse-ai.pages.dev, https://preview-123.pulse-ai.pages.dev, https://blood-donor.org"
        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": custom}):
            result = _parse_origins("CORS_ALLOWED_ORIGINS", default)
            self.assertEqual(
                result,
                [
                    "https://pulse-ai.pages.dev",
                    "https://preview-123.pulse-ai.pages.dev",
                    "https://blood-donor.org",
                ],
            )

    def test_parse_origins_ignores_trailing_and_empty_entries(self):
        default = ["http://localhost:5173"]
        custom = " https://pulse-ai.pages.dev , , https://app.pulse.org , "
        with patch.dict(os.environ, {"CSRF_TRUSTED_ORIGINS": custom}):
            result = _parse_origins("CSRF_TRUSTED_ORIGINS", default)
            self.assertEqual(
                result,
                [
                    "https://pulse-ai.pages.dev",
                    "https://app.pulse.org",
                ],
            )
