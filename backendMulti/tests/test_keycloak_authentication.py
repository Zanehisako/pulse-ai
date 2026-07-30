from unittest.mock import patch

import jwt
from django.test import SimpleTestCase
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.test import APIRequestFactory

from backendMulti.services.authentication import KeycloakAuthentication


class KeycloakAuthenticationTests(SimpleTestCase):
    def test_accepts_swagger_value_with_duplicate_bearer_prefix(self):
        request = APIRequestFactory().get(
            "/api/ml/",
            HTTP_AUTHORIZATION="Bearer Bearer token-from-swagger",
        )

        with patch(
            "backendMulti.services.authentication.validate_token",
            return_value={"active": True, "preferred_username": "pios"},
        ) as validate_token:
            user, token = KeycloakAuthentication().authenticate(request)

        validate_token.assert_called_once_with("token-from-swagger")
        self.assertEqual(token, "token-from-swagger")
        self.assertTrue(user.is_authenticated)

    def test_accepts_standard_bearer_header(self):
        request = APIRequestFactory().get(
            "/api/ml/",
            HTTP_AUTHORIZATION="Bearer token-from-client",
        )

        with patch(
            "backendMulti.services.authentication.validate_token",
            return_value={"active": True, "preferred_username": "pios"},
        ) as validate_token:
            _user, token = KeycloakAuthentication().authenticate(request)

        validate_token.assert_called_once_with("token-from-client")
        self.assertEqual(token, "token-from-client")

    def test_rejects_refresh_token_with_actionable_message(self):
        refresh_token = jwt.encode({"typ": "Refresh"}, key="", algorithm="none")
        request = APIRequestFactory().get(
            "/api/ml/",
            HTTP_AUTHORIZATION=f"Bearer {refresh_token}",
        )

        with self.assertRaises(AuthenticationFailed) as raised:
            KeycloakAuthentication().authenticate(request)

        self.assertIn("access_token", str(raised.exception))
        self.assertIn("refresh_token", str(raised.exception))
