from unittest.mock import patch

from django.test import SimpleTestCase, override_settings
from rest_framework import status
from rest_framework.test import APIRequestFactory

from authApp.views import LoginView, RefreshTokenView
from backendMulti.services.keycloak_auth import (
    KeycloakConfigurationError,
    get_keycloak_client,
)


class KeycloakProviderConfigTests(SimpleTestCase):
    def test_missing_client_secret_fails_before_keycloak_request(self):
        with override_settings(
            KEYCLOAK_CONFIG={
                "SERVER_URL": "http://keycloak:8080/",
                "REALM_NAME": "pios",
                "CLIENT_ID": "django-backend",
                "CLIENT_SECRET": "",
            }
        ):
            with self.assertRaises(KeycloakConfigurationError) as raised:
                get_keycloak_client()

        self.assertIn("CLIENT_SECRET", str(raised.exception))

    def test_login_reports_provider_config_error_without_requiring_bearer_token(self):
        request = APIRequestFactory().post(
            "/v1/auth/login/",
            {"username": "pios", "password": "pios"},
            format="json",
        )

        with patch(
            "authApp.views.login_user",
            side_effect=KeycloakConfigurationError("missing client config"),
        ):
            response = LoginView.as_view()(request)

        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(
            response.data["error"],
            "Authentication provider is not configured",
        )

    def test_refresh_reports_provider_config_error_without_access_token(self):
        request = APIRequestFactory().post(
            "/v1/auth/refresh/",
            {"refresh_token": "refresh-token"},
            format="json",
        )

        with patch(
            "authApp.views.refresh_user_token",
            side_effect=KeycloakConfigurationError("missing client config"),
        ):
            response = RefreshTokenView.as_view()(request)

        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(
            response.data["error"],
            "Authentication provider is not configured",
        )
