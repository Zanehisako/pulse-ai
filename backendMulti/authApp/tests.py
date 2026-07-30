from types import SimpleNamespace
from unittest.mock import patch

from django.test import override_settings
from rest_framework import status
from rest_framework.test import APIClient, APITestCase


class CurrentUserLoginHistoryViewTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.url = "/v1/auth/web/login-history/"

    def test_requires_authentication(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    @override_settings(
        KEYCLOAK_EVENTS_CONFIG={
            "LOGIN_EVENT_TYPES": ["LOGIN", "LOGIN_ERROR"],
            "DEFAULT_MAX_RESULTS": 20,
            "MAX_RESULTS_LIMIT": 50,
        }
    )
    @patch("authApp.views.get_keycloak_login_events")
    def test_returns_normalized_login_events(self, mock_get_keycloak_login_events):
        mock_get_keycloak_login_events.return_value = [
            {
                "id": "evt-2",
                "type": "LOGIN_ERROR",
                "time": 1711111111000,
                "ipAddress": "10.0.0.2",
                "clientId": "pios-web",
                "error": "invalid_user_credentials",
                "details": {"auth_method": "openid-connect", "auth_type": "code"},
            },
            {
                "id": "evt-1",
                "type": "LOGIN",
                "time": 1711111110000,
                "ipAddress": "10.0.0.1",
                "clientId": "pios-web",
                "details": {"auth_method": "openid-connect", "auth_type": "code"},
            },
        ]
        self.client.force_authenticate(
            user=SimpleNamespace(is_authenticated=True, sub="kc-user-1")
        )

        response = self.client.get(self.url, {"max": 10})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["meta"]["count"], 2)
        self.assertEqual(response.data["meta"]["max_results"], 10)
        self.assertEqual(response.data["meta"]["event_types"], ["LOGIN", "LOGIN_ERROR"])
        self.assertEqual(response.data["events"][0]["id"], "evt-2")
        self.assertEqual(response.data["events"][0]["error"], "invalid_user_credentials")
        self.assertEqual(response.data["events"][1]["type"], "LOGIN")
        mock_get_keycloak_login_events.assert_called_once_with(
            user_id="kc-user-1",
            max_results=10,
        )

    @override_settings(
        KEYCLOAK_EVENTS_CONFIG={
            "LOGIN_EVENT_TYPES": ["LOGIN"],
            "DEFAULT_MAX_RESULTS": 20,
            "MAX_RESULTS_LIMIT": 25,
        }
    )
    @patch("authApp.views.get_keycloak_login_events")
    def test_clamps_requested_max_to_config_limit(self, mock_get_keycloak_login_events):
        self.client.force_authenticate(
            user=SimpleNamespace(is_authenticated=True, sub="kc-user-2")
        )

        response = self.client.get(self.url, {"max": 500})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["meta"]["max_results"], 25)
        mock_get_keycloak_login_events.assert_called_once_with(
            user_id="kc-user-2",
            max_results=25,
        )
