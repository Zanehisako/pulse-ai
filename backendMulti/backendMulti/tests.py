from asgiref.sync import async_to_sync
from django.test import TestCase
from unittest.mock import patch

from authApp.models import User
from backendMulti.middleware.ws_auth import get_user_from_token


class WebSocketTokenAuthTests(TestCase):
    def test_keycloak_access_token_resolves_to_django_user(self):
        token_info = {
            "active": True,
            "email": "alerts@example.com",
            "name": "Alerts User",
        }

        with patch("backendMulti.services.authentication.validate_token", return_value=token_info):
            user = async_to_sync(get_user_from_token)("keycloak-access-token")

        self.assertTrue(user.is_authenticated)
        self.assertEqual(user.email, "alerts@example.com")
        self.assertTrue(User.objects.filter(email="alerts@example.com").exists())

    def test_bearer_prefixed_websocket_token_is_supported(self):
        token_info = {
            "active": True,
            "email": "bearer-alerts@example.com",
            "name": "Bearer Alerts",
        }

        with patch("backendMulti.services.authentication.validate_token", return_value=token_info):
            user = async_to_sync(get_user_from_token)("Bearer keycloak-access-token")

        self.assertTrue(user.is_authenticated)
        self.assertEqual(user.email, "bearer-alerts@example.com")
