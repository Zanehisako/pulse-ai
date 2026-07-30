from unittest.mock import patch

from django.conf import settings
from django.test import RequestFactory, SimpleTestCase, override_settings

from backendMulti.api_docs import (
    DocsAutoAuthTokenView,
    normalized_api_docs_token_route,
)
from backendMulti.schema import KeycloakAuthenticationScheme


class OpenApiAuthConfigTests(SimpleTestCase):
    def test_keycloak_authentication_extension_defines_bearer_scheme(self):
        scheme = KeycloakAuthenticationScheme(target=None)

        self.assertEqual(
            scheme.target_class,
            "backendMulti.services.authentication.KeycloakAuthentication",
        )
        self.assertEqual(scheme.name, "Bearer Authentication")
        self.assertEqual(
            scheme.get_security_definition(auto_schema=None),
            {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "Paste the raw Keycloak access_token only. Do not paste the refresh_token or the 'Bearer ' prefix.",
            },
        )

    def test_spectacular_settings_include_global_bearer_security(self):
        spectacular = settings.SPECTACULAR_SETTINGS

        self.assertIn(
            {"Bearer Authentication": []},
            spectacular["SECURITY"],
        )
        self.assertEqual(
            spectacular["APPEND_COMPONENTS"]["securitySchemes"][
                "Bearer Authentication"
            ]["scheme"],
            "bearer",
        )
        self.assertIn(
            "backendMulti.schema.add_global_security",
            spectacular["POSTPROCESSING_HOOKS"],
        )
        self.assertFalse(
            spectacular["SWAGGER_UI_SETTINGS"]["persistAuthorization"],
        )

    def test_api_docs_auto_auth_config_is_debug_gated(self):
        config = settings.API_DOCS_AUTH_CONFIG

        self.assertIn("AUTO_AUTH_ENABLED", config)
        self.assertEqual(config["SECURITY_SCHEME"], "Bearer Authentication")
        self.assertIn("KEYCLOAK_CLIENT_ID", config)
        self.assertEqual(normalized_api_docs_token_route(), "api/docs/auth-token/")

    @override_settings(
        DEBUG=True,
        API_DOCS_AUTH_CONFIG={
            "AUTO_AUTH_ENABLED": True,
            "TOKEN_ROUTE": "api/docs/auth-token/",
            "SECURITY_SCHEME": "Bearer Authentication",
            "KEYCLOAK_USERNAME": "docs-user",
            "KEYCLOAK_PASSWORD": "docs-password",
            "KEYCLOAK_CLIENT_ID": "docs-client",
            "KEYCLOAK_CLIENT_SECRET": "",
        },
    )
    def test_swagger_page_includes_auto_auth_script(self):
        response = self.client.get("/api/docs/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "api/docs/auth\\u002Dtoken/")
        self.assertContains(response, "preauthorizeApiKey")


class ApiDocsAutoAuthTokenTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()

    @override_settings(
        DEBUG=True,
        API_DOCS_AUTH_CONFIG={
            "AUTO_AUTH_ENABLED": True,
            "TOKEN_ROUTE": "api/docs/auth-token/",
            "SECURITY_SCHEME": "Bearer Authentication",
            "KEYCLOAK_USERNAME": "docs-user",
            "KEYCLOAK_PASSWORD": "docs-password",
            "KEYCLOAK_CLIENT_ID": "docs-client",
            "KEYCLOAK_CLIENT_SECRET": "",
        },
        KEYCLOAK_CONFIG={
            "SERVER_URL": "http://keycloak.local/",
            "REALM_NAME": "docs-realm",
        },
    )
    @patch("backendMulti.api_docs.KeycloakOpenID")
    def test_auto_auth_token_returns_access_token_only(self, mock_keycloak_openid):
        mock_keycloak_openid.return_value.token.return_value = {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

        response = DocsAutoAuthTokenView.as_view()(
            self.factory.get("/api/docs/auth-token/")
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.data,
            {
                "access_token": "access-token",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        )
        self.assertNotIn("refresh_token", response.data)
        mock_keycloak_openid.assert_called_once_with(
            server_url="http://keycloak.local/",
            realm_name="docs-realm",
            client_id="docs-client",
            client_secret_key=None,
        )
        mock_keycloak_openid.return_value.token.assert_called_once_with(
            "docs-user",
            "docs-password",
        )

    @override_settings(
        DEBUG=True,
        API_DOCS_AUTH_CONFIG={
            "AUTO_AUTH_ENABLED": False,
            "TOKEN_ROUTE": "api/docs/auth-token/",
            "SECURITY_SCHEME": "Bearer Authentication",
            "KEYCLOAK_USERNAME": "docs-user",
            "KEYCLOAK_PASSWORD": "docs-password",
            "KEYCLOAK_CLIENT_ID": "docs-client",
            "KEYCLOAK_CLIENT_SECRET": "",
        },
    )
    def test_auto_auth_token_404s_when_disabled(self):
        response = DocsAutoAuthTokenView.as_view()(
            self.factory.get("/api/docs/auth-token/")
        )

        self.assertEqual(response.status_code, 404)
