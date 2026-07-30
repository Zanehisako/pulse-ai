from django.conf import settings
from django.http import Http404
from rest_framework.exceptions import APIException
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from keycloak import KeycloakOpenID
from drf_spectacular.utils import extend_schema
from drf_spectacular.views import SpectacularSwaggerView
from rest_framework.reverse import reverse


class DocsAutoAuthTokenUnavailable(APIException):
    status_code = 503
    default_detail = "API docs auto-auth token is unavailable."
    default_code = "docs_auto_auth_unavailable"


def api_docs_auth_config():
    return getattr(settings, "API_DOCS_AUTH_CONFIG", {})


def api_docs_auto_auth_enabled():
    config = api_docs_auth_config()
    return bool(settings.DEBUG and config.get("AUTO_AUTH_ENABLED"))


def normalized_api_docs_token_route():
    route = str(api_docs_auth_config().get("TOKEN_ROUTE", "")).strip().lstrip("/")
    return route if route.endswith("/") else f"{route}/"


def get_docs_keycloak_token(username: str, password: str):
    keycloak_config = settings.KEYCLOAK_CONFIG
    docs_config = api_docs_auth_config()
    missing = [
        key
        for key, value in {
            "SERVER_URL": keycloak_config.get("SERVER_URL"),
            "REALM_NAME": keycloak_config.get("REALM_NAME"),
            "KEYCLOAK_CLIENT_ID": docs_config.get("KEYCLOAK_CLIENT_ID"),
        }.items()
        if not str(value or "").strip()
    ]
    if missing:
        raise DocsAutoAuthTokenUnavailable(
            "Missing required API docs auto-auth configuration: "
            + ", ".join(missing)
        )

    client = KeycloakOpenID(
        server_url=keycloak_config["SERVER_URL"],
        realm_name=keycloak_config["REALM_NAME"],
        client_id=docs_config["KEYCLOAK_CLIENT_ID"],
        client_secret_key=docs_config.get("KEYCLOAK_CLIENT_SECRET") or None,
    )
    return client.token(username, password)


class PiosSwaggerView(SpectacularSwaggerView):
    template_name_js = "backendMulti/swagger_ui_auto_auth.js"

    def get(self, request, *args, **kwargs):
        response = super().get(request, *args, **kwargs)
        if api_docs_auto_auth_enabled():
            config = api_docs_auth_config()
            response.data.update(
                {
                    "docs_auto_auth_enabled": True,
                    "docs_auto_auth_url": reverse(
                        "swagger-auto-auth-token",
                        request=request,
                    ),
                    "docs_auto_auth_security_scheme": config.get("SECURITY_SCHEME"),
                }
            )
        return response


class DocsAutoAuthTokenView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    @extend_schema(exclude=True)
    def get(self, request):
        if not api_docs_auto_auth_enabled():
            raise Http404

        config = api_docs_auth_config()
        username = str(config.get("KEYCLOAK_USERNAME") or "").strip()
        password = str(config.get("KEYCLOAK_PASSWORD") or "")
        if not username or not password:
            raise DocsAutoAuthTokenUnavailable(
                "Missing API docs auto-auth Keycloak username or password."
            )

        try:
            token_payload = get_docs_keycloak_token(username, password)
        except Exception as exc:
            if isinstance(exc, DocsAutoAuthTokenUnavailable):
                raise
            raise DocsAutoAuthTokenUnavailable(
                "Could not get a Keycloak access token for API docs."
            ) from exc

        access_token = token_payload.get("access_token")
        if not access_token:
            raise DocsAutoAuthTokenUnavailable(
                "Keycloak did not return an access token for API docs."
            )

        response_payload = {"access_token": access_token}
        for key in ("expires_in", "token_type"):
            if key in token_payload:
                response_payload[key] = token_payload[key]
        return Response(response_payload)
