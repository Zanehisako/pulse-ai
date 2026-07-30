from drf_spectacular.extensions import OpenApiAuthenticationExtension


class KeycloakAuthenticationScheme(OpenApiAuthenticationExtension):
    target_class = "backendMulti.services.authentication.KeycloakAuthentication"
    name = "Bearer Authentication"

    def get_security_definition(self, auto_schema):
        return {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Paste the raw Keycloak access_token only. Do not paste the refresh_token or the 'Bearer ' prefix.",
        }


def exclude_legacy_ml_endpoints(endpoints, **kwargs):
    return [
        endpoint for endpoint in endpoints
        if not endpoint[0].startswith("/ml/")
    ]


def add_global_security(result, generator, request, public, **kwargs):
    """Postprocessing hook: inject root-level security so every endpoint shows the lock icon."""
    result.setdefault("security", [{"Bearer Authentication": []}])
    return result
