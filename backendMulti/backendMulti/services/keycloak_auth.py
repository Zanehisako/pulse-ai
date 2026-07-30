# keycloak_auth.py
from collections.abc import Iterable

from keycloak import KeycloakOpenID, KeycloakAdmin
from django.conf import settings
from authApp.models import User


class KeycloakConfigurationError(Exception):
    """Raised when required Keycloak client settings are missing."""


def _require_keycloak_config(*keys):
    cfg = settings.KEYCLOAK_CONFIG
    missing = [key for key in keys if not str(cfg.get(key) or "").strip()]
    if missing:
        raise KeycloakConfigurationError(
            "Missing required Keycloak configuration: " + ", ".join(missing)
        )
    return cfg


def get_keycloak_client():
    cfg = _require_keycloak_config(
        "SERVER_URL",
        "REALM_NAME",
        "CLIENT_ID",
        "CLIENT_SECRET",
    )
    return KeycloakOpenID(
        server_url=cfg['SERVER_URL'],
        realm_name=cfg['REALM_NAME'],
        client_id=cfg['CLIENT_ID'],
        client_secret_key=cfg['CLIENT_SECRET'],
    )

def get_keycloak_admin():
    cfg = _require_keycloak_config(
        "SERVER_URL",
        "REALM_NAME",
        "ADMIN_USERNAME",
        "ADMIN_PASSWORD",
    )
    return KeycloakAdmin(
        server_url=cfg['SERVER_URL'],
        username=cfg['ADMIN_USERNAME'],   # ← from settings, not hardcoded
        password=cfg['ADMIN_PASSWORD'],   # ← from settings, not hardcoded
        realm_name=cfg['REALM_NAME'],
        user_realm_name="master",  # Admin API uses master realm
        verify=True,
    )


def get_keycloak_login_events(
    *,
    user_id: str,
    event_types: Iterable[str] | None = None,
    max_results: int | None = None,
    first: int = 0,
):
    kc_admin = get_keycloak_admin()
    configured_types = list(
        event_types or settings.KEYCLOAK_EVENTS_CONFIG["LOGIN_EVENT_TYPES"]
    )

    events = []
    for event_type in configured_types:
        query = {
            "user": user_id,
            "type": event_type,
            "first": first,
        }
        if max_results is not None:
            query["max"] = max_results
        events.extend(kc_admin.get_events(query=query) or [])

    return sorted(events, key=lambda item: item.get("time", 0), reverse=True)

def login_user(username, password):
    kc = get_keycloak_client()
    return kc.token(username, password)  # returns access_token, refresh_token

def register_user(username, email, password, first_name, last_name):
    kc_admin = get_keycloak_admin()
    kc_admin.create_user({
        "username": username,
        "email": email,
        "firstName": first_name,
        "lastName": last_name,
        "enabled": True,
        "emailVerified": True,
        "credentials": [{"type": "password", "value": password, "temporary": False}],
    })
    user_id= kc_admin.get_user_id(username)
    kc_admin.update_user(user_id, {
        "requiredActions": []
    })

def logout_user(refresh_token):
    kc = get_keycloak_client()
    kc.logout(refresh_token)

def validate_token(access_token):
    kc = get_keycloak_client()
    # This calls Keycloak's introspection endpoint
    return kc.introspect(access_token)  # raises exception if invalid

def update_user_password(email, new_password):
    kc_admin = get_keycloak_admin()

    users = kc_admin.get_users({"email": email})
    if not users:
        raise Exception("User not found")

    user_id = users[0]["id"]

    kc_admin.set_user_password(
        user_id=user_id,
        password=new_password,
        temporary=False,
    )

def refresh_user_token(refresh_token):
    kc = get_keycloak_client()
    return kc.refresh_token(refresh_token)  # returns new access_token + refresh_token


def sync_keycloak_user_by_email(email: str):
    normalized_email = (email or "").strip().lower()
    if not normalized_email:
        return None

    kc_admin = get_keycloak_admin()
    matches = kc_admin.get_users({"email": normalized_email}) or []
    user_data = next(
        (
            item
            for item in matches
            if str(item.get("email", "")).strip().lower() == normalized_email
        ),
        None,
    )
    if not user_data:
        return None

    first_name = str(user_data.get("firstName", "")).strip()
    last_name = str(user_data.get("lastName", "")).strip()
    full_name = " ".join(part for part in [first_name, last_name] if part).strip()
    fallback_name = (
        full_name
        or str(user_data.get("username", "")).strip()
        or normalized_email
    )

    user, _created = User.objects.update_or_create(
        email=normalized_email,
        defaults={
            "name": fallback_name,
            "is_active": True,
        },
    )
    return user


def __logout_user(refresh_token):
    """Revoke refresh token - revoke all user sessions"""
    try:
        kc = get_keycloak_client()
        
        token_info = kc.introspect(refresh_token)
        
        if not token_info.get('active'):
            raise Exception("Invalid or expired refresh token")
        
        user_id = token_info.get('sub')
        
        if user_id:
            kc_admin = get_keycloak_admin()
            kc_admin.user_logout(user_id)
        else:
            raise Exception("Cannot extract user from token")
    except Exception as e:
        raise Exception("Logout failed")  
