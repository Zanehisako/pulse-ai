"""
Django settings for backendMulti project — with ML integration.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR.parent / ".env.local")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# Internal ML asset paths
ML_CONFIG_DIR = BASE_DIR / "ml" / "config"
ML_MODELS_DIR = BASE_DIR / "ml" / "models"

DEBUG = os.getenv("DJANGO_DEBUG", "1") == "1"
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY") or (
    "unsafe-local-dev-secret-key" if DEBUG else ""
)
if not SECRET_KEY:
    raise RuntimeError("DJANGO_SECRET_KEY is required when DJANGO_DEBUG is disabled.")
ALLOWED_HOSTS = os.getenv("DJANGO_ALLOWED_HOSTS", "*").split(",")


# INSTALLED APPS
INSTALLED_APPS = [
    # Django core
    "daphne",  # Must be FIRST for ASGI
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "dashboard.apps.DashboardConfig",
    # Third-party
    "rest_framework",
    "corsheaders",
    "channels",
    "django_filters",
    "drf_spectacular",
    # Existing apps
    "authApp",
    "testUrl",
    "centres",
    "communications",
    "notifications",
    "alerts.apps.AlertsConfig",
    "digital_twin.apps.DigitalTwinConfig",
    "ml",
    "workspace",
    "inventory.apps.InventoryConfig",
    "bloodbag",
    # "dashboard",
]


MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "backendMulti.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "backendMulti.wsgi.application"
ASGI_APPLICATION = "backendMulti.asgi.application"


DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("DB_NAME", "pios"),
        # "USER": os.getenv("DB_USER", "postgres"),
        "USER": os.getenv("DB_USER", "admin"),
        "PASSWORD": os.getenv("DB_PASSWORD", "admin"),
        "HOST": os.getenv("DB_HOST", "localhost"),
        "PORT": os.getenv("DB_PORT", "5432"),
    }
}

AUTH_USER_MODEL = "authApp.User"

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    }
}

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'backendMulti.services.authentication.KeycloakAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [        
        'rest_framework.permissions.IsAuthenticated',
    ],
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
    ],
    "DEFAULT_PAGINATION_CLASS": None,
}
SPECTACULAR_SETTINGS = {
    "TITLE": "PIOS API",
    "DESCRIPTION": "Blood donation platform — Auth, Labs, Notifications & ML Prediction.",
    "VERSION": "2.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
    "PREPROCESSING_HOOKS": [
        "backendMulti.schema.exclude_legacy_ml_endpoints",
    ],
    "POSTPROCESSING_HOOKS": [
        "drf_spectacular.hooks.postprocess_schema_enums",
        "backendMulti.schema.add_global_security",
    ],
    # Force the Bearer token input to always appear in Swagger UI
    "APPEND_COMPONENTS": {
        "securitySchemes": {
            "Bearer Authentication": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "Paste the raw Keycloak access_token only. Do not paste the refresh_token or the 'Bearer ' prefix.",
            }
        }
    },
    "SECURITY": [{"Bearer Authentication": []}],
    "SWAGGER_UI_SETTINGS": {
        "persistAuthorization": False,
    },
}

# Keycloak settings for authentication
KEYCLOAK_CONFIG = {
    # Local default uses localhost; Docker overrides this with KEYCLOAK_SERVER_URL=http://keycloak:8080/
    "SERVER_URL": os.getenv("KEYCLOAK_SERVER_URL", "http://localhost:8080/"),
    "REALM_NAME": os.getenv("KEYCLOAK_REALM_NAME", "pios"),
    "CLIENT_ID": os.getenv("KEYCLOAK_CLIENT_ID", "django-backend"),
    "CLIENT_SECRET": os.getenv("KEYCLOAK_CLIENT_SECRET", ""),
    "ADMIN_USERNAME": os.getenv(
        "KEYCLOAK_ADMIN_USERNAME",
        os.getenv("KEYCLOAK_BOOTSTRAP_ADMIN_USERNAME", "pios"),
    ),
    "ADMIN_PASSWORD": os.getenv(
        "KEYCLOAK_ADMIN_PASSWORD",
        os.getenv("KEYCLOAK_BOOTSTRAP_ADMIN_PASSWORD", "pios"),
    ),
}

API_DOCS_AUTH_CONFIG = {
    "AUTO_AUTH_ENABLED": DEBUG
    and _env_bool("PIOS_API_DOCS_AUTO_AUTH_ENABLED", False),
    "TOKEN_ROUTE": os.getenv(
        "PIOS_API_DOCS_TOKEN_ROUTE",
        "api/docs/auth-token/",
    ),
    "SECURITY_SCHEME": os.getenv(
        "PIOS_API_DOCS_SECURITY_SCHEME",
        "Bearer Authentication",
    ),
    "KEYCLOAK_USERNAME": os.getenv(
        "PIOS_API_DOCS_KEYCLOAK_USERNAME",
        os.getenv("KEYCLOAK_APP_USERNAME", ""),
    ),
    "KEYCLOAK_PASSWORD": os.getenv(
        "PIOS_API_DOCS_KEYCLOAK_PASSWORD",
        os.getenv("KEYCLOAK_APP_PASSWORD", ""),
    ),
    "KEYCLOAK_CLIENT_ID": os.getenv(
        "PIOS_API_DOCS_KEYCLOAK_CLIENT_ID",
        os.getenv("VITE_KEYCLOAK_CLIENT_ID", os.getenv("KEYCLOAK_CLIENT_ID", "")),
    ),
    "KEYCLOAK_CLIENT_SECRET": os.getenv(
        "PIOS_API_DOCS_KEYCLOAK_CLIENT_SECRET",
        "",
    ),
}

KEYCLOAK_EVENTS_CONFIG = {
    "LOGIN_EVENT_TYPES": [
        item.strip()
        for item in os.getenv(
            "KEYCLOAK_LOGIN_EVENT_TYPES",
            "LOGIN,LOGIN_ERROR",
        ).split(",")
        if item.strip()
    ],
    "DEFAULT_MAX_RESULTS": int(os.getenv("KEYCLOAK_EVENTS_DEFAULT_MAX_RESULTS", "20")),
    "MAX_RESULTS_LIMIT": int(os.getenv("KEYCLOAK_EVENTS_MAX_RESULTS_LIMIT", "100")),
}

CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5173",
]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_CREDENTIALS = True
if DEBUG:
    CORS_ALLOW_ALL_ORIGINS = True

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
    }
}

CSRF_TRUSTED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

CSRF_TRUSTED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# ══════════════════════════════════════════════════════════
#  PIOS ML SETTINGS
# ══════════════════════════════════════════════════════════

# Path to the local folder where .pkl models live
PIOS_MODELS_DIR = Path(os.getenv("PIOS_MODELS_DIR", str(ML_MODELS_DIR))).resolve()

PIOS_MODEL_CONFIG_PATH = Path(
    os.getenv(
        "PIOS_MODEL_CONFIG",
        str(ML_CONFIG_DIR / "config.json"),
    )
).resolve()

PIOS_ORCHESTRATOR_CONFIG_PATH = Path(
    os.getenv(
        "PIOS_ORCHESTRATOR_MODEL_CONFIG",
        str(ML_CONFIG_DIR / "orchestrator_models.json"),
    )
).resolve()

PIOS_SCHEDULED_PREDICTION_CONFIG_PATH = Path(
    os.getenv(
        "PIOS_SCHEDULED_PREDICTION_CONFIG",
        str(ML_CONFIG_DIR / "scheduled_predictions.json"),
    )
).resolve()

PIOS_SIMULATION_CONFIG_PATH = Path(
    os.getenv(
        "PIOS_SIMULATION_CONFIG",
        str(ML_CONFIG_DIR / "simulation_config.json"),
    )
).resolve()

PIOS_DIGITAL_TWIN_CONFIG_PATH = Path(
    os.getenv(
        "PIOS_DIGITAL_TWIN_CONFIG",
        str(ML_CONFIG_DIR / "digital_twin.json"),
    )
).resolve()

WORKSPACE_WIDGET_SOURCES_CONFIG_PATH = Path(
    os.getenv(
        "WORKSPACE_WIDGET_SOURCES_CONFIG",
        str(BASE_DIR / "workspace" / "config" / "widget_sources.json"),
    )
).resolve()

# xLAM Orchestrator settings
PIOS_XLAM_REPO_ID = os.getenv("PIOS_XLAM_REPO_ID", "bartowski/xLAM-7b-fc-r-GGUF")
PIOS_XLAM_MODEL_DIR = Path(
    os.getenv("PIOS_XLAM_MODEL_DIR", str(ML_MODELS_DIR / "gguf"))
).resolve()
PIOS_XLAM_DISABLE_LLM = os.getenv("PIOS_XLAM_DISABLE_LLM", "0") == "1"
PIOS_XLAM_INIT_MODE = os.getenv("PIOS_XLAM_INIT_MODE", "background").strip().lower()

EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "1") == "1"
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER", "mgryan20@gmail.com")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD", "jxzxcflfqbisuxgh")
