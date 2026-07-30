# authentication.py
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from django.contrib.auth.models import AnonymousUser
from .keycloak_auth import validate_token
from authApp.models import User


def _looks_like_refresh_token(token: str) -> bool:
    try:
        import jwt

        payload = jwt.decode(token, options={"verify_signature": False})
    except Exception:
        return False
    token_type = str(payload.get("typ") or payload.get("token_type") or "").lower()
    return token_type == "refresh"


class KeycloakUser:
    """
    Wrapper utilisateur qui implémente les propriétés attendues par DRF.
    Wraps token_info (dict Keycloak) et un User Django optionnel.
    """
    def __init__(self, token_info: dict, django_user: User = None):
        self.token_info = token_info
        self.django_user = django_user
        self.is_authenticated = True
        self.is_active = token_info.get('active', True)

    def __getattr__(self, name):
        """Retourne les attributs du User Django si disponible, sinon du token_info"""
        if self.django_user and hasattr(self.django_user, name):
            return getattr(self.django_user, name)
        return self.token_info.get(name)

    def __str__(self):
        if self.django_user:
            return str(self.django_user)
        return self.token_info.get('email', 'Unknown')


def django_user_from_token_info(token_info: dict) -> User | None:
    email = token_info.get('email', '')
    if not email:
        return None
    name = token_info.get('name') or token_info.get('preferred_username', email)
    django_user, _ = User.objects.get_or_create(
        email=email,
        defaults={'name': name, 'is_active': True},
    )
    return django_user


class KeycloakAuthentication(BaseAuthentication):
    """
    Authentifie via token Keycloak et retourne un wrapper utilisateur compatible avec DRF.
    """
    def authenticate(self, request):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return None  # let other authenticators try

        token = auth_header.split(' ', 1)[1].strip()
        if token.lower().startswith('bearer '):
            token = token.split(' ', 1)[1].strip()
        if _looks_like_refresh_token(token):
            raise AuthenticationFailed(
                'Use the Keycloak access_token for API authorization. '
                'The refresh_token is only used with the refresh endpoint.'
            )
        try:
            token_info = validate_token(token)
            if not token_info.get('active'):
                raise AuthenticationFailed('Token inactive or expired')

            django_user = django_user_from_token_info(token_info)

            # Retourner un wrapper utilisateur compatible DRF
            keycloak_user = KeycloakUser(token_info, django_user)
            return (keycloak_user, token)

        except AuthenticationFailed:
            raise
        except Exception as e:
            raise AuthenticationFailed(f'Invalid token: {str(e)}')
