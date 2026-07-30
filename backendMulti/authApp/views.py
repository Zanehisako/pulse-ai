import logging

from backendMulti.services.otp_service import clear_otp, generate_otp, is_otp_verified, mark_otp_verified, store_otp, verify_otp
from backendMulti.services.keycloak_auth import (
    KeycloakConfigurationError,
    login_user,
    register_user,
    logout_user,
    __logout_user,
    update_user_password,
    refresh_user_token,
    get_keycloak_login_events,
)
from django.conf import settings
from django.core.mail import send_mail
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated


logger = logging.getLogger(__name__)


def _normalize_keycloak_event(event):
    details = event.get("details") or {}
    return {
        "id": event.get("id"),
        "type": event.get("type"),
        "time": event.get("time"),
        "ip_address": event.get("ipAddress"),
        "client_id": event.get("clientId"),
        "error": event.get("error"),
        "auth_method": details.get("auth_method"),
        "auth_type": details.get("auth_type"),
    }


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')

        if not username or not password:
            return Response(
                {'error': 'Username and password are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            tokens = login_user(username, password)

            from authApp.models import User
            try:
                user = User.objects.filter(
                    email__icontains=username
                ).first() or User.objects.filter(
                    name__icontains=username
                ).first()
                user_name = user.name if user else username
            except Exception:
                user_name = username

            return Response({
                'access_token': tokens['access_token'],
                'refresh_token': tokens['refresh_token'],
                'expires_in': tokens.get('expires_in', 300),
                'name': user_name,
                'email': user.email if user else username,
            }, status=status.HTTP_200_OK)
        except KeycloakConfigurationError:
            logger.exception("Keycloak configuration is incomplete for login.")
            return Response(
                {'error': 'Authentication provider is not configured'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        except Exception:
            return Response(
                {'error': 'Invalid credentials'},
                status=status.HTTP_401_UNAUTHORIZED
            )


class SignupView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get('username')
        email = request.data.get('email')
        first_name = request.data.get('first_name')
        last_name = request.data.get('last_name')
        password = request.data.get('password')

        if not all([username, email, password, first_name, last_name]):
            return Response(
                {'error': 'All fields are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            register_user(username, email, password, first_name, last_name)
            tokens = login_user(username, password)
            return Response({
                'access_token': tokens['access_token'],
                'refresh_token': tokens['refresh_token'],
                'expires_in': tokens.get('expires_in', 300),
                'name': tokens.get('name'),
                'email': tokens.get('email'),
            }, status=status.HTTP_201_CREATED)
        except KeycloakConfigurationError:
            logger.exception("Keycloak configuration is incomplete for signup.")
            return Response(
                {'error': 'Authentication provider is not configured'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )


class LogoutView(APIView):
    def post(self, request):
        refresh_token = request.data.get("refresh_token")

        if not refresh_token:
            return Response(
                {"error": "Refresh token is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            logout_user(refresh_token)  # 
            return Response(
                {"message": "Logged out successfully"},
                status=status.HTTP_200_OK,
            )
        except Exception:
            return Response(
                {"error": "Logout failed"},
                status=status.HTTP_400_BAD_REQUEST,
            )

class ForgotPasswordView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")

        if not email:
            return Response({"error": "Email is required"}, status=400)

        otp = generate_otp()
        store_otp(email, otp)

        send_mail(
            "Your OTP Code",
            f"Your OTP is: {otp}",
            "mgryan20@gmail.com",
            [email],
            fail_silently=False,
        )

        return Response({"message": "OTP sent"}, status=200)
    

class VerifyOTPView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        otp = request.data.get("otp")

        if not email or not otp:
            return Response({"error": "Email and OTP required"}, status=400)

        if not verify_otp(email, otp):
            return Response({"error": "Invalid OTP"}, status=400)

        mark_otp_verified(email)

        return Response({"message": "OTP verified"}, status=200)
    
class ResetPasswordView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        new_password = request.data.get("password")

        if not email or not new_password:
            return Response({"error": "Email and password required"}, status=400)

        if not is_otp_verified(email):
            return Response({"error": "OTP not verified"}, status=403)

        try:
            update_user_password(email, new_password)
            clear_otp(email)

            return Response({"message": "Password reset successful"}, status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=400)

class ProfileView(APIView):
    def get(self, request):
        token_info = request.user
        return Response({
            'username': token_info.get('preferred_username'),
            'email': token_info.get('email'),
            'roles': token_info.get('realm_access', {}).get('roles', []),
        })
      
class RefreshTokenView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        refresh_token = request.data.get('refresh_token')
        if not refresh_token:
            return Response(
                {'error': 'Refresh token is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            tokens = refresh_user_token(refresh_token)
            return Response({
                'access_token': tokens['access_token'],
                'refresh_token': tokens['refresh_token'],
                'expires_in': tokens.get('expires_in', 300),
            }, status=status.HTTP_200_OK)
        except KeycloakConfigurationError:
            logger.exception("Keycloak configuration is incomplete for token refresh.")
            return Response(
                {'error': 'Authentication provider is not configured'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        except Exception:
            return Response(
                {'error': 'Invalid or expired refresh token'},
                status=status.HTTP_401_UNAUTHORIZED
            )


class CurrentUserLoginHistoryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        default_max = settings.KEYCLOAK_EVENTS_CONFIG["DEFAULT_MAX_RESULTS"]
        max_limit = settings.KEYCLOAK_EVENTS_CONFIG["MAX_RESULTS_LIMIT"]
        requested_max = request.query_params.get("max")

        try:
            max_results = int(requested_max) if requested_max is not None else default_max
        except (TypeError, ValueError):
            max_results = default_max

        max_results = max(1, min(max_results, max_limit))

        user_id = getattr(request.user, "sub", None)
        if not user_id:
            return Response(
                {"error": "Authenticated user identifier is missing."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        events = get_keycloak_login_events(
            user_id=user_id,
            max_results=max_results,
        )

        normalized = [_normalize_keycloak_event(event) for event in events[:max_results]]
        return Response(
            {
                "events": normalized,
                "meta": {
                    "count": len(normalized),
                    "max_results": max_results,
                    "event_types": settings.KEYCLOAK_EVENTS_CONFIG["LOGIN_EVENT_TYPES"],
                },
            }
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def __get_current_user(request):
    """Return current authenticated user from Keycloak token"""
    user = request.user
    
    if not user or not user.is_authenticated:
        return Response(
            {'error': 'User information not available'},
            status=status.HTTP_401_UNAUTHORIZED
        )
    
    return Response({
        'username': user.preferred_username,
        'email': user.email,
        'fullName': user.name,
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def __logout_current_user(request):
    """Logs a user out by revoking their refresh token."""
    refresh_token = request.data.get('refresh_token')
    
    if not refresh_token:
        return Response(
            {'error': 'Refresh token is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        __logout_user(refresh_token)
        return Response({'message': 'Logged out successfully'}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response(
            {'error': f'Logout failed: {str(e)}'},
            status=status.HTTP_400_BAD_REQUEST
        )
