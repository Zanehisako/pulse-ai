# middleware.py
from functools import wraps
from django.http import JsonResponse
from .keycloak_auth import get_keycloak_client

def require_auth(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return JsonResponse({'error': 'Missing token'}, status=401)
        
        token = auth_header.split(' ')[1]
        try:
            kc = get_keycloak_client()
            token_info = kc.introspect(token)
            if not token_info.get('active'):
                return JsonResponse({'error': 'Token inactive'}, status=401)
            request.user_info = token_info  # attach user info to request
        except Exception:
            return JsonResponse({'error': 'Invalid token'}, status=401)
        
        return view_func(request, *args, **kwargs)
    return wrapper