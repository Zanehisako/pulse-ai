from __future__ import annotations

import logging
from urllib.parse import parse_qs

from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from django.contrib.auth.models import AnonymousUser

logger = logging.getLogger(__name__)


@database_sync_to_async
def get_user_from_token(token: str):
    """
    Resolve a DRF token or JWT to a User.
    Tries DRF Token first, then falls back to AnonymousUser.
    """
    token = str(token or "").strip()
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
    if not token:
        return AnonymousUser()

    try:
        from rest_framework.authtoken.models import Token

        return Token.objects.select_related("user").get(key=token).user
    except Exception:
        pass

    try:
        from django.contrib.auth import get_user_model
        from rest_framework_simplejwt.tokens import AccessToken

        user_model = get_user_model()
        payload = AccessToken(token)
        return user_model.objects.get(id=payload["user_id"])
    except Exception:
        pass

    try:
        from backendMulti.services.authentication import (
            django_user_from_token_info,
            validate_token,
        )

        token_info = validate_token(token)
        if token_info.get("active"):
            user = django_user_from_token_info(token_info)
            if user is not None:
                return user
    except Exception:
        logger.debug("Unable to authenticate websocket token.", exc_info=True)

    return AnonymousUser()


class TokenAuthMiddleware(BaseMiddleware):
    """
    Reads token from query string: ws://host/ws/alerts/?token=<TOKEN>
    Sets scope["user"] so consumers can check authentication.
    """

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("websocket", "http"):
            query_string = scope.get("query_string", b"").decode()
            params = parse_qs(query_string)
            token_list = params.get("token", [])

            if token_list:
                scope["user"] = await get_user_from_token(token_list[0])
            elif "user" not in scope:
                scope["user"] = AnonymousUser()

        return await super().__call__(scope, receive, send)
