from drf_spectacular.utils import (
    OpenApiExample,
    OpenApiParameter,
    OpenApiTypes,
    extend_schema,
)
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from .models import Notification, NotificationType
from .serializers import (
    EmptySerializer,
    NotificationDeleteResponseSerializer,
    NotificationErrorResponseSerializer,
    NotificationListResponseSerializer,
    NotificationSendRequestSerializer,
    NotificationSendResponseSerializer,
)
from .services import publish_notification


def _authenticated_user_id(user):
    if not user or not getattr(user, "is_authenticated", False):
        return None
    user_id = getattr(user, "pk", None) or getattr(user, "id", None)
    if user_id is not None:
        return user_id
    django_user = getattr(user, "django_user", None)
    return getattr(django_user, "pk", None) or getattr(django_user, "id", None)


class SendNotificationView(APIView):
    serializer_class = NotificationSendRequestSerializer
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Notifications"],
        operation_id="notifications_send",
        summary="Send a notification to connected clients",
        request=NotificationSendRequestSerializer,
        responses={
            status.HTTP_201_CREATED: NotificationSendResponseSerializer,
            status.HTTP_400_BAD_REQUEST: NotificationErrorResponseSerializer,
        },
        examples=[
            OpenApiExample(
                "Broadcast notification",
                value={
                    "title": "Stock alert",
                    "body": "O- units are running low.",
                    "type": "warning",
                    "data": {"blood_type": "O-", "threshold": 8},
                },
                request_only=True,
            ),
        ],
    )
    def post(self, request):
        title = request.data.get("title", "").strip()
        body = request.data.get("body", "").strip()
        notif_type = request.data.get("type", NotificationType.INFO)
        extra_data = request.data.get("data", {})

        if not title or not body:
            return Response(
                {"error": "title and body are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if notif_type not in NotificationType.values:
            return Response(
                {"error": f"type must be one of: {NotificationType.values}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        notification = publish_notification(
            title=title,
            body=body,
            notif_type=notif_type,
            extra_data=extra_data,
            sent_by=str(request.user) if request.user.is_authenticated else "admin",
        )

        return Response(
            {
                "success": True,
                "id": str(notification.id),
                "message": "Notification sent",
            },
            status=status.HTTP_201_CREATED,
        )


class NotificationListView(APIView):
    serializer_class = EmptySerializer
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Notifications"],
        operation_id="notifications_list",
        summary="List recent notifications for the authenticated user",
        parameters=[
            OpenApiParameter(
                name="limit",
                type=OpenApiTypes.INT,
                location=OpenApiParameter.QUERY,
                required=False,
                description="Maximum number of notifications to return.",
            ),
        ],
        responses={status.HTTP_200_OK: NotificationListResponseSerializer},
    )
    def get(self, request):
        limit = int(request.query_params.get("limit", 20))
        user_id = _authenticated_user_id(request.user)
        if not user_id:
            return Response({"count": 0, "results": []})
        notifications = Notification.objects.filter(recipient_id=user_id)[:limit]
        return Response(
            {
                "count": notifications.count(),
                "results": [notification.to_dict() for notification in notifications],
            }
        )


class MarkNotificationAsReadView(APIView):
    serializer_class = EmptySerializer
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Notifications"],
        operation_id="notifications_mark_read",
        summary="Mark a notification as read",
        responses={
            status.HTTP_200_OK: NotificationDeleteResponseSerializer,
            status.HTTP_404_NOT_FOUND: NotificationErrorResponseSerializer,
        },
    )
    def patch(self, request, pk):
        user_id = _authenticated_user_id(request.user)
        if not user_id:
            return Response(
                {"detail": "Authentication required"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        notification = Notification.objects.filter(pk=pk, recipient_id=user_id).first()
        if not notification:
            return Response(
                {"error": "Not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        notification.is_read = True
        notification.save(update_fields=["is_read"])
        return Response({"success": True})

    post = patch


class MarkAllNotificationsAsReadView(APIView):
    serializer_class = EmptySerializer
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Notifications"],
        operation_id="notifications_mark_all_read",
        summary="Mark all notifications as read",
        responses={status.HTTP_200_OK: NotificationDeleteResponseSerializer},
    )
    def patch(self, request):
        user_id = _authenticated_user_id(request.user)
        if not user_id:
            return Response(
                {"detail": "Authentication required"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        Notification.objects.filter(recipient_id=user_id, is_read=False).update(
            is_read=True
        )
        return Response({"success": True})

    post = patch


class DeleteNotificationView(APIView):
    serializer_class = EmptySerializer
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Notifications"],
        operation_id="notifications_delete",
        summary="Delete a notification",
        responses={
            status.HTTP_200_OK: NotificationDeleteResponseSerializer,
            status.HTTP_404_NOT_FOUND: NotificationErrorResponseSerializer,
        },
    )
    def delete(self, request, pk):
        user_id = _authenticated_user_id(request.user)
        if not user_id:
            return Response(
                {"detail": "Authentication required"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        notification = Notification.objects.filter(pk=pk, recipient_id=user_id).first()
        if not notification:
            return Response(
                {"error": "Not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        notification.delete()
        return Response({"success": True})
