from rest_framework import serializers

from .models import NotificationType


class EmptySerializer(serializers.Serializer):
    pass


class NotificationErrorResponseSerializer(serializers.Serializer):
    error = serializers.CharField()


class NotificationSendRequestSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255)
    body = serializers.CharField()
    type = serializers.ChoiceField(choices=NotificationType.values, default=NotificationType.INFO)
    data = serializers.JSONField(required=False, default=dict)


class NotificationItemSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    title = serializers.CharField()
    body = serializers.CharField()
    type = serializers.ChoiceField(choices=NotificationType.values)
    created_at = serializers.DateTimeField()
    extra_data = serializers.JSONField()
    is_read = serializers.BooleanField()
    sent_by = serializers.CharField()
    recipient_id = serializers.IntegerField(allow_null=True)


class NotificationSendResponseSerializer(serializers.Serializer):
    success = serializers.BooleanField()
    id = serializers.UUIDField()
    message = serializers.CharField()


class NotificationListResponseSerializer(serializers.Serializer):
    count = serializers.IntegerField()
    results = NotificationItemSerializer(many=True)


class NotificationDeleteResponseSerializer(serializers.Serializer):
    success = serializers.BooleanField()
