from django.contrib import admin
from .models import Notification


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ["title", "type", "sent_by", "sent_at"]
    list_filter = ["type", "sent_at"]
    search_fields = ["title", "body"]
    ordering = ["-sent_at"]
    readonly_fields = ["id", "sent_at"]

   # ← Direct Send button from Admin Panel

    actions = ["resend_notification"]

    def resend_notification(self, request, queryset):
        from .services import broadcast_notification
        for notification in queryset:
            broadcast_notification(notification)
        self.message_user(request, f"{queryset.count()} notification(s) resent.")

    resend_notification.short_description = "📡 Resend via WebSocket"
