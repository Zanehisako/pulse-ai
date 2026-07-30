from django.db import models
from django.utils import timezone
import uuid
from authApp.models import User


class NotificationType(models.TextChoices):
    INFO = "info", "Info"
    SUCCESS = "success", "Success"
    WARNING = "warning", "Warning"
    BLOOD_DONATION = "blood_donation", "Blood Donation"
    WORKSPACE_SHARE = "workspace_share", "Workspace Share"


class Notification(models.Model):
    """
   Record all sent notifications
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    body = models.TextField()
    type = models.CharField(
        max_length=20,
        choices=NotificationType.choices,
        default=NotificationType.INFO,
    )
    extra_data = models.JSONField(default=dict, blank=True)
    sent_at = models.DateTimeField(default=timezone.now)
    sent_by = models.CharField(max_length=100, default="admin")
    recipient = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="notifications",
    )
    is_read = models.BooleanField(default=False)

    class Meta:
        ordering = ["-sent_at"]
        verbose_name = "Notification"
        verbose_name_plural = "Notifications"

    def __str__(self):
        return f"[{self.type}] {self.title} — {self.sent_at.strftime('%Y-%m-%d %H:%M')}"

    def to_dict(self):
        return {
            "id": str(self.id),
            "title": self.title,
            "body": self.body,
            "type": self.type,
            "created_at": self.sent_at.isoformat(),
            "extra_data": self.extra_data,
            "is_read": self.is_read,
            "sent_by": self.sent_by,
            "recipient_id": self.recipient_id,
        }
