from django.test import TestCase
from rest_framework.test import APIClient

from authApp.models import User
from notifications.models import Notification, NotificationType
from notifications.services import publish_notification


class KeycloakLikeUser:
    is_authenticated = True

    def __init__(self, django_user):
        self.django_user = django_user

    def __getattr__(self, name):
        if name in {"pk", "id"}:
            return getattr(self.django_user, name)
        return None


class NotificationApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email="chaimaa@example.com",
            name="Chaimaa",
            password="secret123",
        )
        self.other_user = User.objects.create_user(
            email="romaissa@example.com",
            name="Romaissa",
            password="secret123",
        )
        self.client.force_authenticate(user=self.user)

    def test_list_notifications_returns_only_authenticated_user_items(self):
        own_notification = publish_notification(
            title="Workspace shared with you",
            body='Romaissa shared this workspace with you: "My Dashboard"',
            notif_type=NotificationType.WORKSPACE_SHARE,
            extra_data={"workspace_id": 12},
            sent_by="romaissa@example.com",
            recipient=self.user,
        )
        publish_notification(
            title="Other notification",
            body="Should not be visible",
            notif_type=NotificationType.INFO,
            sent_by="system",
            recipient=self.other_user,
        )

        response = self.client.get("/api/notifications/?limit=20")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["results"][0]["id"], str(own_notification.id))
        self.assertEqual(payload["results"][0]["created_at"], own_notification.sent_at.isoformat())
        self.assertEqual(payload["results"][0]["extra_data"]["workspace_id"], 12)
        self.assertFalse(payload["results"][0]["is_read"])

    def test_mark_as_read_persists_notification_state(self):
        notification = Notification.objects.create(
            title="Workspace shared with you",
            body='Romaissa shared this workspace with you: "My Dashboard"',
            type=NotificationType.WORKSPACE_SHARE,
            recipient=self.user,
        )

        response = self.client.patch(f"/api/notifications/{notification.id}/mark-as-read/")

        self.assertEqual(response.status_code, 200)
        notification.refresh_from_db()
        self.assertTrue(notification.is_read)

    def test_mark_as_read_accepts_keycloak_user_wrapper(self):
        self.client.force_authenticate(user=KeycloakLikeUser(self.user))
        notification = Notification.objects.create(
            title="Workspace shared with you",
            body='Romaissa shared this workspace with you: "My Dashboard"',
            type=NotificationType.WORKSPACE_SHARE,
            recipient=self.user,
        )

        response = self.client.patch(f"/api/notifications/{notification.id}/mark-as-read/")

        self.assertEqual(response.status_code, 200)
        notification.refresh_from_db()
        self.assertTrue(notification.is_read)

    def test_mark_all_as_read_updates_all_user_notifications(self):
        first = Notification.objects.create(
            title="One",
            body="First",
            type=NotificationType.INFO,
            recipient=self.user,
        )
        second = Notification.objects.create(
            title="Two",
            body="Second",
            type=NotificationType.INFO,
            recipient=self.user,
        )
        Notification.objects.create(
            title="Three",
            body="Other user",
            type=NotificationType.INFO,
            recipient=self.other_user,
        )

        response = self.client.patch("/api/notifications/mark-all-as-read/")

        self.assertEqual(response.status_code, 200)
        first.refresh_from_db()
        second.refresh_from_db()
        self.assertTrue(first.is_read)
        self.assertTrue(second.is_read)
        self.assertFalse(Notification.objects.get(recipient=self.other_user).is_read)

    def test_mark_all_as_read_accepts_keycloak_user_wrapper(self):
        self.client.force_authenticate(user=KeycloakLikeUser(self.user))
        first = Notification.objects.create(
            title="One",
            body="First",
            type=NotificationType.INFO,
            recipient=self.user,
        )
        other = Notification.objects.create(
            title="Other",
            body="Other user",
            type=NotificationType.INFO,
            recipient=self.other_user,
        )

        response = self.client.patch("/api/notifications/mark-all-as-read/")

        self.assertEqual(response.status_code, 200)
        first.refresh_from_db()
        other.refresh_from_db()
        self.assertTrue(first.is_read)
        self.assertFalse(other.is_read)

    def test_delete_accepts_keycloak_user_wrapper(self):
        self.client.force_authenticate(user=KeycloakLikeUser(self.user))
        notification = Notification.objects.create(
            title="Delete me",
            body="Body",
            type=NotificationType.INFO,
            recipient=self.user,
        )

        response = self.client.delete(f"/api/notifications/{notification.id}/")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(Notification.objects.filter(pk=notification.pk).exists())
