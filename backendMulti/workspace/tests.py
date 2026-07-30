from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from django.core.management import call_command
from django.core.exceptions import ImproperlyConfigured
from django.test import TestCase, override_settings
from rest_framework.test import APIClient
from unittest.mock import patch

from authApp.models import User
from inventory.models import BloodSupply, Donor, Hospital
from notifications.models import Notification, NotificationType
from workspace.models import DashboardPersonnel, PartageDashboard, WidgetsLayout
from workspace.services.widget_sources import (
    get_widget_sources_catalog,
    reset_widget_sources_cache,
)


class WorkspaceApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.owner = User.objects.create_user(
            email="owner@example.com",
            name="Owner",
            password="secret123",
        )
        self.collaborator = User.objects.create_user(
            email="collab@example.com",
            name="Collaborator",
            password="secret123",
        )
        self.dashboard = DashboardPersonnel.objects.create(
            id_utilisateur=self.owner,
            nom="Operations Workspace",
            description="Shared operational dashboard.",
            config={},
            auto_refresh_ms=5000,
        )
        WidgetsLayout.objects.create(
            id_dashboard=self.dashboard,
            widget_id="widget-1",
            position_x=0,
            position_y=0,
            width=3,
            height=2,
            ordre_z=0,
            config={
                "type": "bar",
                "title": "Stock overview",
                "table": "stock",
                "group_by": "groupe_sanguin",
                "metric": "sum",
                "field": "quantite",
            },
        )
        self.client.force_authenticate(user=self.owner)

    def test_retrieve_workspace_returns_single_dashboard_payload(self):
        response = self.client.get(f"/api/workspace/workspaces/{self.dashboard.id_dashboard}/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["id_dashboard"], self.dashboard.id_dashboard)
        self.assertEqual(len(payload["widgets_layout"]), 1)
        self.assertTrue(payload["est_owner"])

    def test_favorite_returns_full_workspace_payload_and_persists(self):
        response = self.client.patch(
            f"/api/workspace/workspaces/{self.dashboard.id_dashboard}/favorite/",
            {},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["id_dashboard"], self.dashboard.id_dashboard)
        self.assertTrue(payload["est_favori"])

        self.dashboard.refresh_from_db()
        self.assertTrue(self.dashboard.est_favori)

    def test_share_by_email_persists_permission(self):
        response = self.client.post(
            f"/api/workspace/workspaces/{self.dashboard.id_dashboard}/share/",
            {"user_email": self.collaborator.email, "permission": "edit"},
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["permission"], "edit")
        self.assertEqual(payload["utilisateur_dest"]["email"], self.collaborator.email)
        self.assertTrue(
            PartageDashboard.objects.filter(
                id_dashboard=self.dashboard,
                id_utilisateur_dest=self.collaborator,
                permission="edit",
            ).exists()
        )
        notification = Notification.objects.get(recipient=self.collaborator)
        self.assertEqual(notification.type, NotificationType.WORKSPACE_SHARE)
        self.assertIn("Owner", notification.body)
        self.assertEqual(notification.extra_data["owner_name"], "Owner")
        self.assertEqual(notification.extra_data["workspace_id"], self.dashboard.id_dashboard)

    @patch("workspace.views.sync_keycloak_user_by_email")
    def test_share_by_email_syncs_missing_local_user_from_keycloak(
        self,
        mock_sync_keycloak_user_by_email,
    ):
        mock_sync_keycloak_user_by_email.side_effect = lambda _email: User.objects.create(
            email="romaissaa13@gmail.com",
            name="Romaissaa",
            is_active=True,
        )

        response = self.client.post(
            f"/api/workspace/workspaces/{self.dashboard.id_dashboard}/share/",
            {"user_email": "romaissaa13@gmail.com", "permission": "view"},
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["utilisateur_dest"]["email"], "romaissaa13@gmail.com")
        mock_sync_keycloak_user_by_email.assert_called_once_with("romaissaa13@gmail.com")

    def test_widget_sources_metadata_endpoint_returns_configured_catalog(self):
        response = self.client.get("/api/workspace/metadata/widget-sources/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("sources", payload)
        self.assertTrue(any(source["id"] == "stock" for source in payload["sources"]))

        routed_response = self.client.get(
            "/api/workspace/workspaces/metadata/widget-sources/"
        )
        self.assertEqual(routed_response.status_code, 200)

    def test_shared_endpoint_includes_workspace_owner_identity(self):
        PartageDashboard.objects.create(
            id_dashboard=self.dashboard,
            id_utilisateur_dest=self.collaborator,
            permission="view",
        )
        self.client.force_authenticate(user=self.collaborator)

        response = self.client.get("/api/workspace/workspaces/shared/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload[0]["utilisateur"]["name"], "Owner")
        self.assertEqual(payload[0]["owner_name"], "Owner")

    def test_query_data_uses_metric_field_from_source_config(self):
        hospital = Hospital.objects.create(
            hospital_id="H001",
            name="Centre 1",
            wilaya="montreal",
        )
        BloodSupply.objects.create(
            supply_id="S001",
            hospital=hospital,
            blood_product_type="O+",
            current_stock_units=12,
            usage_today=4,
            lead_time_days=2,
            days_since_last_restock=1,
            stockout_count_90d=0,
            scheduled_surgeries_next7d=1,
            event_timestamp="2026-01-01T00:00:00Z",
        )
        BloodSupply.objects.create(
            supply_id="S002",
            hospital=hospital,
            blood_product_type="O+",
            current_stock_units=7,
            usage_today=2,
            lead_time_days=2,
            days_since_last_restock=1,
            stockout_count_90d=0,
            scheduled_surgeries_next7d=1,
            event_timestamp="2026-01-01T00:00:00Z",
        )

        response = self.client.post(
            "/api/workspace/workspaces/query_data/",
            {
                "model": "stock",
                "group_by": "blood_product_type",
                "metric": "sum",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [{"blood_product_type": "O+", "value": 19.0}])


class WorkspaceWidgetSourceConfigTests(TestCase):
    def tearDown(self):
        reset_widget_sources_cache()

    def test_invalid_widget_source_config_fails_clearly(self):
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "widget_sources.json"
            config_path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "id": "stock",
                                "label": "Stock",
                                "model": "Stock",
                                "group_by": [{"value": "groupe_sanguin", "label": "Groupe sanguin"}],
                                "metrics": [{"value": "median", "label": "Median"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with override_settings(WORKSPACE_WIDGET_SOURCES_CONFIG_PATH=config_path):
                with self.assertRaises(ImproperlyConfigured):
                    get_widget_sources_catalog(reload=True)


class SeedWorkspaceCommandTests(TestCase):
    def tearDown(self):
        reset_widget_sources_cache()

    def test_seed_workspace_creates_demo_dashboards_and_shares(self):
        call_command("seed_workspace")

        self.assertTrue(
            DashboardPersonnel.objects.filter(nom="Seed Team Workspace").exists()
        )
        self.assertTrue(WidgetsLayout.objects.exists())
        self.assertTrue(
            PartageDashboard.objects.filter(permission="share").exists()
        )
        self.assertTrue(Hospital.objects.filter(hospital_id__startswith="WS-H").exists())
        self.assertTrue(BloodSupply.objects.filter(supply_id__startswith="WS-S").exists())
        self.assertTrue(Donor.objects.filter(donor_id__startswith="WS-D").exists())

    def test_seed_workspace_is_idempotent(self):
        call_command("seed_workspace")
        first_counts = {
            "dashboards": DashboardPersonnel.objects.count(),
            "widgets": WidgetsLayout.objects.count(),
            "shares": PartageDashboard.objects.count(),
            "hospitals": Hospital.objects.count(),
            "stock": BloodSupply.objects.count(),
            "donors": Donor.objects.count(),
        }

        call_command("seed_workspace")
        second_counts = {
            "dashboards": DashboardPersonnel.objects.count(),
            "widgets": WidgetsLayout.objects.count(),
            "shares": PartageDashboard.objects.count(),
            "hospitals": Hospital.objects.count(),
            "stock": BloodSupply.objects.count(),
            "donors": Donor.objects.count(),
        }

        self.assertEqual(second_counts, first_counts)
