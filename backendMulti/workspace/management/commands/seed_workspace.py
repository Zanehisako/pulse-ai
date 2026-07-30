from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import transaction

from authApp.models import User
from inventory.models import BloodSupply, Donor, Hospital
from workspace.models import (
    DashboardPersonnel,
    PartageDashboard,
    WidgetsLayout,
)
from workspace.services.widget_sources import list_widget_sources

DEMO_USERS = (
    ("workspace.owner@pios.local", "Workspace Owner"),
    ("workspace.collaborator@pios.local", "Workspace Collaborator"),
    ("workspace.viewer@pios.local", "Workspace Viewer"),
)

HOSPITAL_ROWS = (
    {
        "hospital_id": "WS-H001",
        "name": "PIOS Seed Centre Quebec",
        "wilaya": "quebec",
    },
    {
        "hospital_id": "WS-H002",
        "name": "PIOS Seed Centre Montreal",
        "wilaya": "montreal",
    },
    {
        "hospital_id": "WS-H003",
        "name": "PIOS Seed Centre Laval",
        "wilaya": "laval",
    },
)

DONOR_ROWS = (
    {
        "donor_id": "WS-D0001",
        "name": "Seed Alice",
        "wilaya": "quebec",
        "city_id": "WS-C001",
        "lat": 46.8139,
        "lon": -71.2080,
        "availability": 1,
        "blood_group": "O+",
        "recency_days": 22,
        "frequency_365": 8,
        "days_until_eligible": 0,
        "cluster_id": 1,
    },
    {
        "donor_id": "WS-D0002",
        "name": "Seed Brahim",
        "wilaya": "montreal",
        "city_id": "WS-C002",
        "lat": 45.5019,
        "lon": -73.5674,
        "availability": 1,
        "blood_group": "A+",
        "recency_days": 15,
        "frequency_365": 11,
        "days_until_eligible": 0,
        "cluster_id": 2,
    },
    {
        "donor_id": "WS-D0003",
        "name": "Seed Chloe",
        "wilaya": "laval",
        "city_id": "WS-C003",
        "lat": 45.6066,
        "lon": -73.7124,
        "availability": 0,
        "blood_group": "B+",
        "recency_days": 60,
        "frequency_365": 4,
        "days_until_eligible": 0,
        "cluster_id": 3,
    },
)

BLOOD_SUPPLY_ROWS = (
    {
        "supply_id": "WS-S001",
        "hospital_id": "WS-H001",
        "blood_product_type": "O+",
        "current_stock_units": 42,
        "usage_today": 6,
        "lead_time_days": 2,
        "days_since_last_restock": 1,
        "stockout_count_90d": 0,
        "scheduled_surgeries_next7d": 3,
    },
    {
        "supply_id": "WS-S002",
        "hospital_id": "WS-H001",
        "blood_product_type": "A+",
        "current_stock_units": 28,
        "usage_today": 5,
        "lead_time_days": 3,
        "days_since_last_restock": 2,
        "stockout_count_90d": 1,
        "scheduled_surgeries_next7d": 2,
    },
    {
        "supply_id": "WS-S003",
        "hospital_id": "WS-H002",
        "blood_product_type": "O+",
        "current_stock_units": 35,
        "usage_today": 7,
        "lead_time_days": 2,
        "days_since_last_restock": 1,
        "stockout_count_90d": 0,
        "scheduled_surgeries_next7d": 4,
    },
    {
        "supply_id": "WS-S004",
        "hospital_id": "WS-H002",
        "blood_product_type": "B+",
        "current_stock_units": 19,
        "usage_today": 3,
        "lead_time_days": 4,
        "days_since_last_restock": 3,
        "stockout_count_90d": 2,
        "scheduled_surgeries_next7d": 1,
    },
    {
        "supply_id": "WS-S005",
        "hospital_id": "WS-H003",
        "blood_product_type": "O-",
        "current_stock_units": 11,
        "usage_today": 2,
        "lead_time_days": 5,
        "days_since_last_restock": 4,
        "stockout_count_90d": 3,
        "scheduled_surgeries_next7d": 1,
    },
)


def _widget_payload(source: dict, *, widget_id: str, title_suffix: str, position: int) -> dict:
    metric = source.get("metrics", [])[0]
    group_by = source.get("default_group_by") or source["group_by"][0]["value"]
    return {
        "widget_id": widget_id,
        "position_x": 0,
        "position_y": position,
        "width": 3,
        "height": 2,
        "ordre_z": position,
        "config": {
            "type": "bar",
            "title": f"{source['label']} {title_suffix}",
            "table": source["id"],
            "group_by": group_by,
            "metric": metric["value"],
            "field": metric.get("field", "id"),
        },
    }


class Command(BaseCommand):
    help = "Seed workspace demo data (widget sources, sample dashboards, shares)"

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true")

    def handle(self, *args, **options):
        with transaction.atomic():
            if options["reset"]:
                self._reset_seed_data()

            users = self._seed_users()
            hospitals = self._seed_hospitals()
            self._seed_donors()
            self._seed_blood_supply(hospitals)
            self._seed_dashboards(users)

        self.stdout.write(self.style.SUCCESS("Workspace demo data ready."))

    def _reset_seed_data(self) -> None:
        demo_emails = [email for email, _name in DEMO_USERS]
        demo_users = list(User.objects.filter(email__in=demo_emails))
        demo_dashboards = DashboardPersonnel.objects.filter(id_utilisateur__in=demo_users)

        PartageDashboard.objects.filter(id_dashboard__in=demo_dashboards).delete()
        WidgetsLayout.objects.filter(id_dashboard__in=demo_dashboards).delete()
        demo_dashboards.delete()
        User.objects.filter(email__in=demo_emails).delete()

        hospital_ids = [row["hospital_id"] for row in HOSPITAL_ROWS]
        BloodSupply.objects.filter(supply_id__startswith="WS-S").delete()
        Hospital.objects.filter(hospital_id__in=hospital_ids).delete()
        Donor.objects.filter(donor_id__startswith="WS-D").delete()

    def _seed_users(self) -> dict[str, User]:
        users: dict[str, User] = {}
        for email, name in DEMO_USERS:
            user, _created = User.objects.get_or_create(
                email=email,
                defaults={"name": name, "is_active": True},
            )
            users[email] = user
        return users

    def _seed_hospitals(self) -> dict[str, Hospital]:
        hospitals: dict[str, Hospital] = {}
        for row in HOSPITAL_ROWS:
            hospital, _created = Hospital.objects.get_or_create(**row)
            hospitals[row["hospital_id"]] = hospital
        return hospitals

    def _seed_donors(self) -> None:
        from django.utils import timezone

        for row in DONOR_ROWS:
            Donor.objects.update_or_create(
                donor_id=row["donor_id"],
                defaults={
                    **row,
                    "event_timestamp": timezone.now(),
                },
            )

    def _seed_blood_supply(self, hospitals: dict[str, Hospital]) -> None:
        from django.utils import timezone

        for row in BLOOD_SUPPLY_ROWS:
            hospital_id = row["hospital_id"]
            BloodSupply.objects.update_or_create(
                supply_id=row["supply_id"],
                defaults={
                    "hospital": hospitals[hospital_id],
                    "blood_product_type": row["blood_product_type"],
                    "current_stock_units": row["current_stock_units"],
                    "usage_today": row["usage_today"],
                    "lead_time_days": row["lead_time_days"],
                    "days_since_last_restock": row["days_since_last_restock"],
                    "stockout_count_90d": row["stockout_count_90d"],
                    "scheduled_surgeries_next7d": row["scheduled_surgeries_next7d"],
                    "event_timestamp": timezone.now(),
                },
            )

    def _seed_dashboards(self, users: dict[str, User]) -> None:
        sources = list_widget_sources()
        if not sources:
            raise RuntimeError("No workspace widget sources are configured")

        owner = users["workspace.owner@pios.local"]
        collaborator = users["workspace.collaborator@pios.local"]
        viewer = users["workspace.viewer@pios.local"]

        primary_sources = sources[: min(3, len(sources))]
        template_source = sources[min(3, len(sources) - 1)]

        team_dashboard = self._upsert_dashboard(
            user=owner,
            name="Seed Team Workspace",
            description="Shared workspace seeded for collaboration tests.",
            auto_refresh_ms=15000,
            widgets=[
                _widget_payload(source, widget_id=f"seed-widget-{index}", title_suffix="Overview", position=index)
                for index, source in enumerate(primary_sources)
            ],
        )
        self._upsert_share(team_dashboard, collaborator, "share")
        self._upsert_share(team_dashboard, viewer, "view")

        self._upsert_dashboard(
            user=owner,
            name="Seed Inventory Template",
            description="Template workspace seeded from widget source metadata.",
            est_template=True,
            type_template="manager",
            widgets=[
                _widget_payload(
                    template_source,
                    widget_id="seed-template-widget-0",
                    title_suffix="Template",
                    position=0,
                )
            ],
        )

    def _upsert_dashboard(
        self,
        *,
        user: User,
        name: str,
        description: str,
        widgets: list[dict],
        auto_refresh_ms: int = 0,
        est_template: bool = False,
        type_template: str | None = None,
    ) -> DashboardPersonnel:
        dashboard, _created = DashboardPersonnel.objects.update_or_create(
            id_utilisateur=user,
            nom=name,
            defaults={
                "description": description,
                "config": {},
                "auto_refresh_ms": auto_refresh_ms,
                "est_template": est_template,
                "type_template": type_template,
                "is_draft": False,
            },
        )

        dashboard.widgets_layout.all().delete()
        WidgetsLayout.objects.bulk_create(
            [
                WidgetsLayout(
                    id_dashboard=dashboard,
                    widget_id=widget["widget_id"],
                    position_x=widget["position_x"],
                    position_y=widget["position_y"],
                    width=widget["width"],
                    height=widget["height"],
                    ordre_z=widget["ordre_z"],
                    config=widget["config"],
                )
                for widget in widgets
            ]
        )
        return dashboard

    def _upsert_share(
        self,
        dashboard: DashboardPersonnel,
        user: User,
        permission: str,
    ) -> None:
        PartageDashboard.objects.update_or_create(
            id_dashboard=dashboard,
            id_utilisateur_dest=user,
            defaults={"permission": permission},
        )
