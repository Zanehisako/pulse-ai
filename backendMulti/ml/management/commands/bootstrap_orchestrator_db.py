"""
Create and seed the Postgres tables used by the orchestrator db_search tool.

Usage:
    python manage.py bootstrap_orchestrator_db
"""
from __future__ import annotations

import os
from pathlib import Path

from django.core.management.base import BaseCommand

from ml.core.external_tools import (
    DEFAULT_SQLITE_DB_PATH,
    ensure_postgres_reference_tables,
    ensure_sqlite_reference_tables,
)
from ml.core.utils import safe_slug


PROJECT_ROOT = Path(__file__).resolve().parents[4]
ML_BACKEND_DIR = PROJECT_ROOT / "ml-backend"


class Command(BaseCommand):
    help = "Create and seed the Postgres donors/hospitals tables used by the orchestrator."

    def handle(self, *args, **options):
        donor_csv_path = Path(
            (
                os.getenv("PIOS_ORCH_DB_DONOR_CSV_PATH", "").strip()
                or str(
                    ML_BACKEND_DIR
                    / "datasets"
                    / "blood_registry_sythentic"
                    / "data"
                    / "blood_donation_registry_ml_ready.csv"
                )
            )
        ).expanduser().resolve()
        hospital_csv_path = Path(
            (
                os.getenv("PIOS_ORCH_DB_SUPPLY_CSV_PATH", "").strip()
                or str(
                    ML_BACKEND_DIR
                    / "datasets_featues_labels_seperated"
                    / "hospital_supply_features.parquet"
                )
            )
        ).expanduser().resolve()
        db_schema = safe_slug(os.getenv("PIOS_ORCH_DB_SCHEMA", "public"))
        donors_table_name = safe_slug(
            os.getenv("PIOS_ORCH_DB_DONORS_TABLE", "orchestrator_donors")
        )
        hospitals_table_name = safe_slug(
            os.getenv("PIOS_ORCH_DB_HOSPITALS_TABLE", "orchestrator_hospitals")
        )
        sqlite_path = (
            os.getenv("PIOS_ORCH_DB_SQLITE_PATH", "").strip()
            or str(DEFAULT_SQLITE_DB_PATH)
        )

        try:
            ensure_postgres_reference_tables(
                db_schema=db_schema,
                donors_table_name=donors_table_name,
                hospitals_table_name=hospitals_table_name,
                donor_csv_path=donor_csv_path,
                hospital_csv_path=hospital_csv_path,
                required_tables={"donors", "hospitals"},
            )
            self.stdout.write(
                self.style.SUCCESS(
                    "Bootstrapped orchestrator Postgres tables "
                    f"{db_schema}.{donors_table_name} and {db_schema}.{hospitals_table_name}"
                )
            )
            return
        except Exception as exc:
            self.stderr.write(
                "Postgres bootstrap failed; creating SQLite fallback instead: "
                f"{exc}"
            )

        created_path = ensure_sqlite_reference_tables(
            db_sqlite_path=sqlite_path,
            donors_table_name=donors_table_name,
            hospitals_table_name=hospitals_table_name,
            donor_csv_path=donor_csv_path,
            hospital_csv_path=hospital_csv_path,
            required_tables={"donors", "hospitals"},
        )
        self.stdout.write(
            self.style.SUCCESS(
                "Bootstrapped orchestrator SQLite fallback tables "
                f"{donors_table_name} and {hospitals_table_name} at {created_path}"
            )
        )
