"""
One-time import: local Django ML JSON configs → Django ORM.
Run: python manage.py import_ml_config
"""

import json

from django.conf import settings
from django.core.management.base import BaseCommand

from ml.core.model_config_sync import sync_model_config_file
from ml.models import OrchestratorVariant


class Command(BaseCommand):
    help = "Import local Django ML JSON configs into the Django database"

    def handle(self, *args, **options):
        self._import_models()
        self._import_orchestrator()

    def _import_models(self):
        path = settings.PIOS_MODEL_CONFIG_PATH
        if not path.exists():
            self.stderr.write(f"Not found: {path}")
            return

        summary = sync_model_config_file(path, reason="management-import")
        self.stdout.write(
            self.style.SUCCESS(
                "✅ "
                f"{summary['models_total']} model configs synced "
                f"({summary['created_total']} created, "
                f"{summary['updated_total']} updated, "
                f"{summary['deactivated_removed_total']} deactivated)"
            )
        )

    def _import_orchestrator(self):
        path = settings.PIOS_ORCHESTRATOR_CONFIG_PATH
        if not path.exists():
            self.stderr.write(f"Not found: {path}")
            return

        data = json.loads(path.read_text("utf-8"))
        selected = data.get("selected_model_id")
        count = 0
        for row in data.get("models", []):
            vid = row.get("id", "").strip()
            if not vid:
                continue
            obj, created = OrchestratorVariant.objects.update_or_create(
                variant_id=vid,
                defaults={
                    "name": row.get("name", vid),
                    "description": row.get("description", ""),
                    "repo_id": row.get("repo_id", "bartowski/xLAM-7b-fc-r-GGUF"),
                    "filename": row.get("filename", ""),
                    "size_mb": row.get("size_mb"),
                    "size_bytes": row.get("size_bytes"),
                    "min_ram_gb": row.get("min_ram_gb", 0),
                    "min_vram_mb": row.get("min_vram_mb", 0),
                    "n_ctx": row.get("n_ctx", 4096),
                    "n_batch": row.get("n_batch", 128),
                    "is_selected": vid == selected,
                },
            )
            self.stdout.write(f"  {'Created' if created else 'Updated'}: {vid}")
            count += 1
        self.stdout.write(
            self.style.SUCCESS(f"✅ {count} orchestrator variants imported")
        )
