# testUrl/management/commands/refresh_quebec_data.py
import hashlib, json
from django.core.management.base import BaseCommand
from django.utils import timezone
from testUrl.models import DataVersion

class Command(BaseCommand):
    help = 'Refresh Quebec donneurs data and update version timestamp'

    def handle(self, *args, **kwargs):
       # 1. Here you put the actual data update logic ──────────────────

         

        # ── 2. Calculate the hash of the new data───────────────────────────
        from testUrl.algiers_views import LABS_DATA  
        data_bytes   = json.dumps(LABS_DATA, sort_keys=True).encode()
        version_hash = hashlib.sha256(data_bytes).hexdigest()[:16]

        # ── 3. Update or create the record───────────────────────────────────
        obj, created = DataVersion.objects.update_or_create(
            key='quebec_donneurs',
            defaults={
                'last_updated': timezone.now(),
                'version_hash': version_hash,
            }
        )

        self.stdout.write(
            self.style.SUCCESS(
                f" Version updated → {obj.last_updated} [{version_hash}]"
            )
        )