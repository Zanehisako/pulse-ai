# testUrl/management/commands/import_donneurs.py

import json
import os
import hashlib
from django.core.management.base import BaseCommand
from django.utils import timezone
from testUrl.models import Donneur, DataVersion

JSON_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../data/laboratoires.json'
)

class Command(BaseCommand):
    help = 'Import donneurs from laboratoires.json to database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before import',
        )

    def handle(self, *args, **options):
        # ── 1. قراءة الـ JSON ─────────────────────────────────────────────────
        self.stdout.write(f'📂 Reading: {JSON_PATH}')

        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.stdout.write(f'✅ Found {len(data)} donneurs')

        # ── 2. حذف القديم إذا طلب ────────────────────────────────────────────
        if options['clear']:
            count = Donneur.objects.count()
            Donneur.objects.all().delete()
            self.stdout.write(f'🗑️  Cleared {count} existing records')

        # ── 3. استيراد البيانات ───────────────────────────────────────────────
        created = 0
        updated = 0
        errors  = 0

        for item in data:
            try:
                obj, was_created = Donneur.objects.update_or_create(
                    id=item['id'],
                    defaults={
                        'nom':            item.get('nom', ''),
                        'adresse':        item.get('adresse', ''),
                        'telephone':      item.get('telephone', ''),
                        'latitude':       item.get('latitude', 0.0),
                        'longitude':      item.get('longitude', 0.0),
                        'groupe_sanguin': item.get('groupe_sanguin', ''),
                        'age':            item.get('age', 0),
                        'poids':          item.get('poids', 0.0),
                        'nb_dons':        item.get('nb_dons', 0),
                        'dernier_don':    item.get('dernier_don'),
                        'disponible':     item.get('disponible', ''),
                        'region':         item.get('region', ''),
                        'rating':         item.get('rating', 0.0),
                        'image':          item.get('image', ''),
                    }
                )

                if was_created:
                    created += 1
                else:
                    updated += 1

            except Exception as e:
                errors += 1
                self.stderr.write(f'❌ Error on id={item.get("id")}: {e}')

        # ── 4. تحديث DataVersion ──────────────────────────────────────────────
        raw = json.dumps(data, sort_keys=True).encode('utf-8')
        version_hash = hashlib.md5(raw).hexdigest()

        DataVersion.objects.update_or_create(
            key='quebec_donneurs',
            defaults={
                'last_updated': timezone.now(),
                'version_hash': version_hash,
            }
        )

        # ── 5. تقرير نهائي ────────────────────────────────────────────────────
        self.stdout.write('─' * 40)
        self.stdout.write(f'✅ Created : {created}')
        self.stdout.write(f'🔄 Updated : {updated}')
        self.stdout.write(f'❌ Errors  : {errors}')
        self.stdout.write(f'🏷️  Version : {version_hash}')
        self.stdout.write('✅ Done!')