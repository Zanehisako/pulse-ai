# centres/management/commands/generate_centres.py

import random
from django.core.management.base import BaseCommand
from centres.models import CentreDon

CITIES = [
    # (ville, region, latitude, longitude)
    ("Montréal",         "Montréal / Montérégie",            45.5017, -73.5673),
    ("Longueuil",        "Montréal / Montérégie",            45.5319, -73.5187),
    ("Laval",            "Montréal / Montérégie",            45.6066, -73.7124),
    ("Brossard",         "Montréal / Montérégie",            45.4619, -73.4747),
    ("Saint-Jean",       "Montréal / Montérégie",            45.3068, -73.2638),
    ("Sherbrooke",       "Estrie / Cantons-de-l'Est",        45.4042, -71.8929),
    ("Granby",           "Estrie / Cantons-de-l'Est",        45.4000, -72.7333),
    ("Drummondville",    "Centre-du-Québec",                 45.8833, -72.4833),
    ("Victoriaville",    "Centre-du-Québec",                 46.0500, -71.9667),
    ("Lévis",            "Chaudière-Appalaches",             46.8032, -71.1762),
    ("Saint-Georges",    "Chaudière-Appalaches",             46.1167, -70.6667),
    ("Rivière-du-Loup",  "Bas-Saint-Laurent",                47.8333, -69.5333),
    ("Rimouski",         "Bas-Saint-Laurent",                48.4500, -68.5333),
    ("Matane",           "Gaspésie Nord",                    48.8500, -67.5333),
    ("Gaspé",            "Gaspésie Nord",                    48.8333, -64.4833),
    ("Québec",           "Québec (Ville)",                   46.8139, -71.2080),
    ("Sainte-Foy",       "Québec (Ville)",                   46.7825, -71.2994),
    ("Beauport",         "Québec (Ville)",                   46.8833, -71.1833),
    ("Saint-Jérôme",     "Laurentides",                      45.7745, -74.0032),
    ("Mont-Tremblant",   "Laurentides",                      46.1185, -74.5962),
    ("Joliette",         "Lanaudière",                       46.0167, -73.4500),
    ("Terrebonne",       "Lanaudière",                       45.7000, -73.6333),
    ("Trois-Rivières",   "Mauricie",                         46.3432, -72.5515),
    ("Shawinigan",       "Mauricie",                         46.5500, -72.7500),
    ("Gatineau",         "Outaouais Nord",                   45.4765, -75.7013),
    ("Rouyn-Noranda",    "Abitibi-Témiscamingue",            48.2395, -79.0147),
    ("Val-d'Or",         "Abitibi-Témiscamingue",            48.1000, -77.7833),
    ("La Tuque",         "Haute-Mauricie",                   47.4333, -72.7833),
    ("Saguenay",         "Saguenay–Lac-Saint-Jean Ouest",    48.4220, -71.0537),
    ("Alma",             "Saguenay–Lac-Saint-Jean Est",      48.5500, -71.6500),
    ("Roberval",         "Saguenay–Lac-Saint-Jean Ouest",    48.5167, -72.2333),
    ("Baie-Comeau",      "Côte-Nord Ouest",                  49.2167, -68.1500),
    ("Sept-Îles",        "Côte-Nord Centre",                 50.2167, -66.3833),
    ("Havre-Saint-Pierre","Côte-Nord Est",                   50.2333, -63.6000),
    ("Chibougamau",      "Nord-du-Québec Centre",            49.9167, -74.3667),
    ("Matagami",         "Nord-du-Québec Ouest",             49.7500, -77.6333),
]

TYPES        = ['fixe', 'mobile', 'hopital', 'clinique']
TYPES_WEIGHT = [0.35,   0.25,     0.25,      0.15]

STATUTS        = ['actif', 'actif', 'actif', 'ferme_temporaire', 'en_maintenance']

EQUIPEMENTS_POOL = [
    'don_sang_total', 'don_plasma', 'don_plaquettes',
    'aphérèse', 'don_globules_rouges',
]

HORAIRES_TEMPLATE = {
    'fixe': {
        'lun': '07:30-19:00', 'mar': '07:30-19:00',
        'mer': '07:30-19:00', 'jeu': '07:30-19:00',
        'ven': '07:30-17:00', 'sam': '08:00-16:00',
        'dim': 'fermé',
    },
    'mobile': {
        'lun': 'fermé',       'mar': '09:00-17:00',
        'mer': '09:00-17:00', 'jeu': '09:00-17:00',
        'ven': '09:00-17:00', 'sam': '09:00-14:00',
        'dim': 'fermé',
    },
    'hopital': {
        'lun': '08:00-16:00', 'mar': '08:00-16:00',
        'mer': '08:00-16:00', 'jeu': '08:00-16:00',
        'ven': '08:00-14:00', 'sam': 'fermé',
        'dim': 'fermé',
    },
    'clinique': {
        'lun': '09:00-17:00', 'mar': '09:00-17:00',
        'mer': '09:00-17:00', 'jeu': '09:00-17:00',
        'ven': '09:00-16:00', 'sam': '09:00-13:00',
        'dim': 'fermé',
    },
}

CAPACITE = {
    'fixe':     (80, 150),
    'mobile':   (30,  60),
    'hopital':  (50, 100),
    'clinique': (20,  50),
}

NOM_PREFIXES = {
    'fixe':     'Héma-Québec —',
    'mobile':   'Unité Mobile —',
    'hopital':  'Centre Hospitalier —',
    'clinique': 'Clinique Don de Sang —',
}


class Command(BaseCommand):
    help = 'Generate 100 realistic blood donation centres in Québec'

    def add_arguments(self, parser):
        parser.add_argument('--count',  type=int, default=100)
        parser.add_argument('--clear',  action='store_true')

    def handle(self, *args, **options):
        if options['clear']:
            deleted = CentreDon.objects.count()
            CentreDon.objects.all().delete()
            self.stdout.write(f'🗑️  Deleted {deleted} centres')

        count   = options['count']
        created = 0
        errors  = 0

        # ── distribuer les centres sur les villes ─────────────────────────────
        # plusieurs centres par ville pour atteindre 100
        entries = []
        for i in range(count):
            city_data = CITIES[i % len(CITIES)]
            ville, region, base_lat, base_lng = city_data

            type_centre = random.choices(TYPES, weights=TYPES_WEIGHT)[0]

            # ← légère variation GPS pour éviter superposition
            lat = round(base_lat + random.uniform(-0.05, 0.05), 6)
            lng = round(base_lng + random.uniform(-0.05, 0.05), 6)

            index      = i + 1
            code       = f"QC-{ville[:3].upper()}-{index:03d}"
            nom        = f"{NOM_PREFIXES[type_centre]} {ville} {index}"
            capacite   = random.randint(*CAPACITE[type_centre])
            statut     = random.choice(STATUTS)
            equipements = random.sample(
                EQUIPEMENTS_POOL,
                k=random.randint(1, len(EQUIPEMENTS_POOL))
            )
            horaires   = HORAIRES_TEMPLATE[type_centre]
            annee      = random.randint(1990, 2020)
            mois       = random.randint(1, 12)
            jour       = random.randint(1, 28)

            entries.append(CentreDon(
                code_centre          = code,
                nom                  = nom,
                type                 = type_centre,
                adresse              = f"{random.randint(1, 9999)} Rue Principale",
                code_postal          = f"G{random.randint(1,9)}X {random.randint(1,9)}X{random.randint(1,9)}",
                ville                = ville,
                region               = region,
                pays                 = "Canada",
                latitude             = lat,
                longitude            = lng,
                telephone            = f"4{random.randint(10,99)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
                email                = f"{ville.lower().replace(' ', '').replace('-', '')}{index}@hema-quebec.qc.ca",
                capacite_journaliere = capacite,
                horaires             = horaires,
                statut               = statut,
                equipements          = equipements,
                date_ouverture       = f"{annee}-{mois:02d}-{jour:02d}",
            ))

        try:
            CentreDon.objects.bulk_create(entries, ignore_conflicts=True)
            created = len(entries)
        except Exception as e:
            self.stderr.write(f'❌ Error: {e}')
            errors += 1

        self.stdout.write('─' * 40)
        self.stdout.write(f'✅ Created : {created}')
        self.stdout.write(f'❌ Errors  : {errors}')
        self.stdout.write(f'📊 Total in DB: {CentreDon.objects.count()}')
        self.stdout.write('✅ Done!')