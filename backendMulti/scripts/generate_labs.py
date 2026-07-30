import json
import random
import os

def generate_donneurs(count=500000):
    """Generate random blood donors across all of Québec Province"""

    donneurs = []


    QUEBEC_MIN_LAT = 44.99
    QUEBEC_MAX_LAT = 62.60
    QUEBEC_MIN_LNG = -79.76
    QUEBEC_MAX_LNG = -57.10


    REGIONS = [
        # (nom, center_lat, center_lng, weight%, spread_lat, spread_lng)

       # ── The densely populated South (90% of the population) ─────────────────────────────────
        ("Montréal",              45.5017, -73.5673, 25.0, 0.30, 0.40),
        ("Laval",                 45.6066, -73.7124,  8.0, 0.15, 0.20),
        ("Longueuil",             45.5312, -73.5185,  6.0, 0.15, 0.20),
        ("Québec (Ville)",        46.8139, -71.2080, 12.0, 0.20, 0.25),
        ("Lévis",                 46.6974, -71.1795,  4.0, 0.12, 0.15),
        ("Sherbrooke",            45.4042, -71.8929,  4.0, 0.12, 0.15),
        ("Trois-Rivières",        46.3432, -72.5432,  4.0, 0.12, 0.15),
        ("Saguenay",              48.4281, -71.0537,  3.0, 0.15, 0.20),
        ("Gatineau",              45.4765, -75.7013,  5.0, 0.15, 0.20),
        ("Saint-Jean-sur-Richelieu", 45.3073, -73.2638, 2.0, 0.10, 0.12),
        ("Granby",                45.4009, -72.7331,  2.0, 0.10, 0.12),
        ("Saint-Hyacinthe",       45.6234, -72.9573,  2.0, 0.10, 0.12),
        ("Repentigny",            45.7416, -73.4600,  2.0, 0.10, 0.12),
        ("Drummondville",         45.8836, -72.4827,  2.0, 0.10, 0.12),
        ("Saint-Jérôme",          45.7742, -74.0034,  2.0, 0.10, 0.12),
        ("Rimouski",              48.4489, -68.5317,  1.5, 0.12, 0.15),
        ("Rouyn-Noranda",         48.2395, -79.0188,  1.5, 0.12, 0.15),
        ("Val-d'Or",              48.0975, -77.7930,  1.0, 0.10, 0.12),
        ("Sept-Îles",             50.2037, -66.3806,  1.0, 0.10, 0.12),
        ("Baie-Comeau",           49.2210, -68.1507,  1.0, 0.10, 0.12),

      # ── The Remote North (10% of the population) ──────────────────────────────────
        ("Chibougamau",           49.9166, -74.3694,  0.8, 0.20, 0.25),
        ("Matagami",              49.7515, -77.6326,  0.3, 0.15, 0.20),
        ("Radisson",              53.7918, -77.6143,  0.2, 0.20, 0.25),
        ("Kuujjuaq",              58.1079, -68.4013,  0.3, 0.30, 0.40),
        ("Puvirnituq",            60.0334, -77.2761,  0.2, 0.25, 0.30),
        ("Chisasibi",             53.7957, -78.8987,  0.2, 0.20, 0.25),
    ]

    # ─── donneurs ────────────────────────────────────────────────────────
    prenoms_m = [
        "Jean", "Pierre", "Michel", "François", "André", "Philippe",
        "Marc", "Luc", "Paul", "Nicolas", "Alexandre", "David",
        "Thomas", "Antoine", "Guillaume", "Mathieu", "Julien", "Simon",
    ]
    prenoms_f = [
        "Marie", "Sophie", "Julie", "Isabelle", "Nathalie", "Sylvie",
        "Catherine", "Monique", "Céline", "Émilie", "Audrey", "Sarah",
        "Amélie", "Camille", "Laura", "Vanessa", "Stéphanie", "Anne",
    ]
    noms_famille = [
        "Tremblay", "Gagnon", "Roy", "Côté", "Bouchard", "Gauthier",
        "Morin", "Lavoie", "Fortin", "Gagné", "Ouellet", "Pelletier",
        "Bélanger", "Lévesque", "Bergeron", "Simard", "Landry", "Leblanc",
        "Boucher", "Thibault", "Lacroix", "Girard", "Caron", "Grenier",
    ]

    groupes_sanguins = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    poids_groupes    = [34, 6, 9, 2, 3, 1, 38, 7]  # % réels au Québec

    streets = [
        "Rue Principale",          "Boulevard Saint-Laurent",
        "Avenue du Parc",          "Rue des Érables",
        "Chemin du Lac",           "Boulevard des Laurentides",
        "Rue de la Montagne",      "Avenue Royale",
        "Boulevard Taschereau",    "Rue Notre-Dame",
        "Avenue De la Brunante",   "Rue des Peupliers",
        "Boulevard Sainte-Anne",   "Chemin des Quatre-Bourgeois",
        "Rue du Commissaire",      "Avenue du Pont",
    ]

   
    total_weight  = sum(r[3] for r in REGIONS)
    region_counts = []
    assigned      = 0

    for idx, region in enumerate(REGIONS):
        if idx == len(REGIONS) - 1:
            n = count - assigned
        else:
            n = round(count * region[3] / total_weight)
        region_counts.append(n)
        assigned += n

    # ───  donneurs ────────────────────────────────────────────────────────
    donneur_id = 1

    for region, n in zip(REGIONS, region_counts):
        region_name, center_lat, center_lng, _, spread_lat, spread_lng = region

        for _ in range(n):
           
            lat = round(center_lat + random.uniform(-spread_lat, spread_lat), 6)
            lng = round(center_lng + random.uniform(-spread_lng, spread_lng), 6)

           
            lat = max(QUEBEC_MIN_LAT, min(QUEBEC_MAX_LAT, lat))
            lng = max(QUEBEC_MIN_LNG, min(QUEBEC_MAX_LNG, lng))

            # ─── donneur ─────────────────────────────────────────────
            is_male  = random.random() > 0.5
            prenom   = random.choice(prenoms_m if is_male else prenoms_f)
            nom      = random.choice(noms_famille)
            nom_complet = f"{prenom} {nom}"

            street_num = random.randint(1, 9999)
            street     = random.choice(streets)
            address    = f"{street_num} {street}, {region_name}, Québec"

            area_codes = ['418', '438', '450', '514', '819', '873', '579', '367']
            area_code  = random.choice(area_codes)
            phone      = f"{area_code}-{random.randint(100,999)}-{random.randint(1000,9999)}"

            groupe_sanguin = random.choices(groupes_sanguins, weights=poids_groupes)[0]

            age      = random.randint(18, 65)
            poids    = round(random.uniform(50.0, 110.0), 1)
            nb_dons  = random.randint(0, 50)

            # dernier don
            jours_depuis = random.randint(56, 1825)  # 56 jours min entre dons
            from datetime import date, timedelta
            dernier_don = (date.today() - timedelta(days=jours_depuis)).isoformat()

            disponible = random.choices(
                ['disponible', 'indisponible', 'en_attente'],
                weights=[70, 20, 10]
            )[0]

            rating = round(random.uniform(3.5, 5.0), 1)

            colors = ['FF0000', '0000FF', '00FF00', 'FFA500', 'FF00FF',
                      '00FFFF', 'FF6B6B', '800080', '006994', '228B22']
            color  = random.choice(colors)
            image  = f"https://via.placeholder.com/400x300/{color}/FFFFFF?text=Don+{donneur_id}"

            donneurs.append({
                "id":             donneur_id,
                "nom":            nom_complet,
                "adresse":        address,
                "telephone":      phone,
                "latitude":       lat,
                "longitude":      lng,
                "groupe_sanguin": groupe_sanguin,
                "age":            age,
                "poids":          poids,
                "nb_dons":        nb_dons,
                "dernier_don":    dernier_don,
                "disponible":     disponible,
                "region":         region_name,
                "rating":         rating,
                "image":          image,
            })

            donneur_id += 1

    random.shuffle(donneurs)
    for idx, d in enumerate(donneurs, start=1):
        d["id"] = idx

    return donneurs


# ─── Paths ────────────────────────────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_file  = os.path.join(project_root, 'testUrl', 'data', 'laboratoires.json')

# ─── Generate ─────────────────────────────────────────────────────────────────
print("🔄 Generating 500,000 donneurs de sang across Québec Province...")
donneurs = generate_donneurs(3000)

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(donneurs, f, ensure_ascii=False)

print(f"✅ Done! {len(donneurs)} donneurs generated")
print(f"📂 File: {output_file}")
print(f"📊 Size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

# ─── Stats ────────────────────────────────────────────────────────────────────
print("\n📍 Distribution par région:")
from collections import Counter
region_dist = Counter(d['region'] for d in donneurs)
print(f"{'Région':<30} {'Donneurs':>9}  {'Bar'}")
print("-" * 60)
for region, cnt in region_dist.most_common():
    bar = '█' * (cnt // 5000)
    print(f"  {region:<28} {cnt:>9}  {bar}")

print("\n🩸 Distribution groupes sanguins:")
blood_dist = Counter(d['groupe_sanguin'] for d in donneurs)
for groupe, cnt in sorted(blood_dist.items()):
    pct = cnt / len(donneurs) * 100
    print(f"  {groupe:<4} {cnt:>8}  ({pct:.1f}%)")

print("\n✅ Coverage: toute la province de Québec")
print(f"   lat: 44.99 → 62.60")
print(f"   lng: -79.76 → -57.10")