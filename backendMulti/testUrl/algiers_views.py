import json
import gzip
import os
from collections import defaultdict
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from rest_framework.renderers import BaseRenderer
from rest_framework.permissions import AllowAny
from testUrl.models import DataVersion, Donneur

def _load_from_db():
    return list(
        Donneur.objects.all().values(
            'id', 'nom', 'adresse', 'telephone',
            'latitude', 'longitude',
            'groupe_sanguin', 'age', 'poids', 'nb_dons',
            'dernier_don', 'disponible', 'region',
            'rating', 'image',
        )
    )

LABS_DATA = None

def _get_labs_data():
    global LABS_DATA
    if LABS_DATA is None:
        LABS_DATA = _load_from_db()
    return LABS_DATA

# ─── 6×6 Québec Province Grid ─────────────────────────────────────────────────
QUEBEC_MIN_LAT = 44.99
QUEBEC_MAX_LAT = 62.60
QUEBEC_MIN_LNG = -79.76
QUEBEC_MAX_LNG = -57.10
ROWS     = 6
COLS     = 6
CELL_LAT = (QUEBEC_MAX_LAT - QUEBEC_MIN_LAT) / ROWS   # ~2.935°
CELL_LNG = (QUEBEC_MAX_LNG - QUEBEC_MIN_LNG) / COLS   # ~3.777°

CELL_NAMES = {
    (0, 0): "Montréal / Montérégie",
    (0, 1): "Estrie / Cantons-de-l'Est",
    (0, 2): "Centre-du-Québec",
    (0, 3): "Chaudière-Appalaches",
    (0, 4): "Bas-Saint-Laurent",
    (0, 5): "Gaspésie Sud",
    (1, 0): "Laurentides",
    (1, 1): "Lanaudière",
    (1, 2): "Mauricie",
    (1, 3): "Québec (Ville)",
    (1, 4): "Bas-Saint-Laurent Nord",
    (1, 5): "Gaspésie Nord",
    (2, 0): "Outaouais Nord",
    (2, 1): "Abitibi-Témiscamingue",
    (2, 2): "Haute-Mauricie",
    (2, 3): "Saguenay–Lac-Saint-Jean Ouest",
    (2, 4): "Saguenay–Lac-Saint-Jean Est",
    (2, 5): "Côte-Nord Ouest",
    (3, 0): "Abitibi Nord",
    (3, 1): "Nord-du-Québec Ouest",
    (3, 2): "Nord-du-Québec Centre",
    (3, 3): "Nord-du-Québec Est",
    (3, 4): "Côte-Nord Centre",
    (3, 5): "Côte-Nord Est",
    (4, 0): "Eeyou Istchee Ouest",
    (4, 1): "Eeyou Istchee Centre",
    (4, 2): "Eeyou Istchee Est",
    (4, 3): "Nunavik Ouest",
    (4, 4): "Nunavik Centre",
    (4, 5): "Nunavik Est",
    (5, 0): "Grand Nord Ouest",
    (5, 1): "Grand Nord Centre-Ouest",
    (5, 2): "Grand Nord Centre",
    (5, 3): "Grand Nord Centre-Est",
    (5, 4): "Grand Nord Est",
    (5, 5): "Pointe Nord",
}


DONNEUR_FIELDS = [
    'id', 'nom', 'adresse', 'telephone',
    'latitude', 'longitude',
    'groupe_sanguin', 'age', 'poids', 'nb_dons',
    'dernier_don', 'disponible', 'region',
    'rating', 'image',
]


def get_cell(lat: float, lng: float):
    if not (QUEBEC_MIN_LAT <= lat <= QUEBEC_MAX_LAT and
            QUEBEC_MIN_LNG <= lng <= QUEBEC_MAX_LNG):
        return None
    row = min(int((lat - QUEBEC_MIN_LAT) / CELL_LAT), ROWS - 1)
    col = min(int((lng - QUEBEC_MIN_LNG) / CELL_LNG), COLS - 1)
    return row, col


def slim(donneur: dict) -> dict:
    result = {k: donneur[k] for k in DONNEUR_FIELDS if k in donneur}
    if result.get('dernier_don'):
        result['dernier_don'] = str(result['dernier_don'])
    return result


def grid_size_for_zoom(zoom: int) -> float:
    if zoom <= 10: return 0.05
    if zoom <= 12: return 0.02
    if zoom <= 14: return 0.01
    if zoom <= 16: return 0.005
    return 0.002


class GzipRenderer(BaseRenderer):
    media_type   = 'application/gzip'
    format       = 'gz'
    charset      = None
    render_style = 'binary'

    def render(self, data, accepted_media_type=None, renderer_context=None):
        return data


# ─── Endpoint 1: Index 36 cellules ────────────────────────────────────────────
class AlgiersGridIndexView(APIView):
    """
    GET /testUrl/api/labs/algiers-index/
    Retourne les infos des 36 cellules avec le nombre de donneurs
    """
    renderer_classes = [GzipRenderer]

    def get(self, request):
        labs_data = _get_labs_data()
        cell_counts = {(r, c): 0 for r in range(ROWS) for c in range(COLS)}

        for d in labs_data:
            cell = get_cell(d['latitude'], d['longitude'])
            if cell:
                cell_counts[cell] += 1

        grids = []
        for row in range(ROWS):
            for col in range(COLS):
                grids.append({
                    "row":      row,
                    "col":      col,
                    "name":     CELL_NAMES.get((row, col), f"Zone {row}-{col}"),
                    "filename": f"qc_{row}_{col}.gz",
                    "count":    cell_counts[(row, col)],
                    "bounds": {
                        "min_lat": QUEBEC_MIN_LAT + row * CELL_LAT,
                        "max_lat": QUEBEC_MIN_LAT + (row + 1) * CELL_LAT,
                        "min_lng": QUEBEC_MIN_LNG + col * CELL_LNG,
                        "max_lng": QUEBEC_MIN_LNG + (col + 1) * CELL_LNG,
                    },
                })

        return Response({
            "province":       "Québec",
            "type":           "donneurs_de_sang",
            "grid":           f"{ROWS}x{COLS}",
            "cell_size_km":   f"~{CELL_LAT * 111:.1f}km x {CELL_LNG * 111:.1f}km",
            "total_donneurs": len(labs_data),
            "grids":          grids,
        })


# ─── Endpoint 2: Cellule compressée ───────────────────────────────────────────
class AlgiersGridCellView(APIView):
    """
    GET /testUrl/api/labs/algiers-cell/?row=1&col=3
    Retourne les donneurs de la cellule compressés en GZip
    """

    def get(self, request):
        try:
            row = int(request.query_params.get('row'))
            col = int(request.query_params.get('col'))
        except (TypeError, ValueError):
            return Response(
                {"error": f"row et col sont requis (0-{ROWS-1})"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not (0 <= row < ROWS and 0 <= col < COLS):
            return Response(
                {"error": f"row: 0-{ROWS-1}, col: 0-{COLS-1}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        min_lat = QUEBEC_MIN_LAT + row * CELL_LAT
        max_lat = QUEBEC_MIN_LAT + (row + 1) * CELL_LAT
        min_lng = QUEBEC_MIN_LNG + col * CELL_LNG
        max_lng = QUEBEC_MIN_LNG + (col + 1) * CELL_LNG

        donneurs = [
            slim(d) for d in _get_labs_data()
            if min_lat <= d['latitude'] < max_lat
            and min_lng <= d['longitude'] < max_lng
        ]

        json_bytes = json.dumps(donneurs, ensure_ascii=False).encode('utf-8')
        compressed = gzip.compress(json_bytes, compresslevel=9)

        response = HttpResponse(compressed, content_type='application/gzip')
        response['Content-Disposition'] = f'attachment; filename="qc_{row}_{col}.gz"'
        response['X-Cell-Name']         = CELL_NAMES.get((row, col), '')
        response['X-Donneur-Count']     = len(donneurs)
        response['X-Original-Size']     = len(json_bytes)
        response['X-Compressed-Size']   = len(compressed)
        return response


# ─── Endpoint 3: Résumé des clusters (offline) ────────────────────────────────
class AlgiersClusterSummaryView(APIView):
    """
    GET /testUrl/api/labs/algiers-clusters/
    Retourne un fichier léger avec le centre et le nombre de donneurs par cellule
    """
    renderer_classes = [GzipRenderer]

    def get(self, request):
        labs_data = _get_labs_data()
        clusters = []

        for row in range(ROWS):
            for col in range(COLS):
                min_lat = QUEBEC_MIN_LAT + row * CELL_LAT
                max_lat = QUEBEC_MIN_LAT + (row + 1) * CELL_LAT
                min_lng = QUEBEC_MIN_LNG + col * CELL_LNG
                max_lng = QUEBEC_MIN_LNG + (col + 1) * CELL_LNG

                count = sum(
                    1 for d in labs_data
                    if min_lat <= d['latitude'] < max_lat
                    and min_lng <= d['longitude'] < max_lng
                )

                if count > 0:
                    clusters.append({
                        'row':   row,
                        'col':   col,
                        'lat':   (min_lat + max_lat) / 2,
                        'lng':   (min_lng + max_lng) / 2,
                        'count': count,
                        'name':  CELL_NAMES.get((row, col), ''),
                    })

        json_bytes = json.dumps(clusters, ensure_ascii=False).encode('utf-8')
        compressed = gzip.compress(json_bytes, compresslevel=9)
        return HttpResponse(compressed, content_type='application/gzip')


# ─── Endpoint 4: Clustering dynamique (online) ────────────────────────────────
class LaboratoiresView(APIView):
    """
    GET /testUrl/api/laboratoires/?bbox=minLng,minLat,maxLng,maxLat&zoom=13
    Server-side clustering des donneurs de sang
    """

    def get(self, request):
        try:
            bbox = request.query_params.get('bbox')
            zoom = int(request.query_params.get('zoom', 12))

            if not bbox:
                return Response(
                    {"error": "bbox requis: minLng,minLat,maxLng,maxLat"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            min_lng, min_lat, max_lng, max_lat = map(float, bbox.split(','))

            # ─── Filtre dans bbox ──────────────────────────────────────────────
            filtered = [
                d for d in _get_labs_data()
                if min_lat <= d['latitude']  <= max_lat
                and min_lng <= d['longitude'] <= max_lng
            ]

            # ─── zoom très proche → points directs ────────────────────────────
            if zoom >= 17:
                return Response({
                    "count":   len(filtered),
                    "results": [{**slim(d), "type": "point"} for d in filtered],
                }, status=status.HTTP_200_OK)

            # ─── Clustering par grid ───────────────────────────────────────────
            grid_size = grid_size_for_zoom(zoom)
            buckets   = defaultdict(list)

            for d in filtered:
                key = (
                    round(d['latitude']  / grid_size),
                    round(d['longitude'] / grid_size),
                )
                buckets[key].append(d)

            results = []
            for group in buckets.values():
                if len(group) == 1:
                    results.append({**slim(group[0]), "type": "point"})
                else:
                    avg_lat = sum(d['latitude']  for d in group) / len(group)
                    avg_lng = sum(d['longitude'] for d in group) / len(group)

                    from collections import Counter
                    dominant = Counter(
                        d['groupe_sanguin'] for d in group
                    ).most_common(1)[0][0]

                    results.append({
                        "type":           "cluster",
                        "lat":            round(avg_lat, 6),
                        "lng":            round(avg_lng, 6),
                        "count":          len(group),
                        "groupe_sanguin": dominant,
                    })

            return Response({
                "count":   len(results),
                "results": results,
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ─── Endpoint 5: Labs by Bounds ───────────────────────────────────────────────
class QuebecLabsByBoundsView(APIView):
    """
    GET /testUrl/api/labs/quebec-bounds/
        ?min_lat=45.4&max_lat=46.9&min_lng=-74.0&max_lng=-71.0&zoom=15
    Individual labs or clusters are referred according to Zoom.
    """

    def get(self, request):
        try:
            min_lat = float(request.query_params.get('min_lat'))
            max_lat = float(request.query_params.get('max_lat'))
            min_lng = float(request.query_params.get('min_lng'))
            max_lng = float(request.query_params.get('max_lng'))
            zoom    = int(request.query_params.get('zoom', 13))
        except (TypeError, ValueError):
            return Response(
                {"error": "min_lat, max_lat, min_lng, max_lng, zoom are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        filtered = [
            lab for lab in _get_labs_data()
            if min_lat <= lab['latitude'] <= max_lat
            and min_lng <= lab['longitude'] <= max_lng
        ]

        if zoom <= 11:
            grid_size = 2.0
        elif zoom <= 12:
            grid_size = 1.0
        elif zoom <= 13:
            grid_size = 0.5
        elif zoom <= 14:
            grid_size = 0.2
        elif zoom <= 15:
            grid_size = 0.1
        elif zoom <= 16:
            grid_size = 0.05
        else:
            result = [{'type': 'point', **lab} for lab in filtered[:200]]
            return Response(result)

        clusters = {}
        for lab in filtered:
            row = round(lab['latitude']  / grid_size)
            col = round(lab['longitude'] / grid_size)
            key = f"{row}_{col}"
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(lab)

        result = []
        for group in clusters.values():
            if len(group) == 1:
                result.append({'type': 'point', **group[0]})
            else:
                avg_lat = sum(l['latitude']  for l in group) / len(group)
                avg_lng = sum(l['longitude'] for l in group) / len(group)
                result.append({
                    'type':  'cluster',
                    'lat':   round(avg_lat, 6),
                    'lng':   round(avg_lng, 6),
                    'count': len(group),
                })

        return Response(result)


class DataVersionView(APIView):
    permission_classes = [AllowAny]
    """
    GET /testUrl/api/labs/version/
    The last data update date is returned.
    """
    def get(self, request):
        try:
            v = DataVersion.objects.get(key='quebec_donneurs')
            return Response({
                "key":          "quebec_donneurs",
                "last_updated": v.last_updated.isoformat(),
                "version_hash": v.version_hash,
            })
        except DataVersion.DoesNotExist:
            return Response(
                {"last_updated": None, "version_hash": None}
            )


class ReloadDataView(APIView):
    """
    POST /testUrl/api/labs/reload/
    Recharge LABS_DATA depuis la DB sans redémarrer le serveur
    """
    def post(self, request):
        global LABS_DATA
        LABS_DATA = _load_from_db()
        return Response({
            "status": "reloaded",
            "total":  len(LABS_DATA),
        })