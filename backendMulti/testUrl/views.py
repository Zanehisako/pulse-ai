import json
import os
from collections import defaultdict
from drf_spectacular.utils import OpenApiParameter, OpenApiTypes, extend_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import EmptySerializer, LaboratoriesResponseSerializer, LaboratoryErrorResponseSerializer

JSON_PATH = os.path.join(os.path.dirname(__file__), 'data', 'laboratoires.json')
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    LABS_DATA = json.load(f)


def grid_size_for_zoom(zoom: int) -> float:

    if zoom <= 10:
        return 0.05
    elif zoom <= 12:
        return 0.02
    elif zoom <= 14:
        return 0.01
    elif zoom <= 16:
        return 0.005
    else:
        return 0.002


class LaboratoiresView(APIView):
    """

    GET /testUrl/api/laboratoires/?bbox=minLng,minLat,maxLng,maxLat&zoom=13
    """

    def get(self, request):
        try:
            bbox = request.query_params.get('bbox')
            zoom = int(request.query_params.get('zoom', 12))

            if not bbox:
                return Response(
                    {"error": "bbox is required: minLng,minLat,maxLng,maxLat"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            min_lng, min_lat, max_lng, max_lat = map(float, bbox.split(','))

      
            filtered = [
                lab for lab in LABS_DATA
                if min_lat <= lab["latitude"]  <= max_lat
                and min_lng <= lab["longitude"] <= max_lng
            ]

      
            if zoom >= 17:
                return Response({
                    "count":   len(filtered),
                    "results": [{**lab, "type": "point"} for lab in filtered],
                }, status=status.HTTP_200_OK)

            # clustering بالـ grid
            grid_size = grid_size_for_zoom(zoom)
            clusters  = defaultdict(list)

            for lab in filtered:
                key = (
                    round(lab["latitude"]  / grid_size),
                    round(lab["longitude"] / grid_size),
                )
                clusters[key].append(lab)

            results = []
            for labs in clusters.values():
                if len(labs) == 1:
                    results.append({**labs[0], "type": "point"})
                else:
                    avg_lat = sum(l["latitude"]  for l in labs) / len(labs)
                    avg_lng = sum(l["longitude"] for l in labs) / len(labs)
                    results.append({
                        "type":  "cluster",
                        "lat":   avg_lat,
                        "lng":   avg_lng,
                        "count": len(labs),
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
class TiledLaboratoiresView(APIView):
    """
    Tiled API View with spatial filtering and optional binary format.

    GET /testUrl/api/laboratoires/tiled/
        ?north=62.6&south=44.99&east=-57.1&west=-79.76&zoom=12&format=json

    Query params:
        north, south, east, west  - viewport bounds (float, required for filtering)
        zoom                      - current zoom level (int, optional)
        format                    - 'json' | 'binary' (default: 'json')
    """

    _cached_data = None  # Simple in-memory cache

    @classmethod
    def _load_data(cls):
        if cls._cached_data is not None:
            return cls._cached_data
        json_file_path = os.path.join(
            os.path.dirname(__file__), 'data', 'laboratoires.json'
        )
        with open(json_file_path, 'r', encoding='utf-8') as file:
            cls._cached_data = json.load(file)
        return cls._cached_data

    def get(self, request):
        try:
            data = self._load_data()
        except FileNotFoundError:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        total_count = len(data)

        # ─── Parse query params ────────────────────────────────────────────────
        north = self._parse_float(request.GET.get('north'),  62.60)
        south = self._parse_float(request.GET.get('south'),  44.99)
        east  = self._parse_float(request.GET.get('east'),  -57.10)
        west  = self._parse_float(request.GET.get('west'),  -79.76)
        zoom  = self._parse_int(request.GET.get('zoom'), 12)
        response_format = request.GET.get('format', 'json')

        # ─── Spatial filtering ─────────────────────────────────────────────────
        filtered = [
            lab for lab in data
            if (south <= lab.get('latitude',  0) <= north
            and west  <= lab.get('longitude', 0) <= east)
        ]

        # ─── Zoom-based sampling ───────────────────────────────────────────────
        if zoom <= 6 and len(filtered) > 500:
            step = max(1, len(filtered) // 500)
            filtered = filtered[::step]
        elif zoom <= 8 and len(filtered) > 1000:
            step = max(1, len(filtered) // 1000)
            filtered = filtered[::step]
        elif zoom <= 10 and len(filtered) > 3000:
            step = max(1, len(filtered) // 3000)
            filtered = filtered[::step]
        elif zoom <= 12 and len(filtered) > 8000:
            step = max(1, len(filtered) // 8000)
            filtered = filtered[::step]

        # ─── Return response ───────────────────────────────────────────────────
        if response_format == 'binary':
            return self._binary_response(filtered, total_count)
        else:
            return self._json_response(filtered, total_count)

    def _json_response(self, markers, total_count):
        """
        Compact JSON response.
        ~40-50 bytes per marker vs ~200+ bytes with full lab data.
        """
        results = []
        for lab in markers:
            results.append({
                'id':  lab.get('id', 0),
                'lat': round(lab.get('latitude',  0), 6),
                'lng': round(lab.get('longitude', 0), 6),
                'st':  1 if lab.get('statut') == 'ouvert' else 0,
                'cap': lab.get('capacite_tests_jour', 0) or 0,
            })

        return Response({
            'markers':  results,
            'total':    total_count,
            'filtered': len(results),
        }, status=status.HTTP_200_OK)

    def _binary_response(self, markers, total_count):
        """
        Lightweight binary format — 15 bytes per marker.

        Header (8 bytes):
            total_count:  uint32 LE
            marker_count: uint32 LE

        Per marker (15 bytes):
            id:       int32   LE  (4 bytes)
            lat:      float32 LE  (4 bytes)
            lng:      float32 LE  (4 bytes)
            status:   uint8       (1 byte)
            capacity: uint16  LE  (2 bytes)

        1000 markers = 8 + (1000 × 15) = 15,008 bytes (~15 KB)
        vs JSON:      1000 × ~50 bytes = ~50 KB
        """
        marker_count = len(markers)
        buf = bytearray(8 + marker_count * 15)

        # Header
        struct.pack_into('<II', buf, 0, total_count, marker_count)

        # Markers
        offset = 8
        for lab in markers:
            marker_id   = lab.get('id', 0)
            lat         = round(lab.get('latitude',  0), 6)
            lng         = round(lab.get('longitude', 0), 6)
            status_byte = 1 if lab.get('statut') == 'ouvert' else 0
            capacity    = min(lab.get('capacite_tests_jour', 0) or 0, 65535)

            struct.pack_into('<IffBH', buf, offset,
                             marker_id, lat, lng, status_byte, capacity)
            offset += 15

        response = HttpResponse(
            bytes(buf),
            content_type='application/octet-stream'
        )
        response['Content-Length'] = len(buf)
        response['Cache-Control']  = 'public, max-age=30'
        return response

    @staticmethod
    def _parse_float(value, default):
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_int(value, default):
        try:
            return int(value) if value is not None else default
        except (TypeError, ValueError):
            return default