# centres/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from .models import CentreDon
from .serializers import CentreDonSerializer

class CentreDonListView(APIView):
    """
    GET /centres/api/centres/
    ?min_lat=45.4&max_lat=46.9&min_lng=-74.0&max_lng=-71.0
    &type=fixe,mobile
    &statut=actif
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        queryset = CentreDon.objects.all()

        # ── Filtre par bounds ─────────────────────────────────────────────────
        try:
            min_lat = request.query_params.get('min_lat')
            max_lat = request.query_params.get('max_lat')
            min_lng = request.query_params.get('min_lng')
            max_lng = request.query_params.get('max_lng')

            if all([min_lat, max_lat, min_lng, max_lng]):
                queryset = queryset.filter(
                    latitude__gte  = float(min_lat),
                    latitude__lte  = float(max_lat),
                    longitude__gte = float(min_lng),
                    longitude__lte = float(max_lng),
                )
        except ValueError:
            return Response(
                {"error": "Coordonnées invalides"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # ── Filtre par type ───────────────────────────────────────────────────
        types = request.query_params.get('type')
        if types:
            queryset = queryset.filter(type__in=types.split(','))

        # ── Filtre par statut (défaut: actif) ─────────────────────────────────
        statut = request.query_params.get('statut', 'actif')
        queryset = queryset.filter(statut=statut)

        serializer = CentreDonSerializer(queryset, many=True)
        return Response(serializer.data)


class CentreDonDetailView(APIView):
    """
    GET /centres/api/centres/<id>/
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        try:
            centre = CentreDon.objects.get(pk=pk)
        except CentreDon.DoesNotExist:
            return Response(
                {"error": "Centre introuvable"},
                status=status.HTTP_404_NOT_FOUND
            )
        return Response(CentreDonSerializer(centre).data)