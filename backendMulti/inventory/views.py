from __future__ import annotations

import logging
from datetime import date as date_type

from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from inventory.apps import _run_prediction_chain
from inventory.models import Donor, PredictionResult
from inventory.serializers import (
    BloodSupplySnapshotSerializer,
    DonorSerializer,
    PredictionResultSerializer,
)
from inventory.tasks import configured_donor_score_result_model

logger = logging.getLogger(__name__)


class DonorListView(APIView):
    ORDERING_ALIAS = {
        "donation_count": "frequency_365",
    }
    ALLOWED_ORDERING_FIELDS = {
        "donor_id",
        "event_timestamp",
        "city_id",
        "availability",
        "blood_group",
        "recency_days",
        "frequency_365",
        "days_until_eligible",
        "updated_at",
    }
    DEFAULT_ORDERING = ("-frequency_365", "-updated_at", "donor_id")

    def _normalized_ordering(self, raw_ordering: str | None) -> list[str]:
        if not raw_ordering:
            return list(self.DEFAULT_ORDERING)

        normalized: list[str] = []
        for token in raw_ordering.split(","):
            token = token.strip()
            if not token:
                continue

            descending = token.startswith("-")
            requested = token[1:] if descending else token
            actual = self.ORDERING_ALIAS.get(requested, requested)

            if actual not in self.ALLOWED_ORDERING_FIELDS:
                continue

            normalized.append(f"-{actual}" if descending else actual)

        return normalized or list(self.DEFAULT_ORDERING)

    def get(self, request):
        ordering = self._normalized_ordering(request.query_params.get("ordering"))
        queryset = Donor.objects.all().order_by(*ordering)
        serializer = DonorSerializer(queryset, many=True)

        return Response(
            {
                "count": len(serializer.data),
                "results": serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class PredictionResultListView(APIView):
    def get(self, request):
        model_name = request.query_params.get("model")
        alert_only = str(request.query_params.get("alert_only", "false")).lower() in {
            "1",
            "true",
            "yes",
        }

        queryset = PredictionResult.objects.all()
        if model_name:
            queryset = queryset.filter(model_name=model_name)
        if alert_only:
            queryset = queryset.filter(alert_triggered=True)

        queryset = queryset.order_by(
            "entity_id",
            "-predicted_for_date",
            "-created_at",
        ).distinct("entity_id")

        serializer = PredictionResultSerializer(queryset, many=True)
        return Response(serializer.data)


class PredictionTriggerView(APIView):
    """
    POST /api/predictions/trigger/

    Manually runs the full prediction chain (simulation tick → predictions →
    dashboard sync) synchronously and returns a JSON summary.
    Staff authentication required.
    """

    permission_classes = [IsAdminUser]

    @extend_schema(
        tags=["Predictions"],
        responses={200: None},
        description=(
            "Trigger the full prediction pipeline manually: simulation tick → "
            "ML predictions → dashboard sync. Staff only."
        ),
    )
    def post(self, request):
        summary = _run_prediction_chain()
        if summary["ok"]:
            return Response(
                {"status": "ok", **summary},
                status=status.HTTP_200_OK,
            )
        # Partial or full failure — 207 lets the caller inspect per-step results
        return Response(
            {"status": "partial_failure", **summary},
            status=status.HTTP_207_MULTI_STATUS,
        )


class SupplyHistoryView(APIView):
    """
    GET /api/inventory/supply-history/

    Query params:
      blood_product_type — required
      hospital_id        — optional; when omitted, aggregates across all hospitals
      hours              — optional, default 72, max 2160 (90 days)

    Returns per-tick totals ordered chronologically (oldest first) for sparkline
    rendering.  An empty list means the simulation has not ticked yet — the
    frontend hides the sparkline in that case.
    """

    permission_classes = [AllowAny]

    def get(self, request):
        blood_product_type = request.query_params.get("blood_product_type", "").strip()
        if not blood_product_type:
            return Response(
                {"detail": "blood_product_type is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        hospital_id = request.query_params.get("hospital_id", "").strip()

        try:
            raw_hours = int(request.query_params.get("hours", 72))
            hours = max(1, min(raw_hours, 2160))
        except (TypeError, ValueError):
            hours = 72

        from datetime import timedelta

        from django.db.models import Avg, FloatField, Sum
        from django.db.models.functions import Trunc
        from django.utils import timezone

        from inventory.models import BloodSupplySnapshot

        since = timezone.now() - timedelta(hours=hours)
        qs = BloodSupplySnapshot.objects.filter(
            blood_product_type=blood_product_type,
            recorded_at__gte=since,
        )
        if hospital_id:
            qs = qs.filter(hospital__hospital_id=hospital_id)

        # Group by tick timestamp so each tick is one point regardless of hospital count
        rows = (
            qs.values("recorded_at")
            .annotate(
                current_stock_units=Sum("current_stock_units", output_field=FloatField()),
                usage_today=Sum("usage_today", output_field=FloatField()),
            )
            .order_by("recorded_at")
        )
        return Response(list(rows), status=status.HTTP_200_OK)


class DonorPropensityScoresView(APIView):
    """
    GET /api/inventory/donors/scores/

    Returns configured donor priority scores for every donor for a given date.

    Query params:
      date — ISO date string (YYYY-MM-DD), default: today
    """

    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Predictions"],
        responses={200: None},
        description=(
            "Returns configured donor priority scores per donor, keyed by donor_id."
        ),
    )
    def get(self, request):
        raw_date = request.query_params.get("date", "").strip()
        if raw_date:
            try:
                score_date = date_type.fromisoformat(raw_date)
            except ValueError:
                return Response(
                    {"detail": "Invalid date format. Use YYYY-MM-DD."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            score_date = date_type.today()

        model_name = configured_donor_score_result_model()
        if not model_name:
            return Response(
                {"detail": "No enabled donor score scheduled prediction is configured."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        rows = (
            PredictionResult.objects.filter(
                model_name=model_name,
                predicted_for_date=score_date,
            )
            .values("entity_id", "predicted_value", "blood_type")
            .order_by("entity_id")
        )
        results = list(rows)
        expected_donations = sum(r["predicted_value"] for r in results)
        return Response(
            {
                "date": score_date.isoformat(),
                "model_name": model_name,
                "count": len(results),
                "expected_donations": round(expected_donations, 2),
                "results": results,
            },
            status=status.HTTP_200_OK,
        )
