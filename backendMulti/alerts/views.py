from __future__ import annotations

from drf_spectacular.utils import OpenApiExample, OpenApiParameter, OpenApiTypes, extend_schema
from rest_framework import status
from rest_framework.generics import get_object_or_404
from rest_framework.response import Response
from rest_framework.views import APIView

from alerts.models import AlertEvent, AlertRule, AlertSeverity, AlertStatus
from alerts.serializers import (
    AlertActionSerializer,
    AlertCreateTestSerializer,
    AlertEventSerializer,
    AlertRuleSerializer,
    AlertSummarySerializer,
    EmptySerializer,
)
from alerts.services.event_service import acknowledge_event, create_or_refresh_event, escalate_event, resolve_event, InvalidTransitionError


class AlertRuleListView(APIView):
    serializer_class = AlertRuleSerializer

    @extend_schema(tags=["Alerts"], responses={200: AlertRuleSerializer(many=True)})
    def get(self, request):
        serializer = AlertRuleSerializer(AlertRule.objects.all(), many=True)
        return Response(serializer.data)

    @extend_schema(tags=["Alerts"], request=AlertRuleSerializer, responses={201: AlertRuleSerializer})
    def post(self, request):
        serializer = AlertRuleSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        rule = serializer.save(created_by=str(request.user) if request.user.is_authenticated else "admin")
        return Response(AlertRuleSerializer(rule).data, status=status.HTTP_201_CREATED)


class AlertRuleDetailView(APIView):
    serializer_class = AlertRuleSerializer

    def get_object(self, pk: int) -> AlertRule:
        return get_object_or_404(AlertRule, pk=pk)

    @extend_schema(tags=["Alerts"], responses={200: AlertRuleSerializer})
    def get(self, request, pk):
        return Response(AlertRuleSerializer(self.get_object(pk)).data)

    @extend_schema(tags=["Alerts"], request=AlertRuleSerializer, responses={200: AlertRuleSerializer})
    def put(self, request, pk):
        rule = self.get_object(pk)
        serializer = AlertRuleSerializer(rule, data=request.data, partial=False)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

    @extend_schema(tags=["Alerts"], responses={204: None})
    def delete(self, request, pk):
        self.get_object(pk).delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class AlertListView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["Alerts"],
        parameters=[
            OpenApiParameter("status", OpenApiTypes.STR, OpenApiParameter.QUERY),
            OpenApiParameter("severity", OpenApiTypes.STR, OpenApiParameter.QUERY),
            OpenApiParameter("entity_type", OpenApiTypes.STR, OpenApiParameter.QUERY),
            OpenApiParameter("entity_id", OpenApiTypes.STR, OpenApiParameter.QUERY),
            OpenApiParameter("blood_type", OpenApiTypes.STR, OpenApiParameter.QUERY),
        ],
        responses={200: AlertEventSerializer(many=True)},
    )
    def get(self, request):
        queryset = AlertEvent.objects.select_related("rule").all()
        # By default exclude resolved/muted so the list shows only active alerts.
        # Pass ?status=resolved (or any other value) to override.
        status_filter = request.query_params.get("status")
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        else:
            queryset = queryset.exclude(status__in=[AlertStatus.RESOLVED, AlertStatus.MUTED])
        for field in ("severity", "entity_type", "entity_id", "blood_type"):
            value = request.query_params.get(field)
            if value:
                queryset = queryset.filter(**{field: value})
        return Response(AlertEventSerializer(queryset, many=True).data)


class AlertSummaryView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(tags=["Alerts"], responses={200: AlertSummarySerializer})
    def get(self, request):
        queryset = AlertEvent.objects.all()
        payload = {
            "open": queryset.filter(status=AlertStatus.OPEN).count(),
            "acknowledged": queryset.filter(status=AlertStatus.ACKNOWLEDGED).count(),
            "escalated": queryset.filter(status=AlertStatus.ESCALATED).count(),
            "resolved": queryset.filter(status=AlertStatus.RESOLVED).count(),
            "critical": queryset.filter(severity=AlertSeverity.CRITICAL).count(),
            "warning": queryset.filter(severity=AlertSeverity.WARNING).count(),
        }
        return Response(payload)


class AlertDetailView(APIView):
    serializer_class = EmptySerializer

    def get_object(self, pk):
        return get_object_or_404(AlertEvent.objects.select_related("rule"), pk=pk)

    @extend_schema(tags=["Alerts"], responses={200: AlertEventSerializer})
    def get(self, request, pk):
        return Response(AlertEventSerializer(self.get_object(pk)).data)


class AlertAcknowledgeView(APIView):
    serializer_class = AlertActionSerializer

    @extend_schema(tags=["Alerts"], request=AlertActionSerializer, responses={200: AlertEventSerializer})
    def post(self, request, pk):
        serializer = AlertActionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            event = acknowledge_event(
                get_object_or_404(AlertEvent.objects.select_related("rule"), pk=pk),
                actor=str(request.user) if request.user.is_authenticated else "admin",
                note=serializer.validated_data.get("note", ""),
            )
        except InvalidTransitionError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(AlertEventSerializer(event).data)


class AlertResolveView(APIView):
    serializer_class = AlertActionSerializer

    @extend_schema(tags=["Alerts"], request=AlertActionSerializer, responses={200: AlertEventSerializer})
    def post(self, request, pk):
        serializer = AlertActionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            event = resolve_event(
                get_object_or_404(AlertEvent.objects.select_related("rule"), pk=pk),
                actor=str(request.user) if request.user.is_authenticated else "admin",
                note=serializer.validated_data.get("note", ""),
            )
        except InvalidTransitionError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(AlertEventSerializer(event).data)


class AlertEscalateView(APIView):
    serializer_class = AlertActionSerializer

    @extend_schema(tags=["Alerts"], request=AlertActionSerializer, responses={200: AlertEventSerializer})
    def post(self, request, pk):
        serializer = AlertActionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            event = escalate_event(
                get_object_or_404(AlertEvent.objects.select_related("rule"), pk=pk),
                actor=str(request.user) if request.user.is_authenticated else "admin",
                note=serializer.validated_data.get("note", ""),
            )
        except InvalidTransitionError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(AlertEventSerializer(event).data)


class AlertTestFireView(APIView):
    serializer_class = AlertCreateTestSerializer

    @extend_schema(
        tags=["Alerts"],
        request=AlertCreateTestSerializer,
        responses={201: AlertEventSerializer},
        examples=[
            OpenApiExample(
                "Critical stock alert",
                value={
                    "title": "Critical O- shortage risk",
                    "message": "Hospital H005 may run out of O- within 2 days.",
                    "severity": "critical",
                    "entity_type": "hospital",
                    "entity_id": "H005",
                    "blood_type": "O-",
                    "context": {"current_stock_units": 6, "predicted_days_until_stockout": 2},
                },
                request_only=True,
            )
        ],
    )
    def post(self, request):
        serializer = AlertCreateTestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        rule, _ = AlertRule.objects.get_or_create(
            name=f"manual::{data['title']}",
            defaults={
                "description": data["message"],
                "trigger_type": "manual",
                "severity_base": data["severity"],
                "conditions": {"field": "entity_id", "op": "==", "value": data.get("entity_id", "")},
                "channels": ["websocket", "in_app"],
                "created_by": "manual-test",
            },
        )

        event, _ = create_or_refresh_event(
            rule=rule,
            event_key=f"manual::{data['title']}::{data.get('entity_id', '')}::{data.get('blood_type', '')}",
            title=data["title"],
            message=data["message"],
            context=data.get("context", {}),
            source_type="manual",
            source_ref="manual-test",
            entity_type=data.get("entity_type", ""),
            entity_id=data.get("entity_id", ""),
            blood_type=data.get("blood_type", ""),
            predicted_value=data.get("context", {}).get("predicted_days_until_stockout"),
        )
        if event is None:
            return Response(
                {"detail": "Alert cap reached for this severity — no new event created."},
                status=status.HTTP_429_TOO_MANY_REQUESTS,
            )
        event.severity = data["severity"]
        event.status = data["status"]
        event.save(update_fields=["severity", "status"])
        return Response(AlertEventSerializer(event).data, status=status.HTTP_201_CREATED)