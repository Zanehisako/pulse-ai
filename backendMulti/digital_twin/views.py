from __future__ import annotations

from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.generics import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .config import DigitalTwinConfigError, public_config_payload
from .models import TwinAction, TwinBranch, TwinRun
from .serializers import (
    TwinActionCreateSerializer,
    TwinBranchCreateSerializer,
    TwinEventCreateSerializer,
    TwinPromotionCreateSerializer,
    TwinRecommendationCreateSerializer,
    TwinRunCreateSerializer,
    TwinSnapshotCreateSerializer,
)
from .services import (
    action_payload,
    branch_payload,
    capture_snapshot,
    create_twin_run,
    current_state,
    event_payload,
    execute_action,
    inject_event,
    promote_action,
    recommend_action,
    recommendation_payload,
    run_branch,
    run_payload,
    snapshot_payload,
)


def _actor(request) -> str:
    return str(request.user) if request.user and request.user.is_authenticated else "system"


def _error(exc: Exception) -> Response:
    return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


class TwinConfigView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(tags=["Digital Twin"], responses={200: None})
    def get(self, request):
        return Response(public_config_payload())


class TwinRunListView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = TwinRunCreateSerializer

    @extend_schema(tags=["Digital Twin"], responses={200: None})
    def get(self, request):
        rows = TwinRun.objects.all()[:50]
        return Response([run_payload(row) for row in rows])

    @extend_schema(tags=["Digital Twin"], request=TwinRunCreateSerializer, responses={201: None})
    def post(self, request):
        serializer = TwinRunCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            run = create_twin_run(serializer.validated_data, user=request.user)
            snapshot = capture_snapshot(run, actor=_actor(request))
        except DigitalTwinConfigError as exc:
            return _error(exc)
        return Response(
            {
                "run": run_payload(run),
                "snapshot": snapshot_payload(snapshot),
            },
            status=status.HTTP_201_CREATED,
        )


class TwinRunDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get_object(self, run_id):
        return get_object_or_404(TwinRun, run_id=run_id)

    @extend_schema(tags=["Digital Twin"], responses={200: None})
    def get(self, request, run_id):
        return Response(current_state(self.get_object(run_id)))


class TwinSnapshotView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = TwinSnapshotCreateSerializer

    @extend_schema(tags=["Digital Twin"], request=TwinSnapshotCreateSerializer, responses={201: None})
    def post(self, request, run_id):
        run = get_object_or_404(TwinRun, run_id=run_id)
        serializer = TwinSnapshotCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            snapshot = capture_snapshot(
                run,
                source=serializer.validated_data.get("source"),
                actor=_actor(request),
            )
        except DigitalTwinConfigError as exc:
            return _error(exc)
        return Response(snapshot_payload(snapshot), status=status.HTTP_201_CREATED)


class TwinEventView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = TwinEventCreateSerializer

    @extend_schema(tags=["Digital Twin"], request=TwinEventCreateSerializer, responses={201: None})
    def post(self, request, run_id):
        run = get_object_or_404(TwinRun, run_id=run_id)
        serializer = TwinEventCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        try:
            event = inject_event(
                run,
                event_key=data["event_key"],
                severity=data["severity"],
                start_hour=data["start_hour"],
                source=data["source"],
                payload=data["payload"],
                actor=_actor(request),
            )
        except DigitalTwinConfigError as exc:
            return _error(exc)
        return Response(event_payload(event), status=status.HTTP_201_CREATED)


class TwinActionView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = TwinActionCreateSerializer

    @extend_schema(tags=["Digital Twin"], request=TwinActionCreateSerializer, responses={201: None})
    def post(self, request, run_id):
        run = get_object_or_404(TwinRun, run_id=run_id)
        serializer = TwinActionCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        try:
            action = execute_action(
                run,
                action_key=data["action_key"],
                simulated_hour=data["simulated_hour"],
                source=data["source"],
                payload=data["payload"],
                actor=_actor(request),
            )
        except DigitalTwinConfigError as exc:
            return _error(exc)
        return Response(action_payload(action), status=status.HTTP_201_CREATED)


class TwinBranchView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = TwinBranchCreateSerializer

    @extend_schema(tags=["Digital Twin"], request=TwinBranchCreateSerializer, responses={201: None})
    def post(self, request, run_id):
        run = get_object_or_404(TwinRun, run_id=run_id)
        serializer = TwinBranchCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        try:
            branch = run_branch(
                run,
                branch_key=data["branch_key"],
                policy_overrides=data["policy_overrides"],
                action_overrides=data["action_overrides"],
                event_overrides=data["event_overrides"],
                actor=_actor(request),
            )
        except DigitalTwinConfigError as exc:
            return _error(exc)
        return Response(branch_payload(branch), status=status.HTTP_201_CREATED)


class TwinRecommendationView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = TwinRecommendationCreateSerializer

    @extend_schema(tags=["Digital Twin"], request=TwinRecommendationCreateSerializer, responses={201: None})
    def post(self, request, run_id):
        run = get_object_or_404(TwinRun, run_id=run_id)
        serializer = TwinRecommendationCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        branch = None
        if serializer.validated_data.get("branch_id"):
            branch = get_object_or_404(
                TwinBranch,
                branch_id=serializer.validated_data["branch_id"],
                run=run,
            )
        try:
            recommendation = recommend_action(run, branch=branch, actor=_actor(request))
        except DigitalTwinConfigError as exc:
            return _error(exc)
        return Response(recommendation_payload(recommendation), status=status.HTTP_201_CREATED)


class TwinPromotionView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = TwinPromotionCreateSerializer

    @extend_schema(tags=["Digital Twin"], request=TwinPromotionCreateSerializer, responses={200: None})
    def post(self, request):
        serializer = TwinPromotionCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        action = get_object_or_404(TwinAction, action_id=serializer.validated_data["action_id"])
        try:
            promoted = promote_action(action, actor=_actor(request))
        except DigitalTwinConfigError as exc:
            return _error(exc)
        return Response(action_payload(promoted))
