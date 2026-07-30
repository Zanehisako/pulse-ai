from __future__ import annotations

from rest_framework import serializers


class TwinRunCreateSerializer(serializers.Serializer):
    mode = serializers.CharField(required=False, default="operational_shadow")
    scenario_key = serializers.CharField(required=False, default="baseline")
    strategy_key = serializers.CharField(required=False, default="baseline")
    source_mode = serializers.ChoiceField(required=False, choices=["live", "synthetic"], default="live")
    seed = serializers.IntegerField(required=False, min_value=0, default=100)


class TwinSnapshotCreateSerializer(serializers.Serializer):
    source = serializers.ChoiceField(required=False, choices=["live", "synthetic"])


class TwinEventCreateSerializer(serializers.Serializer):
    event_key = serializers.CharField()
    severity = serializers.FloatField(required=False, min_value=0.0, max_value=1.0, default=0.5)
    start_hour = serializers.FloatField(required=False, min_value=0.0, default=0.0)
    source = serializers.CharField(required=False, default="manual")
    payload = serializers.DictField(required=False, default=dict)


class TwinActionCreateSerializer(serializers.Serializer):
    action_key = serializers.CharField()
    simulated_hour = serializers.FloatField(required=False, min_value=0.0, default=0.0)
    source = serializers.CharField(required=False, default="manual")
    payload = serializers.DictField(required=False, default=dict)


class TwinBranchCreateSerializer(serializers.Serializer):
    branch_key = serializers.CharField()
    policy_overrides = serializers.DictField(required=False, default=dict)
    action_overrides = serializers.ListField(required=False, default=list)
    event_overrides = serializers.ListField(required=False, default=list)


class TwinRecommendationCreateSerializer(serializers.Serializer):
    branch_id = serializers.UUIDField(required=False)


class TwinPromotionCreateSerializer(serializers.Serializer):
    action_id = serializers.UUIDField()
