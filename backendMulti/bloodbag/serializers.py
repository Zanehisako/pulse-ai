from rest_framework import serializers

from bloodbag.models import BloodBag


class BloodBagSerializer(serializers.ModelSerializer):
    class Meta:
        model = BloodBag
        fields = [
            "id",
            "barcode",
            "blood_type",
            "component",
            "status",
            "collection_date",
            "expiry_date",
            "donor_id",
            "volume_ml",
            "notes",
            "created_at",
            "updated_at",
        ]


class SyncOperationSerializer(serializers.Serializer):
    barcode = serializers.CharField()
    operation = serializers.ChoiceField(choices=["upsert", "delete"])
    bag = BloodBagSerializer(required=False, allow_null=True)
    created_at = serializers.DateTimeField(required=False)
