from django.db import transaction
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from bloodbag.models import BloodBag
from bloodbag.serializers import BloodBagSerializer, SyncOperationSerializer


class BloodBagListView(APIView):
    def get(self, request):
        queryset = BloodBag.objects.all().order_by("expiry_date", "barcode")
        serializer = BloodBagSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class BloodBagSyncView(APIView):
    @transaction.atomic
    def post(self, request):
        serializer = SyncOperationSerializer(
            data=request.data.get("operations", []),
            many=True,
        )
        serializer.is_valid(raise_exception=True)

        for operation in serializer.validated_data:
            if operation["operation"] == "delete":
                BloodBag.objects.filter(barcode=operation["barcode"]).delete()
                continue

            bag_payload = operation.get("bag")
            if not bag_payload:
                continue

            payload = dict(bag_payload)
            payload.pop("id", None)
            BloodBag.objects.update_or_create(
                barcode=payload["barcode"],
                defaults=payload,
            )

        bags = BloodBag.objects.all().order_by("expiry_date", "barcode")
        return Response(
            {
                "bags": BloodBagSerializer(bags, many=True).data,
            },
            status=status.HTTP_200_OK,
        )
