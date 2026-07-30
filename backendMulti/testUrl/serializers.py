from rest_framework import serializers


class EmptySerializer(serializers.Serializer):
    pass


class LaboratoryErrorResponseSerializer(serializers.Serializer):
    error = serializers.CharField()


class LaboratoriesResponseSerializer(serializers.Serializer):
    count = serializers.IntegerField()
    results = serializers.ListField(child=serializers.JSONField())
