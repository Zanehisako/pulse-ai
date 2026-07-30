# testUrl/serializers.py

from rest_framework import serializers
from .models import CentreDon

class CentreDonSerializer(serializers.ModelSerializer):
    class Meta:
        model  = CentreDon
        fields = [
            'id', 'code_centre', 'nom', 'type',
            'adresse', 'code_postal', 'ville', 'region', 'pays',
            'latitude', 'longitude',
            'telephone', 'email',
            'capacite_journaliere', 'horaires',
            'statut', 'equipements', 'date_ouverture',
        ]