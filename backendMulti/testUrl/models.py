# testUrl/models.py

from django.db import models

class Donneur(models.Model):
    nom            = models.CharField(max_length=200)
    adresse        = models.TextField()
    telephone      = models.CharField(max_length=50)
    latitude       = models.FloatField()
    longitude      = models.FloatField()
    groupe_sanguin = models.CharField(max_length=10)
    age            = models.IntegerField()
    poids          = models.FloatField()
    nb_dons        = models.IntegerField(default=0)
    dernier_don    = models.DateField(null=True, blank=True)
    disponible     = models.CharField(max_length=50)
    region         = models.CharField(max_length=100)
    rating         = models.FloatField(default=0.0)
    image          = models.URLField(max_length=500)

    class Meta:
        db_table = 'donneurs'

    def __str__(self):
        return f"{self.nom} ({self.groupe_sanguin})"


class DataVersion(models.Model):
    key          = models.CharField(max_length=100, unique=True)
    last_updated = models.DateTimeField()
    version_hash = models.CharField(max_length=64)

    class Meta:
        db_table = 'data_versions'

    def __str__(self):
        return f"{self.key} → {self.last_updated}"    