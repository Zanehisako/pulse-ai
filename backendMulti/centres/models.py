# testUrl/models.py

from django.db import models

class CentreDon(models.Model):
    TYPE_CHOICES = [
        ('fixe',     'Fixe'),
        ('mobile',   'Mobile'),
        ('hopital',  'Hôpital'),
        ('clinique', 'Clinique'),
    ]
    STATUT_CHOICES = [
        ('actif',              'Actif'),
        ('ferme_temporaire',   'Fermé temporairement'),
        ('en_maintenance',     'En maintenance'),
        ('ferme_definitif',    'Fermé définitivement'),
    ]

    code_centre          = models.CharField(max_length=20, unique=True)
    nom                  = models.CharField(max_length=150)
    type                 = models.CharField(max_length=20, choices=TYPE_CHOICES)
    adresse              = models.TextField(blank=True)
    code_postal          = models.CharField(max_length=10, blank=True)
    ville                = models.CharField(max_length=100, blank=True)
    region               = models.CharField(max_length=100, blank=True)
    pays                 = models.CharField(max_length=50, default='France')
    latitude             = models.DecimalField(max_digits=10, decimal_places=8, null=True)
    longitude            = models.DecimalField(max_digits=11, decimal_places=8, null=True)
    telephone            = models.CharField(max_length=15, blank=True)
    email                = models.EmailField(blank=True)
    capacite_journaliere = models.IntegerField(null=True, blank=True)
    horaires             = models.JSONField(default=dict, blank=True)
    statut               = models.CharField(max_length=20, choices=STATUT_CHOICES, default='actif')
    equipements          = models.JSONField(default=list, blank=True)
    date_ouverture       = models.DateField(null=True, blank=True)

    class Meta:
        db_table = 'centre_don'

    def __str__(self):
        return f"{self.nom} ({self.type})"