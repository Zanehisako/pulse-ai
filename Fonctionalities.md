# SUJET 1 : Plateforme Web Responsive de Tableaux de Bord Temps Réel
## Fonctionnalités détaillées et modules

### Module A : Système d’authentification et de profils
1. **Portail de connexion unique (SSO) avec token JWT**
2. **Gestion multi-profils :**
   - **Administrateur :** accès complet, gestion utilisateurs
   - **Gestionnaire opérationnel :** KPI temps réel, alertes
   - **Logisticien :** vue transport, flotte, itinéraires
   - **Coordinateur centre :** stocks locaux, rendez-vous
   - **Consultant :** vue en lecture seule, export
3. **Journal d’audit des connexions et actions sensibles**

### Module B : Tableau de bord principal
4. **Vue d’ensemble à 360° :**
   - Widgets configurables en *drag & drop*
   - Indicateurs synthétiques (niveaux de stock, urgences en cours, collectes du jour)
   - Carte interactive des centres avec code couleur (vert/orange/rouge)
5. **Filtres intelligents :**
   - Par région/centre
   - Par type de produit sanguin
   - Par période (24h, 7j, 30j)
   - Par niveau d’alerte

### Module C : Visualisations temps réel
6. **Graphiques dynamiques :**
   - Courbes de tendance avec prévisions J+1 à J+30
   - Cartes thermiques des réserves par centre et type
   - Diagrammes de Gantt des collectes planifiées vs réalisées
   - Graphiques en chandelier (ouvertures/fermetures de stocks)
7. **Fonctionnalités d’analyse :**
   - Zoom et défilement temporel
   - Comparaison période (vs même jour semaine dernière)
   - Export instantané en PNG / PDF / CSV

### Module D : Système d’alerte et de notification
8. **Alertes configurées par seuils :**
   - Stock critique < 2 jours de consommation
   - Collecte en retard > 2 heures
   - Rupture imminente d’un groupe sanguin rare
9. **Notification multi-canaux :**
   - Badges sur l’interface
   - Notifications push navigateur
   - Mode veille avec écran dédié aux urgences
   - Journal des alertes avec actions de résolution

### Module E : Mode hors-ligne et performance
10. **Cache intelligent :**
    - Sauvegarde automatique des dernières données consultées
    - Synchronisation incrémentielle au retour en ligne
    - Indicateur de fraîcheur des données (dernière mise à jour)
11. **Optimisations :**
    - *Lazy loading* des graphiques complexes
    - Compression des données historiques
    - *Prefetching* des données prévisibles

### Module F : Personnalisation avancée
12. **Espaces de travail personnels :**
    - Sauvegarde des vues favorites
    - Partage de tableaux de bord entre collègues
    - Templates de reporting par métier
13. **Paramètres utilisateur :**
    - Thèmes (clair / sombre / haut contraste)
    - Langue (FR / EN)
    - Unités de mesure configurables

### Module G : Module géospatial temps réel
14. **Cartographie interactive :**
    - Intégration de bibliothèque cartographique open-source (Leaflet.js / MapLibre GL JS)
    - Affichage performant de données GeoJSON et Vector Tiles
    - Visualisations avancées : clusters dynamiques, heatmaps, flux animés, polygones de secteur
15. **Couches interactives :**
    - Filtrage par type d’entité (centre, véhicule, ressource)
    - Filtrage par période temporelle et statut
    - Calculs géospatiaux côté client : distances, isochrones simples, recherche de points dans un rayon
16. **Intégration temps réel :**
    - Mise à jour live des positions et statuts sur la carte
    - Synchronisation avec les alertes géolocalisées
    - Visualisation spatiale des flux logistiques
17. **Interface cartographique unifiée :**
    - Widget carte dans le tableau de bord principal
    - Navigation bidirectionnelle entre carte et données
    - Export des vues cartographiques et données spatiales

---

# SUJET 2 : Application Mobile Hybride avec Fonctionnalités Hors-ligne Avancées
## Fonctionnalités détaillées et modules

### Module A : Gestion des missions terrain
1. **Carnet de route intelligent :**
   - Liste des collectes assignées avec trajet optimisé
   - Scan QR code des poches collectées
   - Photo attestant des conditions de transport
   - Signature électronique des responsables site
2. **Suivi en temps réel :**
   - Position GPS partagée (optionnel, avec consentement)
   - Estimation des temps de trajet
   - Alertes trafic / incidents

### Module B : Gestion des stocks locale
3. **Inventaire mobile :**
   - Scan des codes barres des poches
   - Saisie rapide des entrées / sorties
   - Contrôle visuel de l’état des produits
   - Lecture NFC des étiquettes intelligentes
4. **Alertes locales :**
   - Détection de rupture de chaîne du froid
   - Rappels de dates de péremption (48h avant)
   - Stocks minimums atteints

### Module C : Communication et collaboration
5. **Messagerie interne sécurisée :**
   - Conversations par équipe / mission
   - Envoi de photos et documents
   - Messages priorités (urgences)
6. **Annuaire dynamique :**
   - Contacts par centre / rôle
   - Statut de disponibilité
   - Appel direct intégré

### Module D : Synchronisation Offline-First
7. **Stratégie de sync intelligente :**
   - Différentielle : seuls les changements sont sync
   - Priorisée : données critiques d’abord
   - Opportuniste : utilise toute connexion disponible
8. **Gestion des conflits :**
   - Règles métier de résolution
   - Journal des modifications en conflit
   - Interface de résolution manuelle si besoin

### Module E : Fonctionnalités device native
9. **Géolocalisation :**
   - Géofencing des centres (notification à l’approche)
   - Enregistrement automatique des lieux de collecte
10. **Appareil photo :**
    - Scan documents (fiches de don)
    - Capture d’anomalies (matériel défectueux)
    - OCR intégré pour saisie automatique
11. **Notifications device :**
    - Vibrations pour alertes critiques
    - LED flash pour notifications urgentes
    - Sonneries personnalisables

### Module F : Interface mobile optimisée
12. **Navigation gesture-based :**
    - Swipe pour actions rapides
    - Pull-to-refresh pour synchronisation
    - Navigation vocale (accessibilité)
13. **Mode contraintes terrain :**
    - Interface ultra-simplifiée (gros boutons)
    - Mode mains-libres
    - Compatibilité gants tactiles

---

# SUJET 3 : Système de Prédiction et d’Alerte Précoce par Apprentissage Automatique
## Fonctionnalités détaillées et modules

### Module A : Ingestion et préparation des données
1. **Sources de données multiples :**
   - Historique des dons (5 ans minimum)
   - Données externes : météo, vacances scolaires, événements locaux
   - Données hospitalières : chirurgies planifiées, urgences historiques
   - Données démographiques : âge, sexe, localisation donneurs
2. **Pipeline de feature engineering :**
   - Variables temporelles : jour de semaine, mois, saison, vacances
   - Variables cycliques : encodage sin/cos des heures/jours
   - Variables retardées (*lag features*) : dons J-7, J-30, J-365
   - Variables dérivées : moyennes mobiles, tendances

### Module B : Modèles de prédiction multi-horizons
3. **Architecture modulaire :**
   - Modèle court terme (J+1 à J+7) : LSTM/GRU pour patterns complexes
   - Modèle moyen terme (J+8 à J+30) : XGBoost/LightGBM avec features cycliques
   - Modèle long terme (+1 mois) : Prophet pour tendances saisonnières
   - Modèle d’urgence : détection d’anomalies (Isolation Forest, AutoEncoder)
4. **Ensemble learning :**
   - Stacking des prédictions
   - Poids dynamiques selon la performance récente
   - Intervalles de confiance calculés

### Module C : Pipeline MLOps complet
5. **Entraînement automatisé :**
   - Re-training hebdomadaire avec nouvelles données
   - Validation croisée temporelle (time-series split)
   - Sélection automatique du meilleur modèle (AUC, RMSE)
6. **Déploiement continu :**
   - Packaging avec MLflow
   - Tests A/B sur sous-ensemble de centres
   - Rollback automatique en cas de dégradation
7. **Monitoring production :**
   - Drift detection des données d’entrée
   - Performance tracking vs réalité terrain
   - Alertes techniques (latence, erreurs)

### Module D : Système d’alerte intelligent
8. **Configuration des règles :**
   - Seuils dynamiques : % d’écart à la prévision
   - Combinaisons : stock bas ET prévision faible
   - Escalade : alerte niveau 1 → niveau 2 après X heures
9. **Notifications contextuelles :**
   - Pré-alerte : tendance défavorable détectée
   - Alerte confirmée : seuil franchi
   - Alerte critique : risque imminent
   - Retour à la normale : situation résolue
10. **Dashboard des alertes :**
    - Historique et statut
    - Actions entreprises
    - Temps de résolution moyen

### Module E : Simulateur What-If
11. **Scénarios pré-configurés :**
    - Pandémie : baisse de 30% des dons
    - Catastrophe naturelle : augmentation urgences +50%
    - Grève transport : collectes mobiles annulées
    - Campagne marketing : +15% de nouveaux donneurs
12. **Modélisation interactive :**
    - Curseurs pour ajuster les paramètres
    - Impact en cascade visualisé
   AR RECOMMENDATIONS GENERATED AUTOMATICALLY (RECOMMANDATIONS GÉNÉRÉES AUTOMATIQUEMENT)
13. **Export de scénarios :**
    - Rapports détaillés
    - Présentations exécutives
    - Plans d’action générés

### Module F : Interface de gestion des modèles
14. **Tableau de bord data science :**
    - Performance comparée des modèles
    - Importance des features (SHAP values)
    - Analyse des erreurs de prédiction
15. **Configuration manuelle :**
    - Override des prévisions si besoin
    - Ajustement des paramètres de sensibilité
    - Calibration des modèles par région
16. **Transparence et audit :**
    - Explication des prédictions individuelles
    - Traçabilité des décisions algorithmiques
    - Conformité RGPD / éthique
