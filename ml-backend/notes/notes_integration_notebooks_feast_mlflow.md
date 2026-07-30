# Notes – Intégration des modèles Notebook dans Feast + MLflow

**Date :** 11 Février 2026  
**Contexte :** ML Backend (`/Users/mac/Documents/pios-1/ml-backend`)  
**Objectif :** unifier l’entraînement, la gestion des features et le tracking MLflow des modèles créés dans les notebooks, via **un seul pipeline scriptable** et **dynamique** (nombre variable de modèles et de features).

---

## 1. Résumé Exécutif

Nous avons implémenté un pipeline unique qui :
- lit la configuration des modèles notebook (`config.json`)
- applique une spécification d’entraînement (`training_spec.json`)
- génère les features Feast (offline/online)
- entraîne les modèles (multi-modèles, multi-features)
- logge paramètres/métriques/artifacts dans MLflow
- sauvegarde les modèles entraînés localement

Résultat : une commande unique permet désormais de lancer l’intégration complète Notebook -> Feast -> MLflow.

---

## 2. Pourquoi cette intégration était nécessaire

Avant :
- pipeline MLOps existant orienté principalement modèle unique (logistic regression bloodbank)
- logique de features en partie dispersée entre notebooks et scripts
- difficulté à industrialiser plusieurs modèles notebook avec des feature sets différents

Cible :
- standardiser le passage Notebook -> production MLOps
- éviter le hardcode du nombre de modèles/features
- garder la traçabilité des runs et la reproductibilité

---

## 3. Changements réalisés (code)

### 3.1 Nouveau pipeline dynamique

Fichier ajouté :
- `/Users/mac/Documents/pios-1/ml-backend/pios_ml_backend/mlops/notebook_models_pipeline.py`

Rôle principal :
- chargement des modèles notebook (catalogue)
- chargement des specs d’entraînement (dataset, target, algo, timestamp/entity, etc.)
- préparation des données
- feature engineering ciblé
- génération automatique des définitions Feast
- entraînement + log MLflow pour chaque modèle

### 3.2 Exposition du pipeline dans le package mlops

Fichier modifié :
- `/Users/mac/Documents/pios-1/ml-backend/pios_ml_backend/mlops/__init__.py`

Ajout de l’export :
- `NotebookModelsFeastMLflowPipeline`

### 3.3 Intégration CLI

Fichier modifié :
- `/Users/mac/Documents/pios-1/ml-backend/pios_ml_backend/main.py`

Nouveautés :
- commande `mlops-train-notebooks`
- options :
  - `--model-config`
  - `--training-spec`
  - `--model-id` (répétable)
  - `--strict-features`
  - `--register-models`
  - `--skip-materialize`

### 3.4 Spécification d’entraînement par défaut

Fichier ajouté :
- `/Users/mac/Documents/pios-1/ml-backend/notebooks/ml_models/training_spec.json`

Contient la config de base pour :
- `donor_prediction`
- `xgb_demand_forecast_j+30`

### 3.5 Documentation

Fichier modifié :
- `/Users/mac/Documents/pios-1/ml-backend/README.md`

Ajouts :
- section dédiée à `mlops-train-notebooks`
- exemples de commandes
- options de configuration

---

## 4. Comment le pipeline fonctionne (architecture)

### 4.1 Entrées

- Catalogue notebook :
  - `/Users/mac/Documents/pios-1/ml-backend/notebooks/ml_models/config.json`
- Spécification d’entraînement :
  - `/Users/mac/Documents/pios-1/ml-backend/notebooks/ml_models/training_spec.json`

Le pipeline fusionne ces deux sources pour chaque modèle.

### 4.2 Préparation et features

Le pipeline gère automatiquement :
- parsing timestamp (`date`, `as_of_date`, etc.)
- construction d’entité Feast (`entity_id`)
- conversion/encodage des features catégorielles
- imputation de features manquantes (par défaut, fallback à `0.0`)
- mode strict possible (`--strict-features`)

Feature engineering implémenté (notamment pour modèle demand forecast) :
- `lag1`, `lag7`, `lag30`
- `roll7`, `roll30`
- one-hot attendus dans config (`hospital_*`, `blood_type_*`)
- fallback pour `dow`, `weekend`, `month` si absent

### 4.3 Feast

Le pipeline génère automatiquement :
- parquet par modèle dans `feature_repo/data/`
- définitions Feast dynamiques dans :
  - `/Users/mac/Documents/pios-1/ml-backend/feature_repo/notebook_generated_feature_definitions.py`

Puis exécute :
- `feast apply`
- `materialize_incremental` (sauf si `--skip-materialize`)

### 4.4 Entraînement et MLflow

Pour chaque modèle :
- récupération training set via `get_historical_features`
- split train/test
- entraînement selon algo déclaré
- calcul métriques
- log dans MLflow (params, metrics, artifacts, feature spec)
- sauvegarde modèle local (`models/*_feast_mlflow.pkl`)

Algorithmes supportés :
- Classification :
  - `logistic_regression`
  - `random_forest_classifier`
  - `xgboost_classifier`
- Régression :
  - `linear_regression`
  - `random_forest_regressor`
  - `xgboost_regressor`

---

## 5. Décisions techniques (et pourquoi)

### 5.1 Pipeline orienté config plutôt que hardcode
- **Pourquoi :** permettre l’ajout de modèles notebook sans modifier le code pipeline à chaque fois.

### 5.2 Génération dynamique des objets Feast
- **Pourquoi :** chaque modèle peut avoir son propre schéma de features et son propre `FeatureService`.

### 5.3 Fallback de features manquantes (désactivable)
- **Pourquoi :** robustesse opérationnelle lors de transitions notebooks -> prod.
- Contrôle qualité conservé via `--strict-features`.

### 5.4 Feature engineering minimal mais ciblé
- **Pourquoi :** compatibilité avec les features réellement observées dans notebooks (`lag*`, `roll*`, one-hot hospital/blood type), sans surcomplexifier.

---

## 6. Validation exécutée

### 6.1 Vérifications techniques
- compilation Python des fichiers modifiés : OK
- affichage `--help` CLI : OK

### 6.2 Exécutions end-to-end

Commandes testées :

```bash
python -m pios_ml_backend.main mlops-train-notebooks --model-id donor_prediction --skip-materialize
python -m pios_ml_backend.main mlops-train-notebooks --model-id xgb_demand_forecast_j+30 --skip-materialize
python -m pios_ml_backend.main mlops-train-notebooks --model-id donor_prediction --model-id xgb_demand_forecast_j+30 --skip-materialize
```

Résultats observés :
- runs MLflow créés avec succès
- artifacts modèles générés
- fichiers Feast dynamiques appliqués
- résumé d’exécution écrit

Exemple de `run_id` obtenus :
- `donor_prediction` : `d03996f3cdad4aaa879994cfd64fbd51`
- `xgb_demand_forecast_j+30` : `1b09054c0761460e84b8ce528e5ad0b7`
- run combiné multi-modèles : succès également

---

## 7. Sorties générées

- Résumé runs :
  - `/Users/mac/Documents/pios-1/ml-backend/artifacts/mlflow_notebook_runs.json`
- Features Feast générées :
  - `/Users/mac/Documents/pios-1/ml-backend/feature_repo/notebook_generated_feature_definitions.py`
- Parquet features :
  - `/Users/mac/Documents/pios-1/ml-backend/feature_repo/data/notebook_donor_prediction_features.parquet`
  - `/Users/mac/Documents/pios-1/ml-backend/feature_repo/data/notebook_xgb_demand_forecast_j_30_features.parquet`
- Modèles entraînés :
  - `/Users/mac/Documents/pios-1/ml-backend/models/donor_prediction_feast_mlflow.pkl`
  - `/Users/mac/Documents/pios-1/ml-backend/models/xgb_demand_forecast_j_30_feast_mlflow.pkl`

---

## 8. Utilisation (opérationnelle)

Depuis :

```bash
cd /Users/mac/Documents/pios-1/ml-backend
```

Commande standard :

```bash
python -m pios_ml_backend.main mlops-train-notebooks
```

Options fréquentes :

```bash
# subset de modèles
python -m pios_ml_backend.main mlops-train-notebooks \
  --model-id donor_prediction \
  --model-id xgb_demand_forecast_j+30

# mode strict sur les features
python -m pios_ml_backend.main mlops-train-notebooks --strict-features

# enregistrement model registry MLflow
python -m pios_ml_backend.main mlops-train-notebooks --register-models
```

---

## 9. Limites actuelles

- le pipeline couvre les patterns features observés dans les notebooks actuels ; des features plus exotiques peuvent nécessiter extension
- warning MLflow sur `artifact_path` déprécié (non bloquant)
- sérialisation sklearn via pickle/cloudpickle (warning sécurité standard MLflow)

---

## 10. Conclusion

L’objectif d’intégration est atteint : les modèles notebook sont maintenant industrialisables via un pipeline unique, dynamique, traçable et exécutable en une commande.

Cette base permet de faire évoluer rapidement le nombre de modèles, les features et les stratégies d’entraînement, tout en conservant Feast comme couche feature store et MLflow comme couche de suivi/versioning.

---

