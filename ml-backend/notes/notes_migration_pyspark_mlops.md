# Notes – Migration MLOps vers PySpark DataFrames

**Date :** 15 Février 2026  
**Contexte :** `/Users/mac/Documents/pios-1/ml-backend`  
**Objectif :** remplacer les flux d'entraînement basés sur pandas/scikit-learn par des flux PySpark DataFrames + Spark ML, avec un mode d'exécution documenté.

---

## 1. Résumé des changements

La chaîne d'entraînement MLOps a été migrée vers PySpark pour :
- préparer les datasets via `SparkSession` et DataFrames Spark
- entraîner les modèles avec `pyspark.ml`
- logger/charger les modèles via `mlflow.spark`
- conserver la logique Feast (apply/materialize/get_historical_features)

---

## 2. Fichiers modifiés

- `/Users/mac/Documents/pios-1/ml-backend/pios_ml_backend/mlops/feast_mlflow_pipeline.py`
  - préparation des données en Spark (`prepare_feature_data`)
  - entraînement Spark Logistic Regression
  - scoring online avec Spark + `VectorAssembler`
  - logging modèle avec `mlflow.spark`
  - garde-fou Java (refus explicite si Java >= 24)

- `/Users/mac/Documents/pios-1/ml-backend/pios_ml_backend/mlops/notebook_models_pipeline.py`
  - entraînement notebooks migré vers Spark ML
  - split train/test avec `randomSplit`
  - métriques via `MulticlassClassificationEvaluator` / `RegressionEvaluator`
  - sauvegarde locale des modèles Spark (`.sparkml`)
  - garde-fou Java (même logique)
  - fallback `xgboost_*` vers GBT Spark (avec contrôle binaire pour la classification)

- `/Users/mac/Documents/pios-1/ml-backend/pios_ml_backend/main.py`
  - `mlops-prepare` adapté aux DataFrames Spark (`frame.count()`)

---

## 3. Prérequis d'exécution

## 3.1 Java

PySpark utilisé ici doit tourner avec **Java 17 ou 21** (pas Java 25).

Configuration recommandée :

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"
hash -r
java -version
```

## 3.2 MLflow tracking URI

Si votre `mlflow.db` historique contient une révision Alembic incompatible, utilisez une base tracking dédiée :

```bash
export MLFLOW_TRACKING_URI=sqlite:////Users/mac/Documents/pios-1/ml-backend/mlflow_local.db
```

---

## 4. Commandes pour exécuter les nouveaux flux

Depuis `/Users/mac/Documents/pios-1/ml-backend` :

```bash
python -m pios_ml_backend.main mlops-prepare
python -m pios_ml_backend.main mlops-apply
python -m pios_ml_backend.main mlops-materialize
python -m pios_ml_backend.main mlops-train
```

Pour les modèles notebook :

```bash
python -m pios_ml_backend.main mlops-train-notebooks
```

Exemples ciblés :

```bash
python -m pios_ml_backend.main mlops-train-notebooks --model-id donor_prediction
python -m pios_ml_backend.main mlops-train-notebooks --model-id xgb_demand_forecast_j+30
python -m pios_ml_backend.main mlops-train-notebooks --skip-materialize
```

---

## 5. Sorties attendues

- Features parquet Feast :
  - `/Users/mac/Documents/pios-1/ml-backend/feature_repo/data/bloodbank_features.parquet`
- Résumé du dernier run `mlops-train` :
  - `/Users/mac/Documents/pios-1/ml-backend/artifacts/mlflow_last_run.json`
- Résumé des runs notebook :
  - `/Users/mac/Documents/pios-1/ml-backend/artifacts/mlflow_notebook_runs.json`
- Modèles notebook sauvegardés en format Spark :
  - `/Users/mac/Documents/pios-1/ml-backend/models/*_feast_mlflow.sparkml`

---

## 6. Notes de diagnostic

- Warnings `NativeCodeLoader` : informatifs dans ce contexte local.
- Warnings `BrokenPipeError` dans les workers PySpark : peuvent apparaître pendant certaines phases Spark sans invalider le run final.
- Si erreur `Unsupported Java runtime detected (Java 25)` : recharger `JAVA_HOME` sur Java 21/17.
- Si erreur Alembic `Can't locate revision identified by ...` : changer `MLFLOW_TRACKING_URI` vers une nouvelle base SQLite.
