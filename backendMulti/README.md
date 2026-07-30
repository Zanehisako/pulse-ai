# 🩸 PIOS Backend — Plateforme Intelligente d'Optimisation Sanguine

> Backend Django unifié intégrant l'authentification, les données laboratoires, les notifications temps réel (WebSocket) et un moteur de prédiction ML orchestré par un LLM local GGUF (xLAM par défaut, autres modèles auto-détectés).

---

## 📋 Table des matières

1. [Prérequis](#-prérequis)
2. [Installation](#-installation)
3. [Configuration de la base de données](#-configuration-de-la-base-de-données)
4. [Lancement du serveur](#-lancement-du-serveur)
5. [Vérification du bon fonctionnement](#-vérification-du-bon-fonctionnement)
6. [Endpoints API ML](#-endpoints-api-ml)
7. [Tests avec Postman / cURL](#-tests-avec-postman--curl)
8. [Panneau d'administration](#-panneau-dadministration)
9. [Architecture du projet](#-architecture-du-projet)
10. [Dépannage](#-dépannage)

---

## ✅ Prérequis

| Outil      | Version requise | Vérification       |
| ---------- | --------------- | ------------------ |
| Python     | >= 3.11         | `python --version` |
| PostgreSQL | >= 14           | `psql --version`   |
| pip        | à jour          | `pip --version`    |
| Git        | installé        | `git --version`    |

> ⚠️ **Espace disque** : Le backend peut charger n'importe quel modèle GGUF local détecté dans le répertoire configuré. Assurez-vous d'avoir assez d'espace disque et de RAM pour le modèle choisi.

---

## 🚀 Installation

### 1. Créer l'environnement virtuel

```bash
cd backendMulti
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -e ".[dev]"
```

## 🗄 Configuration de la base de données

### 1. Créer la base PostgreSQL

```sql
-- Se connecter à PostgreSQL
psql -U postgres

-- Créer la base et l'utilisateur
CREATE DATABASE pios;
CREATE USER admin WITH PASSWORD 'admin';
GRANT ALL PRIVILEGES ON DATABASE pios TO admin;
ALTER DATABASE pios OWNER TO admin;
\q
```

### 2. Variables d'environnement (optionnel)

Par défaut, le serveur utilise ces valeurs. Modifiez-les si votre configuration diffère :

```bash
# Windows (PowerShell)
$env:DB_NAME = "pios"
$env:DB_USER = "admin"
$env:DB_PASSWORD = "admin"
$env:DB_HOST = "localhost"
$env:DB_PORT = "5432"

# Linux / macOS
export DB_NAME=pios
export DB_USER=admin
export DB_PASSWORD=admin
export DB_HOST=localhost
export DB_PORT=5432
```

### 3. Appliquer les migrations

```bash
python manage.py migrate
```

### 4. Importer les configurations ML

```bash
python manage.py import_ml_config
# importe les configurations ML embarquées dans backendMulti
```

### 4.1. Générer automatiquement les exemples de scope

```bash
# Prévisualiser les changements
python manage.py seed_model_examples --dry-run

# Appliquer
python manage.py seed_model_examples
```

### 5. Créer un compte administrateur

```bash
python manage.py createsuperuser
```

Entrez un nom d'utilisateur, email et mot de passe.

---

## ▶️ Lancement du serveur

```bash
python manage.py runserver
```

### Test 1 — Le serveur répond

```bash
curl http://localhost:8000/api/ml/
```

### Test 2 — État de santé

```bash
curl http://localhost:8000/api/ml/health/
```

## 📚 Documentation interactive

Une fois le serveur lancé, vous pouvez tester les endpoints depuis Swagger UI :

```text
http://localhost:8000/api/docs/
```

Le schéma OpenAPI brut est disponible ici :

```text
http://localhost:8000/api/schema/
```

## 🔌 Endpoints API ML

### Tableau récapitulatif

| Méthode | URL                                     | Description                           |
| ------- | --------------------------------------- | ------------------------------------- |
| `GET`   | `/api/ml/`                              | Index de l'API ML                     |
| `GET`   | `/api/ml/health/`                       | État de santé du système              |
| `GET`   | `/api/ml/models/`                       | Liste de tous les modèles ML          |
| `GET`   | `/api/ml/models/{id}/`                  | Détail d'un modèle                    |
| `POST`  | `/api/ml/predict/nl/`                   | **Prédiction en langage naturel** ⭐  |
| `POST`  | `/api/ml/models/{id}/predict/`          | Prédiction avec un modèle spécifique  |
| `POST`  | `/api/ml/config/reload/`                | Recharger les configurations          |
| `GET`   | `/api/ml/orchestrator/status/`          | État de l'orchestrateur LLM           |
| `GET`   | `/api/ml/orchestrator/models/`          | Modèles GGUF disponibles              |
| `PUT`   | `/api/ml/orchestrator/models/selected/` | Changer le modèle orchestrateur       |
| `POST`  | `/api/ml/orchestrator/warmup/`          | Forcer le rechargement du LLM         |

### Autres endpoints existants

| Méthode | URL               | Description                     |
| ------- | ----------------- | ------------------------------- |
| `*`     | `/auth/`          | Authentification                |
| `*`     | `/testUrl/`       | Données laboratoires            |
| `*`     | `/notifications/` | Notifications                   |
| `*`     | `/admin/`         | Panneau d'administration Django |

---

## 🧪 Tests avec Postman

### Test 1 — Prédiction en langage naturel (endpoint principal)

1. Méthode : `POST`
2. URL : `http://localhost:8000/api/ml/predict/nl/`
3. Onglet Body → `raw` → `JSON`
4. Contenu :

```json
{
  "query": "Est-ce qu'un homme de 35 ans avec un BMI de 24 est susceptible de donner du sang ?"
}
```

### Test 2 — Prédiction avec caractéristiques complètes

```json
{
  "query": "Vérifier l'éligibilité du donneur",
  "features": {
    "age": 42,
    "sex": "M",
    "bmi": 26.1,
    "recency_days": 30,
    "donation_count_last_12m": 5,
    "blood_type": "A+",
    "is_regular_donor": 1
  },
  "model_id": "donor_v1"
}
```

### Test 3 — Prédiction directe sur un modèle spécifique

Accédez à : **http://localhost:8000/admin/**

Connectez-vous avec le compte superuser créé précédemment.

### Sections disponibles :

| Section                 | Description                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| **ML Models**           | Voir/modifier les configurations des modèles ML. Activer/désactiver des modèles.                 |
| **Prediction Logs**     | Historique complet de toutes les prédictions : utilisateur, requête, résultat, temps de réponse. |
| **Orchestrator Models** | Gérer les modèles GGUF de l'orchestrateur. Les variantes xLAM par défaut et les GGUF locaux auto-détectés sont sélectionnables à chaud. |
| **Notifications**       | Historique des notifications envoyées.                                                           |

### Actions rapides dans l'admin :

- **Désactiver un modèle** : ML Models → décochez `is_active` → Save
- **Changer le modèle orchestrateur** : Orchestrator Models → cochez `is_selected` sur le modèle désiré
- **Consulter les logs** : Prediction Logs → filtrer par date, modèle, succès/échec

---

---

## 📦 Ajouter un nouveau modèle ML

Quand vous ajoutez un nouveau fichier `.pkl` dans le dossier models :

```bash
# 1. Importer la config locale du backend Django
python manage.py import_ml_config

# 2. Générer les exemples automatiquement
python manage.py seed_model_examples

# 3. Redémarrer le serveur
python manage.py runserver
```

### Commandes `seed_model_examples`

| Commande                                                   | Description                         |
| ---------------------------------------------------------- | ----------------------------------- |
| `python manage.py seed_model_examples --dry-run`           | Prévisualiser sans sauvegarder      |
| `python manage.py seed_model_examples`                     | Remplir les modèles sans exemples   |
| `python manage.py seed_model_examples --force`             | Écraser tous les exemples existants |
| `python manage.py seed_model_examples --model-id donor_v1` | Mettre à jour un seul modèle        |

### Détection automatique des tâches

La commande détecte le type de modèle et génère les bons exemples :

| model_id contient              | Tâche détectée       | Exemples générés              |
| ------------------------------ | -------------------- | ----------------------------- |
| `donor`, `agent`, `logistic`   | `donor_individual`   | Prédiction donneur individuel |
| `donation_30d`, `days_to_next` | `donor_horizon`      | Horizon temporel de don       |
| `stockout`, `demand`           | `inventory_forecast` | Prévision stock/demande       |
| autre                          | `general`            | Exécution directe du modèle   |
