# Notes: Runtime Config dynamique + test d'intégration live

Date: 18 février 2026

## Objectif

Mettre en place un chargement de configuration modèle **à chaud** (sans redémarrage du service), piloté par PostgreSQL (`LISTEN/NOTIFY`) avec repli périodique, puis valider le flux complet via un test d'intégration qui:

1. démarre l'API historique,
2. entraîne un modèle,
3. met à jour le feature store,
4. recharge la config runtime,
5. utilise immédiatement le modèle entraîné pendant que le service reste actif.

> Note: ce document décrit un flux de validation historique autour de l'ancienne API ML supprimée. Dans l'architecture actuelle, le backend applicatif exposé est `backendMulti`, tandis que `ml-backend` conserve surtout les utilitaires d'entraînement, de scheduling et de support MLOps.

## Changements implémentés

### 1) Configuration runtime pilotée par PostgreSQL

Fichier: `ml-backend/pios_ml_backend/config_db.py`

- Ajout d'une table `ml_model_config` (`config_key`, `config_json`, `updated_at`).
- Ajout d'un trigger PostgreSQL qui émet `pg_notify` à chaque `INSERT/UPDATE/DELETE`.
- Ajout d'helpers pour:
  - lire un snapshot de config runtime,
  - écrire/mettre à jour la config runtime,
  - ouvrir une connexion `LISTEN`.
- Ajout d'un timeout de connexion DB via `DB_CONNECT_TIMEOUT`.
- Maintien d'un repli sur le fichier JSON local si PostgreSQL est indisponible.

### 2) Rechargement dynamique côté API historique

Fichier historique: `ml-backend/pios_ml_backend/api.py`

- `ModelRegistry` rendu thread-safe (verrous) pour éviter les races pendant les reloads.
- Détection de changements via signature config + artefacts modèles.
- Ajout d'un listener background qui:
  - écoute `NOTIFY` (reload immédiat),
  - fait un poll périodique en secours.
- Démarrage/arrêt propre du listener au cycle de vie de l'API historique.
- Ajout d'endpoints runtime:
  - `GET /config/runtime`
  - `PUT /config/runtime`
  - `POST /config/runtime/reload`
- Exposition de l'état runtime config dans `/health`.

Ces éléments documentent le comportement de l'ancienne couche API retirée. Le backend exposé aujourd'hui pour l'application reste `backendMulti`.

### 3) Documentation

Fichier: `ml-backend/README.md`

- Ajout des endpoints runtime config.
- Ajout d'un exemple `PUT /config/runtime`.
- Ajout des variables d'environnement associées:
  - `PIOS_MODEL_CONFIG_KEY`
  - `PIOS_MODEL_CONFIG_CHANNEL`
  - `PIOS_MODEL_CONFIG_POLL_SECONDS`
  - `PIOS_MODEL_CONFIG_RECONNECT_SECONDS`
  - `DB_CONNECT_TIMEOUT`

## Test d'intégration historique

Fichier historique supprimé: `ml-backend/tests/test_live_runtime_model_reload.py`

### Ce que valide le test

- Le service historique démarre et reste disponible.
- Un modèle sklearn est entraîné pendant le test et sérialisé (`.pkl`).
- Un repo Feast éphémère est créé puis `feast apply` est exécuté.
- La config runtime est mise à jour pendant l'exécution du service.
- Le reload runtime est déclenché (`POST /config/runtime/reload`).
- Le modèle devient visible sur `/models`.
- Une prédiction réussit via `POST /models/{model_alias}/predict`.
- Le service est toujours `ok` sur `/health` après ce flux.

### Logs détaillés

Le test inclut désormais des logs structurés (timestamp + niveau) pour chaque étape critique:

- préparation de l'environnement,
- entraînement,
- création/apply Feast (avec stdout/stderr),
- écriture/reload config,
- réponse de prédiction,
- état de santé final.

Exécution avec logs à l'époque:

```bash
python ml-backend/tests/test_live_runtime_model_reload.py
```

ou

```bash
pytest -s -q ml-backend/tests/test_live_runtime_model_reload.py
```

Ce test a depuis été supprimé avec l'ancienne API FastAPI de `ml-backend`.

## Résultat de validation local

Commande exécutée à l'époque:

```bash
pytest -q ml-backend/tests/test_live_runtime_model_reload.py
```

Résultat historique: `1 passed` (avec warnings de dépréciation liés au cycle de vie de l'ancienne API, préexistants).

## Remarques techniques

- En local sans conteneur PostgreSQL, le système bascule sur le fichier JSON (comportement attendu).
- En environnement Docker/compose avec PostgreSQL disponible, le flux `LISTEN/NOTIFY` fournit la propagation immédiate.
