# Notes : Orchestrateur dynamique, outils externes et plan de test

**Date** : 7 Mars 2026
**Périmètre principal historique** : ancienne API `ml-backend/pios_ml_backend/api.py`, `ml-backend/pios_ml_backend/config_db.py`
**Périmètre actuel** : backend applicatif `backendMulti/ml/orchestrator/service.py` et endpoints Django `/api/ml/...`

---

## 1. Objectif des changements

L’objectif était de faire évoluer l’orchestrateur pour qu’il ne dépende plus uniquement des modèles ML locaux :

- ajout d’un accès à des outils externes de type base de données, recherche et API LLM
- possibilité pour l’orchestrateur de choisir dynamiquement l’outil le plus adapté
- fallback automatique vers `search` ou `llm_api` si aucun outil métier ne peut répondre
- amélioration du catalogue de modèles pour éviter des sélections absurdes
- migration de la configuration runtime vers un stockage plus propre : **une ligne par modèle**

Le besoin concret derrière ces changements était de mieux répondre à des questions de type :

- statistiques nationales
- agrégations par âge / sexe / race / ethnicité
- évolution temporelle
- questions de stock / supply par groupe sanguin

---

## 2. Problèmes observés avant correction

### 2.1 Mauvais routage des requêtes

Des requêtes comme :

- `How has the percentage of total blood donations by age changed over time in the U.S.?`
- `what's the blood supply of O-`

étaient routées vers des modèles de prédiction ML sans rapport direct avec la question.

Exemples de comportements incorrects observés :

- sélection d’un modèle donneur alors qu’il fallait une base de données ou une recherche
- sélection d’un modèle de forecast alors qu’on demandait un stock courant
- résultat numérique isolé, sans fondement analytique réel

### 2.2 xLAM parfois indisponible dans le process

Dans certains cas, le statut retournait :

- `llm_ready = false`
- `llm_attempted = false`

Ce qui signifiait que le warmup n’avait pas été déclenché dans ce process précis.

### 2.3 Catalogue runtime insuffisant

Le stockage de configuration était incorrect ou trop pauvre :

- descriptions incomplètes
- pas de bons / mauvais exemples exploitables par l’orchestrateur
- structure `ml_model_config` peu adaptée à un vrai catalogue dynamique

### 2.4 `db_tool` absent du vrai plan LLM

Au début, `db_tool` n’était utilisé que comme fallback d’exécution.
Le LLM n’avait pas ce tool dans sa liste de planification, donc il ne pouvait pas le choisir directement.

---

## 3. Ce qui a été ajouté / modifié

## 3.1 Outils externes de premier niveau dans l’orchestrateur

L’orchestrateur historique supportait explicitement :

- `db_tool`
- `search`
- `llm_api`

Ces outils sont maintenant exposés à la planification et à l’exécution.

Concrètement :

- le prompt de planification inclut les outils externes
- `db_tool` peut être choisi directement par xLAM
- `search` et `llm_api` restent disponibles si la question sort du périmètre des modèles

---

## 3.2 Fallback externe dynamique

Une chaîne de fallback externe a été ajoutée avec ordre configurable :

- ordre par défaut : `db_tool,search,llm_api`

Variables principales :

- `PIOS_ORCH_ENABLE_EXTERNAL_FALLBACK`
- `PIOS_ORCH_EXTERNAL_FALLBACK_ORDER`
- `PIOS_ORCH_ENABLE_DB_FALLBACK`
- `PIOS_ORCH_DB_TOOL_NAME`
- `PIOS_ORCH_SEARCH_API_URL`
- `PIOS_ORCH_LLM_API_URL`

Le comportement attendu :

- si un tool métier répond directement, il est utilisé
- sinon l’orchestrateur tente les outils externes dans l’ordre
- si `PIOS_ORCH_LLM_API_URL` n’est pas configurée, le fallback LLM peut utiliser le xLAM local si disponible

---

## 3.3 `db_tool` pour les requêtes de supply / stock sanguin

Un `db_tool` concret a été ajouté pour traiter les requêtes de type stock/supply.

Sources supportées :

- SQLite, si `PIOS_ORCH_DB_SQLITE_PATH` est configurée
- CSV en fallback, notamment :
  - `datasets/synthetic_bloodbank_daily.csv`
  - `datasets/synthetic_blood_transfusion.csv`
  - `datasets/synthetic_blood_transfusion_with_features.csv`

Le tool :

- détecte les requêtes orientées supply / stock
- extrait le groupe sanguin si présent
- agrège les données par date
- retourne un `latest`, une `series` et une `answer`

Variables utiles :

- `PIOS_ORCH_DB_SUPPLY_CSV_PATH`
- `PIOS_ORCH_DB_SQLITE_PATH`
- `PIOS_ORCH_DB_SQLITE_TABLE`
- `PIOS_ORCH_DB_SERIES_LIMIT`

---

## 3.4 Détection du périmètre des modèles via métadonnées

La logique de routage n’est plus uniquement basée sur des mots-clés fixes.

Elle utilise maintenant :

- `description`
- `features`
- `feature_info`
- `examples`
  - exemples `good`
  - exemples `bad`

Cela permet de décider si une requête est dans le périmètre réel d’un modèle.

Effet attendu :

- un modèle donneur ne doit plus répondre à une question d’analytics nationale
- un modèle de forecast ne doit plus être choisi pour un stock courant si `db_tool` est plus adapté

---

## 3.5 Passage à un vrai catalogue runtime “une ligne par modèle”

Le stockage runtime a été restructuré.

Avant :

- structure trop centrée sur un blob JSON global

Maintenant :

- chaque modèle a sa propre ligne dans `ml_models`
- le snapshot JSON reste synchronisé, mais la vérité runtime peut être relue depuis les lignes

Colonnes runtime ajoutées / normalisées :

- `model_id`
- `description`
- `file_path`
- `model_type`
- `features_json`
- `feature_info_json`
- `examples_json`
- `defaults_json`
- `enabled`
- `updated_at`

Cela facilite :

- l’inspection
- la maintenance
- l’évolution du catalogue
- le rechargement dynamique

---

## 3.6 Enrichissement automatique des métadonnées modèles

Une logique d’enrichissement a été ajoutée pour reconstruire un catalogue runtime plus utile à partir de :

- fichiers présents dans le dossier modèles
- catalogue de référence éventuel
- inférence de tâche
- génération d’exemples `good` / `bad`
- construction de `feature_info`

Le but est de donner au LLM et au routeur assez de contexte pour choisir correctement.

---

## 3.7 Initialisation xLAM plus robuste

Le chargement du LLM local a été renforcé :

- warmup au démarrage
- lazy init au premier `run()` si le process n’a pas fait le warmup
- fallback local dans `llm_api` si aucune API externe n’est configurée

Cela évite les cas où :

- `llm_ready = false`
- `llm_attempted = false`

alors qu’un modèle local existe bien sur disque.

---

## 3.8 Détection directe vs fallback

Un point important a été ajouté pour lire correctement le résultat d’orchestration.

Cas 1 : sélection directe par le plan LLM

- `plan.steps[0].tool == "db_tool"`
- `planner_mode == "llm"`

Cas 2 : modèle d’abord, puis fallback externe

- `plan.steps[0].tool` = modèle ML
- `planner_mode == "llm_with_external_fallback"`

Cas 3 : xLAM indisponible, plan heuristique

- `planner_mode == "fallback"`

Cela permet de distinguer :

- un vrai choix de planification
- une récupération après échec

---

## 3.9 Correction de l’extraction des groupes sanguins

L’extraction NL a été corrigée pour reconnaître correctement :

- `O-`
- `O+`
- `AB-`
- `AB+`
- `O negative`
- `A positive`

Cette correction était nécessaire, car on observait auparavant :

- `blood_type = null`

sur des requêtes pourtant simples comme :

- `what's the blood supply of O-`

---

## 4. Fichiers principaux concernés

Fichiers backend principaux historiques :

- `ml-backend/pios_ml_backend/api.py`
- `ml-backend/pios_ml_backend/config_db.py`

Fichiers de test historiques ajoutés / mis à jour :

- `ml-backend/tests/test_orchestrator_external_fallback.py`
- `ml-backend/tests/test_runtime_catalog_rows.py`

Travail connexe côté orchestrateur conservé :

- `backendMulti/ml/orchestrator/service.py`

---

## 5. Scénarios de test recommandés

## 5.1 Contexte d’exécution actuel

L’ancienne API FastAPI de `ml-backend` a été supprimée.

Le backend applicatif conservé est désormais `backendMulti`, avec les endpoints Django sous `/api/ml/...`.

Les utilitaires conservés dans `ml-backend` servent au training, au scheduling, au drift monitoring et aux workflows MLOps, mais ne doivent plus être lancés comme backend HTTP principal.

---

## 5.2 Vérifier le statut de l’orchestrateur

```bash
curl http://localhost:8000/api/ml/orchestrator/status/
```

Points à vérifier :

- `llm_ready`
- `llm_attempted`
- `external_fallback.enabled`
- `external_fallback.order`
- `external_fallback.db_enabled`
- `external_fallback.db_csv_exists`

Résultat attendu typique côté backend Django conservé :

- `order = ["db_tool", "search", "llm_api"]`
- `db_enabled = true`

---

## 5.3 Test : supply O- avec sélection directe du `db_tool`

```bash
curl -X POST http://localhost:8000/predict/nl \
  -H "Content-Type: application/json" \
  -d '{"query":"what'\''s the blood supply of O-"}'
```

Après les derniers correctifs, le comportement attendu est :

- `plan.steps[0].tool == "db_tool"`
- `planner_mode == "llm"`
- `execution_results[0].tool == "db_tool"`
- `execution_results[0].output.blood_type == "O-"`

Si le résultat montre :

- `planner_mode == "llm_with_external_fallback"`

alors `db_tool` n’a pas été choisi directement par le plan, mais seulement en secours.

---

## 5.4 Test : analytics nationale hors périmètre d’un modèle donneur

```bash
curl -X POST http://localhost:8000/predict/nl \
  -H "Content-Type: application/json" \
  -d '{"query":"How has the percentage of total blood donations by age changed over time in the U.S.?"}'
```

Comportement attendu :

- l’orchestrateur évite un modèle donneur si les métadonnées indiquent que la requête est hors périmètre
- la réponse passe vers un outil externe plus adapté

Selon la configuration en place :

- `search`
- `llm_api`
- ou autre outil externe futur ajouté au prompt

---

## 5.5 Test : warmup / init LLM

```bash
curl -X POST "http://localhost:8000/orchestrator/warmup?wait=true"
```

Puis :

```bash
curl http://localhost:8000/orchestrator/status
```

Attendu :

- `llm_attempted = true`
- `llm_ready = true` si le modèle local peut se charger

Note :

- même si le warmup n’est pas lancé explicitement, un `run()` doit maintenant pouvoir déclencher une initialisation lazy

---

## 5.6 Test : vérification du catalogue runtime en base

Si le backend de config est SQLite :

```bash
sqlite3 /Users/mac/Documents/pios-1/ml-backend/data/config.db "select model_id, enabled from ml_models order by model_id;"
```

On doit observer :

- une ligne par modèle
- pas un unique enregistrement opaque pour tout le catalogue runtime

Pour vérifier qu’il y a bien du contenu exploitable :

```bash
sqlite3 /Users/mac/Documents/pios-1/ml-backend/data/config.db "select model_id, substr(description,1,80) from ml_models order by model_id;"
```

---

## 5.7 Lancer les tests automatiques

```bash
pytest -q ml-backend/tests/test_orchestrator_external_fallback.py ml-backend/tests/test_runtime_catalog_rows.py
```

Résultat attendu au moment de la rédaction de cette note :

- `5 passed`

Ces tests couvrent notamment :

- fallback externe pour requête analytics
- conservation du chemin modèle pour une vraie requête modèle
- usage du `db_tool` sur une requête supply
- distinction entre sélection directe du `db_tool` et fallback
- persistance row-per-model dans `ml_models`

---

## 6. Comment lire les résultats JSON

Pour interpréter une réponse de `/predict/nl`, il faut regarder en priorité :

### `plan.steps`

Indique ce que le LLM a réellement choisi.

### `execution_results`

Indique ce qui a effectivement été exécuté et avec quel succès.

### `planner_mode`

Interprétation pratique :

- `llm` : le tool a été planifié directement par xLAM
- `llm_with_external_fallback` : xLAM a planifié autre chose, puis l’exécution a basculé vers l’externe
- `external_fallback` : plan initial non exploitable, chaîne externe utilisée
- `fallback` : xLAM indisponible, plan heuristique appliqué

---

## 7. Limites actuelles

- `db_tool` est pour l’instant spécialisé sur le cas supply / stock
- `search` dépend d’une configuration externe si on veut autre chose que la tentative basique actuelle
- `llm_api` externe nécessite une vraie URL si l’on ne veut pas s’appuyer sur le modèle local
- certaines questions analytics nationales nécessiteront encore de nouveaux outils spécialisés ou une vraie base agrégée

---

## 8. Direction future recommandée

Pour rester vraiment dynamique, la bonne direction est :

- garder `db_tool`, `search`, `llm_api` comme outils planifiables
- ajouter d’autres outils externes spécialisés au même niveau
- enrichir les descriptions et exemples par modèle
- ne pas forcer un modèle précis dans le code métier
- laisser le LLM choisir parmi une liste de tools bien décrits, avec garde-fous métadonnées + fallback

En résumé :

- les modèles ML servent aux prédictions ciblées
- les tools externes servent aux questions factuelles, agrégées, documentaires ou analytiques
- l’orchestrateur choisit entre les deux, puis se replie proprement si nécessaire

