# Configuration des Modèles de Don de Sang : Documentation

**Date :** 04 Février 2026  
**Matériel :** CPU (inférence scikit-learn)  
**Objectif :** Utiliser deux modèles entraînés sur un snapshot donneur : (1) probabilité de don dans les 30 jours et (2) nombre de jours estimé avant le prochain don.

---

## 1. Résumé Exécutif

Cette configuration définit un **contrat d’entrée unique** (mêmes features pour tout le monde) et deux modèles :

- **Modèle A (Classification) :** `donation_model_30d.pkl`  
  Sort une **probabilité** `P(don dans 30 jours)` dans l’intervalle `[0, 1]`.

- **Modèle B (Régression) :** `days_to_next_donation_reg.pkl`  
  Sort une **estimation** du **nombre de jours jusqu’au prochain don** (valeur `>= 0`).

Points clés :

- Les deux modèles consomment **les mêmes 5 features**.
- La configuration applique une **validation stricte** (champs requis, types, colonnes inconnues).
- Une couche de **post-traitement** standardise la sortie (seuil pour la classification, clipping pour la régression).

---

## 2. Vue d’Ensemble de `config.yaml`

Le fichier `config.yaml` est structuré en 4 blocs :

1. **Schéma d’entrée commun** : définition des features attendues (nom, type, contraintes)
2. **Liste des modèles** : deux entrées (classifier + régresseur)
3. **Runtime** : chargement au démarrage, cache, logs
4. **Validation** : règles pour rejeter les entrées incorrectes

Ce découpage permet :

- d’ajouter un troisième modèle sans changer les inputs
- de garder une inférence reproductible et stable

---

## 3. Contrat d’Entrée (Schéma Commun)

### Format d’entrée attendu

Un **snapshot donneur** = une seule ligne (par exemple un dictionnaire converti en DataFrame à 1 ligne).

### Features requises

| Feature               |  Type | Signification                                           | Contrainte |
| --------------------- | ----: | ------------------------------------------------------- | ---------: |
| `recency_days`        | float | Jours depuis le dernier don                             |      min 0 |
| `frequency_365`       | float | Nombre de dons sur les 365 derniers jours               |      min 0 |
| `time_months`         | float | Mois depuis le premier don                              |      min 0 |
| `days_until_eligible` | float | Jours restants avant éligibilité (0 si éligible)        |      min 0 |
| `overdue_days`        | float | Jours de retard vs cadence attendue (0 si aucun retard) |      min 0 |

### Ordre des features (imposé)

Les deux modèles utilisent exactement l’ordre suivant :

1. `recency_days`
2. `frequency_365`
3. `time_months`
4. `days_until_eligible`
5. `overdue_days`

Remarque : si ton `.pkl` contient déjà le pipeline de prétraitement, l’ordre sert surtout à **valider** et éviter les erreurs silencieuses.

---

## 4. Modèle A : Probabilité de Don à 30 Jours (Classification)

### Identification

- `model_id` : `donation_propensity_30d_clf`
- `task` : `classification`
- `horizon` : `30d`
- `artifact.path` : `donation_model_30d.pkl`
- `framework` : `sklearn`
- `serialization` : `joblib`

### Entrées

- Schéma : `common_input_schema`
- `allow_extra_features: false`
- `missing_values: "reject"` (les NaN sont refusés sauf si tu changes la règle)

### Sorties

- **Sortie principale :** `p_donate_30d` (float `[0, 1]`)
- **Sortie optionnelle :** `class_label` (0/1) calculée via un seuil

### Post-traitement

- `proba_class_index: 1`  
  On prend la proba de la classe positive via `predict_proba()[0, 1]`
- `decision_threshold: 0.50`
  - si `p_donate_30d >= 0.50` → label = 1
  - sinon → label = 0
- `label_mapping` :
  - 0 → `no_donation_30d`
  - 1 → `donation_30d`
- `risk_bands` (lecture rapide) :
  - **low** : `p < 0.30`
  - **medium** : `0.30 <= p < 0.70`
  - **high** : `p >= 0.70`

### Interprétation

- Plus `p_donate_30d` est proche de 1, plus le donneur est susceptible de donner prochainement.
- Le seuil et les bandes sont **paramétrables** selon la stratégie (rappel vs précision).

---

## 5. Modèle B : Jours Jusqu’au Prochain Don (Régression)

### Identification

- `model_id` : `days_to_next_donation_reg`
- `task` : `regression`
- `horizon` : `point_estimate`
- `artifact.path` : `days_to_next_donation_reg.pkl`
- `framework` : `sklearn`
- `serialization` : `joblib`

### Entrées

- Schéma : `common_input_schema`
- Même ordre de features
- Même politique de valeurs manquantes par défaut (`reject`)

### Sorties

- **Sortie principale :** `pred_days_to_next_donation` (float `>= 0`)

### Post-traitement

- `clip_min: 0.0`  
  Si une prédiction sort négative, elle est ramenée à 0.
- `rounding: "none"`  
  Options possibles : `int`, `ceil`, `floor`

### Interprétation

- Valeur faible → don attendu bientôt
- Valeur élevée → don attendu plus tard
- À compléter idéalement par des métriques de validation (MAE/RMSE) pour quantifier l’incertitude

---

## 6. Runtime & Validation

### Runtime

- `load_on_startup: true` : charge les modèles au démarrage
- `model_cache.enabled: true` : garde les modèles en mémoire
- `max_models_in_memory: 4` : limite du cache
- Logs :
  - `log_inputs: false` (meilleur pour confidentialité)
  - `log_outputs: true`

### Validation

- `enforce_feature_types: true` : refuse des types incorrects (ex: texte à la place d’un nombre)
- `reject_unknown_features: true` : refuse les colonnes inconnues
- `reject_missing_required: true` : refuse une entrée si une feature obligatoire manque

---

## 7. Exemple d’Entrée /après Conversion

### Exemple entrée

```json
{
  "recency_days": 45,
  "frequency_365": 2,
  "time_months": 24,
  "days_until_eligible": 0,
  "overdue_days": 10
}
```

### Example sortie

clf:
{
"p_donate_30d": 0.72,
"class_label": 1
}
reg:
{
"pred_days_to_next_donation": 18.4
}
