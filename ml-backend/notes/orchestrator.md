```
---

**Date de création** : 4 Février 2026  

```
```markdown
# Notes du Projet : Système d'Orchestration Multi-Agents avec LLM

## 1. Vue d'ensemble du projet

Nous avons développé un système complet d'orchestration d'outils (Tool Orchestrator) basé sur Qwen 2.5 1.5B, capable de :
- Sélectionner dynamiquement des outils ML à partir de requêtes en langage naturel
- Exécuter des prédictions avec des modèles sauvegardés en .pkl
- Fournir des explications multilingues (Anglais, Français, Arabe)

---

## 2. Architecture du système

### 2.1 Composants principaux

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Requête User   │────▶│  LLM Orchestrator│────▶│  Model Registry │
│  (Texte libre)  │     │  (Qwen 2.5 1.5B) │     │  (Config + PKL) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────┐          ┌─────────────────┐
                        │  JSON Plan   │          │  ML Model       │
                        │  (Routing)   │          │  (Inference)    │
                        └──────────────┘          └─────────────────┘
                               │                           │
                               └──────────┬────────────────┘
                                          ▼
                                   ┌──────────────┐
                                   │  Explanation │
                                   │  (Langage    │
                                   │   naturel)   │
                                   └──────────────┘
```

### 2.2 Flux de données en deux phases

**Phase 1 - Routage (Technical)** :
- Le LLM extrait les paramètres structurés de la requête utilisateur
- Sortie : JSON strict avec `tool_name` et `arguments`
- Température : 0.01 (déterministe)

**Phase 2 - Explication (Natural)** :
- Le LLM interprète le résultat ML en langage naturel
- Adaptation automatique à la langue de la requête
- Température : 0.3 (créative mais contrôlée)

---

## 3. Entraînement du modèle d'orchestration

### 3.1 Première itération : Single-Step (10k exemples)

**Dataset Generator** : `orchestrator_dataset_generator.py`

Stratégies de nommage variées pour éviter le surapprentissage :
- **Clear** : `MathService`, `WeatherAPI`
- **Snake** : `math_calculator_v1`, `weather_get`
- **Camel** : `weatherService`, `globalProvider`
- **Enterprise** : `AbstractMathProvider`, `GlobalWeatherSingleton`
- **UUID** : `svc-84ba20`, `func-a1`
- **Abstract** : `System_A12`, `Module_5X`

**Format de sortie attendu** :
```json
{
  "plan_type": "single_step",
  "tool_use": {
    "tool_name": "math_calculator_v1",
    "arguments": {
      "expression": "15 * 23 + 5"
    }
  }
}
```

**Résultats** :
- Précision sur la sélection du premier outil : **100%**
- Capacité à généraliser sur des noms abstraits : **Excellente**
- Limite : incapable de gérer les requêtes multi-étapes

### 3.2 Deuxième itération : Tentative Multi-Step (échec partiel)

**Observation** : Le modèle entraîné sur single-step ne généralise pas spontanément vers multi-step.

**Comportement observé** :
- Requête : "Compare weather in Paris and Tokyo"
- Comportement : Sélectionne uniquement `weather_get` pour Paris
- Raison : Biais d'entraînement vers `plan_type: "single_step"`

**Solution nécessaire** : Regénération du dataset avec distribution :
- 40% single_step
- 30% parallel  
- 20% sequential
- 10% conditional

---

## 4. Universal Model Exporter

### 4.1 Motivation

Problème initial : Les modèles custom (comme `PersistentAdamAgent`) ne peuvent pas être inférés sans importer la classe de définition.

Solution : Création d'un **format universel** autonome contenant :
- Les poids du modèle
- Les statistiques de normalisation (moyennes, écarts-types)
- Les encodeurs catégoriels
- L'architecture détectée

### 4.2 Formats supportés

| Framework | Architecture | Méthode d'export | Fichier généré |
|-----------|-------------|------------------|----------------|
| Custom NumPy | `linear_custom` | `export_custom_linear()` | `*_v1.pkl` |
| Scikit-Learn | `sklearn` | `export_sklearn()` | `*.pkl` |
| PyTorch | `pytorch` | `export_pytorch()` | `*.pkl` |
| XGBoost | `xgboost` | `export_xgboost()` | `*.pkl` |

### 4.3 Structure du fichier universel (Custom)

```python
{
    "architecture": "linear_custom",
    "params": {
        "weights": {"recency_days": -0.023, "age": -0.015, ...},
        "intercept": -0.234,
        "means": {"recency_days": 120.5, "age": 42.3, ...},
        "stds": {"recency_days": 45.2, "age": 15.1, ...},
        "encoders": {"sex": ["F", "M"], "blood_type": ["A+", "O-", ...]}
    },
    "metadata": {
        "features": ["recency_days", "age", "bmi", ...],
        "version": "1.0"
    }
}
```

---

## 5. Résultats des tests

### 5.1 Tests de sélection d'outils (Single-Step)

| Requête | Outil sélectionné | Arguments extraits | Statut |
|---------|------------------|-------------------|--------|
| "Calculate 15 * 23 + 5" | `math_calculator_v1` | `{"expression": "15 * 23 + 5"}` | ✅ |
| "Weather in Tokyo" | `weather_get` | `{"city": "Tokyo"}` | ✅ |
| "Translate 'hello' to French" | `System_A12` (abstrait) | `{"t_val": "hello", "l_val": "French"}` | ✅ |
| "Get users from DB" | `db_query` | `{"query_str": "SELECT * FROM users"}` | ✅ |

**Précision globale** : 12/12 (100%)

### 5.2 Tests avec noms abstraits

**Edge Case 1** :
- Outils disponibles : `svc-9x2a1b`, `GlobalWeatherProvider`
- Requête : "Calculate 5 + 10"
- Résultat : Sélectionne `svc-9x2a1b` (correct) ✅

**Edge Case 2** :
- Outil disponible : `System_X1`
- Description : "Translates text"
- Arguments : `{"a": "text", "b": "target_lang"}` (noms abstraits)
- Requête : "Translate 'hello' to French"
- Résultat : Sélectionne `System_X1` avec `{"a": "hello", "b": "French"}` ✅

### 5.3 Tests d'exécution réelle

**Test du modèle de don de sang** :

| Profil | Probabilité prédite | Explication générée |
|--------|--------------------|---------------------|
| Bon donneur (35 ans, récent, régulier) | ~95% | "Oui, vous êtes un excellent candidat!" |
| Mauvais donneur (90 ans, BMI 10, pas récent) | ~10% | "Il est peu probable que vous puissiez donner pour le moment" |

**Multilinguisme** :
- Requête française → Réponse française ✅
- Requête arabe → Réponse arabe ✅

---

## 6. Fichier de configuration (config.json)

Structure pour intégrer un nouveau modèle :

```json
{
  "models": [
    {
      "id": "donor_prediction",
      "description": "Prédit l'éligibilité des donneurs de sang",
      "file_path": "models/donor_v1.pkl",
      "features": ["recency_days", "donation_count_last_12m", "age", "bmi", "sex"],
      "feature_info": {
        "recency_days": {
          "type": "integer",
          "description": "Jours depuis le dernier don",
          "required": true
        },
        "sex": {
          "type": "string", 
          "description": "Sexe (M/F)",
          "required": true
        }
      },
      "examples": [
        {
          "user_query": "Vérifier donneur 35 ans, homme, IMC 24, dernier don il y a 45 jours",
          "extracted_args": {
            "age": 35,
            "sex": "M",
            "bmi": 24,
            "recency_days": 45
          }
        }
      ]
    }
  ]
}
```

---

## 7. Problèmes rencontrés et solutions

### 7.1 Parsing JSON inconsistants

**Problème** : Le modèle génère parfois `kwargs` au lieu de `arguments`.

**Solution** : Couche de normalisation dans `parse_output()` :

```python
def normalize_plan(data):
    if "tool_use" in data:
        return data
    elif "tool_name" in data:
        return {
            "plan_type": "single_step",
            "tool_use": {
                "tool_name": data["tool_name"],
                "arguments": data.get("arguments", data.get("kwargs", {}))
            }
        }
```

### 7.2 Extraction de valeurs extrêmes

**Problème** : Requête "90 year old" → le modèle hésite à extraire 90 (peu commun).

**Solution** : Prompt engineering avec instruction explicite :
```
"If values seem extreme (e.g. Age 90), extract them anyway."
```

### 7.3 Biais de langue dans l'explication

**Problème** : Le modèle répond parfois en anglais même si la requête est en français.

**Solution** : Few-shot examples multilingues dans le prompt :
```
Query: "Puis-je donner?" (Prob: 0.10)
Response: Il semble peu probable...
```

---

## 8. Points clés de réussite

### 8.1 Prévention du surapprentissage (Overfitting)

**Stratégies efficaces** :
- Variation des noms d'outils (6 stratégies différentes)
- Variation des noms d'arguments (`city` → `loc`, `c`, `destination`)
- Ordre aléatoire des outils dans le prompt
- Ajout de distracteurs (outils non pertinents)

### 8.2 Robustesse du parsing

Architecture en deux couches :
1. **Extraction JSON** : Regex + validation JSON
2. **Normalisation** : Gestion des variantes de format

### 8.3 Séparation des responsabilités

| Composant | Rôle | Technologie |
|-----------|------|-------------|
| UniversalInferenceEngine | Exécution ML | NumPy/Pickle |
| ModelRegistry | Gestion des modèles | JSON Config |
| MLOrchestrator | Coordination LLM | Transformers |
| UniversalModelExporter | Portabilité | Pickle standardisé |

---

## 9. Améliorations futures suggérées

1. **Multi-step réel** : Regénérer le dataset avec 30% d'exemples parallel/sequential
2. **Chaînage de dépendances** : Permettre l'utilisation du résultat d'une prédiction comme entrée d'une autre
3. **Validation des arguments** : Ajouter une couche de validation de schéma avant exécution
4. **Cache de prédictions** : Mémoriser les résultats pour les requêtes identiques
5. **Explicabilité ML** : Intégrer SHAP ou LIME pour expliquer les prédictions des modèles opaques

---

## 10. Commandes et utilisation

### Installation des dépendances
```bash
pip install torch transformers peft accelerate bitsandbytes pandas numpy
```

### Entraînement d'un nouveau modèle d'orchestration
```python
# 1. Générer le dataset
python orchestrator_dataset_generator.py  # Génère 10k exemples

# 2. Fine-tuner Qwen
python train_orchestrator.py  # Utilise SFTTrainer avec LoRA
```

### Export d'un modèle custom
```python
from universal_exporter import UniversalModelExporter

exporter = UniversalModelExporter()
exporter.export_custom_linear(
    service_instance=my_trained_service,
    output_path="models/donor_v1.pkl",
    description="Blood donor eligibility predictor"
)
```

### Inférence
```python
orchestrator = MLOrchestrator(
    model_config_path="ml_models/config.json",
    llm_base_model="Qwen/Qwen2.5-1.5B-Instruct",
    llm_adapter_path="/kaggle/working/qwen2.5-orchestrator-10k"
)

result = orchestrator.predict("Can this donor donate? Age 35, BMI 24...")

Ce document couvre l'intégralité du projet, des premiers tests jusqu'à la solution universelle d'export/import, avec les raisons techniques des choix effectués et les résultats obtenus.