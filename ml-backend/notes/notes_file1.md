🩸 **Prédiction de Pénurie & Alerte Précoce (Blood Bank) : Note de Résultats – Notebook “Demand → Stockout Proxy”**
**Date : 31 Janvier 2026**
**Matériel :** CPU (XGBoost – `xgboost`)
**Dataset :** `synthetic_blood_transfusion.csv` (46 752 lignes, 23 colonnes)
**Objectif :** Prédire la **demande future** (`units_used`) à plusieurs horizons (**J+1, J+7, J+30**), puis convertir cette prévision en **alerte risque de rupture** via un **proxy stockout** (stock futur estimé vs `critical_stock`).
_(Format inspiré de la note précédente : résumé exécutif → architecture → features → résultats → conclusions.)_

---

## 1) Résumé Exécutif

Nous avons construit un pipeline “**Time Series Tabulaire**” par **hôpital × groupe sanguin × date** :

- ✅ **Prévision de demande (régression)** stable sur 3 horizons (RMSE ~ 2.36–2.38 unités).
- ✅ **Alerte pénurie (proxy)** en post-traitement avec de très bonnes métriques, surtout à court terme :
  - **J+1 :** Precision **0.944**, Recall **0.982**, F1 **0.963**
  - **J+7 :** Precision **0.902**, Recall **0.937**, F1 **0.919**
  - **J+30 :** Precision **0.879**, Recall **0.917**, F1 **0.897**

---

## 2) Pipeline (du Notebook)

### A) Préparation & anti-fuite (data leakage)

Certaines colonnes “trop proches de la vérité future” ont été supprimées **avant entraînement** :

- `stockout_next_day`, `heavy_demand`, `heavy_demand_next_day`

👉 Objectif : éviter que le modèle “triche” en apprenant une cible quasi-directe.

### B) Feature engineering (past-only)

Création de variables historiques **par (hospital, blood_type)** :

- Lags demande : `lag1`, `lag7`, `lag30`
- Moyennes glissantes demande : `roll7`, `roll30`
- Variables de contexte : calendrier (`dow`, `weekend`, `month`, `holiday`), météo (`temp_c`, `rain_mm`), stress système (`flu_index`, `trauma_cases`, `scheduled_surgeries`, `donation_campaign`, `supply_shock`)
- Variables supply “connues à t” (option A du notebook) : `stock_start`, `units_collected`, `wastage`

### C) Split temporel (réaliste)

Découpage **train/test** sur le temps (pas de shuffle) :

- Split à **80% quantile date** → **split ≈ 2023-03-14**

---

## 3) Modèle entraîné

**Modèle :** `XGBRegressor` (objectif `reg:squarederror`)
Hyperparams (du notebook) :

- `n_estimators=600`, `learning_rate=0.05`, `max_depth=6`
- `subsample=0.8`, `colsample_bytree=0.8`
- `random_state=42`, `n_jobs=-1`

---

## 4) Résultats — Prévision de la Demande (units_used)

| Horizon  |      RMSE |       MAE |     sMAPE |      WAPE |
| -------- | --------: | --------: | --------: | --------: |
| **J+1**  | **2.359** | **1.904** | **58.15** | **40.74** |
| **J+7**  | **2.371** | **1.908** | **58.28** | **40.87** |
| **J+30** | **2.384** | **1.923** | **58.61** | **41.22** |

✅ **Sanity check (important)** : quand on mélange la cible (shuffle), le RMSE monte à **~3.17** (nettement pire) → preuve que le modèle apprend un vrai signal, pas du hasard.

---

## 5) Alerte “Stockout” (Proxy basé sur critical_stock)

### Idée (post-processing)

On transforme la demande prédite en un stock futur estimé :

[
\widehat{stock}_{t+H} = stock_end(t) + \widehat{collect}(t) - \widehat{waste}(t) - \widehat{demand}(t+H)
]

Dans le notebook :

- (\widehat{collect}(t)) ≈ moyenne glissante 7 jours (`col_roll7`)
- (\widehat{waste}(t)) ≈ moyenne glissante 7 jours (`was_roll7`)
- Vérité terrain : `stock_end_future_H = stock_end(t+H)`
- Alerte stockout : `stock_end_future_H < critical_stock(t)`

### Performances (classification proxy)

| Horizon  | Precision |    Recall |        F1 |
| -------- | --------: | --------: | --------: |
| **J+1**  | **0.944** | **0.982** | **0.963** |
| **J+7**  | **0.902** | **0.937** | **0.919** |
| **J+30** | **0.879** | **0.917** | **0.897** |

**Lecture métier :**

- À **J+1**, l’alerte capte presque tous les cas à risque (**recall 98%**) tout en restant précise (**94%**).
- À horizon **J+30**, c’est normal que ça baisse : l’incertitude augmente (supply/demand changent).

---

## 6) Interprétabilité

Le notebook génère un graphique **Top 10 Feature Importance (XGBoost J+1)** pour comprendre quelles variables pilotent la prédiction (utile pour le rapport/mémoire et pour convaincre un encadrant).

---

## 7) Limites & améliorations recommandées

1. **Proxy stockout** = approximation :
   - on utilise des moyennes glissantes pour `units_collected` et `wastage` → bien pour un baseline, mais pas optimal.

2. **Améliorations “production-grade”** :
   - Construire un vrai modèle **classification** pour `stockout_next_day` / `stockout_next_7d` (probabilités + calibration).
   - Faire un vrai “rollout” multi-jours : simuler le stock jour par jour (t+1 … t+H).
   - Ajouter une couche décisionnelle : seuils différents par **groupe sanguin**, pénalité plus forte pour O−, etc. (même logique “score métier” qu’on applique dans l’autre note).
