🩸 **Prédiction de Pénurie & Alerte Précoce (Blood Bank) : Note de Résultats – Modèles “Stockout” & “Heavy Demand”**
**Date : 24 Janvier 2026**
**Matériel :** CPU (RandomForest – scikit-learn)
**Objectif :** Prédire (J+1) **le risque de rupture de stock** et **les pics de demande**, à partir de données temporelles + contexte (trauma, chirurgie, météo, campagne de dons, etc.).

---

## 1) Résumé Exécutif

Nous avons construit un pipeline complet **Data → Features → Modèle → Alertes** à partir d’un dataset **synthétique réaliste** (hospital × groupe sanguin × jour).
Deux cibles ont été testées :

1. ✅ **Stockout Next Day (rupture J+1)** : modèle **performant et exploitable**

- **ROC AUC : 0.8807**
- **PR AUC : 0.8764**
- **Recall(Stockout) : 0.931** → le modèle détecte **93%** des ruptures (très bon pour la sécurité)

2. ⚠️ **Heavy Demand Next Day (forte demande J+1)** : modèle **faible en détection** (au seuil 0.5)

- **ROC AUC : 0.5797**
- **PR AUC : 0.1901**
- **Recall(HeavyDemand) : 0.001** → quasi aucune détection (modèle “trop conservateur”)

**Conclusion :** le modèle “stockout” est solide. Le modèle “heavy demand” nécessite **révision du label + ajustement du seuil + modèle plus adapté** (boosting).

---

## 2) Données & Hypothèses de Simulation (Dataset Synthétique)

**Granularité :** chaque ligne = **un jour** pour un couple **(hôpital, groupe sanguin)**.
**Variables de contexte utilisées :**

- **Trauma cases** (plus le week-end, pluie, jours fériés) → demande ↑
- **Chirurgies planifiées** → demande ↑
- **Flu index** (hiver) → demande ↑
- **Donation campaign** → collecte ↑
- **Supply shock** (incident logistique/ressource) → collecte ↓
- **Wastage/expiry** → stock ↓
- **Blood type effect** : types rares (-, AB) collectés moins souvent → risque de rupture ↑

---

## 3) Architecture de Modélisation (Pipeline)

**Prétraitement**

- Catégorielles : `hospital`, `blood_type` → **One-Hot Encoding**
- Numériques : imputation médiane

**Feature engineering**

- Lags : `stock_end_lag1`, `units_used_lag1`, `units_collected_lag1`
- Moyennes mobiles (MA7) : `used_ma7`, `collect_ma7`
- Calendrier : `dow`, `weekofyear`, `month`, `holiday`

**Split**

- Split temporel : **80% train / 20% test** (anti-leakage)

**Modèle**

- **RandomForestClassifier** (baseline robuste tabulaire)
- **XGBoost**

---

## 4) Résultats – Modèle 1 : Stockout Next Day ✅

**Taux de positifs :** Train 0.548 / Test 0.566 (événement fréquent dans la simulation)

**Métriques**

- **ROC AUC : 0.8807** (forte capacité de séparation)
- **PR AUC : 0.8764** (excellent)
- **Accuracy : 0.826**

**Confusion Matrix**

- TN=1397, FP=632, FN=183, TP=2460

**Interprétation métier**

- **Recall(rupture)=0.931 :** le modèle **attrape presque toutes les ruptures** → bon pour la sécurité patient.
- **Precision(rupture)=0.796 :** ~20% des alertes sont des “faux positifs” → coût opérationnel (transferts, préparation inutile), mais acceptable si la priorité est de **ne pas rater** une rupture.
- **FN=183 :** ce sont les ruptures “ratées” → c’est la métrique la plus critique à réduire si on vise un système d’alerte.

**Décision recommandée**

- En production, on ajuste le **seuil d’alerte** :
  - seuil ↓ → FN ↓ (plus de sécurité) mais FP ↑
  - seuil ↑ → FP ↓ (moins d’alarmes) mais FN ↑

---

## 5) Résultats – Modèle 2 : Heavy Demand Next Day ⚠️

**Taux de positifs :** ~0.149 (≈ 1 jour / 7)

**Métriques**

- **ROC AUC : 0.5797** (faible)
- **PR AUC : 0.1901** (à peine au-dessus du baseline ≈ 0.149)
- **Accuracy : 0.851** (trompeuse)

**Confusion Matrix**

- TN=3977, FP=0, FN=694, TP=1

**Interprétation métier**

- Le modèle **prédit presque toujours “pas de heavy demand”** (seuil 0.5 trop strict).
- **Recall=0.001 :** il rate pratiquement tous les pics → **non exploitable** pour l’alerte.

**Causes probables**

1. **Seuil 0.5** inadapté (probabilités trop basses)
2. Label heavy demand défini par quantile → bruit/instabilité
3. RandomForest moins performant que boosting pour capter des patterns faibles

**Plan correctif**

- **Baisser le seuil** (0.1–0.3) et choisir un compromis precision/recall
- Redéfinir heavy demand (ex: `mean + 2*std` ou seuil absolu par type)
- Tester **XGBoost/LightGBM/CatBoost** + calibration

---

## 6) Conclusion

- ✅ **Le modèle Stockout J+1 est prêt pour un scénario d’alerte** : performances fortes (AUC ~0.88), excellente couverture des ruptures (recall 0.93).
- ⚠️ **Le modèle Heavy Demand J+1 doit être itéré** : le signal est trop faible au seuil standard, et la définition du label doit être stabilisée + modèle boosting conseillé.
