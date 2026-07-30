# 🩸 Optimisation des Donateurs de Sang par IA : Rapport de Projet

**Date :** 24 Janvier 2026
**Matériel :** NVIDIA Tesla P100 (Accélération GPU)
**Objectif :** Identifier les "Donateurs Idéaux" en prédisant la probabilité individuelle de don et en la combinant avec la valeur biologique et la disponibilité logistique.

---

## 1. Résumé Exécutif (Executive Summary)
Nous avons développé avec succès un **Modèle d'Ensemble (Voting Ensemble)** qui segmente les donateurs en quadrants exploitables. Le système va au-delà du simple filtrage basé sur des règles pour prédire le **comportement humain** (la volonté de donner).

*   **Modèle Final :** CatBoost Classifier (GPU)
*   **Performance :** AUC **0.8114** (Prédiction comportementale uniquement)
*   **Innovation Clé :** Séparation des *Règles d'Éligibilité* (Filtres stricts) du *Score de Propension* (Modèle IA), empêchant ainsi la fuite de données (Data Leakage).

---

## 2. Évolution de l'Architecture et des Modèles

Notre approche a évolué à travers trois phases distinctes. Voici l'analyse comparative des modèles et architectures testés.

### Phase 1 : L'Inadéquation des Séries Temporelles (Abandonnée)
*   **Architecture :** LightGBM (Court terme) + Facebook Prophet (Long terme).
*   **Raison de l'échec :** Prophet est conçu pour des séries temporelles univariées (ex: "Approvisionnement quotidien total"), et non pour scorer 30 000 utilisateurs individuels basés sur des caractéristiques démographiques. Cette approche était inefficace informatiquement et statistiquement inappropriée pour des données transversales.

### Phase 2 : Le Réseau de Neurones Hybride (Itération)
*   **Architecture :** LightGBM (Probabilité) + GRU/RNN (Régression de Volume).
*   **Métriques :**
    *   GRU MSE (Erreur Quadratique Moyenne) : **0.4220** (Réduit depuis 0.81).
*   **Résultat :** Le GRU a réussi à apprendre des modèles temporels, mais la complexité de maintenir deux pipelines séparés (Tabulaire vs Tensoriel) offrait des rendements décroissants par rapport aux arbres de gradient boosting modernes.

### Phase 3 : Le Tournoi de Modèles (Sélection Finale)
Nous avons mené une recherche rigoureuse (Grid-Search) sur le GPU P100.

| Modèle | AUC Initial (Avec Fuite) | AUC Final (Comportement Pur) | Statut |
| :--- | :--- | :--- | :--- |
| **Random Forest** | 0.8556 | ~0.78 | Abandonné (Baseline) |
| **XGBoost** | 0.8608 | 0.8050 | Candidat Solide |
| **LightGBM** | 0.8609 | 0.8090 | Candidat Solide |
| **CatBoost** | **0.8612** | **0.8114** | **VAINQUEUR 🏆** |

**Pourquoi CatBoost ?** Il a géré les variables catégorielles (`pays`, `région`, `groupe_sanguin`) nativement sans nécessiter d'encodage complexe (One-Hot Encoding), maintenant l'AUC la plus élevée sur l'ensemble de validation.

---

## 3. La Crise de "Fuite de Données" et sa Résolution

### Le Problème
Lors de la Phase 3, les modèles atteignaient une AUC de **0.86+**. L'investigation a révélé une **Fuite de Données (Data Leakage)**.
La variable `chronic_condition_flag` (maladie chronique) était parfaitement corrélée avec la cible. L'IA n'apprenait pas *qui veut donner*, elle apprenait simplement *qui est médicalement interdit de don*.

### La Solution
Nous avons fondamentalement changé la stratégie d'entraînement :
1.  **Suppression des variables de fuite :** Retrait de `chronic_condition_flag`, `eligible_to_donate`, et `smoker`.
2.  **Filtrage des données d'entraînement :** Nous avons entraîné le modèle **UNIQUEMENT** sur les donateurs éligibles.
3.  **Résultat :** L'AUC est descendue à **0.8114**.
    *   *Pourquoi est-ce mieux ?* Ce score représente la capacité réelle du modèle à prédire la **psychologie et l'habitude**. Les règles médicales sont désormais appliquées *après* la prédiction de l'IA, agissant comme un filtre strict.

---

## 4. Stratégie d'Ingénierie des Fonctionnalités (Feature Engineering)

Nous avons créé trois catégories de variables pour capturer le contexte complet d'un donateur.

### A. Variables Comportementales (Le Moteur d'Habitude)
Ce sont les prédicteurs les plus forts du don.
*   **`reliability_score` (Score de Fiabilité) :** Une métrique composite de `is_regular_donor` $\times$ `fréquence`.
*   **`donation_velocity` (Vélocité) :** Total des dons à vie divisé par les années d'activité.
*   **`recency_days` (Récence) :** Jours depuis la dernière activité (Le prédicteur n°1 de l'attrition).

### B. Variables Biologiques (Le Moteur de Valeur)
Utilisées pour pondérer le score final, et non nécessairement pour prédire la probabilité.
*   **`donor_versatility_score` (Polyvalence) :** Calculé à partir de la table de compatibilité.
    *   O- (Universel) = 1.0
    *   AB+ (Receveur uniquement) = 0.125
*   **`biological_value` :** Une somme pondérée de la Rareté, de la Polyvalence et de la Prévalence dans le pays.

### C. Variables Environnementales/Logistiques (Le Moteur de Réalité)
*   **`weather_suitability` (0.0 - 1.0) :** Récupéré via l'**API Open-Meteo**.
    *   Convertit la Pluie/Température brute en une pénalité probabiliste (ex: Tempête de neige = 0.1, Ensoleillé = 1.0).
*   **`is_holiday_week` :** Indicateur binaire pour les périodes de forte indisponibilité.
*   **`distance_km` :** Simulation de la friction logistique basée sur l'urbanicité.

---

## 5. Logique de Score du “Donateur Idéal”

Le résultat final n’est pas juste une probabilité.  
C’est une métrique métier composite calculée comme suit :

$$
Score_{Final} = P(\text{Volonté}) \times Valeur_{Bio} \times Adéquation_{Météo} \times Filtre_{Éligibilité}
$$

Où :

- **$P(\text{Volonté})$** :  La probabilité (0–1) issue de CatBoost.

- **$Valeur_{Bio}$** :  La rareté et la polyvalence du sang  *(ex. : O− reçoit un multiplicateur supérieur)*.

- **$Adéquation_{Météo}$** :  Pénalise les donateurs pendant les tempêtes  *(éviter de gaspiller des appels)*.

- **$Filtre_{Éligibilité}$** :  Réduit le score à zéro pour toute personne actuellement différée.


---

## 6. Résultats de la Segmentation Finale

En utilisant le modèle final, nous avons segmenté la base de donateurs en quatre quadrants stratégiques :

1.  **🏆 Héros (VIPs) :** Haute Probabilité / Haute Valeur Biologique.
    *   *Action :* Conciergerie personnelle, appels immédiats.
2.  **☁️ Réserve Fiable :** Haute Probabilité / Basse Valeur Biologique (ex: A+).
    *   *Action :* Maintenance automatisée par SMS.
3.  **💎 Géants Endormis (Sleeping Giants) :** Basse Probabilité / Haute Valeur Biologique (ex: O-).
    *   *Action :* **Cible Marketing Prioritaire.** Ce sont les leads les plus précieux à "réveiller".
4.  **zzz Grand Public :** Basse Probabilité / Basse Valeur.
    *   *Action :* Ignorer / Basse priorité.

### Résumé des Métriques
*   **AUC Finale :** 0.8114
*   **Taux de Faux Positifs :** Faible (Le modèle est conservateur concernant la disponibilité).
*   **Top Fonctionnalité :** `donation_count_last_12m` (Comportemental).

---

## 7. Conclusion
Le projet a réussi à passer d'une prédiction "Boîte Noire" à une stratégie segmentée et interprétable. En résolvant le problème de fuite de données et en intégrant des données météorologiques externes, le système optimise désormais pour la **réalité logistique** plutôt que pour une simple disponibilité théorique.



# 📝 Note d'Analyse : Impact des Features Logistiques & Socio-Démographiques
**Date :** 26 Janvier 2026

### 1. Ce qui a été ajouté (Mise à jour "Monde Réel")
Nous avons enrichi le modèle avec des données opérationnelles concrètes (Datasets J & K) :
*   **Logistique (Last Mile) :** `center_distance_km`, `travel_time_min`, `transfer_cost_usd`, `route_feasibility`.
*   **Contexte Socio-économique :** `area_deprivation_index`, `area_education_index`.
*   **Environnement :** `weather_suitability` (Météo temps réel).

### 2. Observation des Performances (IA)
*   **Stabilité de l'AUC :** Le modèle a atteint un AUC de **0.8179**, une légère amélioration par rapport à la version précédente (0.8114).
*   **Domination de l'Habitude :** Comme vous l'avez noté, les graphiques d'importance des features ("Feature Importance") montrent que **`donation_count_last_12m`** et **`recency_days`** écrasent tout le reste.
*   **Classement des Nouvelles Features :** `center_distance_km` et `weather_suitability` apparaissent en bas de liste, avec un impact marginal sur la *prédiction pure*.

### 3. Pourquoi les features "Habitude" dominent-elles ? (Psychologie vs Logistique)
C'est un phénomène classique en modélisation comportementale :
*   **La Loi du Comportement Passé :** La meilleure façon de prédire si quelqu'un va donner demain est de savoir s'il a donné hier. Un donneur engagé (Habitude forte) viendra même s'il pleut ou s'il habite à 20km.
*   **La Friction vs Le Blocage :** La distance et la météo sont des **frictions** (ça rend l'action plus dure), mais ce ne sont pas des blocages absolus pour un "Héros". L'IA a correctement appris que la motivation interne (habitude) surpasse les obstacles externes (logistique).

### 4. L'Importance "Cachée" des Nouvelles Features (Impact Business)
Si l'IA juge ces features "mineures" pour prédire la probabilité, elles sont **majeures** pour la rentabilité (ROI).

Nous les utilisons dans la **Couche de Décision (Post-Processing)** via le score final, et non seulement dans la prédiction brute.

| Feature | Importance Prédictive (IA) | Importance Opérationnelle (Business) |
| :--- | :--- | :--- |
| **Recency / Donation Count** | ⭐⭐⭐⭐⭐ (Critique) | Indique **QUI** est susceptible de dire "Oui". |
| **Distance / Coût Transfert** | ⭐⭐ (Faible) | Indique **COMBIEN** cela coûte de récupérer ce sang. |
| **Météo** | ⭐ (Faible) | Indique **QUAND** appeler (timing tactique). |

### 5. Conclusion Stratégique
Le modèle IA nous donne la **Volonté (Willingness)**, mais les nouvelles features logistiques nous donnent la **Faisabilité (Feasibility)**.

> **Exemple Concret :**
> *   **Donneur A :** Probabilité 90%, Distance 2km.
> *   **Donneur B :** Probabilité 90%, Distance 50km.
>
> Pour l'IA (CatBoost), ces deux donneurs sont presque identiques (car ils ont la même habitude).
> Mais pour votre **Budget Logistique**, le Donneur A est 10x plus rentable.
>
> **C'est pourquoi le `Ideal_Donor_Score` final est supérieur au simple `AI_Probability`. Il intègre ces contraintes "mineures" pour l'IA mais "majeures" pour le portefeuille.**



# 📑 Notes Techniques : Expérimentation Apprentissage en Ligne & Sélection de Features

**Date :** 27 Janvier 2026
**Sujet :** Transition vers l'Apprentissage en Ligne (Online Learning) et analyse de l'impact des interactions complexes.

---

## 1. Objectifs de l'Expérimentation
Nous avons testé une architecture d'**Apprentissage en Ligne (Online Learning)** pour répondre à deux hypothèses :
1.  **Adaptabilité :** L'agent peut-il apprendre en temps réel quelles données sont utiles sans réentraînement massif ?
2.  **Sélection Automatique (Gatekeeper) :** L'agent peut-il rejeter automatiquement des features coûteuses (Météo, Logistique) si elles n'améliorent pas la prédiction ?
3.  **Interactions Complexes :** Est-ce que croiser les données (ex: *Âge × Météo*) améliore la performance ?

---

## 2. Analyse des Résultats

### A. Le "Gatekeeper" (Agent Simple avec Régularisation L1)
*   **Résultat :** ✅ **Succès.**
*   **Observation :** L'agent a drastiquement simplifié le modèle. Sur ~18 features injectées, il n'en a gardé que 2 ou 3 principales.
*   **Performance :** AUC stable (~0.81) malgré la suppression de 80% des données.
*   **Découverte Clé :** Les features logistiques (`center_distance_km`, `weather`, `cost`) ont été rejetées (poids mis à 0).
    *   *Interprétation :* Ces facteurs influencent le **coût** de l'opération, mais pas la **volonté psychologique** du donneur. L'IA a correctement identifié qu'elles étaient du "bruit" pour la prédiction de l'acte de don.

### B. L'Agent à Interactions (Agent Complexe)
*   **Résultat :** ❌ **Échec (Rendements décroissants).**
*   **Observation :** Nous avons forcé l'agent à chercher des interactions entre features et à protéger celles impliquées.
*   **Performance :** L'AUC a stagné entre **0.72 et 0.78**, inférieur au modèle simple.
*   **Problème :**
    1.  **Explosion de complexité :** Le modèle est passé de 3 paramètres à plus de 25 interactions instables.
    2.  **Sur-apprentissage (Overfitting) :** Les lignes de poids "hachées" montrent que le modèle chassait du bruit statistique plutôt que des signaux réels.
    3.  **Features Zombies :** Le mécanisme de "Bouclier de protection" a empêché l'élagage de features faibles, gardant le modèle inutilement lourd.

---

## 3. Comparatif des Métriques

| Architecture | AUC (Précision) | Complexité (Features actives) | Stabilité | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **CatBoost (Baseline)** | **0.8179** | Moyen (Toutes features) | Haute | Référence solide. |
| **Online Gatekeeper (Simple)** | **~0.8100** | **Très Basse (2-3 features)** | **Haute** | **Architecture Optimale (Efficiente).** |
| **Online Interactions (Complexe)**| ~0.7400 | Très Haute (25+ params) | Basse | Inutilement lourd & bruité. |

---

## 4. Conclusions Stratégiques & Architecture Finale

L'expérimentation valide le principe du **Rasoir d'Ockham** : *Les modèles les plus simples sont souvent les meilleurs.*

### Pourquoi l'AUC n'a pas augmenté avec plus de données ?
Le comportement de don est dominé par **l'Habitude** (`donation_count`, `recency`). C'est un signal fort et suffisant. Ajouter la météo ou la distance n'ajoute pas d'information sur la *motivation* du donneur, cela n'ajoute que du contexte logistique.

### Recommandation d'Architecture Finale ("Lean Pipeline")

Nous abandonnons l'agent complexe à interactions pour revenir à une architecture hybride stricte :

1.  **Couche de Prédiction (L'IA) :**
    *   **Rôle :** Prédire la *Volonté* (Probabilité).
    *   **Input :** Uniquement les données CRM historiques (`recency`, `frequency`, `age`, `reliability`).
    *   **Modèle :** Gatekeeper Simple ou CatBoost.

2.  **Couche de Décision (Le Business) :**
    *   **Rôle :** Optimiser le *ROI*.
    *   **Input :** Données Logistiques (`weather`, `distance`, `cost`).
    *   **Mécanisme :** Formule mathématique post-prédiction.
    *   *Formule :* `Score = (Probabilité IA) - (Pénalité Distance) - (Pénalité Météo)`.

**Gain Business :** Nous pouvons cesser de payer pour des flux de données externes (Météo/Trafic) pour l'entraînement du modèle IA, et les utiliser uniquement comme filtres au moment de l'appel.

# 📑 Notes Techniques : Migration vers Online Learning Optimisé & Stabilisation

**Date :** 28 Janvier 2026  
**Sujet :** Stabilisation d'un Agent d'Apprentissage en Ligne (Online Learning) pour la Prédiction de Dons de Sang

---

## 1. Contexte & Problématiques Initiales

Nous sommes partis d'une implémentation naive de **Régression Logistique Online** (SGD vanilla) entraînée sur 30 000 échantillons. Trois blocages critiques ont été identifiés :

1.  **Saturation Sinusoïde (Sigmoid) :** Prédictions bloquées à **1.0** (ou 0.0) en raison de features non standardisées (âge=108, distances brutes) provoquant une explosion des logits (z &gt; ±25).
2.  **Fuite de Données (Data Leakage) :** Présence de features d'identification (`donor_id`, `donor_id`) et de cibles dérivées (`next_6m_donation_count`, `donation_propensity_score`) créant un overfitting parfait (AUC artificiel de 0.99 puis crash à 0.50).
3.  **Performance :** Vitesse d'entraînement trop faible (~150 lignes/sec) pour du vrai temps réel, avec un élagage trop agressif réduisant le modèle à 4 features sur les 30 disponibles.

---

## 2. Solutions Techniques Apportées

### A. Optimisation Adam + Régularisation L2 (Poids & Biais)
*   **Changement :** Remplacement du SGD vanilla par l'optimiseur **Adam** avec découplage des pénalités L1 (sparsité) et L2 (dérive).
*   **Résultat :** Résolution de la dérive de l'intercept (passé de **4.77** à **0.03**) et des poids (réduction de 34.0 à des valeurs sensibles ~2-5).
*   **Impact :** L'intercept reste centré sur le prior du dataset (base rate ~30%), permettant des prédictions calibrées (0.08 pour mauvais donneur, 0.99 pour bon).

### B. Standardisation en Ligne (Welford)
*   **Changement :** Implémentation de l'algorithme de **Welford** pour le calcul streaming de moyenne et variance (z-score) à chaque batch.
*   **Résultat :** Toutes les features (âge, BMI, counts) sont normalisées en temps réel sans nécessiter de passage dataset complet préalable.
*   **Impact :** Élimination totale de la saturation. La sigmoïde reçoit des inputs dans [-5, 5] au lieu de [-800, +108], stabilisant l'apprentissage.

### C. Protection Anti-Fuite & Filtrage Sémantique
*   **Changement :** Création d'une couche de validation `_is_valid_feature()` blacklistant automatiquement :
    *   Les IDs (`donor_id`, `uuid`, `id`)
    *   Les dates brutes (`as_of_date`, `last_donation_date`) — remplacées par des délais (recency)
    *   Les targets dérivées (`next_6m_*`, `propensity_score`, `eligibility_status`)
*   **Résultat :** Suppression explicite de 9 colonnes problématiques avant entraînement.
*   **Impact :** L'AUC se stabilise à **0.80** (réaliste) au lieu de 0.99 (fictif) ou 0.50 (collapse après overfit).

### D. Injection Dynamique de Features (File d'Attente)
*   **Changement :** Passage d'un système "toutes features dès le départ" à une **file d'attente** (queue) injectant une nouvelle feature tous les **2000 steps**.
*   **Mécanisme :** Les 5 features CRM historiques démarrent immédiatement ; les 15 features contextuelles (géo, rareté, démographie) arrivent progressivement.
*   **Impact :** Permet au standardizer de s'adapter feature par feature et évite le "choc" d'ajout massif de dimensions nulles au step 0.

### E. Accélération Computationnelle (10x)
*   **Optimisations :**
    *   Remplacement de `df.iterrows()` par `df.to_dict('records')` (vectorisation Python)
    *   Mise en cache des features actives en variables locales (évitement de lookups dict)
    *   Réduction de fréquence des mises à jour corrélations (toutes les 50 steps vs 1)
    *   Skip des calculs interactions fréquentiels (toutes les 5 steps)
*   **Résultat :** Passage de **~150 lignes/sec** à **~2 500 lignes/sec** (18s pour 30k échantillons vs 206s initialement).

---

## 3. Comparatif Avant/Après Optimisation

| Métrique | Version Initiale (SGD) | Version Optimisée (Adam) | Impact |
|:---|:---:|:---:|:---:|
| **Vitesse d'entraînement** | ~150 row/s | **~2 500 row/s** | **x16 plus rapide** |
| **Intercept final** | 4.77 (saturé) | **0.03 (neutre)** | Calibration corrigée |
| **Features actives** | 4 (sur-élagage) | **19 (stables)** | Richesse préservée |
| **AUC sur test** | 0.50 (collapse) / 0.99 (fuite) | **0.80 (stable)** | Généralisation valide |
| **Prédiction mauvais donneur** | 0.993 (erreur) | **0.082 (correct)** | Logique métier respectée |
| **Prédiction bon donneur** | 1.000 | **0.998** | Saturation évitée |

---

## 4. Rejet de l'Approche PPO

Une suggestion d'évolution vers **PPO (Proximal Policy Optimization)** — typique des Reinforcement Learning — a été évaluée et **rejetée**.

**Justification :**
*   *Nature du problème :* Notre cas est de la **prédiction supervisée** (label immédiat `donated_next_6m` disponible), pas de la prise de décision séquentielle ( MDP avec rewards différés).
*   *Sur-ingénierie :* PPO nécessite des episodes, des buffers de trajectoires, et un environnement simulé (coûts, stocks sanguins) qui n'existent pas dans notre flux de données historiques.
*   *Résultat :* L'Online Logistic Regression optimisée est l'état de l'art pour le streaming supervisé (type FTRL-Proximal utilisé chez Google/FB pour l'ad targeting).

---

## 5. Architecture Finale Validée

### Le Pipeline "Lean & Fast"

#### Couche 1 : Prétraitement Défensif (Input)
*   **Blacklist sémantique :** Suppression auto des IDs, dates brutes, et scores dérivés.
*   **Encodage :** LabelEncoder évolutif (gestion des nouvelles catégories "UNK").
*   **Standardisation :** Z-score online (Welford) avec clipping à ±5σ.

#### Couche 2 : Modèle Online (Core)
*   **Algorithme :** Régression Logistique + Adam (β1=0.9, β2=0.999).
*   **Régularisation :** L1=0.0001 (sparsité légère), L2=0.01 (poids), L2_intercept=0.5 (biais).
*   **Injection :** Queue de 15 features secondaires injectées toutes les 2000 observations.
*   **Persistence :** Sauvegarde pickle incluant les états Adam (m/v), les encoders catégoriels, et les stats du standardizer.

#### Couche 3 : Inférence Continue (Production)
*   **Mode Inference :** Lecture seule, utilisation des stats du standardizer figées.
*   **Mode Learning :** Mise à jour incrémentale possible (partial_fit) avec auto-injection de nouvelles features si disponibles.

---

## 6. Conclusions & Gains Business

1.  **Robustesse :** Le modèle ne sature plus (pas de 0.0 ou 1.0 artificiels) et gère correctement les nouveaux donneurs (cold start via standardizer online).
2.  **Frugalité :** Pas besoin de réentraîner sur tout l'historique. Les nouvelles données s'intègrent en temps réel (12 secondes pour ingérer 30k lignes).
3.  **Transparence :** 19 features actives sont interprétables (top : `recency_days`, `donation_count_last_12m`, `deferral_reason`) vs "boîte noire" complexe.

**Recommandation :** Déployer cette architecture en remplacement du batch training mensuel. La capacité à injecter des features (météo, événements locaux) sans redémarrage l'adapte aux données temps réel.