# Plan d’exécution (Web + Mobile + ML) — PIOS+ (Extrait)

## 📅 Semaine 1 — Démarrage et fondations (21–26 janvier)

### 🧩 Jour 1 — Kickoff & Squelettes

- **Web (Sujet 1)** : créer le projet React + Vite, structure layout (Header, Sidebar, Dashboard vide), configurer le design system (couleurs, typographie, cards).  
  **👉 But :** établir une base stable pour le tableau de bord.

- **Mobile (Sujet 2)** : init Ionic/Capacitor, créer navigation (Login, Missions, Stocks, Alerts), préparer ApiClient mock.  
  **👉 But :** définir le squelette et navigation hors-ligne.

- **ML (Sujet 3)** : définir cible (ex : demande journalière), horizons (J+1, J+7, J+30), métriques (MAPE, RMSE).  
  **👉 But :** cadrer le problème avant d’écrire du code.

---

### ⚙️ Jour 2 — Authentification & Données simulées

- **Web** : matrice rôles-permissions, écrans Login + Forbidden, AuthContext mock JWT.  
  **👉 But :** RBAC clair dès le départ.

- **Mobile** : login écran + stockage token sécurisé, intercepteur API.  
  **👉 But :** même contrat d’auth que le web.

- **ML** : générateur de données synthétiques 5 ans (dons/centres/produits).  
  **👉 But :** données d’entraînement prêtes sans dépendre du réel.

---

### 📊 Jour 3 — Filtres et stockage local

- **Web** : store global (Zustand/Redux) pour filtres (centre, période, produit), UI de filtres.  
  **👉 But :** base UX pour filtrer partout.

- **Mobile** : base SQLite locale (missions, inventory, outbox).  
  **👉 But :** structure offline-first posée.

- **ML** : baseline = moyenne mobile, export metrics.  
  **👉 But :** point de comparaison minimal.

---

### 🌐 Jour 4 — API Mock et UI connectée

- **Web** : OpenAPI v0 (auth, kpi, alerts, stocks), brancher UI sur mock.  
  **👉 But :** contrat stabilisé.

- **Mobile** : détecter offline/online, lire DB locale, indicateur de fraîcheur.  
  **👉 But :** expérience offline lisible.

- **ML** : service FastAPI `/train`, `/predict` (baseline).  
  **👉 But :** API prête à être branchée.

---

### 🪪 Jour 5 — Audit et synchronisation

- **Web** : journal d’audit (logins/actions), page dédiée admin.  
  **👉 But :** traçabilité conforme.

- **Mobile** : outbox v1 (écriture locale + retry).  
  **👉 But :** sync offline réelle.

- **ML** : feature engineering (lags 7/30/365, rolling mean).  
  **👉 But :** pipeline data prêt.

---

## 📅 Semaine 2 — Backend réel & Flux temps réel (27 janv – 1 févr)

### ⚡ Jour 6
- **Web** : JWT réel, UI masquée selon rôle.  
- **Mobile** : auth réelle + refresh minimal.  
- **ML** : dictionnaire features versionné.  
  **👉 But :** socle sécurisé, reproductible.

---

### 🔍 Jour 7
- **Web** : dashboard KPI (cards, mini trends).  
- **Mobile** : écran Missions, statut, ETA mock.  
- **ML** : script d’entraînement auto + MLflow local.  
  **👉 But :** MVP visualisable + ML industrialisé.

---

### 🌍 Jour 8
- **Web** : simulator Python → Postgres → API → polling UI.  
- **Mobile** : sync pull (différentiel).  
- **ML** : endpoint `/predict` (multi horizon).  
  **👉 But :** flux données vivantes.

---

### 🔔 Jour 9
- **Web** : WebSocket temps réel metrics/alerts.  
- **Mobile** : refresh auto + badge alertes.  
- **ML** : rapport quotidien perf.  
  **👉 But :** réactivité + mesure de qualité.

---

### 🧪 Jour 10
- **Web** : tests e2e smoke.  
- **Mobile** : tests offline mode avion.  
- **ML** : design détection anomalies (Isolation Forest).  
  **👉 But :** cycle stable + vision future.

---

## 📅 Semaine 3 — MVP complet (3 – 8 février)

### 📈 Jour 11
- **Web** : chart tendances + prévisions J+1/7.  
- **Mobile** : scan QR code mock.  
- **ML** : modèle court terme (LightGBM).

### 🚨 Jour 12
- **Web** : alertes v1 (liste, actions).  
- **Mobile** : inventaire entrées/sorties.  
- **ML** : job génération alertes règles simples.

### 🔥 Jour 13
- **Web** : heatmap stocks 24h/7j/30j.  
- **Mobile** : photo preuve transport.  
- **ML** : modèle moyen terme J+8–30.

### 🧭 Jour 14
- **Web** : Gantt collectes planifiées.  
- **Mobile** : signature électronique.  
- **ML** : Prophet (tendance long terme).

### 📤 Jour 15
- **Web** : export CSV/PNG.  
- **Mobile** : notifications locales.  
- **ML** : tracking MLflow.

---

## 📅 Semaine 4 — Temps réel et géospatial (10 – 15 février)

### ⚙️ Jour 16
- **Web** : WS metrics → Dashboard.  
- **Mobile** : sync priorisée (alertes > missions > stock).  
- **ML** : détection anomalies v1.

### 🧭 Jour 17
- **Web** : mode veille urgences.  
- **Mobile** : GPS tracking opt-in.  
- **ML** : seuils dynamiques.

### 🗺️ Jour 18
- **Web** : carte centres Leaflet + couleurs stock.  
- **Mobile** : trajet ETA mock.  
- **ML** : features externes (météo, vacances).

### 📊 Jour 19
- **Web** : heatmap avancée + drill-down.  
- **Mobile** : NFC design + mock.  
- **ML** : stacking v0.

### 🧾 Jour 20
- **Web** : audit exports + actions.  
- **Mobile** : annuaire contacts mock.  
- **ML** : monitoring technique.

---

## 📅 Semaine 5 — Personnalisation & stabilité (17 – 22 février)

- **Web** : widgets drag&drop, cache intelligent, chandeliers stocks.  
- **Mobile** : messagerie v0, OCR design, mode terrain UI.  
- **ML** : entraînement hebdo, drift detection, intervalle de confiance.  
  **👉 But :** MVP complet + UX réelle.

---

## 📅 Semaine 6 — Communication & A/B test (24 – 29 février)

- **Web** : templates reporting, carte heatmap + clusters, comparaison périodes.  
- **Mobile** : messagerie sync v1, trafic mock, mode mains-libres.  
- **ML** : modèle urgence, monitoring prod, calibration régionale.  
  **👉 But :** plateforme complète et robuste.

---

## 📅 Semaine 7 — Logistique & MLOps (3 – 8 mars)

- **Web** : flux logistiques animés, RBAC final, What-if v1.  
- **Mobile** : planification route, OCR réel, journal terrain.  
- **ML** : registry MLflow, alertes techniques ML, tuning v1.  
  **👉 But :** production MLOps fonctionnelle.

---

## 📅 Semaine 8 — Scénarios et actions (10 – 15 mars)

- **Web** : seuils dynamiques UI, isochrones, préfetching.  
- **Mobile** : incidents trajet, mode gants tactiles.  
- **ML** : anomalies autoencoder, recommandations actions, rollback v1.  
  **👉 But :** décisionnel complet.

---

## 📅 Semaine 9 — Optimisation & stabilité (17 – 22 mars)

- **Web** : exports PDF multi-sections, vector tiles, historique SLA.  
- **Mobile** : sync conflits guidée, notifications avancées.  
- **ML** : drift trigger retrain, explicabilité SHAP, audit trail.  
  **👉 But :** pré-livrable stable.

---

## 📅 Semaine 10 — Finition UX & sécurité (24 – 29 mars)

- **Web** : drag&drop v1, paramètres user (langue/unités), hardening sécurité.  
- **Mobile** : notifications push, UI conflits, privacy GPS.  
- **ML** : rollback auto, dashboard DS v1.  
  **👉 But :** produits livrables.

---

## 📅 Semaine 11 — Internationalisation & Release RC (31 mars – 5 avril)

- **Web** : i18n FR/EN, accessibilité WCAG, release candidate.  
- **Mobile** : accessibilité mobile, release candidate.  
- **ML** : fairness check, modèles prod gelés.  
  **👉 But :** release finale prête.

---

## 📅 Semaine 12 — Intégration finale et démo (7 – 12 avril)

- **Web/Mobile/ML** : scénarios de bout-en-bout (stock → alerte → action → audit), documentation complète, tests charge, sécurité finale, polish UX/UI.  
  **👉 But :** version démo complète.

---

## 📅 Semaines 13 – 19 (mi-avril → fin mai) — QA + Démo + Docs + Buffer

- **QA Web/Mobile/ML** : correction de bugs, tests multi-device, scénarios terrain, vérification ML drift.  
- **Documentation** : guides utilisateurs, terrain, ML, model cards.  
- **Démo** : scripts scénarisés (pandémie, grève, campagne).  
- **Packaging** : docker-compose, Makefile, build mobile dev, registry ML.  
- **Revue finale** : sécurité, performance, monitoring.  
- **Buffer** : rattrapage, stabilisation, pré-remise.  
  **👉 But :** projet livré complet, documenté, testable et stable avant fin mai.
