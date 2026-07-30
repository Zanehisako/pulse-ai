# File: .github/ISSUE_TEMPLATE/ai-model-issue.md

---
name: "🤖 Problème IA/ML"
about: Rapporter un problème lié aux modèles IA ou pipelines ML
title: "[AI] "
labels: ["ai/ml"]
assignees: ""
---

## Type de problème
- [ ] Régression performance modèle
- [ ] Data leakage
- [ ] Bug preprocessing
- [ ] Instabilité training
- [ ] Problème déploiement
- [ ] Monitoring / Drift
- [ ] Autre: _____

## Modèle concerné
- Nom:
- Version/tag:
- Emplacement (registry/artifact):

## Contexte
- Domaine métier / cas d’usage:
- Équipe / propriétaire:
- Lien vers pipeline d’entraînement (si applicable):
- Lien vers endpoint / service de serving (si applicable):

## Impact
Qu’est-ce qui est cassé ? Gravité ?
- Niveau de gravité:
  - [ ] S0 (bloquant prod)
  - [ ] S1 (impact majeur)
  - [ ] S2 (impact modéré)
  - [ ] S3 (mineur)
- Portée:
  - [ ] Prod
  - [ ] Staging/Préprod
  - [ ] Dev/Local
- Utilisateurs impactés (approx):
- Depuis quand (date/heure):
- Comportement attendu:
- Comportement observé:

## Métriques affectées
- Accuracy:
- Precision:
- Recall:
- F1:
- Latence:
- Coût:

## Détails des métriques (optionnel mais recommandé)
- Seuil attendu / SLO:
- Valeur observée:
- Période d’observation:
- Segment(s) impacté(s) (ex: région, langue, device, classe):
- Écart (delta) vs baseline / version précédente:

## Dataset
- Version dataset:
- Split (train/test/val):
- Sampling/filters:

## Données / Qualité
- Changement de schéma récent:
  - [ ] Oui
  - [ ] Non
- Données manquantes / outliers:
  - [ ] Oui
  - [ ] Non
- Drift détecté (data / concept):
  - [ ] Oui
  - [ ] Non
- Source(s) de données concernées:
- Feature store / tables / vues concernées:

## Pipeline / Exécution
- Run ID / Job ID:
- Date/heure du run:
- Environnement d’exécution:
- Paramètres/config (si applicable):
- Artefacts générés (model, metrics, logs):

## Étapes pour reproduire
1. Charger le modèle '...'
2. Exécuter sur les données '...'
3. Observer '...'

## Entrée minimale pour reproduire (recommandé)
- Exemple input (JSON/CSV):

## Comparaison avec une version stable (si possible)

- Version stable (baseline):

- Version problématique:

- Diff observé (résumé):

## Logs / Outputs

- Coller logs, stack traces, screenshots.

## Environnement

- Framework (PyTorch/TensorFlow/etc.):

- Version:

- Hardware (CPU/GPU):

- Runtime (Docker/K8s/etc.):

## Déploiement / Serving (si applicable)

- Type de serving:

  - [ ] API

  - [ ] Batch

  - [ ] Streaming

  - [ ] Edge/On-device

- Endpoint / service:

- Version déployée:

- Stratégie de rollout:

  - [ ] Blue/Green

  - [ ] Canary

  - [ ] Full

  - [ ] Autre: _____

## Workaround / Mitigation

- Workaround disponible:

  - [ ] Oui

  - [ ] Non

- Action de mitigation appliquée:

- Rollback effectué:

  - [ ] Oui

  - [ ] Non

## Notes additionnelles

- Liens (dashboards, tickets, docs):

- Hypothèse(s) / pistes:

- Owner(s) à notifier: