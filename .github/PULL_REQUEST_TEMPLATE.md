# File: .github/PULL_REQUEST_TEMPLATE.md

## Description
Résumé clair des changements et du pourquoi.

## Type de changement
- [ ] Nouvelle fonctionnalité
- [ ] Correction de bug
- [ ] Refactoring
- [ ] Documentation
- [ ] CI/CD
- [ ] Performance
- [ ] Sécurité

## Domaine affecté
- [ ] Web
- [ ] Mobile
- [ ] API
- [ ] IA/ML
- [ ] Infrastructure
- [ ] Docs
- [ ] Data

## Issue(s) liée(s)
Fixes #___

## Contexte
- Problème initial:
- Solution choisie (résumé):
- Alternatives écartées (optionnel):
- Impacts / risques connus:

## Changements (détails)
- Changement 1:
- Changement 2:
- Changement 3:

## Screenshots / Vidéos (UI)
Si applicable.

## Comment tester
Étapes exactes pour valider la PR.

### Tests manuels
1. ...
2. ...
3. ...

### Tests automatisés
- [ ] Unit tests
- [ ] Integration tests
- [ ] E2E tests
- [ ] Performance tests
- [ ] Security checks
- Commande(s) exécutée(s):
```bash
# ex:
# npm test
# pytest
# make test
```
## Checklist
- [ ] Le code suit les conventions du projet
- [ ] Auto-review effectuée
- [ ] Tests ajoutés/mis à jour (si nécessaire)
- [ ] Documentation mise à jour (si nécessaire)
- [ ] CI passe
- [ ] Aucun secret/config sensible committé
- [ ] Aucune régression connue introduite
- [ ] Logs/metrics ajoutés si pertinent
- [ ] Feature flag utilisé si changement risqué

## Notes pour les reviewers
Risques, edge cases, migrations, rollback.

## Déploiement
- [ ] Migration DB requise
- [ ] Changement de configuration requis
- [ ] Nouvelles variables d’environnement
- [ ] Backward compatible

## Rollback plan (si nécessaire)
- Comment revenir en arrière:
- Données à nettoyer (si applicable):
- Impact rollback:
