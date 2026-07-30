# File: docs/conventions/commit-conventions.md

# Conventions de Commit (Conventional Commits)

## 1) Format

```text
<type>(<scope>): <description>

[body optionnel]

[footer optionnel]
```
## 2) Types

feat : nouvelle fonctionnalité

fix : correction de bug

docs : documentation

style : formatage (sans changement fonctionnel)

refactor : refactoring

test : ajout/modification de tests

chore : maintenance

ci : configuration CI/CD

perf : amélioration de performance

security : correctif sécurité (optionnel)

## 3) Scopes (multi-domaines)

web : frontend web

mobile : application mobile

api : backend / API

ai : IA/ML

infra : DevOps / infrastructure

docs : documentation

deps : dépendances

data : data pipelines / lakehouse / ETL (si applicable)

## 4) Exemples
feat(web): add dark mode toggle
fix(web): resolve mobile menu overflow

feat(mobile): implement push notifications
fix(mobile): fix crash on iOS 15

feat(ai): add text classifier model
fix(ai): handle missing values in preprocessing

feat(api): add pagination to users endpoint
refactor(api): optimize database queries

ci(infra): add cache for build pipeline
chore(deps): bump react-native version
docs(docs): update conventions for branching

## 5) Règles

Description en anglais

Première ligne : 50–72 caractères max

Utiliser l’impératif : add, fix, remove, update

Commits atomiques (une intention claire)

Référencer les issues si applicable : #123

Breaking changes
BREAKING CHANGE: update auth token format