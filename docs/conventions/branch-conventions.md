# File: docs/conventions/branch-conventions.md

# Conventions de Branches

## 1) Branches principales
- `main` : production
- `develop` : intégration des features
- `staging` : pré-production

> Si tu es en trunk-based : garde `main` + branches courtes et supprime `develop/staging`.

---

## 2) Convention de nommage

### Feature
```text
feature/<type>/<ticket>-<short-description>
```
### Fix
```text
fix/<type>/<ticket>-<short-description>
```
### Release
```text
release/vX.Y.Z
```
### Hotfix
```text
hotfix/<short-description>
```
## 3) Types (multi-domaines)

web : frontend web

mobile : app mobile

api : backend / API

ai : IA/ML

infra : DevOps / infra

docs : documentation

data : data pipelines / lakehouse / ETL (optionnel)

## 4) Exemples

### Feature
```text
feature/web/123-add-user-profile
feature/mobile/789-push-notifications
feature/api/404-add-auth-endpoints
feature/ai/202-train-churn-model
feature/infra/88-add-terraform-lock
```
### Fix
```text
fix/web/505-menu-overflow
fix/mobile/606-crash-on-launch
fix/api/707-null-pointer-on-users
fix/ai/808-data-leakage-guard
```
Release / Hotfix
```text
release/v1.2.3
hotfix/critical-login-redirect
```
## 5) Règles de workflow

- Toujours partir de develop (ou main si trunk-based).

- Une branche = un objectif.

- Rebase/merge depuis la base avant PR si branche longue.

- Merge uniquement via PR (pas de push direct).

- Supprimer la branche après merge.