# File: CONTRIBUTING.md

# Guide de Contribution

## Référence (source officielle)
Les conventions détaillées sont dans `docs/conventions/` :
- `naming-conventions.md`
- `commit-conventions.md`
- `branch-conventions.md`
- `api-conventions.md`

---

## Prérequis
- Node.js 18+ (Web/Mobile)
- Python 3.10+ (IA/ML)
- Docker + Docker Compose (si applicable)
- Git 2.30+

---

## Règles minimales (strict)
- Code en **anglais**, documentation en **français**
- **PR obligatoire** (pas de push direct sur `main`)
- **2 approvals** minimum
- **CI OK** (lint + tests)
- Commits **atomiques** (petits, propres)
- Secrets **interdits** dans le repo (configs, tokens, clés)

---

## Workflow
1. Créer une issue (template GitHub)
2. Créer une branche (convention)
3. Développer + commits atomiques
4. Ouvrir une PR (template rempli)
5. Review (2 approvals) + CI OK
6. Merge + suppression de la branche

---

## Qualité
- Tests requis sur toute logique métier / services / endpoints
- UI : screenshots/vidéo dans la PR si changement visuel
- API : exemples payload + erreurs si endpoint modifié
- IA/ML : version dataset + métriques + reproductibilité (seed/config)

---
