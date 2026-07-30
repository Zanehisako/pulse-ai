# Notes – Mise en place de l’infrastructure ML avec PostgreSQL

## Contexte
L’objectif était de mettre en place une **infrastructure propre et scalable** pour un système incluant :
- un **backend ML** (Python)
- une **base de données partagée**
- une future intégration avec **backend core, web et mobile**

Le système devait supporter :
- des **configurations dynamiques à l’exécution**
- une **séparation claire des responsabilités**
- une architecture défendable dans un **mémoire de master**

---

## Choix d’architecture

### 1. Base de données
- Choix : **PostgreSQL**
- Raison :
  - support des accès concurrents
  - transactions ACID
  - gestion des utilisateurs et des schémas
  - adapté aux systèmes dynamiques

### 2. Principe clé
> **Une base de données partagée, mais des schémas isolés par service**

- Une seule instance PostgreSQL
- Un utilisateur + un schéma par service
  - `ml_user` → `ml_schema`
  - (plus tard : `core_user` → `core_schema`)

Cela garantit :
- isolation
- sécurité
- absence de conflits entre services

---

## Organisation du projet

### Structure globale
```text
pios-1/
├── infrastructure/
│   ├── docker-compose.yaml
│   └── postgres/
│       └── init.sql
├── ml-backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── pios_ml_backend/
│       ├── main.py
│       └── config_db.py
├── backend-core/
├── web-app/
├── mobile-app/
└── shared/
```

- `infrastructure/` : orchestration (Docker Compose, DB)
- chaque service **possède son propre Dockerfile**

---

## Mise en place de PostgreSQL

### Initialisation via `init.sql`
Utilisation du mécanisme officiel `docker-entrypoint-initdb.d`.

Contenu principal :
- création des utilisateurs (`ml_user`)
- création des schémas (`ml_schema`)
- attribution des permissions

```sql
CREATE USER ml_user WITH PASSWORD 'ml_pass';
CREATE SCHEMA ml_schema AUTHORIZATION ml_user;
GRANT ALL PRIVILEGES ON SCHEMA ml_schema TO ml_user;
```

⚠️ Les scripts d’initialisation ne s’exécutent **qu’au premier démarrage** → nécessité de supprimer le volume lors des changements :
```bash
docker compose down -v
```

---

## Docker Compose

### Rôle
- démarrer PostgreSQL
- démarrer le backend ML
- gérer le réseau et l’ordre de démarrage

### Problème rencontré
`depends_on` **ne garantit pas** que PostgreSQL soit prêt.

### Solution
Ajout d’un **healthcheck PostgreSQL** :
```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U admin -d pios"]
  interval: 5s
  timeout: 5s
  retries: 10
```

Puis :
```yaml
depends_on:
  postgres:
    condition: service_healthy
```

---

## Backend ML (Python)

### Packaging
- Utilisation de `pyproject.toml`
- Installation via `pip install .`
- Backend structuré comme **package Python** (`pios_ml_backend`)

### Connexion PostgreSQL
Connexion avec :
- utilisateur dédié (`ml_user`)
- schéma dédié (`ml_schema`)

```python
psycopg2.connect(
    host="postgres",
    dbname="pios",
    user="ml_user",
    password="ml_pass",
    options="-c search_path=ml_schema"
)
```

Cela évite toute écriture accidentelle dans le schéma `public`.

---

## Problèmes rencontrés et solutions

### 1. Erreur TOML
- cause : tableau non fermé dans `pyproject.toml`
- solution : correction de la syntaxe

### 2. Authentification PostgreSQL
- cause : utilisateur inexistant
- solution : création via `init.sql`

### 3. Connexion refusée
- cause : backend démarrait avant PostgreSQL
- solution : healthcheck + attente

### 4. Permission refusée sur `public`
- cause : bonne isolation des schémas
- solution : définir `search_path=ml_schema`

---

## État final

À la fin :
- PostgreSQL démarre correctement
- les utilisateurs et schémas sont créés
- le backend ML se connecte sans erreur
- les tables sont créées dans `ml_schema`
- l’infrastructure est stable et reproductible

Log final attendu :
```
Container pios-postgres Healthy
PIOS ML Backend started with Postgres config DB
```

---

## Conclusion

Cette mise en place fournit :
- une base solide pour l’intégration web/mobile
- une architecture claire et sécurisée
- une infrastructure défendable académiquement

Elle marque la transition entre :
> **un projet de recherche** → **un système logiciel structuré**

---

