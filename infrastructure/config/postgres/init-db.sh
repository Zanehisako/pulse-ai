#!/usr/bin/env bash
# Database Initialization
# Runs ONCE on first container start
# Env vars injected by docker-compose:
#   POSTGRES_USER, POSTGRES_DB         (built-in postgres vars)
#   CORE_DB_USER, CORE_DB_PASSWORD
#   ML_DB_USER,   ML_DB_PASSWORD

set -e

echo "🐘 Initializing PIOS database..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL

    -- ══════════════════════════════════════════
    --  Extensions
    -- ══════════════════════════════════════════
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    -- ══════════════════════════════════════════
    --  Users (idempotent)
    -- ══════════════════════════════════════════
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$CORE_DB_USER') THEN
            CREATE USER $CORE_DB_USER WITH PASSWORD '$CORE_DB_PASSWORD';
        END IF;
    END
    \$\$;

    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$ML_DB_USER') THEN
            CREATE USER $ML_DB_USER WITH PASSWORD '$ML_DB_PASSWORD';
        END IF;
    END
    \$\$;

    -- ══════════════════════════════════════════
    --  Schemas
    -- ══════════════════════════════════════════
    CREATE SCHEMA IF NOT EXISTS core_schema AUTHORIZATION $CORE_DB_USER;
    CREATE SCHEMA IF NOT EXISTS ml_schema   AUTHORIZATION $ML_DB_USER;

    -- ══════════════════════════════════════════
    --  Permissions: core_user → core_schema (r/w)
    -- ══════════════════════════════════════════
    GRANT ALL PRIVILEGES ON SCHEMA core_schema TO $CORE_DB_USER;
    ALTER DEFAULT PRIVILEGES FOR USER $CORE_DB_USER IN SCHEMA core_schema
        GRANT ALL PRIVILEGES ON TABLES TO $CORE_DB_USER;
    ALTER DEFAULT PRIVILEGES FOR USER $CORE_DB_USER IN SCHEMA core_schema
        GRANT ALL PRIVILEGES ON SEQUENCES TO $CORE_DB_USER;

    -- ══════════════════════════════════════════
    --  Permissions: ml_user → ml_schema (r/w)
    -- ══════════════════════════════════════════
    GRANT ALL PRIVILEGES ON SCHEMA ml_schema TO $ML_DB_USER;
    ALTER DEFAULT PRIVILEGES FOR USER $ML_DB_USER IN SCHEMA ml_schema
        GRANT ALL PRIVILEGES ON TABLES TO $ML_DB_USER;
    ALTER DEFAULT PRIVILEGES FOR USER $ML_DB_USER IN SCHEMA ml_schema
        GRANT ALL PRIVILEGES ON SEQUENCES TO $ML_DB_USER;

    -- ══════════════════════════════════════════
    --  Permissions: ml_user → core_schema (r/o)
    -- ══════════════════════════════════════════
    GRANT USAGE ON SCHEMA core_schema TO $ML_DB_USER;
    ALTER DEFAULT PRIVILEGES FOR USER $CORE_DB_USER IN SCHEMA core_schema
        GRANT SELECT ON TABLES TO $ML_DB_USER;

EOSQL

echo "✅ PIOS schemas and users initialized!"
echo "   👤 $CORE_DB_USER → core_schema (read/write)"
echo "   👤 $ML_DB_USER   → ml_schema (read/write) + core_schema (read-only)"

# ══════════════════════════════════════════
#  Keycloak database
# ══════════════════════════════════════════
echo "🔐 Creating Keycloak database..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL

    SELECT 'CREATE DATABASE keycloak OWNER $POSTGRES_USER'
    WHERE NOT EXISTS (
        SELECT FROM pg_database WHERE datname = 'keycloak'
    )\gexec

    GRANT ALL PRIVILEGES ON DATABASE keycloak TO $POSTGRES_USER;

EOSQL

# ══════════════════════════════════════════
#  MLflow database
# ══════════════════════════════════════════
echo "📊 Creating MLflow database..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL

    SELECT 'CREATE DATABASE mlflow OWNER $POSTGRES_USER'
    WHERE NOT EXISTS (
        SELECT FROM pg_database WHERE datname = 'mlflow'
    )\gexec

    GRANT ALL PRIVILEGES ON DATABASE mlflow TO $POSTGRES_USER;

EOSQL

echo "✅ Everything initialized!"