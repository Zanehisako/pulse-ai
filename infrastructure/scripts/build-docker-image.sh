#!/bin/bash
#  PIOS — Start Local Docker Services

set -e
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/infrastructure/docker/docker-compose.local.yaml"
ENV_FILE="$ROOT_DIR/.env.local"

echo "🐳 Starting local Docker services..."

mkdir -p "$ROOT_DIR/ml-backend/models/gguf"
mkdir -p "$ROOT_DIR/mlflow_artifacts"

if [ ! -f "$ENV_FILE" ]; then
  echo "📄 Creating .env.local from .env.example..."
  cp "$ROOT_DIR/.env.example" "$ENV_FILE"
    echo "⚠️  Fill KEYCLOAK_CLIENT_SECRET in .env.local, then run this command again."
  echo "   Find it in Keycloak: realm pios → Clients → django-backend → Credentials."
  echo "   Or search infrastructure/config/keycloak-export/pios-realm.json for clientId django-backend."
  exit 1
fi

export SONAR_TOKEN="${SONAR_TOKEN:-}"

KEYCLOAK_CLIENT_SECRET_VALUE="$(
  grep -E '^KEYCLOAK_CLIENT_SECRET=' "$ENV_FILE" 2>/dev/null \
    | tail -n 1 \
    | cut -d'=' -f2-
)"

if [ -z "$KEYCLOAK_CLIENT_SECRET_VALUE" ]; then
  REALM_FILE="$ROOT_DIR/infrastructure/config/keycloak-export/pios-realm.json"
  if [ -f "$REALM_FILE" ]; then
    KEYCLOAK_CLIENT_SECRET_VALUE=$(python3 -c "
import json
with open('$REALM_FILE') as f:
    data = json.load(f)
for client in data.get('clients', []):
    if client.get('clientId') == 'django-backend':
        print(client.get('secret', ''))
        break
" 2>/dev/null || true)
  fi

  if [ -n "$KEYCLOAK_CLIENT_SECRET_VALUE" ]; then
    echo "🔑 Auto-detected KEYCLOAK_CLIENT_SECRET from keycloak export."
    if grep -q '^KEYCLOAK_CLIENT_SECRET=' "$ENV_FILE"; then
      sed -i '' "s/^KEYCLOAK_CLIENT_SECRET=.*/KEYCLOAK_CLIENT_SECRET=$KEYCLOAK_CLIENT_SECRET_VALUE/" "$ENV_FILE" 2>/dev/null || \
      sed -i "s/^KEYCLOAK_CLIENT_SECRET=.*/KEYCLOAK_CLIENT_SECRET=$KEYCLOAK_CLIENT_SECRET_VALUE/" "$ENV_FILE" 2>/dev/null || true
    else
      echo "KEYCLOAK_CLIENT_SECRET=$KEYCLOAK_CLIENT_SECRET_VALUE" >> "$ENV_FILE"
    fi
  else
    echo "❌ KEYCLOAK_CLIENT_SECRET is required in .env.local for backend Keycloak validation."
    echo "   Find it in Keycloak: realm pios → Clients → django-backend → Credentials."
    echo "   Or search infrastructure/config/keycloak-export/pios-realm.json for clientId django-backend."
    exit 1
  fi
fi

APP_VERSION=$(python3 -c "
import tomllib
with open('$ROOT_DIR/backendMulti/pyproject.toml', 'rb') as f:
    print(tomllib.load(f)['project']['version'])
" 2>/dev/null || python -c "
import tomllib
with open('$ROOT_DIR/backendMulti/pyproject.toml', 'rb') as f:
    print(tomllib.load(f)['project']['version'])
" 2>/dev/null || echo "dev")

export APP_VERSION
echo "📦 Backend version: $APP_VERSION"

docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d --build

echo "⏳ Waiting for PostgreSQL..."
RETRIES=0
until docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" exec -T postgres pg_isready -U "${POSTGRES_USER:-admin}" > /dev/null 2>&1 || \
      { PG_CID=$(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps -q postgres 2>/dev/null) && [ -n "$PG_CID" ] && docker exec "$PG_CID" pg_isready -U "${POSTGRES_USER:-admin}" > /dev/null 2>&1; }; do
  RETRIES=$((RETRIES + 1))
  if [ "$RETRIES" -ge 30 ]; then
    echo "❌ PostgreSQL did not become ready after 60s."
    exit 1
  fi
  sleep 2
done
echo "✅ PostgreSQL ready!"

echo ""
echo "🐳 Docker services running:"
echo "   🐘 PostgreSQL → localhost:5432"
echo "   🔬 MLflow     → http://localhost:8889"
echo "   🔑 Keycloak   → http://localhost:8080"
echo "   🐍 Backend    → http://localhost:8000"