#!/bin/bash
#  PIOS — Start Local Docker Services

set -e
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/infrastructure/docker/docker-compose.local.yaml"
ENV_FILE="$ROOT_DIR/.env.local"

echo "🐳 Starting local Docker services..."

if [ ! -f "$ENV_FILE" ]; then
  echo "📄 Creating .env.local from .env.example..."
  cp "$ROOT_DIR/.env.example" "$ENV_FILE"
    echo "⚠️  Fill KEYCLOAK_CLIENT_SECRET in .env.local, then run this command again."
  echo "   Find it in Keycloak: realm pios → Clients → django-backend → Credentials."
  echo "   Or search infrastructure/config/keycloak-export/pios-realm.json for clientId django-backend."
  exit 1
fi

KEYCLOAK_CLIENT_SECRET_VALUE="$(
  grep -E '^KEYCLOAK_CLIENT_SECRET=' "$ENV_FILE" 2>/dev/null \
    | tail -n 1 \
    | cut -d'=' -f2-
)"
if [ -z "$KEYCLOAK_CLIENT_SECRET_VALUE" ]; then
  echo "❌ KEYCLOAK_CLIENT_SECRET is required in .env.local for backend Keycloak validation."
  echo "   Find it in Keycloak: realm pios → Clients → django-backend → Credentials."
  echo "   Or search infrastructure/config/keycloak-export/pios-realm.json for clientId django-backend."
  exit 1
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
until docker exec pios-postgres pg_isready -U admin -d pios > /dev/null 2>&1; do
  sleep 2
done
echo "✅ PostgreSQL ready!"

echo ""
echo "🐳 Docker services running:"
echo "   🐘 PostgreSQL → localhost:5432"
echo "   🔬 MLflow     → http://localhost:8889"
echo "   🔑 Keycloak   → http://localhost:8080"
echo "   🐍 Backend    → http://localhost:8000"