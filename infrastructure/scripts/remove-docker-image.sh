#!/bin/bash
# PIOS — Stop Local Docker Services

set -e
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/infrastructure/docker/docker-compose.local.yaml"
ENV_FILE="$ROOT_DIR/.env.local"

echo "🐳 Stopping local Docker services..."

if [ -f "$ENV_FILE" ]; then
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down -v
else
  docker compose -f "$COMPOSE_FILE" down -v
fi

echo ""
echo "✅ All services stopped and volumes removed:"
echo "   🐘 PostgreSQL → removed"
echo "   🔬 MLflow     → removed"
echo "   🔑 Keycloak   → removed"
echo "   🐍 Backend    → removed"
