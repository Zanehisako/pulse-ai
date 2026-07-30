#!/bin/bash
# PIOS — Stop Local Docker Services (keep volumes)

set -e
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/infrastructure/docker/docker-compose.local.yaml"
ENV_FILE="$ROOT_DIR/.env.local"

echo "🐳 Stopping local Docker services..."

if [ -f "$ENV_FILE" ]; then
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down
else
  docker compose -f "$COMPOSE_FILE" down
fi

echo ""
echo "✅ All services stopped (volumes preserved):"
echo "   🐘 PostgreSQL → stopped"
echo "   🔬 MLflow     → stopped"
echo "   🔑 Keycloak   → stopped"
echo "   🐍 Backend    → stopped"
