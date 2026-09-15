#!/usr/bin/env bash
# ==============================================================================
# PulseAI - Cloudflare Quick Tunnel for Local Django Backend
# ==============================================================================
# Exposes the local Django server (default: http://localhost:8000) via a secure
# Cloudflare Quick Tunnel (https://*.trycloudflare.com).
#
# This enables Cloudflare Pages Functions (Edge API) and frontend preview
# environments to route natural language predictions through the full Django
# DynamicXLAMOrchestrator and Tool Registry.
# ==============================================================================

set -e

# Visual colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PORT="${1:-${DJANGO_PORT:-8000}}"
LOCAL_URL="http://localhost:${PORT}"

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  PulseAI — Cloudflare Quick Tunnel for Django Orchestrator    ${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}\n"

echo -e "${YELLOW}Target Local Service:${NC} ${LOCAL_URL}"

# 1. Check if local Django server is reachable
if curl -s -o /dev/null -w "%{http_code}" "${LOCAL_URL}/api/ml/orchestrator/status/" 2>/dev/null | grep -qE "200|401|403|404|503"; then
  echo -e "${GREEN}✓ Local Django service detected on port ${PORT}.${NC}"
else
  echo -e "${YELLOW}⚠️  Notice: Local service not detected on ${LOCAL_URL}.${NC}"
  echo -e "   Make sure your Django server is started with:"
  echo -e "   ${CYAN}cd backendMulti && python manage.py runserver 0.0.0.0:${PORT}${NC}\n"
fi

# 2. Locate or install cloudflared
if command -v cloudflared &>/dev/null; then
  CLOUDFLARED_CMD="cloudflared"
elif [ -x "/opt/homebrew/bin/cloudflared" ]; then
  CLOUDFLARED_CMD="/opt/homebrew/bin/cloudflared"
elif [ -x "/usr/local/bin/cloudflared" ]; then
  CLOUDFLARED_CMD="/usr/local/bin/cloudflared"
elif command -v docker &>/dev/null; then
  echo -e "${YELLOW}cloudflared binary not found on PATH. Falling back to Docker...${NC}"
  # On macOS/Windows Docker, host.docker.internal reaches the host machine
  if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
    TUNNEL_TARGET="http://host.docker.internal:${PORT}"
  else
    TUNNEL_TARGET="http://localhost:${PORT}"
  fi
  echo -e "${GREEN}Starting Cloudflare Tunnel via Docker pointing to ${TUNNEL_TARGET}...${NC}\n"
  exec docker run --rm -it --network=host cloudflare/cloudflared:latest tunnel --url "${TUNNEL_TARGET}"
else
  echo -e "${RED}❌ cloudflared is not installed and Docker is not available.${NC}"
  echo -e "Please install cloudflared using:"
  echo -e "  ${CYAN}brew install cloudflared${NC}"
  echo -e "Or download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
  exit 1
fi

echo -e "\n${GREEN}Starting Cloudflare Quick Tunnel...${NC}"
echo -e "${YELLOW}Look for the generated tunnel URL below:${NC}"
echo -e "  Example: ${CYAN}https://your-tunnel-name.trycloudflare.com${NC}\n"
echo -e "Then copy the URL and set it in your Cloudflare Pages environment variables:"
echo -e "  ${CYAN}DJANGO_BACKEND_URL=https://your-tunnel-name.trycloudflare.com${NC}\n"
echo -e "Press ${YELLOW}Ctrl+C${NC} anytime to terminate the tunnel.\n"

exec "$CLOUDFLARED_CMD" tunnel --url "${LOCAL_URL}"
