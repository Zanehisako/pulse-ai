#!/bin/bash
# PIOS — Orchestrator

set -e
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$ROOT_DIR/infrastructure/scripts"

# ── COLORS ────────────────────────────────────
if [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ]; then
  G='\033[0;32m' Y='\033[1;33m' B='\033[0;34m' R='\033[0;31m' N='\033[0m'
else
  G='' Y='' B='' R='' N=''
fi

# ── HELP ──────────────────────────────────────
show_help() {
  echo -e "${B}══════════════════════════════════════════════${N}"
  echo -e "${B}           🧠 PIOS AI & MLOps Platform        ${N}"
  echo -e "${B}══════════════════════════════════════════════${N}"
  echo ""
  echo "Usage: ./run.sh [command]"
  echo ""
  echo "Commands:"
  echo "  setup       Create local env and install local dependencies"
  echo "  start       Start all Docker services"
  echo "  stop        Stop Docker services (keep volumes)"
  echo "  remove      Stop Docker services and remove volumes"
  echo "  restart     Remove then start fresh"
  echo "  scan        Run local SonarQube scan"
  echo "  status      Show running containers"
  echo "  help        Show this help"
  echo ""
}

# ── COMMANDS ──────────────────────────────────
case "${1:-help}" in

  setup)
    echo -e "${G}🛠️  Setting up PIOS...${N}"
    bash "$SCRIPTS_DIR/setup.sh"
    ;;

  start)
    echo -e "${G}🚀 Starting PIOS...${N}"
    bash "$SCRIPTS_DIR/build-docker-image.sh"
    ;;

  stop)
    echo -e "${Y}🛑 Stopping PIOS (volumes preserved)...${N}"
    bash "$SCRIPTS_DIR/stop-docker-image.sh"
    ;;

  remove)
    echo -e "${R}🗑️  Removing PIOS (volumes will be deleted)...${N}"
    read -p "   Are you sure? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    bash "$SCRIPTS_DIR/remove-docker-image.sh"
    ;;

  restart)
    echo -e "${Y}🔄 Restarting PIOS...${N}"
    bash "$SCRIPTS_DIR/stop-docker-image.sh"
    bash "$SCRIPTS_DIR/build-docker-image.sh"
    ;;

  scan)
    echo -e "${B}🔎 Running PIOS scan...${N}"
    bash "$ROOT_DIR/start-local.sh" scan
    ;;

  status)
    echo -e "${B}📊 PIOS Container Status:${N}"
    docker ps --filter "name=pios" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    ;;

  help|--help|-h)
    show_help
    ;;

  *)
    echo -e "${R}❌ Unknown command: $1${N}"
    show_help
    exit 1
    ;;
esac
