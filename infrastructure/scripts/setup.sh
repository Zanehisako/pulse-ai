#!/usr/bin/env bash
# PIOS — setup script for local development

set -e

# Colors
G='\033[0;32m'  Y='\033[1;33m'  R='\033[0;31m'  N='\033[0m'

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_SELECTOR="$ROOT_DIR/infrastructure/scripts/select_python.py"
PYTHON_PROJECT_DIR="${PIOS_PYTHON_PROJECT_DIR:-$ROOT_DIR/backendMulti}"

BOOTSTRAP_PYTHON="$(command -v python3 || command -v python || true)"
if [ -z "$BOOTSTRAP_PYTHON" ]; then
  echo -e "${R}❌ Python not found in PATH. Install a supported Python first.${N}"
  exit 1
fi

PYTHON_REQUIREMENT="$("$BOOTSTRAP_PYTHON" "$PYTHON_SELECTOR" requirement "$PYTHON_PROJECT_DIR")" || exit 1
PYTHON="$("$BOOTSTRAP_PYTHON" "$PYTHON_SELECTOR" select "$PYTHON_PROJECT_DIR")" || exit 1

# # ─────────────────────────────────────────────
# # Guards: check required tools are on PATH
# # ─────────────────────────────────────────────
# for TOOL in docker npm flutter; do
#   command -v "$TOOL" &>/dev/null || {
#     echo -e "${R}❌ '$TOOL' not found in PATH. Please install it before running setup.${N}"
#     exit 1
#   }
# done

# ─────────────────────────────────────────────
# Docker Compose: support both plugin and legacy
# ─────────────────────────────────────────────
if docker compose version &>/dev/null 2>&1; then
  DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &>/dev/null; then
  DOCKER_COMPOSE="docker-compose"
else
  echo -e "${R}❌ Neither 'docker compose' nor 'docker-compose' found.${N}"
  exit 1
fi

activate_venv() {
  local VENV_PATH="$1"
  local VENV_PYTHON=""

  if [ -f "$VENV_PATH/Scripts/python.exe" ]; then
    VENV_PYTHON="$VENV_PATH/Scripts/python.exe"
  elif [ -f "$VENV_PATH/bin/python" ]; then
    VENV_PYTHON="$VENV_PATH/bin/python"
  fi

  if [ -n "$VENV_PYTHON" ] && ! "$BOOTSTRAP_PYTHON" "$PYTHON_SELECTOR" check "$PYTHON_PROJECT_DIR" "$VENV_PYTHON" >/dev/null 2>&1; then
    echo -e "${Y}⚙️ Recreating virtual environment at $VENV_PATH for Python $PYTHON_REQUIREMENT...${N}"
    rm -rf "$VENV_PATH"
    VENV_PYTHON=""
  fi

  if [ ! -d "$VENV_PATH" ]; then
    echo -e "${Y}⚙️ Creating virtual environment at $VENV_PATH...${N}"
    $PYTHON -m venv "$VENV_PATH" || {
      echo -e "${R}❌ Failed to create virtual environment${N}"
      exit 1
    }
  fi

  if [ -f "$VENV_PATH/Scripts/activate" ]; then
    source "$VENV_PATH/Scripts/activate"   # Windows (Git Bash)
  elif [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"       # Mac/Linux
  else
    echo -e "${R}❌ No activation script found in $VENV_PATH${N}"
    exit 1
  fi

  echo -e "${G}   ✅ Virtual environment ready${N}"
}

echo -e "${Y}🔧 Setting up PIOS project...${N}\n"

# ─────────────────────────────────────────────
# Step 0: .env
# ─────────────────────────────────────────────
if [ ! -f "$ROOT_DIR/.env.local" ]; then
  echo -e "${Y}📄 Creating .env.local...${N}"
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env.local"
  echo -e "${G}   ✅ Created .env.local${N}"
else
  echo -e "${G}   ✅ .env.local already exists${N}"
fi

# ─────────────────────────────────────────────
# Python version check
# ─────────────────────────────────────────────
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
  echo -e "${R}❌ Python 3.11+ required. Found: $PY_MAJOR.$PY_MINOR${N}"
  exit 1
fi

echo -e "${G}   ✅ Python $PY_MAJOR.$PY_MINOR detected${N}"

ensure_project_venv() {
  local project_dir="$1"
  local project_name="$2"

  echo -e "${Y}   🧪 Activating virtual environment for ${project_name}...${N}"
  activate_venv "$project_dir"
  python -m pip install --upgrade pip
}

# ─────────────────────────────────────────────
# Step 1: Django Backend
# ─────────────────────────────────────────────
echo -e "\n${Y}🐍 Step 1: Django Backend...${N}"
cd "$ROOT_DIR/backendMulti"
activate_venv ".venv"
pip install -e ".[dev]" --prefer-binary

# ─────────────────────────────────────────────
# Step 2: ML Backend
# ─────────────────────────────────────────────
# echo -e "\n${Y}🤖 Step 2: ML Backend...${N}"
# cd "$ROOT_DIR/ml-backend"
# activate_venv ".venv"

# # Optimization: install llama-cpp-python wheel first (avoid local source builds on Windows)
# PY_VERSION=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
# WHEEL_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cpu/$PY_VERSION"

# if ! python -c "import llama_cpp" &>/dev/null; then
#   echo -e "${Y}   ⚙️ Installing llama-cpp-python (CPU wheel only)...${N}"
#   pip install "llama-cpp-python==0.3.19" \
#     --extra-index-url "$WHEEL_INDEX" \
#     --only-binary=:all: \
#     --timeout 120 \
#     --retries 10
# fi

# pip install -e ".[dev,mlops]" --prefer-binary

# ─────────────────────────────────────────────
# Step 3: Feast
# ─────────────────────────────────────────────
echo -e "\n${Y}🍽️ Step 3: Feast Registry...${N}"
cd "$ROOT_DIR/ml-backend/feature_repo"

if [ ! -f "feature_store.yaml" ]; then
  echo -e "${R}❌ feature_store.yaml not found in feature_repo/. Cannot run 'feast apply'.${N}"
  exit 1
fi

feast apply || {
  echo -e "${R}❌ 'feast apply' failed. Check your feature_store.yaml and registry config.${N}"
  exit 1
}

# ─────────────────────────────────────────────
# Step 4: Database
# ─────────────────────────────────────────────
echo -e "\n${Y}🗄️ Step 4: PostgreSQL...${N}"

$DOCKER_COMPOSE -f "$ROOT_DIR/infrastructure/docker/docker-compose.local.yaml" up -d postgres

# Wait for Postgres with a timeout (max 60s)
RETRIES=0
until docker exec pios-postgres pg_isready -U admin -d pios > /dev/null 2>&1; do
  RETRIES=$((RETRIES + 1))
  if [ "$RETRIES" -ge 30 ]; then
    echo -e "${R}❌ PostgreSQL did not become ready after 60s. Check Docker logs.${N}"
    exit 1
  fi
  sleep 2
done

echo -e "${G}   ✅ PostgreSQL ready${N}"

# ─────────────────────────────────────────────
# Step 5: Django migrate
# ─────────────────────────────────────────────
echo -e "\n${Y}📦 Step 5: Migrations...${N}"

cd "$ROOT_DIR/backendMulti"
activate_venv ".venv"

DB_HOST=localhost python manage.py migrate
DB_HOST=localhost python manage.py import_ml_config

# ─────────────────────────────────────────────
echo -e "\n${G}✅ Setup complete!${N}"
echo -e "   Run ./run.sh start to launch all PIOS AI services."
