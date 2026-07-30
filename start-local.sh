#!/bin/bash

#  PIOS — Main Project Runner (cross-platform: macOS / Linux / Windows Git-Bash / WSL)

set -e
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$ROOT_DIR/infrastructure/scripts"
DOCKER_DIR="$ROOT_DIR/infrastructure/docker"
DEFAULT_PYTHON="$(command -v python3 || command -v python || true)"

# ── OS DETECTION ──────────────────────────────
# Sets OS_TYPE to one of: macos, linux, wsl, windows
detect_os() {
  case "$(uname -s)" in
    Darwin*)  OS_TYPE="macos"   ;;
    Linux*)
      if grep -qiE '(microsoft|wsl)' /proc/version 2>/dev/null; then
        OS_TYPE="wsl"
      else
        OS_TYPE="linux"
      fi
      ;;
    CYGWIN*|MINGW*|MSYS*)  OS_TYPE="windows" ;;
    *)  OS_TYPE="unknown" ;;
  esac
}
detect_os

# ── COLORS (disabled when stdout is not a tty) ───
if [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ]; then
  G='\033[0;32m'  Y='\033[1;33m'  B='\033[0;34m'  R='\033[0;31m'  N='\033[0m'
else
  G=''  Y=''  B=''  R=''  N=''
fi

# ── LOAD .env.local EARLY (suppresses SONAR_TOKEN etc. warnings) ──
if [ -f "$ROOT_DIR/.env.local" ]; then
  set -a; source "$ROOT_DIR/.env.local"; set +a
fi
export SONAR_TOKEN="${SONAR_TOKEN:-}"

# ── DOCKER COMPOSE DETECTION ─────────────────
# Prefer "docker compose" (v2 plugin), fall back to "docker-compose" (standalone)
detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
  else
    echo -e "${R}❌ Neither 'docker compose' (plugin) nor 'docker-compose' (standalone) found.${N}"
    echo "   Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
  fi
}

# Wrapper: call compose with the right binary
dc() {
  $COMPOSE_CMD "$@"
}

# ── PORTABLE sed -i ──────────────────────────
# macOS BSD sed requires -i '', GNU sed does not.
sed_inplace() {
  if [ "$OS_TYPE" = "macos" ]; then
    sed -i '' "$@"
  else
    sed -i "$@"
  fi
}

# ── KILL PROCESS ON PORT (cross-platform) ────
kill_port() {
  local port="$1"
  case "$OS_TYPE" in
    macos)
      local pid
      pid=$(lsof -ti:"$port" 2>/dev/null || true)
      [ -n "$pid" ] && kill -9 $pid 2>/dev/null || true
      ;;
    linux|wsl)
      if command -v lsof >/dev/null 2>&1; then
        local pid
        pid=$(lsof -ti:"$port" 2>/dev/null || true)
        [ -n "$pid" ] && kill -9 $pid 2>/dev/null || true
      elif command -v ss >/dev/null 2>&1; then
        # ss -tlnp output: LISTEN 0 128 *:8000 *:* users:(("python",pid=1234,fd=5))
        local pid
        pid=$(ss -tlnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
      elif command -v fuser >/dev/null 2>&1; then
        fuser -k "${port}/tcp" 2>/dev/null || true
      fi
      ;;
    windows)
      # Git Bash / MSYS on Windows
      local pid
      pid=$(netstat -ano 2>/dev/null | grep -E " 0\.0\.0\.0:${port} | 127\.0\.0\.1:${port} " | awk '{print $5}' | head -1 || true)
      [ -n "$pid" ] && taskkill //F //PID "$pid" 2>/dev/null || true
      ;;
  esac
}

# ── ENSURE KEYCLOAK DATABASE EXISTS ─────────
ensure_keycloak_database() {
  local postgres_user="${POSTGRES_USER:-admin}"
  local postgres_db="${POSTGRES_DB:-pios}"
  local db_name="${KEYCLOAK_DB_NAME:-keycloak}"
  local db_owner="${KEYCLOAK_DB_OWNER:-$postgres_user}"
  echo -e "${Y}🔐 Ensuring Keycloak database exists...${N}"

  dc -f "$COMPOSE_FILE" exec -T postgres psql \
    -v ON_ERROR_STOP=1 \
    -U "$postgres_user" \
    -d "$postgres_db" \
    -v db_name="$db_name" \
    -v db_owner="$db_owner" <<-'EOSQL'
SELECT format('CREATE DATABASE %I OWNER %I', :'db_name', :'db_owner')
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = :'db_name'
)\gexec
EOSQL

  echo -e "${G}   ✅ Keycloak database ready${N}"
}

# ── ENSURE MLFLOW DATABASE EXISTS ───────────
ensure_mlflow_database() {
  local postgres_user="${POSTGRES_USER:-admin}"
  local postgres_db="${POSTGRES_DB:-pios}"
  local db_name="mlflow"
  local db_owner="$postgres_user"
  echo -e "${Y}📊 Ensuring MLflow database exists...${N}"

  dc -f "$COMPOSE_FILE" exec -T postgres psql \
    -v ON_ERROR_STOP=1 \
    -U "$postgres_user" \
    -d "$postgres_db" \
    -v db_name="$db_name" \
    -v db_owner="$db_owner" <<-'EOSQL'
SELECT format('CREATE DATABASE %I OWNER %I', :'db_name', :'db_owner')
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = :'db_name'
)\gexec
EOSQL

  echo -e "${G}   ✅ MLflow database ready${N}"
}

# ── KEYCLOAK BOOTSTRAP ──────────────────────
is_truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

json_value() {
  local key="$1"
  "${DEFAULT_PYTHON:-python3}" -c "import json,sys; print(json.load(sys.stdin).get('$key', ''))"
}

wait_for_keycloak() {
  local keycloak_url="${1%/}"
  local realm="$2"
  local max_wait="${3:-120}"
  local waited=0

  echo -e "${Y}⏳ Waiting for Keycloak...${N}"
  until curl -sf "$keycloak_url/realms/master" > /dev/null 2>&1 \
    && curl -sf "$keycloak_url/realms/$realm/.well-known/openid-configuration" > /dev/null 2>&1; do
    if [ "$waited" -ge "$max_wait" ]; then
      echo -e "${R}❌ Keycloak did not become ready within ${max_wait}s.${N}"
      echo "   Checked URL: $keycloak_url"
      echo "   Checked realm: $realm"
      exit 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo -e "${G}   ✅ Keycloak ready${N}"
}

get_keycloak_admin_token() {
  local keycloak_url="${1%/}"
  local admin_username="${KEYCLOAK_BOOTSTRAP_ADMIN_USERNAME:-${KEYCLOAK_ADMIN:-admin}}"
  local admin_password="${KEYCLOAK_BOOTSTRAP_ADMIN_PASSWORD:-${KEYCLOAK_ADMIN_PASSWORD:-admin}}"

  curl -sf -X POST "$keycloak_url/realms/master/protocol/openid-connect/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "grant_type=password" \
    -d "client_id=admin-cli" \
    -d "username=$admin_username" \
    -d "password=$admin_password" | json_value "access_token"
}

fail_keycloak_admin_token() {
  local keycloak_url="${1%/}"
  local admin_username="${KEYCLOAK_BOOTSTRAP_ADMIN_USERNAME:-${KEYCLOAK_ADMIN:-admin}}"

  echo -e "${R}❌ Keycloak admin token failed.${N}"
  echo "   Checked URL: $keycloak_url/realms/master/protocol/openid-connect/token"
  echo "   Admin username: $admin_username"
  echo "   Client ID: admin-cli"
  echo "   Possible causes:"
  echo "   - Keycloak is not fully ready"
  echo "   - KEYCLOAK_BOOTSTRAP_ADMIN_USERNAME / KEYCLOAK_BOOTSTRAP_ADMIN_PASSWORD are wrong"
  echo "   - start-local.sh is using a host URL while the caller needs a container URL"
  echo "   - the Keycloak container was first created before bootstrap admin env vars were set"
}

ensure_keycloak_realm_exists() {
  local keycloak_url="${1%/}"
  local realm="$2"
  local token="$3"
  local status

  status=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $token" \
    "$keycloak_url/admin/realms/$realm")

  if [ "$status" != "200" ]; then
    echo -e "${R}❌ Keycloak realm '$realm' is not available (HTTP $status).${N}"
    echo "   Check the realm import under infrastructure/config/keycloak-export."
    exit 1
  fi
}

get_keycloak_client_secret() {
  local keycloak_url="${1%/}"
  local realm="$2"
  local client_id="$3"
  local token="$4"
  local clients_json client_uuid

  clients_json=$(curl -sf -G "$keycloak_url/admin/realms/$realm/clients" \
    -H "Authorization: Bearer $token" \
    --data-urlencode "clientId=$client_id")
  client_uuid=$(printf '%s' "$clients_json" | "${DEFAULT_PYTHON:-python3}" -c "import json,sys; client_id=sys.argv[1]; clients=json.load(sys.stdin); match=next((client for client in clients if client.get('clientId') == client_id), {}); print(match.get('id', ''))" "$client_id")

  if [ -z "$client_uuid" ]; then
    return 1
  fi

  curl -sf "$keycloak_url/admin/realms/$realm/clients/$client_uuid/client-secret" \
    -H "Authorization: Bearer $token" | json_value "value"
}

ensure_keycloak_client_secret() {
  if [ -n "${KEYCLOAK_CLIENT_SECRET:-}" ]; then
    return 0
  fi

  local keycloak_url="${KEYCLOAK_SERVER_URL:-http://localhost:8080/}"
  local realm="${KEYCLOAK_REALM_NAME:-pios}"
  local client_id="${KEYCLOAK_CLIENT_ID:-django-backend}"
  keycloak_url="${keycloak_url%/}"

  echo -e "${Y}🔐 Resolving Keycloak client secret for '$client_id'...${N}"

  local token secret
  if ! token="$(get_keycloak_admin_token "$keycloak_url" 2>/dev/null)" || [ -z "$token" ]; then
    fail_keycloak_admin_token "$keycloak_url"
    exit 1
  fi

  if ! secret="$(get_keycloak_client_secret "$keycloak_url" "$realm" "$client_id" "$token" 2>/dev/null)" || [ -z "$secret" ]; then
    echo -e "${R}❌ Could not resolve Keycloak client secret for '$client_id'.${N}"
    echo "   Check the client import under infrastructure/config/keycloak-export."
    exit 1
  fi

  export KEYCLOAK_CLIENT_SECRET="$secret"
  echo -e "${G}   ✅ Keycloak client secret resolved${N}"
}

ensure_keycloak_pios_user() {
  if is_truthy "${SKIP_KEYCLOAK_BOOTSTRAP:-false}"; then
    echo -e "${Y}🔐 SKIP_KEYCLOAK_BOOTSTRAP=true — skipping Keycloak user verification${N}"
    return 0
  fi

  local keycloak_url="${KEYCLOAK_SERVER_URL:-http://localhost:8080/}"
  local realm="${KEYCLOAK_REALM_NAME:-pios}"
  local app_user="${KEYCLOAK_APP_USERNAME:-pios}"
  local app_pass="${KEYCLOAK_APP_PASSWORD:-pios}"
  local app_email="${KEYCLOAK_APP_EMAIL:-$app_user@example.com}"

  keycloak_url="${keycloak_url%/}"

  echo -e "${Y}🔐 Ensuring Keycloak user '$app_user' exists...${N}"

  local token
  if ! token="$(get_keycloak_admin_token "$keycloak_url" 2>/dev/null)" || [ -z "$token" ]; then
    fail_keycloak_admin_token "$keycloak_url"
    exit 1
  fi

  ensure_keycloak_realm_exists "$keycloak_url" "$realm" "$token"

  local users_json user_id
  users_json=$(curl -sf -G "$keycloak_url/admin/realms/$realm/users" \
    -H "Authorization: Bearer $token" \
    --data-urlencode "username=$app_user" \
    --data-urlencode "exact=true")
  user_id=$(printf '%s' "$users_json" | "${DEFAULT_PYTHON:-python3}" -c "import json,sys; users=json.load(sys.stdin); print(users[0].get('id', '') if users else '')")

  if [ -n "$user_id" ]; then
    echo -e "${G}   ✅ Keycloak user '$app_user' already exists${N}"
  else
    local create_status
    create_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$keycloak_url/admin/realms/$realm/users" \
      -H "Authorization: Bearer $token" \
      -H "Content-Type: application/json" \
      -d "{
        \"username\": \"$app_user\",
        \"enabled\": true,
        \"email\": \"$app_email\",
        \"emailVerified\": true,
        \"firstName\": \"$app_user\",
        \"lastName\": \"$app_user\"
      }")

    if [ "$create_status" != "201" ]; then
      echo -e "${R}❌ Could not create Keycloak user '$app_user' (HTTP $create_status).${N}"
      exit 1
    fi

    users_json=$(curl -sf -G "$keycloak_url/admin/realms/$realm/users" \
      -H "Authorization: Bearer $token" \
      --data-urlencode "username=$app_user" \
      --data-urlencode "exact=true")
    user_id=$(printf '%s' "$users_json" | "${DEFAULT_PYTHON:-python3}" -c "import json,sys; users=json.load(sys.stdin); print(users[0].get('id', '') if users else '')")
    if [ -z "$user_id" ]; then
      echo -e "${R}❌ Created Keycloak user '$app_user' but could not resolve its id.${N}"
      exit 1
    fi
    echo -e "${G}   ✅ Keycloak user '$app_user' created${N}"
  fi

  local password_status
  password_status=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "$keycloak_url/admin/realms/$realm/users/$user_id/reset-password" \
    -H "Authorization: Bearer $token" \
    -H "Content-Type: application/json" \
    -d "{\"type\":\"password\",\"value\":\"$app_pass\",\"temporary\":false}")

  if [ "$password_status" != "204" ]; then
    echo -e "${R}❌ Could not set password for Keycloak user '$app_user' (HTTP $password_status).${N}"
    exit 1
  fi
  echo -e "${G}   ✅ Keycloak user '$app_user' password verified${N}"
}

# ── ENSURE DOCKER ENGINE IS RUNNING ──────────
ensure_docker() {
  local max_wait=90   # seconds — generous for cold-start on Windows/WSL
  local waited=0

  # Quick check
  if docker info >/dev/null 2>&1; then
    echo -e "${G}   ✅ Docker engine is running${N}"
    detect_compose
    return 0
  fi

  echo -e "${Y}⏳ Docker engine not responding — attempting to start it...${N}"

  case "$OS_TYPE" in
    macos)
      # open is idempotent; -g = don't bring to foreground
      open -g -a Docker 2>/dev/null || true
      ;;
    linux)
      # Try systemd first, then legacy init
      if command -v systemctl >/dev/null 2>&1; then
        sudo systemctl start docker 2>/dev/null || true
      elif command -v service >/dev/null 2>&1; then
        sudo service docker start 2>/dev/null || true
      else
        echo -e "${R}   Cannot auto-start Docker on this Linux system.${N}"
      fi
      ;;
    wsl)
      # WSL2 with Docker Desktop integration — start via Windows side
      if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -Command "Start-Process 'C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe' -WindowStyle Hidden" 2>/dev/null || true
      elif command -v cmd.exe >/dev/null 2>&1; then
        cmd.exe /c "start \"\" \"C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe\"" 2>/dev/null || true
      else
        echo -e "${R}   Cannot auto-start Docker Desktop from WSL. Please start it manually.${N}"
      fi
      ;;
    windows)
      # Git Bash / MSYS — use start to launch Docker Desktop
      start "" "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe" 2>/dev/null \
        || cmd.exe /c "start \"\" \"C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe\"" 2>/dev/null \
        || true
      ;;
    *)
      echo -e "${Y}   Unknown OS — cannot auto-start Docker. Please start it manually.${N}"
      ;;
  esac

  # Poll until the daemon responds
  while ! docker info >/dev/null 2>&1; do
    if [ "$waited" -ge "$max_wait" ]; then
      echo ""
      echo -e "${R}❌ Docker engine did not start within ${max_wait}s.${N}"
      echo -e "${R}   Please start Docker Desktop (or the Docker daemon) manually, then re-run this script.${N}"
      exit 1
    fi
    printf "."
    sleep 2
    waited=$((waited + 2))
  done

  echo ""
  echo -e "${G}   ✅ Docker engine is running (took ~${waited}s)${N}"
  detect_compose
}

# ── HELP ──────────────────────────────────────
show_help() {
  echo -e "${B}══════════════════════════════════════════════${N}"
  echo -e "${B}                    🩸 PIOS                   ${N}"
  echo -e "${B}══════════════════════════════════════════════${N}"
  echo ""
  echo "Usage: ./run.sh [command] [environment]"
  echo ""
  echo "Commands:"
  echo "  start [env]     Start all services (default: local)"
  echo "  stop [env]      Stop Docker services"
  echo "  restart [env]   Restart Docker services"
  echo "  setup           First-time project setup"
  echo "  train           Train ML models"
  echo "  scan            Run SonarQube analysis"
  echo "  migrate         Run Django migrations"
  echo "  status          Show running containers"
  echo "  web             Start React web app only"
  echo "  help            Show this help"
  echo ""
}

ENV="${2:-local}"
COMPOSE_FILE="$DOCKER_DIR/docker-compose.${ENV}.yaml"

# Check compose file exists
check_compose() {
  if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${R}❌ Compose file not found: $COMPOSE_FILE${N}"
    echo "   Available: local"
    exit 1
  fi
}

# ── VENV ACTIVATION ───────────────────────────
activate_venv() {
  local project_dir="$1"
  if [ -f "$project_dir/pyproject.toml" ]; then
    VENV_DIR="$project_dir/.venv"
  elif [ -f "$project_dir/requirements.txt" ]; then
    VENV_DIR="$project_dir/venv"
  else
    echo -e "${R} No pyproject.toml or requirements.txt found in $project_dir${N}"
    exit 1
  fi

  # Activate — handle Windows vs Unix venv layout
  if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"       # Windows (Git Bash / MSYS)
  elif [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"           # Linux / macOS / WSL
  else
    echo -e "${R}❌ Venv not found at $VENV_DIR — run ./run.sh setup first${N}"
    exit 1
  fi

  echo -e "${G}   ✅ Venv activated: $VENV_DIR${N}"
}

resolve_simulation_studio_python() {
  # backendMulti venv (preferred)
  if [ -x "$ROOT_DIR/backendMulti/.venv/Scripts/python.exe" ]; then
    echo "$ROOT_DIR/backendMulti/.venv/Scripts/python.exe"
    return 0
  fi

  if [ -x "$ROOT_DIR/backendMulti/.venv/bin/python" ]; then
    echo "$ROOT_DIR/backendMulti/.venv/bin/python"
    return 0
  fi

  # fallback to root venv
  if [ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]; then
    echo "$ROOT_DIR/.venv/Scripts/python.exe"
    return 0
  fi

  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    echo "$ROOT_DIR/.venv/bin/python"
    return 0
  fi

  # fallback to system python
  if [ -n "$DEFAULT_PYTHON" ]; then
    echo "$DEFAULT_PYTHON"
    return 0
  fi

  return 1
}

# ── SONAR TOKEN AUTO-SETUP ────────────────────
setup_sonar_token() {
  ENV_FILE="$ROOT_DIR/.env.local"

  # Ensure .env.local exists
  if [ ! -f "$ENV_FILE" ]; then
    echo -e "${Y}📄 Creating .env.local from .env.example...${N}"
    cp "$ROOT_DIR/.env.example" "$ENV_FILE"
  fi

  EXISTING_TOKEN=$(grep "^SONAR_TOKEN=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2)

  if [ -n "$EXISTING_TOKEN" ]; then
    STATUS=$(curl -sf -u "$EXISTING_TOKEN:" http://localhost:9000/api/authentication/validate | grep -o '"valid":[^,}]*' | cut -d: -f2)
    if [ "$STATUS" = "true" ]; then
      echo -e "${G}   ✅ SONAR_TOKEN valid${N}"
      export SONAR_TOKEN="$EXISTING_TOKEN"
      return 0
    fi
    echo -e "${Y}   ⚠️  SONAR_TOKEN expired or invalid — regenerating...${N}"
  fi

  echo -e "${Y}🔑 No SONAR_TOKEN found — generating one automatically...${N}"

  SONAR_ADMIN_PASS="${SONAR_ADMIN_PASSWORD:-admin123}"

  RESPONSE=$(curl -sf -u "admin:${SONAR_ADMIN_PASS}" \
    -X POST "http://localhost:9000/api/user_tokens/generate" \
    -d "name=pios-local-$(date +%s)" \
    -d "type=GLOBAL_ANALYSIS_TOKEN")

  if [ $? -ne 0 ] || [ -z "$RESPONSE" ]; then
    echo -e "${R}❌ Failed to generate token. Is the admin password still 'admin'?${N}"
    echo -e "   Set SONAR_ADMIN_PASSWORD in .env.local if you changed it."
    exit 1
  fi

  TOKEN=$(echo "$RESPONSE" | grep -o '"token":"[^"]*"' | cut -d'"' -f4)

  if [ -z "$TOKEN" ]; then
    echo -e "${R}❌ Could not parse token from response${N}"
    exit 1
  fi

  # Portable in-place sed
  if grep -q "^SONAR_TOKEN=" "$ENV_FILE"; then
    sed_inplace "s/^SONAR_TOKEN=.*/SONAR_TOKEN=$TOKEN/" "$ENV_FILE"
  else
    echo "SONAR_TOKEN=$TOKEN" >> "$ENV_FILE"
  fi
  export SONAR_TOKEN="$TOKEN"

  echo -e "${G}   ✅ Token generated and saved to .env.local${N}"
}

# ═══════════════════════════════════════════════
#  COMMANDS
# ═══════════════════════════════════════════════
case "${1:-start}" in

  # ── SETUP ─────────────────────────────────
  setup)
    bash "$SCRIPTS_DIR/setup.sh"
    ;;

  # ── START ─────────────────────────────────
  start)
    check_compose

    # Clear ports (cross-platform)
    for port in 8000 8010 8888 8889 5173; do
      kill_port "$port"
    done

    echo -e "${B}══════════════════════════════════════════════${N}"
    echo -e "${B}    Starting PIOS [${ENV}]                  ${N}"
    echo -e "${B}══════════════════════════════════════════════${N}"

    # Ensure Docker engine is actually responding before any compose commands
    ensure_docker

    # Start postgres first so we can create auxiliary DBs (keycloak, mlflow)
    # before starting the services that depend on them. Otherwise MLflow and
    # Keycloak crash on first boot when running against an existing volume that
    # predates the auxiliary-DB init script.
    echo -e "\n${Y}🐳 Starting DB in Docker...${N}"
    if ! docker info > /dev/null 2>&1; then
      echo -e "${R} Docker Desktop is not running. Please start it first.${N}"
      exit 1
    fi
    dc -f "$COMPOSE_FILE" up -d postgres

    # Wait for Postgres
    echo -e "${Y}⏳ Waiting for PostgreSQL...${N}"
    until dc -f "$COMPOSE_FILE" exec -T postgres pg_isready -U admin -d pios > /dev/null 2>&1; do
      sleep 2
    done
    echo -e "${G}   ✅ PostgreSQL ready${N}"

    ensure_keycloak_database
    ensure_mlflow_database

    # Now safe to start services that depend on those databases
    dc -f "$COMPOSE_FILE" up -d mlflow keycloak
    
    export KEYCLOAK_SERVER_URL="${KEYCLOAK_SERVER_URL:-http://localhost:8080/}"
    export KEYCLOAK_REALM_NAME="${KEYCLOAK_REALM_NAME:-pios}"
    export KEYCLOAK_BOOTSTRAP_ADMIN_USERNAME="${KEYCLOAK_BOOTSTRAP_ADMIN_USERNAME:-admin}"
    export KEYCLOAK_BOOTSTRAP_ADMIN_PASSWORD="${KEYCLOAK_BOOTSTRAP_ADMIN_PASSWORD:-admin}"
    export KEYCLOAK_APP_USERNAME="${KEYCLOAK_APP_USERNAME:-pios}"
    export KEYCLOAK_APP_PASSWORD="${KEYCLOAK_APP_PASSWORD:-pios}"
    export KEYCLOAK_APP_EMAIL="${KEYCLOAK_APP_EMAIL:-${KEYCLOAK_APP_USERNAME}@example.com}"

    wait_for_keycloak "$KEYCLOAK_SERVER_URL" "$KEYCLOAK_REALM_NAME"

    ensure_keycloak_pios_user

    export PYTHONUTF8=1

    # Prevent OpenMP/MKL thread deadlocks inside the async Daphne server.
    # XGBoost, LightGBM, and NumPy all spawn OpenMP threads that can deadlock
    # with Python's asyncio event loop when running under an ASGI server.
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export BLIS_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1

    # Start Django locally
    echo -e "\n${Y}🐍 Starting Django backend...${N}"
    activate_venv "$ROOT_DIR/backendMulti"
    cd "$ROOT_DIR/backendMulti"
    
    export KEYCLOAK_CLIENT_ID="${KEYCLOAK_CLIENT_ID:-django-backend}"
    ensure_keycloak_client_secret
    export KEYCLOAK_ADMIN_USERNAME="${KEYCLOAK_ADMIN_USERNAME:-$KEYCLOAK_BOOTSTRAP_ADMIN_USERNAME}"
    export KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD:-$KEYCLOAK_BOOTSTRAP_ADMIN_PASSWORD}"
    export PIOS_MODELS_DIR="${PIOS_MODELS_DIR:-$ROOT_DIR/ml-backend/models}"

    python manage.py migrate --noinput --run-syncdb

    echo -e "${Y}⚙️  Syncing ML config into DB...${N}"
    python manage.py import_ml_config

    echo -e "${Y}📊 Generating SOTA model stats for dashboard cards...${N}"
    export PIOS_SOTA_CATALOG_PATH="${PIOS_SOTA_CATALOG_PATH:-$ROOT_DIR/ml-backend/config/sota_model_catalog.json}"
    export PIOS_SOTA_DATASETS_DIR="${PIOS_SOTA_DATASETS_DIR:-$ROOT_DIR/ml-backend/datasets_featues_labels_seperated}"
    if python manage.py generate_sota_model_stats; then
      echo -e "${G}   ✅ SOTA model stats ready${N}"
    else
      echo -e "${Y}   ⚠️  SOTA model stats generation failed — dashboard will use existing stats if present${N}"
    fi

    # Wait for MLflow before seeding (up to 30s). Synthetic prediction history
    # always seeds (no MLflow needed). Live ML predictions are skipped if
    # MLflow / model preload is unavailable, so the trend detector and dynamic
    # thresholds still have data on first run.
    echo -e "${Y}⏳ Waiting for MLflow...${N}"
    MLFLOW_URL="${MLFLOW_TRACKING_URI:-http://localhost:8889}"
    SEED_FLAGS=""
    MLFLOW_READY=false
    for i in $(seq 1 15); do
      if curl -sf "${MLFLOW_URL}/health" > /dev/null 2>&1; then
        MLFLOW_READY=true
        break
      fi
      sleep 2
    done
    if [ "$MLFLOW_READY" = "true" ]; then
      echo -e "${G}   ✅ MLflow ready${N}"
      echo -e "${Y}⚙️  Preloading prediction models...${N}"
      if python manage.py preload_prediction_models; then
        echo -e "${G}   ✅ Prediction models ready${N}"
      else
        echo -e "${Y}   ⚠️  Prediction model preload failed — skipping live predictions only${N}"
        SEED_FLAGS="--skip-predict-live"
      fi
    else
      echo -e "${Y}   ⚠️  MLflow not ready — skipping live predictions only${N}"
      SEED_FLAGS="--skip-predict-live"
    fi

    python manage.py seed_all --force $SEED_FLAGS
    export DJANGO_SETTINGS_MODULE=backendMulti.settings
    daphne -b 0.0.0.0 -p 8000 backendMulti.asgi:application &
    DJANGO_PID=$!
    cd "$ROOT_DIR"

    # Wait for Django
    echo -e "${Y}⏳ Waiting for Django backend...${N}"
    until curl -sf http://localhost:8000/admin/ > /dev/null 2>&1; do
      sleep 20
    done
    echo -e "${G}   ✅ Django backend ready${N}"

    # Start Simulation Studio (API only)
    echo -e "\n${Y}🧪 Starting Simulation Studio FastAPI backend...${N}"
    export PIOS_SIM_OFFLINE="${PIOS_SIM_OFFLINE:-1}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
    export PIOS_STANDALONE_PREDICTIONS="${PIOS_STANDALONE_PREDICTIONS:-1}"
    export PIOS_ML_CONFIG_ROOT="${PIOS_ML_CONFIG_ROOT:-$ROOT_DIR/backendMulti/ml/config}"
    export PIOS_MODELS_DIR="${PIOS_MODELS_DIR:-$ROOT_DIR/ml-backend/models}"
    export PIOS_MLFLOW_HTTP_TIMEOUT_SECONDS="${PIOS_MLFLOW_HTTP_TIMEOUT_SECONDS:-30}"
    export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:8889}"
    SIM_STUDIO_PYTHON="$(resolve_simulation_studio_python || true)"
    if [ -z "$SIM_STUDIO_PYTHON" ]; then
      echo -e "${R}❌ Could not find a Python interpreter for simulation_studio.${N}"
      exit 1
    fi
    if ! "$SIM_STUDIO_PYTHON" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
      echo -e "${R}❌ FastAPI/Uvicorn not available for simulation_studio.${N}"
      echo "   Install the simulation_studio dependencies in $ROOT_DIR/.venv or in your default Python environment."
      exit 1
    fi
    activate_venv "$ROOT_DIR"/backendMulti
    cd "$ROOT_DIR"
    uvicorn simulation_studio.app.main:app --reload --port 8010 &
    SIM_STUDIO_PID=$!
    echo -e "${Y}⏳ Waiting for Simulation Studio backend...${N}"
    until curl -sf http://localhost:8010/api/health > /dev/null 2>&1; do
      sleep 2
    done
    echo -e "${G}   ✅ Simulation Studio backend ready${N}"

    # Summary
    echo -e "\n${B}══════════════════════════════════════════════${N}"
    echo -e "${G}  🎉 PIOS [${ENV}] is running!               ${N}"
    echo -e "${B}══════════════════════════════════════════════${N}"
    echo ""
    echo -e "   🐘 PostgreSQL   → localhost:5432"
    echo -e "   🔐 Keycloak     → http://localhost:8080"
    echo -e "   🐍 Django API   → http://localhost:8000"
    echo -e "   📖 API Docs     → http://localhost:8000/api/docs/"
    echo -e "   🧪 Studio API   → http://localhost:8010"
    echo -e "   📘 Studio Docs  → http://localhost:8010/docs"
    if [ "$ENV" = "local" ]; then
      echo -e "   🌐 React App    → http://localhost:5173"
    fi
    echo ""
    echo -e "${Y}   Press Ctrl+C to stop everything${N}"

    # Graceful shutdown
    cleanup() {
      echo -e "\n${R}🛑 Shutting down PIOS...${N}"
      [ -n "$DJANGO_PID" ]  && kill $DJANGO_PID  2>/dev/null || true
      [ -n "$SIM_STUDIO_PID" ] && kill $SIM_STUDIO_PID 2>/dev/null || true
      [ -n "$REACT_PID" ]   && kill $REACT_PID   2>/dev/null || true
      [ -n "$FLUTTER_PID" ] && kill $FLUTTER_PID  2>/dev/null || true
      [ -n "$FEAST_PID" ]   && kill $FEAST_PID    2>/dev/null || true
      dc -f "$COMPOSE_FILE" stop
      sleep 20
      echo -e "${G}✅ Everything stopped.${N}"
      exit 0
    }
    trap cleanup SIGINT SIGTERM
    wait
    ;;

  # ── STOP ──────────────────────────────────
  stop)
    check_compose
    ensure_docker
    echo -e "${R}🛑 Stopping PIOS [${ENV}]...${N}"
    dc -f "$COMPOSE_FILE" down
    echo -e "${G}✅ Stopped.${N}"
    ;;

  # ── TRAIN ─────────────────────────────────
  train)
    export PYTHONUTF8=1
    check_compose
    ensure_docker

    echo -e "\n${Y}🐳 PostgreSQL starting...${N}"
    dc -f "$COMPOSE_FILE" up -d postgres

    # Wait until Postgres is ready
    echo -e "${Y}⏳ Waiting for PostgreSQL to be ready...${N}"
    until dc -f "$COMPOSE_FILE" exec -T postgres pg_isready -U admin -d pios > /dev/null 2>&1; do
      sleep 2
    done
    echo -e "${G}   ✅ PostgreSQL ready${N}"

    ensure_keycloak_database
    ensure_mlflow_database

    # Training only needs MLflow (tracking) and Postgres (Django DB writes).
    # Keycloak is not used by the training pipeline.
    dc -f "$COMPOSE_FILE" up -d mlflow

    # Run training
    echo -e "\n${Y}🏋️  Running training...${N}"
    activate_venv "$ROOT_DIR/backendMulti"
    cd "$ROOT_DIR/ml-backend/scripts"
    if [ -f "$ROOT_DIR/.env.local" ]; then
      sed 's/\r//' "$ROOT_DIR/.env.local" > /tmp/.env.clean
      set -a; source /tmp/.env.clean; set +a
    fi
    if [ -f "$ROOT_DIR/.env.local" ]; then set -a; source "$ROOT_DIR/.env.local"; set +a; fi
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export PYTHONWARNINGS="ignore"
    python -u train_models.py
    echo -e "${G}   ✅ Training complete${N}"

    echo -e "\n${Y}🔄 Syncing model config to DB...${N}"
    activate_venv "$ROOT_DIR/backendMulti"
    cd "$ROOT_DIR/backendMulti"
    python manage.py import_ml_config

    #dc -f "$COMPOSE_FILE" stop postgres mlflow
    ;;

  # ── SCAN (SonarQube) ──────────────────────
  scan)
    check_compose

    # Ensure .env.local exists
    if [ ! -f "$ROOT_DIR/.env.local" ]; then
      echo -e "${Y}📄 Creating .env.local from .env.example...${N}"
      cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env.local"
    fi

    # Reload .env.local
    set -a; source "$ROOT_DIR/.env.local"; set +a

    ensure_docker

    echo -e "${B}══════════════════════════════════════════════${N}"
    echo -e "${B}  🔍 Running SonarQube Analysis [${ENV}]      ${N}"
    echo -e "${B}══════════════════════════════════════════════${N}"

    # 1. Start SonarQube
    echo -e "\n${Y}🐳 Starting SonarQube services...${N}"
    dc -f "$COMPOSE_FILE" --profile sonar up -d

    # 2. Wait until ready
    echo -e "${Y}⏳ Waiting for SonarQube to be ready...${N}"
    until curl -sf http://localhost:9000/api/system/status | grep -q '"status":"UP"'; do
      printf "."
      sleep 5
    done
    echo -e "\n${G}   ✅ SonarQube ready${N}"

    # 3. Auto-generate token if missing
    setup_sonar_token

    # 4. Run scanner
    echo -e "\n${Y}🔬 Running scanner...${N}"
    dc -f "$COMPOSE_FILE" --profile scan run --rm scanner

    echo -e "\n${G}   ✅ Scan complete!${N}"
    echo -e "   📊 Results → http://localhost:9000/projects"
    ;;

  # ── RESTART ───────────────────────────────
  restart)
    check_compose
    ensure_docker
    echo -e "${Y}🔄 Restarting PIOS [${ENV}]...${N}"
    dc -f "$COMPOSE_FILE" down
    dc -f "$COMPOSE_FILE" up -d --build
    echo -e "${G}✅ Restarted.${N}"
    ;;

  # ── STATUS ────────────────────────────────
  status)
    echo -e "${B}📊 PIOS Container Status:${N}"
    docker ps --filter "name=pios" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    ;;

  # ── MIGRATE ───────────────────────────────
  migrate)
    check_compose
    ensure_docker
    echo -e "${Y}🐳 Ensuring database is running...${N}"
    dc -f "$COMPOSE_FILE" up -d postgres
    until dc -f "$COMPOSE_FILE" exec -T postgres pg_isready -U admin -d pios > /dev/null 2>&1; do
      sleep 2
    done

    echo -e "${Y}📦 Running Django migrations...${N}"
    activate_venv "$ROOT_DIR/backendMulti"
    cd "$ROOT_DIR/backendMulti"
    python manage.py migrate
    echo -e "${G}✅ Migrations complete.${N}"
    ;;

  # ── HELP ──────────────────────────────────
  help|--help|-h)
    show_help
    ;;

  *)
    echo -e "${R}❌ Unknown command: $1${N}"
    show_help
    exit 1
    ;;
esac
