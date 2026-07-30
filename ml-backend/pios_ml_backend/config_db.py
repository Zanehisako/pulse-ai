from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


DEFAULT_CONFIG_KEY = "runtime_models"
DEFAULT_CONFIG_CHANNEL = "pios_model_config_changed"
_VALID_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_MODEL_COLUMNS = {
    "model_id",
    "description",
    "file_path",
    "model_type",
    "features",
    "feature_info",
    "examples",
    "defaults",
    "enabled",
}


@dataclass
class ConfigSnapshot:
    payload: dict[str, Any]
    source: str
    version: str | None = None
    error: str | None = None


def get_db_backend() -> str:
    raw = (os.getenv("PIOS_CONFIG_DB_BACKEND") or "").strip().lower()
    if raw in {"postgres", "sqlite"}:
        return raw
    return "sqlite"


def supports_listen_notify() -> bool:
    return get_db_backend() == "postgres"


def _sqlite_db_path() -> Path:
    raw = os.getenv("PIOS_CONFIG_DB_PATH")
    if raw and raw.strip():
        return Path(raw).expanduser().resolve()
    return (_PROJECT_ROOT / "data" / "config.db").resolve()


def get_connection(*, autocommit: bool = False):
    backend = get_db_backend()
    if backend == "sqlite":
        db_path = _sqlite_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(db_path),
            timeout=max(1, int(os.getenv("DB_CONNECT_TIMEOUT", "3"))),
        )
        if autocommit:
            conn.isolation_level = None
        return conn

    host = os.getenv("DB_HOST") or os.getenv("PGHOST") or "localhost"
    port = int(os.getenv("DB_PORT") or os.getenv("PGPORT") or "5432")
    dbname = os.getenv("DB_NAME") or os.getenv("PGDATABASE") or "pios"
    user = os.getenv("DB_USER") or os.getenv("PGUSER") or "ml_user"
    password = os.getenv("DB_PASSWORD") or os.getenv("PGPASSWORD") or "ml_pass"
    connection = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        connect_timeout=max(1, int(os.getenv("DB_CONNECT_TIMEOUT", "3"))),
        options="-c search_path=ml_schema",
    )
    if autocommit:
        connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return connection


def _validate_identifier(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be empty.")
    if not _VALID_SQL_IDENTIFIER.match(normalized):
        raise ValueError(
            f"Invalid {label}: {value!r}. Use letters, digits, and underscores only."
        )
    return normalized


def _sqlite_table_columns(cur: sqlite3.Cursor, table_name: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table_name})")
    return {str(row[1]) for row in cur.fetchall() if len(row) > 1}


def _postgres_table_columns(cur, table_name: str) -> set[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'ml_schema' AND table_name = %s
        """,
        (table_name,),
    )
    return {str(row[0]) for row in cur.fetchall()}


def _normalize_runtime_model_row(raw_row: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw_row, dict):
        return None
    model_id = str(raw_row.get("id") or raw_row.get("model_id") or "").strip()
    if not model_id:
        return None
    description = str(raw_row.get("description", "")).strip()
    file_path = str(raw_row.get("file_path", "")).strip()
    model_type = str(raw_row.get("type") or raw_row.get("model_type") or "unknown").strip() or "unknown"
    features = raw_row.get("features", [])
    if not isinstance(features, list):
        features = []
    features = [str(item) for item in features if isinstance(item, str) and item.strip()]
    feature_info = raw_row.get("feature_info", {})
    if not isinstance(feature_info, dict):
        feature_info = {}
    examples = raw_row.get("examples", [])
    if not isinstance(examples, list):
        examples = []
    defaults = raw_row.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    enabled_raw = raw_row.get("enabled", raw_row.get("is_active", True))
    enabled = bool(enabled_raw)
    return {
        "id": model_id,
        "description": description,
        "file_path": file_path,
        "type": model_type,
        "features": features,
        "feature_info": feature_info,
        "examples": examples,
        "defaults": defaults,
        "enabled": enabled,
    }


def _normalize_runtime_models(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        model_row = _normalize_runtime_model_row(row)
        if model_row is None:
            continue
        model_id = str(model_row["id"]).lower()
        if model_id in seen:
            continue
        seen.add(model_id)
        normalized.append(model_row)
    return normalized


def _sync_runtime_models_table(cur, backend: str, models: list[dict[str, Any]]) -> None:
    if backend == "sqlite":
        ids = [row["id"] for row in models]
        if ids:
            placeholders = ",".join(["?"] * len(ids))
            cur.execute(
                f"UPDATE ml_models SET enabled = 0 WHERE model_id NOT IN ({placeholders})",
                ids,
            )
        else:
            cur.execute("UPDATE ml_models SET enabled = 0")
        for row in models:
            cur.execute(
                """
                INSERT INTO ml_models (
                    model_id, description, file_path, model_type,
                    features_json, feature_info_json, examples_json, defaults_json,
                    enabled, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                ON CONFLICT(model_id)
                DO UPDATE SET
                    description = excluded.description,
                    file_path = excluded.file_path,
                    model_type = excluded.model_type,
                    features_json = excluded.features_json,
                    feature_info_json = excluded.feature_info_json,
                    examples_json = excluded.examples_json,
                    defaults_json = excluded.defaults_json,
                    enabled = excluded.enabled,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                """,
                (
                    row["id"],
                    row["description"],
                    row["file_path"],
                    row["type"],
                    json.dumps(row["features"]),
                    json.dumps(row["feature_info"]),
                    json.dumps(row["examples"]),
                    json.dumps(row["defaults"]),
                    1 if row["enabled"] else 0,
                ),
            )
        return

    ids = [row["id"] for row in models]
    if ids:
        cur.execute(
            "UPDATE ml_models SET enabled = FALSE WHERE model_id <> ALL(%s)",
            (ids,),
        )
    else:
        cur.execute("UPDATE ml_models SET enabled = FALSE")
    for row in models:
        cur.execute(
            """
            INSERT INTO ml_models (
                model_id, description, file_path, model_type,
                features_json, feature_info_json, examples_json, defaults_json,
                enabled, updated_at
            )
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, NOW())
            ON CONFLICT (model_id)
            DO UPDATE SET
                description = EXCLUDED.description,
                file_path = EXCLUDED.file_path,
                model_type = EXCLUDED.model_type,
                features_json = EXCLUDED.features_json,
                feature_info_json = EXCLUDED.feature_info_json,
                examples_json = EXCLUDED.examples_json,
                defaults_json = EXCLUDED.defaults_json,
                enabled = EXCLUDED.enabled,
                updated_at = NOW()
            """,
            (
                row["id"],
                row["description"],
                row["file_path"],
                row["type"],
                json.dumps(row["features"]),
                json.dumps(row["feature_info"]),
                json.dumps(row["examples"]),
                json.dumps(row["defaults"]),
                bool(row["enabled"]),
            ),
        )


def _fetch_runtime_models_from_rows(cur, backend: str) -> tuple[list[dict[str, Any]], str | None]:
    if backend == "sqlite":
        cur.execute(
            """
            SELECT
                model_id,
                description,
                file_path,
                model_type,
                features_json,
                feature_info_json,
                examples_json,
                defaults_json,
                enabled,
                updated_at
            FROM ml_models
            WHERE enabled = 1
            ORDER BY model_id
            """
        )
    else:
        cur.execute(
            """
            SELECT
                model_id,
                description,
                file_path,
                model_type,
                features_json,
                feature_info_json,
                examples_json,
                defaults_json,
                enabled,
                updated_at
            FROM ml_models
            WHERE enabled = TRUE
            ORDER BY model_id
            """
        )

    rows = cur.fetchall()
    models: list[dict[str, Any]] = []
    latest_version: str | None = None
    for row in rows:
        features_raw = row[4]
        feature_info_raw = row[5]
        examples_raw = row[6]
        defaults_raw = row[7]
        updated_at = row[9]
        if updated_at is not None:
            latest_version = updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at)

        def _decode(raw: Any, fallback: Any):
            if isinstance(raw, (list, dict)):
                return raw
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    return parsed
                except json.JSONDecodeError:
                    return fallback
            return fallback

        features = _decode(features_raw, [])
        feature_info = _decode(feature_info_raw, {})
        examples = _decode(examples_raw, [])
        defaults = _decode(defaults_raw, {})
        if not isinstance(features, list):
            features = []
        if not isinstance(feature_info, dict):
            feature_info = {}
        if not isinstance(examples, list):
            examples = []
        if not isinstance(defaults, dict):
            defaults = {}

        models.append(
            {
                "id": str(row[0]),
                "description": str(row[1] or ""),
                "file_path": str(row[2] or ""),
                "type": str(row[3] or "unknown"),
                "features": [str(item) for item in features if isinstance(item, str)],
                "feature_info": feature_info,
                "examples": examples,
                "defaults": defaults,
                "enabled": bool(row[8]),
            }
        )
    return models, latest_version


def init_db(
    *,
    config_key: str = DEFAULT_CONFIG_KEY,
    channel: str = DEFAULT_CONFIG_CHANNEL,
) -> None:
    config_key = config_key.strip() or DEFAULT_CONFIG_KEY
    seed_payload = {"models": []}

    if get_db_backend() == "sqlite":
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    file_path TEXT NOT NULL DEFAULT '',
                    model_type TEXT NOT NULL DEFAULT 'unknown',
                    features_json TEXT NOT NULL DEFAULT '[]',
                    feature_info_json TEXT NOT NULL DEFAULT '{}',
                    examples_json TEXT NOT NULL DEFAULT '[]',
                    defaults_json TEXT NOT NULL DEFAULT '{}',
                    enabled INTEGER DEFAULT 1,
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                )
                """
            )
            existing_cols = _sqlite_table_columns(cur, "ml_models")
            if "model_id" not in existing_cols and {"name", "task"}.issubset(existing_cols):
                # Legacy schema migration path.
                cur.execute("ALTER TABLE ml_models RENAME TO ml_models_legacy")
                cur.execute(
                    """
                    CREATE TABLE ml_models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL UNIQUE,
                        description TEXT NOT NULL DEFAULT '',
                        file_path TEXT NOT NULL DEFAULT '',
                        model_type TEXT NOT NULL DEFAULT 'unknown',
                        features_json TEXT NOT NULL DEFAULT '[]',
                        feature_info_json TEXT NOT NULL DEFAULT '{}',
                        examples_json TEXT NOT NULL DEFAULT '[]',
                        defaults_json TEXT NOT NULL DEFAULT '{}',
                        enabled INTEGER DEFAULT 1,
                        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO ml_models(model_id, description, model_type, enabled)
                    SELECT name, '', task, enabled
                    FROM ml_models_legacy
                    """
                )
                cur.execute("DROP TABLE ml_models_legacy")
                existing_cols = _sqlite_table_columns(cur, "ml_models")

            sqlite_additions = {
                "description": "ALTER TABLE ml_models ADD COLUMN description TEXT NOT NULL DEFAULT ''",
                "file_path": "ALTER TABLE ml_models ADD COLUMN file_path TEXT NOT NULL DEFAULT ''",
                "model_type": "ALTER TABLE ml_models ADD COLUMN model_type TEXT NOT NULL DEFAULT 'unknown'",
                "features_json": "ALTER TABLE ml_models ADD COLUMN features_json TEXT NOT NULL DEFAULT '[]'",
                "feature_info_json": "ALTER TABLE ml_models ADD COLUMN feature_info_json TEXT NOT NULL DEFAULT '{}'",
                "examples_json": "ALTER TABLE ml_models ADD COLUMN examples_json TEXT NOT NULL DEFAULT '[]'",
                "defaults_json": "ALTER TABLE ml_models ADD COLUMN defaults_json TEXT NOT NULL DEFAULT '{}'",
                "updated_at": "ALTER TABLE ml_models ADD COLUMN updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
            }
            for col, ddl in sqlite_additions.items():
                if col not in existing_cols:
                    cur.execute(ddl)

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_model_config (
                    config_key TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                )
                """
            )
            cur.execute(
                """
                INSERT OR IGNORE INTO ml_model_config (config_key, config_json)
                VALUES (?, ?)
                """,
                (config_key, json.dumps(seed_payload)),
            )
            conn.commit()
        finally:
            conn.close()
        return

    channel = _validate_identifier(channel, "notification channel")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_models (
                id SERIAL PRIMARY KEY,
                model_id TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                file_path TEXT NOT NULL DEFAULT '',
                model_type TEXT NOT NULL DEFAULT 'unknown',
                features_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                feature_info_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                examples_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                defaults_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                enabled BOOLEAN DEFAULT TRUE,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        existing_cols = _postgres_table_columns(cur, "ml_models")
        postgres_additions = {
            "model_id": "ALTER TABLE ml_models ADD COLUMN model_id TEXT",
            "description": "ALTER TABLE ml_models ADD COLUMN description TEXT NOT NULL DEFAULT ''",
            "file_path": "ALTER TABLE ml_models ADD COLUMN file_path TEXT NOT NULL DEFAULT ''",
            "model_type": "ALTER TABLE ml_models ADD COLUMN model_type TEXT NOT NULL DEFAULT 'unknown'",
            "features_json": "ALTER TABLE ml_models ADD COLUMN features_json JSONB NOT NULL DEFAULT '[]'::jsonb",
            "feature_info_json": "ALTER TABLE ml_models ADD COLUMN feature_info_json JSONB NOT NULL DEFAULT '{}'::jsonb",
            "examples_json": "ALTER TABLE ml_models ADD COLUMN examples_json JSONB NOT NULL DEFAULT '[]'::jsonb",
            "defaults_json": "ALTER TABLE ml_models ADD COLUMN defaults_json JSONB NOT NULL DEFAULT '{}'::jsonb",
            "updated_at": "ALTER TABLE ml_models ADD COLUMN updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
        }
        for col, ddl in postgres_additions.items():
            if col not in existing_cols:
                cur.execute(ddl)
        # Legacy schema support: promote name/task columns to model_id/model_type if needed.
        if "name" in existing_cols and "model_id" in _postgres_table_columns(cur, "ml_models"):
            cur.execute("UPDATE ml_models SET model_id = COALESCE(NULLIF(model_id, ''), name)")
        if "task" in existing_cols and "model_type" in _postgres_table_columns(cur, "ml_models"):
            cur.execute("UPDATE ml_models SET model_type = COALESCE(NULLIF(model_type, ''), task)")
        cur.execute("ALTER TABLE ml_models ALTER COLUMN model_id SET NOT NULL")
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_indexes
                    WHERE schemaname = 'ml_schema' AND indexname = 'ml_models_model_id_key'
                ) THEN
                    EXECUTE 'CREATE UNIQUE INDEX ml_models_model_id_key ON ml_models(model_id)';
                END IF;
            END
            $$;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_model_config (
                config_key TEXT PRIMARY KEY,
                config_json JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            INSERT INTO ml_model_config (config_key, config_json)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (config_key) DO NOTHING
            """,
            (config_key, json.dumps(seed_payload)),
        )
        cur.execute(
            f"""
            CREATE OR REPLACE FUNCTION notify_model_config_change()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            DECLARE
                payload TEXT;
            BEGIN
                payload := json_build_object(
                    'operation', TG_OP,
                    'config_key', COALESCE(NEW.config_key, OLD.config_key),
                    'updated_at', NOW()
                )::text;
                PERFORM pg_notify('{channel}', payload);
                RETURN COALESCE(NEW, OLD);
            END;
            $$;
            """
        )
        cur.execute(
            """
            DROP TRIGGER IF EXISTS trg_notify_model_config_change ON ml_model_config
            """
        )
        cur.execute(
            """
            CREATE TRIGGER trg_notify_model_config_change
            AFTER INSERT OR UPDATE OR DELETE ON ml_model_config
            FOR EACH ROW
            EXECUTE FUNCTION notify_model_config_change()
            """
        )
        conn.commit()
    finally:
        conn.close()


def fetch_model_config(
    *,
    config_key: str = DEFAULT_CONFIG_KEY,
) -> tuple[dict[str, Any] | None, str | None]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        if get_db_backend() == "sqlite":
            cur.execute(
                """
                SELECT config_json, updated_at
                FROM ml_model_config
                WHERE config_key = ?
                """,
                (config_key,),
            )
        else:
            cur.execute(
                """
                SELECT config_json, updated_at
                FROM ml_model_config
                WHERE config_key = %s
                """,
                (config_key,),
            )
        row = cur.fetchone()
        if row is None:
            return None, None
        raw_payload = row[0]
        payload: dict[str, Any] | None = None
        if isinstance(raw_payload, dict):
            payload = raw_payload
        elif isinstance(raw_payload, str):
            try:
                decoded = json.loads(raw_payload)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                payload = decoded
        if payload is None:
            version = None
            if row[1] is not None:
                version = row[1].isoformat() if hasattr(row[1], "isoformat") else str(row[1])
            return None, version
        version = None
        if row[1] is not None:
            version = row[1].isoformat() if hasattr(row[1], "isoformat") else str(row[1])
        return payload, version
    finally:
        conn.close()


def load_model_config_snapshot(
    *,
    config_key: str = DEFAULT_CONFIG_KEY,
) -> ConfigSnapshot:
    source = get_db_backend()
    backend = get_db_backend()
    try:
        if config_key == DEFAULT_CONFIG_KEY:
            conn = get_connection()
            try:
                cur = conn.cursor()
                row_models, row_version = _fetch_runtime_models_from_rows(cur, backend)
            finally:
                conn.close()
            if row_models:
                return ConfigSnapshot(
                    payload={"models": row_models},
                    source=f"{source}:ml_models_rows",
                    version=row_version,
                )

        payload, version = fetch_model_config(config_key=config_key)
        if isinstance(payload, dict):
            models = payload.get("models")
            if not isinstance(models, list):
                payload["models"] = []
            return ConfigSnapshot(
                payload=payload,
                source=source,
                version=version,
            )
        return ConfigSnapshot(
            payload={"models": []},
            source=source,
            version=version,
            error=f"No config row found for key '{config_key}'.",
        )
    except Exception as exc:  # noqa: BLE001
        db_error = f"{type(exc).__name__}: {exc}"
    return ConfigSnapshot(
        payload={"models": []},
        source=source,
        version=None,
        error=db_error,
    )


def upsert_model_config(
    *,
    payload: dict[str, Any],
    config_key: str = DEFAULT_CONFIG_KEY,
) -> str | None:
    runtime_models = _normalize_runtime_models(payload.get("models", []))
    if config_key == DEFAULT_CONFIG_KEY:
        payload = {"models": runtime_models}

    conn = get_connection()
    try:
        cur = conn.cursor()
        backend = get_db_backend()
        if config_key == DEFAULT_CONFIG_KEY:
            _sync_runtime_models_table(cur, backend, runtime_models)

        if backend == "sqlite":
            cur.execute(
                """
                INSERT INTO ml_model_config (config_key, config_json, updated_at)
                VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                ON CONFLICT (config_key)
                DO UPDATE SET
                    config_json = excluded.config_json,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                """,
                (config_key, json.dumps(payload)),
            )
            cur.execute(
                """
                SELECT updated_at FROM ml_model_config WHERE config_key = ?
                """,
                (config_key,),
            )
            row = cur.fetchone()
        else:
            cur.execute(
                """
                INSERT INTO ml_model_config (config_key, config_json, updated_at)
                VALUES (%s, %s::jsonb, NOW())
                ON CONFLICT (config_key)
                DO UPDATE SET
                    config_json = EXCLUDED.config_json,
                    updated_at = NOW()
                RETURNING updated_at
                """,
                (config_key, json.dumps(payload)),
            )
            row = cur.fetchone()
        conn.commit()
        if row is None or row[0] is None:
            return None
        return row[0].isoformat() if hasattr(row[0], "isoformat") else str(row[0])
    finally:
        conn.close()


def ensure_model_config(
    *,
    payload: dict[str, Any],
    config_key: str,
) -> bool:
    runtime_models = _normalize_runtime_models(payload.get("models", []))
    if config_key == DEFAULT_CONFIG_KEY:
        payload = {"models": runtime_models}

    conn = get_connection()
    try:
        cur = conn.cursor()
        backend = get_db_backend()
        if config_key == DEFAULT_CONFIG_KEY and runtime_models:
            _sync_runtime_models_table(cur, backend, runtime_models)

        if backend == "sqlite":
            cur.execute(
                """
                INSERT OR IGNORE INTO ml_model_config (config_key, config_json)
                VALUES (?, ?)
                """,
                (config_key, json.dumps(payload)),
            )
            inserted = cur.rowcount > 0
        else:
            cur.execute(
                """
                INSERT INTO ml_model_config (config_key, config_json)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (config_key) DO NOTHING
                RETURNING config_key
                """,
                (config_key, json.dumps(payload)),
            )
            row = cur.fetchone()
            inserted = row is not None
        conn.commit()
        return inserted
    finally:
        conn.close()


def open_listen_connection(channel: str):
    if not supports_listen_notify():
        raise RuntimeError("LISTEN/NOTIFY is only supported with postgres backend.")
    normalized_channel = _validate_identifier(channel, "notification channel")
    conn = get_connection(autocommit=True)
    cur = conn.cursor()
    cur.execute(f"LISTEN {normalized_channel};")
    cur.close()
    return conn
