from __future__ import annotations

import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schemas import DataSnapshotRequest

STUDIO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDIO_ROOT.parent
BACKEND_ROOT = REPO_ROOT / "backendMulti"
ML_MODELS_ROOT = REPO_ROOT / "ml-backend" / "models"
NOTEBOOK_MODELS_ROOT = REPO_ROOT / "ml-backend" / "notebooks" / "models"
SIMULATOR_MODELS_ROOT = STUDIO_ROOT / "simulator"
VALID_BLOOD_TYPES = ("O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-")
DEFAULT_DONOR_MODEL = "authApp.User"
DEFAULT_DONOR_TABLE = "donneur"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "active", "eligible"}:
        return True
    if lowered in {"0", "false", "no", "n", "inactive", "ineligible"}:
        return False
    return default


def parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raw = str(value).strip()
    if not raw:
        return None
    raw = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def iso_or_none(value: Any) -> str | None:
    parsed = parse_dt(value)
    return parsed.replace(microsecond=0).isoformat() if parsed else None


def recency_days(value: Any, *, as_of: datetime | None = None) -> float | None:
    parsed = parse_dt(value)
    if parsed is None:
        return None
    anchor = as_of or datetime.now(timezone.utc)
    delta = anchor - parsed
    return max(delta.total_seconds() / 86_400.0, 0.0)


def choose_value(row: dict[str, Any], *candidates: str) -> Any:
    lowered = {str(key).lower(): key for key in row}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None:
            return row.get(key)
    return None


def normalize_blood_type(value: Any, *, fallback_seed: int = 0) -> str:
    if value is not None:
        normalized = str(value).upper().replace(" ", "")
        if normalized in VALID_BLOOD_TYPES:
            return normalized
    return VALID_BLOOD_TYPES[fallback_seed % len(VALID_BLOOD_TYPES)]


def estimate_dropout_risk(row: dict[str, Any], last_donation_days: float | None) -> float:
    explicit = choose_value(
        row,
        "dropout_risk",
        "attrition_risk",
        "churn_risk",
        "risk_score",
    )
    if explicit is not None:
        return clamp(safe_float(explicit), 0.0, 1.0)

    donation_count = safe_int(
        choose_value(
            row,
            "donation_count",
            "total_donations",
            "completed_donations",
            "lifetime_donations",
        ),
        0,
    )
    regular = safe_bool(
        choose_value(row, "is_regular_donor", "regular_donor", "is_regular"),
        donation_count >= 3,
    )
    age_penalty = 0.0
    if last_donation_days is not None:
        if last_donation_days > 365:
            age_penalty += 0.30
        elif last_donation_days > 180:
            age_penalty += 0.15
        elif last_donation_days < 56:
            age_penalty += 0.08
    loyalty_credit = 0.18 if regular else 0.0
    volume_credit = min(donation_count * 0.02, 0.18)
    active_penalty = 0.25 if safe_bool(choose_value(row, "is_active"), True) else 0.45
    risk = 0.42 + age_penalty + active_penalty - loyalty_credit - volume_credit
    return clamp(risk, 0.05, 0.95)


def infer_eligibility(row: dict[str, Any], last_donation_days: float | None) -> bool:
    next_eligible = choose_value(
        row,
        "prochain_don_eligible",
        "next_eligible_donation",
        "eligible_after",
    )
    next_eligible_dt = parse_dt(next_eligible)
    if next_eligible_dt is not None:
        return next_eligible_dt <= datetime.now(timezone.utc)
    explicit = choose_value(
        row,
        "is_eligible",
        "eligible",
        "can_donate",
        "donor_eligible",
    )
    if explicit is not None:
        return safe_bool(explicit, True)
    status = str(
        choose_value(row, "statut", "status", "eligibility_status", "state") or ""
    ).strip().lower()
    if status in {"eligible", "actif", "active"}:
        return True
    if status in {"non_eligible", "suspendu", "suspended", "inactif", "inactive"}:
        return False
    if last_donation_days is not None and last_donation_days < 56:
        return False
    return safe_bool(choose_value(row, "is_active"), True)


@dataclass(frozen=True)
class DonorRecord:
    donor_id: str
    display_name: str
    blood_type: str
    eligible: bool
    regular_donor: bool
    donation_count: int
    dropout_risk: float
    recency_days: float | None
    last_donation_at: str | None
    latitude: float | None = None
    longitude: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "donor_id": self.donor_id,
            "display_name": self.display_name,
            "blood_type": self.blood_type,
            "eligible": self.eligible,
            "regular_donor": self.regular_donor,
            "donation_count": self.donation_count,
            "dropout_risk": round(self.dropout_risk, 4),
            "recency_days": round(self.recency_days, 2)
            if self.recency_days is not None
            else None,
            "last_donation_at": self.last_donation_at,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class HistoricalEventRecord:
    key: str
    title: str
    severity: float
    occurred_at: str | None
    category: str
    modifiers: dict[str, float] = field(default_factory=dict)
    forced_weather: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "severity": round(self.severity, 4),
            "occurred_at": self.occurred_at,
            "category": self.category,
            "modifiers": dict(self.modifiers),
            "forced_weather": self.forced_weather,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_id: str
    description: str
    path: str
    model_type: str
    is_active: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "description": self.description,
            "path": self.path,
            "model_type": self.model_type,
            "is_active": self.is_active,
        }


@dataclass(frozen=True)
class SimulationSnapshot:
    snapshot_id: str
    generated_at: str
    source: str
    donor_records: list[DonorRecord]
    historical_events: list[HistoricalEventRecord]
    model_registry: list[ModelRegistryEntry]
    donor_summary: dict[str, Any]
    calibration: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "generated_at": self.generated_at,
            "source": self.source,
            "donor_records": [row.as_dict() for row in self.donor_records],
            "historical_events": [row.as_dict() for row in self.historical_events],
            "model_registry": [row.as_dict() for row in self.model_registry],
            "donor_summary": dict(self.donor_summary),
            "calibration": dict(self.calibration),
            "warnings": list(self.warnings),
        }


def build_donor_summary(
    donors: list[DonorRecord],
    events: list[HistoricalEventRecord],
    models: list[ModelRegistryEntry],
) -> tuple[dict[str, Any], dict[str, Any]]:
    donor_count = len(donors)
    if not donors:
        summary = {
            "total_donors": 0,
            "eligible_rate": 0.0,
            "regular_donor_share": 0.0,
            "average_dropout_risk": 0.0,
            "active_last_180d_share": 0.0,
            "blood_type_distribution": {},
            "critical_event_count": len(
                [event for event in events if event.severity >= 0.75]
            ),
            "available_model_count": len(models),
        }
        calibration = {
            "estimated_donor_show_factor": 0.85,
            "estimated_eligible_rate": 0.85,
            "estimated_donor_inter_arrival_h": 6.0,
            "alert_pressure": 0.0,
        }
        return summary, calibration

    eligible_rate = sum(1 for donor in donors if donor.eligible) / donor_count
    regular_share = sum(1 for donor in donors if donor.regular_donor) / donor_count
    avg_dropout = sum(donor.dropout_risk for donor in donors) / donor_count
    active_share = (
        sum(
            1
            for donor in donors
            if donor.recency_days is None or donor.recency_days <= 180.0
        )
        / donor_count
    )
    blood_counts = Counter(donor.blood_type for donor in donors)
    critical_event_count = len([event for event in events if event.severity >= 0.75])
    alert_pressure = critical_event_count / max(len(events), 1)
    estimated_show_factor = clamp(
        0.70 + (regular_share * 0.22) + (active_share * 0.14) - (avg_dropout * 0.28),
        0.45,
        1.25,
    )
    estimated_eligible_rate = clamp((eligible_rate * 0.85) + (active_share * 0.10), 0.35, 0.99)
    estimated_daily_attempts = max(
        donor_count * max(estimated_show_factor * estimated_eligible_rate, 0.12) / 45.0,
        0.5,
    )
    estimated_inter_arrival_h = clamp(24.0 / estimated_daily_attempts, 0.20, 48.0)

    summary = {
        "total_donors": donor_count,
        "eligible_rate": round(eligible_rate, 4),
        "regular_donor_share": round(regular_share, 4),
        "average_dropout_risk": round(avg_dropout, 4),
        "active_last_180d_share": round(active_share, 4),
        "blood_type_distribution": dict(sorted(blood_counts.items())),
        "critical_event_count": critical_event_count,
        "available_model_count": len(models),
    }
    calibration = {
        "estimated_donor_show_factor": round(estimated_show_factor, 4),
        "estimated_eligible_rate": round(estimated_eligible_rate, 4),
        "estimated_donor_inter_arrival_h": round(estimated_inter_arrival_h, 4),
        "alert_pressure": round(alert_pressure, 4),
    }
    return summary, calibration


def synthetic_donor_records(limit: int) -> list[DonorRecord]:
    donors: list[DonorRecord] = []
    for index in range(limit):
        blood_type = VALID_BLOOD_TYPES[index % len(VALID_BLOOD_TYPES)]
        regular = index % 4 in {0, 1}
        recency = 30.0 + (index % 12) * 18.0
        donors.append(
            DonorRecord(
                donor_id=f"synthetic-{index + 1}",
                display_name=f"Synthetic Donor {index + 1}",
                blood_type=blood_type,
                eligible=recency >= 56.0,
                regular_donor=regular,
                donation_count=2 + (index % 7),
                dropout_risk=clamp(0.18 + ((index % 9) * 0.06), 0.05, 0.92),
                recency_days=recency,
                last_donation_at=None,
                latitude=46.80 + ((index % 10) * 0.003),
                longitude=-71.28 + ((index % 8) * 0.004),
                attributes={"source": "synthetic"},
            )
        )
    return donors


def local_model_registry(limit: int = 24) -> list[ModelRegistryEntry]:
    rows: list[ModelRegistryEntry] = []
    seen: set[str] = set()
    for root in (ML_MODELS_ROOT, NOTEBOOK_MODELS_ROOT, SIMULATOR_MODELS_ROOT):
        if not root.exists():
            continue
        for path in sorted(root.glob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".pkl", ".pt", ".zip", ".gguf"}:
                continue
            model_id = path.stem
            if model_id in seen:
                continue
            seen.add(model_id)
            rows.append(
                ModelRegistryEntry(
                    model_id=model_id,
                    description="Local model artifact discovered from the ML backend.",
                    path=str(path),
                    model_type=path.suffix.lower().lstrip("."),
                    is_active=True,
                )
            )
            if len(rows) >= limit:
                return rows
    return rows


def synthetic_snapshot(request: DataSnapshotRequest, warnings: list[str] | None = None) -> SimulationSnapshot:
    donors = synthetic_donor_records(request.donor_limit if request.include_raw_donors else 64)
    if not request.include_raw_donors:
        donors = donors[: min(len(donors), 64)]
    models = local_model_registry() if request.include_model_registry else []
    events: list[HistoricalEventRecord] = []
    summary, calibration = build_donor_summary(donors, events, models)
    return SimulationSnapshot(
        snapshot_id=f"snapshot-{int(datetime.now(timezone.utc).timestamp())}",
        generated_at=utc_now_iso(),
        source="synthetic",
        donor_records=donors if request.include_raw_donors else [],
        historical_events=events,
        model_registry=models,
        donor_summary=summary,
        calibration=calibration,
        warnings=list(warnings or []),
    )


@lru_cache(maxsize=1)
def get_django_apps():
    if not BACKEND_ROOT.exists():
        raise RuntimeError(f"Backend workspace is unavailable at {BACKEND_ROOT}.")
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
    import django

    django.setup()
    from django.apps import apps

    return apps


def get_django_connection():
    get_django_apps()
    from django.db import connection

    return connection


def sanitize_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise RuntimeError(f"Unsafe SQL identifier: {value!r}")
    return value


def table_exists(table_name: str) -> bool:
    connection = get_django_connection()
    sanitized = sanitize_identifier(table_name)
    return sanitized in set(connection.introspection.table_names())


def sql_table_description(table_name: str) -> list[str]:
    connection = get_django_connection()
    quoted_table = connection.ops.quote_name(sanitize_identifier(table_name))
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT * FROM {quoted_table} LIMIT 0")
        return [column[0] for column in cursor.description or []]


def donor_table_name(request: DataSnapshotRequest) -> str:
    return (
        request.donor_table
        or os.environ.get("PIOS_SNAPSHOT_DONOR_TABLE")
        or DEFAULT_DONOR_TABLE
    )


def detect_snapshot_backend() -> dict[str, Any]:
    if not BACKEND_ROOT.exists():
        return {
            "source": "synthetic",
            "adapter": "standalone",
            "label": "Synthetic snapshot fallback (backend workspace unavailable)",
            "donor_table": None,
            "donor_model": None,
        }
    table_name = os.environ.get("PIOS_SNAPSHOT_DONOR_TABLE") or DEFAULT_DONOR_TABLE
    model_path = os.environ.get("PIOS_SNAPSHOT_DONOR_MODEL") or DEFAULT_DONOR_MODEL
    try:
        if table_exists(table_name):
            return {
                "source": "django",
                "adapter": "sql_table",
                "label": f"Raw donor table: {table_name}",
                "donor_table": table_name,
                "donor_model": None,
            }
    except Exception:
        pass
    return {
        "source": "django",
        "adapter": "django_model",
        "label": f"Django model: {model_path}",
        "donor_table": table_name,
        "donor_model": model_path,
    }


def load_django_donors(request: DataSnapshotRequest) -> list[DonorRecord]:
    apps = get_django_apps()
    model_path = request.donor_model or os.environ.get(
        "PIOS_SNAPSHOT_DONOR_MODEL", DEFAULT_DONOR_MODEL
    )
    if "." not in model_path:
        raise RuntimeError(
            f"Invalid donor model path '{model_path}'. Expected 'app_label.ModelName'."
        )
    app_label, model_name = model_path.split(".", 1)
    model = apps.get_model(app_label, model_name)
    fields = [
        field.name
        for field in model._meta.get_fields()
        if getattr(field, "concrete", False) and not getattr(field, "many_to_many", False)
    ]
    order_field = next(
        (
            field_name
            for field_name in ("updated_at", "created_at", "date_joined", "id")
            if field_name in fields
        ),
        None,
    )
    queryset = model.objects.all()
    as_of = parse_dt(request.as_of)
    if as_of and order_field in {"updated_at", "created_at", "date_joined"}:
        queryset = queryset.filter(**{f"{order_field}__lte": as_of})
    if order_field is not None:
        queryset = queryset.order_by(f"-{order_field}")
    rows = list(queryset.values(*fields)[: request.donor_limit])

    donors: list[DonorRecord] = []
    for index, row in enumerate(rows):
        last_donation = choose_value(
            row,
            "last_donation_at",
            "last_donation_date",
            "dernier_don",
            "donated_at",
            "last_seen_at",
            "updated_at",
            "date_joined",
        )
        last_donation_days = recency_days(last_donation, as_of=as_of)
        donation_count = safe_int(
            choose_value(
                row,
                "donation_count",
                "total_donations",
                "completed_donations",
                "lifetime_donations",
                "nombre_total_dons",
            ),
            0,
        )
        regular = safe_bool(
            choose_value(row, "is_regular_donor", "regular_donor", "is_regular"),
            donation_count >= 3,
        )
        donor_id_raw = choose_value(
            row,
            "id",
            "uuid",
            "identifier",
            "donor_id",
            "user_id",
            "id_donneur",
            "numero_donneur",
            "email",
        )
        first_name = choose_value(row, "prenom", "first_name")
        last_name = choose_value(row, "nom", "last_name", "surname")
        display_name = str(
            " ".join(part for part in [first_name, last_name] if part).strip()
            or choose_value(row, "name", "display_name", "full_name", "email")
            or f"Donor {index + 1}"
        )
        donors.append(
            DonorRecord(
                donor_id=str(donor_id_raw or f"donor-{index + 1}"),
                display_name=display_name,
                blood_type=normalize_blood_type(
                    choose_value(
                        row,
                        "blood_type",
                        "blood_group",
                        "group",
                        "groupe_sanguin",
                    ),
                    fallback_seed=index,
                ),
                eligible=infer_eligibility(row, last_donation_days),
                regular_donor=regular,
                donation_count=donation_count,
                dropout_risk=estimate_dropout_risk(row, last_donation_days),
                recency_days=last_donation_days,
                last_donation_at=iso_or_none(last_donation),
                latitude=(
                    safe_float(choose_value(row, "latitude", "lat"), 0.0) or None
                ),
                longitude=(
                    safe_float(choose_value(row, "longitude", "lon", "lng"), 0.0)
                    or None
                ),
                attributes={
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "id",
                        "uuid",
                        "identifier",
                        "donor_id",
                        "user_id",
                        "name",
                        "display_name",
                        "full_name",
                        "email",
                    }
                },
            )
        )
    return donors


def load_sql_donors(request: DataSnapshotRequest, table_name: str) -> list[DonorRecord]:
    connection = get_django_connection()
    sanitized = sanitize_identifier(table_name)
    quoted_table = connection.ops.quote_name(sanitized)
    columns = sql_table_description(sanitized)
    if not columns:
        return []

    order_field = next(
        (
            field_name
            for field_name in (
                "date_inscription",
                "dernier_don",
                "updated_at",
                "created_at",
                "id_donneur",
            )
            if field_name in columns
        ),
        columns[0],
    )

    as_of = parse_dt(request.as_of)
    where_parts: list[str] = []
    params: list[Any] = []
    if as_of is not None and order_field in {
        "date_inscription",
        "dernier_don",
        "updated_at",
        "created_at",
    }:
        where_parts.append(f"{connection.ops.quote_name(order_field)} <= %s")
        params.append(as_of)

    where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
    select_columns = ", ".join(connection.ops.quote_name(column) for column in columns)
    sql = (
        f"SELECT {select_columns} FROM {quoted_table}"
        f"{where_sql} ORDER BY {connection.ops.quote_name(order_field)} DESC NULLS LAST LIMIT %s"
    )
    params.append(int(request.donor_limit))

    donors: list[DonorRecord] = []
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        result_columns = [column[0] for column in cursor.description or []]

    for index, raw_row in enumerate(rows):
        row = {result_columns[pos]: raw_row[pos] for pos in range(len(result_columns))}
        last_donation = choose_value(row, "dernier_don", "last_donation_at", "last_donation_date")
        last_donation_days = recency_days(last_donation, as_of=as_of)
        donation_count = safe_int(
            choose_value(row, "nombre_total_dons", "donation_count", "total_donations"),
            0,
        )
        regular = donation_count >= 3
        donor_id_raw = choose_value(row, "numero_donneur", "id_donneur", "email")
        display_name = " ".join(
            part
            for part in [
                str(choose_value(row, "prenom") or "").strip(),
                str(choose_value(row, "nom") or "").strip(),
            ]
            if part
        ).strip()
        consent = safe_bool(choose_value(row, "consentement_email", "consentement_sms"), True)
        dropout = estimate_dropout_risk(row, last_donation_days)
        if not consent:
            dropout = clamp(dropout + 0.08, 0.0, 1.0)
        donors.append(
            DonorRecord(
                donor_id=str(donor_id_raw or f"donor-{index + 1}"),
                display_name=display_name or f"Donor {index + 1}",
                blood_type=normalize_blood_type(
                    choose_value(row, "groupe_sanguin", "blood_type"),
                    fallback_seed=index,
                ),
                eligible=infer_eligibility(row, last_donation_days),
                regular_donor=regular,
                donation_count=donation_count,
                dropout_risk=dropout,
                recency_days=last_donation_days,
                last_donation_at=iso_or_none(last_donation),
                latitude=(
                    safe_float(choose_value(row, "latitude"), 0.0) or None
                ),
                longitude=(
                    safe_float(choose_value(row, "longitude"), 0.0) or None
                ),
                attributes={
                    "city": choose_value(row, "ville", "city"),
                    "postal_code": choose_value(row, "code_postal", "postal_code"),
                    "status": choose_value(row, "statut", "status"),
                    "next_eligible_at": iso_or_none(
                        choose_value(row, "prochain_don_eligible")
                    ),
                },
            )
        )
    return donors


def alert_event_modifiers(title: str, message: str, severity: float) -> tuple[dict[str, float], str | None]:
    text = f"{title} {message}".lower()
    modifiers: dict[str, float] = {}
    forced_weather: str | None = None
    if any(token in text for token in ("stock", "shortage", "supply", "critical")):
        modifiers["demand_surge_factor"] = round(1.0 + (severity * 0.35), 4)
        modifiers["reserve_target_days"] = round(1.0 + (severity * 0.12), 4)
    if any(token in text for token in ("transport", "courier", "road", "route")):
        modifiers["transport_penalty"] = round(1.0 + (severity * 0.50), 4)
        modifiers["regional_replenishment_rate"] = round(max(0.45, 1.0 - (severity * 0.35)), 4)
    if any(token in text for token in ("lab", "processing", "test", "screening")):
        modifiers["lab_time_factor"] = round(1.0 + (severity * 0.45), 4)
        modifiers["proc_time_factor"] = round(1.0 + (severity * 0.25), 4)
    if any(token in text for token in ("snow", "blizzard", "ice storm", "storm")):
        forced_weather = "snow" if "snow" in text else "blizzard"
        modifiers["transport_penalty"] = round(max(modifiers.get("transport_penalty", 1.0), 1.0 + (severity * 0.60)), 4)
    if any(token in text for token in ("donor", "dropout", "attendance", "no-show")):
        modifiers["donor_show_factor"] = round(max(0.55, 1.0 - (severity * 0.22)), 4)
        modifiers["donor_inter_arrival_h"] = round(1.0 + (severity * 0.28), 4)
    return modifiers, forced_weather


def load_django_events(limit: int) -> list[HistoricalEventRecord]:
    apps = get_django_apps()
    try:
        model = apps.get_model("alerts", "AlertEvent")
    except LookupError:
        return []
    rows = list(
        model.objects.all()
        .order_by("-opened_at")
        .values(
            "event_key",
            "title",
            "message",
            "severity",
            "status",
            "source_type",
            "entity_type",
            "blood_type",
            "opened_at",
            "context",
        )[:limit]
    )
    severity_map = {"warning": 0.45, "critical": 0.85, "resolved": 0.15}
    events: list[HistoricalEventRecord] = []
    for row in rows:
        severity = severity_map.get(str(row.get("severity", "")).lower(), 0.35)
        modifiers, forced_weather = alert_event_modifiers(
            str(row.get("title", "")),
            str(row.get("message", "")),
            severity,
        )
        category = str(row.get("entity_type") or row.get("source_type") or "alert")
        events.append(
            HistoricalEventRecord(
                key=str(row.get("event_key") or row.get("title") or "alert"),
                title=str(row.get("title") or "Historical alert"),
                severity=severity,
                occurred_at=iso_or_none(row.get("opened_at")),
                category=category,
                modifiers=modifiers,
                forced_weather=forced_weather,
                metadata={
                    "status": row.get("status"),
                    "blood_type": row.get("blood_type"),
                    "context": row.get("context") or {},
                },
            )
        )
    return events


def load_django_model_registry() -> list[ModelRegistryEntry]:
    apps = get_django_apps()
    try:
        model = apps.get_model("ml", "MLModelConfig")
    except LookupError:
        return local_model_registry()
    rows = list(
        model.objects.all()
        .order_by("model_id")
        .values("model_id", "description", "file_path", "model_type", "is_active")
    )
    if not rows:
        return local_model_registry()
    return [
        ModelRegistryEntry(
            model_id=str(row.get("model_id")),
            description=str(row.get("description") or ""),
            path=str(row.get("file_path") or ""),
            model_type=str(row.get("model_type") or "unknown"),
            is_active=safe_bool(row.get("is_active"), True),
        )
        for row in rows
    ]


def capture_data_snapshot(request: DataSnapshotRequest) -> SimulationSnapshot:
    if request.source == "synthetic":
        return synthetic_snapshot(request)

    warnings: list[str] = []
    try:
        table_name = donor_table_name(request)
        donors: list[DonorRecord]
        used_adapter = "django_model"
        if request.donor_model and request.donor_model.startswith("table:"):
            table_name = request.donor_model.split(":", 1)[1]
            donors = load_sql_donors(request, table_name)
            used_adapter = "sql_table"
        elif request.donor_table is not None:
            donors = load_sql_donors(request, table_name)
            used_adapter = "sql_table"
        elif table_exists(table_name):
            donors = load_sql_donors(request, table_name)
            used_adapter = "sql_table"
        else:
            donors = load_django_donors(request)
        events = (
            load_django_events(limit=24) if request.include_alert_events else []
        )
        models = load_django_model_registry() if request.include_model_registry else []
        summary, calibration = build_donor_summary(donors, events, models)
        if used_adapter == "sql_table":
            warnings.append(f"Snapshot used raw donor table '{table_name}'.")
        return SimulationSnapshot(
            snapshot_id=f"snapshot-{int(datetime.now(timezone.utc).timestamp())}",
            generated_at=utc_now_iso(),
            source="django",
            donor_records=donors if request.include_raw_donors else [],
            historical_events=events,
            model_registry=models,
            donor_summary=summary,
            calibration=calibration,
            warnings=warnings,
        )
    except Exception as exc:
        warnings.append(
            "Django/Postgres snapshot failed; synthetic fallback was used instead: "
            f"{exc}"
        )
        return synthetic_snapshot(request, warnings=warnings)
