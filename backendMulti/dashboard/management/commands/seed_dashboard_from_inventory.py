"""
Dashboard sync command - fixed version.

Stock snapshot shape is now:
{
    "O+": 291,
    "A-": 39,
    ...
    "stock_status": {
        "O+": "medium",
        "A-": "low",
        ...
    },
    "stockout_days": {
        "O+": 12.4,   # min days until stockout across hospitals in this wilaya
        "A-": 3.1,    # from configured stockout-days PredictionResult rows
    }
}

If no PredictionResult rows exist yet, stockout_days is omitted
and the frontend falls back gracefully to hiding the stockout row.
"""
from collections import Counter
from datetime import date, timedelta

from django.core.management.base import BaseCommand
from django.db.models import Max, Sum
from django.core.cache import cache

from dashboard.models import ModelResponse
from inventory.models import BloodSupply, Donor, PredictionResult
from inventory.tasks import (
    configured_dashboard_forecast_cap_policy,
    configured_dashboard_forecast_jobs,
    configured_stockout_days_result_models,
)

CACHE_KEY = "dashboard_seeded"

BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
DEFAULT_WILAYAS = ["Québec", "Montréal"]

# Maps French wilaya names to valid ModelResponse.Region enum values.
# Add entries here when new wilayas are added — no code changes required.
WILAYA_TO_REGION: dict[str, str] = {
    "Montréal": ModelResponse.Region.MONTREAL,
}

# The set of region values that are valid in this codebase.
# Any ModelResponse row whose region is NOT in this set is an orphaned legacy row.
CANONICAL_REGIONS: frozenset[str] = frozenset(
    v for v in (ModelResponse.Region.QUEBEC, ModelResponse.Region.MONTREAL)
)


def _normalize_wilaya_to_region(wilaya: str) -> str:
    """Return the ModelResponse.Region value for a given wilaya name."""
    return WILAYA_TO_REGION.get(wilaya, ModelResponse.Region.QUEBEC)

CRITICAL_THRESHOLD = 80
LOW_THRESHOLD      = 200
DONATION_HISTORY_DAYS = 90
DONATION_HISTORY_WINDOW_DAYS = 365
STOCK_CAPACITY_UNITS = 1000
STOCK_STATUS_BANDS = (
    ("low", 0, 30),
    ("medium", 30, 70),
    ("high", 70, 101),
)


def derive_alert_level(stock_map: dict) -> str:
    for bt in ["O-", "AB-", "O+", "B-"]:
        units = stock_map.get(bt, 0)
        if units < CRITICAL_THRESHOLD:
            return ModelResponse.AlertLevel.HIGH
        if units < LOW_THRESHOLD:
            return ModelResponse.AlertLevel.MEDIUM
    all_units = list(stock_map.values())
    if any(u < CRITICAL_THRESHOLD for u in all_units):
        return ModelResponse.AlertLevel.HIGH
    if any(u < LOW_THRESHOLD for u in all_units):
        return ModelResponse.AlertLevel.MEDIUM
    return ModelResponse.AlertLevel.LOW


def get_stock_percentage(units: int | float) -> int:
    percentage = int((float(units) / STOCK_CAPACITY_UNITS) * 100)
    return max(0, min(100, percentage))


def derive_stock_status(units: int | float) -> str:
    percentage = get_stock_percentage(units)
    for status, min_percentage, max_percentage in STOCK_STATUS_BANDS:
        if min_percentage <= percentage < max_percentage:
            return status
    return STOCK_STATUS_BANDS[0][0]


def build_stock_status(stock_data: dict) -> dict:
    return {
        blood_type: derive_stock_status(units)
        for blood_type, units in stock_data.items()
        if blood_type in BLOOD_TYPES
    }


def build_stockout_days(wilaya: str) -> dict | None:
    """
    For each blood type in this wilaya, find the minimum predicted
    days until stockout across all hospitals (worst-case hospital wins).
    Uses each supply's latest available prediction, so the dashboard
    can still render stockout cards even when today's batch has not run yet.
    Returns None if no PredictionResult rows exist yet.
    """
    result_model_names = configured_stockout_days_result_models()
    if not result_model_names:
        return None

    supply_ids = list(
        BloodSupply.objects
        .filter(hospital__wilaya=wilaya)
        .values_list("supply_id", flat=True)
    )
    if not supply_ids:
        return None

    rows = (
        PredictionResult.objects
        .filter(
            model_name__in=result_model_names,
            entity_id__in=supply_ids,
        )
        .order_by("entity_id", "-predicted_for_date", "-created_at")
        .values("entity_id", "predicted_value")
    )

    if not rows.exists():
        return None

    supply_to_type = dict(
        BloodSupply.objects
        .filter(supply_id__in=supply_ids)
        .values_list("supply_id", "blood_product_type")
    )

    days_per_type: dict[str, float] = {}
    latest_supply_ids: set[str] = set()
    for row in rows:
        supply_id = row["entity_id"]
        if supply_id in latest_supply_ids:
            continue
        latest_supply_ids.add(supply_id)

        bt  = supply_to_type.get(supply_id)
        val = float(row["predicted_value"])
        if bt and (bt not in days_per_type or val < days_per_type[bt]):
            days_per_type[bt] = round(val, 1)

    return days_per_type or None


def build_donation_history(
    donor_rows,
    *,
    reference_date: date | None = None,
    days: int = DONATION_HISTORY_DAYS,
) -> dict:
    """
    Build the dashboard donation time series from donor recency/frequency data.

    Seeded donors do not store every historical donation event, so this expands
    frequency_365 into an approximate per-day history anchored on recency_days.
    """
    if days <= 0:
        return {"data": []}

    end_date = reference_date or date.today()
    start_date = end_date - timedelta(days=days - 1)
    counts: Counter[date] = Counter()

    for row in donor_rows:
        recency_days = row.get("recency_days")
        frequency_365 = row.get("frequency_365")
        try:
            recency = max(0, int(recency_days))
        except (TypeError, ValueError):
            continue

        try:
            frequency = max(0, int(frequency_365))
        except (TypeError, ValueError):
            frequency = 0

        if recency >= DONATION_HISTORY_WINDOW_DAYS and frequency == 0:
            continue

        donation_count = max(
            frequency,
            1 if recency < DONATION_HISTORY_WINDOW_DAYS else 0,
        )
        if donation_count <= 0:
            continue

        interval_days = max(1, DONATION_HISTORY_WINDOW_DAYS // donation_count)
        for donation_index in range(donation_count):
            donation_recency = recency + donation_index * interval_days
            if donation_recency >= DONATION_HISTORY_WINDOW_DAYS:
                break

            donation_date = end_date - timedelta(days=donation_recency)
            if start_date <= donation_date <= end_date:
                counts[donation_date] += 1

    return {
        "data": [
            {
                "date": (start_date + timedelta(days=offset)).isoformat(),
                "value": counts[start_date + timedelta(days=offset)],
            }
            for offset in range(days)
        ]
    }


def build_wilaya_donation_history(wilaya: str) -> dict:
    donor_rows = Donor.objects.filter(wilaya=wilaya).values(
        "recency_days",
        "frequency_365",
    )
    return build_donation_history(donor_rows)


def build_forecast(wilaya: str) -> dict | None:
    """
    For each blood type in this wilaya, aggregate the latest available
    configured forecast PredictionResult rows across hospitals for each
    configured horizon.
    Returns None if no configured forecast results exist yet.
    """
    forecast_jobs = configured_dashboard_forecast_jobs()
    if not forecast_jobs:
        return None

    supply_rows = list(
        BloodSupply.objects
        .filter(hospital__wilaya=wilaya)
        .values_list("supply_id", "blood_product_type")
    )
    if not supply_rows:
        return None

    supply_ids = [supply_id for supply_id, _blood_type in supply_rows]
    supply_to_type = dict(supply_rows)
    model_to_horizon = {
        row["result_model_name"]: horizon
        for horizon, row in forecast_jobs.items()
    }
    horizon_types = {
        horizon: str(row.get("horizon_type") or "absolute").strip().lower()
        for horizon, row in forecast_jobs.items()
    }
    cap_policy = configured_dashboard_forecast_cap_policy()
    cap_enabled = bool(cap_policy.get("enabled"))
    horizon_max_factors = (
        dict(cap_policy.get("horizon_max_factors") or {}) if cap_enabled else {}
    )
    default_max_factor = (
        float(cap_policy.get("default_max_factor")) if cap_enabled else None
    )
    min_floor_units = (
        float(cap_policy.get("min_floor_units")) if cap_enabled else None
    )
    current_stock_totals: dict[str, float] = {}
    for _supply_id, blood_type in supply_rows:
        current_stock_totals.setdefault(blood_type, 0.0)
    for row in (
        BloodSupply.objects
        .filter(hospital__wilaya=wilaya)
        .values("blood_product_type")
        .annotate(total_units=Sum("current_stock_units"))
    ):
        current_stock_totals[str(row["blood_product_type"])] = float(row["total_units"] or 0.0)

    rows = (
        PredictionResult.objects
        .filter(
            entity_id__in=supply_ids,
            model_name__in=list(model_to_horizon.keys()),
        )
        .order_by("entity_id", "model_name", "-predicted_for_date", "-created_at")
        .values("entity_id", "model_name", "predicted_value")
    )
    if not rows.exists():
        return None

    latest_per_supply_model: dict[tuple[str, str], float] = {}
    for row in rows:
        cache_key = (str(row["entity_id"]), str(row["model_name"]))
        if cache_key in latest_per_supply_model:
            continue
        latest_per_supply_model[cache_key] = float(row["predicted_value"])

    if not latest_per_supply_model:
        return None

    sums: dict[tuple[str, str], float] = {}
    for (supply_id, model_name), predicted_value in latest_per_supply_model.items():
        blood_type = supply_to_type.get(supply_id)
        horizon = model_to_horizon.get(model_name)
        if not blood_type or not horizon:
            continue
        sum_key = (blood_type, horizon)
        sums[sum_key] = sums.get(sum_key, 0.0) + predicted_value

    result: dict[str, dict] = {}
    for blood_type in BLOOD_TYPES:
        horizons: dict[str, dict | None] = {}
        current_total = current_stock_totals.get(blood_type, 0.0)
        for horizon in forecast_jobs:
            predicted = sums.get((blood_type, horizon))
            if predicted is not None and horizon_types.get(horizon) == "delta":
                predicted = max(0.0, current_total + predicted)
            if (
                predicted is not None
                and current_total > 0
                and cap_enabled
                and default_max_factor is not None
                and min_floor_units is not None
            ):
                factor = float(horizon_max_factors.get(horizon, default_max_factor))
                cap = max(current_total * factor, current_total + min_floor_units)
                predicted = min(predicted, cap)
            horizons[horizon] = (
                {"predicted_units": round(predicted, 2)}
                if predicted is not None
                else None
            )
        result[blood_type] = horizons

    return result


def get_available_wilayas() -> list[str]:
    seeded_wilayas = list(
        BloodSupply.objects.order_by("hospital__wilaya").values_list(
            "hospital__wilaya",
            flat=True,
        ).distinct()
    )
    return seeded_wilayas or DEFAULT_WILAYAS


def write_snapshot(typeModel, jsonResponse, wilaya, alert_level):
    """Replace the previous snapshot for this wilaya + typeModel.

    Also removes any rows whose ``region`` is not in CANONICAL_REGIONS so that
    legacy orphaned rows (e.g. ``gatineau``, ``saguenay``) created before the
    WILAYA_TO_REGION normalisation was introduced can never re-accumulate.
    """
    region = _normalize_wilaya_to_region(wilaya)
    # Delete the current region's snapshot.
    ModelResponse.objects.filter(
        typeModel=typeModel,
        region=region,
        product=ModelResponse.Product.BLOOD,
        period=ModelResponse.Period.H24,
    ).delete()

    # Purge any orphaned rows for non-canonical regions.
    ModelResponse.objects.filter(
        typeModel=typeModel,
        product=ModelResponse.Product.BLOOD,
        period=ModelResponse.Period.H24,
    ).exclude(region__in=CANONICAL_REGIONS).delete()

    ModelResponse.objects.create(
        typeModel=typeModel,
        jsonResponse=jsonResponse,
        region=region,
        product=ModelResponse.Product.BLOOD,
        period=ModelResponse.Period.H24,
        alert_level=alert_level,
    )


class Command(BaseCommand):
    help = "Sync dashboard snapshots from inventory + predictions (one row per wilaya)"

    def add_arguments(self, parser):
        parser.add_argument("--wilaya", type=str, default=None)
        parser.add_argument("--force",  action="store_true")

    def handle(self, *args, **options):
        if not options["force"] and cache.get(CACHE_KEY):
            self.stdout.write(
                self.style.WARNING("Already synced. Use --force to refresh.")
            )
            return

        wilayas = [options["wilaya"]] if options["wilaya"] else get_available_wilayas()

        for wilaya in wilayas:
            self.stdout.write(f"\n-- {wilaya.upper()} --")

            # STOCK
            stocks = (
                BloodSupply.objects
                .filter(hospital__wilaya=wilaya)
                .values("blood_product_type")
                .annotate(total=Sum("current_stock_units"))
            )
            stock_map  = {r["blood_product_type"]: round(r["total"]) for r in stocks}
            stock_data = {bt: stock_map.get(bt, 0) for bt in BLOOD_TYPES}
            stock_data["stock_status"] = build_stock_status(stock_data)

            # Load previous snapshot for carry-forward when ML predictions are
            # partial or missing (e.g. right after a restart or DB reset).
            _region = _normalize_wilaya_to_region(wilaya)
            _prev_row = ModelResponse.objects.filter(
                typeModel="stock",
                region=_region,
                product=ModelResponse.Product.BLOOD,
                period=ModelResponse.Period.H24,
            ).values("jsonResponse").first()
            _prev_json: dict = {}
            if _prev_row:
                try:
                    _j = _prev_row["jsonResponse"]
                    _prev_json = _j if isinstance(_j, dict) else __import__("json").loads(_j)
                except Exception:
                    pass
            _prev_stockout: dict = _prev_json.get("stockout_days", {})
            _prev_forecast: dict = _prev_json.get("forecast", {})

            # Attach stockout_days — carry forward previous values for blood
            # types whose PredictionResult rows are temporarily unavailable so
            # every card always renders its stockout section.
            stockout_days = build_stockout_days(wilaya)
            if stockout_days is None:
                if _prev_stockout:
                    stockout_days = dict(_prev_stockout)
                    self.stdout.write(
                        "  info no PredictionResult rows - carried forward previous stockout_days"
                    )
                else:
                    self.stdout.write(
                        "  info no PredictionResult rows today - stockout_days omitted"
                    )
            else:
                # Fill in blood types that are missing from today's predictions.
                for bt in BLOOD_TYPES:
                    if bt not in stockout_days and bt in _prev_stockout:
                        stockout_days[bt] = _prev_stockout[bt]
                self.stdout.write(f"  ok stockout_days attached: {stockout_days}")
            if stockout_days:
                stock_data["stockout_days"] = stockout_days

            # Attach forecast — carry forward previous values for blood types
            # with temporarily missing predictions.
            forecast = build_forecast(wilaya)
            if forecast is None:
                if _prev_forecast:
                    forecast = dict(_prev_forecast)
                    self.stdout.write(
                        "  info no forecast rows - carried forward previous forecast"
                    )
                else:
                    self.stdout.write("  info no StockForecast rows today - forecast omitted")
            else:
                # Fill in blood types missing from today's forecast.
                for bt in BLOOD_TYPES:
                    if bt not in forecast and bt in _prev_forecast:
                        forecast[bt] = _prev_forecast[bt]
                self.stdout.write("  ok forecast attached")
            if forecast:
                stock_data["forecast"] = forecast

            alert_level = derive_alert_level(stock_map)
            write_snapshot("stock", stock_data, wilaya, alert_level)
            self.stdout.write(f"  ok stock -> alert={alert_level}")

            # METRICS
            total_donors = Donor.objects.filter(wilaya=wilaya).count()
            active_donors = Donor.objects.filter(wilaya=wilaya, availability=1).count()
            # Use the most recent predicted_for_date available, not today,
            # so the count stays correct after restarts on different days.
            _supply_ids_for_wilaya = BloodSupply.objects.filter(
                hospital__wilaya=wilaya
            ).values_list("supply_id", flat=True)
            _latest = PredictionResult.objects.filter(
                model_name__in=configured_stockout_days_result_models(),
                entity_id__in=_supply_ids_for_wilaya,
            ).aggregate(d=Max("predicted_for_date"))["d"]
            _critical_date = _latest or date.today()
            critical_count = PredictionResult.objects.filter(
                model_name__in=configured_stockout_days_result_models(),
                alert_triggered=True,
                predicted_for_date=_critical_date,
                entity_id__in=_supply_ids_for_wilaya,
            ).count()

            metrics = [
                {"label": "Total Donors",        "value": total_donors,   "change": 0, "icon": "users",         "color": "primary"},
                {"label": "Active Donors",        "value": active_donors,  "change": 0, "icon": "droplets",      "color": "warning"},
                {"label": "Critical Stock Types", "value": critical_count, "change": 0, "icon": "alertTriangle", "color": "danger"},
            ]
            write_snapshot("metrics", metrics, wilaya, alert_level)
            self.stdout.write(
                f"  ok metrics -> donors={total_donors} active={active_donors} critical={critical_count}"
            )

            # DONATIONS
            donations = build_wilaya_donation_history(wilaya)
            write_snapshot("donations", donations, wilaya, alert_level)
            total_recent_donations = sum(point["value"] for point in donations["data"])
            self.stdout.write(
                f"  ok donations -> {total_recent_donations} in last {DONATION_HISTORY_DAYS} days"
            )

        cache.set(CACHE_KEY, True, timeout=None)
        self.stdout.write(self.style.SUCCESS("\nDashboard synced"))
