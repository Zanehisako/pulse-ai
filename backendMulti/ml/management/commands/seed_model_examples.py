"""
Auto-seed good/bad examples for all ML models based on their metadata.

Usage:
    python manage.py seed_model_examples
    python manage.py seed_model_examples --force     # Overwrite existing examples
    python manage.py seed_model_examples --dry-run   # Preview without saving
"""
from __future__ import annotations

import re
from typing import Any

from django.core.management.base import BaseCommand

from ml.models import MLModelConfig


# ── Task detection from model_id and features ────────────

def _infer_task(model: MLModelConfig) -> str:
    model_id = model.model_id.lower()
    features = model.features or []
    joined_features = " ".join(f.lower() for f in features)

    # Check model_id patterns first
    if "stockout" in model_id or "demand" in model_id:
        return "inventory_forecast"
    if "donation_30d" in model_id or "days_to_next" in model_id:
        return "donor_horizon"
    if "donor" in model_id or "agent" in model_id or "logistic" in model_id:
        return "donor_individual"

    # Check feature patterns
    if any(t in joined_features for t in ("stock", "hospital", "units_used", "units_collected")):
        return "inventory_forecast"
    if any(t in joined_features for t in ("recency", "donation_count", "blood_type", "eligible", "bmi")):
        return "donor_individual"

    return "general"


# ── Build features hint for examples ─────────────────────

def _feature_hint(model: MLModelConfig) -> str:
    """Build a human-readable feature string for good examples."""
    features = model.features or []
    hints = []
    for f in features[:5]:  # Max 5 features in example
        name = f.lower().replace("_", " ")
        if "age" in name:
            hints.append("age 35")
        elif "bmi" in name:
            hints.append("BMI 24")
        elif "sex" in name:
            hints.append("male")
        elif "recency" in name:
            hints.append("recency 30 days")
        elif "donation_count" in name:
            hints.append("3 donations in last 12 months")
        elif "blood_type" in name:
            hints.append("blood type A+")
        elif "hospital" in name:
            hints.append("hospital Central")
        elif "stock" in name:
            hints.append("current stock 50 units")
        elif "units_used" in name:
            hints.append("20 units used daily")
        elif "eligible" in name:
            hints.append("eligible to donate")
        elif "regular" in name:
            hints.append("regular donor")
        else:
            hints.append(f"{name} value")
    return ", ".join(hints) if hints else "provided feature values"


# ── Example generators per task ──────────────────────────

def _donor_individual_examples(model: MLModelConfig) -> list[dict[str, Any]]:
    hint = _feature_hint(model)
    model_name = model.model_id.replace("_", " ").title()
    return [
        {
            "kind": "good",
            "user_query": f"Is a donor with {hint} likely to donate?",
            "why": f"Single donor-level prediction using {model_name} features.",
        },
        {
            "kind": "good",
            "user_query": f"Check if a 42 year old female with BMI 26 is eligible to donate blood.",
            "why": "Individual donor eligibility check with demographic features.",
        },
        {
            "kind": "good",
            "user_query": f"Predict donation probability for donor with {hint}.",
            "why": "Direct prediction request with concrete donor features.",
        },
        {
            "kind": "bad",
            "user_query": "What is the U.S. blood type prevalence by ethnicity over time?",
            "why": "Population analytics query; requires database/search, not donor-level model.",
        },
        {
            "kind": "bad",
            "user_query": "How has the percentage of total blood donations by age changed over time?",
            "why": "Aggregate trend analytics, not individual donor prediction.",
        },
        {
            "kind": "bad",
            "user_query": "What's the blood supply of O- in hospitals?",
            "why": "Inventory/supply question; requires db_tool, not donor model.",
        },
    ]


def _donor_horizon_examples(model: MLModelConfig) -> list[dict[str, Any]]:
    hint = _feature_hint(model)
    return [
        {
            "kind": "good",
            "user_query": f"How many days until the next donation for a donor with {hint}?",
            "why": "Individual time-to-event estimation using donor features.",
        },
        {
            "kind": "good",
            "user_query": "Will this donor donate within 30 days given their recent activity?",
            "why": "Donor horizon prediction with temporal features.",
        },
        {
            "kind": "good",
            "user_query": f"Predict days to next donation for donor with {hint}.",
            "why": "Direct horizon prediction request.",
        },
        {
            "kind": "bad",
            "user_query": "Break down national donation percentages by age band over time.",
            "why": "Aggregate trend analytics outside donor horizon scope.",
        },
        {
            "kind": "bad",
            "user_query": "What is the current blood inventory for all hospitals?",
            "why": "Inventory question, not donor timeline prediction.",
        },
        {
            "kind": "bad",
            "user_query": "Explain the Rh factor in blood typing.",
            "why": "General knowledge question, not a prediction task.",
        },
    ]


def _inventory_forecast_examples(model: MLModelConfig) -> list[dict[str, Any]]:
    hint = _feature_hint(model)
    return [
        {
            "kind": "good",
            "user_query": f"Given {hint}, forecast days until stockout in the next month.",
            "why": "Operational inventory horizon forecast using hospital features.",
        },
        {
            "kind": "good",
            "user_query": "Predict blood demand for hospital Central in the next 30 days.",
            "why": "Hospital-level demand forecasting request.",
        },
        {
            "kind": "good",
            "user_query": f"How many days until stockout with {hint}?",
            "why": "Stockout prediction using operational features.",
        },
        {
            "kind": "bad",
            "user_query": "Is a 35 year old male with BMI 24 likely to donate?",
            "why": "Individual donor prediction, not inventory forecasting.",
        },
        {
            "kind": "bad",
            "user_query": "What percentage of U.S. presenting donors are deferred nationally?",
            "why": "National donor analytics, not hospital inventory forecasting.",
        },
        {
            "kind": "bad",
            "user_query": "What is the Rh factor in blood typing?",
            "why": "General knowledge question, not a prediction task.",
        },
    ]


def _general_examples(model: MLModelConfig) -> list[dict[str, Any]]:
    hint = _feature_hint(model)
    return [
        {
            "kind": "good",
            "user_query": f"Run prediction with {hint}.",
            "why": "Direct model execution request with features.",
        },
        {
            "kind": "good",
            "user_query": f"What does model {model.model_id} predict for these values?",
            "why": "Explicit model invocation.",
        },
        {
            "kind": "bad",
            "user_query": "Compute national prevalence trends from U.S. donor population tables.",
            "why": "Aggregate analytics should use database/search tools.",
        },
        {
            "kind": "bad",
            "user_query": "What's the blood supply of O- in all hospitals?",
            "why": "Supply/inventory query should use db_tool.",
        },
    ]


# ── Description generators per task ─────────────────────

TASK_DESCRIPTIONS = {
    "donor_individual": (
        "Individual donor propensity model. "
        "Use for single-donor probability/ranking based on age, sex, BMI, recency, donation history. "
        "NOT for population-level statistics, trends, or aggregate analytics."
    ),
    "donor_horizon": (
        "Donor time-horizon model. "
        "Predicts when an individual donor is likely to donate next, or if they will donate within a specific window. "
        "NOT for aggregate trends or inventory forecasting."
    ),
    "inventory_forecast": (
        "Hospital inventory and demand forecasting model. "
        "Predicts days until stockout, demand levels, and supply needs from operational features. "
        "NOT for individual donor predictions or general knowledge questions."
    ),
    "general": (
        "General prediction model. "
        "Run with provided feature values for model-specific predictions."
    ),
}

EXAMPLE_GENERATORS = {
    "donor_individual": _donor_individual_examples,
    "donor_horizon": _donor_horizon_examples,
    "inventory_forecast": _inventory_forecast_examples,
    "general": _general_examples,
}


class Command(BaseCommand):
    help = "Auto-generate good/bad examples and descriptions for all ML models based on their metadata."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite existing examples (default: skip models that already have examples).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview changes without saving to database.",
        )
        parser.add_argument(
            "--model-id",
            type=str,
            default=None,
            help="Only update a specific model by ID.",
        )

    def handle(self, *args, **options):
        force = options["force"]
        dry_run = options["dry_run"]
        target_model_id = options["model_id"]

        models = MLModelConfig.objects.filter(is_active=True)
        if target_model_id:
            models = models.filter(model_id=target_model_id)

        if not models.exists():
            self.stdout.write(self.style.WARNING("No active models found."))
            return

        updated = 0
        skipped = 0

        for model in models:
            # Skip if already has examples (unless --force)
            has_examples = isinstance(model.examples, list) and len(model.examples) > 0
            if has_examples and not force:
                self.stdout.write(f"  ⏭️  {model.model_id} — already has {len(model.examples)} examples (use --force to overwrite)")
                skipped += 1
                continue

            task = _infer_task(model)
            generator = EXAMPLE_GENERATORS.get(task, _general_examples)
            examples = generator(model)
            description = TASK_DESCRIPTIONS.get(task, TASK_DESCRIPTIONS["general"])

            if dry_run:
                self.stdout.write(f"\n  🔍 {model.model_id}")
                self.stdout.write(f"     Task: {task}")
                self.stdout.write(f"     Description: {description[:80]}...")
                self.stdout.write(f"     Examples ({len(examples)}):")
                for ex in examples:
                    icon = "✅" if ex["kind"] == "good" else "❌"
                    self.stdout.write(f"       {icon} [{ex['kind']}] {ex['user_query'][:70]}...")
                continue

            model.examples = examples
            model.description = description
            model.save(update_fields=["examples", "description"])
            updated += 1
            self.stdout.write(self.style.SUCCESS(
                f"  ✅ {model.model_id} — task={task}, {len(examples)} examples"
            ))

        self.stdout.write("")
        if dry_run:
            self.stdout.write(self.style.WARNING(f"🔍 Dry run complete. {models.count()} model(s) previewed. No changes saved."))
        else:
            self.stdout.write(self.style.SUCCESS(f"🏁 Done. {updated} updated, {skipped} skipped."))
