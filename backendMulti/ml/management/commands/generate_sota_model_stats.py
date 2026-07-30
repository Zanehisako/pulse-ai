from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from ml.core.model_loading import ModelSourceResolutionError
from ml.core.sota_model_stats import (
    SotaModelStatsConfigError,
    generate_sota_model_stats,
)


class Command(BaseCommand):
    help = "Generate and persist ModelStats rows for configured SOTA joblib models."

    def add_arguments(self, parser):
        parser.add_argument("--runtime-config", help="Override prediction_runtime.json path.")
        parser.add_argument("--catalog-path", help="Override SOTA catalog path.")
        parser.add_argument("--datasets-dir", help="Override SOTA feature datasets directory.")
        parser.add_argument("--models-dir", help="Override SOTA model artifact directory.")

    def handle(self, *args, **options):
        try:
            summary = generate_sota_model_stats(
                runtime_config_path=options.get("runtime_config"),
                catalog_path=options.get("catalog_path"),
                datasets_dir=options.get("datasets_dir"),
                models_dir=options.get("models_dir"),
            )
        except (ModelSourceResolutionError, SotaModelStatsConfigError) as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                f"Persisted SOTA ModelStats for {summary['persisted']} model(s)."
            )
        )
        for row in summary["processed"]:
            self.stdout.write(
                "  OK {model_id}: {available_features}/{configured_features} features, task={task_type}".format(
                    **row
                )
            )
        for row in summary["skipped"]:
            self.stdout.write(
                self.style.WARNING("  SKIP {model_id}: {reason}".format(**row))
            )
