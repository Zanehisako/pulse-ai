import argparse
import json
import logging
import signal
import time
from pathlib import Path

from pios_ml_backend.config_db import init_db
from pios_ml_backend.mlops import (
    DependencyError,
    FeastMLflowPipeline,
    NotebookModelsFeastMLflowPipeline,
    PipelineError,
)
from pios_ml_backend.training_scheduler import start_scheduler, stop_scheduler

logger = logging.getLogger("pios.main")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PIOS ML backend commands (DB bootstrap + Feast/MLflow workflows)."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=[
            "run",
            "init-db",
            "mlops-prepare",
            "mlops-apply",
            "mlops-materialize",
            "mlops-train",
            "mlops-train-notebooks",
            "mlops-score",
            "mlops-entities",
        ],
    )
    parser.add_argument(
        "--skip-materialize",
        action="store_true",
        help="Skip Feast online materialization during mlops-train.",
    )
    parser.add_argument(
        "--inventory-id",
        action="append",
        help="Inventory id in the form <hospital>__<blood_type>. Repeat for multiple entities.",
    )
    parser.add_argument(
        "--run-id",
        help="Specific MLflow run id for mlops-score. Defaults to the latest run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max number of entity ids returned by mlops-entities.",
    )
    parser.add_argument(
        "--model-config",
        help=(
            "Notebook model config JSON path "
            "(default: ml-backend/notebooks/ml_models/config.json)."
        ),
    )
    parser.add_argument(
        "--training-spec",
        help=(
            "Training spec JSON path "
            "(default: ml-backend/notebooks/ml_models/training_spec.json)."
        ),
    )
    parser.add_argument(
        "--model-id",
        action="append",
        help="Train only these model ids from notebook config (repeatable).",
    )
    parser.add_argument(
        "--strict-features",
        action="store_true",
        help="Fail when a requested feature is missing instead of auto-filling with 0.",
    )
    parser.add_argument(
        "--register-models",
        action="store_true",
        help="Register trained models in MLflow registry.",
    )
    return parser


def _run_service_loop() -> None:
    init_db()

    # Start the background training scheduler
    _scheduler = start_scheduler()  # noqa: F841
    logger.info("Training scheduler started alongside service loop.")

    # Ensure clean shutdown on SIGINT / SIGTERM
    def _handle_signal(signum: int, frame: object) -> None:
        print(f"\nReceived signal {signum} — shutting down scheduler …")
        stop_scheduler()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("PIOS ML backend started with Postgres config DB.")
    print("Service loop running (training scheduler active). Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    finally:
        stop_scheduler()


def _run_mlops(args: argparse.Namespace) -> None:
    pipeline = FeastMLflowPipeline.from_env()

    if args.command == "mlops-prepare":
        frame = pipeline.prepare_feature_data()
        print(
            f"Prepared {frame.count()} rows for Feast source at "
            f"{pipeline.settings.feature_parquet}"
        )
        return

    if args.command == "mlops-apply":
        pipeline.apply_feast_repo()
        print("Feast definitions applied successfully.")
        return

    if args.command == "mlops-materialize":
        pipeline.materialize_incremental()
        print("Feast online store materialized incrementally.")
        return

    if args.command == "mlops-train":
        result = pipeline.sync_and_train(materialize_online=not args.skip_materialize)
        print(json.dumps(result, indent=2))
        return

    if args.command == "mlops-train-notebooks":
        notebook_pipeline = NotebookModelsFeastMLflowPipeline.from_env()
        result = notebook_pipeline.run(
            model_config_path=Path(args.model_config).resolve()
            if args.model_config
            else None,
            training_spec_path=Path(args.training_spec).resolve()
            if args.training_spec
            else None,
            selected_model_ids=args.model_id,
            materialize_online=not args.skip_materialize,
            allow_missing_features=not args.strict_features,
            register_models=args.register_models,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "mlops-score":
        if not args.inventory_id:
            raise PipelineError("Provide at least one --inventory-id for mlops-score.")
        scores = pipeline.score_online(args.inventory_id, run_id=args.run_id)
        print(json.dumps(scores, indent=2))
        return

    if args.command == "mlops-entities":
        ids = pipeline.list_inventory_ids(limit=max(1, args.limit))
        print(json.dumps(ids, indent=2))
        return

    raise PipelineError(f"Unsupported mlops command: {args.command}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "run":
            _run_service_loop()
        elif args.command == "init-db":
            init_db()
            print("Database initialized.")
        else:
            _run_mlops(args)
    except (DependencyError, PipelineError) as exc:
        parser.exit(status=1, message=f"{exc}\n")


if __name__ == "__main__":
    main()
