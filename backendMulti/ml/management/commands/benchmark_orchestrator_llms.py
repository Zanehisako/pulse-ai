from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from ml.orchestrator.benchmark import (
    DEFAULT_GGUF_DIR,
    DEFAULT_OUTPUT_ROOT,
    OrchestratorDynamicBenchmark,
)


class Command(BaseCommand):
    help = (
        "Benchmark all local orchestrator GGUF LLMs on dynamic natural-language "
        "tool-routing tasks and generate plots/report artifacts."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-dir",
            type=str,
            default=str(DEFAULT_OUTPUT_ROOT),
            help="Root directory where a timestamped benchmark artifact folder will be created.",
        )
        parser.add_argument(
            "--gguf-dir",
            type=str,
            default=str(DEFAULT_GGUF_DIR),
            help="Directory containing the local orchestrator GGUF models.",
        )
        parser.add_argument(
            "--models",
            type=str,
            default="",
            help="Optional comma-separated orchestrator model ids to benchmark.",
        )
        parser.add_argument(
            "--cases",
            type=str,
            default="",
            help="Optional comma-separated benchmark case ids to run.",
        )
        parser.add_argument(
            "--quick",
            action="store_true",
            help="Run a reduced smoke benchmark instead of the full matrix.",
        )
        parser.add_argument(
            "--cpu-threads",
            type=int,
            default=4,
            help="Number of CPU threads to give llama.cpp during benchmark runs.",
        )
        parser.add_argument(
            "--n-ctx",
            type=int,
            default=2048,
            help="Planner context length to use when loading each GGUF model.",
        )
        parser.add_argument(
            "--n-batch",
            type=int,
            default=128,
            help="Planner batch size to use when loading each GGUF model.",
        )
        parser.add_argument(
            "--n-gpu-layers",
            type=int,
            default=0,
            help="Number of llama.cpp GPU layers to use. Default keeps the benchmark CPU-safe.",
        )

    def handle(self, *args, **options):
        output_dir = Path(options["output_dir"]).expanduser().resolve()
        gguf_dir = Path(options["gguf_dir"]).expanduser().resolve()
        selected_models = [
            item.strip()
            for item in str(options.get("models") or "").split(",")
            if item.strip()
        ]
        selected_cases = [
            item.strip()
            for item in str(options.get("cases") or "").split(",")
            if item.strip()
        ]

        if not gguf_dir.exists():
            raise CommandError(f"GGUF directory does not exist: {gguf_dir}")

        benchmark = OrchestratorDynamicBenchmark(
            gguf_dir=gguf_dir,
            output_root=output_dir,
            cpu_threads=int(options["cpu_threads"]),
            n_ctx=int(options["n_ctx"]),
            n_batch=int(options["n_batch"]),
            n_gpu_layers=int(options["n_gpu_layers"]),
        )

        try:
            summary = benchmark.run(
                selected_model_ids=selected_models or None,
                selected_case_ids=selected_cases or None,
                quick=bool(options["quick"]),
            )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS(f"Benchmark artifacts written to: {summary.output_dir}"))
        self.stdout.write(self.style.SUCCESS(f"Report: {summary.report_path}"))
        self.stdout.write(self.style.SUCCESS(f"Summary JSON: {summary.summary_path}"))
        self.stdout.write(self.style.SUCCESS(f"Raw results JSON: {summary.raw_results_path}"))
        self.stdout.write("")
        self.stdout.write("Overall summary:")
        self.stdout.write(summary.overall_table.round(4).to_string(index=False))

