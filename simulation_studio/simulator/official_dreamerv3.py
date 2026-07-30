from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from functools import partial as bind
from pathlib import Path

import numpy as np
from dreamerv3_runtime_paths import (
    discover_official_runtime_paths,
    format_official_runtime_error,
    inject_official_runtime_paths,
    resolve_official_source_root,
)

inject_official_runtime_paths()

CACHE_ROOT = Path(tempfile.gettempdir()) / "pios_dreamer_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "jax_compilation_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", str(CACHE_ROOT / "jax_compilation_cache")
)

SIMULATOR_ROOT = Path(__file__).resolve().parent
DEFAULT_LOGDIR = SIMULATOR_ROOT / "dreamerv3_runs" / "default"


@dataclass(frozen=True)
class JaxRuntimeConfig:
    platform: str
    compute_dtype: str
    prealloc: bool
    reason: str


class BufferedJSONLOutput:
    def __init__(self, logdir, filename="metrics.jsonl", pattern=r".*", strings=False):
        self._pattern = re.compile(pattern)
        self._strings = strings
        self._logdir = Path(logdir)
        self._logdir.mkdir(parents=True, exist_ok=True)
        self._filename = self._logdir / filename
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._flushed = False

    def __call__(self, summaries):
        import numpy as np

        bystep = collections.defaultdict(dict)
        for step, name, value in summaries:
            if not self._pattern.search(name):
                continue
            if isinstance(value, str) and self._strings:
                bystep[step][name] = value
            if isinstance(value, np.ndarray) and len(value.shape) == 0:
                bystep[step][name] = float(value)
        lines = [
            json.dumps({"step": step, **scalars}) + "\n"
            for step, scalars in bystep.items()
        ]
        if not lines:
            return
        with self._lock:
            self._lines.extend(lines)
            self._flushed = False

    def wait(self):
        with self._lock:
            if self._flushed or not self._lines:
                return
            lines = "".join(self._lines)
            self._lines.clear()
            self._flushed = True
        print(f"Writing buffered metrics: {self._filename}")
        with self._filename.open("a") as f:
            f.write(lines)


def _probe_optional_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


@lru_cache(maxsize=1)
def probe_metal_runtime_available() -> bool:
    runtime_paths = discover_official_runtime_paths()
    code = f"""
import os
import sys
import tempfile
from pathlib import Path

for path in reversed({runtime_paths!r}):
    if path and path not in sys.path:
        sys.path.insert(0, path)

cache_root = Path(tempfile.gettempdir()) / "pios_dreamer_cache_probe"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_platforms", "METAL")
jax.devices()
jax.device_get(jnp.arange(4, dtype=jnp.float32))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def is_apple_silicon_mac() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def canonical_platform_name(platform_name: str) -> str:
    value = (platform_name or "").strip()
    if value.lower() == "metal":
        return "METAL"
    return value.lower() if value.lower() in {"cpu", "cuda", "gpu", "tpu"} else value


def choose_jax_runtime(
    requested_platform: str,
    requested_dtype: str,
    *,
    is_apple_silicon: bool | None = None,
    has_jax: bool | None = None,
    has_metal_plugin: bool | None = None,
    metal_runtime_available: bool | None = None,
) -> JaxRuntimeConfig:
    platform_request = (requested_platform or "auto").strip().lower()
    dtype_request = (requested_dtype or "auto").strip().lower()

    if is_apple_silicon is None:
        is_apple_silicon = is_apple_silicon_mac()
    if has_jax is None:
        has_jax = _probe_optional_module("jax") and _probe_optional_module("jaxlib")
    if has_metal_plugin is None:
        has_metal_plugin = _probe_optional_module(
            "jax_plugins.metal_plugin"
        ) or _probe_optional_module("jax_metal")
    if (
        metal_runtime_available is None
        and is_apple_silicon
        and has_jax
        and has_metal_plugin
    ):
        metal_runtime_available = probe_metal_runtime_available()

    if platform_request == "auto":
        if (
            is_apple_silicon
            and has_jax
            and has_metal_plugin
            and metal_runtime_available
        ):
            platform_name = "METAL"
            reason = "Auto-selected Apple Metal because JAX and a Metal plugin were detected."
        elif (
            is_apple_silicon
            and has_jax
            and has_metal_plugin
            and metal_runtime_available is False
        ):
            platform_name = "cpu"
            reason = "Fell back to CPU because the Metal backend probe failed on this machine."
        elif is_apple_silicon and has_jax and not has_metal_plugin:
            platform_name = "cpu"
            reason = "Fell back to CPU because jax-metal was not detected in the Dreamer runtime."
        elif not has_jax:
            platform_name = "cpu"
            reason = "Fell back to CPU because JAX is not importable from the Dreamer runtime."
        else:
            platform_name = "cpu"
            reason = "Fell back to CPU because no accelerator backend was requested explicitly."
    elif platform_request == "gpu":
        if is_apple_silicon:
            if has_jax and has_metal_plugin and metal_runtime_available:
                platform_name = "METAL"
                reason = "Mapped generic GPU request to Apple Metal on Apple Silicon."
            elif has_jax and has_metal_plugin and metal_runtime_available is False:
                platform_name = "cpu"
                reason = "Mapped generic GPU request to CPU because the Metal backend probe failed."
            else:
                platform_name = "cpu"
                reason = "Mapped generic GPU request to CPU because jax-metal is not available."
        else:
            platform_name = "cuda"
            reason = "Mapped generic GPU request to CUDA."
    elif platform_request == "metal":
        if (
            is_apple_silicon
            and has_jax
            and has_metal_plugin
            and metal_runtime_available
        ):
            platform_name = "METAL"
            reason = "Using the requested Apple Metal backend."
        elif not is_apple_silicon:
            platform_name = "cpu"
            reason = (
                "Requested Metal on a non-Apple-Silicon host, so the run is using CPU."
            )
        elif not has_jax:
            platform_name = "cpu"
            reason = (
                "Requested Metal, but JAX is not importable from the Dreamer runtime."
            )
        elif metal_runtime_available is False:
            platform_name = "cpu"
            reason = (
                "Requested Metal, but the Metal backend probe failed on this machine."
            )
        else:
            platform_name = "cpu"
            reason = "Requested Metal, but jax-metal was not detected in the Dreamer runtime."
    else:
        platform_name = canonical_platform_name(platform_request)
        reason = f"Using the requested {platform_name} backend."

    if dtype_request == "auto":
        compute_dtype = (
            "float32"
            if canonical_platform_name(platform_name) in {"cpu", "METAL"}
            else "bfloat16"
        )
    else:
        compute_dtype = dtype_request

    prealloc = canonical_platform_name(platform_name) not in {"cpu", "METAL"}
    return JaxRuntimeConfig(platform_name, compute_dtype, prealloc, reason)


def resolve_env_count(
    requested_envs: int,
    platform_name: str,
    *,
    eval_mode: bool = False,
    cpu_count: int | None = None,
) -> int:
    if requested_envs > 0:
        return requested_envs
    if eval_mode:
        return 1

    workers = max(1, int(cpu_count or os.cpu_count() or 1))
    platform_name = canonical_platform_name(platform_name)
    if platform_name == "METAL":
        return min(4, max(1, workers // 2))
    if platform_name == "cuda":
        return min(8, max(1, workers // 2))
    return 1


def resolve_jax_profiler_mode(requested_mode: str, platform_name: str) -> bool:
    platform_name = canonical_platform_name(platform_name)
    mode = (requested_mode or "auto").strip().lower()
    if mode == "on":
        return True
    if mode == "off":
        return False
    return platform_name != "METAL"


def configure_process_env(
    runtime: JaxRuntimeConfig, *, allow_online_city_graph: bool = False
) -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(bool(runtime.prealloc)).lower()
    if canonical_platform_name(runtime.platform) == "METAL":
        os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")
    if not allow_online_city_graph:
        os.environ.setdefault("PIOS_SIM_OFFLINE", "1")


def import_official_runtime():
    try:
        import dreamerv3.main as official_main
        import elements
        import embodied
        import portal
        import ruamel.yaml as yaml
        from dreamerv3.agent import Agent
        from official_dreamerv3_env import OfficialDreamerBloodEnv
        from official_dreamerv3_env_budget import OfficialDreamerBloodBudgetEnv
        from plot_dreamerv3_metrics import plot_metrics_for_logdir
        from scenarios import HOLDOUT_SCENARIO_KEYS, SCENARIOS, TRAINING_SCENARIO_KEYS
        from train_mappo import load_training_context
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised only in missing-deps runtime.
        raise RuntimeError(format_official_runtime_error(exc)) from exc

    return {
        "elements": elements,
        "embodied": embodied,
        "portal": portal,
        "yaml": yaml,
        "OfficialDreamerBloodEnv": OfficialDreamerBloodEnv,
        "OfficialDreamerBloodBudgetEnv": OfficialDreamerBloodBudgetEnv,
        "plot_metrics_for_logdir": plot_metrics_for_logdir,
        "SCENARIOS": SCENARIOS,
        "TRAINING_SCENARIO_KEYS": TRAINING_SCENARIO_KEYS,
        "HOLDOUT_SCENARIO_KEYS": HOLDOUT_SCENARIO_KEYS,
        "load_training_context": load_training_context,
        "official_main": official_main,
        "Agent": Agent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official DreamerV3 implementation on the blood supply simulator."
    )
    parser.add_argument(
        "--script",
        choices=["train", "train_eval", "eval_only"],
        default="train",
        help="Official DreamerV3 run mode.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=str(DEFAULT_LOGDIR),
        help="Run directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default="",
        help="Checkpoint path for resume or eval_only. Accepts either a completed save folder or <logdir>/ckpt.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50_000,
        help="Environment steps to run.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="train",
        help="Scenario key, comma-separated list, or one of: train, holdout, all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed.",
    )
    parser.add_argument(
        "--step-hours",
        type=float,
        default=6.0,
        help="Simulator hours advanced per action.",
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=0,
        help="Number of training environments. Use 0 to auto-tune for the selected backend.",
    )
    parser.add_argument(
        "--eval-envs",
        type=int,
        default=0,
        help="Number of evaluation environments for train_eval. Use 0 for the backend-aware default.",
    )
    parser.add_argument(
        "--eval-eps",
        type=int,
        default=3,
        help="Episodes for eval in train_eval mode.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=32.0,
        help="Official DreamerV3 train ratio.",
    )
    parser.add_argument(
        "--imag-length",
        type=int,
        default=None,
        help="Override the imagination horizon (imag_length) used for actor/critic "
        "training. Default None keeps the config value (15). Training-only — does "
        "not affect inference. Used by the imagination-horizon ablation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--batch-length",
        type=int,
        default=64,
        help="Sequence length.",
    )
    parser.add_argument(
        "--size",
        choices=["size1m", "size12m", "size25m", "size50m", "debug"],
        default="size1m",
        help="Official model size preset.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=30,
        help="Seconds between logger flushes.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=120,
        help="Seconds between report passes.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=300,
        help="Seconds between checkpoint saves.",
    )
    parser.add_argument(
        "--jax-platform",
        choices=["auto", "cpu", "gpu", "cuda", "metal"],
        default=os.environ.get("PIOS_DREAMERV3_JAX_PLATFORM", "auto"),
        help="JAX backend selection. 'auto' prefers Metal on Apple Silicon when jax-metal is installed.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["auto", "float32", "bfloat16"],
        default=os.environ.get("PIOS_DREAMERV3_COMPUTE_DTYPE", "auto"),
        help="JAX compute dtype. 'auto' uses float32 on CPU/Metal and bfloat16 on CUDA.",
    )
    parser.add_argument(
        "--online-city-graph",
        action="store_true",
        help="Allow live OpenStreetMap city-graph downloads instead of the fast offline fallback.",
    )
    parser.add_argument(
        "--jax-profiler",
        choices=["auto", "on", "off"],
        default="auto",
        help="JAX profiler mode. 'auto' disables the profiler on Metal to avoid known crashes.",
    )
    parser.add_argument(
        "--metrics-write-mode",
        choices=["buffered", "live"],
        default="buffered",
        help="Write JSONL metrics incrementally or buffer them until shutdown/interruption.",
    )
    parser.add_argument(
        "--budget-variant",
        action="store_true",
        help=(
            "Use the budget-aware environment variant.  The reward function "
            "heavily penalises budget overspend and action cost, and the "
            "episode terminates early if the agent exhausts its budget."
        ),
    )
    parser.add_argument(
        "--fixed-budget",
        type=float,
        default=None,
        help=(
            "Override per-scenario budgets with a single fixed budget.  "
            "Only used when --budget-variant is enabled.  A lower value "
            "(e.g. 4000-5000) forces the agent to be more frugal."
        ),
    )
    return parser.parse_args()


def parse_scenarios(
    raw: str,
    available_scenarios: dict[str, object],
    *,
    training_scenario_keys: tuple[str, ...] = (),
    holdout_scenario_keys: tuple[str, ...] = (),
) -> list[str]:
    value = raw.strip().lower()
    if value == "all":
        return list(available_scenarios.keys())
    if value == "train":
        return list(training_scenario_keys or available_scenarios.keys())
    if value in {"holdout", "benchmark", "generalization"}:
        if not holdout_scenario_keys:
            raise ValueError("No holdout scenarios are registered.")
        return list(holdout_scenario_keys)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in values if item not in available_scenarios]
    if invalid:
        raise ValueError(
            f"Unknown scenarios: {invalid}. Available: {sorted(available_scenarios.keys())}"
        )
    if not values:
        raise ValueError("At least one scenario is required.")
    return values


def resolve_official_checkpoint_path(raw_path: str) -> str:
    value = (raw_path or "").strip()
    if not value:
        return value

    path = Path(value).expanduser().resolve()

    if path.is_dir() and (path / "done").exists():
        return str(path)

    if path.is_file() and path.name == "latest":
        latest_name = path.read_text().strip()
        latest_dir = (path.parent / latest_name).resolve()
        if (latest_dir / "done").exists():
            return str(latest_dir)
        raise FileNotFoundError(
            f"Checkpoint latest marker points to a missing or incomplete save: {latest_dir}"
        )

    if path.is_dir() and path.name == "ckpt":
        latest_file = path / "latest"
        if latest_file.exists():
            return resolve_official_checkpoint_path(str(latest_file))
        raise FileNotFoundError(f"Checkpoint directory has no 'latest' marker: {path}")

    if path.is_dir() and (path / "ckpt").is_dir():
        return resolve_official_checkpoint_path(str(path / "ckpt"))

    raise FileNotFoundError(
        "Checkpoint path must point to a completed official DreamerV3 save folder "
        f"or its parent `ckpt` directory. Received: {path}"
    )


def load_official_config(args: argparse.Namespace, runtime: JaxRuntimeConfig):
    import elements
    import ruamel.yaml as yaml

    source_root = resolve_official_source_root()
    if source_root is None:
        source_hint = os.environ.get("PIOS_DREAMERV3_SRC", "/tmp/dreamerv3-official")
        raise FileNotFoundError(
            "Official DreamerV3 source checkout could not be located. "
            f"Expected a checkout root with `dreamerv3/configs.yaml`, but none was found. Last hint: {source_hint}"
        )
    config_path = source_root / "dreamerv3" / "configs.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Official DreamerV3 config not found at {config_path}. "
            "Set `PIOS_DREAMERV3_SRC` to the official source checkout."
        )
    configs = yaml.YAML(typ="safe").load(config_path.read_text())

    config = elements.Config(configs["defaults"])
    config = config.update(configs["dmc_proprio"])
    if args.size != "size1m":
        config = config.update(configs[args.size])

    config = config.update(
        logdir=args.logdir,
        script=args.script,
        seed=args.seed,
        task="gym_pios",
        random_agent=False,
    )
    config = config.update(
        logger={
            **config.logger,
            "outputs": ["jsonl"],
            "filter": "episode/score|episode/length|epstats/|train/loss/|report/",
            "timer": True,
        }
    )
    config = config.update(
        batch_size=args.batch_size,
        batch_length=args.batch_length,
        report_length=max(16, min(args.batch_length, 32)),
    )
    # Imagination-horizon ablation override (training-only; default keeps 15).
    # imag_length lives under the nested `agent.` section, so set it via the flat
    # dotted-key form (same pattern as the jax.profiler override below).
    if getattr(args, "imag_length", None) is not None:
        config = elements.Config(
            {**config.flat, "agent.imag_length": int(args.imag_length)}
        )
    config = config.update(
        run={
            **config.run,
            "steps": int(args.steps),
            "envs": resolve_env_count(int(args.envs), runtime.platform),
            "eval_envs": resolve_env_count(
                int(args.eval_envs), runtime.platform, eval_mode=True
            ),
            "eval_eps": int(args.eval_eps),
            "train_ratio": float(args.train_ratio),
            "log_every": int(args.log_every),
            "report_every": int(args.report_every),
            "save_every": int(args.save_every),
            "from_checkpoint": args.from_checkpoint,
        }
    )
    config = config.update(
        jax={
            **config.jax,
            "platform": runtime.platform,
            "compute_dtype": runtime.compute_dtype,
            "policy_devices": [0],
            "train_devices": [0],
            "prealloc": runtime.prealloc,
        }
    )
    config = elements.Config(
        {
            **config.flat,
            "jax.profiler": resolve_jax_profiler_mode(
                args.jax_profiler, runtime.platform
            ),
        }
    )
    config = config.update(
        replay={
            **config.replay,
            "size": int(2e5 if args.size != "debug" else 1e4),
        }
    )
    return config


def build_run_args(config):
    import elements

    return elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
        from_checkpoint_regex="",
    )


def build_policy_spaces(elements_module):
    from ppo_shared import (
        CONTINUOUS_ACTION_HIGH,
        CONTINUOUS_ACTION_LOW,
        DREAMER_ACTION_DIM,
        OBS_DIM,
    )

    obs_space = {
        "vector": elements_module.Space(np.float32, (OBS_DIM,), 0.0, 1.0),
        "reward": elements_module.Space(np.float32),
        "is_first": elements_module.Space(bool),
        "is_last": elements_module.Space(bool),
        "is_terminal": elements_module.Space(bool),
    }
    act_space = {
        "action": elements_module.Space(
            np.float32,
            (DREAMER_ACTION_DIM,),
            CONTINUOUS_ACTION_LOW,
            CONTINUOUS_ACTION_HIGH,
        ),
    }
    return obs_space, act_space


def load_official_policy_agent(
    checkpoint_path: str,
    *,
    logdir: str | None = None,
    size: str = "size1m",
    seed: int = 0,
    step_hours: float = 6.0,
    jax_platform: str = "auto",
    compute_dtype: str = "auto",
    online_city_graph: bool = False,
    jax_profiler: str = "auto",
):
    checkpoint_path = resolve_official_checkpoint_path(checkpoint_path)
    runtime = choose_jax_runtime(jax_platform, compute_dtype)
    configure_process_env(runtime, allow_online_city_graph=online_city_graph)
    modules = import_official_runtime()

    elements = modules["elements"]
    Agent = modules["Agent"]

    runtime_logdir = (
        Path(logdir).expanduser().resolve()
        if logdir
        else (Path(checkpoint_path).parent / "policy_runtime").resolve()
    )
    args = argparse.Namespace(
        script="eval_only",
        logdir=str(runtime_logdir),
        from_checkpoint=checkpoint_path,
        steps=1,
        scenario="baseline",
        seed=int(seed),
        step_hours=float(step_hours),
        envs=1,
        eval_envs=1,
        eval_eps=1,
        train_ratio=32.0,
        batch_size=16,
        batch_length=64,
        size=size,
        log_every=30,
        report_every=120,
        save_every=300,
        jax_platform=jax_platform,
        compute_dtype=compute_dtype,
        online_city_graph=online_city_graph,
        jax_profiler=jax_profiler,
        metrics_write_mode="buffered",
    )
    config = load_official_config(args, runtime)
    config = elements.Config({**config.flat, "jax.precompile": False})

    obs_space, act_space = build_policy_spaces(elements)
    agent = Agent(
        obs_space,
        act_space,
        elements.Config(
            **config.agent,
            logdir=config.logdir,
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
        ),
    )

    checkpoint = elements.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(checkpoint_path, keys=["agent"])
    return agent


def make_agent_factory(config, make_env, Agent):
    import elements

    def make_agent():
        env = make_env(0)
        notlog = lambda key: not key.startswith("log/")
        obs_space = {key: value for key, value in env.obs_space.items() if notlog(key)}
        act_space = {
            key: value for key, value in env.act_space.items() if key != "reset"
        }
        env.close()
        return Agent(
            obs_space,
            act_space,
            elements.Config(
                **config.agent,
                logdir=config.logdir,
                seed=config.seed,
                jax=config.jax,
                batch_size=config.batch_size,
                batch_length=config.batch_length,
                replay_context=config.replay_context,
                report_length=config.report_length,
                replica=config.replica,
                replicas=config.replicas,
            ),
        )

    return make_agent


def make_logger_factory(config, metrics_write_mode: str):
    import elements

    step = elements.Counter()
    logdir = config.logdir
    multiplier = config.env.get(config.task.split("_")[0], {}).get("repeat", 1)
    outputs = []
    outputs.append(elements.logger.TerminalOutput(config.logger.filter, "Agent"))
    for output in config.logger.outputs:
        if output == "jsonl":
            if metrics_write_mode == "buffered":
                outputs.append(BufferedJSONLOutput(logdir, "metrics.jsonl"))
                outputs.append(
                    BufferedJSONLOutput(logdir, "scores.jsonl", "episode/score")
                )
            else:
                outputs.append(elements.logger.JSONLOutput(logdir, "metrics.jsonl"))
                outputs.append(
                    elements.logger.JSONLOutput(logdir, "scores.jsonl", "episode/score")
                )
        elif output == "scope":
            outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
        elif output == "tensorboard":
            outputs.append(elements.logger.TensorBoardOutput(logdir, config.logger.fps))
        elif output == "wandb":
            name = "/".join(logdir.split("/")[-4:])
            outputs.append(elements.logger.WandBOutput(name))
        else:
            raise NotImplementedError(output)

    logger = elements.Logger(step, outputs, multiplier)
    return lambda: logger


def main() -> None:
    args = parse_args()
    if args.from_checkpoint:
        args.from_checkpoint = resolve_official_checkpoint_path(args.from_checkpoint)
    runtime = choose_jax_runtime(args.jax_platform, args.compute_dtype)
    configure_process_env(runtime, allow_online_city_graph=args.online_city_graph)
    modules = import_official_runtime()

    elements = modules["elements"]
    embodied = modules["embodied"]
    portal = modules["portal"]
    OfficialDreamerBloodEnv = modules["OfficialDreamerBloodEnv"]
    plot_metrics_for_logdir = modules["plot_metrics_for_logdir"]
    SCENARIOS = modules["SCENARIOS"]
    TRAINING_SCENARIO_KEYS = modules["TRAINING_SCENARIO_KEYS"]
    HOLDOUT_SCENARIO_KEYS = modules["HOLDOUT_SCENARIO_KEYS"]
    load_training_context = modules["load_training_context"]
    official_main = modules["official_main"]
    Agent = modules["Agent"]

    scenario_keys = parse_scenarios(
        args.scenario,
        SCENARIOS,
        training_scenario_keys=TRAINING_SCENARIO_KEYS,
        holdout_scenario_keys=HOLDOUT_SCENARIO_KEYS,
    )
    sim_context = load_training_context()
    config = load_official_config(args, runtime)
    run_args = build_run_args(config)

    print(
        "[official_dreamerv3] JAX runtime | "
        f"platform={runtime.platform} dtype={runtime.compute_dtype} "
        f"prealloc={str(runtime.prealloc).lower()} | {runtime.reason}"
    )
    print(
        "[official_dreamerv3] Parallel envs | "
        f"train={config.run.envs} eval={config.run.eval_envs} "
        f"logdir={args.logdir}"
    )
    print(
        "[official_dreamerv3] JAX profiler | "
        f"{'enabled' if config.jax.profiler else 'disabled'}"
    )
    print(f"[official_dreamerv3] Metrics writes | {args.metrics_write_mode}")
    print(
        "[official_dreamerv3] Simulator graph | "
        f"{'online fetch enabled' if args.online_city_graph else 'offline fast path enabled'}"
    )
    if args.budget_variant:
        budget_desc = (
            f"fixed={args.fixed_budget}"
            if args.fixed_budget is not None
            else "per-scenario defaults"
        )
        print(
            f"[official_dreamerv3] Budget variant | "
            f"ENABLED — reward penalises budget/cost, early termination on "
            f"exhaustion, budget={budget_desc}"
        )

    def init():
        elements.timer.global_timer.enabled = config.logger.timer

    portal.setup(
        errfile=False,
        clientkw=dict(logging_color="cyan"),
        serverkw=dict(logging_color="cyan"),
        initfns=[init],
        ipv6=config.ipv6,
    )

    OfficialDreamerBloodBudgetEnv = modules["OfficialDreamerBloodBudgetEnv"]

    def make_env(index: int):
        if args.budget_variant:
            env = OfficialDreamerBloodBudgetEnv(
                scenario_keys,
                sim_context,
                step_hours=args.step_hours,
                seed=args.seed + index * 1000,
                fixed_budget=args.fixed_budget,
            )
        else:
            env = OfficialDreamerBloodEnv(
                scenario_keys,
                sim_context,
                step_hours=args.step_hours,
                seed=args.seed + index * 1000,
            )
        return official_main.wrap_env(env, config)

    make_agent = make_agent_factory(config, make_env, Agent)
    make_replay = bind(official_main.make_replay, config, "replay")
    make_eval_replay = bind(official_main.make_replay, config, "eval_replay", "eval")
    make_stream = bind(official_main.make_stream, config)
    make_logger = make_logger_factory(config, args.metrics_write_mode)
    logger = make_logger()
    interrupted = False

    try:
        if args.script == "train":
            embodied.run.train(
                make_agent,
                make_replay,
                make_env,
                make_stream,
                lambda: logger,
                run_args,
            )
        elif args.script == "train_eval":
            embodied.run.train_eval(
                make_agent,
                make_replay,
                make_eval_replay,
                make_env,
                make_env,
                make_stream,
                lambda: logger,
                run_args,
            )
        elif args.script == "eval_only":
            embodied.run.eval_only(
                make_agent,
                make_env,
                lambda: logger,
                run_args,
            )
        else:
            raise NotImplementedError(args.script)
    except KeyboardInterrupt:
        interrupted = True
        print(
            "[official_dreamerv3] Interrupted by user. Flushing buffered metrics before exit."
        )
    finally:
        logger.close()

    plot_path = plot_metrics_for_logdir(Path(args.logdir))
    if plot_path is not None:
        print(f"Saved metrics plot to {plot_path}")
    else:
        print(f"No plottable metrics found in {Path(args.logdir) / 'metrics.jsonl'}")
    if interrupted:
        print("[official_dreamerv3] Training stopped early after flushing metrics.")


if __name__ == "__main__":
    main()
