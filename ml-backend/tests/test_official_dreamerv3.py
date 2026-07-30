from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

SIMULATOR_DIR = (Path(__file__).resolve().parents[1] / "simulator").resolve()
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

import engine
import dreamerv3_runtime_paths as runtime_paths

from official_dreamerv3 import (
    BufferedJSONLOutput,
    JaxRuntimeConfig,
    build_policy_spaces,
    choose_jax_runtime,
    configure_process_env,
    resolve_official_checkpoint_path,
    resolve_env_count,
    resolve_jax_profiler_mode,
)
from dreamerv3_runtime_paths import (
    discover_official_runtime_paths,
    resolve_official_source_root,
)
from ppo_shared import (
    ACTION_KEYS,
    CONTINUOUS_ACTION_HIGH,
    CONTINUOUS_ACTION_LOW,
    DREAMER_ACTION_DIM,
    continuous_action_levels,
)


def test_choose_jax_runtime_prefers_metal_on_apple_silicon():
    runtime = choose_jax_runtime(
        "auto",
        "auto",
        is_apple_silicon=True,
        has_jax=True,
        has_metal_plugin=True,
        metal_runtime_available=True,
    )

    assert runtime.platform == "METAL"
    assert runtime.compute_dtype == "float32"
    assert runtime.prealloc is False


def test_choose_jax_runtime_falls_back_to_cpu_when_metal_probe_fails():
    runtime = choose_jax_runtime(
        "auto",
        "auto",
        is_apple_silicon=True,
        has_jax=True,
        has_metal_plugin=True,
        metal_runtime_available=False,
    )

    assert runtime.platform == "cpu"
    assert runtime.compute_dtype == "float32"
    assert "probe failed" in runtime.reason


def test_choose_jax_runtime_falls_back_to_cpu_when_metal_is_missing():
    runtime = choose_jax_runtime(
        "metal",
        "auto",
        is_apple_silicon=True,
        has_jax=True,
        has_metal_plugin=False,
    )

    assert runtime.platform == "cpu"
    assert runtime.compute_dtype == "float32"
    assert "jax-metal" in runtime.reason


def test_resolve_env_count_uses_parallel_envs_for_metal():
    assert resolve_env_count(0, "METAL", cpu_count=10) == 4
    assert resolve_env_count(0, "METAL", eval_mode=True, cpu_count=10) == 1
    assert resolve_env_count(3, "METAL", cpu_count=10) == 3
    assert resolve_env_count(0, "cpu", cpu_count=10) == 1


def test_configure_process_env_sets_runtime_defaults(monkeypatch):
    monkeypatch.delenv("ENABLE_PJRT_COMPATIBILITY", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.delenv("PIOS_SIM_OFFLINE", raising=False)

    configure_process_env(
        JaxRuntimeConfig(
            platform="METAL",
            compute_dtype="float32",
            prealloc=False,
            reason="test",
        )
    )

    assert os.environ["ENABLE_PJRT_COMPATIBILITY"] == "1"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert os.environ["PIOS_SIM_OFFLINE"] == "1"


def test_configure_process_env_can_allow_online_city_graph(monkeypatch):
    monkeypatch.delenv("PIOS_SIM_OFFLINE", raising=False)

    configure_process_env(
        JaxRuntimeConfig(
            platform="cpu",
            compute_dtype="float32",
            prealloc=False,
            reason="test",
        ),
        allow_online_city_graph=True,
    )

    assert "PIOS_SIM_OFFLINE" not in os.environ


def test_resolve_jax_profiler_mode_disables_profiler_on_metal_by_default():
    assert resolve_jax_profiler_mode("auto", "METAL") is False
    assert resolve_jax_profiler_mode("auto", "cpu") is True
    assert resolve_jax_profiler_mode("off", "cpu") is False
    assert resolve_jax_profiler_mode("on", "METAL") is True


def test_buffered_jsonl_output_flushes_only_on_wait(tmp_path):
    output = BufferedJSONLOutput(tmp_path, "metrics.jsonl")
    summaries = (
        (10, "episode/score", np.asarray(12.5)),
        (10, "episode/length", np.asarray(57)),
    )

    output(summaries)
    assert not (tmp_path / "metrics.jsonl").exists()

    output.wait()

    content = (tmp_path / "metrics.jsonl").read_text()
    assert '"step": 10' in content
    assert '"episode/score": 12.5' in content


def test_resolve_official_checkpoint_path_accepts_completed_save_dir(tmp_path):
    save_dir = tmp_path / "ckpt" / "20260331T000000F000000"
    save_dir.mkdir(parents=True)
    (save_dir / "done").write_bytes(b"")

    resolved = resolve_official_checkpoint_path(str(save_dir))

    assert resolved == str(save_dir.resolve())


def test_resolve_official_checkpoint_path_expands_ckpt_dir_to_latest_save(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    save_dir = ckpt_dir / "20260331T000000F000000"
    save_dir.mkdir(parents=True)
    (save_dir / "done").write_bytes(b"")
    (ckpt_dir / "latest").write_text(save_dir.name)

    resolved = resolve_official_checkpoint_path(str(ckpt_dir))

    assert resolved == str(save_dir.resolve())


def test_engine_resolve_dreamer_checkpoint_path_auto_discovers_available_run(
    monkeypatch, tmp_path
):
    runs_root = tmp_path / "dreamerv3_runs"
    ckpt_dir = runs_root / "m1_continuous_run8_kpi_aligned" / "ckpt"
    save_dir = ckpt_dir / "20260331T000000F000000"
    save_dir.mkdir(parents=True)
    (save_dir / "done").write_bytes(b"")
    (ckpt_dir / "latest").write_text(save_dir.name, encoding="utf-8")

    fake_engine = tmp_path / "engine.py"
    fake_engine.write_text("", encoding="utf-8")
    monkeypatch.setattr(engine, "__file__", str(fake_engine))

    resolved = engine.resolve_dreamer_checkpoint_path(engine.DREAMER_V3_CONTROLLER_KEY)

    assert resolved == str((runs_root / "m1_continuous_run8_kpi_aligned").resolve())


def test_build_policy_spaces_uses_continuous_dreamer_actions():
    class FakeSpace:
        def __init__(self, dtype, shape=(), low=None, high=None):
            self.dtype = dtype
            self.shape = shape
            self.low = low
            self.high = high

    class FakeElements:
        Space = FakeSpace

    _obs_space, act_space = build_policy_spaces(FakeElements)
    action_space = act_space["action"]

    assert action_space.dtype == np.float32
    assert action_space.shape == (DREAMER_ACTION_DIM,)
    assert action_space.low == CONTINUOUS_ACTION_LOW
    assert action_space.high == CONTINUOUS_ACTION_HIGH


def test_discover_official_runtime_paths_accepts_checkout_root(tmp_path):
    checkout = tmp_path / "dreamerv3-official"
    (checkout / "dreamerv3").mkdir(parents=True)
    (checkout / "dreamerv3" / "configs.yaml").write_text("defaults: {}\n")
    (checkout / "embodied").mkdir()

    paths = discover_official_runtime_paths([checkout])

    assert paths[0] == str(checkout.resolve())
    assert str(checkout.resolve()) in paths
    assert resolve_official_source_root([checkout]) == checkout.resolve()


def test_discover_official_runtime_paths_accepts_nested_package_path(tmp_path):
    checkout = tmp_path / "dreamerv3-official"
    package_dir = checkout / "dreamerv3"
    package_dir.mkdir(parents=True)
    (package_dir / "configs.yaml").write_text("defaults: {}\n")
    (checkout / "embodied").mkdir()

    paths = discover_official_runtime_paths([package_dir])

    assert paths[0] == str(checkout.resolve())
    assert str(checkout.resolve()) in paths
    assert resolve_official_source_root([package_dir]) == checkout.resolve()


def test_discover_official_runtime_paths_accepts_virtualenv_root(tmp_path):
    env_root = tmp_path / "dreamerv3-pkgs"
    site_packages = env_root / "lib" / "python3.12" / "site-packages"
    (site_packages / "dreamerv3").mkdir(parents=True)
    (site_packages / "embodied").mkdir()
    (site_packages / "portal").mkdir()

    paths = discover_official_runtime_paths([env_root])

    assert paths[0] == str(site_packages.resolve())
    assert str(site_packages.resolve()) in paths


def test_discover_official_runtime_paths_prefers_explicit_pkg_env_without_fallback_pkgs(tmp_path, monkeypatch):
    explicit_pkgs = tmp_path / "dreamerv3-pkgs-metal"
    (explicit_pkgs / "dreamerv3").mkdir(parents=True)
    (explicit_pkgs / "embodied").mkdir()
    (explicit_pkgs / "portal").mkdir()
    (explicit_pkgs / "jax").mkdir()

    checkout = tmp_path / "dreamerv3-official"
    (checkout / "dreamerv3").mkdir(parents=True)
    (checkout / "dreamerv3" / "configs.yaml").write_text("defaults: {}\n")
    (checkout / "embodied").mkdir(exist_ok=True)

    fallback_pkgs = tmp_path / "fallback-pkgs"
    (fallback_pkgs / "dreamerv3").mkdir(parents=True)
    (fallback_pkgs / "embodied").mkdir()
    (fallback_pkgs / "portal").mkdir()
    (fallback_pkgs / "jax").mkdir()

    monkeypatch.setenv("PIOS_DREAMERV3_PKGS", str(explicit_pkgs))

    paths = discover_official_runtime_paths([checkout])

    assert paths[0] == str(explicit_pkgs.resolve())
    assert str(checkout.resolve()) in paths
    assert str(fallback_pkgs.resolve()) not in paths


def test_default_runtime_candidates_prefer_isolated_metal_cache():
    candidates = runtime_paths.DEFAULT_RUNTIME_CANDIDATES
    assert candidates.index(SIMULATOR_DIR / "cache" / "dreamerv3-pkgs-metal011") < candidates.index(
        SIMULATOR_DIR / "cache" / "dreamerv3-pkgs"
    )


def test_continuous_action_levels_use_full_range_as_intensity():
    raw_action = np.zeros(DREAMER_ACTION_DIM, dtype=np.float32)
    raw_action[0] = -1.7
    raw_action[1] = 0.0
    raw_action[2] = 0.45
    raw_action[3] = 1.7

    levels = continuous_action_levels(raw_action)

    assert levels[ACTION_KEYS[0]] == 0.0
    assert np.isclose(levels[ACTION_KEYS[1]], 0.5)
    assert np.isclose(levels[ACTION_KEYS[2]], 0.725)
    assert levels[ACTION_KEYS[3]] == 1.0
