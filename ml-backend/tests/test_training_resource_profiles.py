from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ML_BACKEND = Path(__file__).resolve().parents[1]
if str(ML_BACKEND) not in sys.path:
    sys.path.insert(0, str(ML_BACKEND))

from pios_ml_backend import training_resource_profiles as profiles  # noqa: E402


def _write_profile_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "training_resource_profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _profile_payload() -> dict:
    return {
        "selection_env_var": "PIOS_TRAINING_PROFILE",
        "default_profile": "default",
        "profiles": {
            "default": {
                "enabled": True,
                "environment": {},
                "retry_on_exit_codes": [],
                "retry_profiles": [],
                "training_overrides": {},
            },
            "quality": {
                "enabled": True,
                "environment": {
                    "PIOS_TREE_MODEL_THREADS": "4",
                    "OMP_NUM_THREADS": "1",
                },
                "retry_on_exit_codes": [-11],
                "retry_profiles": ["quality_retry"],
                "training_overrides": {
                    "*": {
                        "resource": {
                            "thread_env_var": "PIOS_TREE_MODEL_THREADS",
                            "tree_model_threads": 4,
                        },
                        "mlflow": {
                            "evaluate": {
                                "enabled": False,
                            },
                        },
                    },
                    "donor_model": {
                        "explainability": {
                            "shap": {
                                "enabled": False,
                            },
                        },
                    },
                },
            },
            "quality_retry": {
                "inherits": "quality",
                "enabled": True,
                "environment": {
                    "PIOS_TREE_MODEL_THREADS": "2",
                },
                "retry_on_exit_codes": [],
                "retry_profiles": [],
            },
        },
    }


def test_default_training_resource_profile_config_validates():
    payload = profiles.load_training_resource_profiles()

    assert "local_m1_quality" in payload["profiles"]
    assert payload["selection_env_var"] == "PIOS_TRAINING_PROFILE"
    assert -6 in profiles.resolve_training_resource_profile(
        "local_m1_quality"
    )["retry_on_exit_codes"]


def test_profile_environment_and_training_overrides_deep_merge(tmp_path: Path):
    config_path = _write_profile_config(tmp_path, _profile_payload())
    env = {"PIOS_TRAINING_PROFILE": "quality"}

    profiles.apply_selected_profile_environment(env, config_path=config_path)
    config = profiles.apply_profile_to_model_config(
        {
            "id": "donor_model",
            "registered_model_name": "donor_model_registry",
            "training": {
                "mlflow": {
                    "input_example": {
                        "enabled": True,
                    },
                },
            },
        },
        profile_name="quality",
        config_path=config_path,
    )

    assert env["PIOS_TREE_MODEL_THREADS"] == "4"
    assert env["OMP_NUM_THREADS"] == "1"
    assert config["training"]["resource"]["thread_env_var"] == "PIOS_TREE_MODEL_THREADS"
    assert config["training"]["resource"]["profile_name"] == "quality"
    assert config["training"]["mlflow"]["input_example"]["enabled"] is True
    assert config["training"]["mlflow"]["evaluate"]["enabled"] is False
    assert config["training"]["explainability"]["shap"]["enabled"] is False


def test_invalid_profile_config_fails_clearly(tmp_path: Path):
    payload = _profile_payload()
    payload["profiles"]["quality"]["environment"] = ["not", "a", "mapping"]
    config_path = _write_profile_config(tmp_path, payload)

    with pytest.raises(ValueError, match="environment must be an object"):
        profiles.load_training_resource_profiles(config_path)


def test_subprocess_runner_retries_segfault_with_configured_profile(monkeypatch, tmp_path: Path):
    config_path = _write_profile_config(tmp_path, _profile_payload())
    calls: list[dict[str, str]] = []

    def fake_run(args, *, env, cwd=None):  # noqa: ANN001
        calls.append(dict(env))
        return subprocess.CompletedProcess(
            args=args,
            returncode=-11 if len(calls) == 1 else 0,
        )

    monkeypatch.setattr(profiles.subprocess, "run", fake_run)
    retries: list[str] = []

    result = profiles.run_subprocess_with_profile_retries(
        ["python", "train.py"],
        env={"PIOS_TRAINING_PROFILE": "quality"},
        config_path=config_path,
        on_retry=lambda _result, retry_profile: retries.append(retry_profile),
    )

    assert result.returncode == 0
    assert retries == ["quality_retry"]
    assert [call["PIOS_TREE_MODEL_THREADS"] for call in calls] == ["4", "2"]
