"""Unit tests for DreamerV4 Metal GPU / MPS optimisation helpers.

These tests validate the new Apple-Silicon-aware device selection,
MPS runtime configuration, ``torch.compile`` wrapper, and model-build
pipeline introduced to bring DreamerV4 native training speed in line
with the DreamerV3 JAX-Metal runner.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

SIMULATOR_DIR = Path(__file__).resolve().parents[1] / "simulator"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from dreamerv4_agent import (
    RSSM,
    Actor,
    Critic,
    _auto_detect_device,
    _build_models_from_config,
    _configure_mps_runtime,
    _is_apple_silicon,
    _maybe_compile,
    _mps_is_available,
    _resolve_device,
    _resolve_native_env_count,
    _resolve_native_model_config,
)

# ---------------------------------------------------------------------------
# _is_apple_silicon
# ---------------------------------------------------------------------------


def test_is_apple_silicon_on_darwin_arm64():
    with patch("dreamerv4_agent.platform") as mock_platform:
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"
        assert _is_apple_silicon() is True


def test_is_apple_silicon_on_darwin_aarch64():
    with patch("dreamerv4_agent.platform") as mock_platform:
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "aarch64"
        assert _is_apple_silicon() is True


def test_is_apple_silicon_on_linux_x86():
    with patch("dreamerv4_agent.platform") as mock_platform:
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"
        assert _is_apple_silicon() is False


def test_is_apple_silicon_on_darwin_x86():
    with patch("dreamerv4_agent.platform") as mock_platform:
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "x86_64"
        assert _is_apple_silicon() is False


# ---------------------------------------------------------------------------
# _mps_is_available
# ---------------------------------------------------------------------------


def test_mps_is_available_returns_bool():
    result = _mps_is_available()
    assert isinstance(result, bool)


def test_mps_is_available_false_when_no_mps_attr():
    with patch.object(torch.backends, "mps", create=False, new=None):
        # When mps attribute doesn't behave as expected, should not crash
        result = _mps_is_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# _configure_mps_runtime
# ---------------------------------------------------------------------------


def test_configure_mps_runtime_sets_env_vars(monkeypatch):
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    _configure_mps_runtime()

    assert os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") == "0.0"
    assert os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO") == "0.0"
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_configure_mps_runtime_does_not_override_existing_env(monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
    monkeypatch.setenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    _configure_mps_runtime()

    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.7"
    assert os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.3"
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


# ---------------------------------------------------------------------------
# _auto_detect_device
# ---------------------------------------------------------------------------


def test_auto_detect_device_returns_torch_device():
    device = _auto_detect_device()
    assert isinstance(device, torch.device)


def test_auto_detect_device_falls_back_to_cpu_when_no_gpu():
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("dreamerv4_agent._mps_is_available", return_value=False),
    ):
        device = _auto_detect_device()
        assert device.type == "cpu"


def test_auto_detect_device_prefers_cuda_over_mps():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("dreamerv4_agent._mps_is_available", return_value=True),
    ):
        device = _auto_detect_device()
        assert device.type == "cuda"


def test_auto_detect_device_selects_mps_when_available_and_no_cuda():
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("dreamerv4_agent._mps_is_available", return_value=True),
        patch("dreamerv4_agent._configure_mps_runtime") as mock_cfg,
    ):
        device = _auto_detect_device()
        assert device.type == "mps"
        mock_cfg.assert_called_once()


# ---------------------------------------------------------------------------
# _resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_auto_delegates_to_auto_detect():
    with patch("dreamerv4_agent._auto_detect_device") as mock:
        mock.return_value = torch.device("cpu")
        result = _resolve_device("auto")
        assert result == torch.device("cpu")
        mock.assert_called_once()


def test_resolve_device_none_delegates_to_auto_detect():
    with patch("dreamerv4_agent._auto_detect_device") as mock:
        mock.return_value = torch.device("cpu")
        result = _resolve_device(None)
        assert result == torch.device("cpu")
        mock.assert_called_once()


def test_resolve_device_explicit_cpu():
    device = _resolve_device("cpu")
    assert device == torch.device("cpu")


def test_resolve_device_explicit_mps_configures_runtime():
    with patch("dreamerv4_agent._configure_mps_runtime") as mock_cfg:
        device = _resolve_device("mps")
        assert device.type == "mps"
        mock_cfg.assert_called_once()


def test_resolve_device_explicit_cuda():
    with patch("dreamerv4_agent._configure_mps_runtime") as mock_cfg:
        device = _resolve_device("cuda")
        assert device.type == "cuda"
        mock_cfg.assert_not_called()


# ---------------------------------------------------------------------------
# _maybe_compile
# ---------------------------------------------------------------------------


class _TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_maybe_compile_returns_module_on_cpu():
    module = _TinyModule()
    result = _maybe_compile(module, torch.device("cpu"))
    # On CPU, compilation is skipped — the original module is returned.
    assert result is module


def test_maybe_compile_respects_disable_flag():
    module = _TinyModule()
    result = _maybe_compile(module, torch.device("mps"), disable=True)
    assert result is module


def test_maybe_compile_works_on_mps_device():
    """When MPS is available, _maybe_compile should attempt aot_eager compilation."""
    if not _mps_is_available():
        pytest.skip("MPS not available on this machine")

    module = _TinyModule().to("mps")
    result = _maybe_compile(module, torch.device("mps"))
    # The compiled module should still be callable.
    x = torch.randn(2, 4, device="mps")
    out = result(x)
    assert out.shape == (2, 2)


def test_maybe_compile_graceful_on_old_pytorch():
    """If torch.compile is missing (PyTorch < 2.0), just return the module."""
    module = _TinyModule()
    with patch.object(torch, "compile", create=False, new=None):
        # Remove compile attribute to simulate old PyTorch
        delattr_needed = hasattr(torch, "compile")
        if delattr_needed:
            original = torch.compile
            delattr(torch, "compile")
        try:
            result = _maybe_compile(module, torch.device("mps"))
            assert result is module
        finally:
            if delattr_needed:
                torch.compile = original  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _resolve_native_env_count
# ---------------------------------------------------------------------------


def test_resolve_native_env_count_respects_explicit():
    assert _resolve_native_env_count(8) == 8


def test_resolve_native_env_count_auto_positive():
    count = _resolve_native_env_count(0)
    assert isinstance(count, int)
    assert 1 <= count <= 4


# ---------------------------------------------------------------------------
# _resolve_native_model_config
# ---------------------------------------------------------------------------


def test_resolve_native_model_config_default_preset():
    config = _resolve_native_model_config(obs_dim=42, action_dim=7, preset="default")
    assert config["obs_dim"] == 42
    assert config["action_dim"] == 7
    assert config["hidden_size"] == 256
    assert config["stoch_categories"] == 16
    assert config["stoch_classes"] == 16
    assert config["mlp_units"] == 256


def test_resolve_native_model_config_fast_preset():
    config = _resolve_native_model_config(obs_dim=42, action_dim=7, preset="fast")
    assert config["hidden_size"] == 128
    assert config["stoch_categories"] == 8


def test_resolve_native_model_config_overrides():
    config = _resolve_native_model_config(
        obs_dim=10,
        action_dim=3,
        preset="default",
        hidden_size=64,
        mlp_units=32,
    )
    assert config["hidden_size"] == 64
    assert config["mlp_units"] == 32
    # Non-overridden fields keep the preset value
    assert config["stoch_categories"] == 16


def test_resolve_native_model_config_rejects_unknown_preset():
    with pytest.raises(ValueError, match="Unknown native model preset"):
        _resolve_native_model_config(obs_dim=10, action_dim=3, preset="nonexistent")


def test_resolve_native_model_config_rejects_zero_override():
    with pytest.raises(ValueError, match="hidden_size must be >= 1"):
        _resolve_native_model_config(
            obs_dim=10, action_dim=3, preset="default", hidden_size=0
        )


# ---------------------------------------------------------------------------
# _build_models_from_config — with and without compile
# ---------------------------------------------------------------------------


_SMALL_CONFIG = {
    "obs_dim": 10,
    "action_dim": 3,
    "hidden_size": 16,
    "stoch_categories": 4,
    "stoch_classes": 4,
    "mlp_units": 16,
}


def test_build_models_from_config_returns_four_models():
    rssm, actor, critic, target_critic = _build_models_from_config(
        _SMALL_CONFIG, torch.device("cpu")
    )
    assert isinstance(rssm, (RSSM, nn.Module))
    assert isinstance(actor, (Actor, nn.Module))
    assert isinstance(critic, (Critic, nn.Module))
    assert isinstance(target_critic, (Critic, nn.Module))


def test_build_models_from_config_compile_false_on_cpu():
    """compile_models=True should still work on CPU (compilation is skipped)."""
    rssm, actor, critic, target_critic = _build_models_from_config(
        _SMALL_CONFIG, torch.device("cpu"), compile_models=True
    )
    # On CPU, _maybe_compile returns the original module unchanged.
    assert isinstance(rssm, (RSSM, nn.Module))


def test_build_models_target_critic_matches_critic():
    _, _, critic, target_critic = _build_models_from_config(
        _SMALL_CONFIG, torch.device("cpu")
    )
    for p, tp in zip(critic.parameters(), target_critic.parameters()):
        assert torch.equal(p.data, tp.data)


@pytest.mark.skipif(not _mps_is_available(), reason="MPS not available")
def test_build_models_on_mps_with_compile():
    rssm, actor, critic, target_critic = _build_models_from_config(
        _SMALL_CONFIG, torch.device("mps"), compile_models=True
    )
    # Verify the models are usable on MPS after compilation
    state = rssm.initial_state(2, torch.device("mps"))
    assert state["h"].device.type == "mps"


# ---------------------------------------------------------------------------
# RSSM / Actor / Critic — basic forward pass on CPU (smoke test)
# ---------------------------------------------------------------------------


def test_rssm_forward_cpu():
    rssm = RSSM(
        obs_dim=10,
        action_dim=3,
        hidden_size=16,
        stoch_categories=4,
        stoch_classes=4,
        mlp_units=16,
    )
    B, T = 2, 5
    obs = torch.randn(B, T, 10)
    actions = torch.randn(B, T, 3)
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0

    out = rssm.observe(obs, actions, is_first)
    assert out["feat_seq"].shape == (B, T, 16 + 4 * 4)
    assert out["final_state"]["h"].shape == (B, 16)
    assert out["final_state"]["z"].shape == (B, 4 * 4)


def test_actor_forward_cpu():
    actor = Actor(state_feat_dim=32, action_dim=3, mlp_units=16)
    feat = torch.randn(4, 32)
    action, log_prob, entropy = actor(feat, sample=True)
    assert action.shape == (4, 3)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)
    assert (action >= -1.0).all() and (action <= 1.0).all()


def test_critic_forward_cpu():
    critic = Critic(state_feat_dim=32, mlp_units=16)
    feat = torch.randn(4, 32)
    value = critic(feat)
    assert value.shape == (4, 1)


# ---------------------------------------------------------------------------
# MPS integration smoke tests (skipped when hardware unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _mps_is_available(), reason="MPS not available")
def test_rssm_forward_on_mps():
    rssm = RSSM(
        obs_dim=10,
        action_dim=3,
        hidden_size=16,
        stoch_categories=4,
        stoch_classes=4,
        mlp_units=16,
    ).to("mps")
    B, T = 2, 3
    obs = torch.randn(B, T, 10, device="mps")
    actions = torch.randn(B, T, 3, device="mps")
    is_first = torch.zeros(B, T, device="mps")

    out = rssm.observe(obs, actions, is_first)
    assert out["feat_seq"].device.type == "mps"


@pytest.mark.skipif(not _mps_is_available(), reason="MPS not available")
def test_actor_critic_on_mps():
    actor = Actor(state_feat_dim=32, action_dim=3, mlp_units=16).to("mps")
    critic = Critic(state_feat_dim=32, mlp_units=16).to("mps")
    feat = torch.randn(4, 32, device="mps")

    action, lp, ent = actor(feat, sample=True)
    assert action.device.type == "mps"

    value = critic(feat)
    assert value.device.type == "mps"


@pytest.mark.skipif(not _mps_is_available(), reason="MPS not available")
def test_non_blocking_transfer_to_mps():
    """Verify non_blocking=True does not error on MPS."""
    arr = torch.randn(4, 10)
    t = arr.to("mps", non_blocking=True)
    assert t.device.type == "mps"


@pytest.mark.skipif(not _mps_is_available(), reason="MPS not available")
def test_mps_synchronize_available():
    """torch.mps.synchronize should be callable without error."""
    assert hasattr(torch.mps, "synchronize")
    torch.mps.synchronize()


@pytest.mark.skipif(not _mps_is_available(), reason="MPS not available")
def test_mps_empty_cache_available():
    """torch.mps.empty_cache should be callable without error."""
    assert hasattr(torch.mps, "empty_cache")
    torch.mps.empty_cache()
