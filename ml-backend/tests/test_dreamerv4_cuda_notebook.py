from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

SIMULATOR_DIR = Path(__file__).resolve().parents[1] / "simulator"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

import dreamerv4_agent
import dreamerv4_notebook
import engine


def test_resolve_dreamerv4_checkpoint_artifact_accepts_checkpoint_directory(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True)
    checkpoint = ckpt_dir / "dreamerv4_agent.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")

    resolved = dreamerv4_agent.resolve_dreamerv4_checkpoint_artifact(ckpt_dir)

    assert resolved == str(checkpoint.resolve())


def test_resolve_dreamerv4_checkpoint_artifact_accepts_model_pt_checkpoint_directory(
    tmp_path,
):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True)
    checkpoint = ckpt_dir / "model.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")

    resolved = dreamerv4_agent.resolve_dreamerv4_checkpoint_artifact(ckpt_dir)

    assert resolved == str(checkpoint.resolve())


def test_publish_dreamerv4_checkpoint_copies_to_dropin_location(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"dreamerv4-checkpoint")
    target_dir = tmp_path / "native" / "ckpt"

    published = dreamerv4_agent.publish_dreamerv4_checkpoint(source, target_dir)

    published_path = Path(published)
    assert published_path == (target_dir / "dreamerv4_agent.pt").resolve()
    assert published_path.read_bytes() == b"dreamerv4-checkpoint"

    latest_path = target_dir / "latest"
    assert latest_path.exists()
    if latest_path.is_symlink():
        assert os.readlink(str(latest_path)) == "dreamerv4_agent.pt"
    else:
        assert latest_path.read_text(encoding="utf-8").strip() == "dreamerv4_agent.pt"


def test_train_from_notebook_requires_cuda_when_requested(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA training was requested"):
        dreamerv4_notebook.train_from_notebook()


def test_notebook_runtime_report_lists_recommended_batch_size(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda idx: f"GPU-{idx}")
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda idx: (7, 5))

    report = dreamerv4_notebook.runtime_report()

    assert report["cuda_device_count"] == 2
    assert report["recommended_batch_size"] == 512
    assert report["cuda_devices"] == ["GPU-0", "GPU-1"]


def test_train_from_notebook_publishes_dropin_checkpoint(tmp_path, monkeypatch):
    source_ckpt = tmp_path / "runs" / "ckpt" / "dreamerv4_agent.pt"
    source_ckpt.parent.mkdir(parents=True)
    source_ckpt.write_bytes(b"checkpoint")

    def fake_train(_args):
        return {
            "checkpoint_path": str(source_ckpt),
            "metrics_path": str(tmp_path / "runs" / "metrics.jsonl"),
            "logdir": str(tmp_path / "runs"),
            "device": "cuda",
            "env_steps": 123,
            "grad_steps": 45,
            "episodes": 6,
            "train_ratio": 1.0,
            "envs": 4,
            "model_config": {"hidden_size": 128},
        }

    published_path = tmp_path / "published" / "dreamerv4_agent.pt"

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(dreamerv4_notebook, "train_dreamerv4_native", fake_train)
    monkeypatch.setattr(
        dreamerv4_notebook,
        "publish_dreamerv4_checkpoint",
        lambda checkpoint_path, destination: str(published_path),
    )

    result = dreamerv4_notebook.train_from_notebook(
        dreamerv4_notebook.DreamerV4NotebookConfig(
            steps=100,
            export_dir=str(tmp_path / "published"),
        )
    )

    assert result["checkpoint_path"] == str(source_ckpt)
    assert result["published_checkpoint"] == str(published_path)
    assert result["dropin_checkpoint"] == str(published_path)
    assert result["config"]["steps"] == 100


def test_notebook_config_defaults_to_larger_cuda_batch_and_data_parallel():
    cfg = dreamerv4_notebook.DreamerV4NotebookConfig()

    assert cfg.batch_size == 256
    assert cfg.use_data_parallel is True
    assert cfg.max_cuda_devices == 0


def test_load_dreamerv4_agent_accepts_checkpoint_directory(monkeypatch, tmp_path):
    ckpt_dir = tmp_path / "exported" / "ckpt"
    ckpt_dir.mkdir(parents=True)
    checkpoint = ckpt_dir / "dreamerv4_agent.pt"
    checkpoint.write_bytes(b"checkpoint")

    def fake_load(path, *, seed, step_hours):
        return {
            "path": path,
            "seed": seed,
            "step_hours": step_hours,
        }

    monkeypatch.setattr(
        dreamerv4_agent,
        "load_dreamerv4_pytorch_agent",
        fake_load,
    )

    loaded = engine.load_dreamerv4_agent(str(ckpt_dir), seed=17, step_hours=3.0)

    assert loaded["path"] == str(checkpoint.resolve())
    assert loaded["seed"] == 17
    assert loaded["step_hours"] == 3.0


def test_load_dreamerv4_agent_auto_discovers_model_pt_from_env_dir(
    monkeypatch, tmp_path
):
    ckpt_dir = tmp_path / "native" / "ckpt"
    ckpt_dir.mkdir(parents=True)
    checkpoint = ckpt_dir / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    def fake_load(path, *, seed, step_hours):
        return {
            "path": path,
            "seed": seed,
            "step_hours": step_hours,
        }

    monkeypatch.setenv("PIOS_DREAMERV4_CHECKPOINT", str(ckpt_dir))
    monkeypatch.setattr(
        dreamerv4_agent,
        "load_dreamerv4_pytorch_agent",
        fake_load,
    )

    loaded = engine.load_dreamerv4_agent(seed=23, step_hours=4.5)

    assert loaded["path"] == str(checkpoint.resolve())
    assert loaded["seed"] == 23
    assert loaded["step_hours"] == 4.5


def test_colab_notebook_is_valid_and_self_contained_for_multi_gpu_training():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "dreamerv4_colab_training.ipynb"
    )

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 8
    joined_sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "StandaloneBloodSupplyEnv" in joined_sources
    assert "Using standalone notebook trainer with optimized DDP." in joined_sources
    assert "OPTIMIZED_DDP_NOTEBOOK = True" in joined_sources
    assert "if 'OPTIMIZED_DDP_NOTEBOOK = True' in entry:" in joined_sources
    assert "from dreamerv4_notebook import" not in joined_sources
    assert "dreamerv4_notebook.py" not in joined_sources
    assert "DistributedDataParallel as DDP" in joined_sources
    assert "\"torch.distributed.run\"" in joined_sources
    assert "torch.cuda.amp.GradScaler" in joined_sources
    assert "torch.autocast" in joined_sources
    assert "use_ddp=True" in joined_sources
    assert "mixed_precision=True" in joined_sources
    assert "recommended_model_preset" in joined_sources
    assert "model_preset=\"default\"" in joined_sources
    assert "train_ratio=0.5" in joined_sources
    assert "max_train_steps_per_iter=2" in joined_sources
    assert "train_start_buffer" in joined_sources
    assert "nn.DataParallel(" not in joined_sources
    assert "recommended_batch_size" in joined_sources
    assert "github.com/mesellemahmed/pios.git" not in joined_sources
    assert "git clone" not in joined_sources
    assert "pip" in joined_sources
