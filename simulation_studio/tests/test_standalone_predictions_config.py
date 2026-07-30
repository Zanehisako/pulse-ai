import json

import pytest

from simulation_studio.app.standalone_predictions_config import (
    STUDIO_ROOT,
    StandalonePredictionsConfigError,
    load_standalone_predictions_config,
    ml_config_root,
    validate_standalone_predictions_config,
)


def test_ml_config_root_resolves_from_studio_root_not_config_subdir():
    cfg = load_standalone_predictions_config()
    root = ml_config_root(cfg, config_dir=STUDIO_ROOT / "config")
    assert (root / "scheduled_predictions.json").is_file()


def test_validate_requires_entity_source_map(tmp_path):
    config_path = tmp_path / "standalone_predictions.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "1",
                "enabled": True,
                "ml_config_root": ".",
                "entity_source_map": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StandalonePredictionsConfigError):
        validate_standalone_predictions_config(
            json.loads(config_path.read_text(encoding="utf-8")),
            config_dir=config_path.parent,
        )