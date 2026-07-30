from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT_DIR / "training_scripts"
BACKEND_DIR = ROOT_DIR.parent / "backendMulti"
SCHEDULE_PATH = ROOT_DIR / "config" / "training_schedule.json"

for path in (ROOT_DIR, TRAINING_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_enabled_training_scripts_import_from_schedule():
    schedule = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    enabled_scripts = [
        entry["script"]
        for entry in schedule.get("scripts", {}).values()
        if entry.get("enabled") is True and entry.get("script")
    ]

    assert enabled_scripts

    for script_name in enabled_scripts:
        script_path = TRAINING_DIR / script_name
        spec = importlib.util.spec_from_file_location(
            f"{script_path.stem}_import_test",
            script_path,
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
