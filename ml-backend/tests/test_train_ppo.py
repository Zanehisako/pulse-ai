from __future__ import annotations

import sys
from pathlib import Path


SIMULATOR_DIR = Path(__file__).resolve().parents[1] / "simulator"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from scenarios import SCENARIOS
from train_ppo import _parse_scenario_keys


def test_parse_scenario_keys_supports_all():
    assert _parse_scenario_keys("all", SCENARIOS) == list(SCENARIOS.keys())


def test_parse_scenario_keys_supports_comma_separated_values():
    assert _parse_scenario_keys("baseline, donor_decrease", SCENARIOS) == [
        "baseline",
        "donor_decrease",
    ]
