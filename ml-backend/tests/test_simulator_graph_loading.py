from __future__ import annotations

import sys
from pathlib import Path


SIMULATOR_DIR = Path(__file__).resolve().parents[1] / "simulator"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

import core


def test_load_city_graph_uses_offline_fallback_without_hitting_osm(monkeypatch):
    monkeypatch.setenv("PIOS_SIM_OFFLINE", "1")

    def fail(*args, **kwargs):
        raise AssertionError("offline mode should not query OpenStreetMap")

    monkeypatch.setattr(core.ox, "geocode_to_gdf", fail)
    monkeypatch.setattr(core.ox, "graph_from_place", fail)

    graph, north, south, east, west = core.load_city_graph()

    assert graph is None
    assert (north, south, east, west) == core.DEFAULT_CITY_BOUNDS
