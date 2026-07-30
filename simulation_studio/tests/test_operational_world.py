from simulation_studio.app.operational_world import (
    advance_operational_tick,
    reset_operational_world,
    seed_operational_world,
)


def test_seed_operational_world_has_supplies_and_donors():
    world = seed_operational_world()
    assert len(world.hospitals) >= 2
    assert len(world.blood_supplies) >= 8
    assert len(world.donors) >= 10


def test_advance_operational_tick_mutates_stock():
    world = reset_operational_world(seed_operational_world())
    before = float(world.blood_supplies[0]["current_stock_units"])
    advance_operational_tick(world)
    after = float(world.blood_supplies[0]["current_stock_units"])
    assert before != after or float(world.blood_supplies[0]["usage_today"]) > 0