from __future__ import annotations

import simpy

from simulation_studio.app.operational_seed import (
    build_operational_seed_plan,
    build_simulation_centers,
    prestock_centers_from_operational,
)


class _Strategy:
    extra_nurses = 0
    extra_lab_staff = 0
    extra_processing_staff = 0
    hours_extension_h = 0
    mobile_units = 0


def _sample_payload() -> dict:
    return {
        "hospitals": [
            {"hospital_id": "H-1", "name": "General Hospital", "wilaya": "Algiers"},
            {"hospital_id": "H-2", "name": "Care Hospital", "wilaya": "Oran"},
            {"hospital_id": "H-3", "name": "North Clinic", "wilaya": "Blida"},
            {"hospital_id": "H-4", "name": "South Medical Center", "wilaya": "Setif"},
        ],
        "blood_supplies": [
            {
                "hospital_id": "H-1",
                "hospital_name": "General Hospital",
                "blood_product_type": "O+",
                "current_stock_units": 12,
                "usage_today": 4,
            },
            {
                "hospital_id": "H-2",
                "hospital_name": "Care Hospital",
                "blood_product_type": "A+",
                "current_stock_units": 18,
                "usage_today": 6,
            },
            {
                "hospital_id": "H-3",
                "hospital_name": "North Clinic",
                "blood_product_type": "B+",
                "current_stock_units": 9,
                "usage_today": 2,
            },
            {
                "hospital_id": "H-4",
                "hospital_name": "South Medical Center",
                "blood_product_type": "AB+",
                "current_stock_units": 7,
                "usage_today": 3,
            },
        ],
    }


def test_build_operational_seed_plan_uses_all_hospitals():
    plan = build_operational_seed_plan(_sample_payload())
    assert plan is not None
    assert plan.enabled
    assert plan.hospital_count == 4
    hospital_configs = [row for row in plan.center_configs if row["type"] == "hospital"]
    assert len(hospital_configs) == 4
    assert hospital_configs[0]["external_id"] == "H-1"
    assert hospital_configs[0]["initial_usage_today"] == 4


def test_prestock_centers_from_operational_assigns_per_hospital_inventory():
    plan = build_operational_seed_plan(_sample_payload())
    assert plan is not None
    env = simpy.Environment()
    centers = build_simulation_centers(env, _Strategy(), center_configs=plan.center_configs)
    seeded = prestock_centers_from_operational(centers, env, plan.supplies)
    assert seeded is True

    hospitals = [center for center in centers if center.ctype == "hospital"]
    assert len(hospitals) == 4
    totals = {center.external_id: sum(center.inventory_by_component().values()) for center in hospitals}
    assert totals["H-1"] == 12
    assert totals["H-2"] == 18
    assert totals["H-3"] == 9
    assert totals["H-4"] == 7