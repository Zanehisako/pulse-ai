from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from .operational_seed import load_operational_simulation_config

router = APIRouter(prefix="/api/studio", tags=["simulation-lab"])

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "operational_simulation.json"


class HospitalCapacityInput(BaseModel):
    nurses: int | None = Field(default=None, ge=1, le=50)
    lab: int | None = Field(default=None, ge=1, le=50)
    processing: int | None = Field(default=None, ge=1, le=50)


class GeoBBoxInput(BaseModel):
    north: float | None = Field(default=None, ge=-90, le=90)
    south: float | None = Field(default=None, ge=-90, le=90)
    east: float | None = Field(default=None, ge=-180, le=180)
    west: float | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def check_bounds(self) -> "GeoBBoxInput":
        if self.north is not None and self.south is not None and self.north < self.south:
            raise ValueError("north must be greater than or equal to south")
        if self.east is not None and self.west is not None and self.east < self.west:
            raise ValueError("east must be greater than or equal to west")
        return self


class ComponentSplitInput(BaseModel):
    RBC: float | None = Field(default=None, ge=0, le=1)
    PLATELETS: float | None = Field(default=None, ge=0, le=1)
    PLASMA: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def check_sum(self) -> "ComponentSplitInput":
        values = [self.RBC, self.PLATELETS, self.PLASMA]
        if all(v is not None for v in values):
            total = sum(values)
            if not (0.99 <= total <= 1.01):
                raise ValueError("component split must sum to 1.0")
        return self


class HospitalHoursInput(BaseModel):
    open: float | None = Field(default=None, ge=0, le=24)
    close: float | None = Field(default=None, ge=0, le=24)

    @model_validator(mode="after")
    def check_order(self) -> "HospitalHoursInput":
        if self.open is not None and self.close is not None and self.close < self.open:
            raise ValueError("close must be greater than or equal to open")
        return self


class OperationalSimulationConfigUpdate(BaseModel):
    enabled: bool | None = None
    max_hospitals: int | None = Field(default=None, ge=1, le=100)
    min_hospitals: int | None = Field(default=None, ge=0, le=100)
    include_blood_banks: bool | None = None
    include_mobile_from_strategy: bool | None = None
    hospital_capacity: HospitalCapacityInput | None = None
    hospital_hours: HospitalHoursInput | None = None
    geo_bbox: GeoBBoxInput | None = None
    component_split: ComponentSplitInput | None = None
    fallback_to_default_centers: bool | None = None
    demand_weight_from_usage: bool | None = None
    usage_demand_weight_floor: float | None = Field(default=None, ge=0.1, le=5.0)
    usage_demand_weight_cap: float | None = Field(default=None, ge=0.1, le=5.0)
    use_django_centers: bool | None = None
    max_django_centers: int | None = Field(default=None, ge=1, le=500)

    @model_validator(mode="after")
    def check_hospital_bounds(self) -> "OperationalSimulationConfigUpdate":
        if self.max_hospitals is not None and self.min_hospitals is not None:
            if self.max_hospitals < self.min_hospitals:
                raise ValueError("max_hospitals must be greater than or equal to min_hospitals")
        return self

    @field_validator("usage_demand_weight_floor", "usage_demand_weight_cap")
    @classmethod
    def check_floor_cap(cls, value: float | None, info: Any) -> float | None:
        return value

    @model_validator(mode="after")
    def check_floor_leq_cap(self) -> "OperationalSimulationConfigUpdate":
        if self.usage_demand_weight_floor is not None and self.usage_demand_weight_cap is not None:
            if self.usage_demand_weight_floor > self.usage_demand_weight_cap:
                raise ValueError("usage_demand_weight_floor must be <= usage_demand_weight_cap")
        return self


def _load_config() -> dict[str, Any]:
    return load_operational_simulation_config()


def _save_config(payload: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@router.get("/config")
def get_operational_config() -> dict[str, Any]:
    return _load_config()


@router.put("/config")
def update_operational_config(payload: OperationalSimulationConfigUpdate) -> dict[str, Any]:
    current = _load_config()
    updates = payload.model_dump(exclude_none=True)

    for key in ("hospital_capacity", "geo_bbox", "component_split", "hospital_hours"):
        if key in updates:
            nested = updates.pop(key)
            current.setdefault(key, {})
            if key == "hospital_hours":
                hours = current.get("hospital_hours") or [0, 24]
                if nested.get("open") is not None:
                    hours[0] = nested["open"]
                if nested.get("close") is not None:
                    hours[1] = nested["close"]
                current[key] = hours
            else:
                for sub_key, sub_value in nested.items():
                    if sub_value is not None:
                        current[key][sub_key] = sub_value

    for key, value in updates.items():
        current[key] = value

    _save_config(current)
    return current
