"""
calibration.py - Real-world baseline assumptions for the Quebec City simulator.

The simulator remains a stylised operations model, but these constants anchor the
default demand and supply levels in public Hema-Quebec and Statistics Canada data.
"""

from __future__ import annotations

QUEBEC_PROVINCE_POPULATION_2021 = 8_501_833
QUEBEC_CMA_POPULATION_2021 = 839_311
QUEBEC_CITY_POPULATION_SHARE = (
    QUEBEC_CMA_POPULATION_2021 / QUEBEC_PROVINCE_POPULATION_2021
)

# Hema-Quebec, annual report 2024-2025.
# We treat this as completed donor events, not raw attempted arrivals.
HEMA_ANNUAL_COMPLETED_DONOR_EVENTS_2024_2025 = 146_735
HEMA_ANNUAL_LABILE_PRODUCTS_2024_2025 = 309_113

# Hema-Quebec hospital distribution page.
HEMA_DAILY_HOSPITAL_ORDERS = 200

# Approximate baseline yield from attempted arrivals to completed donations in the
# simulator after show-up and eligibility filtering.
QC_BASELINE_COMPLETION_YIELD = 0.66

QC_DAILY_COMPLETED_DONATIONS_EST = (
    HEMA_ANNUAL_COMPLETED_DONOR_EVENTS_2024_2025 / 365.0
) * QUEBEC_CITY_POPULATION_SHARE
QC_DAILY_DONOR_ATTEMPTS_EST = (
    QC_DAILY_COMPLETED_DONATIONS_EST / QC_BASELINE_COMPLETION_YIELD
)
QC_DAILY_LABILE_PRODUCTS_EST = (
    HEMA_ANNUAL_LABILE_PRODUCTS_2024_2025 / 365.0
) * QUEBEC_CITY_POPULATION_SHARE
QC_DAILY_HOSPITAL_ORDERS_EST = (
    HEMA_DAILY_HOSPITAL_ORDERS * QUEBEC_CITY_POPULATION_SHARE
)
QC_AVG_UNITS_PER_ORDER_EST = (
    QC_DAILY_LABILE_PRODUCTS_EST / QC_DAILY_HOSPITAL_ORDERS_EST
)

# Approximate citywide collection window; used to convert scaled donor activity
# into a city-level inter-arrival time.
# Donor generator active window: 7:00–21:00 = 14 h.  Previous value of 13.0
# under-counted the generation window, inflating the arrival rate ~7.7%.
QC_EFFECTIVE_COLLECTION_HOURS = 14.0
QC_BASE_DONOR_INTERARRIVAL_H = (
    QC_EFFECTIVE_COLLECTION_HOURS / QC_DAILY_DONOR_ATTEMPTS_EST
)
QC_BASE_DEMAND_INTERARRIVAL_H = 24.0 / QC_DAILY_HOSPITAL_ORDERS_EST

QC_BASE_BUFFER_DAYS = 4.0

# Quebec City demand mix proxy for labile products.
COMPONENT_DEMAND_WEIGHTS = {"RBC": 0.58, "PLATELETS": 0.27, "PLASMA": 0.15}
COMPONENT_COLLECTION_WEIGHTS = {"RBC": 0.50, "PLATELETS": 0.20, "PLASMA": 0.30}

REAL_WORLD_NOTES = [
    (
        "Quebec CMA population share: "
        f"{QUEBEC_CMA_POPULATION_2021:,} / {QUEBEC_PROVINCE_POPULATION_2021:,} "
        f"= {QUEBEC_CITY_POPULATION_SHARE:.2%}"
    ),
    (
        "Scaled blood-product demand: "
        f"{QC_DAILY_LABILE_PRODUCTS_EST:.1f} labile products/day"
    ),
    (
        "Scaled hospital-order cadence: "
        f"{QC_DAILY_HOSPITAL_ORDERS_EST:.1f} orders/day "
        f"(~1 every {QC_BASE_DEMAND_INTERARRIVAL_H:.2f} h)"
    ),
    (
        "Scaled completed donations target: "
        f"{QC_DAILY_COMPLETED_DONATIONS_EST:.1f} successful donations/day"
    ),
    (
        "Simulated donor arrivals needed to hit that target: "
        f"{QC_DAILY_DONOR_ATTEMPTS_EST:.1f} arrival attempts/day "
        f"(~1 every {QC_BASE_DONOR_INTERARRIVAL_H:.2f} h during collection hours)"
    ),
]
