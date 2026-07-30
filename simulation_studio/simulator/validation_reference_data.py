"""
validation_reference_data.py — Ground-truth reference data for validating the
Quebec City blood supply chain simulation against real-world Héma-Québec benchmarks.

This module provides structured, citable reference values drawn from:
    • Héma-Québec annual reports (2024-2025)
    • Canadian Blood Services (CBS) annual reports (2022-2023)
    • WHO Global Status Report on Blood Safety (2021)
    • Statistics Canada Census (2021)
    • Peer-reviewed operations-research literature

Every reference entry carries its source citation, applicable year, unit, and
an expected value or plausible range.  The data is organised by category so
that simulation outputs can be compared systematically during validation.

Usage
-----
>>> from simulator.validation_reference_data import QUEBEC_REFERENCE_DATA
>>> collection = get_references_by_category("collection_donation")
>>> for ref in collection:
...     print(ref.metric, ref.expected_low, ref.expected_high, ref.unit)

Master's thesis validation reference.
Last updated: 2025-07
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# §1  Data-class for a single real-world reference entry
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RealWorldReference:
    """A single ground-truth data point for simulation validation.

    Parameters
    ----------
    metric : str
        Human-readable name of the metric (e.g. "Annual provincial completed
        donations").
    expected_value : float or None
        Point estimate when a single expected value is available.  Set to
        ``None`` when only a range is meaningful.
    expected_low : float
        Lower bound of the plausible real-world range (inclusive).
    expected_high : float
        Upper bound of the plausible real-world range (inclusive).
    unit : str
        Physical or logical unit (e.g. "donations/year", "%", "minutes").
    source : str
        Short citation key — look up the full reference in
        :data:`LITERATURE_CITATIONS`.
    category : str
        Grouping key that matches a top-level key in
        :data:`QUEBEC_REFERENCE_DATA`.
    year : str
        Reporting year or range (e.g. "2024-2025", "2022").
    notes : str
        Free-text annotation on derivation, caveats, or applicability.
    confidence : str
        Subjective confidence in the reference value: "high", "medium", or
        "low".
    """

    metric: str
    expected_value: Optional[float]
    expected_low: float
    expected_high: float
    unit: str
    source: str
    category: str
    year: str
    notes: str = ""
    confidence: str = "medium"

    # -- Convenience helpers -------------------------------------------------

    @property
    def midpoint(self) -> float:
        """Arithmetic mean of the range bounds."""
        return (self.expected_low + self.expected_high) / 2.0

    @property
    def range_str(self) -> str:
        """Human-readable range string."""
        if self.expected_low == self.expected_high:
            return f"{self.expected_low:,.4g} {self.unit}"
        return f"{self.expected_low:,.4g}–{self.expected_high:,.4g} {self.unit}"

    def contains(self, value: float) -> bool:
        """Return ``True`` if *value* lies within [expected_low, expected_high]."""
        return self.expected_low <= value <= self.expected_high


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Comprehensive reference data organised by category
# ═══════════════════════════════════════════════════════════════════════════════
#
# Key derivations (documented here for reproducibility):
#
#   QUEBEC_CMA_POPULATION_2021     = 839_311
#   QUEBEC_PROVINCE_POPULATION_2021 = 8_501_833
#   QUEBEC_CITY_POPULATION_SHARE   = 839_311 / 8_501_833  ≈ 0.09872  (~9.87 %)
#
#   QC CMA daily completed donations ≈ 146_735 / 365 × 0.09872 ≈ 39.68
#   QC CMA daily labile products     ≈ 309_113 / 365 × 0.09872 ≈ 83.59
#   QC CMA daily hospital orders     ≈ 200     × 0.09872        ≈ 19.74
#   Average products per donation    ≈ 309_113 / 146_735         ≈ 2.107
# ═══════════════════════════════════════════════════════════════════════════════

_POP_SHARE = 839_311 / 8_501_833  # ≈ 0.09872

QUEBEC_REFERENCE_DATA: Dict[str, List[RealWorldReference]] = {
    # -----------------------------------------------------------------------
    # (a) Collection & Donation
    # -----------------------------------------------------------------------
    "collection_donation": [
        RealWorldReference(
            metric="Annual provincial completed donations",
            expected_value=146_735.0,
            expected_low=146_735.0,
            expected_high=146_735.0,
            unit="donations/year",
            source="hema_quebec_2024",
            category="collection_donation",
            year="2024-2025",
            notes="Héma-Québec completed donor events for the fiscal year.",
            confidence="high",
        ),
        RealWorldReference(
            metric="Annual provincial labile products distributed",
            expected_value=309_113.0,
            expected_low=309_113.0,
            expected_high=309_113.0,
            unit="products/year",
            source="hema_quebec_2024",
            category="collection_donation",
            year="2024-2025",
            notes="Total labile blood products distributed province-wide.",
            confidence="high",
        ),
        RealWorldReference(
            metric="Daily provincial hospital orders",
            expected_value=200.0,
            expected_low=180.0,
            expected_high=220.0,
            unit="orders/day",
            source="hema_quebec_2024",
            category="collection_donation",
            year="2024-2025",
            notes=(
                "Héma-Québec hospital distribution page states ~200 orders/day; "
                "range accounts for day-of-week variation."
            ),
            confidence="high",
        ),
        RealWorldReference(
            metric="Quebec City CMA daily completed donations",
            expected_value=round(146_735 / 365.0 * _POP_SHARE, 1),
            expected_low=round(146_735 / 365.0 * _POP_SHARE * 0.85, 1),
            expected_high=round(146_735 / 365.0 * _POP_SHARE * 1.15, 1),
            unit="donations/day",
            source="hema_quebec_2024",
            category="collection_donation",
            year="2024-2025",
            notes=(
                "Derived: 146 735 / 365 × 0.0987 ≈ 39.7.  ±15 % accounts for "
                "regional collection-site density variation."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="Quebec City CMA daily labile products",
            expected_value=round(309_113 / 365.0 * _POP_SHARE, 1),
            expected_low=round(309_113 / 365.0 * _POP_SHARE * 0.85, 1),
            expected_high=round(309_113 / 365.0 * _POP_SHARE * 1.15, 1),
            unit="products/day",
            source="hema_quebec_2024",
            category="collection_donation",
            year="2024-2025",
            notes="Derived: 309 113 / 365 × 0.0987 ≈ 83.6.",
            confidence="medium",
        ),
        RealWorldReference(
            metric="Quebec City CMA daily hospital orders",
            expected_value=round(200.0 * _POP_SHARE, 1),
            expected_low=round(200.0 * _POP_SHARE * 0.85, 1),
            expected_high=round(200.0 * _POP_SHARE * 1.15, 1),
            unit="orders/day",
            source="hema_quebec_2024",
            category="collection_donation",
            year="2024-2025",
            notes="Derived: 200 × 0.0987 ≈ 19.7.",
            confidence="medium",
        ),
        RealWorldReference(
            metric="National whole blood donation rate",
            expected_value=15.8,
            expected_low=14.0,
            expected_high=18.0,
            unit="donations per 1,000 population",
            source="cbs_2022",
            category="collection_donation",
            year="2022",
            notes=(
                "Canadian Blood Services reports ~15.8 per 1 000 nationally; "
                "Quebec operates its own system (Héma-Québec) so the rate may "
                "differ slightly."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="Donor completion yield (show-up × eligibility)",
            expected_value=0.66,
            expected_low=0.60,
            expected_high=0.72,
            unit="fraction",
            source="hema_quebec_2024",
            category="collection_donation",
            year="2024-2025",
            notes=(
                "Fraction of attempted donor arrivals that result in a completed "
                "donation, after no-shows, deferrals, and screening failures."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="Average labile products per donation",
            expected_value=round(309_113 / 146_735, 2),
            expected_low=1.8,
            expected_high=2.5,
            unit="products/donation",
            source="hema_quebec_2024",
            category="collection_donation",
            year="2024-2025",
            notes=(
                "Derived: 309 113 / 146 735 ≈ 2.11. Whole blood typically yields "
                "RBC + plasma ± platelets (buffy coat) ≈ 2–3 components."
            ),
            confidence="high",
        ),
    ],
    # -----------------------------------------------------------------------
    # (b) Blood Type Distribution (Canadian population)
    # -----------------------------------------------------------------------
    "blood_type_distribution": [
        RealWorldReference(
            metric="O-positive prevalence",
            expected_value=39.0,
            expected_low=39.0,
            expected_high=46.0,
            unit="%",
            source="cbs_2022",
            category="blood_type_distribution",
            year="2022",
            notes="Most common type in Canada; range reflects ethnic mix.",
            confidence="high",
        ),
        RealWorldReference(
            metric="A-positive prevalence",
            expected_value=30.0,
            expected_low=27.0,
            expected_high=34.0,
            unit="%",
            source="cbs_2022",
            category="blood_type_distribution",
            year="2022",
            notes="Second most common type in Canada.",
            confidence="high",
        ),
        RealWorldReference(
            metric="B-positive prevalence",
            expected_value=9.0,
            expected_low=7.5,
            expected_high=12.0,
            unit="%",
            source="cbs_2022",
            category="blood_type_distribution",
            year="2022",
            notes="Varies with South/East Asian demographics.",
            confidence="high",
        ),
        RealWorldReference(
            metric="AB-positive prevalence",
            expected_value=3.0,
            expected_low=2.5,
            expected_high=5.0,
            unit="%",
            source="cbs_2022",
            category="blood_type_distribution",
            year="2022",
            notes="Rarest common positive type.",
            confidence="high",
        ),
        RealWorldReference(
            metric="O-negative prevalence",
            expected_value=7.0,
            expected_low=4.0,
            expected_high=9.0,
            unit="%",
            source="cbs_2022",
            category="blood_type_distribution",
            year="2022",
            notes="Universal donor for RBC; critical for emergency supply.",
            confidence="high",
        ),
        RealWorldReference(
            metric="A-negative prevalence",
            expected_value=5.0,
            expected_low=3.0,
            expected_high=7.0,
            unit="%",
            source="cbs_2022",
            category="blood_type_distribution",
            year="2022",
            notes="Relatively uncommon Rh-negative type.",
            confidence="high",
        ),
        RealWorldReference(
            metric="B-negative prevalence",
            expected_value=1.5,
            expected_low=1.0,
            expected_high=2.0,
            unit="%",
            source="cbs_2022",
            category="blood_type_distribution",
            year="2022",
            notes="Rare type; donors actively recruited.",
            confidence="high",
        ),
        RealWorldReference(
            metric="AB-negative prevalence",
            expected_value=0.7,
            expected_low=0.5,
            expected_high=1.0,
            unit="%",
            source="cbs_2022",
            category="blood_type_distribution",
            year="2022",
            notes="Rarest ABO/Rh combination in Canadian population.",
            confidence="high",
        ),
    ],
    # -----------------------------------------------------------------------
    # (c) Donor Deferral & Rejection
    # -----------------------------------------------------------------------
    "deferral_rejection": [
        RealWorldReference(
            metric="Overall donor deferral rate",
            expected_value=12.0,
            expected_low=7.0,
            expected_high=15.0,
            unit="%",
            source="cbs_2022",
            category="deferral_rejection",
            year="2022",
            notes=(
                "CBS and WHO benchmark.  Includes temporary and permanent "
                "deferrals at screening."
            ),
            confidence="high",
        ),
        RealWorldReference(
            metric="Low hemoglobin deferral rate",
            expected_value=5.0,
            expected_low=3.0,
            expected_high=8.0,
            unit="%",
            source="who_2021",
            category="deferral_rejection",
            year="2021",
            notes=(
                "Higher in female donors (up to 10 %); single largest cause of "
                "deferral in high-income countries."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="TTI-positive rejection rate",
            expected_value=0.2,
            expected_low=0.1,
            expected_high=0.5,
            unit="%",
            source="who_2021",
            category="deferral_rejection",
            year="2021",
            notes=(
                "Transfusion-transmissible infection (HIV, HBV, HCV, syphilis) "
                "confirmed-positive rate in repeat donors in high-income settings."
            ),
            confidence="high",
        ),
        RealWorldReference(
            metric="Age-based ineligibility rate",
            expected_value=2.0,
            expected_low=1.0,
            expected_high=3.0,
            unit="%",
            source="hema_quebec_2024",
            category="deferral_rejection",
            year="2024-2025",
            notes=(
                "Donors outside 18–70 age window (Héma-Québec) or those with "
                "age-related medical conditions."
            ),
            confidence="low",
        ),
        RealWorldReference(
            metric="Medication deferral rate",
            expected_value=2.0,
            expected_low=1.0,
            expected_high=3.0,
            unit="%",
            source="aabb_2020",
            category="deferral_rejection",
            year="2020",
            notes="Antibiotics, anticoagulants, isotretinoin, etc.",
            confidence="medium",
        ),
        RealWorldReference(
            metric="Recent tattoo/piercing deferral rate",
            expected_value=1.0,
            expected_low=0.5,
            expected_high=2.0,
            unit="%",
            source="hema_quebec_2024",
            category="deferral_rejection",
            year="2024-2025",
            notes=(
                "Québec reduced deferral period from 6 to 3 months in recent "
                "years; rate trending downward."
            ),
            confidence="low",
        ),
    ],
    # -----------------------------------------------------------------------
    # (d) Wastage & Expiry
    # -----------------------------------------------------------------------
    "wastage_expiry": [
        RealWorldReference(
            metric="RBC outdating rate",
            expected_value=2.0,
            expected_low=1.0,
            expected_high=5.0,
            unit="% of collected",
            source="cbs_2022",
            category="wastage_expiry",
            year="2022",
            notes=(
                "CBS target is < 2 %; actual rates may be higher in smaller "
                "hospitals with lower turnover."
            ),
            confidence="high",
        ),
        RealWorldReference(
            metric="Platelet outdating rate",
            expected_value=10.0,
            expected_low=5.0,
            expected_high=20.0,
            unit="% of collected",
            source="cbs_2022",
            category="wastage_expiry",
            year="2022",
            notes=(
                "5-day shelf life drives high wastage; apheresis platelets have "
                "slightly better utilisation than pooled buffy-coat."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="Plasma wastage rate",
            expected_value=0.5,
            expected_low=0.0,
            expected_high=1.0,
            unit="% of collected",
            source="cbs_2022",
            category="wastage_expiry",
            year="2022",
            notes="Frozen plasma has a 1-year shelf life; wastage is minimal.",
            confidence="high",
        ),
        RealWorldReference(
            metric="Overall blood product wastage rate",
            expected_value=4.0,
            expected_low=2.0,
            expected_high=8.0,
            unit="%",
            source="beliën_2012",
            category="wastage_expiry",
            year="2012",
            notes=(
                "Weighted average across all component types.  Literature range "
                "is 2–8 % for well-managed systems."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="RBC shelf life utilisation (average age at transfusion)",
            expected_value=17.0,
            expected_low=14.0,
            expected_high=21.0,
            unit="days",
            source="pierskalla_2005",
            category="wastage_expiry",
            year="2005",
            notes=(
                "Mean age of RBC units at time of transfusion; 42-day max shelf "
                "life. FIFO policies push average toward 14–18 days."
            ),
            confidence="medium",
        ),
    ],
    # -----------------------------------------------------------------------
    # (e) Service Levels & Shortage
    # -----------------------------------------------------------------------
    "service_levels": [
        RealWorldReference(
            metric="Overall order fulfillment rate",
            expected_value=97.0,
            expected_low=95.0,
            expected_high=99.5,
            unit="%",
            source="hema_quebec_2024",
            category="service_levels",
            year="2024-2025",
            notes=(
                "Target fill rate for hospital orders.  Héma-Québec targets "
                "> 99 % for routine orders."
            ),
            confidence="high",
        ),
        RealWorldReference(
            metric="Acceptable shortage rate",
            expected_value=2.0,
            expected_low=0.5,
            expected_high=5.0,
            unit="%",
            source="who_2021",
            category="service_levels",
            year="2021",
            notes=(
                "Complement of fulfillment rate.  Chronic shortages above 5 % "
                "trigger national-level alerts."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="Critical shortage events per year (national)",
            expected_value=4.0,
            expected_low=2.0,
            expected_high=6.0,
            unit="events/year",
            source="cbs_2022",
            category="service_levels",
            year="2022",
            notes=(
                "National-level inventory alerts where O-negative RBC drops "
                "below 2 days of supply."
            ),
            confidence="low",
        ),
        RealWorldReference(
            metric="O-negative emergency release share of all RBC issues",
            expected_value=7.0,
            expected_low=5.0,
            expected_high=10.0,
            unit="%",
            source="aabb_2020",
            category="service_levels",
            year="2020",
            notes=(
                "Uncrossmatched O-neg issued for massive haemorrhage protocol; "
                "varies by trauma centre volume."
            ),
            confidence="medium",
        ),
    ],
    # -----------------------------------------------------------------------
    # (f) Inventory Levels
    # -----------------------------------------------------------------------
    "inventory_levels": [
        RealWorldReference(
            metric="National target RBC days of supply",
            expected_value=5.0,
            expected_low=3.0,
            expected_high=7.0,
            unit="days",
            source="cbs_2022",
            category="inventory_levels",
            year="2022",
            notes="CBS and Héma-Québec aim for ≥ 5 days for routine planning.",
            confidence="high",
        ),
        RealWorldReference(
            metric="Platelet days of supply",
            expected_value=2.0,
            expected_low=1.0,
            expected_high=3.0,
            unit="days",
            source="cbs_2022",
            category="inventory_levels",
            year="2022",
            notes="Short 5-day shelf life constrains buffer stock.",
            confidence="high",
        ),
        RealWorldReference(
            metric="Plasma days of supply",
            expected_value=10.0,
            expected_low=5.0,
            expected_high=14.0,
            unit="days",
            source="cbs_2022",
            category="inventory_levels",
            year="2022",
            notes="Frozen plasma (FFP) with 1-year shelf life; larger buffers feasible.",
            confidence="medium",
        ),
        RealWorldReference(
            metric="Optimal operational buffer (Héma-Québec target)",
            expected_value=4.0,
            expected_low=3.0,
            expected_high=5.0,
            unit="days",
            source="hema_quebec_2024",
            category="inventory_levels",
            year="2024-2025",
            notes=(
                "Héma-Québec internal target for labile product inventory at "
                "distribution centres."
            ),
            confidence="high",
        ),
    ],
    # -----------------------------------------------------------------------
    # (g) Processing Times
    # -----------------------------------------------------------------------
    "processing_times": [
        RealWorldReference(
            metric="Registration and screening time",
            expected_value=15.0,
            expected_low=10.0,
            expected_high=25.0,
            unit="minutes",
            source="aabb_2020",
            category="processing_times",
            year="2020",
            notes=(
                "Includes identity verification, health questionnaire, vital "
                "signs, and hemoglobin check."
            ),
            confidence="high",
        ),
        RealWorldReference(
            metric="Whole blood collection time",
            expected_value=12.0,
            expected_low=10.0,
            expected_high=15.0,
            unit="minutes",
            source="aabb_2020",
            category="processing_times",
            year="2020",
            notes="Phlebotomy for a standard 450 mL whole-blood unit.",
            confidence="high",
        ),
        RealWorldReference(
            metric="Lab testing (infectious disease markers)",
            expected_value=2.0,
            expected_low=1.0,
            expected_high=3.0,
            unit="hours",
            source="aabb_2020",
            category="processing_times",
            year="2020",
            notes=(
                "NAT + serological testing for HIV, HBV, HCV, syphilis, "
                "and other markers.  Batched processing at central lab."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="Component processing (separation and labelling)",
            expected_value=3.0,
            expected_low=2.0,
            expected_high=4.0,
            unit="hours",
            source="aabb_2020",
            category="processing_times",
            year="2020",
            notes=(
                "Centrifugation, plasma extraction, buffy-coat pooling, "
                "leukoreduction, and final labelling."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="Quarantine and release time",
            expected_value=48.0,
            expected_low=24.0,
            expected_high=72.0,
            unit="hours",
            source="hema_quebec_2024",
            category="processing_times",
            year="2024-2025",
            notes=(
                "Units held until all test results are confirmed negative "
                "and quality checks pass."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="Total collection-to-release turnaround",
            expected_value=48.0,
            expected_low=24.0,
            expected_high=72.0,
            unit="hours",
            source="hema_quebec_2024",
            category="processing_times",
            year="2024-2025",
            notes=(
                "End-to-end time from phlebotomy to product available for "
                "distribution.  Typically 24–72 h depending on test batching."
            ),
            confidence="medium",
        ),
    ],
    # -----------------------------------------------------------------------
    # (h) Transport & Logistics
    # -----------------------------------------------------------------------
    "transport_logistics": [
        RealWorldReference(
            metric="Urban courier delivery time (Quebec City)",
            expected_value=30.0,
            expected_low=15.0,
            expected_high=45.0,
            unit="minutes",
            source="hema_quebec_2024",
            category="transport_logistics",
            year="2024-2025",
            notes=(
                "Routine delivery from Héma-Québec distribution centre to "
                "hospitals in the Quebec City CMA."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="Emergency delivery time",
            expected_value=20.0,
            expected_low=10.0,
            expected_high=30.0,
            unit="minutes",
            source="hema_quebec_2024",
            category="transport_logistics",
            year="2024-2025",
            notes="Stat / Code-Red deliveries; priority dispatch.",
            confidence="medium",
        ),
        RealWorldReference(
            metric="Average urban travel speed",
            expected_value=40.0,
            expected_low=30.0,
            expected_high=50.0,
            unit="km/h",
            source="stats_canada_2021",
            category="transport_logistics",
            year="2021",
            notes="Typical urban arterial speed in Quebec City.",
            confidence="medium",
        ),
        RealWorldReference(
            metric="Weather impact factor — snow",
            expected_value=0.70,
            expected_low=0.60,
            expected_high=0.80,
            unit="speed multiplier",
            source="beliën_2012",
            category="transport_logistics",
            year="2012",
            notes="Multiplicative factor applied to baseline travel speed during snow.",
            confidence="low",
        ),
        RealWorldReference(
            metric="Weather impact factor — ice storm",
            expected_value=0.45,
            expected_low=0.40,
            expected_high=0.50,
            unit="speed multiplier",
            source="beliën_2012",
            category="transport_logistics",
            year="2012",
            notes="Severe icing; roads partially impassable.",
            confidence="low",
        ),
        RealWorldReference(
            metric="Weather impact factor — blizzard",
            expected_value=0.25,
            expected_low=0.20,
            expected_high=0.30,
            unit="speed multiplier",
            source="beliën_2012",
            category="transport_logistics",
            year="2012",
            notes="Near whiteout conditions; delivery may be suspended.",
            confidence="low",
        ),
    ],
    # -----------------------------------------------------------------------
    # (i) Seasonal Patterns
    # -----------------------------------------------------------------------
    "seasonal_patterns": [
        RealWorldReference(
            metric="Summer collection drop (June–August)",
            expected_value=15.0,
            expected_low=10.0,
            expected_high=20.0,
            unit="% below baseline",
            source="hema_quebec_2024",
            category="seasonal_patterns",
            year="2024-2025",
            notes=(
                "Vacation season reduces donor availability; Héma-Québec runs "
                "targeted summer campaigns."
            ),
            confidence="high",
        ),
        RealWorldReference(
            metric="Holiday period collection drop (December–January)",
            expected_value=20.0,
            expected_low=15.0,
            expected_high=25.0,
            unit="% below baseline",
            source="hema_quebec_2024",
            category="seasonal_patterns",
            year="2024-2025",
            notes="Christmas / New Year period; compounded by winter weather.",
            confidence="high",
        ),
        RealWorldReference(
            metric="Flu season donor deferral increase (January–March)",
            expected_value=30.0,
            expected_low=20.0,
            expected_high=40.0,
            unit="% increase in deferral",
            source="who_2021",
            category="seasonal_patterns",
            year="2021",
            notes=(
                "Respiratory illness (influenza, RSV, COVID) temporarily defers "
                "symptomatic donors."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="Quebec winter snow event frequency",
            expected_value=25.0,
            expected_low=20.0,
            expected_high=30.0,
            unit="% of winter days",
            source="stats_canada_2021",
            category="seasonal_patterns",
            year="2021",
            notes="Approximate fraction of December–March days with snowfall.",
            confidence="medium",
        ),
        RealWorldReference(
            metric="Quebec winter ice event frequency",
            expected_value=10.0,
            expected_low=5.0,
            expected_high=15.0,
            unit="% of winter days",
            source="stats_canada_2021",
            category="seasonal_patterns",
            year="2021",
            notes="Freezing rain / ice storm days during winter months.",
            confidence="low",
        ),
        RealWorldReference(
            metric="Quebec winter blizzard event frequency",
            expected_value=5.0,
            expected_low=2.0,
            expected_high=8.0,
            unit="% of winter days",
            source="stats_canada_2021",
            category="seasonal_patterns",
            year="2021",
            notes="Major blizzard events with > 25 cm snowfall or whiteout.",
            confidence="low",
        ),
    ],
    # -----------------------------------------------------------------------
    # (j) Component Demand Mix
    # -----------------------------------------------------------------------
    "component_demand_mix": [
        RealWorldReference(
            metric="RBC share of hospital demand",
            expected_value=58.0,
            expected_low=55.0,
            expected_high=65.0,
            unit="%",
            source="cbs_2022",
            category="component_demand_mix",
            year="2022",
            notes="Red blood cells dominate transfusion medicine demand.",
            confidence="high",
        ),
        RealWorldReference(
            metric="Platelet share of hospital demand",
            expected_value=27.0,
            expected_low=20.0,
            expected_high=30.0,
            unit="%",
            source="cbs_2022",
            category="component_demand_mix",
            year="2022",
            notes=(
                "Growing demand driven by oncology / haematology services; "
                "proportion increasing over time."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="Plasma share of hospital demand",
            expected_value=15.0,
            expected_low=10.0,
            expected_high=20.0,
            unit="%",
            source="cbs_2022",
            category="component_demand_mix",
            year="2022",
            notes=(
                "Fresh frozen plasma (FFP) and cryoprecipitate; some demand "
                "met by fractionated plasma products."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="Crossmatch-to-transfusion ratio",
            expected_value=1.8,
            expected_low=1.5,
            expected_high=2.5,
            unit="ratio",
            source="aabb_2020",
            category="component_demand_mix",
            year="2020",
            notes=(
                "Ratio of units crossmatched to units actually transfused.  "
                "Best-practice target is < 2.0; higher ratios indicate "
                "over-ordering."
            ),
            confidence="medium",
        ),
    ],
    # -----------------------------------------------------------------------
    # (k) Donor Demographics (Quebec / Canada)
    # -----------------------------------------------------------------------
    "donor_demographics": [
        RealWorldReference(
            metric="Mean donor age",
            expected_value=40.0,
            expected_low=35.0,
            expected_high=45.0,
            unit="years",
            source="hema_quebec_2024",
            category="donor_demographics",
            year="2024-2025",
            notes="Héma-Québec donor pool skews slightly older than CBS nationally.",
            confidence="medium",
        ),
        RealWorldReference(
            metric="Female donor proportion",
            expected_value=46.0,
            expected_low=42.0,
            expected_high=50.0,
            unit="%",
            source="hema_quebec_2024",
            category="donor_demographics",
            year="2024-2025",
            notes=(
                "Roughly balanced; females have higher deferral rates due to "
                "lower hemoglobin thresholds."
            ),
            confidence="high",
        ),
        RealWorldReference(
            metric="Repeat donor rate",
            expected_value=70.0,
            expected_low=65.0,
            expected_high=75.0,
            unit="%",
            source="hema_quebec_2024",
            category="donor_demographics",
            year="2024-2025",
            notes=(
                "Fraction of donations from donors with ≥ 2 lifetime donations. "
                "Repeat donors have lower deferral and TTI rates."
            ),
            confidence="medium",
        ),
        RealWorldReference(
            metric="Mean hemoglobin — male donors",
            expected_value=15.0,
            expected_low=14.0,
            expected_high=15.5,
            unit="g/dL",
            source="aabb_2020",
            category="donor_demographics",
            year="2020",
            notes="Eligibility threshold typically 13.0 g/dL for males.",
            confidence="high",
        ),
        RealWorldReference(
            metric="Mean hemoglobin — female donors",
            expected_value=13.2,
            expected_low=12.5,
            expected_high=14.0,
            unit="g/dL",
            source="aabb_2020",
            category="donor_demographics",
            year="2020",
            notes="Eligibility threshold typically 12.5 g/dL for females.",
            confidence="high",
        ),
        RealWorldReference(
            metric="Mean weight — male donors",
            expected_value=80.0,
            expected_low=75.0,
            expected_high=85.0,
            unit="kg",
            source="stats_canada_2021",
            category="donor_demographics",
            year="2021",
            notes="Canadian adult male average; minimum 50 kg for donation.",
            confidence="medium",
        ),
        RealWorldReference(
            metric="Mean weight — female donors",
            expected_value=66.0,
            expected_low=60.0,
            expected_high=72.0,
            unit="kg",
            source="stats_canada_2021",
            category="donor_demographics",
            year="2021",
            notes="Canadian adult female average; minimum 50 kg for donation.",
            confidence="medium",
        ),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Flat list of all reference entries
# ═══════════════════════════════════════════════════════════════════════════════

ALL_REFERENCES: List[RealWorldReference] = [
    ref for category_refs in QUEBEC_REFERENCE_DATA.values() for ref in category_refs
]
"""Flat list of every :class:`RealWorldReference` entry for convenient iteration."""


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Lookup helper
# ═══════════════════════════════════════════════════════════════════════════════


def get_references_by_category(category: str) -> List[RealWorldReference]:
    """Return all reference entries for the given *category* key.

    Parameters
    ----------
    category : str
        A key from :data:`QUEBEC_REFERENCE_DATA`, e.g.
        ``"collection_donation"``, ``"blood_type_distribution"``.

    Returns
    -------
    list[RealWorldReference]
        Matching entries, or an empty list if the category is not found.

    Examples
    --------
    >>> refs = get_references_by_category("wastage_expiry")
    >>> len(refs) >= 4
    True
    """
    return list(QUEBEC_REFERENCE_DATA.get(category, []))


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Validation thresholds for automated pass/fail checks
# ═══════════════════════════════════════════════════════════════════════════════

VALIDATION_THRESHOLDS: Dict[str, Dict[str, object]] = {
    "blood_type_distribution": {
        "method": "chi_squared",
        "p_value_min": 0.05,
        "description": (
            "Chi-squared goodness-of-fit test against expected Canadian ABO/Rh "
            "frequencies.  Simulated distribution must not be statistically "
            "different at α = 0.05."
        ),
    },
    "daily_donations": {
        "method": "relative_deviation",
        "tolerance_pct": 20.0,
        "reference_value": round(146_735 / 365.0 * _POP_SHARE, 1),
        "description": (
            "Simulated daily completed donations in the Quebec City CMA must "
            "be within ±20 % of the derived estimate (~39.7)."
        ),
    },
    "daily_demand": {
        "method": "relative_deviation",
        "tolerance_pct": 20.0,
        "reference_value": round(200.0 * _POP_SHARE, 1),
        "description": (
            "Simulated daily hospital orders must be within ±20 % of the "
            "derived estimate (~19.7)."
        ),
    },
    "deferral_rate": {
        "method": "absolute_deviation",
        "tolerance_pp": 5.0,
        "reference_low": 7.0,
        "reference_high": 15.0,
        "description": (
            "Simulated overall deferral rate must fall within 7–15 % ± 5 "
            "percentage points (i.e. 2–20 %)."
        ),
    },
    "wastage_rate": {
        "method": "range_check",
        "reference_low": 2.0,
        "reference_high": 8.0,
        "description": (
            "Simulated overall wastage rate must fall within the literature "
            "range of 2–8 %."
        ),
    },
    "shortage_rate": {
        "method": "range_check",
        "reference_low": 0.5,
        "reference_high": 5.0,
        "description": (
            "Simulated shortage rate must fall within the acceptable range of 0.5–5 %."
        ),
    },
    "service_rate": {
        "method": "range_check",
        "reference_low": 95.0,
        "reference_high": 99.5,
        "description": ("Simulated order fulfillment rate must fall within 95–99.5 %."),
    },
}
"""Automated pass/fail thresholds keyed by metric name.

Each entry specifies:
    method          — the comparison strategy
    tolerance_pct   — relative tolerance in percent (for ``relative_deviation``)
    tolerance_pp    — absolute tolerance in percentage points (for
                      ``absolute_deviation``)
    reference_low/high — acceptable range bounds (for ``range_check``)
    p_value_min     — minimum p-value for statistical tests (for ``chi_squared``)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Literature citations for thesis bibliography
# ═══════════════════════════════════════════════════════════════════════════════

LITERATURE_CITATIONS: Dict[str, str] = {
    "hema_quebec_2024": (
        "Héma-Québec, Rapport annuel 2024-2025 : Au cœur de l'innovation, "
        "Héma-Québec, Québec, QC, Canada, 2025."
    ),
    "cbs_2022": (
        "Canadian Blood Services, Annual Report 2022-2023: Navigating a Complex "
        "Landscape, Canadian Blood Services, Ottawa, ON, Canada, 2023."
    ),
    "who_2021": (
        "World Health Organization, Global Status Report on Blood Safety and "
        "Availability 2021, WHO, Geneva, Switzerland, 2022."
    ),
    "beliën_2012": (
        "J. Beliën and H. Forcé, 'Supply chain management of blood products: "
        "A literature review,' European Journal of Operational Research, "
        "vol. 217, no. 1, pp. 1–16, 2012, doi: 10.1016/j.ejor.2011.05.026."
    ),
    "katsaliaki_2007": (
        "K. Katsaliaki and S. C. Brailsford, 'Using simulation to improve the "
        "blood supply chain,' Journal of the Operational Research Society, "
        "vol. 58, no. 2, pp. 219–227, 2007, doi: 10.1057/palgrave.jors.2602195."
    ),
    "stats_canada_2021": (
        "Statistics Canada, Census Profile, 2021 Census of Population, "
        "Catalogue no. 98-316-X2021001, Statistics Canada, Ottawa, ON, "
        "Canada, 2022."
    ),
    "pierskalla_2005": (
        "W. P. Pierskalla, 'Supply chain management of blood banks,' in "
        "Operations Research and Health Care: A Handbook of Methods and "
        "Applications, M. L. Brandeau, F. Sainfort, and W. P. Pierskalla, Eds., "
        "Springer, New York, NY, 2005, pp. 103–145, "
        "doi: 10.1007/1-4020-8066-2_5."
    ),
    "aabb_2020": (
        "AABB (American Association of Blood Banks), Technical Manual, "
        "20th ed., M. K. Fung, A. A. R. Grossman, C. D. Hillyer, and "
        "C. M. Westhoff, Eds., AABB Press, Bethesda, MD, 2020."
    ),
}
"""Short citation keys mapped to full bibliographic strings.

These keys are used in the ``source`` field of :class:`RealWorldReference`
entries.  The full strings are formatted for inclusion in a thesis bibliography
(IEEE / Vancouver hybrid style).
"""


# ═══════════════════════════════════════════════════════════════════════════════
# §7  Module-level summary (convenience for interactive use)
# ═══════════════════════════════════════════════════════════════════════════════

_CATEGORIES = list(QUEBEC_REFERENCE_DATA.keys())
_N_REFS = len(ALL_REFERENCES)

__all__ = [
    "RealWorldReference",
    "QUEBEC_REFERENCE_DATA",
    "ALL_REFERENCES",
    "get_references_by_category",
    "VALIDATION_THRESHOLDS",
    "LITERATURE_CITATIONS",
]

if __name__ == "__main__":
    # Quick self-test / summary when run as a script.
    print("=" * 80)
    print("  VALIDATION REFERENCE DATA — Quebec City Blood Supply Chain Simulation")
    print("=" * 80)
    print(f"\n  Categories : {len(_CATEGORIES)}")
    print(f"  Total refs : {_N_REFS}")
    print(f"  Citations  : {len(LITERATURE_CITATIONS)}")
    print(f"  Thresholds : {len(VALIDATION_THRESHOLDS)}")
    print()

    for cat_key in _CATEGORIES:
        refs = QUEBEC_REFERENCE_DATA[cat_key]
        print(f"  [{cat_key}]  ({len(refs)} entries)")
        for ref in refs:
            print(f"    • {ref.metric}: {ref.range_str}  [{ref.source}]")
    print()

    # Validate internal consistency.
    for ref in ALL_REFERENCES:
        assert ref.expected_low <= ref.expected_high, (
            f"Range inverted for '{ref.metric}': "
            f"{ref.expected_low} > {ref.expected_high}"
        )
        assert ref.source in LITERATURE_CITATIONS, (
            f"Unknown source key '{ref.source}' in '{ref.metric}'"
        )
        assert ref.category in QUEBEC_REFERENCE_DATA, (
            f"Unknown category '{ref.category}' in '{ref.metric}'"
        )

    print("  ✓ All internal consistency checks passed.")
    print()
