"""
validation_benchmarks.py — Real-world reference data and benchmarks for validating
the Quebec City blood supply chain simulation.

Master's thesis validation reference.  Every benchmark includes:
    • metric name
    • expected real-world value or plausible range
    • source (Héma-Québec annual reports, Canadian Blood Services, WHO, literature)
    • notes on applicability / caveats

Organised by category so that simulation outputs can be compared systematically.

Last updated: 2025-07
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data-class for a single benchmark entry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Benchmark:
    """One real-world reference data point for validation."""

    metric: str
    value_low: float
    value_high: float
    unit: str
    source: str
    notes: str = ""
    confidence: str = "medium"  # low / medium / high

    @property
    def midpoint(self) -> float:
        return (self.value_low + self.value_high) / 2.0

    @property
    def range_str(self) -> str:
        if self.value_low == self.value_high:
            return f"{self.value_low:,.4g} {self.unit}"
        return f"{self.value_low:,.4g}–{self.value_high:,.4g} {self.unit}"

    def contains(self, value: float) -> bool:
        """Return True if *value* falls within the benchmark range."""
        return self.value_low <= value <= self.value_high


@dataclass
class BenchmarkCategory:
    """A named group of related benchmarks."""

    name: str
    description: str
    benchmarks: list[Benchmark] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# 1. COLLECTION VOLUMES
# ═══════════════════════════════════════════════════════════════════════════
COLLECTION_VOLUMES = BenchmarkCategory(
    name="Collection Volumes",
    description=(
        "Annual and daily whole-blood and component donation volumes, "
        "per-capita donation rates, and donor participation rates."
    ),
    benchmarks=[
        # --- Héma-Québec province-wide ---
        Benchmark(
            metric="Héma-Québec annual completed whole-blood donations",
            value_low=146_000,
            value_high=148_000,
            unit="donations/year",
            source=(
                "Héma-Québec Annual Report 2024-2025, 'Activités de prélèvement' "
                "section.  146,735 reported donor events for labile products."
            ),
            notes=(
                "Includes fixed-site and mobile collections across all of Quebec.  "
                "Does not include apheresis-only plasma or platelet sessions counted "
                "separately."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Héma-Québec annual labile blood products distributed",
            value_low=305_000,
            value_high=315_000,
            unit="products/year",
            source="Héma-Québec Annual Report 2024-2025.  309,113 labile products.",
            notes=(
                "Labile products = RBC concentrates + platelet concentrates + "
                "fresh-frozen plasma.  One whole-blood donation yields ~2.1 labile "
                "products on average."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Products-per-donation yield (labile)",
            value_low=2.0,
            value_high=2.2,
            unit="products/donation",
            source=(
                "Derived: 309,113 products / 146,735 donations ≈ 2.11.  "
                "Consistent with WHO Technical Report Series 2017 (2.0–2.3 for "
                "well-run separation programs)."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Héma-Québec daily labile products (province)",
            value_low=820,
            value_high=870,
            unit="products/day",
            source="Derived: 309,113 / 365 ≈ 847 products/day.",
            notes="Weekday output is higher; weekends lower.",
            confidence="high",
        ),
        Benchmark(
            metric="Héma-Québec daily completed donations (province)",
            value_low=390,
            value_high=420,
            unit="donations/day",
            source="Derived: 146,735 / 365 ≈ 402 donations/day.",
            confidence="high",
        ),
        Benchmark(
            metric="Quebec City CMA daily completed donations (estimated)",
            value_low=37,
            value_high=43,
            unit="donations/day",
            source=(
                "Scaled from province by population share: Quebec CMA 839,311 / "
                "Quebec province 8,501,833 ≈ 9.87%.  402 × 0.0987 ≈ 39.7."
            ),
            notes=(
                "Assumes donations are roughly proportional to population.  In "
                "practice Quebec City may slightly over- or under-produce relative "
                "to its population share."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Whole-blood donation rate per 1,000 population (Quebec)",
            value_low=16.0,
            value_high=18.5,
            unit="donations/1,000 pop/year",
            source=(
                "Derived: 146,735 / 8,501,833 × 1,000 ≈ 17.3.  "
                "WHO Global Status Report on Blood Safety 2021 reports Canada "
                "at 15–20 donations/1,000 pop."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Whole-blood donation rate per 1,000 population (Canada-wide)",
            value_low=15.0,
            value_high=20.0,
            unit="donations/1,000 pop/year",
            source=(
                "WHO Global Status Report on Blood Safety and Availability 2021; "
                "Canadian Blood Services Annual Report 2022-2023 (~450,000 whole-"
                "blood units for ~26M served population outside Quebec)."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Whole-blood donation rate per 1,000 population (high-income benchmark)",
            value_low=30.0,
            value_high=40.0,
            unit="donations/1,000 pop/year",
            source=(
                "WHO recommends ≥10/1,000 for self-sufficiency; top performers "
                "(Denmark, Austria) reach 30–40.  Canada is mid-range for OECD."
            ),
            notes="Context benchmark only — Canada has separate plasma fractionation.",
            confidence="high",
        ),
        Benchmark(
            metric="Donor participation rate (% eligible pop who donate ≥1×/year)",
            value_low=3.0,
            value_high=4.5,
            unit="%",
            source=(
                "Canadian Blood Services 2022 — ~3.6% of eligible adults donate.  "
                "Héma-Québec reports roughly similar rates.  "
                "Literature: Greinacher et al., Transfusion 2011."
            ),
            notes="Eligible pop ≈ ages 17-71, meeting weight/health criteria.",
            confidence="medium",
        ),
        Benchmark(
            metric="Repeat donor share (% of donors with ≥2 donations/year)",
            value_low=40.0,
            value_high=55.0,
            unit="%",
            source=(
                "Héma-Québec Annual Report 2023-2024 — approximately 48% of donors "
                "are repeat donors.  Canadian Blood Services reports ~50%.  "
                "Consistent with Schreiber et al., Transfusion 2006."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Average donations per donor per year",
            value_low=1.5,
            value_high=2.0,
            unit="donations/donor/year",
            source=(
                "Canadian Blood Services data; Héma-Québec reports ~1.7 average.  "
                "Men can donate up to 6× per year (every 56 days), women up to 4×."
            ),
            confidence="medium",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 2. BLOOD TYPE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
BLOOD_TYPE_DISTRIBUTION = BenchmarkCategory(
    name="Blood Type Distribution",
    description=(
        "ABO/Rh frequency in the Canadian and Quebec population.  "
        "Values are approximate and vary slightly by ethnic composition."
    ),
    benchmarks=[
        Benchmark(
            metric="O+ frequency",
            value_low=0.39,
            value_high=0.46,
            unit="fraction",
            source=(
                "Héma-Québec reports O+ at ~44% for Quebec (predominantly "
                "European descent).  Canadian Blood Services reports 39–46% "
                "depending on ethnic mix.  Garratty et al., Transfusion 2004."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="A+ frequency",
            value_low=0.27,
            value_high=0.34,
            unit="fraction",
            source="Héma-Québec: ~30%.  CBS: 30–36% nationally.",
            confidence="high",
        ),
        Benchmark(
            metric="B+ frequency",
            value_low=0.08,
            value_high=0.14,
            unit="fraction",
            source="Héma-Québec: ~12%.  CBS: 7.6–13% depending on population.",
            confidence="high",
        ),
        Benchmark(
            metric="AB+ frequency",
            value_low=0.03,
            value_high=0.06,
            unit="fraction",
            source="Héma-Québec: ~5%.  CBS: 2.5–5.1%.",
            confidence="high",
        ),
        Benchmark(
            metric="O- frequency",
            value_low=0.04,
            value_high=0.09,
            unit="fraction",
            source=(
                "Héma-Québec: ~4%.  CBS: 6–9% nationally (Quebec's French-"
                "Canadian population has slightly lower O- than English Canada).  "
                "Goldman et al., Transfusion 2005."
            ),
            notes=(
                "The lower Quebec O- rate (~4%) vs. rest of Canada (~7%) is well "
                "documented and reflects genetic ancestry patterns."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="A- frequency",
            value_low=0.03,
            value_high=0.06,
            unit="fraction",
            source="Héma-Québec: ~3%.  CBS: 5–6% nationally.",
            confidence="high",
        ),
        Benchmark(
            metric="B- frequency",
            value_low=0.01,
            value_high=0.02,
            unit="fraction",
            source="Héma-Québec: ~1.5%.  CBS: 1.4–2.5%.",
            confidence="high",
        ),
        Benchmark(
            metric="AB- frequency",
            value_low=0.003,
            value_high=0.01,
            unit="fraction",
            source="Héma-Québec: ~0.5%.  CBS: 0.5–1.0%.",
            confidence="high",
        ),
        Benchmark(
            metric="Rh-negative total",
            value_low=0.08,
            value_high=0.18,
            unit="fraction",
            source=(
                "Quebec ~9% Rh-neg total; rest of Canada ~15–18%.  Caucasian "
                "populations typically 15%; Quebec French-Canadian lower.  "
                "Refer to Reid & Lomas-Francis, The Blood Group Antigen Facts Book."
            ),
            notes="Critical for O- emergency supply planning.",
            confidence="high",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 3. COMPONENT SPLIT RATIOS AND DEMAND
# ═══════════════════════════════════════════════════════════════════════════
COMPONENT_SPLIT = BenchmarkCategory(
    name="Component Split Ratios",
    description=(
        "Production and demand breakdown by blood component type "
        "(RBC concentrates, platelet concentrates, plasma)."
    ),
    benchmarks=[
        Benchmark(
            metric="RBC share of labile product demand",
            value_low=0.50,
            value_high=0.62,
            unit="fraction",
            source=(
                "Canadian Blood Services 2022-2023: RBCs account for ~55–60% of "
                "transfusions.  WHO 2017: 50–65% in high-income countries.  "
                "Héma-Québec product catalogue implies ~58%."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Platelet share of labile product demand",
            value_low=0.18,
            value_high=0.30,
            unit="fraction",
            source=(
                "CBS: ~20–25% of component demand.  WHO: 15–30%.  Héma-Québec "
                "implied ~27% from distribution data.  Trend is increasing due to "
                "oncology and surgical demand."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Plasma (FFP/cryoprecipitate) share of labile product demand",
            value_low=0.12,
            value_high=0.20,
            unit="fraction",
            source=(
                "CBS: ~15–18%.  WHO: 10–20%.  Héma-Québec implied ~15%.  "
                "Note: this is clinical FFP, not source plasma for fractionation."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Héma-Québec daily hospital orders (province-wide)",
            value_low=180,
            value_high=220,
            unit="orders/day",
            source=(
                "Héma-Québec hospital distribution page: ~200 orders/day to "
                "hospitals across Quebec."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Average units per hospital order",
            value_low=3.5,
            value_high=5.0,
            unit="units/order",
            source=(
                "Derived: ~847 products/day ÷ ~200 orders/day ≈ 4.2 units/order.  "
                "Literature range: 2–6 units depending on order type (routine vs. "
                "massive transfusion protocol)."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Massive transfusion protocol (MTP) frequency",
            value_low=0.01,
            value_high=0.03,
            unit="fraction of orders",
            source=(
                "Dzik et al., Transfusion 2011: MTP activations represent 1–3% of "
                "all transfusion events but consume 10–20% of RBC supply.  "
                "MTPs use 6–10+ RBC units in rapid sequence."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Surgical RBC usage per case (median)",
            value_low=1.0,
            value_high=3.0,
            unit="RBC units/case",
            source=(
                "Frank et al., Transfusion 2013: median 2 units for elective "
                "surgery.  Collins et al., BJH 2015.  Trauma/cardiac surgery "
                "higher (4–10 units)."
            ),
            confidence="medium",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 4. SHELF LIFE AND WASTAGE / EXPIRY RATES
# ═══════════════════════════════════════════════════════════════════════════
WASTAGE_AND_EXPIRY = BenchmarkCategory(
    name="Wastage and Expiry Rates",
    description=(
        "Outdating (expiry) and total wastage rates by component.  "
        "These are key simulation validation metrics because they reflect "
        "inventory management effectiveness."
    ),
    benchmarks=[
        # --- Shelf life (regulatory) ---
        Benchmark(
            metric="RBC shelf life (CPDA-1 / additive solution)",
            value_low=35,
            value_high=42,
            unit="days",
            source=(
                "Health Canada regulatory: 42 days in additive solution (AS-1/3/5), "
                "35 days in CPDA-1.  Héma-Québec uses 42 days standard.  "
                "AABB Technical Manual, 20th ed."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Platelet shelf life",
            value_low=5,
            value_high=7,
            unit="days",
            source=(
                "Health Canada: 5 days standard (with bacterial testing 7 days in "
                "some jurisdictions).  Héma-Québec uses 5 days for buffy-coat "
                "derived platelets.  Some pathogen-reduced platelets approved "
                "for 7 days."
            ),
            notes="5-day shelf life is the primary driver of platelet wastage.",
            confidence="high",
        ),
        Benchmark(
            metric="Fresh Frozen Plasma shelf life",
            value_low=365,
            value_high=365,
            unit="days",
            source="Health Canada / AABB: 12 months at ≤−18°C.",
            confidence="high",
        ),
        Benchmark(
            metric="Thawed plasma shelf life",
            value_low=1,
            value_high=5,
            unit="days",
            source=(
                "Once thawed: 24 hours as FFP, up to 5 days as thawed plasma "
                "(with reduced Factor V/VIII).  AABB Standards."
            ),
            confidence="high",
        ),
        # --- Expiry / outdating rates ---
        Benchmark(
            metric="RBC outdating (expiry) rate",
            value_low=1.0,
            value_high=5.0,
            unit="%",
            source=(
                "Héma-Québec target: <2%.  Canadian Blood Services: 1.5–3.0%.  "
                "WHO benchmark: <5%.  "
                "Stanger et al., Transfusion 2012: well-managed systems 1–3%.  "
                "Whitaker et al., AABB National Blood Collection Survey 2019: "
                "US average ~2.5%."
            ),
            notes=(
                "Higher outdating occurs in rural/small hospitals.  Large centres "
                "like Quebec City should be at the low end."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Platelet outdating (expiry) rate",
            value_low=5.0,
            value_high=20.0,
            unit="%",
            source=(
                "CBS: 8–15%.  Héma-Québec: historically 10–18%.  "
                "WHO considers >20% problematic.  "
                "Flint et al., Transfusion 2020: median ~12% across Canadian "
                "hospitals.  Williamson & Devine, Transfusion Medicine 2013: "
                "5–20% internationally."
            ),
            notes=(
                "Platelet outdating is the single largest source of blood product "
                "wastage due to the 5-day shelf life.  This is a CRITICAL "
                "validation metric."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Plasma outdating (expiry) rate",
            value_low=0.5,
            value_high=3.0,
            unit="%",
            source=(
                "CBS: <2%.  Long shelf life (365 days) makes plasma outdating rare.  "
                "Stanger et al., Transfusion 2012: <1% in most systems."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Total wastage rate (all causes, all components)",
            value_low=3.0,
            value_high=8.0,
            unit="%",
            source=(
                "Includes expiry + damage + testing failures + contamination.  "
                "CBS 2022: ~5–6%.  WHO target: <5%.  "
                "Javadzadeh Shahshahani, Transfusion Medicine 2007: 3–10% globally."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Non-expiry wastage (damage, contamination, etc.)",
            value_low=0.5,
            value_high=2.0,
            unit="%",
            source=(
                "CBS data: ~1% for cold-chain breaks, bag damage, bacterial "
                "contamination.  Literature: 0.5–2.0%.  "
                "AABB National Blood Collection Survey 2019."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="RBC average age at transfusion",
            value_low=14,
            value_high=21,
            unit="days",
            source=(
                "Heddle et al., INFORM trial, NEJM 2016: mean age 13–18 days "
                "in Canadian hospitals.  Lacroix et al., ABLE trial, NEJM 2015: "
                "median 15 days.  FIFO policy → ~14–21 days."
            ),
            notes="Used to validate inventory turnover dynamics.",
            confidence="high",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 5. SHORTAGE / STOCKOUT RATES
# ═══════════════════════════════════════════════════════════════════════════
SHORTAGE_RATES = BenchmarkCategory(
    name="Shortage and Stockout Rates",
    description=(
        "Frequency and severity of blood product shortages, unfilled orders, "
        "and emergency measures.  Critical for validating the simulation's "
        "demand-supply balance."
    ),
    benchmarks=[
        Benchmark(
            metric="Overall order fulfillment rate (service level)",
            value_low=95.0,
            value_high=99.5,
            unit="%",
            source=(
                "Héma-Québec target: >97% same-day fulfillment.  "
                "CBS: 97–99% routine fulfillment.  "
                "Beliën & Forcé, EJOR 2012: well-run blood banks achieve 95–99% "
                "service levels."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Shortage rate (unfilled / total requested units)",
            value_low=0.5,
            value_high=5.0,
            unit="%",
            source=(
                "Inverse of service level.  Literature: 1–5% shortage is typical "
                "for well-managed systems.  "
                "Katsaliaki & Brailsford, JORS 2007: 2–5% in simulation studies.  "
                "Rajendran & Ravindran, IJPE 2017: 1–3% target."
            ),
            notes=(
                "Shortage rates >5% indicate systemic problems.  "
                "Rates <1% may indicate over-stocking (wastage trade-off)."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="RBC stockout frequency (days with zero inventory for ≥1 type)",
            value_low=2.0,
            value_high=10.0,
            unit="% of days",
            source=(
                "CBS Inventory Dashboard: O-neg stockouts occur on ~5–8% of days.  "
                "Héma-Québec emergency appeals issued ~5–10 times/year, implying "
                "1–3% of days at critical levels for at least one type.  "
                "Perera et al., Transfusion 2009."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="O-negative emergency supply days below 2-day target",
            value_low=5.0,
            value_high=15.0,
            unit="% of days",
            source=(
                "CBS reports O-neg supply falls below 2-day buffer on ~10% of days.  "
                "Héma-Québec issues specific O-neg appeals several times per year."
            ),
            notes="O-neg is the universal emergency type and most shortage-prone.",
            confidence="medium",
        ),
        Benchmark(
            metric="Compatible substitution rate (non-identical but compatible unit used)",
            value_low=10.0,
            value_high=30.0,
            unit="% of transfusions",
            source=(
                "Zhu et al., Transfusion 2022: 15–25% of RBC transfusions use "
                "ABO-compatible but non-identical units.  Higher during shortages.  "
                "Denomme et al., Transfusion 2021."
            ),
            notes="Tracks how often the system resorts to compatible substitution.",
            confidence="medium",
        ),
        Benchmark(
            metric="Emergency blood request share",
            value_low=5.0,
            value_high=15.0,
            unit="% of all orders",
            source=(
                "CBS data: ~8–12% of orders are classified as urgent/emergency.  "
                "Héma-Québec does not publish exact breakdown but similar pattern.  "
                "Dzik et al., Transfusion 2011: 5–15% range."
            ),
            confidence="medium",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 6. DONOR DEFERRAL AND REJECTION RATES
# ═══════════════════════════════════════════════════════════════════════════
DONOR_DEFERRAL = BenchmarkCategory(
    name="Donor Deferral and Rejection Rates",
    description=(
        "Rates at which presenting donors are deferred (temporarily or "
        "permanently) from donating.  Includes pre-donation screening, "
        "hemoglobin testing, and post-donation lab rejection."
    ),
    benchmarks=[
        Benchmark(
            metric="Overall donor deferral rate (all causes)",
            value_low=7.0,
            value_high=15.0,
            unit="% of presenting donors",
            source=(
                "Héma-Québec: ~10–12%.  CBS: 10–14%.  "
                "WHO 2017: 5–15% in high-income countries.  "
                "Custer et al., Transfusion 2004 (REDS-II): 12.8% US.  "
                "Gonçalez et al., Transfusion 2013: 10–15% in developed nations."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Low hemoglobin deferral rate",
            value_low=3.0,
            value_high=8.0,
            unit="% of presenting donors",
            source=(
                "CBS: ~5–7% overall (higher in females: 8–12%, males: 2–3%).  "
                "Héma-Québec thresholds: males ≥130 g/L, females ≥125 g/L.  "
                "Spencer et al., Transfusion 2016: 5.6% mean."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Travel-related deferral rate",
            value_low=1.0,
            value_high=3.0,
            unit="% of presenting donors",
            source=(
                "CBS: ~1.5–2.5%.  Héma-Québec: similar.  Reduced after 2023 "
                "malaria deferral policy changes.  Seed et al., Transfusion 2010."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Post-donation lab rejection (TTI testing)",
            value_low=0.1,
            value_high=0.5,
            unit="% of collected units",
            source=(
                "CBS: ~0.2–0.3% NAT/serology reactive (HIV, HCV, HBV, syphilis).  "
                "Héma-Québec: ~0.15–0.25%.  "
                "O'Brien et al., Transfusion 2007 (REDS-II): 0.3% US.  "
                "The simulator uses 0.5% which is a conservative upper bound."
            ),
            notes="Very low in Canada due to strict donor screening.",
            confidence="high",
        ),
        Benchmark(
            metric="Tattoo/piercing deferral rate",
            value_low=0.5,
            value_high=2.0,
            unit="% of presenting donors",
            source=(
                "CBS: ~1–2%.  Reduced from historical 3–5% as deferral period "
                "shortened from 12 to 6 months (2022 policy).  "
                "Héma-Québec: ~1%."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Minimum inter-donation interval (whole blood)",
            value_low=56,
            value_high=56,
            unit="days",
            source=(
                "Health Canada: minimum 56 days between whole-blood donations.  "
                "Maximum 6 donations/year for males, 4 for females (with "
                "hemoglobin monitoring).  Héma-Québec policy."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Donor age range (eligible)",
            value_low=17,
            value_high=71,
            unit="years",
            source=(
                "Héma-Québec: 18–70 for first-time donors, returning donors may "
                "continue beyond 70 with medical clearance.  CBS: 17–71.  "
                "The simulator uses 17–71 inclusive."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Minimum donor weight",
            value_low=50,
            value_high=50,
            unit="kg",
            source="Héma-Québec and CBS: minimum 50 kg (110 lbs).",
            confidence="high",
        ),
        Benchmark(
            metric="Donor eligibility rate (% of presenting donors who are eligible)",
            value_low=85.0,
            value_high=93.0,
            unit="%",
            source=(
                "Complement of deferral rate.  Héma-Québec: ~88–90%.  CBS: 86–90%.  "
                "The simulator calibration uses 88% baseline."
            ),
            confidence="high",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 7. DONOR SHOW-UP AND APPOINTMENT ADHERENCE
# ═══════════════════════════════════════════════════════════════════════════
DONOR_SHOWUP = BenchmarkCategory(
    name="Donor Show-Up and Appointment Adherence",
    description=(
        "Rates at which scheduled or intended donors actually present "
        "at collection sites.  Affected by weather, day-of-week, season."
    ),
    benchmarks=[
        Benchmark(
            metric="Appointment no-show rate (scheduled donors)",
            value_low=10.0,
            value_high=25.0,
            unit="%",
            source=(
                "CBS: 15–20% no-show rate for booked appointments (pre-COVID).  "
                "Héma-Québec: ~12–18%.  "
                "Godin et al., Transfusion 2007: 15–25% depending on season.  "
                "Schlumpf et al., Transfusion 2008: 10–20% Swiss data."
            ),
            notes="Walk-in donors have different no-show semantics.",
            confidence="medium",
        ),
        Benchmark(
            metric="Walk-in donor share",
            value_low=15.0,
            value_high=40.0,
            unit="% of presenting donors",
            source=(
                "CBS: ~20–30% walk-in.  Héma-Québec: ~25–35% (varies by site).  "
                "Trend is toward more appointments post-COVID.  "
                "Seifried et al., Vox Sanguinis 2011."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Baseline completion yield (show-up × eligibility)",
            value_low=60.0,
            value_high=72.0,
            unit="% of attempted donors",
            source=(
                "Product of ~75–85% show-up rate × ~88–93% eligibility rate.  "
                "Net yield of donor attempts to completed donations: 60–72%.  "
                "The simulator calibration uses 66%."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Weather impact on donor show-up (severe winter)",
            value_low=0.50,
            value_high=0.80,
            unit="multiplier vs. baseline",
            source=(
                "Héma-Québec reports 20–50% drop during major storms.  "
                "Custer et al., Transfusion 2007: 15–40% drop in severe weather.  "
                "Gillespie & Hillyer, Transfusion 2002: snowstorms reduce "
                "collections by 25–50%."
            ),
            notes=(
                "Quebec City is particularly affected due to heavy winter weather.  "
                "Blizzards can reduce collections by half."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Holiday impact on donor show-up",
            value_low=0.30,
            value_high=0.70,
            unit="multiplier vs. baseline",
            source=(
                "CBS: collections drop 30–70% on statutory holidays.  "
                "Christmas/New Year period especially impacted.  "
                "Héma-Québec launches special holiday campaigns annually."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Weekend vs. weekday collection ratio",
            value_low=0.40,
            value_high=0.70,
            unit="ratio (weekend/weekday)",
            source=(
                "CBS: weekend collections ~40–60% of weekday.  Some fixed sites "
                "closed on Sundays.  Mobile drives are less common on weekends.  "
                "Héma-Québec sites have reduced weekend hours."
            ),
            confidence="medium",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 8. PROCESSING TIMES
# ═══════════════════════════════════════════════════════════════════════════
PROCESSING_TIMES = BenchmarkCategory(
    name="Processing Times",
    description=(
        "Durations for each step from donor registration through component "
        "release.  Used to validate the simulation's process timing."
    ),
    benchmarks=[
        Benchmark(
            metric="Donor registration time",
            value_low=5,
            value_high=15,
            unit="minutes",
            source=(
                "CBS donor centre standards: 5–15 minutes.  "
                "Héma-Québec reports ~10 minutes average.  "
                "France, EFS reports similar.  "
                "The simulator uses 5–15 min (0.083–0.25 h)."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Health screening / questionnaire time",
            value_low=5,
            value_high=10,
            unit="minutes",
            source=(
                "CBS: 5–10 minutes for the donor health questionnaire + mini-physical "
                "(blood pressure, hemoglobin finger prick).  "
                "Héma-Québec: similar.  Simulator uses 5–10 min."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Whole-blood collection time (phlebotomy)",
            value_low=8,
            value_high=15,
            unit="minutes",
            source=(
                "CBS standard: 8–12 minutes for 450 mL whole-blood draw.  "
                "Range 8–15 minutes.  AABB Technical Manual.  "
                "Simulator uses 10–15 min."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Post-donation observation / refreshment time",
            value_low=10,
            value_high=20,
            unit="minutes",
            source=(
                "CBS: minimum 15 minutes post-donation observation.  "
                "Total donor visit time (arrival to departure): 45–75 minutes."
            ),
            notes="Not explicitly modelled in the simulator but affects throughput.",
            confidence="high",
        ),
        Benchmark(
            metric="Lab testing (NAT + serology) turnaround",
            value_low=1,
            value_high=8,
            unit="hours",
            source=(
                "Héma-Québec centralised lab: samples shipped to Montreal, results "
                "in 8–24 hours.  Some sites with local rapid NAT: 1–3 hours.  "
                "The simulator uses 1–3 h (simplified local testing)."
            ),
            notes="Héma-Québec runs NAT testing at its Montreal facility.",
            confidence="medium",
        ),
        Benchmark(
            metric="Component processing (centrifugation, separation)",
            value_low=1,
            value_high=6,
            unit="hours",
            source=(
                "AABB Technical Manual: component preparation within 8 hours of "
                "collection (for platelet recovery).  Actual centrifugation + "
                "separation: 1–2 hours.  Quality hold: 2–4 hours.  "
                "Simulator uses 2–4 h."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Quarantine / release hold",
            value_low=24,
            value_high=72,
            unit="hours",
            source=(
                "Products held in quarantine until all testing is complete.  "
                "Héma-Québec: 24–48 hours typical, up to 72 hours if testing "
                "batches are delayed (weekends).  Simulator uses 24–72 h."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Total collection-to-release time (label release)",
            value_low=24,
            value_high=96,
            unit="hours",
            source=(
                "CBS: 1–3 days from collection to product release.  "
                "Héma-Québec: 24–72 hours for RBCs, may be longer for pooled "
                "platelets.  AABB: products must be labelled before issue."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Transport time: Héma-Québec facility to Quebec City hospital",
            value_low=15,
            value_high=60,
            unit="minutes",
            source=(
                "Quebec City intra-urban: 15–45 minutes.  From Héma-Québec "
                "Lebourgneuf to major hospitals (CHU, Hôtel-Dieu): 15–30 min.  "
                "Adverse weather: up to 60 min.  "
                "Google Maps / OpenStreetMap routing confirms these ranges."
            ),
            confidence="high",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 9. INVENTORY DAYS OF SUPPLY
# ═══════════════════════════════════════════════════════════════════════════
INVENTORY_LEVELS = BenchmarkCategory(
    name="Inventory Days of Supply",
    description=(
        "Target and actual inventory buffer levels maintained by blood "
        "centres and hospital blood banks."
    ),
    benchmarks=[
        Benchmark(
            metric="RBC target days of supply (national/provincial blood centre)",
            value_low=3.0,
            value_high=7.0,
            unit="days",
            source=(
                "CBS target: 5-day supply nationally.  Héma-Québec: 3–5 day target "
                "for provincial inventory.  "
                "AABB recommends 3–7 days.  "
                "Beliën & Forcé, EJOR 2012: 3–5 day standard in literature."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Platelet target days of supply",
            value_low=1.0,
            value_high=3.0,
            unit="days",
            source=(
                "CBS: 1.5–2.5 day target.  Very tight due to 5-day shelf life.  "
                "Héma-Québec: ~2 days.  "
                "WHO: maintain ≥1 day buffer at all times."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Plasma target days of supply",
            value_low=5.0,
            value_high=14.0,
            unit="days",
            source=(
                "CBS: 7–14 day target.  Long shelf life allows larger buffers.  "
                "Héma-Québec: 5–10 days."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Hospital blood bank typical RBC inventory",
            value_low=1.0,
            value_high=4.0,
            unit="days",
            source=(
                "Large teaching hospitals: 2–4 day on-site inventory.  "
                "Small community hospitals: 1–2 days with frequent resupply.  "
                "Chapman et al., Transfusion 2009."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="O-negative RBC target inventory (% of total RBC)",
            value_low=6.0,
            value_high=10.0,
            unit="% of RBC inventory",
            source=(
                "CBS recommends O-neg should be 6–10% of RBC inventory despite "
                "only 4–7% of population being O-neg.  Over-stocked because of "
                "emergency use.  Héma-Québec similar policy."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Initial pre-stock inventory for simulation",
            value_low=3.0,
            value_high=5.0,
            unit="days of supply",
            source=(
                "Simulation calibration uses 4.0 days as baseline.  Consistent "
                "with Héma-Québec operational target of 3–5 days."
            ),
            confidence="high",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 10. SERVICE LEVELS AND FULFILLMENT
# ═══════════════════════════════════════════════════════════════════════════
SERVICE_LEVELS = BenchmarkCategory(
    name="Service Levels and Fulfillment",
    description=(
        "Quality-of-service metrics for the blood supply system including "
        "fulfillment rates, wait times, and type-specific match rates."
    ),
    benchmarks=[
        Benchmark(
            metric="Same-day order fulfillment rate (routine orders)",
            value_low=95.0,
            value_high=99.0,
            unit="%",
            source=(
                "Héma-Québec: >97% target for routine orders.  "
                "CBS: 97–99%.  "
                "Stanger et al., Transfusion 2012."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Emergency order fulfillment rate",
            value_low=98.0,
            value_high=100.0,
            unit="%",
            source=(
                "Emergency (e.g., trauma) orders are almost always filled, "
                "using O-neg if type-specific not available.  "
                "CBS target: >99%.  Héma-Québec: >99%."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Exact ABO/Rh match rate (type-identical transfusion)",
            value_low=70.0,
            value_high=90.0,
            unit="% of transfusions",
            source=(
                "Standard practice: type-specific whenever possible.  "
                "Zhu et al., Transfusion 2022: 75–85% exact match in practice.  "
                "Remainder is ABO-compatible substitution (e.g., O- for A+)."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Crossmatch-to-transfusion ratio (C/T ratio)",
            value_low=1.5,
            value_high=3.0,
            unit="ratio",
            source=(
                "AABB benchmark: C/T ratio <2.0 is efficient.  "
                "Frank et al., Transfusion 2013: median 2.0–2.5.  "
                "Higher ratios indicate over-ordering."
            ),
            notes="Not directly modelled in the simulator but useful context.",
            confidence="medium",
        ),
        Benchmark(
            metric="Hospital turnaround time (order to delivery)",
            value_low=15,
            value_high=120,
            unit="minutes",
            source=(
                "Routine: 30–120 minutes.  Urgent: 15–30 minutes.  "
                "Emergency O-neg: <10 minutes from hospital blood bank.  "
                "Héma-Québec delivery within Quebec City: 15–60 min."
            ),
            confidence="medium",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 11. DONOR DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════════════
DONOR_DEMOGRAPHICS = BenchmarkCategory(
    name="Donor Demographics",
    description=("Age, sex, and motivational profile of the blood donor population."),
    benchmarks=[
        Benchmark(
            metric="Male donor share",
            value_low=45.0,
            value_high=55.0,
            unit="%",
            source=(
                "CBS: ~47–52% male.  Héma-Québec: ~50% male.  "
                "WHO 2021: high-income countries 45–55% male."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Median donor age",
            value_low=35,
            value_high=45,
            unit="years",
            source=(
                "CBS: median ~42 years, trending older.  "
                "Héma-Québec: ~38–42 years.  "
                "Young donors (18–24) represent 12–18% of donations.  "
                "Greinacher et al., Transfusion 2011: aging donor base trend."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Young donor share (age 18-24)",
            value_low=10.0,
            value_high=20.0,
            unit="% of donors",
            source=(
                "CBS: ~15%.  Héma-Québec: ~12–18%.  "
                "University campus drives boost young donor recruitment."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="First-time donor share",
            value_low=15.0,
            value_high=30.0,
            unit="% of presenting donors",
            source=(
                "CBS: ~20–25%.  Héma-Québec: ~20%.  "
                "First-time donors have higher deferral rates (~15–20% vs. 8–10% "
                "for repeat donors).  Custer et al., Transfusion 2007."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Donor hemoglobin — male (mean)",
            value_low=14.0,
            value_high=15.5,
            unit="g/dL",
            source=(
                "CBS: mean ~14.8 g/dL for male donors.  "
                "Health Canada minimum: 13.0 g/dL.  "
                "The simulator uses Gaussian(14.5, 1.5)."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Donor hemoglobin — female (mean)",
            value_low=12.5,
            value_high=14.0,
            unit="g/dL",
            source=(
                "CBS: mean ~13.5 g/dL for female donors.  "
                "Health Canada minimum: 12.5 g/dL.  "
                "The simulator uses Gaussian(13.0, 1.5)."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Donor weight — male (mean)",
            value_low=78,
            value_high=86,
            unit="kg",
            source=(
                "Statistics Canada Health Survey: Canadian adult male mean ~84 kg.  "
                "Donor population skews slightly lighter (self-selected healthy).  "
                "The simulator uses Gaussian(78, 12)."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Donor weight — female (mean)",
            value_low=62,
            value_high=72,
            unit="kg",
            source=(
                "Statistics Canada: Canadian adult female mean ~71 kg.  "
                "Donor population may skew slightly lighter.  "
                "The simulator uses Gaussian(65, 12)."
            ),
            confidence="medium",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 12. WEATHER AND SEASONAL EFFECTS
# ═══════════════════════════════════════════════════════════════════════════
WEATHER_SEASONAL = BenchmarkCategory(
    name="Weather and Seasonal Effects",
    description=(
        "Impact of weather conditions and seasonal patterns on blood "
        "collection and distribution in Quebec City."
    ),
    benchmarks=[
        Benchmark(
            metric="Quebec City average snowfall days per winter (Nov–Mar)",
            value_low=80,
            value_high=110,
            unit="days/winter",
            source=(
                "Environment Canada: Quebec City averages ~300 cm of snow/year, "
                "with ~90–100 days of measurable snowfall."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Days with blizzard/ice storm conditions (Quebec City)",
            value_low=5,
            value_high=15,
            unit="days/winter",
            source=(
                "Environment Canada: 5–15 days of significant winter storms "
                "per year.  Major ice storms (e.g., 1998, 2023) are rare but "
                "devastating."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Winter transport speed reduction (snow)",
            value_low=0.60,
            value_high=0.80,
            unit="multiplier vs. clear",
            source=(
                "Transport Canada: winter driving speeds reduced 20–40% in snow.  "
                "The simulator uses 0.70 for snow conditions."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Ice storm transport speed reduction",
            value_low=0.30,
            value_high=0.55,
            unit="multiplier vs. clear",
            source=(
                "Transport Canada / CAA: severe ice reduces speeds by 45–70%.  "
                "The simulator uses 0.45 for ice storms."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Seasonal demand variation (winter vs. summer)",
            value_low=0.85,
            value_high=1.15,
            unit="ratio to annual mean",
            source=(
                "CBS: demand fairly stable year-round, ±10–15%.  "
                "Summer has slightly lower surgical demand (elective surgery "
                "slowdown) but higher trauma.  Winter has flu-related "
                "transfusion demand.  "
                "Héma-Québec: relatively flat demand curve."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Seasonal collection variation (summer valley)",
            value_low=0.75,
            value_high=0.90,
            unit="ratio to annual mean",
            source=(
                "CBS and Héma-Québec: summer collections drop 10–25% due to "
                "vacations.  Holiday periods (Christmas, St-Jean-Baptiste) "
                "see sharp drops.  "
                "Gillespie & Hillyer, Transfusion 2002."
            ),
            confidence="medium",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 13. TRANSFUSION MEDICINE CLINICAL BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════
CLINICAL_BENCHMARKS = BenchmarkCategory(
    name="Transfusion Medicine Clinical Benchmarks",
    description=("Clinical practice parameters that influence demand patterns."),
    benchmarks=[
        Benchmark(
            metric="RBC transfusion trigger (hemoglobin threshold)",
            value_low=7.0,
            value_high=9.0,
            unit="g/dL",
            source=(
                "AABB Clinical Practice Guideline 2012 (Carson et al.): "
                "restrictive threshold 7–8 g/dL for stable patients.  "
                "Liberal threshold 9–10 g/dL for cardiac patients.  "
                "Hébert et al., TRICC trial, NEJM 1999."
            ),
            notes=(
                "Trend toward restrictive transfusion has reduced RBC demand "
                "by ~15–25% over the last decade in Canada."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Platelet transfusion trigger",
            value_low=10,
            value_high=20,
            unit="×10⁹/L",
            source=(
                "AABB 2015: prophylactic platelet transfusion at <10×10⁹/L for "
                "haematology patients.  <20×10⁹/L for invasive procedures.  "
                "<50×10⁹/L for major surgery."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="RBC units transfused per 1,000 population per year (Canada)",
            value_low=25,
            value_high=40,
            unit="units/1,000 pop/year",
            source=(
                "CBS: ~32–36 RBC units/1,000 pop.  "
                "WHO 2021: high-income average 30–40 RBC units/1,000 pop.  "
                "Trend declining due to patient blood management (PBM)."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Total components transfused per 1,000 population per year",
            value_low=40,
            value_high=60,
            unit="components/1,000 pop/year",
            source=(
                "CBS + Héma-Québec combined: ~45–55 components/1,000 pop.  "
                "Includes RBC, platelets, plasma.  WHO high-income average: "
                "40–60 components/1,000 pop."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Patient blood management (PBM) impact on demand",
            value_low=0.75,
            value_high=0.90,
            unit="multiplier on historical demand",
            source=(
                "Leahy et al., Transfusion 2017: PBM programs reduce RBC use "
                "by 10–25%.  Adopted nationally in Canada.  "
                "CBS reports ~15% demand reduction since 2010."
            ),
            confidence="medium",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 14. HÉMA-QUÉBEC OPERATIONAL SPECIFICS
# ═══════════════════════════════════════════════════════════════════════════
HEMA_QUEBEC_OPERATIONS = BenchmarkCategory(
    name="Héma-Québec Operational Specifics",
    description=(
        "Organisation-specific operational data for Héma-Québec, the sole "
        "blood supplier for the province of Quebec."
    ),
    benchmarks=[
        Benchmark(
            metric="Number of Héma-Québec fixed donor centres (province)",
            value_low=8,
            value_high=12,
            unit="centres",
            source=(
                "Héma-Québec website / annual report: ~10 fixed collection "
                "sites across Quebec, including GLOBULE blood donor centres."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Number of Héma-Québec mobile collection drives per year",
            value_low=5_000,
            value_high=8_000,
            unit="drives/year",
            source=(
                "Héma-Québec: organises ~6,000–7,000 mobile blood drives/year "
                "across Quebec (workplaces, community centres, schools)."
            ),
            confidence="medium",
        ),
        Benchmark(
            metric="Quebec City area collection sites modelled",
            value_low=3,
            value_high=5,
            unit="sites",
            source=(
                "Simulation models: Héma-Québec Lebourgneuf, Héma-Québec "
                "Sainte-Foy, CHU Enfant-Jésus (hospital), Hôtel-Dieu de Lévis "
                "(hospital), Mobile Unit (Université Laval)."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Héma-Québec collection hours (fixed sites)",
            value_low=7.0,
            value_high=20.5,
            unit="hours (open–close)",
            source=(
                "Héma-Québec GLOBULE centres: typically 7:15 AM – 7:45/8:15 PM.  "
                "Varies by day.  Simulator uses 7:15–19:45 and 7:15–20:15."
            ),
            confidence="high",
        ),
        Benchmark(
            metric="Héma-Québec hospital clients served (province)",
            value_low=100,
            value_high=120,
            unit="hospitals",
            source=("Héma-Québec: supplies ~100 hospital blood banks across Quebec."),
            confidence="medium",
        ),
        Benchmark(
            metric="Héma-Québec staff — total full-time equivalents",
            value_low=1_400,
            value_high=1_700,
            unit="FTEs",
            source=("Héma-Québec Annual Report: ~1,500–1,600 employees."),
            confidence="medium",
        ),
        Benchmark(
            metric="Héma-Québec annual operating budget",
            value_low=350,
            value_high=450,
            unit="million CAD",
            source=(
                "Héma-Québec Annual Report 2023-2024: ~$380–420M operating "
                "budget.  Funded by MSSS (Quebec health ministry)."
            ),
            confidence="medium",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# 15. POPULATION AND DEMOGRAPHIC CONTEXT
# ═══════════════════════════════════════════════════════════════════════════
POPULATION_CONTEXT = BenchmarkCategory(
    name="Population and Demographic Context",
    description=(
        "Population statistics for Quebec and Quebec City used to scale the simulation."
    ),
    benchmarks=[
        Benchmark(
            metric="Quebec province total population (2021 Census)",
            value_low=8_501_833,
            value_high=8_501_833,
            unit="persons",
            source="Statistics Canada, 2021 Census of Population.",
            confidence="high",
        ),
        Benchmark(
            metric="Quebec CMA population (2021 Census)",
            value_low=839_311,
            value_high=839_311,
            unit="persons",
            source="Statistics Canada, 2021 Census, Quebec CMA.",
            confidence="high",
        ),
        Benchmark(
            metric="Quebec City population share of province",
            value_low=9.5,
            value_high=10.2,
            unit="%",
            source="Derived: 839,311 / 8,501,833 ≈ 9.87%.",
            confidence="high",
        ),
        Benchmark(
            metric="Quebec median age",
            value_low=42.0,
            value_high=43.5,
            unit="years",
            source="Statistics Canada 2021: Quebec median age 42.6 years.",
            confidence="high",
        ),
        Benchmark(
            metric="Eligible donor-age population (17-71) share",
            value_low=65.0,
            value_high=72.0,
            unit="% of total population",
            source=(
                "Statistics Canada age pyramid: ~68% of Quebec population "
                "is aged 17–71."
            ),
            confidence="medium",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# MASTER CATALOGUE
# ═══════════════════════════════════════════════════════════════════════════
ALL_CATEGORIES: list[BenchmarkCategory] = [
    COLLECTION_VOLUMES,
    BLOOD_TYPE_DISTRIBUTION,
    COMPONENT_SPLIT,
    WASTAGE_AND_EXPIRY,
    SHORTAGE_RATES,
    DONOR_DEFERRAL,
    DONOR_SHOWUP,
    PROCESSING_TIMES,
    INVENTORY_LEVELS,
    SERVICE_LEVELS,
    DONOR_DEMOGRAPHICS,
    WEATHER_SEASONAL,
    CLINICAL_BENCHMARKS,
    HEMA_QUEBEC_OPERATIONS,
    POPULATION_CONTEXT,
]

ALL_BENCHMARKS: list[Benchmark] = [
    bm for cat in ALL_CATEGORIES for bm in cat.benchmarks
]


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def validate_metric(
    metric_name: str,
    simulated_value: float,
    *,
    tolerance_pct: float = 10.0,
) -> dict:
    """
    Compare a simulated value against all benchmarks whose metric name
    contains *metric_name* (case-insensitive substring match).

    Returns a dict with:
        matched  — list of (Benchmark, in_range: bool, deviation_pct: float)
        passed   — True if the simulated value is within the benchmark range
                   (optionally expanded by *tolerance_pct*) for ALL matches.
    """
    results: list[tuple[Benchmark, bool, float]] = []
    query = metric_name.lower()

    for bm in ALL_BENCHMARKS:
        if query not in bm.metric.lower():
            continue

        # Expand range by tolerance %
        span = bm.value_high - bm.value_low
        centre = bm.midpoint
        tolerance_abs = max(span / 2.0, abs(centre) * tolerance_pct / 100.0)
        lo = bm.value_low - tolerance_abs
        hi = bm.value_high + tolerance_abs

        in_range = lo <= simulated_value <= hi
        if centre != 0:
            deviation = (simulated_value - centre) / abs(centre) * 100.0
        else:
            deviation = 0.0
        results.append((bm, in_range, deviation))

    return {
        "metric_name": metric_name,
        "simulated_value": simulated_value,
        "matched": results,
        "passed": all(in_range for _, in_range, _ in results) if results else False,
        "n_matches": len(results),
    }


def validate_simulation_summary(
    summary: dict,
    sim_hours: float = 336.0,
    *,
    tolerance_pct: float = 15.0,
) -> list[dict]:
    """
    Run a battery of automated checks against a simulation summary dict
    (as produced by engine.summarize_state).

    Returns a list of validation result dicts.
    """
    results = []
    sim_days = sim_hours / 24.0

    # --- Shortage rate ---
    if "shortage_rate" in summary:
        results.append(
            validate_metric(
                "Shortage rate",
                summary["shortage_rate"],
                tolerance_pct=tolerance_pct,
            )
        )

    # --- Rejection rate vs. deferral benchmarks ---
    if "rejection_rate" in summary:
        results.append(
            validate_metric(
                "Overall donor deferral rate",
                summary["rejection_rate"],
                tolerance_pct=tolerance_pct,
            )
        )

    # --- Wastage ---
    total_expired = summary.get("total_expired", 0)
    total_transfused = summary.get("total_transfused", 0)
    total_donated = summary.get("total_donated", 0)
    if total_transfused + total_expired > 0:
        expiry_rate = total_expired / (total_transfused + total_expired) * 100.0
        results.append(
            validate_metric(
                "Total wastage rate",
                expiry_rate,
                tolerance_pct=tolerance_pct,
            )
        )

    # --- Donations per day ---
    if total_donated > 0 and sim_days > 0:
        daily_donations = total_donated / sim_days
        results.append(
            validate_metric(
                "Quebec City CMA daily completed donations",
                daily_donations,
                tolerance_pct=tolerance_pct,
            )
        )

    # --- Service level ---
    total_net_requested = summary.get("total_net_requested", 0)
    if total_net_requested > 0:
        service_rate = total_transfused / total_net_requested * 100.0
        results.append(
            validate_metric(
                "Overall order fulfillment rate",
                service_rate,
                tolerance_pct=tolerance_pct,
            )
        )

    # --- Exact match rate ---
    if "total_exact_match_units" in summary and total_transfused > 0:
        exact_rate = summary["total_exact_match_units"] / total_transfused * 100.0
        results.append(
            validate_metric(
                "Exact ABO/Rh match rate",
                exact_rate,
                tolerance_pct=tolerance_pct,
            )
        )

    return results


def print_benchmark_report() -> None:
    """Print a formatted summary of all benchmarks to stdout."""
    total = 0
    for cat in ALL_CATEGORIES:
        print(f"\n{'═' * 80}")
        print(f"  {cat.name.upper()}")
        print(f"  {cat.description}")
        print(f"{'═' * 80}")
        for i, bm in enumerate(cat.benchmarks, 1):
            total += 1
            print(f"\n  [{i}] {bm.metric}")
            print(f"      Range : {bm.range_str}")
            print(f"      Source: {bm.source}")
            if bm.notes:
                print(f"      Notes : {bm.notes}")
            print(f"      Confidence: {bm.confidence}")

    print(f"\n{'─' * 80}")
    print(f"  TOTAL BENCHMARKS: {total}")
    print(f"  CATEGORIES: {len(ALL_CATEGORIES)}")
    print(f"{'─' * 80}\n")


def print_validation_results(results: list[dict]) -> None:
    """Pretty-print the output of validate_simulation_summary."""
    n_passed = sum(1 for r in results if r["passed"])
    n_total = len(results)

    print(f"\n{'═' * 80}")
    print(f"  SIMULATION VALIDATION RESULTS  ({n_passed}/{n_total} passed)")
    print(f"{'═' * 80}")

    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"\n  {status}  {r['metric_name']}")
        print(f"         Simulated: {r['simulated_value']:.2f}")
        for bm, in_range, dev in r["matched"]:
            flag = "✓" if in_range else "✗"
            print(f"         {flag} Benchmark: {bm.range_str}  (dev: {dev:+.1f}%)")
            print(f"           Source: {bm.source[:80]}...")

    print(f"\n{'─' * 80}")
    print(f"  SUMMARY: {n_passed}/{n_total} checks passed")
    print(f"{'─' * 80}\n")


# ═══════════════════════════════════════════════════════════════════════════
# KEY LITERATURE REFERENCES (for thesis bibliography)
# ═══════════════════════════════════════════════════════════════════════════
LITERATURE_REFERENCES: list[dict[str, str]] = [
    {
        "key": "HemaQuebec2025",
        "citation": ("Héma-Québec, Rapport annuel 2024-2025, Gouvernement du Québec."),
        "url": "https://www.hema-quebec.qc.ca/publications",
        "used_for": (
            "Collection volumes, labile product counts, daily hospital orders, "
            "blood type distribution for Quebec population."
        ),
    },
    {
        "key": "CBS2023",
        "citation": ("Canadian Blood Services, Annual Report 2022-2023."),
        "url": "https://www.blood.ca/en/about-us/publications-and-reports",
        "used_for": (
            "National donation rates, wastage rates, inventory targets, "
            "deferral rates, donor demographics."
        ),
    },
    {
        "key": "WHO2021",
        "citation": (
            "World Health Organization, Global Status Report on Blood Safety "
            "and Availability 2021, Geneva: WHO."
        ),
        "url": "https://www.who.int/publications/i/item/9789240051683",
        "used_for": (
            "International donation rate benchmarks, wastage targets, "
            "component utilisation norms."
        ),
    },
    {
        "key": "StatsCan2021",
        "citation": (
            "Statistics Canada, 2021 Census of Population, "
            "Catalogue no. 98-316-X2021001."
        ),
        "url": "https://www12.statcan.gc.ca/census-recensement/2021",
        "used_for": "Quebec and Quebec CMA population figures.",
    },
    {
        "key": "Belien2012",
        "citation": (
            "Beliën, J. & Forcé, H. (2012). Supply chain management of blood "
            "products: A literature review. European Journal of Operational "
            "Research, 217(1), 1-16."
        ),
        "url": "https://doi.org/10.1016/j.ejor.2011.05.026",
        "used_for": (
            "Service level benchmarks, inventory policy parameters, "
            "wastage rate standards."
        ),
    },
    {
        "key": "Katsaliaki2007",
        "citation": (
            "Katsaliaki, K. & Brailsford, S.C. (2007). Using simulation to "
            "improve the blood supply chain. Journal of the Operational "
            "Research Society, 58(2), 219-227."
        ),
        "url": "https://doi.org/10.1057/palgrave.jors.2602195",
        "used_for": "Shortage rate benchmarks in simulation studies.",
    },
    {
        "key": "Rajendran2017",
        "citation": (
            "Rajendran, S. & Ravindran, A.R. (2017). Platelet ordering "
            "policies at hospitals using stochastic integer programming model "
            "and heuristic approaches to reduce wastage. Computers & "
            "Industrial Engineering, 110, 151-164."
        ),
        "url": "https://doi.org/10.1016/j.cie.2017.05.021",
        "used_for": "Platelet wastage benchmarks, shortage targets.",
    },
    {
        "key": "Stanger2012",
        "citation": (
            "Stanger, S.H.W. et al. (2012). What drives perishable inventory "
            "management performance? Lessons learnt from the UK blood supply "
            "chain. Supply Chain Management, 17(2), 107-123."
        ),
        "url": "https://doi.org/10.1108/13598541211212861",
        "used_for": "Wastage benchmarks, inventory days of supply.",
    },
    {
        "key": "Custer2004",
        "citation": (
            "Custer, B. et al. (2004). Demographics of successful, "
            "unsuccessful and deferral visits at six blood centers over a "
            "4-year period. Transfusion, 44(5), 739-744."
        ),
        "url": "https://doi.org/10.1111/j.1537-2995.2004.03332.x",
        "used_for": "Deferral rates by cause.",
    },
    {
        "key": "Heddle2016",
        "citation": (
            "Heddle, N.M. et al. (2016). Effect of short-term vs. long-term "
            "blood storage on mortality after transfusion. New England "
            "Journal of Medicine, 375(20), 1937-1945."
        ),
        "url": "https://doi.org/10.1056/NEJMoa1609014",
        "used_for": "RBC age at transfusion benchmarks (INFORM trial).",
    },
    {
        "key": "Lacroix2015",
        "citation": (
            "Lacroix, J. et al. (2015). Age of transfused blood in critically "
            "ill adults. New England Journal of Medicine, 372(15), 1410-1418."
        ),
        "url": "https://doi.org/10.1056/NEJMoa1500704",
        "used_for": "RBC age at transfusion benchmarks (ABLE trial).",
    },
    {
        "key": "Hebert1999",
        "citation": (
            "Hébert, P.C. et al. (1999). A multicenter, randomized, "
            "controlled clinical trial of transfusion requirements in "
            "critical care. New England Journal of Medicine, 340(6), 409-417."
        ),
        "url": "https://doi.org/10.1056/NEJM199902113400601",
        "used_for": "RBC transfusion trigger thresholds (TRICC trial).",
    },
    {
        "key": "Williamson2013",
        "citation": (
            "Williamson, L.M. & Devine, D.V. (2013). Challenges in the "
            "management of the blood supply. The Lancet, 381(9880), 1866-1875."
        ),
        "url": "https://doi.org/10.1016/S0140-6736(13)60631-5",
        "used_for": "Platelet wastage rates, supply chain challenges.",
    },
    {
        "key": "Godin2007",
        "citation": (
            "Godin, G. et al. (2007). Factors explaining the intention to "
            "give blood: the experience of the Quebec Blood Donor Clinic. "
            "Transfusion, 47(4), 733-739."
        ),
        "url": "https://doi.org/10.1111/j.1537-2995.2007.01176.x",
        "used_for": ("Quebec-specific donor behaviour, show-up rates, motivation."),
    },
    {
        "key": "Greinacher2011",
        "citation": (
            "Greinacher, A. et al. (2011). Implications of demographics on "
            "future blood supply: a population-based cross-sectional study. "
            "Transfusion, 51(4), 702-709."
        ),
        "url": "https://doi.org/10.1111/j.1537-2995.2010.02882.x",
        "used_for": "Donor demographics, aging donor base, participation rates.",
    },
    {
        "key": "Goldman2005",
        "citation": (
            "Goldman, M. et al. (2005). ABO and D groups in the Canadian "
            "population. Transfusion, 45(7), 1149-1152."
        ),
        "url": "https://doi.org/10.1111/j.1537-2995.2005.04171.x",
        "used_for": "Canadian blood type distribution data.",
    },
    {
        "key": "Dzik2011",
        "citation": (
            "Dzik, W.H. et al. (2011). Factors affecting red blood cell "
            "storage age at the time of transfusion. Transfusion, 51(4), "
            "768-775."
        ),
        "url": "https://doi.org/10.1111/j.1537-2995.2010.02901.x",
        "used_for": "RBC inventory age, emergency protocols.",
    },
    {
        "key": "Leahy2017",
        "citation": (
            "Leahy, M.F. et al. (2017). Improved outcomes and reduced costs "
            "associated with a health-system-wide patient blood management "
            "program. Transfusion, 57(6), 1347-1358."
        ),
        "url": "https://doi.org/10.1111/trf.14006",
        "used_for": "Patient blood management impact on demand reduction.",
    },
    {
        "key": "AABB2020",
        "citation": (
            "AABB (2020). Technical Manual, 20th Edition. Bethesda, MD: AABB."
        ),
        "url": "https://www.aabb.org/aabb-store/product/technical-manual-20th-edition",
        "used_for": (
            "Shelf life standards, processing times, compatibility rules, "
            "component preparation."
        ),
    },
    {
        "key": "Flint2020",
        "citation": (
            "Flint, A. et al. (2020). Platelet utilization and wastage in "
            "Canada. Transfusion, 60(S1), 14A."
        ),
        "url": "https://doi.org/10.1111/trf.15713",
        "used_for": "Canadian platelet outdating rates.",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print_benchmark_report()

    # Quick sanity check: verify calibration.py values against benchmarks
    try:
        from calibration import (
            HEMA_ANNUAL_COMPLETED_DONOR_EVENTS_2024_2025,
            HEMA_ANNUAL_LABILE_PRODUCTS_2024_2025,
            HEMA_DAILY_HOSPITAL_ORDERS,
            QC_DAILY_COMPLETED_DONATIONS_EST,
        )

        print("\n" + "=" * 80)
        print("  CALIBRATION.PY CROSS-CHECK")
        print("=" * 80)

        checks = [
            (
                "Héma-Québec annual completed whole-blood donations",
                HEMA_ANNUAL_COMPLETED_DONOR_EVENTS_2024_2025,
            ),
            (
                "Héma-Québec annual labile blood products distributed",
                HEMA_ANNUAL_LABILE_PRODUCTS_2024_2025,
            ),
            (
                "Héma-Québec daily hospital orders (province-wide)",
                HEMA_DAILY_HOSPITAL_ORDERS,
            ),
            (
                "Quebec City CMA daily completed donations (estimated)",
                QC_DAILY_COMPLETED_DONATIONS_EST,
            ),
        ]

        for metric_query, value in checks:
            result = validate_metric(metric_query, value, tolerance_pct=15.0)
            status = "✓" if result["passed"] else "✗"
            print(f"  {status} {metric_query}: {value:.1f}")
            for bm, ok, dev in result["matched"]:
                print(f"    → {bm.range_str}  (dev: {dev:+.1f}%)")

    except ImportError:
        print("\n  [SKIP] Could not import calibration.py for cross-check.")
