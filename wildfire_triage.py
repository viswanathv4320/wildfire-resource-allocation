"""
wildfire_triage.py
Wildfire Resource Allocation Model — Risk Scoring + IP Optimizer + Spread Simulation

Final model components:
  - AHP-derived risk scoring with absolute weather normalization
  - Planning-horizon resource-hour MILP optimizer
  - Cost plus risk-weighted residual damage objective
  - Terrain/access-adjusted ground resource effectiveness
  - Nonlinear size-effectiveness multiplier for large fires
  - Elliptical spread footprint for OSM asset exposure scoring
  - Sensitivity analysis for budget, planning horizon, and risk preference

Usage:
    python wildfire_triage.py

Requires:
    pip install pandas numpy pulp networkx
"""

import pandas as pd
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
from pulp import *


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

FIRES_CSV     = "wildfire_data/scenario_fires.csv"
RESOURCES_CSV = "wildfire_data/resources.csv"
# Budget calibration: with 1 air tanker ($40k), 3 helicopters ($90k), and scarce ground resources,
# $150k creates genuine prioritization tension — not every fire can receive aviation.
DAILY_BUDGET  = 950_000

# Optimizer: Planning-horizon formulation
PLANNING_HORIZON_HOURS  = 6      # hours — replaces "daily" framing in run_resource_hour_optimizer
HORIZON_BUDGET          = 950_000  # budget for the planning horizon (not per day)

# Optimizer: Terrain effectiveness
SLOPE_PENALTY            = 0.60   # fraction lost at 100% slope (ground resources)
ROAD_PENALTY             = 0.30   # fraction lost at max road distance
MAX_ROAD_KM              = 25.0   # road distance anchor
MIN_GROUND_EFFECTIVENESS = 0.15   # floor: ground resources never fully useless

# Layer 3 — spread simulation
GRID_SIZE   = 40      # cells per side (40×40 = 4km × 4km)
CELL_M      = 100     # metres per cell
HOUR_SECS   = 3600
CHECK_HOURS = [0, 3, 6, 9, 12]

# A: Suppression feedback — resources slow fire spread
# v_effective = v_base / (1 + ALPHA_SUPPRESSION * S_f)
# S_f = suppression_strength = sum(units * suppression_power) at fire f
# Higher ALPHA → resources have stronger effect on spread rate
ALPHA_SUPPRESSION = 0.15

# Suppression power per resource type (unitless effectiveness score)
# Aerial resources suppress large fronts; hand crews/engines hold lines
SUPPRESSION_POWER = {
    "Type-1 Engine"       : 1.0,
    "Heavy Dozer"         : 2.0,
    "Type-1 Helicopter"   : 3.5,
    "Air Tanker"          : 5.0,
    "Hand Crew (20-person)": 1.5,
}

# B: Dynamic weather — deterministic diurnal profiles
# Based on typical fire weather day: peak danger ~14:00 local time
# Assumes t=0 represents ~08:00 (morning briefing), peak at t=6h (14:00)
# This is more defensible than random walk: same fire, same day, same trend
DIURNAL_PEAK_HOUR   = 6.0    # hours after t=0 when conditions peak (14:00 if t=0=08:00)
DIURNAL_WIND_AMP    = 0.25   # max wind amplification at peak (±25% of base)
DIURNAL_RH_DROP     = 20.0   # max RH drop at peak (percentage points)
DIURNAL_TEMP_RISE   = 8.0    # max temp rise at peak (°C)
# Wind direction diurnal shift: upslope/onshore during day, backing at night
DIURNAL_WDIR_SHIFT  = 15.0   # degrees clockwise at peak

# C: Scenario analysis (NOT uncertainty quantification — perturbations are assumed,
#    not learned from historical data. Present as "plausible scenario range".)
N_ENSEMBLE = 20   # simulations per scenario (fast; Dijkstra is cheap)
ENSEMBLE_SCENARIOS = {
    "optimistic": {"wind_scale": 0.85, "humidity_delta": +8,  "fuel_scale": 0.80},
    "expected"  : {"wind_scale": 1.00, "humidity_delta":  0,  "fuel_scale": 1.00},
    "worst_case": {"wind_scale": 1.15, "humidity_delta": -8,  "fuel_scale": 1.20},
}
# Per-run perturbation within each scenario (represents local variability, not full UQ)
ENSEMBLE_WIND_NOISE    = 0.10   # ±10% wind noise within scenario
ENSEMBLE_HUMIDITY_NOISE= 5.0    # ±5% RH noise within scenario
ENSEMBLE_FUEL_NOISE    = 0.15   # ±15% fuel multiplier noise within scenario

# D: Terrain / slope effects
# v_slope = v * (1 + BETA_SLOPE * slope_grade)
# slope_grade: positive = uphill (faster), negative = downhill (slower)
# Synthetic Gaussian hill centred on grid — configurable peak elevation
BETA_SLOPE       = 0.5    # slope sensitivity coefficient
TERRAIN_PEAK_M   = 300.0  # synthetic hill peak elevation (metres)
TERRAIN_SIGMA    = 12.0   # hill spread in grid cells (12 cells = 1.2 km radius)

# Layer 4 — OSM asset fetch radius (separate from spread grid to avoid overwrite bug)
# Previously half_m was set to CELL_M*centre then immediately overwritten to 20_000
ASSET_FETCH_RADIUS_M = 20_000   # 20 km radius for OSM asset queries

# NOAA absolute meteorological thresholds for normalization
# Source: NOAA Red Flag Warning criteria + fire weather operational standards
WIND_MAX_MPS  = 20.0   # ~45 mph — catastrophic fire wind threshold
TEMP_MIN_C    =  0.0   # lower anchor (cool/wet conditions)
TEMP_MAX_C    = 45.0   # upper anchor (extreme heat)

# Nonlinear weather risk parameters
# Humidity: exponential decay — low RH increases risk faster than linearly
# risk = (exp(-K_h*RH) - exp(-K_h*100)) / (1 - exp(-K_h*100))
K_HUMIDITY    = 0.03

# Wind: exponential growth — high wind increases risk faster than linearly
# risk = (exp(K_w*W_kmh) - 1) / (exp(K_w*W_max_kmh) - 1)
K_WIND        = 0.05
WIND_MAX_KMH  = WIND_MAX_MPS * 3.6   # ~72 km/h

# Weather sub-weights (updated: wind dominates short-term escalation)
WEATHER_W_HUMIDITY = 0.30
WEATHER_W_WIND     = 0.45
WEATHER_W_TEMP     = 0.25

# Damage tradeoff parameter (lambda)
# Controls weight of residual damage vs suppression cost in objective
# Higher = care more about leaving fire uncontrolled
# Calibrated so that at full budget, solver still prefers covering high-risk fires
LAMBDA_DAMAGE = 50.0

# Fix 2: Objective dimensional calibration
# Suppression cost is in $/day. Residual damage needs to be in comparable dollar terms.
# Calibration: USFS/NIFC wildfire damage estimates average ~$200-800/acre for structure
# exposure + suppression difficulty. We use $500/acre as a mid-range anchor.
# residual_damage[f] is then dimensioned as:
#   danger[f]/100 (fraction of worst-case)
#   × asset[f]/10 (fraction of highest-exposure fire)
#   × uncovered[f] (acres)
#   × DAMAGE_COST_PER_ACRE ($/acre)
# = dollars of expected unmitigated damage per acre left uncovered.
# LAMBDA then acts as a pure preference multiplier (dimensionless), not a unit converter.
# LAMBDA=1 → treat uncovered damage at face value; >1 → risk-averse.
DAMAGE_COST_PER_ACRE = 500.0   # $/acre — calibrated to USFS average wildfire damage cost


# ════════════════════════════════════════════════════════════════════════════
# LAYER 0: AHP WEIGHT DERIVATION
# ════════════════════════════════════════════════════════════════════════════

def compute_ahp_weights(verbose: bool = True) -> dict:
    """
    Derive risk component weights using the Analytic Hierarchy Process (Saaty 1980).

    Pairwise comparison matrix justified by:
      - NWCG initial attack prioritization doctrine
      - NOAA Red Flag Warning operational criteria
      - Granda et al. (2023) literature review findings

    Saaty scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme importance.

    Matrix interpretation (row vs column):
      Behavior vs Size     = 4  (behavior directly drives spread rate; size is lagging)
      Behavior vs Weather  = 2  (behavior is the observed outcome of weather)
      Behavior vs Complexity = 3 (complexity is a lagging indicator of past behavior)
      Weather  vs Size     = 3  (small fire + extreme weather > large fire + benign weather)
      Weather  vs Complexity = 2 (weather is forward-looking; complexity backward-looking)
      Complexity vs Size   = 2  (Type 1 implies escaped initial attack — more informative)

    Consistency Ratio (CR) must be < 0.10 to be acceptable.
    """
    criteria = ["Size", "Weather", "Behavior", "Complexity"]

    A = np.array([
        [1,    1/3,  1/4,  1/2],   # Size
        [3,    1,    1/2,  2  ],   # Weather
        [4,    2,    1,    3  ],   # Behavior
        [2,    1/2,  1/3,  1  ],   # Complexity
    ], dtype=float)

    # Normalize columns
    col_sums = A.sum(axis=0)
    A_norm   = A / col_sums

    # Row average = weight vector
    weights  = A_norm.mean(axis=1)

    # Consistency check
    lam_max  = (A @ weights / weights).mean()
    n        = len(criteria)
    CI       = (lam_max - n) / (n - 1)
    RI       = 0.90    # Saaty's Random Index for n=4
    CR       = CI / RI

    if verbose:
        print("\n── AHP Weight Derivation ──────────────────────────────────")
        print("  Pairwise comparison matrix (Saaty scale 1-9):")
        header = f"  {'':12}" + "".join(f"{c:>12}" for c in criteria)
        print(header)
        for i, row_label in enumerate(criteria):
            vals = "".join(
                f"{A[i,j]:>12.3f}" if A[i,j] < 1 else f"{int(A[i,j]):>12d}"
                for j in range(n)
            )
            print(f"  {row_label:<12}{vals}")
        print()
        for c, w in zip(criteria, weights):
            print(f"  {c:<12}: {w:.4f}  ({w*100:.1f}%)")
        print(f"\n  λ_max = {lam_max:.4f}")
        print(f"  CR    = {CR:.4f}  "
              f"({'✓ Consistent (< 0.10)' if CR < 0.10 else '✗ Revise matrix'})")

    return {
        "size"      : weights[0],
        "weather"   : weights[1],
        "behavior"  : weights[2],
        "complexity": weights[3],
        "CR"        : CR,
    }


# ════════════════════════════════════════════════════════════════════════════
# LAYER 1: RISK SCORING  (pure fire danger — no asset score here)
# ════════════════════════════════════════════════════════════════════════════

def compute_risk_scores(fires: pd.DataFrame,
                        weights: dict) -> pd.DataFrame:
    """
    Compute a composite fire danger score (0-100) for each fire.
    Asset value is intentionally excluded here — it enters only in the
    IP objective to avoid double-counting infrastructure exposure.

    Components and normalization:
      size_score       — log-normalized discovery acreage (relative to batch)
      weather_score    — absolute scale anchored to NOAA Red Flag thresholds
      behavior_score   — NIFC FireBehaviorGeneral ordinal mapping
      complexity_score — NIFC FireMgmtComplexity ordinal mapping

    Weights from AHP (compute_ahp_weights).
    """
    df = fires.copy()

    df["wind_speed_mps"] = df["wind_speed_mps"].fillna(3.0)
    df["wind_dir_deg"]   = df["wind_dir_deg"].fillna(270.0)
    df["temperature_c"]  = df["temperature_c"].fillna(15.0)

    # ── Size: log-normalized within batch (relative sizing still valid)
    # Use same demand hierarchy as optimizer for size scoring
    if "incident_size_6h" in df.columns and df["incident_size_6h"].notna().any():
        size_basis = df["incident_size_6h"].fillna(df["discovery_acres"])
    elif "incident_size" in df.columns and df["incident_size"].notna().any():
        size_basis = df["incident_size"].fillna(df["discovery_acres"])
    elif "current_acres" in df.columns:
        size_basis = df["current_acres"]
    else:
        size_basis = df["discovery_acres"]
    df["size_score"] = np.log1p(size_basis)
    df["size_score"] = (df["size_score"] / df["size_score"].max()).clip(0, 1)

    # ── Weather: nonlinear scaling anchored to NOAA Red Flag thresholds
    #
    # Humidity: exponential decay — risk rises steeply as RH drops below ~30%
    #   normalized: (exp(-K_h*RH) - exp(-K_h*100)) / (1 - exp(-K_h*100))
    rh = df["humidity_pct"].clip(0, 100)
    denom_h = 1.0 - np.exp(-K_HUMIDITY * 100)
    df["humidity_risk"] = (
        (np.exp(-K_HUMIDITY * rh) - np.exp(-K_HUMIDITY * 100)) / denom_h
    ).clip(0, 1)

    # Wind: exponential growth — risk accelerates at high wind speeds
    #   normalized: (exp(K_w*W_kmh) - 1) / (exp(K_w*W_max_kmh) - 1)
    w_kmh = (df["wind_speed_mps"] * 3.6).clip(0, WIND_MAX_KMH)
    denom_w = np.exp(K_WIND * WIND_MAX_KMH) - 1.0
    df["wind_risk"] = (
        (np.exp(K_WIND * w_kmh) - 1.0) / denom_w
    ).clip(0, 1)

    # Temperature: linear 0°C → 0, 45°C → 1.0 (operationally clean anchors)
    df["temp_risk"] = (
        (df["temperature_c"] - TEMP_MIN_C) / (TEMP_MAX_C - TEMP_MIN_C)
    ).clip(0, 1)

    # Weather sub-weights: wind dominates short-term escalation (updated)
    df["weather_score"] = (
        WEATHER_W_HUMIDITY * df["humidity_risk"] +
        WEATHER_W_WIND     * df["wind_risk"]     +
        WEATHER_W_TEMP     * df["temp_risk"]
    )

    # ── Behavior: NIFC FireBehaviorGeneral → ordinal 0-1
    behavior_map = {
        "Minimal": 0.20, "Moderate": 0.50,
        "Active":  0.80, "Extreme":  1.00,
    }
    df["behavior_score"] = df["fire_behavior"].map(behavior_map).fillna(0.30)

    # ── Complexity: NIFC FireMgmtComplexity → ordinal 0-1
    complexity_map = {
        "Type 1 Incident": 1.00, "Type 2 Incident": 0.75,
        "Type 3 Incident": 0.50, "Type 4 Incident": 0.25,
        "Type 5 Incident": 0.10,
    }
    df["complexity_score"] = df["mgmt_complexity"].map(complexity_map).fillna(0.30)

    # ── Composite fire danger score (AHP weights)
    df["fire_danger"] = (
        weights["size"]       * df["size_score"]       +
        weights["weather"]    * df["weather_score"]    +
        weights["behavior"]   * df["behavior_score"]   +
        weights["complexity"] * df["complexity_score"]
    )

    # Absolute normalization against theoretical maximum.
    # Since all component scores are in [0, 1] and AHP weights sum to 1.0,
    # the theoretical maximum fire_danger is exactly 1.0 (all components = 1).
    # risk_score_100 = fire_danger × 100 — batch-independent and absolute.
    # A fire with risk_score_100 = 75 means 75% of the worst conceivable fire,
    # regardless of how many other fires are in the dataset.
    THEORETICAL_MAX_DANGER = 1.0   # sum(weights) = 1.0, all components clipped to [0,1]
    df["risk_score_100"] = (df["fire_danger"] / THEORETICAL_MAX_DANGER * 100).round(1).clip(0, 100)
    df["priority_rank"]  = df["fire_danger"].rank(ascending=False).astype(int)

    return df


# ════════════════════════════════════════════════════════════════════════════
# LAYER 2: IP OPTIMIZER — cost minimization formulation
# ════════════════════════════════════════════════════════════════════════════

def run_optimizer(fires: pd.DataFrame,
                  resources: pd.DataFrame,
                  asset_scores: dict = None,
                  budget: float = DAILY_BUDGET) -> dict:
    """
    MILP resource allocation with cost-minimization objective.

    Formulation (sets, variables, constraints):

    Sets:
        F = active fires,  R = resource types

    Decision variables:
        x[r,f] ∈ Z≥0   — units of resource r assigned to fire f
        c[f]   ≥ 0      — covered acres at fire f  (explicit, aids explanation)
        u[f]   ≥ 0      — uncovered acres at fire f (slack variable)

    Objective (both terms in $/day equivalent):
        min  Σ_{r,f} C_r·x[r,f]
           + λ · Σ_f (R_f/100) · (A_f/10) · u[f] · K

        C_r = cost per day ($/day)
        R_f = risk score [0,100] / 100 → fraction of worst-case danger
        A_f = asset score [1,10] / 10  → fraction of highest-exposure fire
        K   = DAMAGE_COST_PER_ACRE = $500/acre (USFS calibration)
        λ   = LAMBDA_DAMAGE (dimensionless risk-preference multiplier)
              λ=1: risk-neutral · λ>1: risk-averse

    Constraints:
        C1  supply   : Σ_f x[r,f] ≤ U_r                      ∀ r
        C2  budget   : Σ_{r,f} C_r·x[r,f] ≤ B
        C3  coverage : c[f] = Σ_r q_r·x[r,f]                 ∀ f
        C4  uncovered: u[f] ≥ D_f − c[f]                     ∀ f
        C5  overcap  : c[f] ≤ D_f + max_r(q_r)               ∀ f
            (allows one large-unit overshoot for lumpy resources,
             without the hard-to-justify 200% cap)

    Demand D_f:
        Priority: current_acres (Layer 3 live projection)
                > incident_size  (operational size at scenario selection time)
                > discovery_acres (size at detection — last resort)
        incident_size reflects the fire's known operational footprint when
        scenario fires were selected, giving the IP a realistic workload
        rather than the trivially small detection acreage.
    """
    fire_names     = fires["fire_name"].tolist()
    resource_names = resources["resource"].tolist()

    units_available = dict(zip(resource_names, resources["units_available"]))
    # Support both old (per_day) and new (per_hour) resource CSVs
    # Use explicit per-hour rates — do not derive from per_day/24
    # resources.csv must have cost_per_hour and acres_per_hour columns
    if "cost_per_hour" not in resources.columns:
        raise ValueError(
            "resources.csv must have 'cost_per_hour' and 'acres_per_hour' columns. "
            "Dividing cost_per_day/24 assumes 24 productive hours — unrealistic for aircraft."
        )
    # Convert to per-day equivalents using productive_hours_per_day for reporting only
    prod_hrs = dict(zip(resource_names,
                        resources["productive_hours_per_day"]
                        if "productive_hours_per_day" in resources.columns
                        else [10] * len(resource_names)))
    acres_per_hour = dict(zip(resource_names, resources["acres_per_hour"]))
    cost_per_hour  = dict(zip(resource_names, resources["cost_per_hour"]))
    # Keep acres_per_day/cost_per_day as aliases for downstream reporting
    acres_per_day  = {r: acres_per_hour[r] * prod_hrs[r] for r in resource_names}
    cost_per_day   = {r: cost_per_hour[r]  * prod_hrs[r] for r in resource_names}

    # Demand hierarchy:
    #   1. incident_size_6h — 6h suppression demand (explicit operational workload)
    #   2. incident_size    — operational footprint at scenario selection time
    #   3. current_acres    — Layer 3 dynamic projection (fallback during simulation)
    #   4. discovery_acres  — detection size (last resort; often trivially small)
    #
    # incident_size_6h is prioritised over current_acres because it is the
    # explicitly designed demand figure for the planning horizon, not a live
    # simulation projection that may not exist at scenario-design time.
    if "incident_size_6h" in fires.columns and fires["incident_size_6h"].notna().any():
        demand     = dict(zip(fire_names, fires["incident_size_6h"].fillna(fires["discovery_acres"])))
        demand_src = "incident_size_6h (6h suppression demand)"
    elif "incident_size" in fires.columns and fires["incident_size"].notna().any():
        demand     = dict(zip(fire_names, fires["incident_size"].fillna(fires["discovery_acres"])))
        demand_src = "incident_size"
    elif "current_acres" in fires.columns:
        demand     = dict(zip(fire_names, fires["current_acres"]))
        demand_src = "current_acres"
    else:
        demand     = dict(zip(fire_names, fires["discovery_acres"]))
        demand_src = "discovery_acres"

    danger = dict(zip(fires["fire_name"], fires["risk_score_100"]))

    # Default asset score: 5.0 (mid-scale neutral) when OSM data is missing.
    # 1.0 would mean "lowest exposure" — understating residual damage.
    # 5.0 → A_f/10 = 0.5, which is a neutral assumption.
    asset  = asset_scores or {f: 5.0 for f in fire_names}

    # Largest single-resource coverage per day — used for overcoverage cap
    max_unit_coverage = max(acres_per_day.values())

    model = LpProblem("Wildfire_Triage_Daily", LpMinimize)

    # x[r,f]: integer units dispatched
    x = {
        (r, f): LpVariable(
            f"x_{r.replace(' ','_').replace('-','_')}_{f.replace(' ','_')}",
            lowBound=0, cat="Integer"
        )
        for r in resource_names for f in fire_names
    }

    # c[f]: explicit coverage variable — aids explanation and diagnostics
    c = {
        f: LpVariable(f"cov_{f.replace(' ','_')}", lowBound=0)
        for f in fire_names
    }

    # u[f]: uncovered acres (slack) — enters residual damage objective
    u = {
        f: LpVariable(f"uncov_{f.replace(' ','_')}", lowBound=0)
        for f in fire_names
    }

    suppression_cost = lpSum(
        cost_per_day[r] * x[(r, f)]
        for r in resource_names for f in fire_names
    )
    residual_damage = lpSum(
        LAMBDA_DAMAGE
        * (danger[f] / 100.0)
        * (asset.get(f, 5.0) / 10.0)
        * u[f]
        * DAMAGE_COST_PER_ACRE
        for f in fire_names
    )
    model += suppression_cost + residual_damage

    # C1: supply limits
    for r in resource_names:
        model += (
            lpSum(x[(r, f)] for f in fire_names) <= units_available[r],
            f"supply_{r.replace(' ','_')}"
        )

    # C2: budget
    model += (
        lpSum(cost_per_day[r] * x[(r, f)]
              for r in resource_names for f in fire_names) <= budget,
        "budget"
    )

    # C3: coverage definition (explicit variable)
    for f in fire_names:
        model += (
            c[f] == lpSum(acres_per_day[r] * x[(r, f)] for r in resource_names),
            f"coverage_def_{f.replace(' ','_')}"
        )

    # C4: uncovered = demand - coverage
    for f in fire_names:
        model += (
            u[f] >= demand[f] - c[f],
            f"uncovered_{f.replace(' ','_')}"
        )

    # C5: overcoverage limit — demand + largest single resource unit
    # More defensible than 2×demand: allows one lumpy unit overshoot,
    # prevents arbitrarily large over-allocation on small fires.
    for f in fire_names:
        model += (
            c[f] <= demand[f] + max_unit_coverage,
            f"overcap_{f.replace(' ','_')}"
        )

    model.solve(PULP_CBC_CMD(msg=0))

    allocation = {
        (r, f): int(value(x[(r, f)]) or 0)
        for r in resource_names for f in fire_names
    }
    coverage_vals = {f: float(value(c[f]) or 0) for f in fire_names}
    uncovered     = {f: float(value(u[f]) or 0) for f in fire_names}

    return {
        "status"        : LpStatus[model.status],
        "objective"     : value(model.objective),
        "allocation"    : allocation,
        "coverage"      : coverage_vals,   # explicit c[f] values
        "uncovered"     : uncovered,
        "fire_names"    : fire_names,
        "resource_names": resource_names,
        "acres_per_day" : acres_per_day,
        "cost_per_day"  : cost_per_day,
        "demand"        : demand,
        "risk_scores"   : danger,
        "asset_scores"  : asset,
    }


# ════════════════════════════════════════════════════════════════════════════
# REPORTING
# ════════════════════════════════════════════════════════════════════════════

def print_risk_report(fires: pd.DataFrame, weights: dict):
    print("\n" + "=" * 65)
    print("  LAYER 1 — RISK SCORING  (AHP weights, absolute normalization)")
    print("=" * 65)
    print(f"\n  Weights: Size={weights['size']:.3f}  Weather={weights['weather']:.3f}"
          f"  Behavior={weights['behavior']:.3f}  Complexity={weights['complexity']:.3f}")
    print(f"  CR={weights['CR']:.4f} ({'✓' if weights['CR'] < 0.10 else '✗'})\n")

    cols = ["fire_name", "discovery_acres", "humidity_pct", "wind_speed_mps",
            "temperature_c", "fire_behavior", "mgmt_complexity",
            "weather_score", "behavior_score", "risk_score_100", "priority_rank"]
    pd.set_option("display.float_format", "{:.2f}".format)
    print(fires[cols].sort_values("priority_rank").to_string(index=False))

    print("\n── Priority ranking ─────────────────────────────────────────")
    for _, row in fires.sort_values("priority_rank").iterrows():
        print(f"  #{int(row['priority_rank'])}  {row['fire_name']:<20}  "
              f"Risk={row['risk_score_100']:5.1f}/100  |  "
              f"behavior={row['fire_behavior']}  "
              f"humidity={row['humidity_pct']:.1f}%  "
              f"wind={row['wind_speed_mps']:.1f}m/s  "
              f"acres={row['discovery_acres']:.0f}")


def print_allocation_report(result: dict, fires: pd.DataFrame,
                             resources: pd.DataFrame):
    print("\n" + "=" * 65)
    print(f"  LAYER 2 — IP OPTIMIZER  [{result['status']}]")
    print(f"  Objective (suppress cost + λ×residual damage): "
          f"{result['objective']:,.0f}")
    print("=" * 65)

    fire_names     = result["fire_names"]
    resource_names = result["resource_names"]
    allocation     = result["allocation"]
    acres_per_day  = result["acres_per_day"]
    cost_per_day   = result["cost_per_day"]
    demand         = result["demand"]
    danger         = result["risk_scores"]
    asset          = result["asset_scores"]
    uncovered      = result["uncovered"]
    coverage_var   = result.get("coverage", {})  # explicit c[f] LP variable values

    # Show which demand source was used
    if "current_acres" in fires.columns:
        demand_src = "current_acres (Layer 3 projection)"
    elif "incident_size" in fires.columns and fires["incident_size"].notna().any():
        demand_src = "incident_size (operational footprint)"
    else:
        demand_src = "discovery_acres (detection size)"
    print(f"  Demand source    : {demand_src}")

    print("\n── Units dispatched ─────────────────────────────────────────")
    alloc_df = pd.DataFrame(index=resource_names, columns=fire_names, data=0)
    for (r, f), v in allocation.items():
        alloc_df.loc[r, f] = v
    print(alloc_df.to_string())

    print("\n── Allocation summary ───────────────────────────────────────")
    rank_map   = dict(zip(fires["fire_name"], fires["priority_rank"]))
    total_cost = 0

    for f in sorted(fire_names, key=lambda f: rank_map[f]):
        units    = {r: allocation[(r, f)] for r in resource_names}
        # Use explicit c[f] LP variable value if available, else recompute
        coverage = coverage_var.get(f, sum(units[r] * acres_per_day[r] for r in resource_names))
        fcost    = sum(units[r] * cost_per_day[r]  for r in resource_names)
        total_cost += fcost
        pct      = min(coverage / demand[f] * 100, 100) if demand[f] > 0 else 0
        res_str  = ", ".join(f"{v} {r}" for r, v in units.items() if v > 0)
        dmg_dollar = (LAMBDA_DAMAGE * (danger[f]/100.0) *
                      (asset.get(f, 5.0)/10.0) * uncovered[f] * DAMAGE_COST_PER_ACRE)

        print(f"\n  #{rank_map[f]}  {f}  "
              f"(danger={danger[f]:.1f}/100  asset={asset.get(f,5.0):.1f}/10)")
        print(f"      Demand (D_f)  : {demand[f]:,.0f} ac  [{demand_src}]")
        print(f"      Coverage c[f] : {coverage:,.0f} ac  ({pct:.0f}%)")
        print(f"      Uncovered u[f]: {uncovered[f]:,.0f} ac")
        print(f"      Residual dmg  : ${dmg_dollar:,.0f}/day equiv  "
              f"(λ × danger/100 × asset/10 × uncov × $500/ac)")
        print(f"      Suppress cost : ${fcost:,.0f}/day")
        print(f"      Resources     : {res_str or 'none'}")

    total_units = sum(allocation.values())
    print(f"\n  ── Totals ──────────────────────────────────────────────")
    print(f"  Suppression cost : ${total_cost:>10,.0f}  (budget=${DAILY_BUDGET:,})")
    print(f"  Units deployed   : {total_units} of "
          f"{int(resources['units_available'].sum())} available")


# ════════════════════════════════════════════════════════════════════════════
# LAYER 3: FIRE SPREAD SIMULATION + DYNAMIC REALLOCATION
# ════════════════════════════════════════════════════════════════════════════

# Fuel-type spread multipliers
# Source: relative rate-of-spread by fuel category (Rothermel-inspired)
# grass/shrub spreads fastest, timber is baseline, nonburnable/urban slows spread.
# If fuel_group is missing, behavior is used to infer a reasonable proxy.
FUEL_SPREAD_MULTIPLIER = {
    # NIFC PredominantFuelGroup values
    "grass"        : 1.6,
    "grass/shrub"  : 1.4,
    "shrub"        : 1.3,
    "timber litter": 1.0,
    "timber"       : 1.0,
    "slash"        : 0.9,
    "nonburnable"  : 0.3,
    "urban"        : 0.4,
    "agriculture"  : 0.7,
    "water"        : 0.1,
    # Fallback by behavior when fuel_group is absent
    "_behavior_extreme"  : 1.5,
    "_behavior_active"   : 1.2,
    "_behavior_moderate" : 1.0,
    "_behavior_minimal"  : 0.7,
}

def get_fuel_multiplier(fuel_group: str, fire_behavior: str) -> float:
    """
    Return a spread rate multiplier based on fuel type.
    Falls back to fire behavior when fuel_group is missing or unrecognized.
    """
    if fuel_group and pd.notna(fuel_group):
        key = str(fuel_group).strip().lower()
        for k, v in FUEL_SPREAD_MULTIPLIER.items():
            if k.startswith("_"):
                continue
            if key == k or key.startswith(k):
                return v
    # Behavior fallback
    beh = str(fire_behavior or "").strip().lower()
    if   "extreme"  in beh: return FUEL_SPREAD_MULTIPLIER["_behavior_extreme"]
    elif "active"   in beh: return FUEL_SPREAD_MULTIPLIER["_behavior_active"]
    elif "moderate" in beh: return FUEL_SPREAD_MULTIPLIER["_behavior_moderate"]
    else:                   return FUEL_SPREAD_MULTIPLIER["_behavior_minimal"]


def fetch_srtm_elevation(fire_lat: float, fire_lon: float,
                          grid_size: int = GRID_SIZE,
                          cell_m: float = CELL_M) -> np.ndarray:
    """
    D: Fetch real SRTM elevation data via Open Elevation API (free, no key needed).
    Falls back to synthetic Gaussian hill if API is unavailable.

    API: https://api.open-elevation.com/api/v1/lookup
    Resolution: ~30m SRTM. Grid is sampled at CELL_M spacing.

    Returns elevation[row, col] in metres, shape (grid_size, grid_size).
    """
    LAT_PER_M = 1 / 111_320
    lon_per_m = 1 / (111_320 * np.cos(np.radians(fire_lat)))
    centre    = grid_size // 2

    # Build list of (lat, lon) for each grid cell
    locations = []
    for r in range(grid_size):
        for c in range(grid_size):
            dr = r - centre
            dc = c - centre
            lat = fire_lat - dr * cell_m * LAT_PER_M
            lon = fire_lon + dc * cell_m * lon_per_m
            locations.append({"latitude": round(lat, 6), "longitude": round(lon, 6)})

    try:
        import requests as req_lib
        resp = req_lib.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": locations},
            timeout=20,
        )
        resp.raise_for_status()
        results  = resp.json()["results"]
        elev_arr = np.array([r["elevation"] for r in results],
                             dtype=float).reshape(grid_size, grid_size)
        print(f"    SRTM elevation fetched: min={elev_arr.min():.0f}m  "
              f"max={elev_arr.max():.0f}m  range={elev_arr.max()-elev_arr.min():.0f}m")
        return elev_arr
    except Exception as e:
        print(f"    SRTM fetch failed ({e}) — using synthetic Gaussian terrain")
        return None   # caller will fall back to synthetic


def build_slope_field(grid_size: int = GRID_SIZE,
                      peak_m: float = TERRAIN_PEAK_M,
                      sigma: float = TERRAIN_SIGMA,
                      peak_row: float = None,
                      peak_col: float = None,
                      fire_lat: float = None,
                      fire_lon: float = None,
                      use_srtm: bool = True) -> np.ndarray:
    """
    D: Elevation field for slope effect on fire spread.

    Priority order:
      1. SRTM fetch via Open Elevation API (if fire_lat/lon provided and use_srtm=True)
      2. Synthetic Gaussian hill (always available as fallback)

    Synthetic hill: offset from ignition centre to create asymmetric spread.
    SRTM: real terrain at ~30m resolution, resampled to CELL_M grid spacing.
    """
    if fire_lat is not None and fire_lon is not None and use_srtm:
        elev = fetch_srtm_elevation(fire_lat, fire_lon, grid_size)
        if elev is not None:
            return elev

    # Synthetic fallback
    if peak_row is None:
        peak_row = grid_size * 0.35
    if peak_col is None:
        peak_col = grid_size * 0.45

    r_idx = np.arange(grid_size)
    c_idx = np.arange(grid_size)
    rr, cc = np.meshgrid(r_idx, c_idx, indexing="ij")
    elev = peak_m * np.exp(
        -((rr - peak_row)**2 + (cc - peak_col)**2) / (2 * sigma**2)
    )
    return elev


def get_weather_at_t(fire_row: pd.Series, t_hours: float,
                     rng: np.random.Generator = None) -> dict:
    """
    B: Deterministic diurnal weather profile at time t_hours.

    Physical basis:
      - Fire weather follows a well-known diurnal cycle: conditions worsen
        from morning to early afternoon, then recover through evening.
      - Peak danger typically occurs ~14:00 local (t=6h if t=0 = 08:00).
      - Model uses a sinusoidal envelope scaled by amplitude constants.

    Variables:
      wind_speed: peaks at DIURNAL_PEAK_HOUR, ±DIURNAL_WIND_AMP × base
      humidity:   minimum at DIURNAL_PEAK_HOUR, drops DIURNAL_RH_DROP pts
      temperature:rises DIURNAL_TEMP_RISE°C at peak, returns at t=12h
      wind_dir:   backs DIURNAL_WDIR_SHIFT° clockwise at peak (thermal effect)

    More defensible than random walk: same fire conditions each run, deterministic.
    rng parameter retained for compatibility but unused.
    """
    base_wind = float(fire_row.get("wind_speed_mps") or 3.0)
    base_rh   = float(fire_row.get("humidity_pct")   or 50.0)
    base_temp = float(fire_row.get("temperature_c")  or 15.0)
    base_wdir = float(fire_row.get("wind_dir_deg")   or 270.0)

    # Sinusoidal diurnal phase: 0 at t=0, peaks at DIURNAL_PEAK_HOUR, back to 0 at t=12h
    # sin(π × t / (2 × peak)) rises to 1 at peak, then sin(π × (12-t)/(2×(12-peak)))
    if t_hours <= DIURNAL_PEAK_HOUR:
        phase = np.sin(np.pi * t_hours / (2.0 * DIURNAL_PEAK_HOUR))
    else:
        phase = np.sin(np.pi * (12.0 - t_hours) / (2.0 * (12.0 - DIURNAL_PEAK_HOUR)))
    phase = float(np.clip(phase, 0.0, 1.0))

    wind_t = float(np.clip(base_wind * (1.0 + DIURNAL_WIND_AMP * phase),
                            0.5, WIND_MAX_MPS))
    rh_t   = float(np.clip(base_rh - DIURNAL_RH_DROP * phase, 5.0, 100.0))
    temp_t = float(np.clip(base_temp + DIURNAL_TEMP_RISE * phase,
                            TEMP_MIN_C, TEMP_MAX_C))
    wdir_t = (base_wdir + DIURNAL_WDIR_SHIFT * phase) % 360

    return {
        "wind_speed_mps": wind_t,
        "wind_dir_deg"  : wdir_t,
        "humidity_pct"  : rh_t,
        "temperature_c" : temp_t,
        "_diurnal_phase": phase,   # stored for reporting
    }


def compute_suppression_strength(fire_name: str,
                                  allocation: dict,
                                  resource_names: list) -> float:
    """
    Compute total suppression strength S_f (used for reporting only).
    Actual spread effect is now sectoral via get_suppression_sectors().
    """
    S_f = 0.0
    for r in resource_names:
        units = int(allocation.get((r, fire_name), 0) or 0)
        if units == 0:
            continue
        power = 1.0
        r_lower = r.lower()
        for key, val in SUPPRESSION_POWER.items():
            if key.lower() in r_lower or r_lower in key.lower():
                power = val
                break
        S_f += units * power
    return S_f


def get_suppression_sectors(fire_name: str,
                              allocation: dict,
                              resource_names: list,
                              wind_to_deg: float) -> dict:
    """
    A (sectoral): Map each resource type to the grid sector it protects.

    Operational logic:
      Air Tanker      → head fire sector (downwind, ±45° of spread direction)
                        Retardant drops slow the advancing front.
      Type-1 Helicopter→ flanks (±90°–135° from spread direction)
                        Recon + water drops on flanking runs.
      Heavy Dozer     → one flank (left flank, 90°–180° from head)
                        Dozers build line; assigned to one side.
      Hand Crew       → backfire/heel sector (upwind, ±60° opposite spread)
                        Crews hold the heel and burn out from anchor points.
      Type-1 Engine   → structure/corridor protection (all sectors, weak effect)
                        Engines defend specific points, not large sectors.

    Returns:
      dict mapping bearing_deg → suppression_strength for each 45° sector
      Sector bearing = centre of the 45° wedge (0=N, 90=E, 180=S, 270=W)
      Strength is additive across resources protecting that sector.
    """
    # 8-direction sector centres (compass bearings)
    sector_centres = [0, 45, 90, 135, 180, 225, 270, 315]
    sector_strength = {s: 0.0 for s in sector_centres}

    for r in resource_names:
        units = int(allocation.get((r, fire_name), 0) or 0)
        if units == 0:
            continue

        r_lower = r.lower()

        # Head fire sector = downwind direction (where fire is spreading TO)
        head_bearing = wind_to_deg % 360
        # Flanks = ±90° from head
        left_flank   = (head_bearing - 90) % 360
        right_flank  = (head_bearing + 90) % 360
        # Heel = upwind, opposite of head
        heel_bearing = (head_bearing + 180) % 360

        if "air tanker" in r_lower or "tanker" in r_lower:
            # Retardant on head fire — blocks ±45° cone around spread direction
            power = SUPPRESSION_POWER.get("Air Tanker", 5.0) * units
            for sc in sector_centres:
                angle_diff = abs(((sc - head_bearing) + 180) % 360 - 180)
                if angle_diff <= 45:
                    sector_strength[sc] += power * (1 - angle_diff / 90)

        elif "helicopter" in r_lower:
            # Helicopter works flanks ±90°–135° from head
            power = SUPPRESSION_POWER.get("Type-1 Helicopter", 3.5) * units
            for sc in sector_centres:
                for flank in [left_flank, right_flank]:
                    angle_diff = abs(((sc - flank) + 180) % 360 - 180)
                    if angle_diff <= 45:
                        sector_strength[sc] += power * (1 - angle_diff / 90)

        elif "dozer" in r_lower:
            # Dozer builds line on left flank
            power = SUPPRESSION_POWER.get("Heavy Dozer", 2.0) * units
            for sc in sector_centres:
                angle_diff = abs(((sc - left_flank) + 180) % 360 - 180)
                if angle_diff <= 45:
                    sector_strength[sc] += power * (1 - angle_diff / 90)

        elif "hand crew" in r_lower or "crew" in r_lower:
            # Crew holds heel and burns out
            power = SUPPRESSION_POWER.get("Hand Crew (20-person)", 1.5) * units
            for sc in sector_centres:
                angle_diff = abs(((sc - heel_bearing) + 180) % 360 - 180)
                if angle_diff <= 60:
                    sector_strength[sc] += power * (1 - angle_diff / 120)

        else:
            # Engine: weak distributed protection across all sectors
            power = SUPPRESSION_POWER.get("Type-1 Engine", 1.0) * units * 0.3
            for sc in sector_centres:
                sector_strength[sc] += power

    return sector_strength


def build_spread_graph(wind_speed_mps: float,
                       wind_dir_deg: float,
                       fuel_multiplier: float = 1.0,
                       suppression_factor: float = 0.0,
                       slope_field: np.ndarray = None,
                       sector_strengths: dict = None) -> nx.DiGraph:
    """
    Directed weighted grid graph for fire spread simulation.

    Physical factors incorporated:
      1. Wind: base_speed × wind_factor (Rothermel-inspired cosine)
      2. Fuel: base_speed scaled by fuel_multiplier
      3. Suppression (A — sectoral): each resource type blocks a specific
         angular sector rather than globally slowing all spread.
         Edges in suppressed sectors get travel_time × (1 + sector_strength).
         This means: airtanker slows head fire, crews hold heel, dozers hold flank.
      4. Slope (D): v_slope = v × (1 + BETA_SLOPE × slope_grade)

    suppression_factor: total S_f (for backward compat / reporting)
    sector_strengths: dict {bearing: strength} from get_suppression_sectors()
                      If provided, used instead of global suppression_factor.
    """
    G = nx.DiGraph()
    G.add_nodes_from([(r, c)
                      for r in range(GRID_SIZE)
                      for c in range(GRID_SIZE)])

    wind_to_deg = (wind_dir_deg + 180) % 360

    # If no sector strengths, fall back to mild global suppression for compatibility
    use_sectoral = sector_strengths is not None and any(v > 0 for v in sector_strengths.values())
    if not use_sectoral and suppression_factor > 0:
        global_reduction = 1.0 / (1.0 + ALPHA_SUPPRESSION * suppression_factor)
    else:
        global_reduction = 1.0

    base_speed = max(0.03 * wind_speed_mps * fuel_multiplier * global_reduction, 0.001)

    directions = {
        (-1, 0): 0,   (-1, 1): 45,  (0, 1): 90,   (1, 1): 135,
        (1,  0): 180, (1, -1): 225, (0, -1): 270, (-1, -1): 315,
    }

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            for (dr, dc), bearing in directions.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    theta       = np.radians(bearing - wind_to_deg)
                    wind_factor = max(1 + 2.0 * np.cos(theta), 0.1)
                    dist_m      = CELL_M * (np.sqrt(2) if abs(dr) + abs(dc) == 2 else 1)

                    # D: Slope effect
                    slope_mod = 1.0
                    if slope_field is not None:
                        grade     = (slope_field[nr, nc] - slope_field[r, c]) / dist_m
                        slope_mod = max(1.0 + BETA_SLOPE * grade, 0.1)

                    # A: Sectoral suppression — slows edges in protected sectors.
                    # Fix: use cosine taper instead of hard 45° cutoff.
                    # Hard cutoff causes discontinuous spread behavior when wind direction
                    # shifts 5-10°, because edges jump in/out of the suppressed zone.
                    # Cosine taper: effect = strength × 0.5 × (1 + cos(π × diff / half_width))
                    # This is zero at ±half_width, peaks smoothly at 0°, continuous everywhere.
                    suppression_mod = 1.0
                    if use_sectoral and sector_strengths:
                        edge_bearing = bearing
                        sector_effect = 0.0
                        HALF_WIDTH_RAD = np.radians(90)   # cosine taper over ±90° full window
                        for sc_bearing, strength in sector_strengths.items():
                            if strength <= 0:
                                continue
                            angle_diff = abs(((edge_bearing - sc_bearing) + 180) % 360 - 180)
                            if angle_diff < 90:
                                # Cosine taper: full at 0°, smoothly → 0 at ±90°
                                taper = 0.5 * (1.0 + np.cos(np.radians(angle_diff) / HALF_WIDTH_RAD * np.pi))
                                overlap = taper * strength
                                sector_effect = max(sector_effect, overlap)
                        suppression_mod = 1.0 + ALPHA_SUPPRESSION * sector_effect

                    travel_h = (dist_m * suppression_mod /
                                (base_speed * wind_factor * slope_mod * HOUR_SECS))
                    G.add_edge((r, c), (nr, nc), weight=travel_h)
    return G


def get_spread_cells(G: nx.DiGraph,
                     t_hours: float) -> set:
    """
    Return set of (row, col) grid cells reachable from ignition within t_hours.
    Includes the ignition cell itself (arrival time = 0).
    """
    ignition = (GRID_SIZE // 2, GRID_SIZE // 2)
    lengths  = nx.single_source_dijkstra_path_length(G, ignition, weight="weight")
    return {cell for cell, t in lengths.items() if t <= t_hours}


def acres_at_time(G: nx.DiGraph, t_hours: float) -> float:
    """
    Net new acres burned beyond discovery at time t_hours.

    The grid represents incremental spread area from the ignition point.
    discovery_acres already accounts for the fire's existing footprint at t=0.
    The ignition cell (centre of grid) is always reachable at t=0 and represents
    the starting point — not new area. We subtract 1 cell so that acres_at_time(0)=0,
    meaning zero additional spread at t=0, which is consistent with the accounting:

        current_acres(t) = discovery_acres + acres_at_time(t)
        current_acres(0) = discovery_acres + 0  ✓

    1 cell = 1 ha (100m × 100m) = 2.471 acres.
    """
    cells = get_spread_cells(G, t_hours)
    net_cells = max(len(cells) - 1, 0)   # subtract ignition cell (already counted in discovery_acres)
    return net_cells * 2.471


def run_spread_ensemble(fire_row: pd.Series,
                        t_hours: float,
                        slope_field: np.ndarray = None,
                        suppression_factor: float = 0.0,
                        n_runs: int = N_ENSEMBLE) -> dict:
    """
    C: Scenario analysis over spread footprint.

    Runs N_ENSEMBLE simulations per scenario (optimistic/expected/worst_case).
    Each run perturbs: wind speed ±10%, fuel multiplier ±15%.

    NOTE: Humidity is NOT perturbed here. The spread graph depends on wind speed,
    wind direction, fuel multiplier, slope, and suppression. Humidity affects risk
    scoring (Layer 1) but does not directly enter the spread graph edge weights.
    Perturbing humidity in the spread ensemble would have zero effect on footprint.
    Humidity remains a scenario-level label only (shown in reporting).

    Returns per-cell reachability probability for each scenario.
    Output: {scenario: {"prob": 2D array, "p10_acres", "p50_acres", "p90_acres"}}
    """
    fuel_mult_base = get_fuel_multiplier(
        fire_row.get("fuel_group"), fire_row.get("fire_behavior")
    )
    results = {}

    for scenario_name, params in ENSEMBLE_SCENARIOS.items():
        cell_hit_count = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        acres_list     = []
        rng = np.random.default_rng(seed=42)

        for run_i in range(n_runs):
            # Perturb wind and fuel only — these directly affect spread graph edge weights
            wind_noise = 1.0 + rng.uniform(-ENSEMBLE_WIND_NOISE, ENSEMBLE_WIND_NOISE)
            fuel_noise = 1.0 + rng.uniform(-ENSEMBLE_FUEL_NOISE, ENSEMBLE_FUEL_NOISE)

            wind_t = float(np.clip(
                fire_row.get("wind_speed_mps", 3.0) * params["wind_scale"] * wind_noise,
                0.5, WIND_MAX_MPS
            ))
            fuel_t = float(np.clip(
                fuel_mult_base * params["fuel_scale"] * fuel_noise, 0.1, 3.0
            ))
            wdir_t = float(fire_row.get("wind_dir_deg", 270.0) or 270.0)

            G = build_spread_graph(
                wind_speed_mps    = wind_t,
                wind_dir_deg      = wdir_t,
                fuel_multiplier   = fuel_t,
                suppression_factor= suppression_factor,
                slope_field       = slope_field,
            )
            cells = get_spread_cells(G, t_hours)
            net_cells = max(len(cells) - 1, 0)
            for (r, c) in cells:
                cell_hit_count[r, c] += 1
            acres_list.append(net_cells * 2.471)

        prob = cell_hit_count / n_runs
        acres_arr = np.array(sorted(acres_list))
        results[scenario_name] = {
            "prob"      : prob,
            "p10_acres" : float(np.percentile(acres_arr, 10)),
            "p50_acres" : float(np.percentile(acres_arr, 50)),
            "p90_acres" : float(np.percentile(acres_arr, 90)),
            # Humidity label (informational only — does not affect spread footprint)
            "humidity_label": f"RH {params['humidity_delta']:+.0f}pp (risk scoring only)",
        }

    return results


def compute_risk_at_t(fires_df: pd.DataFrame,
                       graphs: dict,
                       t_hours: float,
                       weights: dict,
                       weather_override: dict = None) -> pd.DataFrame:
    """
    Re-score risk at time t using projected fire size from spread model.
    B: Accepts weather_override dict {fire_name: {wind_speed_mps, humidity_pct, ...}}
       so dynamic weather conditions feed into risk scoring at each timestep.
    """
    df = fires_df.copy()

    for i, row in df.iterrows():
        f = row["fire_name"]
        projected = acres_at_time(graphs[f], t_hours)
        df.at[i, "current_acres"] = row["discovery_acres"] + projected

        # B: Apply dynamic weather if provided
        if weather_override and f in weather_override:
            wx = weather_override[f]
            for col in ["wind_speed_mps", "humidity_pct", "temperature_c", "wind_dir_deg"]:
                if col in wx:
                    df.at[i, col] = wx[col]

    df["size_score"] = np.log1p(df["current_acres"])
    df["size_score"] = (df["size_score"] / df["size_score"].max()).clip(0, 1)

    # Exponential humidity risk
    rh = df["humidity_pct"].clip(0, 100)
    denom_h = 1.0 - np.exp(-K_HUMIDITY * 100)
    df["humidity_risk"] = (
        (np.exp(-K_HUMIDITY * rh) - np.exp(-K_HUMIDITY * 100)) / denom_h
    ).clip(0, 1)

    # Exponential wind risk
    w_kmh = (df["wind_speed_mps"] * 3.6).clip(0, WIND_MAX_KMH)
    denom_w = np.exp(K_WIND * WIND_MAX_KMH) - 1.0
    df["wind_risk"] = (
        (np.exp(K_WIND * w_kmh) - 1.0) / denom_w
    ).clip(0, 1)

    # Linear temperature 0–45°C
    df["temp_risk"] = (
        (df["temperature_c"] - TEMP_MIN_C) / (TEMP_MAX_C - TEMP_MIN_C)
    ).clip(0, 1)

    df["weather_score"] = (
        WEATHER_W_HUMIDITY * df["humidity_risk"] +
        WEATHER_W_WIND     * df["wind_risk"]     +
        WEATHER_W_TEMP     * df["temp_risk"]
    )

    behavior_map   = {"Minimal":0.2,"Moderate":0.5,"Active":0.8,"Extreme":1.0}
    complexity_map = {"Type 1 Incident":1.0,"Type 2 Incident":0.75,
                      "Type 3 Incident":0.5,"Type 4 Incident":0.25,
                      "Type 5 Incident":0.1}
    df["behavior_score"]   = df["fire_behavior"].map(behavior_map).fillna(0.3)
    df["complexity_score"] = df["mgmt_complexity"].map(complexity_map).fillna(0.3)

    df["fire_danger"] = (
        weights["size"]       * df["size_score"]       +
        weights["weather"]    * df["weather_score"]    +
        weights["behavior"]   * df["behavior_score"]   +
        weights["complexity"] * df["complexity_score"]
    )
    # Absolute normalization — batch-independent (see compute_risk_scores)
    df["risk_score_100"] = (df["fire_danger"] * 100).round(1).clip(0, 100)
    df["priority_rank"]  = df["fire_danger"].rank(ascending=False).astype(int)
    return df


def compute_elliptical_asset_scores(fires_df: pd.DataFrame,
                                     graphs: dict,
                                     assets_gdf,
                                     t_hours: float = 12) -> dict:
    """
    Compute asset scores using actual Dijkstra spread footprint (elliptical)
    rather than a haversine circle. Fixes the circular vs elliptical flaw.

    For each fire:
      1. Get the set of grid cells reachable within t_hours (elliptical footprint)
      2. Convert those cells to real lat/lon bounding boxes
      3. Spatially join with OSM assets
      4. Sum asset weights within the actual spread footprint
    """
    if assets_gdf is None:
        return {fire["fire_name"]: 1.0 for _, fire in fires_df.iterrows()}

    LAT_PER_M = 1 / 111_320
    scores = {}

    for _, fire in fires_df.iterrows():
        name    = fire["fire_name"]
        G       = graphs[name]
        cells   = get_spread_cells(G, t_hours)
        centre  = GRID_SIZE // 2
        lon_m   = 1 / (111_320 * np.cos(np.radians(fire["fire_lat"])))

        # Convert threatened cells to lat/lon bounding boxes
        cell_bounds = []
        for (r, c) in cells:
            dr, dc  = r - centre, c - centre
            lat_top = fire["fire_lat"] - dr       * CELL_M * LAT_PER_M
            lat_bot = fire["fire_lat"] - (dr + 1) * CELL_M * LAT_PER_M
            lon_lft = fire["fire_lon"] + dc       * CELL_M * lon_m
            lon_rgt = fire["fire_lon"] + (dc + 1) * CELL_M * lon_m
            cell_bounds.append((
                min(lat_top, lat_bot), max(lat_top, lat_bot),
                min(lon_lft, lon_rgt), max(lon_lft, lon_rgt)
            ))

        # Find assets within any threatened cell
        fire_assets = assets_gdf[assets_gdf["fire_name"] == name]
        if fire_assets.empty or not cell_bounds:
            scores[name] = 0.0
            continue

        total_weight = 0.0
        for _, asset in fire_assets.iterrows():
            alat = asset["centroid_lat"]
            alon = asset["centroid_lon"]
            for (lat_min, lat_max, lon_min, lon_max) in cell_bounds:
                if lat_min <= alat <= lat_max and lon_min <= alon <= lon_max:
                    total_weight += asset["asset_weight"]
                    break   # count each asset once even if in multiple cells

        scores[name] = total_weight

    # Percentile-anchored normalization to 1–10.
    # Problem with pure min-max: if all fires have similar exposure, normalization
    # spreads small absolute differences across the full 1–10 range, making the
    # optimizer treat modest differences as major strategic distinctions.
    # Fix: anchor p5 → 1 and p95 → 10. Scores outside that range are clipped.
    # This preserves meaningful differences while not exaggerating clustered values.
    s = pd.Series(scores)
    if s.max() <= 0 or s.nunique() == 1:
        return {k: 5.0 for k in scores}

    p5  = float(np.percentile(s.values, 5))
    p95 = float(np.percentile(s.values, 95))

    if p95 <= p5:
        # Fallback to min-max if percentile range collapses (e.g. only 2 fires)
        p5, p95 = float(s.min()), float(s.max())

    return {
        k: float(np.clip(1.0 + 9.0 * (v - p5) / (p95 - p5), 1.0, 10.0))
        for k, v in scores.items()
    }


def run_dynamic_allocation(fires_df: pd.DataFrame,
                            resources: pd.DataFrame,
                            weights: dict,
                            asset_scores: dict = None,
                            prebuilt_graphs: dict = None,
                            slope_fields: dict = None) -> tuple:
    """
    Rolling reallocation with full feedback loop (A + B):

    At each CHECK_HOURS step:
      1. B: Update weather conditions (afternoon drying, wind fluctuation)
      2. A: Compute suppression strength from previous allocation
      3. A: Rebuild spread graphs with suppression-adjusted base speed
      4. Project new acreage from updated graphs
      5. Re-score risk with updated size + weather
      6. Re-run IP optimizer
      7. Store allocation → feeds back into step 2 at next timestep

    This closes the Layer 2 ↔ Layer 3 feedback loop:
      optimizer → suppression_strength → spread_graph → acreage → risk → optimizer
    """
    fire_names     = fires_df["fire_name"].tolist()
    resource_names = resources["resource"].tolist()
    if "cost_per_hour" not in resources.columns:
        raise ValueError("resources.csv must have cost_per_hour and acres_per_hour columns.")
    prod_hrs_dyn = dict(zip(resource_names,
                            resources["productive_hours_per_day"]
                            if "productive_hours_per_day" in resources.columns
                            else [10] * len(resource_names)))
    acres_per_day = {r: resources.loc[resources["resource"]==r,"acres_per_hour"].values[0]
                        * prod_hrs_dyn[r] for r in resource_names}
    fuel_mults     = {
        fire["fire_name"]: get_fuel_multiplier(
            fire.get("fuel_group"), fire.get("fire_behavior")
        )
        for _, fire in fires_df.iterrows()
    }

    # Use prebuilt graphs for t=0 (no suppression yet, base weather)
    if prebuilt_graphs is not None:
        graphs = prebuilt_graphs
        print("\n  Reusing pre-built spread graphs for t=0 …")
    else:
        print("\n  Building initial spread graphs …")
        graphs = {
            fire["fire_name"]: build_spread_graph(
                fire["wind_speed_mps"], fire["wind_dir_deg"],
                fuel_multiplier=fuel_mults[fire["fire_name"]],
                slope_field=slope_fields.get(fire["fire_name"]) if slope_fields else None,
            )
            for _, fire in fires_df.iterrows()
        }

    timeline     = []
    fires_last   = None
    prev_ranks   = {}
    allocation   = {(r, f): 0 for r in resource_names for f in fire_names}  # start: no resources
    rng_weather  = np.random.default_rng(seed=7)

    for t in CHECK_HOURS:
        # B: Dynamic weather at this timestep
        weather_at_t = {}
        for _, fire in fires_df.iterrows():
            f  = fire["fire_name"]
            wx = get_weather_at_t(fire, t, rng=rng_weather)
            weather_at_t[f] = wx

        # A: Sectoral suppression — rebuild graphs with resource-specific sector protection
        if t > 0:
            print(f"\n  t={t}h — rebuilding spread graphs with sectoral suppression …")
            for f in fire_names:
                S_f  = compute_suppression_strength(f, allocation, resource_names)
                wx   = weather_at_t[f]
                wind_to_deg = (wx["wind_dir_deg"] + 180) % 360
                sectors = get_suppression_sectors(f, allocation, resource_names, wind_to_deg)
                graphs[f] = build_spread_graph(
                    wind_speed_mps    = wx["wind_speed_mps"],
                    wind_dir_deg      = wx["wind_dir_deg"],
                    fuel_multiplier   = fuel_mults[f],
                    suppression_factor= S_f,
                    slope_field       = slope_fields.get(f) if slope_fields else None,
                    sector_strengths  = sectors,
                )
                if S_f > 0:
                    top_sectors = sorted(sectors.items(), key=lambda x: -x[1])[:2]
                    sec_str = ", ".join(f"{b}°={s:.1f}" for b, s in top_sectors if s > 0)
                    print(f"    {f:<20}  S_f={S_f:.1f}  sectors protected: [{sec_str}]")

        # Re-score risk with updated weather and projected acreage
        fires_t = compute_risk_at_t(fires_df, graphs, t, weights,
                                    weather_override=weather_at_t)

        # Run optimizer with updated risk scores
        result     = run_optimizer(fires_t, resources,
                                   asset_scores=asset_scores,
                                   budget=DAILY_BUDGET if "acres_per_day" in resources.columns else HORIZON_BUDGET)
        allocation = result["allocation"]  # feeds back into next iteration

        snap = {"hour": t}
        for f in fire_names:
            cov      = sum(int(allocation[(r, f)] or 0) * acres_per_day[r]
                          for r in resource_names)
            rank_now = int(fires_t.loc[fires_t["fire_name"]==f, "priority_rank"].values[0])
            S_f      = compute_suppression_strength(f, allocation, resource_names)
            wx       = weather_at_t[f]
            wind_to  = (wx["wind_dir_deg"] + 180) % 360
            sectors  = get_suppression_sectors(f, allocation, resource_names, wind_to)
            top_sec  = max(sectors.items(), key=lambda x: x[1]) if sectors else (0, 0)

            snap[f"{f}_acres"]           = round(fires_t.loc[fires_t["fire_name"]==f,
                                                              "current_acres"].values[0], 0)
            snap[f"{f}_risk"]            = fires_t.loc[fires_t["fire_name"]==f,
                                                        "risk_score_100"].values[0]
            snap[f"{f}_rank"]            = rank_now
            snap[f"{f}_rank_prev"]       = prev_ranks.get(f, rank_now)
            snap[f"{f}_coverage"]        = round(cov, 0)
            snap[f"{f}_suppression"]     = round(S_f, 2)
            snap[f"{f}_top_sector_deg"]  = top_sec[0]
            snap[f"{f}_top_sector_str"]  = round(top_sec[1], 1)
            snap[f"{f}_wind"]            = round(wx["wind_speed_mps"], 2)
            snap[f"{f}_humidity"]        = round(wx["humidity_pct"], 1)
            snap[f"{f}_diurnal_phase"]   = round(wx.get("_diurnal_phase", 0.0), 2)
            prev_ranks[f] = rank_now

        timeline.append(snap)
        fires_last = fires_t

    return pd.DataFrame(timeline), fires_last, graphs


def print_dynamic_report(timeline: pd.DataFrame, fire_names: list):
    print("\n" + "=" * 65)
    print("  LAYER 3 — DYNAMIC SPREAD + ROLLING IP ALLOCATION")
    print("  Wind-driven directional exposure model, not a fire forecast.")
    print("=" * 65)
    print("""
  How this works:
    At each 3-hour checkpoint, the spread model projects how many acres
    each fire has grown (using its wind speed, direction, and fuel type).
    The updated acreage feeds back into the risk score, and the IP optimizer
    re-runs to decide if resources should be shifted.
    Rank changes indicate when a fire's relative danger has shifted enough
    to change the optimizer's allocation decision.
""")

    rows_by_t = {int(snap["hour"]): snap for _, snap in timeline.iterrows()}
    hours = [int(h) for h in timeline["hour"].tolist()]

    for i, t in enumerate(hours):
        snap = rows_by_t[t]
        print(f"  ── t = {t}h {'─'*48}")
        print(f"  {'Fire':<20} {'Acres':>8} {'Growth':>7} {'Risk':>7} {'Rank':>5} {'Coverage':>12}  Notes")
        print(f"  {'-'*80}")

        prev = rows_by_t[hours[i-1]] if i > 0 else None

        for f in sorted(fire_names, key=lambda f: snap[f"{f}_rank"]):
            acres_now  = snap[f"{f}_acres"]
            risk_now   = snap[f"{f}_risk"]
            rank_now   = int(snap[f"{f}_rank"])
            cov_now    = snap[f"{f}_coverage"]

            # Growth since t=0
            acres_t0   = rows_by_t[0][f"{f}_acres"]
            growth_x   = acres_now / acres_t0 if acres_t0 > 0 else 1.0

            # Rank change note
            notes = []
            if prev is not None:
                rank_prev = int(prev[f"{f}_rank"])
                cov_prev  = prev[f"{f}_coverage"]
                if rank_now < rank_prev:
                    notes.append(f"↑ rank {rank_prev}→{rank_now}")
                elif rank_now > rank_prev:
                    notes.append(f"↓ rank {rank_prev}→{rank_now}")
                if cov_now > cov_prev + 50:
                    notes.append("more resources allocated")
                elif cov_now < cov_prev - 50:
                    notes.append("resources pulled back")

            print(f"  {f:<20} {acres_now:>8,.0f} {growth_x:>6.1f}×"
                  f" {risk_now:>7.1f} {'#'+str(rank_now):>5}"
                  f" {cov_now:>10,.0f} ac"
                  f"  {'  |  '.join(notes) if notes else ''}")
        print()

    # Summary: what changed and why
    print("  ── What drove reallocation ─────────────────────────────────")
    snap_0  = rows_by_t[0]
    snap_12 = rows_by_t[hours[-1]]
    for f in fire_names:
        acres_0  = snap_0[f"{f}_acres"]
        acres_12 = snap_12[f"{f}_acres"]
        rank_0   = int(snap_0[f"{f}_rank"])
        rank_12  = int(snap_12[f"{f}_rank"])
        growth   = acres_12 / acres_0 if acres_0 > 0 else 1.0

        rank_str = (f"rank stable at #{rank_0}" if rank_0 == rank_12
                    else f"rank shifted #{rank_0} → #{rank_12}")
        print(f"  {f:<20}  grew {growth:.1f}×  ({acres_0:,.0f}→{acres_12:,.0f} ac)  {rank_str}")

    print("\n\n  ── Projected acres burned ──────────────────────────────────")
    print(f"  {'Fire':<20}", end="")
    for t in hours:
        print(f"  {int(t):>5}h", end="")
    print()
    print(f"  {'-'*65}")
    for f in fire_names:
        print(f"  {f:<20}", end="")
        for t in hours:
            print(f"  {rows_by_t[t][f'{f}_acres']:>5,.0f}", end="")
        print()





# ════════════════════════════════════════════════════════════════════════════
# Optimizer: TERRAIN / ACCESS SCORING
# ════════════════════════════════════════════════════════════════════════════

def compute_terrain_scores(fires: pd.DataFrame) -> dict:
    """
    Per-fire terrain accessibility score in [0, 1].
    1.0 = flat + road at doorstep. Towards 0.0 = steep + remote.
    Uses terrain_slope_pct and road_distance_km columns if present.
    Falls back to 0.5 (neutral) if columns missing.
    """
    scores = {}
    for _, row in fires.iterrows():
        slope_pct = float(row.get("terrain_slope_pct", 15.0) or 15.0)
        road_km   = float(row.get("road_distance_km",  5.0)  or 5.0)
        score = (1.0
                 - SLOPE_PENALTY * np.clip(slope_pct / 100.0, 0.0, 1.0)
                 - ROAD_PENALTY  * np.clip(road_km   / MAX_ROAD_KM, 0.0, 1.0))
        scores[row["fire_name"]] = float(np.clip(score, 0.0, 1.0))
    return scores


def terrain_adjusted_capacity(base_aph: float,
                               terrain_score: float,
                               resource: str) -> float:
    """
    Adjust resource's effective acres/hour for terrain.
    Air resources (helicopters, tankers) are unaffected.
    Ground resources scale between MIN_GROUND_EFFECTIVENESS and 1.0.
    """
    r_lower = resource.lower()
    if "helicopter" in r_lower or "tanker" in r_lower:
        return base_aph
    eff = MIN_GROUND_EFFECTIVENESS + (1.0 - MIN_GROUND_EFFECTIVENESS) * terrain_score
    return base_aph * eff


# ════════════════════════════════════════════════════════════════════════════
# Optimizer: MILP OPTIMIZER — cost + residual damage minimization
# Final MILP optimizer: cost + residual damage minimization
# ════════════════════════════════════════════════════════════════════════════

# Nonlinear suppression effectiveness
# Sources: Holmes & Calkin (2013) empirical bounds (14–93% of standard rates).
# eff[f] = 1 / (1 + K_SIZE * demand[f])
# At demand=100 ac  → eff ≈ 0.95  (near-standard productivity)
# At demand=1800 ac → eff ≈ 0.82  (moderately degraded on a large fire)
# At demand=50000 ac → eff ≈ 0.14  (severely degraded — very large fires)
K_SIZE = 1.25e-4   # conservative default; sensitivity: 1.25e-4, 5e-4, 1e-3
K_SIZE_VALUES = [1.25e-4, 5e-4, 1e-3]   # for sensitivity analysis


def run_resource_hour_optimizer(fires: pd.DataFrame,
                     resources: pd.DataFrame,
                     asset_scores: dict = None,
                     budget: float = HORIZON_BUDGET,
                     horizon_hours: float = PLANNING_HORIZON_HOURS,
                     terrain_scores: dict = None,
                     lam: float = None) -> dict:
    """
    Planning-horizon MILP — cost + residual damage minimization.

    Objective:
        min  Σ_{r,f} cph[r] × h[r,f]
           + λ × Σ_f (danger[f]/100) × (asset[f]/10) × u[f] × DAMAGE_COST_PER_ACRE

    Decision variables:
        h[r,f]  ∈ Z≥0   resource-hours of type r assigned to fire f
        c[f]    ≥ 0     effective acres covered at fire f
        u[f]    ≥ 0     uncovered acres at fire f (slack)

    Coverage includes nonlinear size-effectiveness multiplier:
        eff[f] = 1 / (1 + K_SIZE × demand[f])
        c[f] = Σ_r aph_eff[r,f] × eff[f] × h[r,f]

    Constraints:
        C1  Supply  : Σ_f h[r,f] ≤ units[r] × horizon_hours        ∀ r
        C2  Budget  : Σ_{r,f} cph[r] × h[r,f] ≤ B
        C3  Coverage: c[f] = Σ_r aph_eff[r,f] × eff[f] × h[r,f]   ∀ f
        C4  Uncovered: u[f] ≥ demand[f] − c[f]                     ∀ f
        C5  Overcap : c[f] ≤ demand[f] + max_single_aph             ∀ f

    Demand hierarchy: incident_size_6h > incident_size > current_acres > discovery_acres
    """
    if lam is None:
        lam = LAMBDA_DAMAGE

    fire_names     = fires["fire_name"].tolist()
    resource_names = resources["resource"].tolist()

    units_avail = dict(zip(resource_names, resources["units_available"]))

    if "cost_per_hour" not in resources.columns:
        raise ValueError(
            "resources.csv must have 'cost_per_hour' and 'acres_per_hour' columns."
        )
    aph_base = dict(zip(resource_names, resources["acres_per_hour"]))
    cph      = dict(zip(resource_names, resources["cost_per_hour"]))

    # Demand hierarchy
    if "incident_size_6h" in fires.columns and fires["incident_size_6h"].notna().any():
        demand     = dict(zip(fire_names, fires["incident_size_6h"].fillna(fires["discovery_acres"])))
        demand_src = "incident_size_6h"
    elif "incident_size" in fires.columns and fires["incident_size"].notna().any():
        demand     = dict(zip(fire_names, fires["incident_size"].fillna(fires["discovery_acres"])))
        demand_src = "incident_size"
    elif "current_acres" in fires.columns:
        demand     = dict(zip(fire_names, fires["current_acres"]))
        demand_src = "current_acres"
    else:
        demand     = dict(zip(fire_names, fires["discovery_acres"]))
        demand_src = "discovery_acres"

    danger  = dict(zip(fire_names, fires["risk_score_100"]))
    asset   = asset_scores or {f: 5.0 for f in fire_names}
    tscores = terrain_scores or {f: 0.5 for f in fire_names}

    # Terrain-adjusted acres per hour (ground resources only)
    aph_eff = {(r, f): terrain_adjusted_capacity(aph_base[r], tscores.get(f, 0.5), r)
               for r in resource_names for f in fire_names}

    # Nonlinear size effectiveness multiplier per fire
    eff = {f: 1.0 / (1.0 + K_SIZE * demand[f]) for f in fire_names}

    max_rh     = {r: units_avail[r] * horizon_hours for r in resource_names}
    max_single = max(aph_base.values())

    m = LpProblem("Wildfire_Triage__MinCost", LpMinimize)

    h = {(r, f): LpVariable(
            f"h_{r.replace(' ','_').replace('-','_')}_{f.replace(' ','_').replace('-','_')}",
            lowBound=0, cat="Integer")
         for r in resource_names for f in fire_names}

    c = {f: LpVariable(f"c_{f.replace(' ','_').replace('-','_')}", lowBound=0)
         for f in fire_names}

    u = {f: LpVariable(f"u_{f.replace(' ','_').replace('-','_')}", lowBound=0)
         for f in fire_names}

    # Objective: suppression cost + λ × residual damage
    suppression_cost = lpSum(cph[r] * h[(r, f)]
                             for r in resource_names for f in fire_names)
    residual_damage  = lpSum(
        lam * (danger[f] / 100.0) * (asset.get(f, 5.0) / 10.0) * u[f] * DAMAGE_COST_PER_ACRE
        for f in fire_names
    )
    m += suppression_cost + residual_damage

    # C1: resource-hour supply
    for r in resource_names:
        m += lpSum(h[(r, f)] for f in fire_names) <= max_rh[r]

    # C2: budget
    total_cost_expr = lpSum(cph[r] * h[(r, f)]
                            for r in resource_names for f in fire_names)
    m += total_cost_expr <= budget

    for f in fire_names:
        # C3: coverage with size-effectiveness multiplier
        m += c[f] == lpSum(aph_eff[(r, f)] * eff[f] * h[(r, f)]
                           for r in resource_names)
        # C4: uncovered slack
        m += u[f] >= demand[f] - c[f]
        # C5: overcoverage cap
        m += c[f] <= demand[f] + max_single

    m.solve(PULP_CBC_CMD(msg=0))

    alloc    = {(r, f): int(value(h[(r, f)]) or 0)
                for r in resource_names for f in fire_names}
    coverage = {f: float(value(c[f]) or 0) for f in fire_names}
    uncov    = {f: float(value(u[f]) or 0) for f in fire_names}
    cost_h   = {f: sum(alloc[(r, f)] * cph[r] for r in resource_names) for f in fire_names}

    return {
        "status"         : LpStatus[m.status],
        "objective"      : value(m.objective),
        "allocation"     : alloc,
        "coverage"       : coverage,
        "uncovered"      : uncov,
        "horizon_cost"   : cost_h,
        "fire_names"     : fire_names,
        "resource_names" : resource_names,
        "acres_per_hour" : aph_eff,
        "cost_per_hour"  : cph,
        "demand"         : demand,
        "demand_src"     : demand_src,
        "risk_scores"    : danger,
        "asset_scores"   : asset,
        "terrain_scores" : tscores,
        "horizon_hours"  : horizon_hours,
        "budget"         : budget,
        "eff"            : eff,
    }


def print_resource_hour_allocation_report(result: dict, fires: pd.DataFrame):
    """Print resource-hours allocation report ( formulation)."""
    fnames    = result["fire_names"]
    rnames    = result["resource_names"]
    alloc     = result["allocation"]
    coverage  = result["coverage"]
    uncov     = result["uncovered"]
    demand    = result["demand"]
    danger    = result["risk_scores"]
    asset     = result["asset_scores"]
    tscores   = result["terrain_scores"]
    cph       = result["cost_per_hour"]
    hz        = result["horizon_hours"]
    rank_map  = dict(zip(fires["fire_name"], fires["priority_rank"]))

    print("\n" + "=" * 70)
    print(f"  LAYER 2 — RESOURCE-HOURS OPTIMIZER  [{result['status']}]")
    print(f"  Planning horizon : {hz}h   |   Budget: ${result.get("budget", HORIZON_BUDGET):,}")
    print(f"  Objective (suppression cost + λ×residual damage, lower=better): {result['objective']:.4f}")
    print("=" * 70)

    print("\n── Resource-hours dispatched (h[r,f]) ──────────────────────────────")
    alloc_df = pd.DataFrame(index=rnames, columns=fnames, data=0)
    for (r, f), v in alloc.items():
        alloc_df.loc[r, f] = v
    print(alloc_df.to_string())

    total_cost = 0
    print("\n── Allocation summary ───────────────────────────────────────────────")
    for f in sorted(fnames, key=lambda f: rank_map[f]):
        cov    = coverage[f]
        dem    = demand[f]
        fcost  = sum(alloc[(r, f)] * cph[r] for r in rnames)
        total_cost += fcost
        pct    = min(cov / dem * 100, 100) if dem > 0 else 0
        ts     = tscores.get(f, 0.5)
        dmg    = (LAMBDA_DAMAGE * (danger[f]/100.0) *
                  (asset.get(f,5.0)/10.0) * uncov[f] * DAMAGE_COST_PER_ACRE)
        rh_parts = [f"{r.split('(')[0].strip()}: {alloc[(r,f)]}h"
                    for r in rnames if alloc[(r,f)] > 0]
        print(f"\n  #{rank_map[f]}  {f}  "
              f"(danger={danger[f]:.1f}/100  asset={asset.get(f,5.0):.1f}/10  "
              f"terrain={ts:.2f})")
        print(f"      Demand (6h)     : {dem:,.0f} ac")
        print(f"      Response demand met: {pct:.0f}%  ({cov:,.0f} ac assigned)")
        print(f"      Unmet demand    : {uncov[f]:,.0f} ac"
              f"{'  ← expected for lower priority' if pct < 30 else ''}")
        print(f"      Residual damage : ${dmg:,.0f}")
        print(f"      Cost over {hz}h  : ${fcost:,.0f}")
        print(f"      Dispatch        : {', '.join(rh_parts) or 'none'}")

    print(f"\n  Budget for {hz}h : ${HORIZON_BUDGET:,}")
    print(f"  Budget used     : ${total_cost:,.0f}  ({total_cost/HORIZON_BUDGET*100:.0f}%)")
    print(f"  Budget remaining: ${HORIZON_BUDGET - total_cost:,.0f}")


# ════════════════════════════════════════════════════════════════════════════
# Optimizer: UNIFIED SENSITIVITY ANALYSES (horizon + lambda + budget via run_resource_hour_optimizer)
# ════════════════════════════════════════════════════════════════════════════

def planning_horizon_sensitivity(fires: pd.DataFrame,
                         resources: pd.DataFrame,
                         asset_scores: dict = None,
                         terrain_scores: dict = None) -> None:
    """Sweep planning horizon [2,4,6,8,12]h using unified run_resource_hour_optimizer."""
    print("\n" + "=" * 65)
    print("  SENSITIVITYOptimizer: Planning Horizon")
    print("=" * 65)
    demand = {}
    if "incident_size_6h" in fires.columns and fires["incident_size_6h"].notna().any():
        demand = dict(zip(fires["fire_name"], fires["incident_size_6h"].fillna(fires["discovery_acres"])))
    else:
        demand = dict(zip(fires["fire_name"], fires["discovery_acres"]))

    horizons = [2, 4, 6, 8, 12]
    print(f"\n  {'Fire':<20}", end="")
    for h in horizons:
        print(f"  {h:>3}h", end="")
    print("  (demand met %)")
    for f in fires["fire_name"]:
        print(f"  {f:<20}", end="")
        for h in horizons:
            r = run_resource_hour_optimizer(fires, resources, asset_scores=asset_scores,
                                  terrain_scores=terrain_scores, horizon_hours=h)
            dem = demand.get(f, 1)
            pct = min(r["coverage"][f] / dem * 100, 100) if dem > 0 else 0
            print(f"  {pct:>5.0f}%", end="")
        print()


def budget_sensitivity(fires: pd.DataFrame,
                            resources: pd.DataFrame,
                            asset_scores: dict = None,
                            terrain_scores: dict = None) -> None:
    """Sweep budget values using unified run_resource_hour_optimizer."""
    print("\n" + "=" * 65)
    print("  SENSITIVITYOptimizer: Budget for Planning Horizon")
    print("=" * 65)
    demand = {}
    if "incident_size_6h" in fires.columns and fires["incident_size_6h"].notna().any():
        demand = dict(zip(fires["fire_name"], fires["incident_size_6h"].fillna(fires["discovery_acres"])))
    else:
        demand = dict(zip(fires["fire_name"], fires["discovery_acres"]))

    budgets = [50_000, 75_000, 100_000, 125_000, 150_000, 200_000, 250_000]
    print(f"\n  {'Fire':<20}", end="")
    for b in budgets:
        print(f"  ${b//1000:>3}k", end="")
    print("  (demand met %)")
    for f in fires["fire_name"]:
        print(f"  {f:<20}", end="")
        for b in budgets:
            r = run_resource_hour_optimizer(fires, resources, asset_scores=asset_scores,
                                  terrain_scores=terrain_scores, budget=b)
            dem = demand.get(f, 1)
            pct = min(r["coverage"][f] / dem * 100, 100) if dem > 0 else 0
            print(f"  {pct:>5.0f}%", end="")
        print()


# ════════════════════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSES
# ════════════════════════════════════════════════════════════════════════════

def complexity_sensitivity(fires: pd.DataFrame,
                            resources: pd.DataFrame,
                            weights: dict,
                            asset_scores: dict = None) -> None:
    """
    Test three complexity weight variants to check if Behavior/Complexity overlap
    materially changes rankings or allocations.

    Variants:
      A) Current model — full AHP complexity weight (16.1%)
      B) No complexity — weight redistributed to behavior
      C) Reduced complexity — half weight, remainder to behavior
    """
    print("\n" + "=" * 65)
    print("  SENSITIVITY: Complexity Weight Variants")
    print("=" * 65)

    variants = {
        "A) Current (complexity=16.1%)": weights,
        "B) No complexity (0%)": {
            **weights,
            "behavior":   weights["behavior"] + weights["complexity"],
            "complexity": 0.0,
        },
        "C) Reduced complexity (8%)": {
            **weights,
            "behavior":   weights["behavior"] + weights["complexity"] * 0.5,
            "complexity": weights["complexity"] * 0.5,
        },
    }

    for label, w in variants.items():
        fires_v = compute_risk_scores(fires.copy(), w)
        result  = run_optimizer(fires_v, resources,
                                asset_scores=asset_scores,
                                budget=DAILY_BUDGET if "acres_per_day" in resources.columns else HORIZON_BUDGET)
        alloc   = result["allocation"]
        apd     = result["acres_per_day"]
        rnames  = result["resource_names"]

        print(f"\n  {label}")
        print(f"  {'Fire':<20} {'Rank':>5} {'Risk':>7} {'Coverage%':>10} {'Cost':>10}")
        print(f"  {'-'*58}")
        for _, row in fires_v.sort_values("priority_rank").iterrows():
            f   = row["fire_name"]
            cov = sum(alloc[(r, f)] * apd[r] for r in rnames)
            pct = min(cov / row["discovery_acres"] * 100, 100) if row["discovery_acres"] > 0 else 0
            cst = sum(alloc[(r, f)] * result["cost_per_day"][r] for r in rnames)
            print(f"  {f:<20} {'#'+str(int(row['priority_rank'])):>5} "
                  f"{row['risk_score_100']:>7.1f} {pct:>9.0f}% ${cst:>8,.0f}")


def lambda_sensitivity(fires: pd.DataFrame,
                        resources: pd.DataFrame,
                        asset_scores: dict = None) -> None:
    """
    Test how allocation changes across lambda values (damage tradeoff parameter).
    Tests: 10, 25, 50, 75, 100
    Shows: coverage, uncovered acres, cost, objective value per fire.
    """
    print("\n" + "=" * 65)
    print("  SENSITIVITY: Lambda (Damage Tradeoff) Values")
    print("=" * 65)

    global LAMBDA_DAMAGE
    original_lambda = LAMBDA_DAMAGE
    fire_names = fires["fire_name"].tolist()

    print(f"\n  {'Fire':<20}", end="")
    for lam in [10, 25, 50, 75, 100]:
        print(f"  λ={lam:>3}", end="")
    print("  (coverage %)")

    for f in fire_names:
        print(f"  {f:<20}", end="")
        for lam in [10, 25, 50, 75, 100]:
            LAMBDA_DAMAGE = lam
            result = run_optimizer(fires, resources,
                                   asset_scores=asset_scores,
                                   budget=DAILY_BUDGET if "acres_per_day" in resources.columns else HORIZON_BUDGET)
            alloc  = result["allocation"]
            apd    = result["acres_per_day"]
            rnames = result["resource_names"]
            dem    = result["demand"][f]
            cov    = sum(alloc[(r, f)] * apd[r] for r in rnames)
            pct    = min(cov / dem * 100, 100) if dem > 0 else 0
            print(f"  {pct:>5.0f}%", end="")
        print()

    LAMBDA_DAMAGE = original_lambda  # restore


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  WILDFIRE TRIAGE MODEL")
    print("  Suppression Feedback · Dynamic Weather · Ensemble Uncertainty · Slope Terrain")
    print("=" * 65)

    fires     = pd.read_csv(FIRES_CSV,     index_col=0).reset_index(drop=True)
    resources = pd.read_csv(RESOURCES_CSV)

    print(f"\n  Fires loaded    : {len(fires)}")
    print(f"  Resource types  : {len(resources)}")
    print(f"  Daily budget    : ${DAILY_BUDGET:,}")
    print(f"  Lambda (damage) : {LAMBDA_DAMAGE}")

    # Layer 0: AHP
    weights = compute_ahp_weights(verbose=True)

    # Layer 1: Risk scoring (pure fire danger — no asset score)
    fires = compute_risk_scores(fires, weights)
    print_risk_report(fires, weights)

    # D: Build terrain — attempt SRTM fetch, fall back to synthetic
    print("\n  Building terrain elevation fields (D: SRTM if available, else synthetic) …")
    slope_fields = {}
    for i, (_, fire) in enumerate(fires.iterrows()):
        print(f"    {fire['fire_name']} ({fire['fire_lat']:.3f}, {fire['fire_lon']:.3f})")
        slope_fields[fire["fire_name"]] = build_slope_field(
            peak_row=GRID_SIZE * (0.3 + i * 0.05),
            peak_col=GRID_SIZE * (0.4 + i * 0.05),
            fire_lat=fire.get("fire_lat"),
            fire_lon=fire.get("fire_lon"),
            use_srtm=True,
        )
    print(f"    Slope sensitivity β={BETA_SLOPE}")

    # Layer 3 graphs built with slope + fuel (no suppression yet at t=0)
    print("\n  Building spread graphs for asset scoring (with slope + fuel) …")
    spread_graphs = {
        fire["fire_name"]: build_spread_graph(
            fire["wind_speed_mps"], fire["wind_dir_deg"],
            fuel_multiplier=get_fuel_multiplier(
                fire.get("fuel_group"), fire.get("fire_behavior")
            ),
            suppression_factor=0.0,
            slope_field=slope_fields[fire["fire_name"]],
        )
        for _, fire in fires.iterrows()
    }
    print("\n  Fuel multipliers applied:")
    for _, fire in fires.iterrows():
        fm  = get_fuel_multiplier(fire.get("fuel_group"), fire.get("fire_behavior"))
        src = "fuel_group" if (fire.get("fuel_group") and pd.notna(fire.get("fuel_group"))) else "behavior fallback"
        print(f"    {fire['fire_name']:<20}  multiplier={fm:.1f}  (source: {src})")

    # Layer 4: Elliptical asset scoring
    assets_gdf  = None
    ASSETS_PATH = "wildfire_data/osm_assets.geojson"
    try:
        import geopandas as gpd
        assets_gdf = gpd.read_file(ASSETS_PATH)
        print(f"  OSM assets loaded: {len(assets_gdf)} features")
    except Exception as e:
        print(f"  OSM assets not found ({e}) — using uniform asset scores")

    print("\n── Layer 4: Elliptical Asset Scoring ───────────────────────")
    asset_scores = compute_elliptical_asset_scores(
        fires, spread_graphs, assets_gdf, t_hours=12
    )
    for f, score in sorted(asset_scores.items(),
                            key=lambda x: fires.loc[fires["fire_name"]==x[0],
                                                     "priority_rank"].values[0]):
        rank = fires.loc[fires["fire_name"]==f, "priority_rank"].values[0]
        print(f"  #{rank}  {f:<20}  asset_score={score:.2f}")

    # Final optimizer uses resource-hours over the planning horizon.
    # ── Terrain scoring + resource-hour optimizer ──────────────────
    print("\n── Terrain / Access Scoring ─────────────────────────────────")
    terrain_scores = compute_terrain_scores(fires)
    for name, ts in terrain_scores.items():
        slope = float(fires.loc[fires["fire_name"]==name, "terrain_slope_pct"].values[0]) \
                if "terrain_slope_pct" in fires.columns else float("nan")
        road  = float(fires.loc[fires["fire_name"]==name, "road_distance_km"].values[0]) \
                if "road_distance_km" in fires.columns else float("nan")
        print(f"  {name:<22}  slope={slope:.0f}%  road={road:.1f}km  access={ts:.2f}")

    print("\n── Resource-Hour Optimizer ─────────────────────────────────")
    result_opt = run_resource_hour_optimizer(fires, resources,
                                  asset_scores=asset_scores,
                                  terrain_scores=terrain_scores)
    print_resource_hour_allocation_report(result_opt, fires)

    # Sensitivity analyses using the final resource-hour optimizer
    planning_horizon_sensitivity(fires, resources, asset_scores, terrain_scores)
    budget_sensitivity(fires, resources, asset_scores, terrain_scores)

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)