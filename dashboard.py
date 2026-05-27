"""
dashboard.py
Wildfire Triage — Decision Support Dashboard

Story:
  Tab 1 — Decision Summary    : What should we do?
  Tab 2 — Why This Allocation : How did the optimizer arrive here?
  Tab 3 — Compare Alternatives: Does IP beat simple rules?
  Tab 4 — Sensitivity         : How robust is the recommendation?

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
from pulp import *

st.set_page_config(
    page_title="Wildfire Triage",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1a1a1a;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; max-width: 100%; }

[data-testid="stSidebar"] h3 {
    font-size: 0.72rem;
    font-weight: 600;
    color: #888;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 1rem;
}

.fire-card {
    background: white;
    border-radius: 8px;
    border: 1px solid #e5e5e5;
    border-top: 4px solid #ddd;
    padding: 1.2rem;
}
.fire-rank-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.fire-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a1a1a;
    margin: 0.25rem 0;
}
.risk-num {
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    font-family: 'DM Mono', monospace;
}
.fire-stat {
    font-size: 0.78rem;
    color: #555;
    margin-top: 0.5rem;
    font-family: 'DM Mono', monospace;
    line-height: 1.7;
}
.tag {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    margin-right: 0.3rem;
    margin-top: 0.4rem;
}
.tag-active   { background:#fee2e2; color:#dc2626; }
.tag-minimal  { background:#e0f2fe; color:#0369a1; }
.tag-moderate { background:#fef3c7; color:#d97706; }
.tag-extreme  { background:#fce7f3; color:#9d174d; }
.tag-t1       { background:#fef3c7; color:#92400e; }
.tag-other    { background:#f3f4f6; color:#4b5563; }

.insight-box {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-left: 4px solid #f59e0b;
    padding: 0.8rem 1.1rem;
    border-radius: 4px;
    font-size: 0.88rem;
    color: #78350f;
    margin-bottom: 1.2rem;
    line-height: 1.55;
}
.stat-row {
    background: white;
    border: 1px solid #e5e5e5;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.stat-val {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    color: #1a1a1a;
}
.stat-label {
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODEL CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

GRID_SIZE = 40; CELL_M = 100
BEHAVIOR_MAP   = {"Minimal":0.2,"Moderate":0.5,"Active":0.8,"Extreme":1.0}
COMPLEXITY_MAP = {"Type 1 Incident":1.0,"Type 2 Incident":0.75,
                  "Type 3 Incident":0.5,"Type 4 Incident":0.25,"Type 5 Incident":0.1}
RANK_COLORS = {1:"#dc2626", 2:"#ea580c", 3:"#ca8a04", 4:"#2563eb"}
RANK_LABELS = {1:"#1 — Highest Priority", 2:"#2 — High Priority",
               3:"#3 — Moderate Priority", 4:"#4 — Lowest Priority"}

# AHP-derived weights (CR=0.0115)
AHP_W = {"size": 0.0960, "weather": 0.2771, "behavior": 0.4658, "complexity": 0.1611}

WIND_MAX_MPS = 20.0
WIND_MAX_KMH = WIND_MAX_MPS * 3.6
TEMP_MIN_C   =  0.0
TEMP_MAX_C   = 45.0
K_HUMIDITY   = 0.03
K_WIND       = 0.05
WEATHER_W_HUMIDITY = 0.30
WEATHER_W_WIND     = 0.45
WEATHER_W_TEMP     = 0.25

LAMBDA_DAMAGE = 50.0
DAMAGE_COST_PER_ACRE = 500.0

# Optimizer: Terrain effectiveness
SLOPE_PENALTY            = 0.60
ROAD_PENALTY             = 0.30
MAX_ROAD_KM              = 25.0
PLANNING_HORIZON_HOURS   = 6
HORIZON_BUDGET           = 150_000
MIN_GROUND_EFFECTIVENESS = 0.15

FUEL_SPREAD_MULTIPLIER = {
    "grass": 1.6, "grass/shrub": 1.4, "shrub": 1.3,
    "timber litter": 1.0, "timber": 1.0, "slash": 0.9,
    "nonburnable": 0.3, "urban": 0.4, "agriculture": 0.7, "water": 0.1,
    "_behavior_extreme": 1.5, "_behavior_active": 1.2,
    "_behavior_moderate": 1.0, "_behavior_minimal": 0.7,
}


# ════════════════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    fires     = pd.read_csv("wildfire_data/scenario_fires.csv", index_col=0).reset_index(drop=True)
    resources = pd.read_csv("wildfire_data/resources.csv")
    try:
        import geopandas as gpd
        assets = gpd.read_file("wildfire_data/osm_assets.geojson")
    except Exception:
        assets = None
    return fires, resources, assets


def get_fuel_multiplier(fuel_group, fire_behavior):
    if fuel_group and pd.notna(fuel_group):
        key = str(fuel_group).strip().lower()
        for k, v in FUEL_SPREAD_MULTIPLIER.items():
            if k.startswith("_"): continue
            if key == k or key.startswith(k): return v
    beh = str(fire_behavior or "").strip().lower()
    if   "extreme"  in beh: return FUEL_SPREAD_MULTIPLIER["_behavior_extreme"]
    elif "active"   in beh: return FUEL_SPREAD_MULTIPLIER["_behavior_active"]
    elif "moderate" in beh: return FUEL_SPREAD_MULTIPLIER["_behavior_moderate"]
    else:                   return FUEL_SPREAD_MULTIPLIER["_behavior_minimal"]


@st.cache_data
def build_slope_field_cached(fire_idx, fire_lat=None, fire_lon=None):
    r_idx = np.arange(GRID_SIZE)
    c_idx = np.arange(GRID_SIZE)
    rr, cc = np.meshgrid(r_idx, c_idx, indexing="ij")
    peak_row = GRID_SIZE * (0.3 + fire_idx * 0.05)
    peak_col = GRID_SIZE * (0.4 + fire_idx * 0.05)
    elev = 300.0 * np.exp(-((rr - peak_row)**2 + (cc - peak_col)**2) / (2 * 12.0**2))
    return elev


@st.cache_data
def build_spread_graph(wind_speed_mps, wind_dir_deg, fuel_multiplier=1.0,
                       _slope_key=None, _slope_data=None):
    G = nx.DiGraph()
    G.add_nodes_from([(r,c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)])
    dirs = {(-1,0):0,(-1,1):45,(0,1):90,(1,1):135,
            (1,0):180,(1,-1):225,(0,-1):270,(-1,-1):315}
    wind_to_deg = (wind_dir_deg + 180) % 360
    slope_field = _slope_data
    base = max(0.03 * wind_speed_mps * fuel_multiplier, 0.001)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            for (dr,dc), bearing in dirs.items():
                nr,nc = r+dr, c+dc
                if 0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE:
                    theta = np.radians(bearing - wind_to_deg)
                    wf    = max(1 + 2.0*np.cos(theta), 0.1)
                    dist  = CELL_M*(np.sqrt(2) if abs(dr)+abs(dc)==2 else 1)
                    slope_mod = 1.0
                    if slope_field is not None:
                        grade = (slope_field[nr,nc] - slope_field[r,c]) / dist
                        slope_mod = max(1.0 + 0.5 * grade, 0.1)
                    v = base * wf * slope_mod
                    G.add_edge((r,c),(nr,nc), weight=dist/max(v,0.001)/3600)
    return G


@st.cache_data
def build_all_graphs(wind_spd, _fires_tuple):
    fires_df = pd.DataFrame(_fires_tuple[1], columns=_fires_tuple[0])
    graphs = {}
    for i, (_, row) in enumerate(fires_df.iterrows()):
        slope = build_slope_field_cached(
            i,
            fire_lat=float(row.get("fire_lat") or 0) or None,
            fire_lon=float(row.get("fire_lon") or 0) or None,
        )
        graphs[row["fire_name"]] = build_spread_graph(
            wind_spd,
            float(row.get("wind_dir_deg", 270) or 270),
            fuel_multiplier=get_fuel_multiplier(row.get("fuel_group"), row.get("fire_behavior")),
            _slope_key=i,
            _slope_data=slope,
        )
    return graphs


def get_spread_cells(G, t_hours):
    L = nx.single_source_dijkstra_path_length(G, (GRID_SIZE//2, GRID_SIZE//2), weight="weight")
    return {cell for cell, t in L.items() if t <= t_hours}


def compute_asset_scores(fires_df, assets, graphs):
    if assets is None:
        return {fire["fire_name"]: 5.0 for _, fire in fires_df.iterrows()}

    LAT_PER_M = 1 / 111_320
    scores = {}
    for _, fire in fires_df.iterrows():
        name  = fire["fire_name"]
        G     = graphs[name]
        cells = get_spread_cells(G, 12)
        centre = GRID_SIZE // 2
        lon_m  = 1 / (111_320 * np.cos(np.radians(fire["fire_lat"])))

        cell_bounds = []
        for (r, c) in cells:
            dr, dc  = r - centre, c - centre
            lat_top = fire["fire_lat"] - dr       * CELL_M * LAT_PER_M
            lat_bot = fire["fire_lat"] - (dr + 1) * CELL_M * LAT_PER_M
            lon_lft = fire["fire_lon"] + dc       * CELL_M * lon_m
            lon_rgt = fire["fire_lon"] + (dc + 1) * CELL_M * lon_m
            cell_bounds.append((min(lat_top,lat_bot),max(lat_top,lat_bot),
                                min(lon_lft,lon_rgt),max(lon_lft,lon_rgt)))

        fire_assets = assets[assets["fire_name"] == name]
        if fire_assets.empty or not cell_bounds:
            scores[name] = 0.0; continue

        total = 0.0
        for _, asset in fire_assets.iterrows():
            alat, alon = asset["centroid_lat"], asset["centroid_lon"]
            for (lat_min, lat_max, lon_min, lon_max) in cell_bounds:
                if lat_min <= alat <= lat_max and lon_min <= alon <= lon_max:
                    total += asset["asset_weight"]; break
        scores[name] = total

    s = pd.Series(scores)
    if s.max() <= 0 or s.nunique() == 1:
        return {k: 5.0 for k in scores}
    p5, p95 = float(np.percentile(s.values, 5)), float(np.percentile(s.values, 95))
    if p95 <= p5: p5, p95 = float(s.min()), float(s.max())
    return {k: float(np.clip(1.0 + 9.0*(v-p5)/(p95-p5), 1.0, 10.0)) for k, v in scores.items()}


def compute_risk(fires_df):
    df = fires_df.copy()
    # Size: log-normalized using the same demand basis as the IP optimizer
    # (incident_size > discovery_acres), so risk scoring and optimizer workload are consistent.
    demand_series = pd.Series(get_demand(df)).reindex(df["fire_name"].values)
    demand_series.index = df.index
    log_demand = np.log1p(demand_series)
    df["size_score"] = (log_demand / log_demand.max()).clip(0, 1)
    rh = df["humidity_pct"].clip(0, 100)
    denom_h = 1.0 - np.exp(-K_HUMIDITY * 100)
    df["humidity_risk"] = ((np.exp(-K_HUMIDITY*rh) - np.exp(-K_HUMIDITY*100)) / denom_h).clip(0,1)
    w_kmh = (df["wind_speed_mps"] * 3.6).clip(0, WIND_MAX_KMH)
    denom_w = np.exp(K_WIND * WIND_MAX_KMH) - 1.0
    df["wind_risk"]     = ((np.exp(K_WIND*w_kmh) - 1.0) / denom_w).clip(0, 1)
    df["temp_risk"]     = ((df["temperature_c"] - TEMP_MIN_C) / (TEMP_MAX_C - TEMP_MIN_C)).clip(0,1)
    df["weather_score"] = (WEATHER_W_HUMIDITY*df["humidity_risk"] +
                           WEATHER_W_WIND*df["wind_risk"] +
                           WEATHER_W_TEMP*df["temp_risk"])
    df["behavior_score"]   = df["fire_behavior"].map(BEHAVIOR_MAP).fillna(0.3)
    df["complexity_score"] = df["mgmt_complexity"].map(COMPLEXITY_MAP).fillna(0.3)
    df["risk_score"] = (AHP_W["size"]       * df["size_score"]     +
                        AHP_W["weather"]    * df["weather_score"]  +
                        AHP_W["behavior"]   * df["behavior_score"] +
                        AHP_W["complexity"] * df["complexity_score"])
    df["risk_score_100"] = (df["risk_score"] * 100).round(1).clip(0, 100)
    df["priority_rank"]  = df["risk_score"].rank(ascending=False).astype(int)
    return df


def get_demand(fires_df):
    """Demand hierarchy: incident_size_6h > incident_size > current_acres > discovery_acres."""
    fnames = fires_df["fire_name"].tolist()
    if "incident_size_6h" in fires_df.columns and fires_df["incident_size_6h"].notna().any():
        return dict(zip(fnames, fires_df["incident_size_6h"].fillna(fires_df["discovery_acres"])))
    elif "incident_size" in fires_df.columns and fires_df["incident_size"].notna().any():
        return dict(zip(fnames, fires_df["incident_size"].fillna(fires_df["discovery_acres"])))
    elif "current_acres" in fires_df.columns:
        return dict(zip(fnames, fires_df["current_acres"]))
    return dict(zip(fnames, fires_df["discovery_acres"]))


# All dashboard tabs use the same resource-hour optimizer


# ════════════════════════════════════════════════════════════════════════════
# Optimizer: TERRAIN HELPERS
# ════════════════════════════════════════════════════════════════════════════

def compute_terrain_scores_dash(fires_df: pd.DataFrame) -> dict:
    scores = {}
    for _, row in fires_df.iterrows():
        slope_pct = float(row.get("terrain_slope_pct", 15.0) or 15.0)
        road_km   = float(row.get("road_distance_km",  5.0)  or 5.0)
        score = (1.0
                 - SLOPE_PENALTY * np.clip(slope_pct / 100.0, 0.0, 1.0)
                 - ROAD_PENALTY  * np.clip(road_km   / MAX_ROAD_KM, 0.0, 1.0))
        scores[row["fire_name"]] = float(np.clip(score, 0.0, 1.0))
    return scores


def terrain_adj_cap(base_aph: float, terrain_score: float, resource: str) -> float:
    r_lower = resource.lower()
    if "helicopter" in r_lower or "tanker" in r_lower:
        return base_aph
    eff = MIN_GROUND_EFFECTIVENESS + (1.0 - MIN_GROUND_EFFECTIVENESS) * terrain_score
    return base_aph * eff


def get_optimizer_demand(fires_df: pd.DataFrame) -> dict:
    """Demand hierarchy: incident_size_6h > incident_size > discovery_acres."""
    fnames = fires_df["fire_name"].tolist()
    if "incident_size_6h" in fires_df.columns and fires_df["incident_size_6h"].notna().any():
        s = fires_df["incident_size_6h"].fillna(fires_df["discovery_acres"])
    elif "incident_size" in fires_df.columns and fires_df["incident_size"].notna().any():
        s = fires_df["incident_size"].fillna(fires_df["discovery_acres"])
    else:
        s = fires_df["discovery_acres"]
    return dict(zip(fnames, s.values))


def run_optimizer(fires_df, resources, budget, asset_scores, horizon_hours, terrain_scores, lam=None):
    """
    Optimizer: minimize suppression cost + λ × residual damage.
    Uses nonlinear size-effectiveness multiplier (default K_SIZE = 1.25e-4).
    Returns allocation, coverage, cost, solver status, objective, uncovered acres, demand, and diagnostics.
    """
    if lam is None:
        lam = LAMBDA_DAMAGE
    fnames = fires_df["fire_name"].tolist()
    rnames = resources["resource"].tolist()
    ua     = dict(zip(rnames, resources["units_available"]))

    if "acres_per_hour" in resources.columns:
        aph_base = dict(zip(rnames, resources["acres_per_hour"]))
        cph      = dict(zip(rnames, resources["cost_per_hour"]))
    else:
        aph_base = dict(zip(rnames, resources["acres_per_day"] / 24.0))
        cph      = dict(zip(rnames, resources["cost_per_day"]  / 24.0))

    demand  = get_optimizer_demand(fires_df)
    danger  = dict(zip(fnames, fires_df["risk_score_100"]))
    ascore  = asset_scores or {f: 5.0 for f in fnames}
    tscores = terrain_scores or {f: 0.5 for f in fnames}

    aph_eff    = {(r,f): terrain_adj_cap(aph_base[r], tscores.get(f,0.5), r)
                  for r in rnames for f in fnames}
    max_rh     = {r: ua[r] * horizon_hours for r in rnames}
    max_single = max(aph_base.values())

    # Nonlinear size-effectiveness multiplier
    # eff[f] = 1 / (1 + K_SIZE * demand[f])
    # Holmes & Calkin (2013): empirical rates 14-93% of standard on large fires
    K_SIZE = 1.25e-4
    eff = {f: 1.0 / (1.0 + K_SIZE * demand[f]) for f in fnames}

    m = LpProblem("triage_mincost", LpMinimize)

    h = {(r,f): LpVariable(
            f"h_{r.replace(' ','_').replace('-','_')}_{f.replace(' ','_').replace('-','_')}",
            lowBound=0, cat="Integer")
         for r in rnames for f in fnames}
    c = {f: LpVariable(f"c_{f.replace(' ','_').replace('-','_')}", lowBound=0) for f in fnames}
    u = {f: LpVariable(f"u_{f.replace(' ','_').replace('-','_')}", lowBound=0) for f in fnames}

    # Objective: suppression cost + λ × residual damage
    total_cost_expr = lpSum(cph[r]*h[(r,f)] for r in rnames for f in fnames)
    residual_damage = lpSum(
        lam * (danger[f]/100.0) * (ascore.get(f,5.0)/10.0) * u[f] * DAMAGE_COST_PER_ACRE
        for f in fnames
    )
    m += total_cost_expr + residual_damage

    # C1: supply
    for r in rnames:
        m += lpSum(h[(r,f)] for f in fnames) <= max_rh[r]
    # C2: budget
    m += total_cost_expr <= budget

    for f in fnames:
        # C3: coverage with size-effectiveness multiplier
        m += c[f] == lpSum(aph_eff[(r,f)] * eff[f] * h[(r,f)] for r in rnames)
        # C4: uncovered slack
        m += u[f] >= demand[f] - c[f]
        # C5: overcoverage cap
        m += c[f] <= demand[f] + max_single

    m.solve(PULP_CBC_CMD(msg=0))

    alloc    = {f:{r:int(value(h[(r,f)]) or 0) for r in rnames} for f in fnames}
    coverage = {f:float(value(c[f]) or 0) for f in fnames}
    cost_h   = {f:sum(alloc[f][r]*cph[r] for r in rnames) for f in fnames}
    uncov    = {f:max(demand[f]-coverage[f], 0.0) for f in fnames}
    diagnostics = {f: {} for f in fnames}

    return alloc, coverage, cost_h, LpStatus[m.status], value(m.objective), uncov, demand, diagnostics

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

fires_raw, resources, assets = load_data()

with st.sidebar:
    st.markdown("## 🔥 Wildfire Triage")
    st.caption("Adjust conditions — model updates in real time.")
    st.markdown("---")

    st.markdown("### Weather")
    wind_spd = st.slider("Wind speed (m/s)", 0.5, 15.0,
                         float(round(fires_raw["wind_speed_mps"].fillna(5.0).mean(), 1)), 0.1)
    humidity = st.slider("Relative humidity (%)", 5.0, 95.0, 50.0, 1.0)
    temp_val = st.slider("Temperature (°C)", 5.0, 45.0, 18.0, 0.5, format="%.1f°C")

    st.markdown("---")
    st.markdown("### Planning horizon")
    horizon_hours = st.slider("Horizon (hours)", 2, 12, 6, 1,
                              help="Resource deployment window. Replaces daily framing.")

    st.markdown("---")
    st.markdown("### Budget for planning horizon")
    budget = st.slider("Budget ($)", 50_000, 300_000,
                       150_000, 10_000, format="$%d")

    st.markdown("---")
    st.markdown("### Fire Behavior Override")
    st.caption("Force a behavior to test what-if scenarios.")
    beh_options = ["actual", "Minimal", "Moderate", "Active", "Extreme"]
    fire_behaviors = {}
    for _, fire in fires_raw.iterrows():
        actual = str(fire.get("fire_behavior", "Minimal") or "Minimal")
        sel = st.selectbox(fire["fire_name"], beh_options, index=0,
                           key=f"beh_{fire['fire_name']}")
        fire_behaviors[fire["fire_name"]] = sel if sel != "actual" else actual

    st.markdown("---")
    use_assets = st.checkbox("Include infrastructure value (OSM)", value=True)


# ════════════════════════════════════════════════════════════════════════════
# APPLY OVERRIDES + RUN MODEL
# ════════════════════════════════════════════════════════════════════════════

fires = fires_raw.copy()
fires["wind_speed_mps"] = wind_spd
fires["humidity_pct"]   = humidity
fires["temperature_c"]  = temp_val
for f, beh in fire_behaviors.items():
    fires.loc[fires["fire_name"]==f, "fire_behavior"] = beh

_fires_tuple = (list(fires.columns), [list(r) for r in fires.itertuples(index=False)])
spread_graphs = build_all_graphs(wind_spd, _fires_tuple)

asset_scores  = compute_asset_scores(fires, assets, spread_graphs) if use_assets else None
fires_scored  = compute_risk(fires)
terrain_scores = compute_terrain_scores_dash(fires_scored)

# Single optimizer: resource-hours + terrain-adjusted capacity + nonlinear size effectiveness
alloc_opt, coverage_opt, horizon_cost, ip_status, obj_val, uncov, demand_map_opt, diagnostics = run_optimizer(
    fires_scored, resources, budget, asset_scores, horizon_hours, terrain_scores
)

demand_map       = demand_map_opt
alloc            = alloc_opt      # alias used in Tab 3 baseline comparisons
coverage         = coverage_opt   # alias
total_cost       = sum(horizon_cost.values())
budget_remaining = budget - total_cost
fire_order       = fires_scored.sort_values("priority_rank")["fire_name"].tolist()
sorted_names     = fire_order
rnames           = resources["resource"].tolist()
ua_map           = dict(zip(rnames, resources["units_available"]))
fnames           = fires_scored["fire_name"].tolist()
aph_map          = dict(zip(rnames, resources["acres_per_hour"] if "acres_per_hour" in resources.columns
                             else resources["acres_per_day"]/24.0))
cph_map          = dict(zip(rnames, resources["cost_per_hour"] if "cost_per_hour" in resources.columns
                             else resources["cost_per_day"]/24.0))


# ════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("# 🔥 Wildfire Resource Allocation — Washington State")
h1,h2,h3,h4,h5,h6,h7,h8 = st.columns(8)
h1.metric("Wind", f"{wind_spd} m/s")
h2.metric("Humidity", f"{humidity:.0f}%")
h3.metric("Temp", f"{temp_val:.1f}°C")
h4.metric("Horizon", f"{horizon_hours}h")
h5.metric("Budget", f"${budget:,}")
h6.metric("Cost over horizon", f"${total_cost:,.0f}", f"{total_cost/budget*100:.0f}% used")
h7.metric("Budget remaining", f"${budget_remaining:,.0f}")
h8.metric("Solver", ip_status)

top = fires_scored.sort_values("priority_rank").iloc[0]
if use_assets:
    insight = (
        f"🔑 <b>{top['fire_name']}</b> is top priority (risk {top['risk_score_100']:.0f}/100). "
        f"Infrastructure value is active — fires near hospitals, schools, and residential areas "
        f"score higher in the damage objective. Toggle off in the sidebar to see how the recommended allocation changes."
    )
else:
    insight = (
        f"🔑 <b>{top['fire_name']}</b> is top priority (risk {top['risk_score_100']:.0f}/100) "
        f"based on fire behavior and weather alone. "
        f"Enable 'Include infrastructure value' to factor in nearby hospitals and schools."
    )
st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📋  Decision Summary",
    "📊  Why This Allocation?",
    "⚖️  Compare Alternatives",
    "🔬  Sensitivity",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DECISION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("### Priority ranking & recommended allocation")
    st.caption(
        "Fires ranked by composite risk score (AHP-weighted: behavior 46.6%, weather 27.7%, "
        "complexity 16.1%, size 9.6%). Resources allocated by MILP optimizer minimizing "
        "suppression cost + λ × residual damage."
    )

    cols = st.columns(4)
    for i, (_, row) in enumerate(fires_scored.sort_values("priority_rank").iterrows()):
        f        = row["fire_name"]
        rank     = int(row["priority_rank"])
        color    = RANK_COLORS[rank]
        beh      = str(row.get("fire_behavior","—") or "—")
        cpx      = str(row.get("mgmt_complexity","—") or "—")
        dem      = demand_map.get(f, row["discovery_acres"])
        cov      = coverage[f]
        useful   = min(cov, dem)                         # acres actually meeting demand
        excess   = max(cov - dem, 0)                     # overshoot from lumpy resources
        dem_pct  = min(useful/dem*100, 100) if dem > 0 else 0
        beh_cls  = f"tag-{beh.lower()}" if beh.lower() in ["active","minimal","moderate","extreme"] else "tag-other"
        cpx_tag  = '<span class="tag tag-t1">Type 1</span>' if "Type 1" in cpx else ""
        ts        = terrain_scores.get(f, 0.5)
        cov_opt    = coverage_opt[f]
        dem_opt    = demand_map_opt.get(f, dem)
        useful_opt = min(cov_opt, dem_opt)
        dem_pct   = min(useful_opt/dem_opt*100, 100) if dem_opt > 0 else 0
        terrain_lbl = ("⛰ Steep/Remote" if ts < 0.35
                       else "〰 Moderate" if ts < 0.65 else "✅ Accessible")
        rh_parts  = [f"{v}h {r.split('(')[0].strip()}"
                     for r, v in alloc_opt[f].items() if v > 0]
        dispatch  = ", ".join(rh_parts) or "monitor only"
        with cols[i]:
            st.markdown(f"""
            <div class="fire-card" style="border-top-color:{color};">
              <div class="fire-rank-label" style="color:{color};">{RANK_LABELS[rank]}</div>
              <div class="fire-name">{f}</div>
              <div class="risk-num" style="color:{color};">{row['risk_score_100']:.0f}
                <span style="font-size:1rem;font-weight:400;color:#aaa;">/100</span>
              </div>
              <div style="margin-top:0.4rem;">
                <span class="tag {beh_cls}">{beh}</span>{cpx_tag}
              </div>
              <div class="fire-stat">
                📍 {dem_opt:,.0f} ac 6h response demand<br>
                ✅ {dem_pct:.0f}% demand met ({useful_opt:,.0f} ac)<br>
                {terrain_lbl} (access={ts:.2f})<br>
                💰 ${horizon_cost[f]:,.0f} over {horizon_hours}h<br>
                🚒 {dispatch}
              </div>
              <div style="background:#f0f0f0;border-radius:4px;height:5px;margin-top:0.7rem;">
                <div style="background:{color};height:5px;border-radius:4px;width:{dem_pct:.0f}%;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    total_rh = sum(sum(alloc_opt[f].values()) for f in fire_order)
    avg_cov  = np.mean([min(min(coverage_opt[f], demand_map_opt.get(f,1))/demand_map_opt.get(f,1)*100, 100)
                        for f in fire_order if demand_map_opt.get(f,0) > 0])
    with s1:
        st.markdown(f'<div class="stat-row"><div class="stat-val">${total_cost:,.0f}</div>'
                    f'<div class="stat-label">Cost over {horizon_hours}h horizon</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-row"><div class="stat-val">${budget_remaining:,.0f}</div>'
                    f'<div class="stat-label">Budget remaining</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-row"><div class="stat-val">{total_rh}</div>'
                    f'<div class="stat-label">Resource-hours deployed</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-row"><div class="stat-val">{avg_cov:.0f}%</div>'
                    f'<div class="stat-label">Avg response demand met</div></div>', unsafe_allow_html=True)
    with s5:
        st.markdown(f'<div class="stat-row"><div class="stat-val">{obj_val:,.0f}</div>'
                    f'<div class="stat-label">IP objective</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dispatch table (resource-hours)")
    st.caption("Values show resource-hours assigned. e.g. '4 Air Tanker' = 4 tanker-hours over the planning horizon.")
    dispatch_rows = []
    for f in sorted_names:
        rank = int(fires_scored.loc[fires_scored["fire_name"]==f, "priority_rank"].values[0])
        dem  = demand_map_opt.get(f, float(fires_scored.loc[fires_scored["fire_name"]==f,"discovery_acres"].values[0]))
        cov  = coverage_opt[f]
        ts   = terrain_scores.get(f, 0.5)
        row_d = {"Rank": f"#{rank}", "Fire": f,
                 "Risk Score": f"{fires_scored.loc[fires_scored['fire_name']==f,'risk_score_100'].values[0]:.1f}",
                 "Terrain Access": f"{ts:.2f}"}
        for r in rnames:
            v = alloc_opt[f].get(r, 0)
            row_d[f"{r.split('(')[0].strip()} (h)"] = v if v > 0 else "—"
        row_d["Demand (ac)"]            = f"{dem:,.0f}"
        row_d["Response demand met"]    = f"{min(min(cov,dem)/dem*100,100):.0f}%" if dem > 0 else "—"
        row_d["Capacity assigned (ac)"] = f"{cov:,.0f}"
        row_d["Unmet demand (ac)"]      = f"{max(dem-cov,0):,.0f}"
        row_d[f"Cost ({horizon_hours}h)"] = f"${horizon_cost[f]:,.0f}"
        dispatch_rows.append(row_d)
    st.dataframe(pd.DataFrame(dispatch_rows), hide_index=True, use_container_width=True)

    with st.expander("Terrain accessibility detail"):
        t_rows = []
        for _, row in fires_scored.sort_values("priority_rank").iterrows():
            name  = row["fire_name"]
            slope = float(fires.loc[fires["fire_name"]==name,"terrain_slope_pct"].values[0]) if "terrain_slope_pct" in fires.columns else float("nan")
            road  = float(fires.loc[fires["fire_name"]==name,"road_distance_km"].values[0]) if "road_distance_km" in fires.columns else float("nan")
            ts    = terrain_scores.get(name, 0.5)
            ground_eff = round(MIN_GROUND_EFFECTIVENESS + (1-MIN_GROUND_EFFECTIVENESS)*ts, 2)
            t_rows.append({"Fire": name, "Slope (%)": f"{slope:.0f}", "Road dist (km)": f"{road:.1f}",
                           "Access score": f"{ts:.2f}", "Ground eff": f"{ground_eff:.0%}", "Air eff": "100%"})
        st.dataframe(pd.DataFrame(t_rows), hide_index=True, use_container_width=True)
        st.caption("Ground effectiveness = fraction of nominal acres/hour that ground resources deliver on this terrain. Air resources unaffected.")

    with st.expander("Component scores (technical detail)"):
        comp_cols = ["fire_name","priority_rank","risk_score_100",
                     "size_score","weather_score","behavior_score","complexity_score",
                     "humidity_risk","wind_risk","temp_risk"]
        display_df = fires_scored[comp_cols].sort_values("priority_rank").round(3)
        display_df.columns = [c.replace("_"," ").title() for c in display_df.columns]
        st.dataframe(display_df, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — WHY THIS ALLOCATION?
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### What is driving each fire's risk score?")
    st.caption(
        "Stacked bars show each factor's absolute contribution to the danger score (0–100). "
        "Score = component × AHP weight × 100. Max possible = 100 (all components at ceiling). "
        "Asset value appears only in the IP damage objective — not the danger score — to avoid double-counting."
    )

    fs = fires_scored.set_index("fire_name")
    fig_risk = go.Figure()
    components = [
        ("Size (log acres)",  "size_score",       AHP_W["size"],       "#94a3b8"),
        ("Weather",           "weather_score",    AHP_W["weather"],    "#3b82f6"),
        ("Fire Behavior",     "behavior_score",   AHP_W["behavior"],   "#ef4444"),
        ("Mgmt Complexity",   "complexity_score", AHP_W["complexity"], "#f59e0b"),
    ]
    for label, col, weight, color in components:
        vals = fs.loc[sorted_names, col] * weight * 100
        fig_risk.add_trace(go.Bar(
            name=label, x=sorted_names, y=vals.values,
            marker_color=color, marker_line_width=0,
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.1f}} pts<extra></extra>",
        ))

    if use_assets and asset_scores:
        for f in sorted_names:
            ascore = asset_scores.get(f, 1.0)
            if ascore > 1.0:
                fig_risk.add_annotation(
                    x=f, y=105,
                    text=f"Asset: {ascore:.1f}×",
                    showarrow=False,
                    font=dict(size=9, color="#10b981"),
                    bgcolor="rgba(16,185,129,0.1)",
                )

    fig_risk.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        font=dict(family="DM Sans", size=12, color="#1a1a1a"),
        legend=dict(traceorder="reversed", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0, font_size=11),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(gridcolor="#f0f0f0", linecolor="#e5e5e5"),
        yaxis=dict(gridcolor="#f0f0f0", linecolor="#e5e5e5",
                   title="Points contributed to danger score (out of 100)",
                   title_font_size=11),
        height=360,
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")
    st.markdown("### Objective breakdown — suppression cost vs residual damage")
    st.caption(
        f"Objective: minimize suppression cost + λ × residual damage (both in $ equivalent). "
        f"Suppression cost = Σ resources × cost/day. "
        f"Residual damage = λ × (danger/100) × (asset/10) × uncovered acres × $500/acre. "
        f"λ={int(LAMBDA_DAMAGE)} (risk-aversion multiplier). "
        f"$500/acre = USFS average wildfire damage calibration. "
        f"Demand = 6h response demand (incident_size_6h) when available, "
        f"otherwise incident size or discovery acres."
    )

    diag_rows = []
    total_suppress, total_resid = 0, 0
    for f in sorted_names:
        danger_f = float(fires_scored.loc[fires_scored["fire_name"]==f, "risk_score_100"].values[0])
        asset_f  = asset_scores.get(f, 5.0) if asset_scores else 5.0
        uncov_f  = uncov.get(f, 0.0)
        suppress = horizon_cost[f]
        resid    = LAMBDA_DAMAGE * (danger_f/100.0) * (asset_f/10.0) * uncov_f * DAMAGE_COST_PER_ACRE
        total_suppress += suppress
        total_resid    += resid
        diag_rows.append({
            "Fire"                  : f,
            "Danger/100"            : f"{danger_f/100:.3f}",
            "Asset/10"              : f"{asset_f/10:.2f}",
            "Uncovered (ac)"        : f"{uncov_f:,.0f}",
            "Suppress cost ($/d)"   : f"${suppress:,.0f}",
            "Residual damage ($/d)" : f"${resid:,.0f}",
            "Obj contribution"      : f"${suppress + resid:,.0f}",
        })

    st.dataframe(pd.DataFrame(diag_rows), hide_index=True, use_container_width=True)

    d1, d2, d3 = st.columns(3)
    d1.metric("Suppression cost", f"${total_suppress:,.0f}", f"{total_suppress/budget*100:.0f}% of budget")
    d2.metric("Residual damage (λ-weighted)", f"${total_resid:,.0f}")
    d3.metric("Total objective", f"${obj_val:,.0f}")

    st.markdown("---")
    st.markdown("### Asset layer impact on allocation")
    st.caption(
        "Asset score enters the IP objective as a damage multiplier (danger × asset × uncovered). "
        "Enabling it shifts resources toward fires threatening hospitals, schools, and residential areas. "
        "The table compares uncovered acres and residual damage with and without the asset layer."
    )

    _, cov_no, _, _, _, uncov_no, _, _ = run_optimizer(fires_scored, resources, budget, None, horizon_hours, terrain_scores)
    _, cov_as, _, _, _, uncov_as, _, _ = run_optimizer(fires_scored, resources, budget, asset_scores, horizon_hours, terrain_scores)

    cmp_rows = []
    for f in sorted_names:
        dem    = demand_map.get(f, float(fires_scored.loc[fires_scored["fire_name"]==f,"discovery_acres"].values[0]))
        danger = float(fires_scored.loc[fires_scored["fire_name"]==f,"risk_score_100"].values[0])
        ascore = asset_scores.get(f, 5.0) if asset_scores else 5.0
        cmp_rows.append({
            "Fire"                      : f,
            "Danger score"              : f"{danger:.1f}",
            "Asset score"               : f"{ascore:.1f}",
            "Coverage (no assets)"      : f"{min(cov_no[f]/dem*100,100):.0f}%",
            "Coverage (with assets)"    : f"{min(cov_as[f]/dem*100,100):.0f}%",
            "Residual dmg (no assets)"  : f"${(danger/100)*1.0*uncov_no.get(f,0)*500:,.0f}",
            "Residual dmg (with assets)": f"${(danger/100)*(ascore/10)*uncov_as.get(f,0)*500:,.0f}",
        })
    st.dataframe(pd.DataFrame(cmp_rows), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Directional spread footprint")
    st.caption(
        "Wind-weighted Dijkstra spread model — a local 4 km × 4 km grid (40×40 cells, 100m each) "
        "centered on the reported fire coordinate. This is not a full incident perimeter map. "
        "Downwind ~3× faster than upwind. Used internally to identify which assets fall "
        "within each fire's 12-hour threat envelope."
    )

    sel_fire = st.selectbox(
        "Fire to inspect",
        sorted_names,
        format_func=lambda f: f"#{int(fires_scored.loc[fires_scored['fire_name']==f,'priority_rank'].values[0])} {f}"
    )
    t_hours = st.slider("Exposure horizon (hours)", 1, 12, 6, 1, key="spread_t")

    fire_row = fires_scored[fires_scored["fire_name"] == sel_fire].iloc[0]
    G_sel    = spread_graphs[sel_fire]
    ignition = (GRID_SIZE // 2, GRID_SIZE // 2)
    lengths  = nx.single_source_dijkstra_path_length(G_sel, ignition, weight="weight")
    arrival  = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    for (r, c), t in lengths.items():
        arrival[r, c] = t

    heatmap_z     = np.where(np.isnan(arrival), t_hours*2, np.minimum(arrival, t_hours*1.5))
    heatmap_z_inv = t_hours*1.5 - heatmap_z
    heatmap_z_inv = np.where(np.isnan(arrival), 0, heatmap_z_inv)

    fig_heat = go.Figure()
    fig_heat.add_trace(go.Heatmap(
        z=heatmap_z_inv,
        colorscale=[[0.0,"rgba(240,240,240,0.3)"],[0.3,"rgba(255,200,100,0.5)"],
                    [0.6,"rgba(255,120,40,0.8)"],[1.0,"rgba(180,20,20,1.0)"]],
        showscale=True,
        colorbar=dict(
            title=dict(text="Spread pressure<br>(early = hot)", font=dict(size=10)),
            tickvals=[], ticktext=[], thickness=12, len=0.6,
        ),
        hovertemplate="Row %{y}, Col %{x}<br>Arrival: %{customdata:.1f}h<extra></extra>",
        customdata=arrival,
        zmin=0, zmax=t_hours*1.5,
    ))
    fig_heat.add_trace(go.Scatter(
        x=[ignition[1]], y=[ignition[0]], mode="markers",
        marker=dict(symbol="star", size=14, color="yellow", line=dict(color="black", width=1.5)),
        name="Ignition", hovertemplate="Ignition<extra></extra>",
    ))

    wind_from  = float(fire_row.get("wind_dir_deg") or 270)
    wind_to_r  = np.radians((wind_from + 180) % 360)
    ax = ignition[1] + 6*np.sin(wind_to_r)
    ay = ignition[0] - 6*np.cos(wind_to_r)
    fig_heat.add_annotation(
        x=ax, y=ay, ax=ignition[1], ay=ignition[0],
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3, arrowcolor="white",
        text=f"Wind →<br>{wind_spd:.1f} m/s",
        font=dict(color="white", size=10), bgcolor="rgba(0,0,0,0.5)", borderpad=3,
    )

    if assets is not None:
        fire_assets = assets[assets["fire_name"] == sel_fire]
        if not fire_assets.empty:
            LAT_PER_M = 1/111_320
            lon_m     = 1/(111_320*np.cos(np.radians(fire_row["fire_lat"])))
            centre    = GRID_SIZE//2
            acols, arows, awts, alabels = [], [], [], []
            for _, asset in fire_assets.iterrows():
                dr = (fire_row["fire_lat"] - asset["centroid_lat"]) / (CELL_M*LAT_PER_M)
                dc = (asset["centroid_lon"] - fire_row["fire_lon"]) / (CELL_M*lon_m)
                rg, cg = centre+dr, centre+dc
                if 0 <= rg < GRID_SIZE and 0 <= cg < GRID_SIZE:
                    arows.append(rg); acols.append(cg)
                    awts.append(asset.get("asset_weight", 1.0))
                    for col in ["amenity","building","landuse"]:
                        if col in asset.index and pd.notna(asset[col]):
                            alabels.append(str(asset[col])); break
                    else:
                        alabels.append("asset")
            if acols:
                threatened = [not np.isnan(arrival[int(round(r)),int(round(c))]) and
                              arrival[int(round(r)),int(round(c))] <= t_hours
                              for r, c in zip(arows, acols)]
                fig_heat.add_trace(go.Scatter(
                    x=acols, y=arows, mode="markers",
                    marker=dict(symbol="diamond",
                                size=[max(6, w*2) for w in awts],
                                color=["rgba(255,50,50,0.9)" if t else "rgba(100,200,100,0.7)"
                                       for t in threatened],
                                line=dict(color="white", width=1)),
                    name="Assets (red=threatened)",
                    hovertemplate="%{text}<extra></extra>", text=alabels,
                ))

    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111",
        font=dict(family="DM Sans", size=12),
        xaxis=dict(title="Grid column (E →)", showgrid=False, range=[0,GRID_SIZE], constrain="domain"),
        yaxis=dict(title="Grid row (N →)", showgrid=False, range=[GRID_SIZE,0],
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10), height=440,
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    fm = get_fuel_multiplier(fire_row.get("fuel_group"), fire_row.get("fire_behavior"))
    st.caption(
        f"**{sel_fire}** · fuel multiplier {fm:.1f}× · wind {wind_spd:.1f} m/s from {wind_from:.0f}° · "
        f"t={t_hours}h footprint shown. Assets within the red zone are included in the damage penalty."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — COMPARE ALTERNATIVES
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("### Does the IP optimizer outperform simple rules?")
    st.caption(
        "All strategies use the same budget and resource pool. "
        "Coverage = acres covered / demand acres. "
        "Risk-weighted demand met weights each fire's result by its danger score — "
        "it measures whether resources went to the most dangerous fires."
    )

    danger_map = dict(zip(fnames, fires_scored["risk_score_100"]))

    # Baselines use resource-hours × aph to compute coverage (consistent with optimizer)
    max_rh_base = {r: ua_map[r] * horizon_hours for r in rnames}

    def cov_from_alloc(alloc_d, f):
        dem = demand_map.get(f, 0)
        size_eff = 1.0 / (1.0 + 1.25e-4 * dem)   # same default K_SIZE as optimizer
        return sum(alloc_d[f].get(r, 0) * aph_map[r] * size_eff for r in rnames)

    def total_avg_cov(alloc_d):
        total = 0
        for f in fnames:
            cov = cov_from_alloc(alloc_d, f)
            dem = demand_map.get(f, 1)
            total += min(cov/dem*100, 100) if dem > 0 else 0
        return total / len(fnames)

    def risk_weighted_cov(alloc_d):
        num, den = 0, 0
        for f in fnames:
            cov = cov_from_alloc(alloc_d, f)
            dem = demand_map.get(f, 1)
            pct = min(cov/dem*100, 100) if dem > 0 else 0
            num += danger_map[f] * pct
            den += danger_map[f]
        return num / den if den > 0 else 0

    def uncovered_high_risk(alloc_d, top_n=2):
        top_fires = fires_scored.sort_values("priority_rank").head(top_n)["fire_name"].tolist()
        return sum(max(demand_map.get(f,0) - cov_from_alloc(alloc_d, f), 0) for f in top_fires)

    def total_spend(alloc_d):
        return sum(alloc_d[f].get(r,0)*cph_map[r] for f in fnames for r in rnames)

    # Risk-score proportional baseline (resource-hours)
    risk_alloc = {f:{r:0 for r in rnames} for f in fnames}
    risk_total = sum(danger_map.values())
    rem_budget = budget
    for r in rnames:
        rh_left = max_rh_base[r]
        for f in sorted(fnames, key=lambda f: -danger_map[f]):
            share = min(int(danger_map[f] / risk_total * max_rh_base[r]), rh_left)
            if share * cph_map[r] <= rem_budget:
                risk_alloc[f][r] = share
                rh_left   -= share
                rem_budget -= share * cph_map[r]

    # Acreage-proportional baseline (resource-hours)
    size_alloc = {f:{r:0 for r in rnames} for f in fnames}
    size_total = sum(demand_map.values())
    rem_budget2 = budget
    for r in rnames:
        rh_left2 = max_rh_base[r]
        for f in sorted(fnames, key=lambda f: -demand_map.get(f,0)):
            share = min(int(demand_map.get(f,0) / max(size_total,1) * max_rh_base[r]), rh_left2)
            if share * cph_map[r] <= rem_budget2:
                size_alloc[f][r] = share
                rh_left2    -= share
                rem_budget2 -= share * cph_map[r]

    # Equal-split baseline (resource-hours)
    equal_alloc = {f:{r:0 for r in rnames} for f in fnames}
    rem_budget3 = budget
    for r in rnames:
        per_fire = max_rh_base[r] // len(fnames)
        for f in fnames:
            if per_fire * cph_map[r] <= rem_budget3:
                equal_alloc[f][r] = per_fire
                rem_budget3 -= per_fire * cph_map[r]

    baselines = {
        "🏆 IP Optimizer (ours)" : alloc_opt,
        "Risk-score proportional": risk_alloc,
        "Acreage proportional"   : size_alloc,
        "Equal split"            : equal_alloc,
    }

    baseline_rows = []
    for label, alloc_b in baselines.items():
        avg_c    = total_avg_cov(alloc_b)
        rwc      = risk_weighted_cov(alloc_b)
        uncov_hr = uncovered_high_risk(alloc_b)
        cost     = total_spend(alloc_b)
        baseline_rows.append({
            "Strategy"                        : label,
            "Avg response demand met"         : f"{avg_c:.1f}%",
            "Risk-weighted demand met"        : f"{rwc:.1f}%",
            "Unmet demand (top-2 risk fires)" : f"{uncov_hr:,.0f} ac",
            f"Cost over {horizon_hours}h"     : f"${cost:,.0f}",
        })
    st.dataframe(pd.DataFrame(baseline_rows), hide_index=True, use_container_width=True)
    st.caption(
        "Risk-weighted response demand met shows whether resources went to the most dangerous fires. "
        "The IP objective minimizes suppression cost + λ × risk-weighted residual damage. "
        "Risk-weighted demand met is an evaluation metric, not the optimizer objective."
    )

    st.markdown("---")
    st.markdown("### Per-fire response demand met by strategy")

    fig_comp = go.Figure()
    strategy_colors = {
        "🏆 IP Optimizer (ours)" : "#dc2626",
        "Risk-score proportional": "#3b82f6",
        "Acreage proportional"   : "#f59e0b",
        "Equal split"            : "#6b7280",
    }

    for label, alloc_b in baselines.items():
        coverages = []
        for f in sorted_names:
            cov = cov_from_alloc(alloc_b, f)
            dem = demand_map.get(f, 1)
            coverages.append(min(cov/dem*100, 100) if dem > 0 else 0)
        fig_comp.add_trace(go.Bar(
            name=label,
            x=[f"#{int(fires_scored.loc[fires_scored['fire_name']==f,'priority_rank'].values[0])} {f}"
               for f in sorted_names],
            y=coverages,
            marker_color=strategy_colors[label],
            opacity=0.85,
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.0f}}% demand met<extra></extra>",
        ))

    fig_comp.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        font=dict(family="DM Sans", size=12, color="#1a1a1a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font_size=11),
        xaxis=dict(gridcolor="#f0f0f0", linecolor="#e5e5e5"),
        yaxis=dict(title="Response demand met (%)", gridcolor="#f0f0f0", linecolor="#e5e5e5",
                   range=[0,105], ticksuffix="%"),
        margin=dict(l=10, r=10, t=60, b=10), height=360,
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("### How robust is the recommendation?")
    st.caption(
        "Two key assumptions are tested here: the budget for planning horizon (an operational constraint) "
        "and λ (a risk-preference parameter). All runs use the optimizer. "
        "If the allocation remains stable across these ranges, the recommendation is robust to these assumptions."
    )

    st.markdown("#### Budget for planning horizon — sensitivity")
    st.caption(
        "How does per-fire response demand met change as the budget varies? "
        ""
    )

    budget_steps = [50_000, 75_000, 100_000, 125_000, 150_000,
                    175_000, 200_000, 250_000, 300_000]
    sensitivity  = {f: [] for f in fires_scored["fire_name"]}

    for b in budget_steps:
        _, cov_b, _, _, _, _, dem_b, _ = run_optimizer(
            fires_scored, resources, b, asset_scores, horizon_hours, terrain_scores)
        for f in fires_scored["fire_name"]:
            dem = dem_b.get(f, 1)
            sensitivity[f].append(min(cov_b[f]/dem*100, 100) if dem > 0 else 0)

    fig_bud = go.Figure()
    for _, row in fires_scored.sort_values("priority_rank").iterrows():
        f    = row["fire_name"]
        rank = int(row["priority_rank"])
        fig_bud.add_trace(go.Scatter(
            x=[b/1000 for b in budget_steps], y=sensitivity[f],
            name=f"#{rank} {f}", mode="lines+markers",
            line=dict(color=RANK_COLORS[rank], width=2.5), marker=dict(size=6),
            hovertemplate=f"<b>{f}</b><br>$%{{x}}k → %{{y:.0f}}% demand met<extra></extra>",
        ))
    fig_bud.add_vline(x=budget/1000, line_dash="dash", line_color="#666", line_width=1.5,
                      annotation_text=f" Current: ${budget//1000}k",
                      annotation_font=dict(family="DM Sans", size=11, color="#666"),
                      annotation_position="top right")
    fig_bud.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        font=dict(family="DM Sans", size=12, color="#1a1a1a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font_size=11),
        xaxis=dict(title="Budget for planning horizon ($k)", gridcolor="#f0f0f0", linecolor="#e5e5e5",
                   range=[40, 320]),
        yaxis=dict(title="Response demand met (%)", gridcolor="#f0f0f0", linecolor="#e5e5e5",
                   range=[0,105], ticksuffix="%"),
        margin=dict(l=10, r=10, t=50, b=10), height=340,
    )
    st.plotly_chart(fig_bud, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Lambda (λ) sensitivity — damage tradeoff")
    st.caption(
        "λ controls how much the optimizer penalizes leaving high-risk fires uncovered. "
        f"Current λ={int(LAMBDA_DAMAGE)}. Higher λ = more risk-averse. "
        "Flat lines = that fire's response demand met is insensitive to this choice."
    )

    lambda_vals = [10, 25, 50, 75, 100]
    lam_sens    = {f: [] for f in fnames}

    for lam in lambda_vals:
        _, cov_l, _, _, _, _, dem_l, _ = run_optimizer(
            fires_scored, resources, budget, asset_scores,
            horizon_hours, terrain_scores, lam=lam)
        for f in fires_scored["fire_name"]:
            dem = dem_l.get(f, 1)
            lam_sens[f].append(min(cov_l[f]/dem*100, 100) if dem > 0 else 0)

    fig_lam = go.Figure()
    for _, row in fires_scored.sort_values("priority_rank").iterrows():
        f    = row["fire_name"]
        rank = int(row["priority_rank"])
        fig_lam.add_trace(go.Scatter(
            x=lambda_vals, y=lam_sens[f],
            name=f"#{rank} {f}", mode="lines+markers",
            line=dict(color=RANK_COLORS[rank], width=2.5), marker=dict(size=6),
            hovertemplate=f"<b>{f}</b><br>λ=%{{x}} → %{{y:.0f}}% demand met<extra></extra>",
        ))
    fig_lam.add_vline(x=LAMBDA_DAMAGE, line_dash="dash", line_color="#666", line_width=1.5,
                      annotation_text=f" Current λ={int(LAMBDA_DAMAGE)}",
                      annotation_font=dict(size=11, color="#666"))
    fig_lam.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        font=dict(family="DM Sans", size=12, color="#1a1a1a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font_size=11),
        xaxis=dict(title="Lambda (λ)", gridcolor="#f0f0f0", linecolor="#e5e5e5"),
        yaxis=dict(title="Response demand met (%)", gridcolor="#f0f0f0", linecolor="#e5e5e5",
                   range=[0,105], ticksuffix="%"),
        margin=dict(l=10, r=10, t=50, b=10), height=340,
    )
    st.plotly_chart(fig_lam, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Planning horizon — response demand met")
    st.caption(f"How does response demand met change as the planning horizon shifts? Current: {horizon_hours}h. Uses unified optimizer.")

    horizon_steps = [2, 3, 4, 6, 8, 10, 12]
    hor_sens = {f: [] for f in fnames}
    for h in horizon_steps:
        _, cov_h, _, _, _, _, dem_h, _ = run_optimizer(fires_scored, resources, budget,
                                                   asset_scores, h, terrain_scores)
        for f in fnames:
            dem = dem_h.get(f, 1)
            hor_sens[f].append(min(cov_h[f]/dem*100, 100) if dem > 0 else 0)

    fig_hor = go.Figure()
    for _, row in fires_scored.sort_values("priority_rank").iterrows():
        f    = row["fire_name"]
        rank = int(row["priority_rank"])
        fig_hor.add_trace(go.Scatter(
            x=horizon_steps, y=hor_sens[f],
            name=f"#{rank} {f}", mode="lines+markers",
            line=dict(color=RANK_COLORS[rank], width=2.5), marker=dict(size=6),
            hovertemplate=f"<b>{f}</b><br>%{{x}}h → %{{y:.0f}}% demand met<extra></extra>",
        ))
    fig_hor.add_vline(x=horizon_hours, line_dash="dash", line_color="#666", line_width=1.5,
                      annotation_text=f" Current: {horizon_hours}h")
    fig_hor.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        font=dict(family="DM Sans", size=12, color="#1a1a1a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font_size=11),
        xaxis=dict(title="Planning horizon (hours)", gridcolor="#f0f0f0", linecolor="#e5e5e5"),
        yaxis=dict(title="Response demand met (%)", gridcolor="#f0f0f0", linecolor="#e5e5e5",
                   range=[0, 105], ticksuffix="%"),
        margin=dict(l=10, r=10, t=50, b=10), height=340,
    )
    st.plotly_chart(fig_hor, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Pipeline: AHP risk scoring (nonlinear weather) → MILP optimizer (PuLP/CBC) → "
    "Elliptical Dijkstra asset exposure (OSM). "
    "Data: NIFC WFIGS · NOAA weather.gov · OpenStreetMap. "
    "Spread model is a directional exposure approximation, not a wildfire forecast."
)