# Wildfire Resource Allocation — Decision Support System

A four-layer decision optimization pipeline that answers a real operational question: **given 4 simultaneous wildfires and a fixed daily budget, which fire gets resources first, and how many units go where?**

Built on real data from NIFC, NOAA, and OpenStreetMap. Interactive dashboard built with Streamlit.

---

## The Problem

When multiple fires ignite simultaneously, fire operations centers must decide how to allocate limited resources — engines, dozers, helicopters, air tankers, hand crews — across all active incidents. Prioritizing by raw acreage alone is misleading: a 500-acre fire with active behavior threatening a hospital cluster may be more urgent than a 15,000-acre fire burning in uninhabited forest with high humidity.

This system formalizes that decision with a four-layer model.

---

## Architecture

```
Layer 0: AHP Weight Derivation
    ↓
Layer 1: Risk Scoring (absolute final scale; scenario-relative size component)
    ↓
Layer 2: IP Optimizer  ←─────────────────────────────┐
    ↓                                                  │
Layer 3: Spread Simulation (dynamic, suppression      │ rolling reallocation
         feedback, diurnal weather, terrain)          │ with sectoral suppression
    ↓                                                  │
Layer 4: OSM Asset Value (elliptical footprint) ──────┘
```

---

## Layer 0 — AHP Weight Derivation

Weights are derived using the Analytic Hierarchy Process (Saaty 1980) from a pairwise comparison matrix justified by NWCG doctrine and NOAA Red Flag criteria.

| Component | AHP Weight | Rationale |
|-----------|-----------|-----------|
| Fire behavior | 46.6% | Observed outcome — directly drives spread rate |
| Weather | 27.7% | Forward-looking escalation signal |
| Complexity | 16.1% | Lagging indicator of escaped initial attack |
| Size | 9.6% | Least predictive of future danger |

Consistency Ratio CR = 0.0115 (< 0.10 threshold — acceptable).

---

## Layer 1 — Risk Scoring

Computes a composite fire danger score (0–100) for each fire. Weather, behavior, and complexity are scaled on fixed absolute/ordinal scales. Size uses log normalization within the selected scenario, so the size component is scenario-relative. The final weighted score is **not rescaled to the top fire**; it is reported as `fire_danger × 100`. A score of 75 means the weighted component score is 75% of the theoretical maximum under the current scoring design.

**Weather subweights (nonlinear):**
- Humidity (30%): exponential decay `exp(−0.03 × RH)` — risk rises steeply below 30% RH
- Wind speed (45%): exponential growth `exp(0.05 × W_kmh)` — risk accelerates at high speed
- Temperature (25%): linear 0°C → 45°C

Size score: log-normalized within the selected scenario (relative sizing).

Behavior and complexity: NIFC ordinal lookup (Minimal=0.2 → Extreme=1.0; Type 5=0.1 → Type 1=1.0).

Asset value is intentionally excluded from the risk score — it enters the IP objective only, to avoid double-counting.

---

## Layer 2 — Integer Programming Optimizer

Formulated as a MILP using PuLP (CBC solver).

**Decision variable:** `x[r][f]` = integer units of resource type `r` dispatched to fire `f`

**Objective (dimensionally calibrated — both terms in $/day equivalent):**
```
minimize  suppression_cost + λ × residual_damage

suppression_cost   = Σ cost_per_day[r] × x[r][f]                      ($/day)
residual_damage[f] = (danger[f]/100) × (asset[f]/10)
                     × uncovered[f] × $500/acre                         ($/day equiv)
```

- `danger[f]/100` = fraction of theoretical worst-case fire (absolute, batch-independent)
- `asset[f]/10` = fraction of highest-exposure fire
- `$500/acre` = calibrated to USFS average wildfire damage cost
- `λ` = dimensionless risk-preference multiplier (λ=1: risk-neutral; λ>1: risk-averse)

Objective values are interpretable in dollar-equivalent terms.

**Constraints:**
- `C1` Supply: units dispatched ≤ units available per resource type
- `C2` Budget: total daily cost ≤ daily budget
- `C3` Cap: coverage ≤ 2× demand per fire (no over-deployment)
- `C4` Uncovered: slack variable ≥ demand − coverage (non-negative by variable bound)

**Demand** is fixed at `discovery_acres` throughout the optimization horizon. Layer 3 dynamic spread updates risk scores but not optimizer demand — this keeps coverage % metrics consistent across timesteps and is operationally defensible (demand reflects the known footprint at dispatch time, not projected future spread).

---

## Layer 3 — Fire Spread Simulation

Models fire propagation as a shortest-path problem on a directed weighted graph. **This is a simplified directional exposure model, not a wildfire forecast.**

- **Grid:** 40×40 cells, 100m × 100m (4km × 4km per fire)
- **Edges:** 8-connected, directed

**Physical factors on edge weights (multiplicative):**

1. **Wind** (Rothermel-inspired): `wind_factor = 1 + 2 × cos(bearing − wind_to_deg)`  
   Downwind ~3× faster than upwind. NOAA FROM-direction corrected to TO-direction (+180°).

2. **Fuel type**: spread multiplier by NIFC `PredominantFuelGroup`  
   (grass=1.6×, shrub=1.3×, timber=1.0×, nonburnable=0.3×). Falls back to behavior when field missing.

3. **Suppression (sectoral)**: each resource type protects a specific angular sector:
   - Air tanker → head fire (downwind ±90°, cosine-tapered)
   - Helicopter → flanks
   - Dozer → left flank
   - Hand crew → heel (upwind)
   - Effect: `travel_h × (1 + α × sector_strength)` where α=0.15
   - Cosine taper (not hard cutoff) prevents discontinuous behavior from small wind shifts

4. **Slope/terrain**: `v_slope = v × (1 + β × slope_grade)`, β=0.5  
   SRTM elevation via Open Elevation API; falls back to synthetic Gaussian hill if unavailable.

5. **Dynamic weather (diurnal profile)**:  
   Assumes t=0 = 08:00 local. Sinusoidal peak at t=6h (14:00):
   - Wind: +25% at peak
   - RH: −20 percentage points at peak
   - Temp: +8°C at peak
   - Wind direction: +15° clockwise at peak (thermal upslope effect)
   Deterministic — more defensible than random walk.

**Acreage accounting:**
```
current_acres(t) = discovery_acres + acres_at_time(t)
acres_at_time(0) = 0   (ignition cell excluded — already in discovery_acres)
```

**Rolling reallocation (feedback loop):**
At t = 0, 3, 6, 9, 12 hours:
1. Dynamic weather updated via diurnal profile
2. Suppression strength from previous allocation → sectoral graph rebuild
3. Risk re-scored with updated acreage + weather
4. IP optimizer re-runs
5. New allocation feeds back into next step

**Scenario analysis (Layer 3C):**
3 scenarios × 20 runs each, perturbing wind (±10%) and fuel multiplier (±15%).
**Note:** This is scenario analysis, not uncertainty quantification. Perturbations are assumed, not calibrated from historical data. Humidity is not perturbed — it affects risk scoring (Layer 1) but not spread graph edge weights.

---

## Layer 4 — OSM Infrastructure Asset Value

Replaces acreage proxy in the objective with spatially-aware damage exposure.

- Assets from OpenStreetMap via `osmnx`: hospitals, schools, fire stations, residential/commercial areas
- Spatial join via elliptical Dijkstra spread footprint (not haversine circle) — assets upwind are not counted
- **Normalization:** percentile-anchored (p5→1, p95→10). Prevents small absolute differences between similarly-exposed fires from being artificially amplified across the full 1–10 scale.

**Asset weights (life-safety hierarchy):**

| Asset type | Weight |
|------------|--------|
| Hospital | 10 |
| Fire station | 9 |
| School | 8 |
| Clinic | 7 |
| Police | 6 |
| Apartments | 5 |
| Residential | 4 |
| Industrial | 3 |
| Commercial | 2 |

---

## Key Result

With infrastructure value disabled, **Sawmill Creek** ranks #1 — Active behavior, Type 1 complexity, highest behavioral risk.

With infrastructure value enabled, **Thorp Road** may rank higher despite lower behavioral risk — its spread footprint contains more critical infrastructure. **A model that ignores asset value sends resources to the wrong fire.**

---

## Known Model Limitations

These are documented here for transparency, not as deficiencies — they reflect deliberate tradeoffs appropriate for a prototype decision-support system.

### Mathematical limitations

**Correlated risk components.** Behavior, complexity, and weather are not independent — behavior is partly driven by weather; complexity reflects historical behavior. The AHP weighted sum aggregates partially correlated signals, which weakens strict interpretability. The planned complexity sensitivity analysis (variants A/B/C) tests whether removing complexity materially changes rankings.

**Linear suppression effectiveness.** The optimizer assumes 2 engines suppress exactly double 1 engine. Real suppression saturates — the first crew matters most. This makes the MILP clean and solvable but less behaviorally realistic. A piecewise-linear or concave coverage function would be more accurate.

**No interaction effects in risk scoring.** The model cannot express that high wind + low humidity together are disproportionately dangerous (super-additive). A multiplicative or interaction-term formulation would capture this, at the cost of interpretability.

**AHP weights are subjective.** CR < 0.10 confirms internal consistency but not objective correctness. Weights reflect expert judgment, not empirical calibration. This is unavoidable in most DSS frameworks; the sensitivity analysis tests how robust rankings are to weight perturbations.

**Objective λ is a preference parameter.** Even with dollar calibration, λ is ultimately a choice about risk tolerance, not a derived quantity. Different values of λ produce different allocations; sensitivity analysis over λ = 10–100 is included in the dashboard.

### Physical limitations

**Flat terrain** (with synthetic/SRTM slope approximation). The terrain model captures first-order slope effects but ignores aspect, canopy cover, or fine-scale topographic channeling.

**Static fuel map.** One fuel multiplier per fire, derived from `PredominantFuelGroup` or fire behavior. Real fires move through spatially heterogeneous fuel mosaics.

**Suppression sectors assumed fixed.** Resources are assigned to a sector at dispatch time. In reality, redeployment within a fire is continuous.

**4km × 4km grid.** For large fires (Bolt Creek: 14,820 ac), the spread model covers only a fraction of the actual fire footprint. The grid represents the *marginal* spread dynamics at the active perimeter, not the full incident.

---

## Data Sources

| Source | What it provides |
|--------|-----------------|
| [NIFC WFIGS](https://data-nifc.opendata.arcgis.com) | Fire locations, acreage, behavior, complexity, fuel model |
| [NOAA weather.gov API](https://api.weather.gov) | Wind speed, direction, temperature, humidity |
| [OpenStreetMap via osmnx](https://osmnx.readthedocs.io) | Infrastructure assets |
| [Open Elevation API](https://api.open-elevation.com) | SRTM terrain elevation |

All sources are free and require no API keys.

---

## Project Structure

```
Forest_Fire_Project/
├── wildfire_data/
│   ├── scenario_fires.csv       # Fire scenario with weather
│   ├── resources.csv            # Firefighting resource types and costs
│   ├── osm_assets.geojson       # OSM infrastructure assets
│   ├── asset_scores.csv         # Per-fire asset scores
│   └── dynamic_timeline.csv     # Rolling allocation timeline (t=0..12h)
│
├── fetch_data.py                # Pull fire + weather data from NIFC/NOAA
├── fetch_osm_assets.py          # Pull OSM infrastructure assets
├── wildfire_triage.py           # Layers 0–3: AHP + risk + IP + spread
├── asset_layer.py               # Layer 4: Asset value integration
├── dashboard.py                 # Streamlit interactive dashboard
│
├── .streamlit/
│   └── config.toml              # Forces light theme
└── README.md
```

---

## Running the Project

### 1. Install dependencies

```bash
pip install pandas numpy networkx pulp osmnx geopandas shapely \
            streamlit plotly requests
```

### 2. Fetch data (requires internet)

```bash
python fetch_data.py          # pulls NIFC fire locations + NOAA weather
python fetch_osm_assets.py    # pulls OSM infrastructure assets
```

### 3. Run the full model

```bash
python wildfire_triage.py     # Layers 0–3: prints results + sensitivity analyses
```

### 4. Launch the dashboard

```bash
streamlit run dashboard.py
```

---

## Dashboard

Four tabs:

| Tab | Contents |
|-----|----------|
| **Overview** | Priority ranking, resource dispatch, coverage %, cost |
| **Risk Analysis** | AHP factor contributions, dispatch table, component scores |
| **Spread Exposure** | Directional heatmap (before/after suppression toggle), scenario analysis, terrain, diurnal weather |
| **Diagnostics & Sensitivity** | Objective breakdown, baseline comparisons, λ sensitivity, budget sensitivity, coverage-cap sensitivity, asset layer impact |

Sidebar: adjust wind speed, humidity, temperature, budget, fire behavior overrides, infrastructure toggle. IP re-runs on every change.

---

## Cost Function

Based on the **C+NVC (Cost + Net Value Change)** framework (Martell 1982, Granda et al. 2023):

```
Objective = suppression_cost + λ × residual_damage
          = Σ (cost_per_day[r] × units[r][f])
          + λ × Σ (danger[f]/100) × (asset[f]/10) × uncovered[f] × $500/acre
```

Both terms are in dollar-equivalent units. λ is a dimensionless risk-preference multiplier (default 50, tested 10–100 in sensitivity analysis).

---

## References

- Granda, J.M. et al. (2023). *A review of optimization models for wildfire suppression resource allocation.* — Literature review of 36 papers; source of C+NVC framework.
- Martell, D.L. (1982). *A review of operational research studies in forest fire management.* — Original C+NVC framework.
- Saaty, T.L. (1980). *The Analytic Hierarchy Process.* — AHP weight derivation methodology.
- Cal Poly Pomona Operations Research Group. *Wildfire Resource Allocation using Linear Programming.* — Resource cost and capability estimates.
