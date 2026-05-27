# Wildfire Resource Allocation — Decision Support System

A four-layer decision-support pipeline that answers a real operational question: **given simultaneous wildfires and a fixed budget, which fire gets resources first, and how many units go where?**

Built on real fire data from NIFC and NOAA. Interactive dashboard built with Streamlit.

---

## The Problem

When multiple fires ignite simultaneously, fire operations centers must decide how to allocate limited resources — engines, dozers, helicopters, air tankers, hand crews — across all active incidents. Prioritizing by raw acreage alone is misleading: a 200-acre fire threatening an urban corridor may be more urgent than a 1,800-acre fire in remote terrain with lower behavioral intensity.

This system formalizes that decision with a four-layer model grounded in NWCG doctrine and fire economics literature.

---

## Architecture

```
Layer 0: AHP Weight Derivation
    ↓
Layer 1: Risk Scoring
    ↓
Layer 2: MILP Optimizer (minimize suppression cost + λ × residual damage)
    ↓
Layer 3: Spread Simulation (dynamic, suppression feedback, terrain)
    ↓
Layer 4: OSM Asset Value (elliptical footprint)
```

---

## Layer 0 — AHP Weight Derivation

Weights derived using the Analytic Hierarchy Process (Saaty 1980), justified by NWCG doctrine and NOAA Red Flag criteria.

| Component | AHP Weight | Rationale |
|-----------|-----------|-----------|
| Fire behavior | 46.6% | Observed outcome — directly drives spread rate |
| Weather | 27.7% | Forward-looking escalation signal |
| Complexity | 16.1% | Lagging indicator of escaped initial attack |
| Size | 9.6% | Least predictive of future danger |

Consistency Ratio CR = 0.0115 (< 0.10 — acceptable).

---

## Layer 1 — Risk Scoring

Composite fire danger score (0–100) per fire. All components use fixed absolute or ordinal scales; size uses log-normalization on `incident_size_6h` (6h suppression demand) within the selected scenario. Asset value is excluded here — it enters the optimizer only, avoiding double-counting.

**Weather subweights (nonlinear):**
- Humidity (30%): exponential decay — risk rises steeply below 30% RH
- Wind speed (45%): exponential growth — risk accelerates at high speed
- Temperature (25%): linear 0°C → 45°C

**Demand hierarchy for size scoring:**
`incident_size_6h` → `incident_size` → `current_acres` → `discovery_acres`

---

## Layer 2 — MILP Optimizer

Formulated as a Mixed-Integer Linear Program using PuLP (CBC solver).

**Objective:**
```
minimize  suppression_cost + λ × residual_damage

suppression_cost   = Σ_{r,f} cost_per_hour[r] × resource_hours[r,f]
residual_damage[f] = (danger[f]/100) × (asset[f]/10) × uncovered[f] × $500/acre
```

Both terms are in dollar-equivalent units. λ is a dimensionless risk-preference multiplier (default 50).

**Nonlinear suppression effectiveness:**

Empirical work suggests realized suppression productivity on large wildland fires can be substantially below standard rates. The model captures this idea with a size-effectiveness multiplier:

```
eff[f] = 1 / (1 + K_SIZE × demand[f])
```

Default `K_SIZE = 1.25e-4` (conservative). Sensitivity analysis uses `[1.25e-4, 5e-4, 1e-3]` to show how the allocation changes under weaker or stronger large-fire productivity penalties.

At default K: a 200-ac fire gets eff ≈ 0.976 (near-standard); a 1,800-ac fire gets eff ≈ 0.816 (~18% degraded). Larger fires are harder to suppress per resource-hour — the optimizer naturally accounts for this.

**Coverage constraint:**
```
c[f] = Σ_r aph_terrain[r,f] × eff[f] × resource_hours[r,f]
```

**Constraints:**
- C1 Supply: resource-hours ≤ units × horizon
- C2 Budget: total cost ≤ budget
- C3 Coverage: c[f] = terrain-adjusted effective acres
- C4 Uncovered: u[f] ≥ demand[f] − c[f]
- C5 Overcap: c[f] ≤ demand[f] + max single-resource capacity

**Demand hierarchy (same as risk scoring):**
`incident_size_6h` → `incident_size` → `current_acres` → `discovery_acres`

---

## Layer 3 — Fire Spread Simulation

Models fire propagation as a shortest-path problem on a directed weighted graph. **This is a simplified directional exposure model, not a wildfire forecast.**

- Grid: 40×40 cells, 100m × 100m (4km × 4km per fire)
- Wind factor: downwind ~3× faster than upwind (Rothermel-inspired)
- Fuel multipliers by NIFC PredominantFuelGroup (grass=1.6×, shrub=1.3×, timber=1.0×)
- Suppression feedback: resources reduce spread rate in their assigned sector
- Terrain: slope adjustment via SRTM (falls back to synthetic if unavailable)
- Diurnal weather: sinusoidal peak at 14:00 — wind +25%, RH −20pp, temp +8°C

Rolling reallocation every 3 hours over a 12-hour horizon.

---

## Layer 4 — OSM Infrastructure Asset Value

- Assets from OpenStreetMap: hospitals, schools, fire stations, residential/commercial areas
- Spatial join via elliptical Dijkstra footprint (wind-direction aware — upwind assets excluded)
- Normalization: percentile-anchored (p5→1, p95→10)
- Life-safety weight hierarchy: hospitals (10) → fire stations (9) → schools (8) → clinics (7) → residential (4) → commercial (2)

---

## Scenario Design

The default scenario is built around four fires from the September 2020 Pacific Northwest Labor Day firestorm — a period of genuine national resource scarcity (NMAC Preparedness Level 5):

| Fire | 6h Demand | Behavior | Complexity | Role in scenario |
|------|----------:|----------|------------|-----------------|
| BEACHIE CREEK | 1,800 ac | Active | Type 1 | High risk, large demand |
| LIONSHEAD | 1,500 ac | Minimal | Type 2 | Large resource sink, lower behavior |
| ALMEDA DRIVE | 200 ac | Moderate | Type 1 | Small but high asset exposure |
| RIVERSIDE | 600 ac | Moderate | Type 3 | Mid-size competitor |

**Total 6h demand: 4,100 acres vs ~900 acres budget-constrained coverage capacity** — forcing genuine allocation tradeoffs. The optimizer cannot cover all fires; it must triage.

---

## Key Modeling Decision

The optimizer does not simply send everything to the highest-risk fire. It solves:

> Which resource allocation gives the best reduction in risk-weighted uncovered demand per dollar and per effective resource-hour?

A smaller, accessible fire with moderate risk may receive more resources than a larger, highest-risk fire if:
- The large fire's demand far exceeds what resources can effectively suppress in the planning horizon
- The smaller fire is fully containable — eliminating all residual damage there
- The productivity gap (from the size-effectiveness multiplier) makes the smaller fire more efficient to cover

This mirrors real operational triage logic: incident commanders sometimes accept strategic defensive posture on one fire to enable containment of another.

---

## Known Model Limitations

**Linear cost model.** Within each fire, cost scales linearly with resource-hours. Fixed mobilization costs are not modeled.

**Fireline production vs suppression.** The `acres_per_hour` values in `resources.csv` represent fireline production capacity per NWCG PMS 437, not direct fire area suppressed. An engine producing 0.35 ac/hr of fireline is not equivalent to extinguishing 0.35 acres of active fire. The model uses fireline production as a proxy for demand covered per resource-hour — a standard simplification in fire operations research but one that understates the nonlinearity of real suppression outcomes.

**Static demand.** The 6h demand figure is fixed at scenario design time. The optimizer does not re-estimate containability mid-horizon (though Layer 3 dynamic reallocation partially addresses this).

**K_SIZE calibration.** The effectiveness multiplier shape is derived from Holmes & Calkin (2013) empirical bounds (14–93% of standard rates) but the specific functional form `1/(1+K×size)` is an approximation. Sensitivity analysis over three K values is included.

**AHP weights are judgment-based.** CR < 0.10 confirms internal consistency, not objective correctness.

**Suppression sectors fixed at dispatch.** Resources are assigned to a fire's sector at dispatch; real redeployment within a fire is continuous.

**4km × 4km spread grid.** For large fires, the grid covers only marginal perimeter dynamics, not the full incident footprint.

---

## Data Sources

| Source | What it provides |
|--------|-----------------|
| [NIFC WFIGS](https://data-nifc.opendata.arcgis.com) | Fire locations, acreage, behavior, complexity |
| [NOAA weather.gov API](https://api.weather.gov) | Wind, temperature, humidity |
| [OpenStreetMap via osmnx](https://osmnx.readthedocs.io) | Infrastructure assets |
| [Open Elevation API](https://api.open-elevation.com) | SRTM terrain elevation |
| [USFS Aviation Contracting](https://www.fs.usda.gov/managing-land/fire/contracting) | Helicopter hourly flight rates (2020 contract chart) |

All sources are free, no API keys required.

## Resource Cost Assumptions

Resource costs and suppression capacities are modeled per productive resource-hour. This avoids assuming 24 productive operating hours per day, which is unrealistic especially for aircraft that have flight-time limits, turnaround, reloading, and daylight constraints.

| Resource | Cost/hr | Acres/hr | Productive hrs/day | Basis |
|---|---|---|---|---|
| Type-1 Engine | $400 | 0.35 | 10 | USFS all-in daily estimate |
| Heavy Dozer | $550 | 0.40 | 10 | USFS equipment rate |
| Type-1 Helicopter | $4,500 | 4.2 | 7 | USFS 2020 flight rate chart (large helicopter $4,298–4,900/hr) |
| Air Tanker | $7,000 | 25.0 | 5 | Conservative EU rate (~$9,700/day + $6,500/hr flight); 5h productive drops |
| Hand Crew (20-person) | $900 | 0.20 | 10 | USFS labor + overhead estimate |

These are planning estimates, not audited contract rates. Sensitivity analysis over budget and K_SIZE captures uncertainty in these assumptions.

---

## Project Structure

```
Forest_Fire_Project/
├── wildfire_data/
│   ├── scenario_fires.csv       # Fire scenario (Labor Day 2020 PNW)
│   ├── resources.csv            # Resource types, capacity, cost
│   ├── osm_assets.geojson       # OSM infrastructure assets
│   ├── asset_scores.csv         # Per-fire asset scores
│   └── dynamic_timeline.csv     # Rolling allocation timeline
│
├── fetch_data.py                # Pull fire + weather from NIFC/NOAA
├── fetch_osm_assets.py          # Pull OSM infrastructure assets
├── wildfire_triage.py           # Layers 0–3: AHP + risk + MILP + spread
├── asset_layer.py               # Layer 4: Asset value integration
├── dashboard.py                 # Streamlit interactive dashboard
│
├── .streamlit/config.toml
└── README.md
```

---

## Running the Project

```bash
pip install pandas numpy networkx pulp osmnx geopandas shapely streamlit plotly requests

python fetch_data.py          # pulls NIFC fire data + NOAA weather
python fetch_osm_assets.py    # pulls OSM infrastructure assets
python wildfire_triage.py     # runs full pipeline, prints results + sensitivity

streamlit run dashboard.py    # launches interactive dashboard
```

---

## Dashboard

| Tab | Contents |
|-----|----------|
| **Decision Summary** | Priority ranking, resource dispatch, demand met %, cost per fire |
| **Why This Allocation?** | Risk factor contributions, objective breakdown, asset exposure impact, spread footprint |
| **Compare Alternatives** | MILP allocation compared with risk-score, acreage, and equal-split baselines |
| **Sensitivity** | Budget, λ, planning horizon, and K_SIZE sensitivity |

Sidebar: adjust wind speed, humidity, temperature, budget, planning horizon, fire behavior overrides, infrastructure toggle. Model re-runs on every change.

