# Wildfire Resource Allocation — Decision Support System

> **Given limited firefighting resources and multiple simultaneous fires, which fire gets resources first — and does an optimized allocation outperform real-world deployment?**

Built on real incident data from September 8, 2020 — the worst single-day wildfire event in Oregon history. Five simultaneous Oregon fires were modeled as competing for a shared constrained resource pool during NMAC Preparedness Level 5 (maximum national scarcity). The system compares reconstructed historical deployment with what a mathematical optimizer would recommend under approximately the same budget and resource constraints.

---

## The Problem

When multiple fires ignite simultaneously, fire operations centers must decide in real time how to split limited resources — engines, dozers, helicopters, air tankers, hand crews — across all active incidents. Raw acreage is a misleading guide: Almeda Drive destroyed 3,000+ structures at just 200 acres, while Lionshead grew to 204,000 acres in remote terrain with lower life-safety risk.

This system formalizes that triage decision using real incident data, mathematical optimization, and machine learning.

---

## What Makes This Different

Most wildfire data science projects stop at prediction — train a model, report accuracy, done. This project connects the full decision chain:

1. **Real scenario** — September 8, 2020 Oregon Labor Day Firestorm, reconstructed from ICS-209-PLUS incident reports and NOAA Sep 8 2020 historical weather
2. **Risk scoring** — each fire scored 0–100 on behavior, weather, complexity, and size using the Analytic Hierarchy Process (AHP), informed by wildfire prioritization principles
3. **Infrastructure exposure** — OpenStreetMap assets (hospitals, schools, residential areas) scored within each fire's wind-driven spread path
4. **Resource optimization** — Mixed Integer Linear Program (MILP) allocates engines, helicopters, tankers, and crews to minimize suppression cost + risk-weighted residual damage under budget and supply constraints
5. **Retrospective evaluation** — optimizer output compared against reconstructed ICS-209 deployment records from September 8, 2020
6. **ML containment model** — Gradient Boosting trained on 25,312 propensity-score-matched historical incidents predicts early containment probability under each allocation

---

## September 8, 2020 — The Scenario

Five Oregon fires were modeled as competing for a shared constrained resource pool on the same day:

| Fire | 6h Demand | Behavior | Complexity | Key fact |
|------|----------:|----------|------------|----------|
| BEACHIE CREEK | 1,800 ac | Active | Type 1 | Grew 200 → 194k acres during wind event. Opal Creek Wilderness. |
| LIONSHEAD | 1,500 ac | Minimal | Type 2 | 204k final acres. East Cascades, arid shrub/timber. |
| RIVERSIDE | 600 ac | Moderate | Type 3 | 138k acres in first 30h. Highway 26 threatened. |
| ALMEDA DRIVE | 200 ac | Moderate | Type 1 | 3,000+ structures destroyed. Medford/Talent/Phoenix urban corridor. |
| HOLIDAY FARM | 900 ac | Active | Type 2 | 173k final acres. McKenzie River corridor, Highway 126. |

Total 6h suppression demand: 5,000 acres. Budget-constrained coverage capacity: ~1,700 acres. The optimizer cannot cover everything — it must triage.

Sep 8 2020 weather (peer-reviewed: Abatzoglou et al. 2021 GRL, NWS Portland):
- Wind: 10–14 m/s sustained easterly, gusts to 27 m/s
- Relative humidity: 7–10% across all fire locations
- Temperature: 31–38°C
- This was an unprecedented wind/heat/low-RH event, 50+ year return period

---

## Key Results

### Optimization vs actual deployment (same $950k budget, same resource pool)

| Metric | MILP | Actual Sep 8 |
|--------|-----:|-------------:|
| Total modeled demand covered | 34.9% | 13.7% |
| Almeda Drive demand covered | 100% | 61.3% |
| Holiday Farm demand covered | 100% | 16.7% |
| Estimated residual damage | $3.88M | $7.43M |
| **Modeled risk-adjusted damage reduction** | **$3.55M** | — |

The model concentrates resources on fires with the highest combination of danger score and infrastructure exposure, rather than spreading evenly. This is the core triage insight.

### ML containment model

| Metric | Value |
|--------|-------|
| Model | Gradient Boosting |
| Cross-validated AUC | 0.80 |
| Training sample | 25,312 PSM-matched incidents |
| Target | P(contained within first 3 sitrep days) |
| Base rate | 63.9% |
| Exploratory ML-based damage reduction estimate | $108,246 |

The conservative estimate uses only fires that actually receive model resources (Beachie Creek, Holiday Farm, Almeda Drive). Lionshead and Riverside receive zero model resources — their ML predictions are unreliable zero-resource counterfactuals and are excluded from the headline figure.

### Comparison vs simple allocation rules (same budget)

| Strategy | Avg demand covered | Risk-weighted demand met |
|----------|-------------------:|-------------------------:|
| **MILP optimizer** | **34.8%** | **37.3%** |
| Risk-score proportional | 3.4% | 4.1% |
| Acreage proportional | 3.6% | 3.8% |
| Equal split | 2.2% | 2.1% |

---

## Technical Architecture

```
ICS-209-PLUS incident reports + NOAA Sep 8 2020 historical weather
        ↓
Layer 0: AHP Weight Derivation
    Pairwise comparison matrix → behavior 46.6%, weather 27.7%,
    complexity 16.1%, size 9.6%
    Consistency Ratio CR = 0.0115 (< 0.10 threshold ✓)
        ↓
Layer 1: Risk Scoring (0–100 per fire)
    Nonlinear weather submodel:
      Humidity: exponential decay (risk rises steeply below 30% RH)
      Wind: exponential growth (risk accelerates at high speed)
      Temperature: linear 0°C → 45°C
        ↓
Layer 2: MILP Optimizer (PuLP / CBC solver)
    minimize: Σ cost[r,f]×hours[r,f] + λ × Σ (danger/100)×(asset/10)×uncovered×$500/ac
    Constraints: supply, budget, coverage, uncovered demand, overcoverage cap
    Size-effectiveness: eff[f] = 1/(1 + K_SIZE × demand[f])
    λ = 50 (risk-aversion weight), K_SIZE = 1.25e-4
        ↓
Layer 3: Spread Simulation (directional exposure only, not a forecast)
    Wind-weighted Dijkstra on 40×40 grid (100m cells, 4km × 4km per fire)
    Downwind ~3× faster than upwind. Fuel multipliers + SRTM terrain slope.
    Used to identify which infrastructure falls within each fire's threat envelope.
        ↓
Layer 4: OSM Asset Exposure
    Tags: hospitals (10), fire stations (9), schools (8), clinics (7),
          residential (4), commercial (2)
    Wind-aware elliptical footprint. Percentile-anchored normalization p5→1, p95→10.
        ↓
ML Containment Model
    Gradient Boosting + Propensity Score Matching (PSM)
    PSM matches fires of similar severity but different resource levels
    to reduce observed severity imbalance when examining the relationship
    between day-1 resources and early containment.
    Features: fire size, spread rate, day-1 resources, complexity,
              structures at risk, seasonality, climate trend
        ↓
ICS-209 Comparison
    Actual Sep 8 2020 ICS-209 deployment vs MILP recommendation
    Same $950k budget, same resource pool, different allocation logic
```

---

## Honest Model Limitations

This is a **prototype decision-support model**, not an operational fire command tool.

| Limitation | Detail |
|-----------|--------|
| **Scenario is historical** | Parameters are fixed to Sep 8 2020. The sidebar sliders allow weather adjustment but the fire locations and demands are fixed. |
| **Linear cost model** | Fixed mobilization costs and crew rotation costs are not modeled. Cost scales linearly with resource-hours. |
| **Fireline production proxy** | `acres_per_hour` values represent fireline production capacity (NWCG PMS 437), not direct fire area suppressed. Understates nonlinearity of real suppression. |
| **Static 6h demand** | Suppression demand is fixed at scenario design time. The optimizer does not re-estimate containability mid-horizon. |
| **K_SIZE approximation** | Size-effectiveness multiplier shape derived from Holmes & Calkin (2013) empirical bounds, not directly calibrated. |
| **AHP weights are judgment-based** | CR < 0.10 confirms internal consistency, not objective correctness. |
| **ML seasonality confound** | `discovery_doy` is the strongest predictor (35.5% importance). Resource features account for ~25% of total variance — not 48% as naive decomposition suggests. |
| **Zero-resource counterfactuals** | ML predictions for Lionshead and Riverside under zero MILP resources are unreliable. Model was trained on real incidents where Type 1/2 fires always receive some resources. |
| **Real commander constraints not modeled** | Crew fatigue, mandatory rest, road closures from fire activity, political/jurisdictional obligations, contract availability, and real-time uncertainty are all absent from the model. |
| **4km × 4km spread grid** | For large fires, the grid covers only marginal perimeter dynamics, not the full incident footprint. |
| **Asset scores are OSM-dependent** | OSM coverage varies by area. Rural fire locations may have sparse asset data. |
| **Information advantage (critical)** | The model uses post-event information that commanders did not have on September 8, 2020. Asset scores were computed from OSM data after the fact. Weather values come from peer-reviewed post-event analysis, not the imperfect forecasts available that morning. The 6h suppression demand figures are designed parameters, not field-available numbers. Fire behavior severity (Almeda Drive destroying 3,000+ structures) was not fully known at dispatch time. This means the $3.55M damage reduction reflects both better optimization logic AND better information — the two cannot be cleanly separated in this comparison. A truly fair comparison would require restricting the model to only information commanders actually had at 6am on September 8, which this project does not do. |

The comparison against actual deployment does **not** prove commanders made errors. It shows a theoretical optimum under simplified but controlled assumptions — and with an information advantage the model should not claim credit for.

---

## Data Sources

| Source | What it provides | Access |
|--------|-----------------|--------|
| [ICS-209-PLUS (Figshare)](https://figshare.com/articles/dataset/19858927) | 182,826 wildfire sitreps, 1999–2020 — core training + scenario data | Free |
| [NIFC WFIGS](https://data-nifc.opendata.arcgis.com) | Fire locations, acreage, behavior, complexity | Free |
| [NOAA NCEI CDO API](https://www.ncdc.noaa.gov/cdo-web/token) | Sep 8 2020 historical weather per fire location (free token) | Free |
| [NOAA weather.gov API](https://api.weather.gov) | Current conditions for interactive sidebar | Free |
| [OpenStreetMap via osmnx](https://osmnx.readthedocs.io) | Infrastructure assets near each fire | Free |
| [Open Elevation API](https://api.open-elevation.com) | SRTM terrain elevation for slope adjustment | Free |
| Abatzoglou et al. 2021, GRL (doi:10.1029/2021GL092520) | Sep 8 2020 weather validation — RH, wind, unprecedented conditions | Published |
| NWS Portland WAF-D-21-0028.1 | Sep 8 2020 event analysis — wind speeds, gusts, Red Flag conditions | Published |
| Holmes & Calkin 2013 | Size-effectiveness multiplier empirical bounds (14–93% of standard rates) | Published |

---

## Resource Cost Assumptions

Costs are per productive resource-hour — not 24h/day. Aircraft have flight-time limits, crews have mandatory rest.

| Resource | Cost/hr | Fireline ac/hr | Productive hrs/day | Basis |
|----------|---------|---------------|-------------------|-------|
| Type-1 Engine | $400 | 0.35 | 10 | USFS all-in daily estimate |
| Heavy Dozer | $550 | 0.40 | 10 | USFS VIPR equipment rate |
| Type-1 Helicopter | $4,500 | 4.2 | 7 | USFS 2020 flight rate chart (large helicopter) |
| Air Tanker | $7,000 | 25.0 | 5 | USFS LAT contract; 5h productive drops/day |
| Hand Crew (20-person) | $900 | 0.20 | 10 | USFS labor + overhead |

Sep 8 2020 resource pool (from ICS-209 sitreps across all 5 fires):
- 227 engines, 154 hand crews, 35 helicopters, 10 air tankers

---

## Project Structure

```
Forest_Fire_Project/
├── wildfire_data/
│   ├── raw/
│   │   └── ics209-plus-wf_sitreps_1999to2020.csv   # Download separately (see below)
│   ├── scenario_fires.csv           # Sep 8 2020 Oregon fires + real weather
│   ├── resources.csv                # Resource pool calibrated to Sep 8 2020
│   ├── actual_deployment.csv        # ICS-209 ground truth: what was actually deployed
│   ├── comparison_report.csv        # MILP vs actual: per-fire coverage + damage
│   ├── osm_assets.geojson           # OSM infrastructure assets (all 5 fires)
│   ├── asset_scores.csv             # Per-fire risk-weighted exposure scores (1–10)
│   ├── containment_predictions.csv  # ML model: P(contained) actual vs MILP
│   ├── historical_weather_sep8_2020.csv  # NOAA verified Sep 8 conditions
│   └── models/
│       ├── containment_model.pkl    # Trained Gradient Boosting + Platt calibration
│       ├── model_metadata.json      # CV AUC, Brier, feature list, matched sample size
│       └── feature_importance.csv   # Feature importance from best model
│
├── fetch_ics209.py              # Build scenario from ICS-209-PLUS (+ --fallback mode)
├── fetch_historical_weather.py  # NOAA NCEI historical weather for Sep 8 2020
├── fetch_osm_assets.py          # OSM infrastructure assets per fire location
├── asset_layer.py               # Compute + normalize asset exposure scores
├── containment_model.py         # Train ML containment model (PSM + Gradient Boosting)
├── wildfire_triage.py           # AHP + risk scoring + MILP optimizer + spread model
├── compare_deployment.py        # MILP vs actual ICS-209 deployment comparison
├── dashboard.py                 # Streamlit interactive dashboard (4 tabs)
├── archive/
│   └── washington-prototype/   # Earlier exploratory Washington version
├── requirements.txt
└── README.md
```

---

## Running the Project

```bash
pip install pandas numpy networkx pulp osmnx geopandas shapely streamlit plotly requests scikit-learn srtm.py

# Step 1: Download ICS-209-PLUS
#   Go to: https://figshare.com/articles/dataset/19858927
#   Download: ics209-plus-wildfire.zip → extract CSV
#   Place at: wildfire_data/raw/ics209-plus-wf_sitreps_1999to2020.csv
#
#   No CSV? Use fallback (verified Sep 8 2020 parameters from published sources):
python fetch_ics209.py --fallback

# Step 2: Historical weather (NOAA NCEI — free token at ncdc.noaa.gov/cdo-web/token)
python fetch_historical_weather.py --token YOUR_TOKEN
#   No token? Use peer-reviewed fallback values:
python fetch_historical_weather.py --fallback-only

# Step 3: OSM infrastructure assets + exposure scores
python fetch_osm_assets.py
python asset_layer.py

# Step 4: Train ML containment model (~3-5 min)
#   Requires ICS-209-PLUS CSV for full training
python containment_model.py

# Step 5: Run optimizer and generate comparison
python wildfire_triage.py
python compare_deployment.py

# Step 6: Launch dashboard
streamlit run dashboard.py
```

---

## Dashboard

4 tabs, structured for both general and technical readers:

| Tab | Purpose | Key content |
|-----|---------|-------------|
| **Actual vs Model** | Main story | Business impact KPIs, fire-by-fire actual vs MILP comparison, coverage and damage charts, ML summary |
| **Why This Recommendation?** | Explain the logic | Fire priority cards, dispatch table, risk score breakdown, asset exposure, spread footprint (expander) |
| **Does Optimization Help?** | Validate results | MILP vs equal split / risk-proportional / acreage-proportional / actual deployment |
| **Robustness & Technical Details** | For technical readers | ML containment chart, MILP formulation, AHP weights, budget sensitivity, λ sensitivity |

Sidebar controls: wind speed, humidity, temperature, budget, planning horizon. The optimizer and risk scores re-run on every change.

---

## Honest Verdict

**What this project demonstrates:**

> A wildfire resource allocation system that turns real incident data into structured triage decisions under scarcity — validated against actual historical deployment from one of the most demanding multi-fire events in US history.

The optimization layer is the core contribution. It is not just prediction — it is a complete decision pipeline from raw data to actionable resource allocation recommendation, with explicit budget and supply constraints, risk-weighted objectives, and infrastructure exposure scoring.

The ICS-209 comparison gives the project a real-world evaluation story that most portfolio projects lack: the same budget, the same resource pool, a different allocation logic, and a quantified outcome difference.

The ML model adds a data-driven containment probability layer. Propensity-score matching reduces observed severity imbalance, but the results remain associational and may still be affected by unmeasured confounding, seasonal dominance, and the zero-resource counterfactual problem.

**What this is:**
A prototype decision-support tool showing how operations research and machine learning can structure emergency resource allocation under genuine scarcity.

**What this is not:**
An operational wildfire command system, a fire spread forecast, or proof that September 8 2020 commanders made errors.
