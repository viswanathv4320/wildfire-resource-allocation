"""
fetch_data.py
Wildfire Triage Project — Data Collection Script
Run this locally. Requires: pip install requests pandas
"""

import requests
import pandas as pd
import numpy as np
import os
import time

OUTPUT_DIR = "wildfire_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEATTLE_LAT = 47.6062
SEATTLE_LON = -122.3321
RADIUS_DEG  = 2.0   # ~150 miles

HEADERS = {"User-Agent": "wildfire-triage-project/1.0 (student-research)"}


# ════════════════════════════════════════════════════════════════════════════
# 1.  FIRE LOCATIONS  (NIFC WFIGS — corrected field names)
# ════════════════════════════════════════════════════════════════════════════

def fetch_fire_locations():
    print("\n── Fetching Washington State fire locations from NIFC …")

    url = (
        "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
        "WFIGS_Incident_Locations/FeatureServer/0/query"
    )

    params = {
        "where"             : "POOState='US-WA' AND DiscoveryAcres IS NOT NULL",
        "outFields"         : ",".join([
            "IncidentName",
            "FireDiscoveryDateTime",
            "DiscoveryAcres",
            "FinalAcres",
            "IncidentSize",
            "InitialLatitude",
            "InitialLongitude",
            "POOCounty",
            "POOState",
            "FireCause",
            "FireBehaviorGeneral",
            "FireMgmtComplexity",
            "PredominantFuelModel",
            "PredominantFuelGroup",
            "EstimatedCostToDate",
            "EstimatedFinalCost",
            "PercentContained",
        ]),
        "f"                 : "json",
        "resultRecordCount" : 2000,
        "orderByFields"     : "DiscoveryAcres DESC",
    }

    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()

    features = r.json().get("features", [])
    print(f"   Raw records returned: {len(features)}")

    rows = [f["attributes"] for f in features]
    df = pd.DataFrame(rows)

    # Drop rows with no location
    df = df.dropna(subset=["InitialLatitude", "InitialLongitude"])

    # Filter within ~150 miles of Seattle
    df = df[
        df["InitialLatitude"] .between(SEATTLE_LAT - RADIUS_DEG, SEATTLE_LAT + RADIUS_DEG) &
        df["InitialLongitude"].between(SEATTLE_LON - RADIUS_DEG, SEATTLE_LON + RADIUS_DEG)
    ].copy()

    df = df.sort_values("DiscoveryAcres", ascending=False).reset_index(drop=True)

    out = os.path.join(OUTPUT_DIR, "wa_fires_near_seattle.csv")
    df.to_csv(out, index=False)
    print(f"   Saved {len(df)} fires near Seattle → {out}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2.  CONTRASTING-PROFILE SCENARIO SELECTION
#
# Selects 4 fires with maximally different risk drivers — one dominant on
# each of: size, behavior/complexity, weather risk (arid proxy), and data
# quality. This tests whether the optimizer makes better decisions than
# simple size-only triage. Fires are constrained to be geographically
# distinct (different counties where possible) so weather signals diverge.
#
# Selection is a two-stage process:
#   Stage 1 — score every fire on four normalized dimensions
#   Stage 2 — greedy pick: choose one fire that ranks highest on each
#              dimension that hasn't been dominated by a prior pick
#
# Why IncidentSize not DiscoveryAcres for the size dimension:
#   DiscoveryAcres is the reported size at discovery — often a few acres.
#   IncidentSize is the most recent operational size. Using DiscoveryAcres
#   for size scoring would select fires that were large at discovery but
#   may have been fully contained, which is less operationally relevant.
#   The model's IP optimizer still uses DiscoveryAcres as the demand
#   signal (documented as "footprint at dispatch time"), so this is a
#   scenario construction choice, not a model inconsistency.
# ════════════════════════════════════════════════════════════════════════════

# Ordinal encodings for the behavior and complexity fields
BEHAVIOR_SCORE  = {"Minimal": 0.2, "Moderate": 0.5, "Active": 0.8, "Extreme": 1.0}
COMPLEXITY_SCORE = {
    "Type 5 Incident": 0.1, "Type 4 Incident": 0.25,
    "Type 3 Incident": 0.5, "Type 2 Incident": 0.75, "Type 1 Incident": 1.0,
}

# Minimum data requirements to be a valid candidate.
# Weather can be fetched for any location; behavior/complexity are strongly
# preferred but not required (the model handles NaN gracefully).
MIN_DISCOVERY_ACRES = 20.0   # filter out trivial ignitions


def _score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add four normalized dimension scores [0,1] to every candidate fire.

    Dimensions:
      dim_size       — operational size (IncidentSize, log-scaled)
      dim_behavior   — fire behavior × management complexity (ordinal)
      dim_aridity    — latitude-based proxy for arid/low-humidity terrain
                       (eastern WA fires at lower lat tend to burn in drier
                        grass/shrub fuel with higher weather risk)
      dim_completeness — fraction of high-value model fields present
    """
    d = df.copy()

    # Fields the model uses heavily — defined here so all dimensions can reference them
    valued_fields = [
        "FireBehaviorGeneral", "FireMgmtComplexity",
        "PredominantFuelGroup", "EstimatedFinalCost",
    ]

    # ── Dimension 1: Operational size (log-normalized) ───────────────────
    size = d["IncidentSize"].fillna(d["DiscoveryAcres"])
    log_size = np.log1p(size)
    d["dim_size"] = (log_size - log_size.min()) / (log_size.max() - log_size.min() + 1e-9)

    # ── Dimension 2: Behavior × complexity ───────────────────────────────
    bscore = d["FireBehaviorGeneral"].map(BEHAVIOR_SCORE).fillna(0.1)
    cscore = d["FireMgmtComplexity"].map(COMPLEXITY_SCORE).fillna(0.1)
    # Use max rather than product — a Type 1 fire with missing behavior
    # should still rank high; product would zero it out unfairly.
    raw_bc = np.maximum(bscore, cscore)
    d["dim_behavior"] = (raw_bc - raw_bc.min()) / (raw_bc.max() - raw_bc.min() + 1e-9)

    # ── Dimension 3: Arid / weather-risk proxy ───────────────────────────
    # Eastern WA (lon > -121) tends to have drier, grassier fuels and
    # lower humidity — the conditions your model weights most heavily.
    # We encode this as a continuous signal so western WA fires aren't
    # fully excluded; they just score lower on this dimension.
    lon = d["InitialLongitude"]
    lat = d["InitialLatitude"]
    # Positive = more eastern/inland (more arid). Range roughly -124 to -117.
    aridity_raw = (lon - lon.min()) / (lon.max() - lon.min() + 1e-9)  # 0=west, 1=east
    # Modulate by latitude: southern eastern WA is historically drier
    lat_factor  = 1 - (lat - lat.min()) / (lat.max() - lat.min() + 1e-9)  # 1=south
    # Add a small completeness nudge (15%) so we don't pick a fire with
    # zero data fields just because it's in the driest county.
    # The aridity signal still dominates (85%).
    completeness_nudge = d[valued_fields].notna().sum(axis=1) / len(valued_fields)
    d["dim_aridity"] = 0.6 * aridity_raw + 0.25 * lat_factor + 0.15 * completeness_nudge

    # ── Dimension 4: Data completeness ───────────────────────────────────
    # Fields the model uses heavily; missing = model falls back to defaults
    d["dim_completeness"] = d[valued_fields].notna().sum(axis=1) / len(valued_fields)

    return d


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def select_scenario_fires(df: pd.DataFrame,
                           n_fires: int = 4,
                           min_separation_km: float = 80.0) -> pd.DataFrame:
    """
    Select `n_fires` fires with contrasting operational profiles.

    Algorithm:
      1. Filter to candidates with sufficient data and acreage.
      2. Score all candidates on four dimensions.
      3. Greedy pick: for each slot, choose the fire that ranks highest
         on the dimension it's meant to represent, subject to geographic
         separation from already-selected fires.
      4. If a dimension has no unique winner (e.g. same fire tops two
         dimensions), fall back to highest overall score for that slot.

    Parameters
    ----------
    df                : output of fetch_fire_locations()
    n_fires           : number of scenario fires to select (default 4)
    min_separation_km : minimum distance between selected fires (km).
                        Prevents two fires from the same cluster.
    """
    # Stage 1 — filter
    candidates = df[df["DiscoveryAcres"] >= MIN_DISCOVERY_ACRES].copy()
    candidates = _score_candidates(candidates)

    print(f"\n── Contrasting-Profile Scenario Selection ─────────────────────")
    print(f"   Candidates after filtering (≥{MIN_DISCOVERY_ACRES} ac): {len(candidates)}")

    # Stage 2 — greedy slot assignment
    # Slot order: size first (anchor), then behavior, aridity, completeness.
    # This ordering matters: size provides the "acreage isn't everything"
    # baseline; behavior/aridity/completeness create the contrast.
    slots = [
        ("size",         "dim_size",         "Large operational footprint"),
        ("behavior",     "dim_behavior",      "High behavior / complexity"),
        ("aridity",      "dim_aridity",       "High weather risk (arid terrain)"),
        ("completeness", "dim_completeness",  "Best data quality"),
    ]

    selected_rows = []
    selected_idx  = set()

    for slot_name, dim_col, slot_desc in slots[:n_fires]:
        # Exclude already-selected fires
        pool = candidates[~candidates.index.isin(selected_idx)].copy()

        if pool.empty:
            print(f"   ⚠ No candidates left for slot '{slot_name}'")
            break

        # Enforce geographic separation from already-selected fires
        if selected_rows:
            sel_lats = [r["InitialLatitude"]  for r in selected_rows]
            sel_lons = [r["InitialLongitude"] for r in selected_rows]
            too_close = pool.apply(
                lambda row: any(
                    _haversine_km(row["InitialLatitude"], row["InitialLongitude"], lt, ln)
                    < min_separation_km
                    for lt, ln in zip(sel_lats, sel_lons)
                ),
                axis=1
            )
            far_pool = pool[~too_close]
            # If separation constraint kills all candidates, relax it
            if far_pool.empty:
                print(f"   ⚠ Relaxing separation constraint for slot '{slot_name}'")
                far_pool = pool
        else:
            far_pool = pool

        # Pick the highest scorer on this slot's dimension
        best_idx = far_pool[dim_col].idxmax()
        best_row = far_pool.loc[best_idx]
        selected_rows.append(best_row)
        selected_idx.add(best_idx)

        print(f"   Slot '{slot_name:<12}' → {best_row['IncidentName']:<22}"
              f"  size={best_row['IncidentSize']:>8,.0f} ac"
              f"  behavior={best_row.get('FireBehaviorGeneral','—')}"
              f"  complexity={best_row.get('FireMgmtComplexity','—')}"
              f"  {slot_desc}")

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)

    # Print dimension score summary for transparency
    dim_cols = ["IncidentName", "dim_size", "dim_behavior", "dim_aridity", "dim_completeness"]
    print(f"\n   Dimension scores for selected fires (0=low, 1=high on each axis):")
    print(f"   {'Fire':<22} {'Size':>6} {'Behav':>6} {'Arid':>6} {'Compl':>6}")
    for _, row in selected.iterrows():
        print(f"   {row['IncidentName']:<22} "
              f"{row['dim_size']:>6.2f} "
              f"{row['dim_behavior']:>6.2f} "
              f"{row['dim_aridity']:>6.2f} "
              f"{row['dim_completeness']:>6.2f}")

    return selected


# ════════════════════════════════════════════════════════════════════════════
# 2.  WEATHER DATA  (NOAA weather.gov — no API key needed)
# ════════════════════════════════════════════════════════════════════════════

def get_nearest_station(lat, lon):
    url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    obs_url = r.json()["properties"]["observationStations"]
    r2 = requests.get(obs_url, headers=HEADERS, timeout=15)
    r2.raise_for_status()
    stations = r2.json()["features"]
    if not stations:
        return None
    return stations[0]["properties"]["stationIdentifier"]


def get_latest_observation(station_id):
    url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    props = r.json()["properties"]
    return {
        "station"        : station_id,
        "timestamp"      : props.get("timestamp"),
        "temperature_c"  : (props.get("temperature")      or {}).get("value"),
        "wind_speed_mps" : (props.get("windSpeed")        or {}).get("value"),
        "wind_dir_deg"   : (props.get("windDirection")    or {}).get("value"),
        "humidity_pct"   : (props.get("relativeHumidity") or {}).get("value"),
        "dewpoint_c"     : (props.get("dewpoint")         or {}).get("value"),
    }


def fetch_weather_for_fires(fire_df):
    """
    Fetch NOAA weather observations for each fire in fire_df.
    fire_df should already be the scenario selection (output of
    select_scenario_fires), not the full candidate list.
    """
    print(f"\n── Fetching weather for {len(fire_df)} scenario fires …")
    records = []

    for i, row in fire_df.reset_index(drop=True).iterrows():
        lat  = row["InitialLatitude"]
        lon  = row["InitialLongitude"]
        name = row.get("IncidentName", f"Fire_{i+1}")
        print(f"   [{i+1}] {name}  ({lat:.3f}, {lon:.3f})")
        try:
            station = get_nearest_station(lat, lon)
            if not station:
                print(f"       ⚠ No station found")
                continue
            obs = get_latest_observation(station)
            obs.update({
                "fire_name"       : name,
                "fire_lat"        : lat,
                "fire_lon"        : lon,
                "discovery_acres" : row.get("DiscoveryAcres"),
                "incident_size"   : row.get("IncidentSize"),   # operational size — used as IP demand
                "final_acres"     : row.get("FinalAcres"),
                "county"          : row.get("POOCounty"),
                "fire_cause"      : row.get("FireCause"),
                "fuel_group"      : row.get("PredominantFuelGroup"),
                "fuel_model"      : row.get("PredominantFuelModel"),
                "fire_behavior"   : row.get("FireBehaviorGeneral"),
                "mgmt_complexity" : row.get("FireMgmtComplexity"),
                "estimated_cost"  : row.get("EstimatedFinalCost"),
            })
            records.append(obs)
            time.sleep(0.5)
        except Exception as e:
            print(f"       ⚠ Weather fetch failed: {e}")

    df = pd.DataFrame(records)
    out = os.path.join(OUTPUT_DIR, "fire_weather.csv")
    df.to_csv(out, index=False)
    print(f"\n   Saved weather data → {out}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 3.  SCENARIO SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def build_scenario(weather_df):
    print("\n── Scenario fire summary ──────────────────────────────────────")
    cols = [
        "fire_name", "county", "fire_lat", "fire_lon",
        "discovery_acres", "incident_size", "fuel_group",
        "temperature_c", "wind_speed_mps", "wind_dir_deg", "humidity_pct",
        "fire_behavior", "mgmt_complexity", "estimated_cost",
    ]
    cols = [c for c in cols if c in weather_df.columns]
    summary = weather_df[cols].copy()
    summary.index = [f"Fire {i+1}" for i in range(len(summary))]
    print(summary.T.to_string())

    out = os.path.join(OUTPUT_DIR, "scenario_fires.csv")
    summary.to_csv(out)
    print(f"\n   Saved scenario → {out}")
    return summary


# ════════════════════════════════════════════════════════════════════════════
# 4.  RESOURCE DATA
# ════════════════════════════════════════════════════════════════════════════

def save_resource_data():
    print("\n── Saving Washington DNR resource estimates …")
    # Cost sources (2023–2024):
    # Type-1 Engine   : CA CFAA 2023 — $173/hr × 16h equipment + 4-person crew labor → ~$4,500/day
    # Heavy Dozer     : USFS VIPR data — Type 1 avg $3,117/day equipment + operator + fuel → ~$5,800/day
    # Type-1 Helicopter: MT DNRC 2024 EU contract — dry rate $36,900/day; blended ops day → ~$30,000/day
    # Air Tanker      : CO state contract $32,000/day standby (2023); USFS LAT range $35–45k → $40,000/day
    # Hand Crew       : 20 × FFT1/FFT2 @$21/hr × 14h = $5,880 labor + overhead/per-diem → ~$11,000/day
    resources = pd.DataFrame([
        {"resource": "Type-1 Engine",        "units_available": 45,
         "acres_per_day": 3.5,   "cost_per_day": 4500},
        {"resource": "Heavy Dozer",           "units_available": 12,
         "acres_per_day": 4.0,   "cost_per_day": 5800},
        {"resource": "Type-1 Helicopter",     "units_available": 8,
         "acres_per_day": 100.0, "cost_per_day": 30000},
        {"resource": "Air Tanker",            "units_available": 4,
         "acres_per_day": 600.0, "cost_per_day": 40000},
        {"resource": "Hand Crew (20-person)", "units_available": 20,
         "acres_per_day": 2.0,   "cost_per_day": 11000},
    ])
    out = os.path.join(OUTPUT_DIR, "resources.csv")
    resources.to_csv(out, index=False)
    print(f"   Saved resource table → {out}")
    return resources


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Wildfire Triage Project — Data Collection")
    print("=" * 60)

    fire_df = fetch_fire_locations()

    if fire_df.empty:
        print("\n⚠ No fire records found near Seattle. Check connection.")
    else:
        # Select 4 fires with contrasting operational profiles
        scenario_fires = select_scenario_fires(fire_df, n_fires=4)

        weather_df = fetch_weather_for_fires(scenario_fires)

        if weather_df.empty:
            print("\n⚠ Weather fetch returned nothing. Check NOAA API.")
        else:
            scenario  = build_scenario(weather_df)
            resources = save_resource_data()

            print("\n" + "=" * 60)
            print("  Done. Upload these two files here to continue:")
            print(f"    {OUTPUT_DIR}/scenario_fires.csv")
            print(f"    {OUTPUT_DIR}/resources.csv")
            print("=" * 60)