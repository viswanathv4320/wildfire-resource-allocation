"""
build_scenario.py
=================
Two-step tool:

  STEP 1 — Enrich
  ---------------
  Reads wa_fires_near_seattle.csv and appends three new columns:
    terrain_slope_pct   — mean slope (%) within 1 km of fire coordinate
    road_distance_km    — distance to nearest OSM road (km)
    terrain_enriched    — flag: 1 = values computed, 0 = fetch failed

  Writes the enriched data back to wa_fires_near_seattle.csv (in-place).
  Safe to re-run: fires already marked terrain_enriched=1 are skipped.

  STEP 2 — Select
  ---------------
  Prints a ranked table of all enriched fires and lets you choose 4
  (by number) to export as wildfire_data/scenario_fires.csv, which is
  the input file for wildfire_triage.py and dashboard.py.

Usage (run once on your local machine where internet access is open):
    pip install srtm.py requests pandas numpy
    python build_scenario.py

Requirements:
    srtm.py     — SRTM tile download + elevation lookup
    requests    — OSM Overpass API for road distance
    pandas      — CSV I/O
    numpy       — gradient / slope computation

Terrain sources:
    Elevation : SRTM 1-arc-second (NASA/USGS via srtm.kurviger.de)
                ~30 m horizontal resolution
    Roads     : OpenStreetMap Overpass API (overpass-api.de)
                Nearest highway feature within 25 km radius
"""

import math
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
import srtm

# ── Paths ────────────────────────────────────────────────────────────────────

SCENARIO_DIR    = "wildfire_data"
WA_CSV          = os.path.join(SCENARIO_DIR, "wa_fires_near_seattle.csv")
SCENARIO_CSV    = os.path.join(SCENARIO_DIR, "scenario_fires.csv")

# ── Terrain fetch config ──────────────────────────────────────────────────────

# Grid of offsets (degrees) around the fire coordinate used to compute slope.
# 0.009° ≈ 1 km. We sample a 5×5 grid → 25 elevation points → finite-difference slope.
SLOPE_GRID_DEG  = 0.009          # half-width of the sampling box
SLOPE_GRID_N    = 5              # points per side (5×5 = 25 samples)

# Overpass API endpoint for road distance queries
OVERPASS_URL    = "https://overpass-api.de/api/interpreter"
ROAD_SEARCH_KM  = 25.0           # max search radius for nearest road
OVERPASS_TIMEOUT= 30             # seconds

# Pause between API calls to avoid rate-limiting
SLEEP_BETWEEN   = 1.5            # seconds


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 HELPERS — Elevation + Slope
# ════════════════════════════════════════════════════════════════════════════

def fetch_slope_pct(lat: float, lon: float, geo_data) -> float:
    """
    Sample a SLOPE_GRID_N × SLOPE_GRID_N elevation grid around (lat, lon)
    and return the mean slope in percent.

    Slope = rise / run × 100.
    Uses central finite differences on the interior of the grid.
    Cell spacing in metres is computed from the degree offset and latitude.
    """
    half  = SLOPE_GRID_DEG
    n     = SLOPE_GRID_N
    lats  = np.linspace(lat - half, lat + half, n)
    lons  = np.linspace(lon - half, lon + half, n)

    # metres per degree at this latitude
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))

    cell_m_lat = (lats[-1] - lats[0]) / (n - 1) * m_per_deg_lat
    cell_m_lon = (lons[-1] - lons[0]) / (n - 1) * m_per_deg_lon

    # Build elevation grid — None where SRTM has no data
    grid = np.full((n, n), np.nan)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            elev = geo_data.get_elevation(la, lo)
            if elev is not None:
                grid[i, j] = float(elev)

    if np.isnan(grid).all():
        return float("nan")

    # Fill NaN with nearest valid neighbour (simple forward-fill)
    for i in range(n):
        for j in range(n):
            if np.isnan(grid[i, j]):
                # Use nearest non-NaN in flattened order
                flat = grid.flatten()
                valid = flat[~np.isnan(flat)]
                if len(valid):
                    grid[i, j] = valid[0]

    # Central finite differences on interior
    # dz/dy (north-south gradient), dz/dx (east-west gradient)
    dz_dy = np.gradient(grid, cell_m_lat, axis=0)   # rise per metre N-S
    dz_dx = np.gradient(grid, cell_m_lon, axis=1)   # rise per metre E-W

    # Slope magnitude
    slope_frac = np.sqrt(dz_dx**2 + dz_dy**2)
    mean_slope_pct = float(np.nanmean(slope_frac) * 100.0)
    return round(mean_slope_pct, 1)


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 HELPERS — Road Distance via OSM Overpass
# ════════════════════════════════════════════════════════════════════════════

def fetch_road_distance_km(lat: float, lon: float,
                            search_km: float = ROAD_SEARCH_KM) -> float:
    """
    Query OSM Overpass for the nearest highway feature within search_km.
    Returns distance in km to the nearest road node, or NaN on failure.

    Uses the 'around' filter which returns all highway elements within
    the radius. We then compute geodesic distance to each returned node
    and return the minimum.
    """
    search_m = search_km * 1000
    query = f"""
[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  node["highway"](around:{search_m:.0f},{lat},{lon});
  way["highway"](around:{search_m:.0f},{lat},{lon});
);
out center;
"""
    headers = {
        "User-Agent": "WildfireTriageResearch/1.0 (academic wildfire resource allocation project)",
        "Accept"    : "application/json",
    }
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query},
                             headers=headers,
                             timeout=OVERPASS_TIMEOUT + 5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"      Overpass error: {e}")
        return float("nan")

    elements = data.get("elements", [])
    if not elements:
        return float("nan")

    min_dist_km = float("inf")
    for el in elements:
        # nodes have lat/lon directly; ways have center
        if el["type"] == "node":
            elat, elon = el.get("lat"), el.get("lon")
        elif el["type"] == "way" and "center" in el:
            elat, elon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        if elat is None or elon is None:
            continue
        dist_km = haversine_km(lat, lon, elat, elon)
        if dist_km < min_dist_km:
            min_dist_km = dist_km

    return round(min_dist_km, 2) if min_dist_km < float("inf") else float("nan")


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Enrich CSV
# ════════════════════════════════════════════════════════════════════════════

def enrich(csv_path: str) -> pd.DataFrame:
    """
    Load wa_fires CSV, compute terrain_slope_pct and road_distance_km
    for every row that hasn't been enriched yet, and save in-place.
    """
    df = pd.read_csv(csv_path)

    # Add columns if missing
    for col in ["terrain_slope_pct", "road_distance_km", "terrain_enriched"]:
        if col not in df.columns:
            df[col] = None

    # Rows needing enrichment
    todo = df[df["terrain_enriched"].isna() | (df["terrain_enriched"] != 1)].copy()
    total = len(todo)

    if total == 0:
        print("  All rows already enriched. Nothing to do.")
        return df

    print(f"\n  Enriching {total} fires with SRTM slope + OSM road distance …")
    print(f"  Estimated time: ~{total * (SLEEP_BETWEEN + 2):.0f}s  "
          f"({SLEEP_BETWEEN + 2:.1f}s per fire)\n")

    geo_data = srtm.get_data()

    for idx, (row_i, row) in enumerate(todo.iterrows()):
        lat = row["InitialLatitude"]
        lon = row["InitialLongitude"]
        name = row["IncidentName"]

        print(f"  [{idx+1}/{total}]  {name}  ({lat:.4f}, {lon:.4f})")

        # -- Slope
        try:
            slope = fetch_slope_pct(lat, lon, geo_data)
            print(f"          slope = {slope}%")
        except Exception as e:
            slope = float("nan")
            print(f"          slope FAILED: {e}")

        # -- Road distance
        try:
            road_km = fetch_road_distance_km(lat, lon)
            print(f"          road  = {road_km} km")
        except Exception as e:
            road_km = float("nan")
            print(f"          road FAILED: {e}")

        df.at[row_i, "terrain_slope_pct"] = slope
        df.at[row_i, "road_distance_km"]  = road_km
        df.at[row_i, "terrain_enriched"]  = 1 if not (math.isnan(slope) and math.isnan(road_km)) else 0

        # Save after every fire — safe to interrupt and resume
        df.to_csv(csv_path, index=False)

        if idx < total - 1:
            time.sleep(SLEEP_BETWEEN)

    print(f"\n  Done. Enriched {total} fires. Saved → {csv_path}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Interactive fire selector → scenario_fires.csv
# ════════════════════════════════════════════════════════════════════════════

# Column mapping: wa_fires CSV → scenario_fires CSV
COLUMN_MAP = {
    "IncidentName"        : "fire_name",
    "POOCounty"           : "county",
    "InitialLatitude"     : "fire_lat",
    "InitialLongitude"    : "fire_lon",
    "DiscoveryAcres"      : "discovery_acres",
    "IncidentSize"        : "incident_size",
    "PredominantFuelGroup": "fuel_group",
    "FireBehaviorGeneral" : "fire_behavior",
    "FireMgmtComplexity"  : "mgmt_complexity",
    "EstimatedCostToDate" : "estimated_cost",
    "terrain_slope_pct"   : "terrain_slope_pct",
    "road_distance_km"    : "road_distance_km",
}

# Columns added manually (not in wa_fires source)
DEFAULTS = {
    "temperature_c" : 25.0,
    "wind_speed_mps": 5.0,
    "wind_dir_deg"  : 270.0,
    "humidity_pct"  : 20.0,
}


def score_for_selection(row: pd.Series) -> float:
    """
    Quick triage score for display sorting.
    Higher = more operationally interesting for scenario design.
    Weights: large incident_size, active/extreme behavior, complex management.
    """
    size_score = math.log1p(float(row.get("IncidentSize") or row.get("DiscoveryAcres") or 1))

    beh_map = {"Minimal": 0.1, "Moderate": 0.4, "Active": 0.7, "Extreme": 1.0}
    beh_score = beh_map.get(str(row.get("FireBehaviorGeneral") or ""), 0.2)

    cpx_map = {
        "Type 1 Incident": 1.0, "Type 2 Incident": 0.75,
        "Type 3 Incident": 0.5, "Type 4 Incident": 0.25, "Type 5 Incident": 0.1,
    }
    cpx_score = cpx_map.get(str(row.get("FireMgmtComplexity") or ""), 0.2)

    # Terrain diversity bonus: higher slope or longer road distance = more interesting
    slope = float(row.get("terrain_slope_pct") or 15.0)
    road  = float(row.get("road_distance_km")  or 5.0)
    terrain_score = min(slope / 50.0, 1.0) * 0.5 + min(road / 20.0, 1.0) * 0.5

    return 0.4 * size_score + 0.3 * beh_score + 0.2 * cpx_score + 0.1 * terrain_score


def select(df: pd.DataFrame) -> None:
    """
    Display enriched fires ranked by selection score.
    User picks 4 by row number → exports scenario_fires.csv.
    """
    # Only show enriched rows with valid coordinates
    valid = df[
        (df["InitialLatitude"].notna()) &
        (df["InitialLongitude"].notna()) &
        (df["terrain_enriched"] == 1)
    ].copy()

    if len(valid) == 0:
        print("\n  No enriched fires found. Run enrichment (Step 1) first.")
        return

    valid["_score"] = valid.apply(score_for_selection, axis=1)
    valid = valid.sort_values("_score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("  FIRE SELECTION TABLE — ranked by operational interest")
    print("=" * 100)
    print(f"  {'#':>3}  {'Incident':<22}  {'County':<12}  {'IncidentSz':>10}  "
          f"{'Behavior':<10}  {'Complexity':<18}  {'Slope%':>7}  {'Road km':>8}  {'Score':>6}")
    print("  " + "-" * 96)

    for i, (_, row) in enumerate(valid.iterrows()):
        inc_size = row.get("IncidentSize")
        inc_size_str = f"{inc_size:,.0f}" if pd.notna(inc_size) else f"{row.get('DiscoveryAcres',0):,.0f}*"
        slope = row.get("terrain_slope_pct")
        road  = row.get("road_distance_km")
        print(
            f"  {i+1:>3}  {str(row['IncidentName']):<22}  "
            f"{str(row.get('POOCounty','')):<12}  "
            f"{inc_size_str:>10}  "
            f"{str(row.get('FireBehaviorGeneral','')):<10}  "
            f"{str(row.get('FireMgmtComplexity','')):<18}  "
            f"{slope if pd.notna(slope) else '?':>6.1f}%  "
            f"{road if pd.notna(road) else '?':>7.1f}km  "
            f"{row['_score']:>6.2f}"
        )

    print("\n  * IncidentSize not available — showing DiscoveryAcres")
    print(
        "\n  Tip: choose fires that create tradeoffs:\n"
        "    - one large operational fire (high incident size)\n"
        "    - one extreme/active behavior fire\n"
        "    - one steep or road-poor fire (high slope, high road km)\n"
        "    - one high-exposure fire near populated areas\n"
    )

    # Get user selections
    selected_indices = []
    while len(selected_indices) < 4:
        remaining = 4 - len(selected_indices)
        try:
            raw = input(f"  Enter {remaining} more number(s) from the table above "
                        f"(comma-separated or one at a time): ").strip()
            nums = [int(x.strip()) for x in raw.split(",") if x.strip()]
            for n in nums:
                if n < 1 or n > len(valid):
                    print(f"    ✗ {n} is out of range. Pick 1–{len(valid)}.")
                elif n in selected_indices:
                    print(f"    ✗ {n} already selected.")
                else:
                    selected_indices.append(n)
                    row = valid.iloc[n - 1]
                    print(f"    ✓ Added #{n}: {row['IncidentName']}")
                if len(selected_indices) == 4:
                    break
        except ValueError:
            print("    ✗ Enter numbers only.")
        except KeyboardInterrupt:
            print("\n  Aborted.")
            return

    chosen = valid.iloc[[i - 1 for i in selected_indices]].copy()

    # Build scenario_fires.csv
    out_rows = []
    for fire_idx, (_, row) in enumerate(chosen.iterrows()):
        out_row = {v: row.get(k) for k, v in COLUMN_MAP.items()}
        out_row["fire_name"] = f"Fire {fire_idx+1}" if False else row["IncidentName"]
        for col, default in DEFAULTS.items():
            out_row[col] = default
        out_rows.append(out_row)

    scenario = pd.DataFrame(out_rows)

    # Column order expected by v5 model
    col_order = [
        "fire_name", "county", "fire_lat", "fire_lon",
        "discovery_acres", "incident_size", "fuel_group",
        "temperature_c", "wind_speed_mps", "wind_dir_deg", "humidity_pct",
        "fire_behavior", "mgmt_complexity", "estimated_cost",
        "terrain_slope_pct", "road_distance_km",
    ]
    scenario = scenario[[c for c in col_order if c in scenario.columns]]

    os.makedirs(SCENARIO_DIR, exist_ok=True)
    scenario.to_csv(SCENARIO_CSV, index=True, index_label="")
    print(f"\n  Exported {len(scenario)} fires → {SCENARIO_CSV}")
    print("\n  Selected fires:")
    for _, row in scenario.iterrows():
        print(f"    {row['fire_name']:<22}  "
              f"incident_size={row.get('incident_size','?')}ac  "
              f"slope={row.get('terrain_slope_pct','?')}%  "
              f"road={row.get('road_distance_km','?')}km")

    print("\n  Next step: run  streamlit run dashboard.py  or  python wildfire_triage.py")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("  build_scenario.py — WA Wildfire Terrain Enrichment + Selector")
    print("=" * 65)

    if not os.path.exists(WA_CSV):
        print(f"\n  ERROR: {WA_CSV} not found.")
        print(f"  Place wa_fires_near_seattle.csv in the same directory as this script.")
        sys.exit(1)

    df = pd.read_csv(WA_CSV)
    enriched_count = int((df.get("terrain_enriched", pd.Series()) == 1).sum())
    total_count    = len(df)

    print(f"\n  Loaded {WA_CSV}")
    print(f"  Total fires   : {total_count}")
    print(f"  Already enriched: {enriched_count}")
    print(f"  Pending       : {total_count - enriched_count}")

    print("\n  What do you want to do?")
    print("    1) Enrich all fires with terrain data (runs API calls — needs internet)")
    print("    2) Skip enrichment, go straight to fire selection")
    print("    3) Both: enrich then select")

    while True:
        try:
            choice = input("\n  Choice [1/2/3]: ").strip()
        except KeyboardInterrupt:
            print("\n  Aborted.")
            sys.exit(0)

        if choice == "1":
            enrich(WA_CSV)
            break
        elif choice == "2":
            if enriched_count == 0:
                print("  Warning: no fires are enriched yet. "
                      "Terrain columns will show default values in the scenario.")
            break
        elif choice == "3":
            enrich(WA_CSV)
            break
        else:
            print("  Enter 1, 2, or 3.")

    df = pd.read_csv(WA_CSV)
    select(df)


if __name__ == "__main__":
    main()