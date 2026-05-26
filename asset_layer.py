"""
asset_layer.py
Wildfire Triage — Layer 4: Asset Value Integration

Takes OSM asset GeoJSON and rewrites the IP objective from:
    maximize  risk_score × acres_per_day × units
to:
    maximize  risk_score × asset_damage_per_acre × acres_per_day × units

Where asset_damage_per_acre = weighted sum of assets in each fire's spread path,
normalized per acre so fires with denser infrastructure score higher.

Usage:
    python asset_layer.py

Requires:
    wildfire_data/osm_assets.geojson   (from fetch_osm_assets.py)
    wildfire_data/scenario_fires.csv
    wildfire_data/resources.csv
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import warnings
warnings.filterwarnings("ignore")
from pulp import *

FIRES_CSV     = "wildfire_data/scenario_fires.csv"
RESOURCES_CSV = "wildfire_data/resources.csv"
ASSETS_GEOJSON = "wildfire_data/osm_assets.geojson"

GRID_SIZE    = 40
CELL_M       = 100
HOUR_SECS    = 3600
LAT_PER_M    = 1 / 111_320
DAILY_BUDGET = 500_000
COVERAGE_CAP = 2.0

# Risk weight config (same as main model)
W_SIZE = 0.20; W_WEATHER = 0.35; W_BEHAVIOR = 0.30; W_COMPLEXITY = 0.15
W_HUMIDITY = 0.50; W_WIND = 0.30; W_TEMP = 0.20


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: Build grid cell geometries for each fire
# ════════════════════════════════════════════════════════════════════════════

def build_grid_geodataframe(fire_lat, fire_lon, fire_name):
    """
    Create a GeoDataFrame where each row is one 100m×100m grid cell,
    with its real-world geometry (a square polygon in lat/lon space).
    """
    centre    = GRID_SIZE // 2
    lon_per_m = 1 / (111_320 * np.cos(np.radians(fire_lat)))

    rows = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            d_row = r - centre
            d_col = c - centre

            # Lat/lon of cell edges
            lat_top = fire_lat - d_row       * CELL_M * LAT_PER_M
            lat_bot = fire_lat - (d_row + 1) * CELL_M * LAT_PER_M
            lon_lft = fire_lon + d_col       * CELL_M * lon_per_m
            lon_rgt = fire_lon + (d_col + 1) * CELL_M * lon_per_m

            rows.append({
                "row"      : r,
                "col"      : c,
                "fire_name": fire_name,
                "geometry" : box(lon_lft, lat_bot, lon_rgt, lat_top),
            })

    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 & 3: Asset score using 12-hour spread radius (haversine distance)
#
# The grid join approach fails when OSM data was fetched at a larger radius
# than the 4km grid. Instead, we directly measure which assets fall within
# each fire's 12-hour maximum spread distance, computed from wind speed.
# This is more physically meaningful anyway — it answers "what is in the
# fire's path over the next 12 hours?" rather than "what's at the ignition?"
# ════════════════════════════════════════════════════════════════════════════

def spread_radius_km(wind_speed_mps, hours=12):
    """
    Maximum fire spread distance in km over `hours` hours.
    Uses the same Rothermel-inspired formula as the grid model:
        base_speed = 0.03 × wind_speed_mps
        downwind amplification factor = 3×
    """
    base_ms = 0.03 * wind_speed_mps
    max_ms  = base_ms * 3.0
    return (max_ms * hours * 3600) / 1000


def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    """Vectorised haversine distance in km."""
    R    = 6371.0
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a    = (np.sin(dlat/2)**2 +
            np.cos(np.radians(lat1)) * np.cos(np.radians(lat2_arr)) *
            np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))


def compute_fire_asset_score(fires_df, assets_gdf):
    """
    For each fire, sum the asset weights of all OSM features
    within the fire's 12-hour maximum spread radius.
    Normalized to a 1-10 scale across fires.
    """
    print("\n── Computing asset value within 12h spread radius ───────────")
    scores = []

    for _, fire in fires_df.iterrows():
        name   = fire["fire_name"]
        radius = spread_radius_km(fire.get("wind_speed_mps", 3.0))
        fa     = (assets_gdf[assets_gdf["fire_name"] == name].copy()
                  if assets_gdf is not None else pd.DataFrame())

        if fa.empty:
            total_w, n = 0.0, 0
        else:
            dist   = haversine_km(
                fire["fire_lat"], fire["fire_lon"],
                fa["centroid_lat"].values, fa["centroid_lon"].values
            )
            within  = fa[dist <= radius]
            total_w = within["asset_weight"].sum()
            n       = len(within)

        scores.append({
            "fire_name"          : name,
            "spread_radius_km"   : round(radius, 2),
            "n_assets_in_radius" : n,
            "total_asset_weight" : round(total_w, 1),
        })
        print(f"  {name:<20}  radius={radius:.1f}km  "
              f"assets_in_path={n:>4}  total_weight={total_w:>8.1f}")

    score_df = pd.DataFrame(scores)
    mn = score_df["total_asset_weight"].min()
    mx = score_df["total_asset_weight"].max()
    score_df["asset_score_10"] = (
        (1 + 9*(score_df["total_asset_weight"]-mn)/(mx-mn)).round(2)
        if mx > mn else 5.0
    )
    return score_df


# ════════════════════════════════════════════════════════════════════════════
# STEP 4: Rebuild risk scores with asset weighting
# ════════════════════════════════════════════════════════════════════════════

def compute_asset_weighted_risk(fires_df, asset_scores):
    """
    Merge asset scores into fires and recompute risk score with an
    additional asset damage component.

    New risk formula:
        risk = 0.15×size + 0.30×weather + 0.25×behavior +
               0.12×complexity + 0.18×asset_damage
    (asset replaces some complexity weight — it's empirically grounded)
    """
    df = fires_df.merge(asset_scores[["fire_name","asset_score_10"]], on="fire_name")

    df["wind_speed_mps"] = df["wind_speed_mps"].fillna(3.0)
    df["wind_dir_deg"]   = df["wind_dir_deg"].fillna(270.0)
    df["temperature_c"]  = df["temperature_c"].fillna(15.0)

    df["size_score"] = np.log1p(df["discovery_acres"])
    df["size_score"] = df["size_score"] / df["size_score"].max()

    df["humidity_risk"] = 1 - (df["humidity_pct"] / 100)
    df["wind_risk"]     = df["wind_speed_mps"] / df["wind_speed_mps"].max()
    tr = df["temperature_c"].max() - df["temperature_c"].min()
    df["temp_risk"]     = (df["temperature_c"] - df["temperature_c"].min()) / tr if tr > 0 else 0.5
    df["weather_score"] = (W_HUMIDITY*df["humidity_risk"] +
                           W_WIND*df["wind_risk"] + W_TEMP*df["temp_risk"])

    behavior_map   = {"Minimal":0.2,"Moderate":0.5,"Active":0.8,"Extreme":1.0}
    complexity_map = {"Type 1 Incident":1.0,"Type 2 Incident":0.75,
                      "Type 3 Incident":0.5,"Type 4 Incident":0.25,"Type 5 Incident":0.1}
    df["behavior_score"]   = df["fire_behavior"].map(behavior_map).fillna(0.3)
    df["complexity_score"] = df["mgmt_complexity"].map(complexity_map).fillna(0.3)

    # Normalize asset score to 0-1
    df["asset_norm"] = (df["asset_score_10"] - 1) / 9

    # Asset-weighted risk
    df["risk_score_asset"] = (
        0.15 * df["size_score"]       +
        0.30 * df["weather_score"]    +
        0.25 * df["behavior_score"]   +
        0.12 * df["complexity_score"] +
        0.18 * df["asset_norm"]
    )
    df["risk_score_asset_100"] = (
        df["risk_score_asset"] / df["risk_score_asset"].max() * 100
    ).round(1)
    df["priority_rank_asset"] = df["risk_score_asset"].rank(ascending=False).astype(int)

    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 5: IP optimizer with asset-weighted objective
# ════════════════════════════════════════════════════════════════════════════

def run_asset_optimizer(fires_df, resources, asset_scores):
    """
    Same IP structure as before, but objective multiplies by asset_score_10:
        maximize  risk_score × asset_score × acres_per_day × units
    
    This means the optimizer strongly prefers covering high-risk fires
    that also have high-value infrastructure in their path.
    """
    # fires_df already has asset_score_10 merged in compute_asset_weighted_risk
    # fall back gracefully if missing (e.g. all-zero asset areas)
    fires_merged = fires_df.copy()
    if "asset_score_10" not in fires_merged.columns:
        fires_merged = fires_merged.merge(
            asset_scores[["fire_name", "asset_score_10"]], on="fire_name", how="left"
        )
    fires_merged["asset_score_10"] = fires_merged["asset_score_10"].fillna(1.0)

    fire_names     = fires_merged["fire_name"].tolist()
    resource_names = resources["resource"].tolist()
    units_avail    = dict(zip(resource_names, resources["units_available"]))
    acres_pd       = dict(zip(resource_names, resources["acres_per_day"]))
    cost_pd        = dict(zip(resource_names, resources["cost_per_day"]))
    demand         = dict(zip(fire_names, fires_merged["discovery_acres"]))
    risk           = dict(zip(fire_names, fires_merged["risk_score_asset_100"]))
    asset_score    = dict(zip(fire_names, fires_merged["asset_score_10"]))

    model = LpProblem("Wildfire_Asset_Triage", LpMaximize)
    x = {
        (r, f): LpVariable(
            f"x_{r.replace(' ','_').replace('-','_')}_{f.replace(' ','_')}",
            lowBound=0, cat="Integer"
        )
        for r in resource_names for f in fire_names
    }

    # Objective: risk × asset damage × coverage
    model += lpSum(
        risk[f] * asset_score[f] * acres_pd[r] * x[(r, f)]
        for r in resource_names for f in fire_names
    )

    for r in resource_names:
        model += lpSum(x[(r, f)] for f in fire_names) <= units_avail[r]
    model += lpSum(
        cost_pd[r] * x[(r, f)] for r in resource_names for f in fire_names
    ) <= DAILY_BUDGET
    for f in fire_names:
        model += lpSum(x[(r, f)] for r in resource_names) >= 1
    for f in fire_names:
        model += lpSum(
            acres_pd[r] * x[(r, f)] for r in resource_names
        ) <= COVERAGE_CAP * demand[f]

    model.solve(PULP_CBC_CMD(msg=0))

    allocation = {
        (r, f): int(value(x[(r, f)]) or 0)
        for r in resource_names for f in fire_names
    }
    return allocation, LpStatus[model.status], fires_merged, acres_pd, cost_pd, demand


# ════════════════════════════════════════════════════════════════════════════
# REPORTING
# ════════════════════════════════════════════════════════════════════════════

def print_comparison(fires_base, fires_asset, allocation_base,
                     allocation_asset, resource_names, acres_pd):
    print("\n" + "=" * 65)
    print("  LAYER 4 — ASSET VALUE IMPACT ON PRIORITY & ALLOCATION")
    print("=" * 65)

    print("\n── Risk score comparison (before vs after asset layer) ──────")
    print(f"  {'Fire':<20} {'Base Risk':>10} {'Asset Risk':>11} "
          f"{'Base Rank':>10} {'Asset Rank':>11} {'Rank Shift':>11}")
    print(f"  {'-'*70}")

    for _, row in fires_asset.sort_values("priority_rank_asset").iterrows():
        f         = row["fire_name"]
        base_risk = fires_base.loc[fires_base["fire_name"]==f, "risk_score_100"].values[0]
        base_rank = int(fires_base.loc[fires_base["fire_name"]==f, "priority_rank"].values[0])
        shift     = base_rank - row["priority_rank_asset"]
        arrow     = ("↑" * abs(shift) if shift > 0 else
                     "↓" * abs(shift) if shift < 0 else "—")
        print(f"  {f:<20} {base_risk:>10.1f} {row['risk_score_asset_100']:>11.1f} "
              f"{'#'+str(base_rank):>10} {'#'+str(row['priority_rank_asset']):>11} "
              f"{arrow:>11}")

    print("\n── Allocation comparison ─────────────────────────────────────")
    print(f"  {'Fire':<20} {'Coverage (base)':>16} {'Coverage (asset)':>17}")
    print(f"  {'-'*55}")
    for f in fires_asset["fire_name"].tolist():
        cov_base  = sum(allocation_base.get((r,f),0)*acres_pd[r] for r in resource_names)
        cov_asset = sum(allocation_asset.get((r,f),0)*acres_pd[r] for r in resource_names)
        print(f"  {f:<20} {cov_base:>14,.0f} ac  {cov_asset:>14,.0f} ac")

    print("\n── Key insight ───────────────────────────────────────────────")
    rank_changes = [
        row["fire_name"]
        for _, row in fires_asset.iterrows()
        if row["priority_rank_asset"] != int(
            fires_base.loc[fires_base["fire_name"]==row["fire_name"],
                           "priority_rank"].values[0]
        )
    ]
    if rank_changes:
        print(f"  Fires that changed rank: {', '.join(rank_changes)}")
        print("  → Asset density in spread path changed priority ordering.")
        print("    This is the key result: raw acreage alone is misleading.")
    else:
        print("  Rankings unchanged — asset distribution is uniform across fires.")
        print("  Try fires closer to urban areas to see rank shifts.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  LAYER 4 — OSM ASSET VALUE INTEGRATION")
    print("=" * 65)

    fires     = pd.read_csv(FIRES_CSV, index_col=0).reset_index(drop=True)
    resources = pd.read_csv(RESOURCES_CSV)

    # Load base risk scores (from Layer 1)
    from wildfire_triage import compute_risk_scores, run_optimizer
    fires_base = compute_risk_scores(fires.copy())

    # Load OSM assets
    print(f"\n  Loading OSM assets from {ASSETS_GEOJSON} …")
    try:
        assets_gdf = gpd.read_file(ASSETS_GEOJSON)
        print(f"  Loaded {len(assets_gdf)} assets across all fires.")
    except Exception as e:
        print(f"  ⚠ Could not load GeoJSON: {e}")
        print("  Run fetch_osm_assets.py first.")
        exit(1)

    # Step 3: Asset scores per fire
    asset_scores = compute_fire_asset_score(fires_base, assets_gdf)
    asset_scores.to_csv("wildfire_data/asset_scores.csv", index=False)

    # Step 4: Asset-weighted risk
    fires_asset = compute_asset_weighted_risk(fires_base.copy(), asset_scores)

    print("\n── Asset scores ──────────────────────────────────────────────")
    print(asset_scores[["fire_name","n_assets_in_radius","total_asset_weight",
                         "asset_score_10"]].to_string(index=False))

    # Step 5: Base IP (no asset layer)
    result_base = run_optimizer(fires_base, resources)
    alloc_base  = result_base["allocation"]
    apd         = result_base["acres_per_day"]
    rnames      = result_base["resource_names"]

    # Step 5b: Asset-weighted IP
    alloc_asset, status, fires_merged, apd2, cpd2, demand2 = run_asset_optimizer(
        fires_asset, resources, asset_scores
    )

    # Compare
    print_comparison(fires_base, fires_asset, alloc_base, alloc_asset, rnames, apd)

    print("\n  Saved asset_scores.csv → wildfire_data/")
    print("=" * 65)
    print("  Done.")
    print("=" * 65)