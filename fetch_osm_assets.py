"""
fetch_osm_assets.py
Wildfire Triage — OSM Asset Value Layer
Run this locally after fetch_data.py.

Usage:
    pip install osmnx geopandas
    python fetch_osm_assets.py

Output:
    wildfire_data/osm_assets.geojson   ← upload this
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

OUTPUT_DIR = "wildfire_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIRES_CSV = "wildfire_data/scenario_fires.csv"
GRID_SIZE = 40
CELL_M    = 100       # metres per cell
LAT_PER_M = 1 / 111_320

# ── Asset tags to pull from OSM ──────────────────────────────────────────────
TAGS = {
    "amenity" : ["hospital", "clinic", "school", "fire_station", "police"],
    "building": ["apartments", "residential", "commercial", "industrial"],
    "landuse" : ["residential", "industrial", "commercial"],
}

# ── Damage weights: how much does losing this asset hurt? ────────────────────
# Scale 1-10. Used to weight the IP objective per grid cell.
ASSET_WEIGHTS = {
    # Critical infrastructure
    "hospital"    : 10.0,
    "fire_station": 8.0,
    "police"      : 6.0,
    "clinic"      : 5.0,
    "school"      : 7.0,
    # Built environment
    "apartments"  : 4.0,
    "residential" : 3.0,
    "commercial"  : 2.0,
    "industrial"  : 2.0,
}


def get_bbox(fire_lat, fire_lon):
    """Return (min_lon, min_lat, max_lon, max_lat) bounding box for the fire grid."""
    centre    = GRID_SIZE // 2
    half_m = centre * CELL_M 
    half_m = 20_000 
    lon_per_m = 1 / (111_320 * np.cos(np.radians(fire_lat)))
    min_lat   = fire_lat - half_m * LAT_PER_M
    max_lat   = fire_lat + half_m * LAT_PER_M
    min_lon   = fire_lon - half_m * lon_per_m
    max_lon   = fire_lon + half_m * lon_per_m
    return min_lon, min_lat, max_lon, max_lat   # osmnx 2.x order: left,bottom,right,top


def get_weight(row):
    for col in ["amenity", "building", "landuse"]:
        if col in row.index and pd.notna(row[col]):
            return ASSET_WEIGHTS.get(str(row[col]), 1.0)
    return 1.0


def fetch_assets_for_fire(fire_name, fire_lat, fire_lon):
    bbox = get_bbox(fire_lat, fire_lon)
    print(f"  Querying OSM for {fire_name} …", end=" ", flush=True)

    try:
        gdf = ox.features_from_bbox(bbox=bbox, tags=TAGS)
    except Exception as e:
        print(f"⚠ No assets found ({e})")
        return None

    if gdf.empty:
        print("0 assets")
        return None

    # Keep useful columns
    keep = ["geometry"] + [c for c in ["amenity", "building", "landuse"] if c in gdf.columns]
    gdf  = gdf[keep].copy()

    # Use centroid as the representative point for polygons/lines
    gdf["centroid_lat"] = gdf.geometry.centroid.y
    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf["asset_weight"] = gdf.apply(get_weight, axis=1)
    gdf["fire_name"]    = fire_name

    print(f"{len(gdf)} assets  "
          f"(hospitals={len(gdf[gdf.get('amenity','') == 'hospital']) if 'amenity' in gdf.columns else 0}, "
          f"schools={len(gdf[gdf.get('amenity','') == 'school']) if 'amenity' in gdf.columns else 0}, "
          f"residential={len(gdf[gdf.get('landuse','') == 'residential']) if 'landuse' in gdf.columns else 0})")

    return gdf


if __name__ == "__main__":
    print("=" * 60)
    print("  Wildfire Triage — OSM Asset Layer")
    print("=" * 60)

    fires = pd.read_csv(FIRES_CSV, index_col=0)
    all_gdfs = []

    for _, fire in fires.iterrows():
        gdf = fetch_assets_for_fire(
            fire["fire_name"], fire["fire_lat"], fire["fire_lon"]
        )
        if gdf is not None:
            all_gdfs.append(gdf)

    if not all_gdfs:
        print("\n⚠ No assets retrieved. Check internet connection.")
    else:
        combined = pd.concat(all_gdfs, ignore_index=True)

        # Save as GeoJSON — upload this file
        out = os.path.join(OUTPUT_DIR, "osm_assets.geojson")
        combined.to_file(out, driver="GeoJSON")
        print(f"\n✓ Saved {len(combined)} total assets → {out}")

        # Also save a lightweight CSV of centroids for easy inspection
        csv_out = os.path.join(OUTPUT_DIR, "osm_assets_summary.csv")
        combined[["fire_name", "centroid_lat", "centroid_lon",
                  "asset_weight"] +
                 [c for c in ["amenity", "building", "landuse"]
                  if c in combined.columns]
        ].to_csv(csv_out, index=False)
        print(f"✓ Saved summary CSV  → {csv_out}")

        print("\n── Asset counts per fire ────────────────────────────────")
        print(combined.groupby("fire_name")["asset_weight"].agg(["count","sum","mean"])
              .rename(columns={"count":"assets","sum":"total_weight","mean":"avg_weight"})
              .to_string())

        print(f"\nNext: upload osm_assets.geojson here.")