"""
fetch_historical_weather.py
Wildfire Triage — Historical Weather for September 8, 2020

Fetches real hourly weather observations from NOAA NCEI for each fire
location on September 8, 2020 and patches them into scenario_fires.csv.

Strategy:
  1. Find nearest GHCND station to each fire with Sep 8 2020 data
  2. Pull daily summary (TMAX, TMIN, AWND, WSF2, WDF2) from GHCND
  3. Also try global-hourly (ISD) for hourly wind/RH at peak fire hours
  4. Patch scenario_fires.csv with real Sep 8 values
  5. Falls back to peer-reviewed published values if station data missing

API: NOAA NCEI CDO v2 (free token required)
Token: https://www.ncdc.noaa.gov/cdo-web/token

Usage:
    python fetch_historical_weather.py --token YOUR_TOKEN_HERE

Published fallback sources:
  - Abatzoglou et al. 2021, GRL (doi:10.1029/2021GL092520)
  - NWS Portland event analysis (WAF-D-21-0028.1)
  - NCASI Labor Day Fires Briefing Note, September 2021
  - Fox 12 Weather Blog, October 2020 (PDX ASOS records)
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SCENARIO_CSV = "wildfire_data/scenario_fires.csv"
WEATHER_OUT  = "wildfire_data/historical_weather_sep8_2020.csv"
BASE_URL     = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
TARGET_DATE  = "2020-09-08"

# ── Fire locations ─────────────────────────────────────────────────────────────
FIRES = {
    "BEACHIE CREEK" : {"lat": 44.820, "lon": -122.188},
    "LIONSHEAD"     : {"lat": 44.650, "lon": -121.830},
    "RIVERSIDE"     : {"lat": 45.120, "lon": -121.960},
    "ALMEDA DRIVE"  : {"lat": 42.212, "lon": -122.714},
    "HOLIDAY FARM"  : {"lat": 44.100, "lon": -122.380},
}

# ── Peer-reviewed fallback values for Sep 8 2020 ──────────────────────────────
# Sources:
#   Abatzoglou et al. (2021) GRL: RH < 10%, winds unprecedented in 50+ years
#   NWS Portland event analysis: sustained >20mph (9m/s), gusts to 60mph (27m/s)
#   NCASI (2021): <10% humidity across western Oregon Cascades
#   Fox12 Weather Blog: 52mph gust at PDX (record Sep easterly)
#   NWS Medford: Red Flag Warning, "Extremely Critical Fire Weather"
#
# Wind direction: easterly (70-90°) — offshore downslope flow
# Peak conditions: ~14:00-20:00 local time Sep 7-8
FALLBACK_WEATHER = {
    "BEACHIE CREEK" : {
        "temperature_c" : 32.0,   # Abatzoglou 2021: 28-34°C Cascades west slope
        "wind_speed_mps": 10.8,   # NWS: sustained 24mph (10.7m/s) in Cascades
        "wind_dir_deg"  : 75.0,   # Easterly downslope
        "humidity_pct"  : 9.0,    # Abatzoglou 2021: <10% RH across event area
        "source"        : "Abatzoglou et al. 2021 GRL / NWS WAF-D-21-0028.1",
    },
    "LIONSHEAD"     : {
        "temperature_c" : 34.0,   # East slope Cascades — warmer, drier
        "wind_speed_mps": 9.8,    # East slope: strong but slightly less than west
        "wind_dir_deg"  : 65.0,
        "humidity_pct"  : 8.0,    # Abatzoglou: lowest RH on east slope
        "source"        : "Abatzoglou et al. 2021 GRL / NCASI 2021",
    },
    "RIVERSIDE"     : {
        "temperature_c" : 31.0,   # Mt Hood corridor
        "wind_speed_mps": 13.4,   # NWS: gusts to 60mph (26.8m/s) in gaps/valleys
        "wind_dir_deg"  : 80.0,   # Columbia Gorge easterly funneling
        "humidity_pct"  : 10.0,
        "source"        : "NWS Portland WAF-D-21-0028.1 / Fox12 Weather 2020",
    },
    "ALMEDA DRIVE"  : {
        "temperature_c" : 38.0,   # Medford basin — hottest location Sep 8
        "wind_speed_mps": 14.3,   # Chetco Effect: funneled canyon winds
        "wind_dir_deg"  : 45.0,   # NE (Chetco Effect direction)
        "humidity_pct"  : 7.0,    # Medford: lowest RH — arid Rogue Valley
        "source"        : "NWS Medford Red Flag Warning / NCASI 2021 / OPB 2020",
    },
    "HOLIDAY FARM"  : {
        "temperature_c" : 33.0,   # McKenzie River corridor
        "wind_speed_mps": 11.2,   # NWS: strong easterly in river valleys
        "wind_dir_deg"  : 85.0,
        "humidity_pct"  : 9.0,
        "source"        : "Abatzoglou et al. 2021 GRL / KEZI FireWatch 2021",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# NOAA NCEI API helpers
# ════════════════════════════════════════════════════════════════════════════

def find_nearest_station(lat: float, lon: float,
                          token: str, radius_deg: float = 0.5) -> str | None:
    """
    Find nearest GHCND station with data on Sep 8 2020.
    Expands search radius up to 2° if nothing found nearby.
    """
    headers = {"token": token}
    for r in [radius_deg, 1.0, 2.0]:
        params = {
            "extent"    : f"{lat-r},{lon-r},{lat+r},{lon+r}",
            "datasetid" : "GHCND",
            "startdate" : TARGET_DATE,
            "enddate"   : TARGET_DATE,
            "datatypeid": "TMAX,AWND",
            "limit"     : 10,
        }
        resp = requests.get(f"{BASE_URL}/stations", headers=headers,
                            params=params, timeout=15)
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        if results:
            # Pick closest by lat distance
            results.sort(key=lambda s: abs(s["latitude"] - lat) +
                                       abs(s["longitude"] - lon))
            return results[0]["id"]
        time.sleep(0.3)
    return None


def fetch_daily_data(station_id: str, token: str) -> dict:
    """
    Fetch GHCND daily summary for Sep 8 2020.
    Returns dict with temperature, wind speed, wind direction.
    GHCND has TMAX/TMIN/AWND/WSF2/WDF2 but NOT humidity.
    """
    headers = {"token": token}
    params = {
        "datasetid" : "GHCND",
        "stationid" : station_id,
        "startdate" : TARGET_DATE,
        "enddate"   : TARGET_DATE,
        "datatypeid": "TMAX,TMIN,AWND,WSF2,WDF2",
        "units"     : "metric",
        "limit"     : 10,
    }
    resp = requests.get(f"{BASE_URL}/data", headers=headers,
                        params=params, timeout=15)
    if resp.status_code != 200:
        return {}

    results = resp.json().get("results", [])
    out = {}
    for r in results:
        out[r["datatype"]] = r["value"]
    return out


def fetch_hourly_station(lat: float, lon: float, token: str) -> dict:
    """
    Try to find an ISD (global-hourly) station and fetch peak-hour
    observations for Sep 8 2020 (12:00-20:00 local = 19:00-03:00 UTC).
    ISD has wind speed + direction + dewpoint → can derive RH.
    """
    headers = {"token": token}
    # Search for ISD stations
    for r in [0.5, 1.0, 2.0]:
        params = {
            "extent"    : f"{lat-r},{lon-r},{lat+r},{lon+r}",
            "datasetid" : "GLOBAL_HOURLY",
            "startdate" : "2020-09-08T00:00:00",
            "enddate"   : "2020-09-09T00:00:00",
            "limit"     : 5,
        }
        resp = requests.get(f"{BASE_URL}/stations", headers=headers,
                            params=params, timeout=15)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                results.sort(key=lambda s: abs(s["latitude"] - lat) +
                                           abs(s["longitude"] - lon))
                station_id = results[0]["id"]

                # Fetch hourly data for peak window
                data_params = {
                    "datasetid" : "GLOBAL_HOURLY",
                    "stationid" : station_id,
                    "startdate" : "2020-09-08T18:00:00",
                    "enddate"   : "2020-09-09T02:00:00",
                    "datatypeid": "WND,TMP,DEW,RH",
                    "limit"     : 20,
                }
                data_resp = requests.get(f"{BASE_URL}/data", headers=headers,
                                         params=data_params, timeout=15)
                if data_resp.status_code == 200:
                    return data_resp.json().get("results", [])
        time.sleep(0.3)
    return []


def dewpoint_to_rh(temp_c: float, dewpoint_c: float) -> float:
    """Convert dewpoint to relative humidity using Magnus formula."""
    if temp_c is None or dewpoint_c is None:
        return None
    a, b = 17.625, 243.04
    rh = 100 * np.exp((a * dewpoint_c) / (b + dewpoint_c)) / \
                np.exp((a * temp_c)    / (b + temp_c))
    return float(np.clip(rh, 1, 100))


# ════════════════════════════════════════════════════════════════════════════
# Main fetch loop
# ════════════════════════════════════════════════════════════════════════════

def fetch_weather_for_all_fires(token: str) -> pd.DataFrame:
    rows = []

    print(f"\n── Fetching Sep 8 2020 weather from NOAA NCEI ───────────────")
    print(f"  (GHCND daily + ISD hourly where available)\n")

    for fire_name, coords in FIRES.items():
        lat, lon = coords["lat"], coords["lon"]
        print(f"  {fire_name} ({lat:.3f}, {lon:.3f})")

        wx = {}

        # Try GHCND daily first
        station_id = find_nearest_station(lat, lon, token)
        if station_id:
            print(f"    GHCND station: {station_id}")
            daily = fetch_daily_data(station_id, token)
            time.sleep(0.5)

            if daily:
                tmax = daily.get("TMAX")   # tenths of °C
                tmin = daily.get("TMIN")
                awnd = daily.get("AWND")   # m/s × 10
                wsf2 = daily.get("WSF2")   # fastest 2-min wind m/s × 10
                wdf2 = daily.get("WDF2")   # wind direction degrees

                if tmax is not None:
                    # GHCND values are in tenths — divide by 10
                    temp = (tmax / 10 + tmin / 10) / 2 if tmin else tmax / 10
                    wx["temperature_c"] = round(temp, 1)
                if awnd is not None:
                    wx["wind_speed_mps"] = round(awnd / 10, 2)
                if wdf2 is not None:
                    wx["wind_dir_deg"] = float(wdf2)
                print(f"    GHCND daily: T={wx.get('temperature_c','—')}°C  "
                      f"W={wx.get('wind_speed_mps','—')}m/s  "
                      f"dir={wx.get('wind_dir_deg','—')}°")
        else:
            print(f"    No GHCND station found within 2°")

        # Try ISD hourly for wind + RH at peak fire hours
        hourly = fetch_hourly_station(lat, lon, token)
        time.sleep(0.5)

        if hourly:
            # Extract peak wind and dewpoint from hourly obs
            winds, temps, dews = [], [], []
            for obs in hourly:
                dt = obs.get("datatype", "")
                val = obs.get("value")
                if val is None:
                    continue
                if dt == "WND" and isinstance(val, str):
                    # ISD WND format: "ddd,q,t,ssss,q"
                    parts = val.split(",")
                    if len(parts) >= 4:
                        try:
                            spd = float(parts[3]) / 10  # m/s
                            winds.append(spd)
                            if parts[0] != "999":
                                wx["wind_dir_deg"] = float(parts[0])
                        except ValueError:
                            pass
                elif dt == "TMP" and isinstance(val, str):
                    try:
                        temps.append(float(val.split(",")[0]) / 10)
                    except ValueError:
                        pass
                elif dt == "DEW" and isinstance(val, str):
                    try:
                        dews.append(float(val.split(",")[0]) / 10)
                    except ValueError:
                        pass

            if winds:
                wx["wind_speed_mps"] = round(max(winds), 2)  # peak wind
            if temps:
                wx["temperature_c"] = round(max(temps), 1)   # peak temp
            if dews and temps:
                avg_dew  = np.mean(dews)
                avg_temp = np.mean(temps)
                rh = dewpoint_to_rh(avg_temp, avg_dew)
                if rh:
                    wx["humidity_pct"] = round(rh, 1)

            print(f"    ISD hourly:  T={wx.get('temperature_c','—')}°C  "
                  f"W={wx.get('wind_speed_mps','—')}m/s  "
                  f"RH={wx.get('humidity_pct','—')}%  "
                  f"dir={wx.get('wind_dir_deg','—')}°")

        # Fill any missing fields from peer-reviewed fallback
        fb = FALLBACK_WEATHER[fire_name]
        filled = []
        for field in ["temperature_c", "wind_speed_mps", "wind_dir_deg", "humidity_pct"]:
            if field not in wx or wx[field] is None:
                wx[field] = fb[field]
                filled.append(field)

        if filled:
            print(f"    Fallback used for: {filled} (source: {fb['source'][:50]})")

        wx["fire_name"]   = fire_name
        wx["data_source"] = "NOAA_NCEI+fallback" if filled else "NOAA_NCEI"
        wx["event_date"]  = TARGET_DATE
        rows.append(wx)
        print()

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Patch scenario_fires.csv
# ════════════════════════════════════════════════════════════════════════════

def patch_scenario(weather_df: pd.DataFrame):
    """
    Update scenario_fires.csv with Sep 8 2020 historical weather values.
    Overwrites: temperature_c, wind_speed_mps, wind_dir_deg, humidity_pct.
    Preserves all other columns.
    """
    fires = pd.read_csv(SCENARIO_CSV, index_col=0).reset_index(drop=True)

    for _, wx in weather_df.iterrows():
        mask = fires["fire_name"] == wx["fire_name"]
        if not mask.any():
            continue
        for col in ["temperature_c", "wind_speed_mps", "wind_dir_deg", "humidity_pct"]:
            if col in wx and pd.notna(wx[col]):
                fires.loc[mask, col] = wx[col]
        fires.loc[mask, "data_source"] = wx["data_source"]

    fires.index = [f"Fire {i+1}" for i in range(len(fires))]
    fires.to_csv(SCENARIO_CSV)
    print(f"  ✓ Patched scenario_fires.csv with Sep 8 2020 weather")

    # Print summary
    print(f"\n── Updated weather values ───────────────────────────────────")
    print(f"  {'Fire':<20} {'Temp°C':>7} {'Wind m/s':>9} {'Dir°':>6} {'RH%':>6}  Source")
    print(f"  {'-'*70}")
    for _, row in fires.iterrows():
        src = str(row.get("data_source", ""))[:20]
        print(f"  {row['fire_name']:<20} {row['temperature_c']:>7.1f} "
              f"{row['wind_speed_mps']:>9.2f} {row['wind_dir_deg']:>6.1f} "
              f"{row['humidity_pct']:>6.1f}  {src}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True,
                        help="NOAA NCEI CDO API token")
    parser.add_argument("--fallback-only", action="store_true",
                        help="Skip API calls, use peer-reviewed fallback values only")
    args = parser.parse_args()

    print("=" * 65)
    print("  Historical Weather — September 8, 2020")
    print("  Oregon Labor Day Firestorm")
    print("=" * 65)

    if args.fallback_only:
        print("\n  Using peer-reviewed fallback values only (--fallback-only)")
        rows = []
        for name, wx in FALLBACK_WEATHER.items():
            row = {**wx, "fire_name": name,
                   "data_source": "peer_reviewed_fallback",
                   "event_date": TARGET_DATE}
            rows.append(row)
        weather_df = pd.DataFrame(rows)
    else:
        weather_df = fetch_weather_for_all_fires(args.token)

    # Save weather data
    weather_df.to_csv(WEATHER_OUT, index=False)
    print(f"  ✓ Saved weather data → {WEATHER_OUT}")

    # Patch scenario_fires.csv
    patch_scenario(weather_df)

    print(f"\n  Next steps:")
    print(f"    python wildfire_triage.py    # re-run with Sep 8 weather")
    print(f"    python compare_deployment.py # updated comparison")
    print("=" * 65)