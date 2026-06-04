"""
compare_deployment.py
Wildfire Triage — MILP vs Actual ICS-209 Deployment Comparison

This is the core analytical contribution of the project.

What it does:
    1. Loads the real scenario (from fetch_ics209.py)
    2. Loads what was actually deployed (from ICS-209-PLUS ground truth)
    3. Runs the MILP optimizer on the same resource pool
    4. Compares: optimal allocation vs actual deployment
    5. Quantifies the gap in dollars and uncovered demand

The comparison answers:
    "Given the same fires, same resource pool, and same budget,
     how does the MILP allocation differ from what incident commanders
     actually decided on September 8, 2020?"

Important caveat (included in output):
    Real commanders had information the model doesn't:
    political constraints, crew fatigue, contract availability,
    road closures, weather uncertainty. This comparison shows
    the THEORETICAL optimum, not proof of commander error.

Usage:
    python wildfire_triage.py   # must run first to generate risk scores
    python compare_deployment.py

Output:
    wildfire_data/comparison_report.csv
    Printed comparison table
"""

import os
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# Import optimizer from wildfire_triage
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wildfire_triage import (
    compute_ahp_weights,
    compute_risk_scores,
    run_resource_hour_optimizer,
    compute_terrain_scores,
    PLANNING_HORIZON_HOURS,
    DAMAGE_COST_PER_ACRE,
    LAMBDA_DAMAGE,
)

# ── Sep 8 2020 scenario-calibrated budget ─────────────────────────────────────
# Actual ICS-209 deployment across all 5 fires cost ~$936k over 6h.
# We set the MILP budget to match actual spend so the comparison is fair:
# both the model and commanders had the same effective resource envelope.
# This is the key methodological choice: hold budget constant, compare ALLOCATION.
HORIZON_BUDGET = 950_000   # ~actual 6h suppression spend, Sep 8 2020

SCENARIO_CSV    = "wildfire_data/scenario_fires.csv"
RESOURCES_CSV   = "wildfire_data/resources.csv"
ACTUAL_CSV      = "wildfire_data/actual_deployment.csv"
ASSET_CSV       = "wildfire_data/asset_scores.csv"
REPORT_OUT      = "wildfire_data/comparison_report.csv"


# ════════════════════════════════════════════════════════════════════════════
# RESOURCE CONVERSION
# Convert ICS-209 unit counts → resource-hours for fair comparison
# ════════════════════════════════════════════════════════════════════════════

# ICS-209 reports total units on the incident, not resource-hours.
# To compare apples-to-apples with the MILP (which uses resource-hours),
# we convert: resource_hours = units × productive_hours_per_day × 6/24
# (6h planning horizon = 25% of a day)
PRODUCTIVE_HOURS = {
    "Type-1 Engine"        : 10,
    "Heavy Dozer"          : 10,
    "Type-1 Helicopter"    : 7,
    "Air Tanker"           : 5,
    "Hand Crew (20-person)": 10,
}

HORIZON_FRACTION = PLANNING_HORIZON_HOURS / 24.0  # 6h/24h = 0.25

# ICS-209 field → model resource name mapping
ICS209_TO_MODEL = {
    "actual_engines"    : "Type-1 Engine",
    "actual_crews"      : "Hand Crew (20-person)",
    "actual_helicopters": "Type-1 Helicopter",
    "actual_air_tankers": "Air Tanker",
}


def ics209_units_to_resource_hours(actual_df: pd.DataFrame) -> dict:
    """
    Convert ICS-209 unit counts to 6h resource-hours.
    Returns dict: {(resource, fire_name): resource_hours}
    """
    rh = {}
    for _, row in actual_df.iterrows():
        f = row["fire_name"]
        for ics_col, model_res in ICS209_TO_MODEL.items():
            units = float(row.get(ics_col, 0) or 0)
            prod_h = PRODUCTIVE_HOURS.get(model_res, 10)
            # Resource-hours in 6h window = units × prod_h × (6/24)
            rh[(model_res, f)] = units * prod_h * HORIZON_FRACTION
    return rh


def compute_actual_coverage(actual_rh: dict,
                             fire_names: list,
                             resource_names: list,
                             aph: dict,
                             eff: dict) -> dict:
    """
    Compute what the actual ICS-209 deployment actually covers in acres.
    Uses the same effectiveness model as the MILP for fair comparison.
    """
    coverage = {}
    for f in fire_names:
        cov = sum(
            actual_rh.get((r, f), 0) * aph.get((r, f), 0) * eff.get(f, 1.0)
            for r in resource_names
        )
        coverage[f] = cov
    return coverage


def compute_actual_cost(actual_rh: dict,
                         fire_names: list,
                         resource_names: list,
                         cph: dict) -> dict:
    """
    Compute the cost of the actual ICS-209 deployment over the 6h horizon.
    """
    cost = {}
    for f in fire_names:
        cost[f] = sum(
            actual_rh.get((r, f), 0) * cph.get(r, 0)
            for r in resource_names
        )
    return cost


# ════════════════════════════════════════════════════════════════════════════
# MAIN COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def run_comparison():
    print("\n" + "=" * 70)
    print("  MILP vs ACTUAL ICS-209 DEPLOYMENT COMPARISON")
    print("  September 8, 2020 — Oregon Labor Day Firestorm")
    print("  Event: NMAC Preparedness Level 5")
    print("=" * 70)

    # Load data — auto-generate actual_deployment.csv if missing
    if not os.path.exists(SCENARIO_CSV):
        print(f"\n  ERROR: {SCENARIO_CSV} not found.")
        print(f"  Run first: python fetch_ics209.py --fallback")
        return

    fires     = pd.read_csv(SCENARIO_CSV, index_col=0).reset_index(drop=True)
    resources = pd.read_csv(RESOURCES_CSV)

    if not os.path.exists(ACTUAL_CSV):
        print(f"  actual_deployment.csv not found — generating from fallback parameters …")
        from fetch_ics209 import build_actual_deployment, TARGET_FIRES
        empty = {name: None for name in TARGET_FIRES}
        build_actual_deployment(empty, fires)

    actual_df = pd.read_csv(ACTUAL_CSV)

    # Load asset scores if available.
    # Note: existing asset_scores.csv may be from a different scenario — fire
    # names must match the current scenario to be used. If not, neutral defaults
    # are used and a message is printed. Re-run fetch_osm_assets.py to regenerate.
    asset_scores = None
    if os.path.exists(ASSET_CSV):
        asset_df  = pd.read_csv(ASSET_CSV)
        score_col = next((c for c in asset_df.columns
                          if "score" in c.lower() and "asset" in c.lower()), None)
        name_col  = next((c for c in asset_df.columns
                          if "name" in c.lower() or "fire" in c.lower()), None)
        if score_col and name_col:
            loaded  = dict(zip(asset_df[name_col], asset_df[score_col]))
            matched = {f: loaded[f] for f in fires["fire_name"] if f in loaded}
            if matched:
                asset_scores = matched
                print(f"  Asset scores loaded (col='{score_col}'): "
                      f"{len(matched)}/{len(fires)} fires matched")
            else:
                print(f"  asset_scores.csv fire names don't match current scenario")
                print(f"  Re-run fetch_osm_assets.py to generate scores for new fires")
    if not asset_scores:
        print(f"  Using neutral asset defaults (5.0 per fire)")
        asset_scores = {f: 5.0 for f in fires["fire_name"]}

    # Run AHP + risk scoring
    weights = compute_ahp_weights(verbose=False)
    fires   = compute_risk_scores(fires, weights)

    # Terrain scores
    terrain_scores = compute_terrain_scores(fires)

    # Run MILP optimizer
    print("\n  Running MILP optimizer …")
    result = run_resource_hour_optimizer(
        fires, resources,
        asset_scores=asset_scores,
        terrain_scores=terrain_scores,
        budget=HORIZON_BUDGET,
        horizon_hours=PLANNING_HORIZON_HOURS,
    )

    fire_names     = result["fire_names"]
    resource_names = result["resource_names"]
    milp_alloc     = result["allocation"]        # {(r,f): resource_hours}
    milp_coverage  = result["coverage"]          # {f: acres}
    milp_uncov     = result["uncovered"]         # {f: acres}
    demand         = result["demand"]            # {f: acres}
    danger         = result["risk_scores"]       # {f: 0-100}
    asset          = result["asset_scores"]      # {f: 1-10}
    aph_eff        = result["acres_per_hour"]    # {(r,f): aph with terrain}
    cph            = result["cost_per_hour"]     # {r: $/hr}
    eff            = result["eff"]               # {f: size-effectiveness}

    # Convert actual ICS-209 deployment to resource-hours
    actual_rh       = ics209_units_to_resource_hours(actual_df)
    actual_coverage = compute_actual_coverage(actual_rh, fire_names, resource_names, aph_eff, eff)
    actual_cost     = compute_actual_cost(actual_rh, fire_names, resource_names, cph)

    # Build comparison table
    rank_map = dict(zip(fires["fire_name"], fires["priority_rank"]))
    rows = []

    for f in sorted(fire_names, key=lambda x: rank_map[x]):
        dem   = demand[f]
        m_cov = milp_coverage[f]
        a_cov = actual_coverage[f]
        m_pct = min(m_cov / dem * 100, 100) if dem > 0 else 0
        a_pct = min(a_cov / dem * 100, 100) if dem > 0 else 0
        m_cost = sum(milp_alloc.get((r, f), 0) * cph.get(r, 0) for r in resource_names)
        a_cost = actual_cost[f]

        # Residual damage under MILP allocation
        m_uncov  = milp_uncov[f]
        m_dmg    = LAMBDA_DAMAGE * (danger[f]/100) * (asset.get(f,5)/10) * m_uncov * DAMAGE_COST_PER_ACRE
        a_uncov  = max(dem - a_cov, 0)
        a_dmg    = LAMBDA_DAMAGE * (danger[f]/100) * (asset.get(f,5)/10) * a_uncov * DAMAGE_COST_PER_ACRE

        rows.append({
            "fire_name"          : f,
            "priority_rank"      : rank_map[f],
            "risk_score"         : danger[f],
            "asset_score"        : asset.get(f, 5.0),
            "demand_6h_ac"       : dem,
            "milp_coverage_ac"   : round(m_cov, 0),
            "actual_coverage_ac" : round(a_cov, 0),
            "milp_pct_covered"   : round(m_pct, 1),
            "actual_pct_covered" : round(a_pct, 1),
            "coverage_gap_ac"    : round(m_cov - a_cov, 0),
            "milp_cost_6h"       : round(m_cost, 0),
            "actual_cost_6h"     : round(a_cost, 0),
            "cost_diff"          : round(m_cost - a_cost, 0),
            "milp_residual_dmg"  : round(m_dmg, 0),
            "actual_residual_dmg": round(a_dmg, 0),
            "dmg_reduction"      : round(a_dmg - m_dmg, 0),
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(REPORT_OUT, index=False)

    # ── Print report ──────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  COVERAGE COMPARISON  (6h planning horizon)")
    print("─" * 70)
    print(f"  {'Fire':<20} {'Risk':>6} {'Demand':>8} {'MILP%':>7} {'Actual%':>8} {'Gap(ac)':>9}")
    print(f"  {'-'*62}")
    for r in rows:
        gap = r["coverage_gap_ac"]
        gap_str = f"+{gap:,.0f}" if gap >= 0 else f"{gap:,.0f}"
        print(f"  {r['fire_name']:<20} {r['risk_score']:>6.1f} "
              f"{r['demand_6h_ac']:>8,.0f} "
              f"{r['milp_pct_covered']:>6.1f}% "
              f"{r['actual_pct_covered']:>7.1f}% "
              f"{gap_str:>9}")

    total_milp_cov   = sum(r["milp_coverage_ac"]   for r in rows)
    total_actual_cov = sum(r["actual_coverage_ac"] for r in rows)
    total_demand     = sum(r["demand_6h_ac"]        for r in rows)
    print(f"  {'TOTAL':<20} {'':>6} {total_demand:>8,.0f} "
          f"{total_milp_cov/total_demand*100:>6.1f}% "
          f"{total_actual_cov/total_demand*100:>7.1f}%")

    print("\n" + "─" * 70)
    print("  RESIDUAL DAMAGE COMPARISON  ($λ × danger × asset × uncovered × $500/ac)")
    print("─" * 70)
    print(f"  {'Fire':<20} {'MILP_dmg':>12} {'Actual_dmg':>12} {'Reduction':>12}")
    print(f"  {'-'*60}")
    for r in rows:
        print(f"  {r['fire_name']:<20} "
              f"${r['milp_residual_dmg']:>10,.0f} "
              f"${r['actual_residual_dmg']:>10,.0f} "
              f"${r['dmg_reduction']:>10,.0f}")
    total_m_dmg = sum(r["milp_residual_dmg"]   for r in rows)
    total_a_dmg = sum(r["actual_residual_dmg"] for r in rows)
    print(f"  {'TOTAL':<20} ${total_m_dmg:>10,.0f} ${total_a_dmg:>10,.0f} "
          f"${total_a_dmg - total_m_dmg:>10,.0f}")

    print("\n" + "─" * 70)
    print("  COST COMPARISON  (6h horizon suppression cost)")
    print("─" * 70)
    total_m_cost = sum(r["milp_cost_6h"]   for r in rows)
    total_a_cost = sum(r["actual_cost_6h"] for r in rows)
    print(f"  MILP total cost   : ${total_m_cost:>12,.0f}")
    print(f"  Actual total cost : ${total_a_cost:>12,.0f}")
    print(f"  Difference        : ${total_m_cost - total_a_cost:>12,.0f}")

    print("\n" + "─" * 70)
    print("  KEY FINDINGS")
    print("─" * 70)

    # Find the fire where MILP vs actual diverges most
    max_gap = max(rows, key=lambda r: abs(r["coverage_gap_ac"]))
    print(f"\n  Largest allocation divergence: {max_gap['fire_name']}")
    print(f"    MILP coverage  : {max_gap['milp_pct_covered']:.1f}% of 6h demand")
    print(f"    Actual coverage: {max_gap['actual_pct_covered']:.1f}% of 6h demand")
    print(f"    Gap            : {max_gap['coverage_gap_ac']:+,.0f} acres")

    # Fires where actual > MILP (model would have sent less)
    over_allocated = [r for r in rows if r["actual_pct_covered"] > r["milp_pct_covered"] + 5]
    if over_allocated:
        print(f"\n  Fires where actual deployment EXCEEDED MILP recommendation:")
        for r in over_allocated:
            print(f"    {r['fire_name']:<20} actual={r['actual_pct_covered']:.1f}%  "
                  f"milp={r['milp_pct_covered']:.1f}%  "
                  f"(model would have sent resources elsewhere)")

    # Fires where MILP > actual (model would have sent more)
    under_allocated = [r for r in rows if r["milp_pct_covered"] > r["actual_pct_covered"] + 5]
    if under_allocated:
        print(f"\n  Fires where MILP recommendation EXCEEDED actual deployment:")
        for r in under_allocated:
            print(f"    {r['fire_name']:<20} milp={r['milp_pct_covered']:.1f}%  "
                  f"actual={r['actual_pct_covered']:.1f}%  "
                  f"(higher risk score justified more resources)")

    print(f"\n  ✓ Full report saved → {REPORT_OUT}")

    # ── Important caveat ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  IMPORTANT CAVEAT")
    print("─" * 70)
    print("""
  This comparison shows the THEORETICAL optimum under the model's
  assumptions, not proof that real incident commanders made errors.

  Real commanders on September 8, 2020 faced constraints this model
  does not capture:
    - Political/jurisdictional obligations (state vs federal fires)
    - Crew fatigue and mandatory rest requirements (16h on / 8h off)
    - Contract availability (not all resources mobilizable at once)
    - Road closures from fire activity limiting ground access
    - Real-time uncertainty about fire behavior and weather
    - Life-safety priorities that override efficiency
    - Mutual aid agreements with fixed commitments

  The value of this comparison is NOT "the model is right and commanders
  were wrong." It is:
    - Quantifying the COST of operational constraints vs optimal
    - Identifying systematic patterns across many incidents
    - Providing a structured decision-support framework for pre-incident
      planning when constraints are less acute
""")


if __name__ == "__main__":
    run_comparison()