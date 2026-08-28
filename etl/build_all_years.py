"""
Build marts for every plan year we hold, one partition per year.

Each DOL release goes to data/marts/<plan_year>/ rather than a single directory,
so the dashboard loads only the year being viewed. That keeps memory bounded as
years accumulate -- the alternative, one combined mart, grows without limit and
Streamlit Cloud's free tier is already tight on a single year.

A small cross-year rollup (trend_summary.parquet) is written at the top level so
trend views never need to open more than one full year.

Usage:
    python etl/build_all_years.py                     # every year found under data/raw
    python etl/build_all_years.py --years 2023 2024
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_marts import build_marts, log  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
MARTS = REPO_ROOT / "data" / "marts"


def zips_for(year: int):
    """Locate a year's three source zips. 2024 sits at the raw root for historical
    reasons; later downloads are foldered by year."""
    for base in (RAW / str(year), RAW):
        trio = [base / f"F_5500_{year}_Latest.zip",
                base / f"F_SCH_A_{year}_Latest.zip",
                base / f"F_SCH_A_PART1_{year}_Latest.zip"]
        if all(p.exists() for p in trio):
            return trio
    return None


def discover_years():
    found = set()
    for p in RAW.rglob("F_5500_*_Latest.zip"):
        try:
            found.add(int(p.stem.split("_")[2]))
        except (IndexError, ValueError):
            continue
    return sorted(y for y in found if zips_for(y))


def build_trend_summary(years):
    """Cross-year rollup, small enough to load whole however many years accrue."""
    frames = []
    for y in years:
        epc_path = MARTS / str(y) / "employer_product_carrier.parquet"
        if not epc_path.exists():
            continue
        epc = pd.read_parquet(epc_path)
        # Same caps the export applies, so trend lines match the workbook.
        epc = epc[epc["Covered_Lives"] <= 1_500_000]
        epc.loc[epc["ContractCommission"] > 10_000_000, "ProductCommission"] = 0.0
        by_contract = epc.drop_duplicates(["Employer", "Product", "ContractRowID"])
        # ProductPremium, not Premium: the raw column repeats the whole contract's
        # premium on each of its product rows, so summing it inflates ~2.4x.
        g = (by_contract.groupby(["Product", "ProductGroup"], as_index=False)
             .agg(Employers=("EIN", "nunique"),
                  Commission=("ProductCommission", "sum"),
                  Premium=("ProductPremium", "sum")))
        lives = (epc.groupby(["Product", "ProductGroup", "EIN"])["Covered_Lives"].max()
                 .groupby(level=[0, 1]).sum().rename("CoveredLives").reset_index())
        g = g.merge(lives, on=["Product", "ProductGroup"], how="left")
        g["PlanYear"] = y
        g["TotalEmployers"] = epc["EIN"].nunique()
        frames.append(g)
        log(f"trend: plan year {y} summarised ({len(g)} products, "
            f"{epc['EIN'].nunique():,} employers)")

    if not frames:
        log("no years available for trend summary")
        return
    out = pd.concat(frames, ignore_index=True)
    out["Penetration%"] = out["Employers"] / out["TotalEmployers"] * 100.0
    out["CommissionRate%"] = out["Commission"] / out["Premium"].replace(0, pd.NA) * 100.0
    path = MARTS / "trend_summary.parquet"
    out.to_parquet(path, index=False)
    log(f"Writing {path} ({len(out):,} rows across {out['PlanYear'].nunique()} years)")


def main():
    ap = argparse.ArgumentParser(description="Build per-plan-year marts and a trend summary.")
    ap.add_argument("--years", nargs="*", type=int, default=None,
                    help="Plan years to build (default: every year found under data/raw)")
    ap.add_argument("--skip-build", action="store_true",
                    help="Only rebuild trend_summary from marts already on disk")
    a = ap.parse_args()

    years = a.years or discover_years()
    if not years:
        log(f"No source zips found under {RAW}. Expected F_5500_<year>_Latest.zip and friends.")
        return
    log(f"plan years to build: {years}")

    if not a.skip_build:
        for y in years:
            trio = zips_for(y)
            if not trio:
                log(f"SKIP {y}: source zips not found")
                continue
            out = MARTS / str(y)
            log(f"===== plan year {y} -> {out} =====")
            build_marts(trio[0], trio[1], trio[2], out, plan_year=y)

    build_trend_summary(years)
    log("done")


if __name__ == "__main__":
    main()
