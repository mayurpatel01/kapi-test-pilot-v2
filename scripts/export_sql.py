"""
Export the cleaned marts for SQL, one plan year at a time.

The dashboard is a lens on this data, not the only way to it. If you would
rather query in SQL - or the app is struggling on a small instance - this writes
the same cleaned tables in a form any database will read.

Three formats:
  duckdb  one .duckdb file, queryable immediately, no server, keeps types
  csv     one .csv per table, loads into anything
  parquet one .parquet per table, typed and compact, read natively by
          DuckDB / Snowflake / BigQuery / Spark / pandas

What you get is the CLEANED data, not the raw filings: the plausibility caps are
applied and premium and commission are split per product so the columns are
additive. Nothing is filtered by tier, weight or opportunity score - those are
display settings in the app and have no effect here.

Usage:
    python scripts/export_sql.py --plan-year 2024
    python scripts/export_sql.py --plan-year 2024 --format csv
    python scripts/export_sql.py --all-years --format parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "marts"
sys.path.insert(0, str(REPO_ROOT / "etl"))

from quality import flag_contracts  # noqa: E402

COMM_CAP = 10_000_000.0
LIVES_CAP = 1_500_000.0
PREMIUM_CAP = 500_000_000.0


def log(msg):
    print(f"[export_sql] {msg}")


def build_tables(year: int) -> dict:
    """The same cleaning the app and the Excel export apply, kept in one place."""
    d = DATA_DIR / str(year)
    if not d.exists():
        raise FileNotFoundError(f"No marts for plan year {year} at {d}. "
                                "Run: python etl/build_all_years.py")

    epc = pd.read_parquet(d / "employer_product_carrier.parquet")
    ebc = pd.read_parquet(d / "employer_broker_commissions.parquet")
    con = pd.read_parquet(d / "employer_contract.parquet")
    dim = pd.read_parquet(d / "employer_dim.parquet")

    before = len(epc), len(ebc), len(con)

    epc = epc[epc["Covered_Lives"] <= LIVES_CAP].copy()
    epc.loc[epc["ContractCommission"] > COMM_CAP,
            ["ProductCommission", "ContractCommission"]] = 0.0
    ebc = ebc[ebc["total_commissions"] <= COMM_CAP].copy()
    con = con[con["Premium"] <= PREMIUM_CAP].copy()
    con.loc[con["Commission"] > COMM_CAP, "Commission"] = 0.0

    log(f"  caps applied: product rows {before[0]:,}->{len(epc):,}, "
        f"broker rows {before[1]:,}->{len(ebc):,}, contracts {before[2]:,}->{len(con):,}")

    # One row per employer x product - the grain most questions are asked at.
    by_contract = epc.drop_duplicates(["Employer", "Product", "ContractRowID"])
    fact = (by_contract.groupby(["EIN", "Employer", "Product", "ProductGroup"], as_index=False)
            .agg(CoveredLives=("Covered_Lives", "max"),
                 Commission=("ProductCommission", "sum"),
                 Premium=("ProductPremium", "sum"),
                 ContractPremium=("Premium", "sum"),
                 Contracts=("ContractRowID", "nunique"),
                 Carriers=("Carrier", "nunique")))
    fact["PlanYear"] = year

    flagged = flag_contracts(con)
    for t in (epc, ebc, con, dim, flagged):
        t["PlanYear"] = year

    return {
        "fact_employer_product": fact,
        "dim_employer": dim,
        "employer_product_carrier": epc,
        "employer_broker_commission": ebc,
        "employer_contract": con,
        "contract_quality_flags": flagged[
            [c for c in ["ContractRowID", "EIN", "Employer", "Carrier", "Products",
                         "Covered_Lives", "Premium", "Commission", "PremiumPerLife",
                         "QualityFlag", "AnyQualityFlag", "PlanYear"]
             if c in flagged.columns]],
    }


def write_duckdb(tables_by_year: dict, out: Path):
    try:
        import duckdb
    except ImportError:
        log("duckdb is not installed. Install it or use --format parquet/csv:")
        log("    pip install duckdb")
        raise SystemExit(1)

    if out.exists():
        out.unlink()
    con = duckdb.connect(str(out))
    for name in next(iter(tables_by_year.values())):
        frames = [t[name] for t in tables_by_year.values() if name in t]
        df = pd.concat(frames, ignore_index=True)
        con.register("_t", df)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _t')
        con.unregister("_t")
        log(f"  {name}: {len(df):,} rows")
    con.close()
    log(f"wrote {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    log("query it with:  duckdb " + out.name + "  then:  SELECT * FROM fact_employer_product LIMIT 10;")


def write_flat(tables_by_year: dict, out_dir: Path, fmt: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in next(iter(tables_by_year.values())):
        frames = [t[name] for t in tables_by_year.values() if name in t]
        df = pd.concat(frames, ignore_index=True)
        path = out_dir / f"{name}.{fmt}"
        if fmt == "csv":
            # float_format keeps money literal. Without it pandas uses repr and
            # large or tiny values export as 5.4967e+10 / 2.5e-07, which Excel
            # and most SQL loaders then read wrongly or reject.
            df.to_csv(path, index=False, float_format="%.2f")
        else:
            df.to_parquet(path, index=False)
        log(f"  {name}: {len(df):,} rows -> {path.name} "
            f"({path.stat().st_size / 1024 / 1024:.1f} MB)")


def available_years():
    return sorted(int(p.name) for p in DATA_DIR.iterdir()
                  if p.is_dir() and p.name.isdigit()
                  and (p / "employer_product_carrier.parquet").exists())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan-year", type=int, help="Single plan year to export")
    ap.add_argument("--all-years", action="store_true", help="Export every year found")
    ap.add_argument("--format", choices=["duckdb", "csv", "parquet"], default="duckdb")
    ap.add_argument("--out", default=None, help="Output file (duckdb) or directory (csv/parquet)")
    a = ap.parse_args()

    years = available_years() if a.all_years else ([a.plan_year] if a.plan_year else [])
    if not years:
        log(f"Specify --plan-year or --all-years. Available: {available_years()}")
        raise SystemExit(1)

    tables_by_year = {}
    for y in years:
        log(f"plan year {y}")
        tables_by_year[y] = build_tables(y)

    suffix = "_".join(str(y) for y in years)
    if a.format == "duckdb":
        out = Path(a.out) if a.out else REPO_ROOT / "exports" / f"kapi_{suffix}.duckdb"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_duckdb(tables_by_year, out)
    else:
        out = Path(a.out) if a.out else REPO_ROOT / "exports" / f"sql_{suffix}"
        write_flat(tables_by_year, out, a.format)
    log("done")


if __name__ == "__main__":
    main()
