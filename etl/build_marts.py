import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benefits import ALL_PRODUCTS, PRODUCT_GROUP, explode_other_text

OTHER_TEXT_COL = "WLFR_TYPE_BNFT_OTH_TEXT"


def log(msg: str):
    print(f"[build_marts] {msg}")


def first_csv_in_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"No CSV found inside {zip_path}")
        csvs_sorted = sorted(csvs, key=lambda n: z.getinfo(n).file_size, reverse=True)
        return csvs_sorted[0]


def read_zip_csv(zip_path: Path, nrows=None, usecols=None, dtype=str) -> pd.DataFrame:
    csv_name = first_csv_in_zip(zip_path)
    log(f"Reading {zip_path.name} -> {csv_name}")
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_name) as f:
            return pd.read_csv(
                f,
                nrows=nrows,
                usecols=usecols,
                dtype=dtype,
                low_memory=False,
                encoding_errors="ignore",
            )


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def coerce_indicator(s: pd.Series) -> pd.Series:
    return (
        s.fillna(0)
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"Y": "1", "YES": "1", "TRUE": "1", "T": "1"})
        .replace({"N": "0", "NO": "0", "FALSE": "0", "F": "0"})
        .apply(lambda x: 1 if x == "1" else 0)
        .astype("int8")
    )


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def build_marts(zip_a: Path, zip_b: Path, zip_c: Path, out_dir: Path):
    ensure_dir(out_dir)

    # -----------------------------
    # Dataset A: employer metadata
    # -----------------------------
    df_a = read_zip_csv(zip_a, dtype=str)

    employer_name_col = pick_first_existing_column(
        df_a,
        ["SPONSOR_DFE_NAME", "SPONSOR_NAME", "EMPLOYER_NAME", "PLAN_NAME", "SPONS_DFE_PN"],
    )
    employer_id_col = pick_first_existing_column(
        df_a,
        ["SPONS_DFE_PN", "SPONSOR_DFE_EIN", "EIN"],
    )

    if "ACK_ID" not in df_a.columns:
        raise KeyError("Dataset A missing ACK_ID")

    if employer_name_col is None:
        employer_name_col = "ACK_ID"
        log("Warning: employer name column not found in A; using ACK_ID as Employer label.")

    cols_a_keep = ["ACK_ID", employer_name_col]
    if employer_id_col and employer_id_col in df_a.columns:
        cols_a_keep.append(employer_id_col)

    df_a = df_a[cols_a_keep].copy()
    df_a["Employer"] = df_a[employer_name_col].astype(str)
    df_a["Employer_ID"] = df_a[employer_id_col].astype(str) if employer_id_col and employer_id_col in df_a.columns else df_a["ACK_ID"].astype(str)
    df_a = df_a[["ACK_ID", "Employer_ID", "Employer"]].drop_duplicates()

    # -----------------------------
    # Dataset B: carrier/product presence + covered lives
    # IMPORTANT: We do NOT attach commissions here.
    # -----------------------------
    df_b = read_zip_csv(zip_b, dtype=str)

    if "ACK_ID" not in df_b.columns:
        raise KeyError("Dataset B missing ACK_ID")

    carrier_name_col = pick_first_existing_column(df_b, ["INS_CARRIER_NAME", "CARRIER_NAME"])
    covered_lives_col = pick_first_existing_column(df_b, ["INS_PRSN_COVERED_EOY_CNT", "COVERED_LIVES", "COVERED_LIVES_EOY"])

    # Required indicators from your spec:
    ind_life = "WLFR_BNFT_LIFE_INSUR_IND"
    ind_std = "WLFR_BNFT_TEMP_DISAB_IND"
    ind_ltd = "WLFR_BNFT_LONG_TERM_DISAB_IND"

    for ind in [ind_life, ind_std, ind_ltd]:
        if ind not in df_b.columns:
            log(f"Warning: Indicator column missing in Dataset B: {ind}")

    df_b = df_b.copy()
    df_b["Carrier"] = df_b[carrier_name_col].astype(str) if carrier_name_col and carrier_name_col in df_b.columns else "UNKNOWN"
    df_b["Covered_Lives"] = safe_numeric(df_b[covered_lives_col]) if covered_lives_col and covered_lives_col in df_b.columns else 0.0

    df_b["IS_LIFE"] = coerce_indicator(df_b[ind_life]) if ind_life in df_b.columns else 0
    df_b["IS_STD"] = coerce_indicator(df_b[ind_std]) if ind_std in df_b.columns else 0
    df_b["IS_LTD"] = coerce_indicator(df_b[ind_ltd]) if ind_ltd in df_b.columns else 0

    # Expand into long rows by product -- checkbox products first
    parts = []
    for prod, flag in [("Life", "IS_LIFE"), ("STD", "IS_STD"), ("LTD", "IS_LTD")]:
        tmp = df_b[df_b[flag] == 1][["ACK_ID", "Carrier", "Covered_Lives"]].copy()
        tmp["Product"] = prod
        parts.append(tmp)

    # ...then the voluntary products, which have no checkbox and must be parsed
    # out of the OTHER free text. See etl/benefits.py for why AD&D is separated.
    if OTHER_TEXT_COL in df_b.columns:
        vb = explode_other_text(
            df_b,
            text_col=OTHER_TEXT_COL,
            keep_cols=["ACK_ID", "Carrier", "Covered_Lives"],
        )
        log(f"Parsed {len(vb):,} product rows from OTHER free text "
            f"({vb['ACK_ID'].nunique():,} filings, {vb['Product'].nunique()} distinct products)")
        parts.append(vb)
    else:
        log(f"Warning: {OTHER_TEXT_COL} missing -- voluntary products will be absent")

    b_long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ACK_ID", "Carrier", "Covered_Lives", "Product"])
    b_long = b_long.drop_duplicates()
    b_long["ProductGroup"] = b_long["Product"].map(PRODUCT_GROUP).fillna("Other")

    for grp in ["Core", "Voluntary", "Adjacent"]:
        sel = b_long[b_long["ProductGroup"] == grp]
        log(f"  {grp:<10} {len(sel):>8,} rows  {sel['ACK_ID'].nunique():>7,} filings")

    # Attach employer labels
    employer_product_carrier = b_long.merge(df_a, on="ACK_ID", how="left")
    employer_product_carrier["Employer"] = employer_product_carrier["Employer"].fillna(employer_product_carrier["ACK_ID"].astype(str))

    # Save mart: employer_product_carrier
    out_epc = out_dir / "employer_product_carrier.parquet"
    log(f"Writing {out_epc}")
    employer_product_carrier.to_parquet(out_epc, index=False)

    # -----------------------------
    # Employer product matrix (gap table) from B only
    # -----------------------------
    matrix = (
        employer_product_carrier.assign(flag=1)
        .pivot_table(index="Employer", columns="Product", values="flag", aggfunc="max", fill_value=0)
        .reset_index()
    )
    # Keep every known product as a column even when nothing matched, so the
    # downstream schema is stable regardless of what a given filing year contains.
    for col in ALL_PRODUCTS:
        if col not in matrix.columns:
            matrix[col] = 0
    matrix = matrix[["Employer"] + ALL_PRODUCTS]

    out_matrix = out_dir / "employer_product_matrix.parquet"
    log(f"Writing {out_matrix}")
    matrix.to_parquet(out_matrix, index=False)

    # -----------------------------
    # Dataset C: broker commissions (NO carrier/product join)
    # -----------------------------
    df_c = read_zip_csv(zip_c, dtype=str)
    if "ACK_ID" not in df_c.columns:
        raise KeyError("Dataset C missing ACK_ID")

    if "INS_BROKER_NAME" not in df_c.columns:
        raise KeyError("Dataset C missing INS_BROKER_NAME (unexpected based on your sample)")

    if "INS_BROKER_COMM_PD_AMT" not in df_c.columns:
        raise KeyError("Dataset C missing INS_BROKER_COMM_PD_AMT (unexpected based on your sample)")

    broker_comm = df_c[["ACK_ID", "INS_BROKER_NAME", "INS_BROKER_COMM_PD_AMT"]].copy()
    broker_comm["Broker"] = broker_comm["INS_BROKER_NAME"].astype(str)
    broker_comm["Commissions Paid"] = safe_numeric(broker_comm["INS_BROKER_COMM_PD_AMT"])

    # Aggregate safely: employer/broker level
    broker_comm_agg = (
        broker_comm.groupby(["ACK_ID", "Broker"], as_index=False)
        .agg(total_commissions=("Commissions Paid", "sum"))
    )

    # Attach employer
    employer_broker_commissions = broker_comm_agg.merge(df_a, on="ACK_ID", how="left")
    employer_broker_commissions["Employer"] = employer_broker_commissions["Employer"].fillna(employer_broker_commissions["ACK_ID"].astype(str))

    out_ebc = out_dir / "employer_broker_commissions.parquet"
    log(f"Writing {out_ebc}")
    employer_broker_commissions.to_parquet(out_ebc, index=False)

    # -----------------------------
    # Summaries for fast UI + AI
    # -----------------------------
    broker_product_summary = (
        employer_product_carrier.groupby(["Product"], as_index=False)
        .agg(unique_employers=("Employer", "nunique"))
    )

    # Broker summary (commissions)
    broker_summary = (
        employer_broker_commissions.groupby(["Broker"], as_index=False)
        .agg(
            unique_employers=("Employer", "nunique"),
            total_commissions=("total_commissions", "sum"),
        )
        .sort_values("total_commissions", ascending=False)
    )
    out_broker_sum = out_dir / "broker_summary.parquet"
    log(f"Writing {out_broker_sum}")
    broker_summary.to_parquet(out_broker_sum, index=False)

    # Carrier summary (presence + lives)
    carrier_summary = (
        employer_product_carrier.groupby(["Carrier", "Product", "ProductGroup"], as_index=False)
        .agg(
            unique_employers=("Employer", "nunique"),
            covered_lives=("Covered_Lives", "sum"),
        )
        .sort_values(["ProductGroup", "Product", "covered_lives"], ascending=[True, True, False])
    )
    out_carrier_sum = out_dir / "carrier_product_summary.parquet"
    log(f"Writing {out_carrier_sum}")
    carrier_summary.to_parquet(out_carrier_sum, index=False)

    log("✅ Done. New marts created (commissions separated to avoid overcount).")


def parse_args():
    p = argparse.ArgumentParser(description="Build Parquet marts from Form 5500 Schedule A datasets (safe pilot version).")
    p.add_argument("--zip_a", required=True, help="Path to Dataset A ZIP (F_5500_2024_Latest.zip)")
    p.add_argument("--zip_b", required=True, help="Path to Dataset B ZIP (F_SCH_A_2024_Latest.zip)")
    p.add_argument("--zip_c", required=True, help="Path to Dataset C ZIP (F_SCH_A_PART1_2024_Latest.zip)")
    p.add_argument("--out_dir", default="data/marts", help="Output directory for Parquet marts (default: data/marts)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    zip_a = Path(args.zip_a).expanduser().resolve()
    zip_b = Path(args.zip_b).expanduser().resolve()
    zip_c = Path(args.zip_c).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    for z in [zip_a, zip_b, zip_c]:
        if not z.exists():
            log(f"ERROR: File not found: {z}")
            sys.exit(1)

    build_marts(zip_a, zip_b, zip_c, out_dir)