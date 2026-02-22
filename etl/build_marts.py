import argparse
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def log(msg: str):
    print(f"[build_marts] {msg}")


def first_csv_in_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"No CSV found inside {zip_path}")
        # Prefer the biggest CSV if multiple exist
        csvs_sorted = sorted(csvs, key=lambda n: z.getinfo(n).file_size, reverse=True)
        return csvs_sorted[0]


def read_zip_csv(zip_path: Path, usecols=None, dtype=None) -> pd.DataFrame:
    csv_name = first_csv_in_zip(zip_path)
    log(f"Reading {zip_path.name} -> {csv_name}")
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_name) as f:
            return pd.read_csv(
                f,
                usecols=usecols,
                dtype=dtype,
                low_memory=False,
                encoding_errors="ignore",
            )


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def find_join_keys(df_left: pd.DataFrame, df_right: pd.DataFrame) -> list[str]:
    """
    Prefer joining on ACK_ID plus any additional schedule identifiers if they exist on both sides.
    """
    common = set(df_left.columns).intersection(set(df_right.columns))

    # Always include ACK_ID if possible
    keys = []
    if "ACK_ID" in common:
        keys.append("ACK_ID")

    # Common schedule identifiers (if present)
    for k in ["SCH_A_PLAN_NUM", "SCH_A_LINE_NUM", "SCH_A_ITEM_NUM", "SCH_A_SEQ_NUM"]:
        if k in common:
            keys.append(k)

    # If only ACK_ID exists, that's fine
    return keys


def coerce_indicator(s: pd.Series) -> pd.Series:
    # Indicators are often '1', 1, 'Y', 'Yes'. Normalize to 0/1.
    return (
        s.fillna(0)
        .astype(str)
        .str.strip()
        .replace({"Y": "1", "YES": "1", "TRUE": "1", "T": "1"})
        .replace({"N": "0", "NO": "0", "FALSE": "0", "F": "0"})
        .apply(lambda x: 1 if x == "1" else 0)
        .astype("int8")
    )


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Core ETL
# -----------------------------
def build_marts(zip_a: Path, zip_b: Path, zip_c: Path, out_dir: Path):
    ensure_dir(out_dir)

    # --- Read Dataset A (Employer / Plan metadata)
    # Keep it light: just the columns we need
    df_a = read_zip_csv(
        zip_a,
        dtype=str,
    )
    # Normalize column names exactly as they are (Form 5500 uses uppercase)
    # We'll pick fields with fallback logic
    employer_name_col = pick_first_existing_column(
        df_a,
        ["SPONSOR_DFE_NAME", "SPONS_DFE_PN", "PLAN_NAME", "SPONSOR_NAME", "EMPLOYER_NAME"],
    )
    employer_id_col = pick_first_existing_column(df_a, ["SPONS_DFE_PN", "SPONSOR_DFE_EIN", "EIN", "ACK_ID"])

    if employer_name_col is None:
        employer_name_col = "ACK_ID"
        log("Warning: Could not find employer name column in Dataset A. Using ACK_ID as Employer label.")

    # Reduce A
    cols_a_keep = ["ACK_ID"]
    for c in [employer_name_col, employer_id_col, "PLAN_NAME"]:
        if c and c in df_a.columns and c not in cols_a_keep:
            cols_a_keep.append(c)
    df_a = df_a[cols_a_keep].copy()

    df_a["Employer"] = df_a[employer_name_col].astype(str)
    df_a["Employer_ID"] = df_a[employer_id_col].astype(str) if employer_id_col in df_a.columns else df_a["ACK_ID"].astype(str)

    # --- Read Dataset B (Schedule A: Carrier + product indicators)
    df_b = read_zip_csv(zip_b, dtype=str)

    # Indicators (you specified these)
    for ind in ["WLFR_BNFT_LIFE_INSUR_IND", "WLFR_BNFT_TEMP_DISAB_IND", "WLFR_BNFT_LONG_TERM_DISAB_IND"]:
        if ind not in df_b.columns:
            log(f"Warning: Indicator column missing in Dataset B: {ind}")

    # Carrier + lives + commissions columns (try best guess)
    carrier_name_col = pick_first_existing_column(df_b, ["INS_CARRIER_NAME", "CARRIER_NAME", "INSURANCE_CARRIER_NAME"])
    covered_lives_col = pick_first_existing_column(df_b, ["INS_PRSN_COVERED_EOY_CNT", "COVERED_LIVES_EOY", "COVERED_LIVES"])
    comm_total_col = pick_first_existing_column(df_b, ["INS_BROKER_COMM_TOT_AMT", "BROKER_COMM_TOT_AMT", "BROKER_COMMISSIONS_TOTAL"])

    # Ensure ACK_ID exists
    if "ACK_ID" not in df_b.columns:
        raise KeyError("Dataset B does not have ACK_ID. Cannot join datasets.")

    # Normalize indicators to 0/1
    life = coerce_indicator(df_b["WLFR_BNFT_LIFE_INSUR_IND"]) if "WLFR_BNFT_LIFE_INSUR_IND" in df_b.columns else pd.Series([0]*len(df_b))
    std = coerce_indicator(df_b["WLFR_BNFT_TEMP_DISAB_IND"]) if "WLFR_BNFT_TEMP_DISAB_IND" in df_b.columns else pd.Series([0]*len(df_b))
    ltd = coerce_indicator(df_b["WLFR_BNFT_LONG_TERM_DISAB_IND"]) if "WLFR_BNFT_LONG_TERM_DISAB_IND" in df_b.columns else pd.Series([0]*len(df_b))

    df_b = df_b.copy()
    df_b["IS_LIFE"] = life
    df_b["IS_STD"] = std
    df_b["IS_LTD"] = ltd

    # Keep relevant columns from B
    cols_b_keep = ["ACK_ID", "IS_LIFE", "IS_STD", "IS_LTD"]
    for c in [carrier_name_col, covered_lives_col, comm_total_col]:
        if c and c in df_b.columns:
            cols_b_keep.append(c)

    # Also keep common schedule keys if present
    for k in ["SCH_A_PLAN_NUM", "SCH_A_LINE_NUM", "SCH_A_ITEM_NUM", "SCH_A_SEQ_NUM"]:
        if k in df_b.columns:
            cols_b_keep.append(k)

    df_b = df_b[cols_b_keep].copy()

    if carrier_name_col and carrier_name_col in df_b.columns:
        df_b["Carrier"] = df_b[carrier_name_col].astype(str)
    else:
        df_b["Carrier"] = "UNKNOWN"

    if covered_lives_col and covered_lives_col in df_b.columns:
        df_b["Covered_Lives"] = safe_numeric(df_b[covered_lives_col])
    else:
        df_b["Covered_Lives"] = 0.0

    if comm_total_col and comm_total_col in df_b.columns:
        df_b["Commissions_Total_B"] = safe_numeric(df_b[comm_total_col])
    else:
        df_b["Commissions_Total_B"] = 0.0

    # Expand B into one row per product type (Life/STD/LTD) where indicator==1
    product_rows = []
    for prod_name, flag_col in [("Life", "IS_LIFE"), ("STD", "IS_STD"), ("LTD", "IS_LTD")]:
        tmp = df_b[df_b[flag_col] == 1].copy()
        tmp["Product"] = prod_name
        product_rows.append(tmp)
    df_b_long = pd.concat(product_rows, ignore_index=True) if product_rows else df_b.assign(Product="Unknown")

    # --- Read Dataset C (Schedule A Part 1: Brokers + commissions)
    df_c = read_zip_csv(zip_c, dtype=str)

    if "ACK_ID" not in df_c.columns:
        raise KeyError("Dataset C does not have ACK_ID. Cannot join datasets.")

    broker_name_col = pick_first_existing_column(df_c, ["INS_BROKER_NAME", "BROKER_NAME", "INSURANCE_BROKER_NAME"])
    broker_comm_col = pick_first_existing_column(df_c, ["INS_BROKER_COMM_PD_AMT", "BROKER_COMM_PD_AMT", "BROKER_COMMISSIONS_PAID"])

    cols_c_keep = ["ACK_ID"]
    for c in [broker_name_col, broker_comm_col]:
        if c and c in df_c.columns:
            cols_c_keep.append(c)
    for k in ["SCH_A_PLAN_NUM", "SCH_A_LINE_NUM", "SCH_A_ITEM_NUM", "SCH_A_SEQ_NUM"]:
        if k in df_c.columns:
            cols_c_keep.append(k)

    df_c = df_c[cols_c_keep].copy()

    df_c["Broker"] = df_c[broker_name_col].astype(str) if broker_name_col in df_c.columns else "UNKNOWN"
    df_c["Commissions Paid"] = safe_numeric(df_c[broker_comm_col]) if broker_comm_col in df_c.columns else 0.0

    # --- Join A + B_long
    join_keys_ab = find_join_keys(df_a, df_b_long)
    if not join_keys_ab:
        join_keys_ab = ["ACK_ID"]
    log(f"Joining A + B on keys: {join_keys_ab}")

    ab = df_b_long.merge(df_a[["ACK_ID", "Employer", "Employer_ID"]], on="ACK_ID", how="left")

    # --- Join AB + C (broker commissions)
    join_keys_bc = find_join_keys(df_b_long, df_c)
    if not join_keys_bc:
        join_keys_bc = ["ACK_ID"]
    log(f"Joining (B_long) + C on keys: {join_keys_bc}")

    # Use the most specific join keys that exist, but always at least ACK_ID
    # We'll merge product rows with broker rows. If C lacks line/plan keys, this becomes ACK_ID join.
    abc = df_b_long.merge(df_c, on=join_keys_bc, how="left")

    # Attach employer
    abc = abc.merge(df_a[["ACK_ID", "Employer", "Employer_ID"]], on="ACK_ID", how="left")

    # Final facts table
    facts = pd.DataFrame(
        {
            "ACK_ID": abc["ACK_ID"].astype(str),
            "Employer_ID": abc["Employer_ID"].astype(str),
            "Employer": abc["Employer"].astype(str),
            "Product": abc["Product"].astype(str),
            "Carrier": abc["Carrier"].astype(str),
            "Broker": abc["Broker"].astype(str),
            "Commissions Paid": safe_numeric(abc["Commissions Paid"]) if "Commissions Paid" in abc.columns else 0.0,
            "Covered_Lives": safe_numeric(abc["Covered_Lives"]) if "Covered_Lives" in abc.columns else 0.0,
        }
    )

    # Drop empty employers (if any)
    facts["Employer"] = facts["Employer"].replace({"nan": None, "None": None})
    facts = facts.dropna(subset=["Employer"])

    # Save facts mart
    facts_path = out_dir / "employer_broker_carrier.parquet"
    log(f"Writing {facts_path}")
    facts.to_parquet(facts_path, index=False)

    # --- Employer product matrix (gap table)
    # For each employer, determine if they have each product
    pivot = (
        facts.assign(flag=1)
        .pivot_table(index="Employer", columns="Product", values="flag", aggfunc="max", fill_value=0)
        .reset_index()
    )

    # Ensure columns exist
    for col in ["Life", "STD", "LTD"]:
        if col not in pivot.columns:
            pivot[col] = 0

    matrix = pivot[["Employer", "Life", "STD", "LTD"]].copy()

    matrix_path = out_dir / "employer_product_matrix.parquet"
    log(f"Writing {matrix_path}")
    matrix.to_parquet(matrix_path, index=False)

    # --- Broker summary
    broker_summary = (
        facts.groupby(["Broker", "Product"], as_index=False)
        .agg(
            unique_employers=("Employer", "nunique"),
            total_commissions=("Commissions Paid", "sum"),
            unique_carriers=("Carrier", "nunique"),
        )
        .sort_values(["Product", "total_commissions"], ascending=[True, False])
    )
    broker_path = out_dir / "broker_product_summary.parquet"
    log(f"Writing {broker_path}")
    broker_summary.to_parquet(broker_path, index=False)

    # --- Carrier summary
    carrier_summary = (
        facts.groupby(["Carrier", "Product"], as_index=False)
        .agg(
            unique_employers=("Employer", "nunique"),
            total_commissions=("Commissions Paid", "sum"),
            unique_brokers=("Broker", "nunique"),
            covered_lives=("Covered_Lives", "sum"),
        )
        .sort_values(["Product", "covered_lives"], ascending=[True, False])
    )
    carrier_path = out_dir / "carrier_product_summary.parquet"
    log(f"Writing {carrier_path}")
    carrier_summary.to_parquet(carrier_path, index=False)

    log("✅ Done. Parquet marts created in data/marts/.")


def parse_args():
    p = argparse.ArgumentParser(description="Build Parquet marts from Form 5500 Schedule A datasets.")
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