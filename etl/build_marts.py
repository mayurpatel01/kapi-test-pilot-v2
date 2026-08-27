import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benefits import ALL_PRODUCTS, PRODUCT_GROUP, explode_other_text

OTHER_TEXT_COL = "WLFR_TYPE_BNFT_OTH_TEXT"
PREMIUM_COL = "WLFR_PREMIUM_RCVD_AMT"
CHARGES_COL = "WLFR_TOT_CHARGES_PAID_AMT"
RET_COMM_COL = "WLFR_RET_COMMISSIONS_AMT"


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


def build_marts(zip_a: Path, zip_b: Path, zip_c: Path, out_dir: Path, plan_year: int | None = None):
    ensure_dir(out_dir)

    # -----------------------------
    # Dataset A: employer metadata
    # -----------------------------
    df_a = read_zip_csv(zip_a, dtype=str)

    if "ACK_ID" not in df_a.columns:
        raise KeyError("Dataset A missing ACK_ID")

    employer_name_col = pick_first_existing_column(
        df_a, ["SPONSOR_DFE_NAME", "SPONSOR_NAME", "EMPLOYER_NAME", "PLAN_NAME"])
    if employer_name_col is None:
        employer_name_col = "ACK_ID"
        log("Warning: employer name column not found in A; using ACK_ID as Employer label.")

    # ---- Employer identity is EIN, not the filed name.
    #
    # A sponsor name is neither stable nor unique. Measured on 2023 vs 2024,
    # 7,291 companies (5.9% of those filing both years) changed their filed name
    # -- "TERUMO BCT" to "TERUMO BLOOD AND CELL TECHNOLOGIES INC" -- so a
    # name-keyed trend reads each of those as a lost account. In the other
    # direction 1,719 names map to more than one EIN inside a single year, so
    # name-keying silently merges distinct companies even without any trend work.
    # SPONS_DFE_EIN is populated on 100% of filings in every year checked.
    ein_col = pick_first_existing_column(df_a, ["SPONS_DFE_EIN", "SPONSOR_DFE_EIN", "EIN"])
    if ein_col is None:
        raise KeyError("Dataset A has no EIN column - cannot key employers reliably")

    geo_cols = {
        "State": pick_first_existing_column(df_a, ["SPONS_DFE_MAIL_US_STATE", "SPONS_DFE_LOC_US_STATE"]),
        "City": pick_first_existing_column(df_a, ["SPONS_DFE_MAIL_US_CITY", "SPONS_DFE_LOC_US_CITY"]),
        "ZIP": pick_first_existing_column(df_a, ["SPONS_DFE_MAIL_US_ZIP", "SPONS_DFE_LOC_US_ZIP"]),
    }
    py_col = pick_first_existing_column(df_a, ["FORM_PLAN_YEAR_BEGIN_DATE", "FORM_TAX_PRD"])

    keep = ["ACK_ID", employer_name_col, ein_col] + [c for c in geo_cols.values() if c]
    if py_col:
        keep.append(py_col)
    df_a = df_a[[c for c in dict.fromkeys(keep)]].copy()

    df_a["EIN"] = (
        df_a[ein_col].fillna("").astype(str).str.replace(r"\D", "", regex=True)
        .str.zfill(9).str[-9:]
    )
    df_a["NameAsFiled"] = df_a[employer_name_col].fillna("").astype(str).str.strip()
    bad_ein = df_a["EIN"].isin({"000000000", ""})
    if bad_ein.any():
        log(f"Warning: {int(bad_ein.sum()):,} filings have no usable EIN - keyed on name instead")
    df_a.loc[bad_ein, "EIN"] = "NAME:" + df_a.loc[bad_ein, "NameAsFiled"]

    if py_col:
        df_a["PlanYear"] = pd.to_datetime(df_a[py_col], errors="coerce").dt.year
        dom = df_a["PlanYear"].value_counts()
        if len(dom):
            log(f"plan year: {int(dom.idxmax())} on {dom.max() / len(df_a) * 100:.1f}% of filings")
    else:
        df_a["PlanYear"] = pd.NA

    # A DOL release is ~96% the plan year it is named for, with a tail of late and
    # amended filings for earlier years reaching back to the 1980s. Those stragglers
    # also appear in their OWN release, so loading several years without filtering
    # would double-count them. Restricting each release to its own plan year gives
    # clean, non-overlapping slices.
    if plan_year is not None:
        before = len(df_a)
        df_a = df_a[df_a["PlanYear"] == plan_year].copy()
        log(f"filtered to plan year {plan_year}: {len(df_a):,} of {before:,} filings "
            f"({len(df_a) / before * 100:.1f}%) - stragglers from other plan years dropped")
        if not len(df_a):
            raise ValueError(f"No filings for plan year {plan_year} in this release")

    # Canonical display name per EIN: the name that company filed most often,
    # longest name breaking ties so "TERUMO BLOOD AND CELL TECHNOLOGIES INC" wins
    # over the abbreviation rather than by accident of ordering.
    name_counts = (
        df_a[df_a["NameAsFiled"] != ""]
        .groupby(["EIN", "NameAsFiled"]).size().rename("n").reset_index()
    )
    name_counts["_len"] = name_counts["NameAsFiled"].str.len()
    canonical = (
        name_counts.sort_values(["EIN", "n", "_len"], ascending=[True, False, False])
                   .drop_duplicates("EIN").set_index("EIN")["NameAsFiled"]
    )

    # A canonical name shared by several EINs would still collide under any
    # groupby on the label, so those get the EIN appended and stay distinct.
    shared = canonical.value_counts()
    shared = set(shared[shared > 1].index)
    disambiguated = canonical.copy()
    mask = canonical.isin(shared)
    disambiguated[mask] = canonical[mask] + " [EIN " + canonical[mask].index.to_series() + "]"
    if mask.any():
        log(f"disambiguated {int(mask.sum()):,} employer names shared by multiple EINs")

    df_a["Employer"] = df_a["EIN"].map(disambiguated).fillna(df_a["NameAsFiled"])
    df_a["Employer_ID"] = df_a["EIN"]

    for out_name, src in geo_cols.items():
        df_a[out_name] = df_a[src].fillna("").astype(str).str.strip() if src else ""
    df_a["State"] = df_a["State"].str.upper()
    df_a["ZIP"] = df_a["ZIP"].str.replace(r"\D", "", regex=True).str[:5]

    # Employer dimension, one row per EIN. Geo comes from the filing itself, so no
    # name matching is involved -- the previous build inferred it by stripping
    # INC/LLC/CORP from names, which merged distinct companies.
    employer_dim = (
        df_a.sort_values("ACK_ID")
            .groupby("EIN", as_index=False)
            .agg(Employer=("Employer", "first"), NameAsFiled=("NameAsFiled", "last"),
                 State=("State", "first"), City=("City", "first"), ZIP=("ZIP", "first"),
                 PlanYear=("PlanYear", "first"), Filings=("ACK_ID", "nunique"))
    )
    log(f"employer dimension: {len(employer_dim):,} distinct EINs "
        f"from {df_a['ACK_ID'].nunique():,} filings")

    out_dim = out_dir / "employer_dim.parquet"
    log(f"Writing {out_dim}")
    employer_dim.to_parquet(out_dim, index=False)

    # employer_geo keeps its old shape so the dashboard's optional load still
    # works, but is now derived from the filing rather than name-matched.
    out_geo = out_dir / "employer_geo.parquet"
    log(f"Writing {out_geo} (state on "
        f"{(employer_dim['State'].fillna('') != '').mean() * 100:.1f}% of employers)")
    employer_dim[["Employer", "State", "ZIP", "City", "EIN"]].to_parquet(out_geo, index=False)

    df_a = df_a[["ACK_ID", "EIN", "Employer_ID", "Employer", "PlanYear"]].drop_duplicates()

    # -----------------------------
    # Dataset B: carrier/product presence, covered lives, premium, commission
    #
    # Commission DOES attach here, at contract grain. Schedule A Part 1 (dataset
    # C) carries FORM_ID alongside ACK_ID, and (ACK_ID, FORM_ID) is unique per
    # Schedule A row -- so every broker commission row resolves to exactly one
    # contract, and therefore to that contract's products. Verified against the
    # 2024 file: 380,007 of 380,007 Part 1 rows join, 100%.
    #
    # The employer/broker rollup in dataset C below is kept as well, because it
    # is the grain the broker-league tables need and it reconciles against this.
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

    # ---- Premium
    # Schedule A reports the money two ways and a contract fills in one or the
    # other. WLFR_PREMIUM_RCVD_AMT is the experience-rated section and is only
    # ~7% populated; WLFR_TOT_CHARGES_PAID_AMT ("total charges paid for this
    # contract") is ~78% populated and is the premium equivalent for everyone
    # else. Coalescing prefers the explicit premium and falls back to charges,
    # which covers ~84% of Schedule A rows.
    #
    # Left UNCAPPED here on purpose: the marts stay faithful to what was filed
    # and the export applies caps, so every excluded row keeps an audit trail on
    # Data_Quality_Flags rather than vanishing at build time.
    prem_rcvd = safe_numeric(df_b[PREMIUM_COL]) if PREMIUM_COL in df_b.columns else 0.0
    charges = safe_numeric(df_b[CHARGES_COL]) if CHARGES_COL in df_b.columns else 0.0
    df_b["Premium"] = prem_rcvd.where(prem_rcvd > 0, charges) if PREMIUM_COL in df_b.columns else charges
    df_b["PremiumSource"] = "none"
    df_b.loc[charges > 0, "PremiumSource"] = "total charges paid"
    if PREMIUM_COL in df_b.columns:
        df_b.loc[prem_rcvd > 0, "PremiumSource"] = "premium received (experience-rated)"
    df_b["RetainedCommission"] = (
        safe_numeric(df_b[RET_COMM_COL]) if RET_COMM_COL in df_b.columns else 0.0
    )

    log(f"premium populated on {int((df_b['Premium'] > 0).sum()):,} of {len(df_b):,} Schedule A rows "
        f"({(df_b['Premium'] > 0).mean() * 100:.1f}%)")

    # ---- Commission at CONTRACT grain, via the (ACK_ID, FORM_ID) join.
    df_c_early = read_zip_csv(zip_c, dtype=str)
    if "FORM_ID" in df_c_early.columns and "FORM_ID" in df_b.columns:
        df_c_early["_comm"] = safe_numeric(df_c_early["INS_BROKER_COMM_PD_AMT"])
        contract_comm = (
            df_c_early.groupby(["ACK_ID", "FORM_ID"], as_index=False)
                      .agg(ContractCommission=("_comm", "sum"),
                           ContractBrokers=("INS_BROKER_NAME", "nunique"))
        )
        before = len(df_b)
        df_b = df_b.merge(contract_comm, on=["ACK_ID", "FORM_ID"], how="left")
        assert len(df_b) == before, "contract commission join changed row count"
        df_b["ContractCommission"] = df_b["ContractCommission"].fillna(0.0)
        df_b["ContractBrokers"] = df_b["ContractBrokers"].fillna(0).astype(int)

        joined = df_c_early.merge(df_b[["ACK_ID", "FORM_ID"]].drop_duplicates(),
                                  on=["ACK_ID", "FORM_ID"], how="inner")
        log(f"commission joined to contracts: {len(joined):,} of {len(df_c_early):,} Part 1 rows "
            f"({len(joined) / len(df_c_early) * 100:.2f}%)")
        log(f"  contracts carrying commission: {int((df_b['ContractCommission'] > 0).sum()):,} "
            f"of {len(df_b):,}")
    else:
        log("Warning: FORM_ID missing - commission cannot be tied to products, "
            "falling back to employer grain only")
        df_b["ContractCommission"] = 0.0
        df_b["ContractBrokers"] = 0

    df_b["IS_LIFE"] = coerce_indicator(df_b[ind_life]) if ind_life in df_b.columns else 0
    df_b["IS_STD"] = coerce_indicator(df_b[ind_std]) if ind_std in df_b.columns else 0
    df_b["IS_LTD"] = coerce_indicator(df_b[ind_ltd]) if ind_ltd in df_b.columns else 0

    # One Schedule A row can list several benefits, so after the explosion below
    # a single contract appears once per product. ContractRowID identifies the
    # source row so money can be summed over DISTINCT contracts instead of over
    # exploded product rows -- without it, premium double-counts exactly the way
    # covered lives would.
    df_b = df_b.reset_index(drop=True)
    df_b["ContractRowID"] = df_b.index.astype("int64")

    CARRY = ["ACK_ID", "ContractRowID", "Carrier", "Covered_Lives", "Premium",
             "PremiumSource", "RetainedCommission", "ContractCommission", "ContractBrokers"]

    # Expand into long rows by product -- checkbox products first
    parts = []
    for prod, flag in [("Life", "IS_LIFE"), ("STD", "IS_STD"), ("LTD", "IS_LTD")]:
        tmp = df_b[df_b[flag] == 1][CARRY].copy()
        tmp["Product"] = prod
        parts.append(tmp)

    # ...then the voluntary products, which have no checkbox and must be parsed
    # out of the OTHER free text. See etl/benefits.py for why AD&D is separated.
    if OTHER_TEXT_COL in df_b.columns:
        vb = explode_other_text(
            df_b,
            text_col=OTHER_TEXT_COL,
            keep_cols=CARRY,
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

    # ---- Commission per product.
    # A contract reports ONE commission covering whatever benefits it lists. Where
    # it lists a single product that figure is exactly that product's commission,
    # no estimation involved. Where it lists several, the commission is split
    # evenly across them -- Schedule A gives no per-benefit breakdown inside a
    # contract, and premium is likewise reported once for the whole contract, so
    # there is nothing to weight by. ProductsOnContract is kept alongside so the
    # exact rows can always be separated from the split ones.
    b_long["ProductsOnContract"] = b_long.groupby("ContractRowID")["Product"].transform("nunique")
    b_long["ProductCommission"] = b_long["ContractCommission"] / b_long["ProductsOnContract"]
    b_long["CommissionIsExact"] = b_long["ProductsOnContract"] == 1

    _exact = b_long[b_long["CommissionIsExact"] & (b_long["ProductCommission"] > 0)]
    _split = b_long[(~b_long["CommissionIsExact"]) & (b_long["ProductCommission"] > 0)]
    log(f"  commission exact  (single-product contracts): {len(_exact):>7,} rows  "
        f"${_exact['ProductCommission'].sum():,.0f}")
    log(f"  commission split  (multi-product contracts) : {len(_split):>7,} rows  "
        f"${_split['ProductCommission'].sum():,.0f}")

    # Attach employer labels
    employer_product_carrier = b_long.merge(df_a, on="ACK_ID", how="left")
    employer_product_carrier["Employer"] = employer_product_carrier["Employer"].fillna(employer_product_carrier["ACK_ID"].astype(str))

    # Save mart: employer_product_carrier
    out_epc = out_dir / "employer_product_carrier.parquet"
    log(f"Writing {out_epc}")
    employer_product_carrier.to_parquet(out_epc, index=False)

    # -----------------------------
    # Contract-level mart: ONE row per Schedule A contract, pre-explosion.
    # Employer premium must be summed from here, never from the product-exploded
    # table, or a contract covering three benefits is counted three times.
    # -----------------------------
    contracts = (
        employer_product_carrier
        .sort_values("Product")
        .groupby("ContractRowID", as_index=False)
        .agg(
            ACK_ID=("ACK_ID", "first"),
            Employer=("Employer", "first"),
            Carrier=("Carrier", "first"),
            Covered_Lives=("Covered_Lives", "first"),
            Premium=("Premium", "first"),
            PremiumSource=("PremiumSource", "first"),
            RetainedCommission=("RetainedCommission", "first"),
            Commission=("ContractCommission", "first"),
            Brokers=("ContractBrokers", "first"),
            Products=("Product", lambda s: " + ".join(sorted(set(s)))),
            ProductCount=("Product", "nunique"),
        )
    )
    out_contracts = out_dir / "employer_contract.parquet"
    log(f"Writing {out_contracts} ({len(contracts):,} contracts, "
        f"${contracts['Premium'].sum():,.0f} premium as filed)")
    contracts.to_parquet(out_contracts, index=False)

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
    # Already read above for the contract-grain join; reuse rather than re-parse.
    df_c = df_c_early
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
            premium=("Premium", "sum"),
            commission=("ProductCommission", "sum"),
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
    p.add_argument("--plan-year", type=int, default=None,
                   help="Restrict to this plan year. A DOL release is ~96%% the year it is named for; "
                        "without this the late/amended tail from other years is kept, which "
                        "double-counts when several releases are loaded together.")
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

    build_marts(zip_a, zip_b, zip_c, out_dir, plan_year=args.plan_year)