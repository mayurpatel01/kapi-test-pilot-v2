"""
Export the cleaned pilot dataset to a shareable Excel workbook.

Reproduces the same cleanup the dashboard applies (app.py):
  - broker company-name matching  -> AON composite + Tier1 majors + tiering
  - state normalization           -> drops bogus "55 states"
  - employer-level rollups        -> lives = MAX (no double count), commissions = SUM
  - primary broker                -> broker with the largest commissions per employer

Usage:
    python scripts/export_dataset.py
    python scripts/export_dataset.py --out exports/kapi_pilot.xlsx --with-detail

NOTE: the name-matching constants below intentionally mirror app.py so the export
matches what the dashboard shows. If you change them in one place, change both.
"""

import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "marts"


def log(msg: str):
    print(f"[export] {msg}")


# =========================================================
# Cleaning helpers (mirrors app.py)
# =========================================================
def norm(s) -> str:
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def as_flag(series: pd.Series) -> pd.Series:
    s = series.fillna(0)
    if s.dtype == "object":
        s = s.astype(str).str.strip()
        return s.isin(["1", "TRUE", "True", "true", "Y", "YES", "Yes", "yes"])
    return s.astype(int) == 1


VALID_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME",
    "MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI",
    "SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","PR"
}

STATE_NAME_TO_ABBR = {
    "ALABAMA":"AL","ALASKA":"AK","ARIZONA":"AZ","ARKANSAS":"AR","CALIFORNIA":"CA","COLORADO":"CO","CONNECTICUT":"CT",
    "DELAWARE":"DE",
    "DISTRICT OF COLUMBIA":"DC","WASHINGTON DC":"DC","WASHINGTON, DC":"DC",
    "FLORIDA":"FL","GEORGIA":"GA","HAWAII":"HI","IDAHO":"ID","ILLINOIS":"IL","INDIANA":"IN","IOWA":"IA","KANSAS":"KS",
    "KENTUCKY":"KY","LOUISIANA":"LA","MAINE":"ME","MARYLAND":"MD","MASSACHUSETTS":"MA","MICHIGAN":"MI","MINNESOTA":"MN",
    "MISSISSIPPI":"MS","MISSOURI":"MO","MONTANA":"MT","NEBRASKA":"NE","NEVADA":"NV","NEW HAMPSHIRE":"NH","NEW JERSEY":"NJ",
    "NEW MEXICO":"NM","NEW YORK":"NY","NORTH CAROLINA":"NC","NORTH DAKOTA":"ND","OHIO":"OH","OKLAHOMA":"OK","OREGON":"OR",
    "PENNSYLVANIA":"PA","RHODE ISLAND":"RI","SOUTH CAROLINA":"SC","SOUTH DAKOTA":"SD","TENNESSEE":"TN","TEXAS":"TX",
    "UTAH":"UT","VERMONT":"VT","VIRGINIA":"VA","WASHINGTON":"WA","WEST VIRGINIA":"WV","WISCONSIN":"WI","WYOMING":"WY",
    "PUERTO RICO":"PR"
}


def normalize_state(x) -> str:
    s = "" if x is None else str(x).strip().upper()
    s = s.replace(".", "")
    if s in VALID_STATES:
        return s
    return STATE_NAME_TO_ABBR.get(s, s)


# ---- Broker company-name matching ----
AON_ROOTS = [
    "AON CORPORATION",
    "AON RISK SERVICES",
    "AON HEWITT",
    "AON CONSULTING",
    "AON SOLUTIONS",
]

TIER1_PATTERNS = {
    "MARSH": [r"\bMARSH\b", r"\bMARSH\s+MCLENNAN\b", r"\bMMC\b"],
    "WTW": [r"\bWILLIS\b", r"\bTOWERS\b", r"\bWTW\b", r"\bWILLIS\s+TOWERS\s+WATSON\b"],
    "GALLAGHER": [r"\bGALLAGHER\b", r"\bARTHUR\s+J\s+GALLAGHER\b"],
    "BROWN & BROWN": [r"\bBROWN\b.*\bBROWN\b", r"\bBROWN\s*&\s*BROWN\b"],
}


def is_aon_composite(broker_name: str) -> bool:
    n = norm(broker_name)
    for root in AON_ROOTS:
        r = norm(root)
        # Must start with root to avoid DATAONLINE false positives
        if re.match(rf"^{re.escape(r)}(\s+.*)?$", n):
            return True
    return False


def match_any(patterns, text) -> bool:
    return any(re.search(p, text) for p in patterns)


def broker_family(broker_name: str) -> str:
    n = norm(broker_name)
    if not n:
        return "UNKNOWN"
    if is_aon_composite(n):
        return "AON"
    for fam, pats in TIER1_PATTERNS.items():
        if match_any(pats, n):
            return fam
    return "OTHER"


def assign_tiers(broker_agg: pd.DataFrame, tier2_pct: float = 0.10) -> pd.DataFrame:
    """Tier0 = AON, Tier1 = named majors, Tier2 = top X% of 'Other' by lives, else Tier3."""
    out = broker_agg.copy()
    out["Tier"] = "Tier3"
    out.loc[out["BrokerFamily"] == "AON", "Tier"] = "Tier0"
    out.loc[out["BrokerFamily"].isin(list(TIER1_PATTERNS.keys())), "Tier"] = "Tier1"

    mask_other = out["BrokerFamily"].eq("OTHER")
    other = out[mask_other]
    if len(other) > 0:
        thresh = other["CoveredLives"].quantile(1 - tier2_pct)
        out.loc[mask_other & (out["CoveredLives"] >= thresh), "Tier"] = "Tier2"
        out.loc[mask_other & (out["CoveredLives"] < thresh), "Tier"] = "Tier3"
    return out


TIER_LABEL = {
    "Tier0": "Tier0 - AON (incumbent)",
    "Tier1": "Tier1 - Global major",
    "Tier2": "Tier2 - Large regional/other",
    "Tier3": "Tier3 - Small/local",
}


# =========================================================
# Build the cleaned dataset
# =========================================================
def load_marts():
    def rd(name):
        p = DATA_DIR / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing mart: {p}")
        log(f"loading {p.name}")
        return pd.read_parquet(p)

    epc = rd("employer_product_carrier")
    ebc = rd("employer_broker_commissions")
    gaps = rd("employer_product_matrix")
    try:
        geo = rd("employer_geo")
    except FileNotFoundError:
        log("WARNING: employer_geo.parquet missing - State/City/ZIP/EIN will be blank")
        geo = pd.DataFrame(columns=["Employer", "State", "City", "ZIP", "EIN"])
    return epc, ebc, gaps, geo


def build(tier2_pct: float = 0.10, comm_cap: float = 10_000_000.0, lives_cap: float = 1_500_000.0):
    epc, ebc, gaps, geo = load_marts()

    epc["Covered_Lives"] = to_numeric(epc["Covered_Lives"]).fillna(0)
    ebc["total_commissions"] = to_numeric(ebc["total_commissions"]).fillna(0)

    # ---- Data-quality guard -------------------------------------------------
    # Form 5500 is self-reported and a handful of filings carry obvious keying
    # errors (e.g. a single row reporting $563 TRILLION in commissions, or a
    # small employer reporting 6M covered lives). Left in, one row swamps every
    # sum in the workbook. Rows above the caps are removed from all rollups and
    # listed in full on the Data_Quality_Flags sheet - nothing is dropped silently.
    comm_raw_total = float(ebc["total_commissions"].sum())
    lives_raw_max = float(epc["Covered_Lives"].max()) if len(epc) else 0.0

    bad_comm = ebc[ebc["total_commissions"] > comm_cap].copy()
    bad_lives = epc[epc["Covered_Lives"] > lives_cap].copy()

    flags = pd.concat([
        bad_comm.assign(
            Field="Commissions paid",
            Counterparty=bad_comm["Broker"],
            ReportedValue=bad_comm["total_commissions"],
            Reason=lambda d: f"Exceeds plausibility cap of ${comm_cap:,.0f} for a single employer-broker filing",
        )[["Employer", "Field", "Counterparty", "ACK_ID", "ReportedValue", "Reason"]],
        bad_lives.assign(
            Field="Covered lives",
            Counterparty=bad_lives["Carrier"] + " (" + bad_lives["Product"] + ")",
            ReportedValue=bad_lives["Covered_Lives"],
            Reason=lambda d: f"Exceeds plausibility cap of {lives_cap:,.0f} lives for a single employer-carrier row",
        )[["Employer", "Field", "Counterparty", "ACK_ID", "ReportedValue", "Reason"]],
    ], ignore_index=True).rename(columns={"ACK_ID": "FilingID"}).sort_values("ReportedValue", ascending=False)

    if len(flags):
        log(f"data-quality: excluding {len(bad_comm)} commission row(s) and {len(bad_lives)} covered-lives row(s)")

    ebc = ebc[ebc["total_commissions"] <= comm_cap].copy()
    epc = epc[epc["Covered_Lives"] <= lives_cap].copy()

    dq = {
        "comm_raw_total": comm_raw_total,
        "comm_clean_total": float(ebc["total_commissions"].sum()),
        "lives_raw_max": lives_raw_max,
        "comm_rows_excluded": int(len(bad_comm)),
        "lives_rows_excluded": int(len(bad_lives)),
        "comm_cap": comm_cap,
        "lives_cap": lives_cap,
    }

    # ---- Employer-level lives (MAX to avoid double counting across products/filings)
    emp_lives = epc.groupby("Employer", as_index=False).agg(CoveredLives=("Covered_Lives", "max"))

    # ---- Carrier footprint per employer
    carriers = (
        epc.groupby("Employer", as_index=False)
           .agg(CarrierCount=("Carrier", "nunique"))
    )
    top_carrier = (
        epc.groupby(["Employer", "Carrier"], as_index=False)
           .agg(_lives=("Covered_Lives", "max"))
           .sort_values(["Employer", "_lives"], ascending=[True, False])
           .drop_duplicates("Employer")[["Employer", "Carrier"]]
           .rename(columns={"Carrier": "TopCarrier"})
    )

    # ---- Commissions per employer (SUM) + broker fragmentation
    emp_comm = (
        ebc.groupby("Employer", as_index=False)
           .agg(TotalCommissions=("total_commissions", "sum"), BrokerCount=("Broker", "nunique"))
    )

    # ---- Primary broker = largest commissions (incumbent proxy)
    primary = (
        ebc.groupby(["Employer", "Broker"], as_index=False)
           .agg(BrokerCommissions=("total_commissions", "sum"))
           .sort_values(["Employer", "BrokerCommissions"], ascending=[True, False])
    )
    primary_broker = (
        primary.drop_duplicates("Employer")[["Employer", "Broker", "BrokerCommissions"]]
               .rename(columns={"Broker": "PrimaryBroker", "BrokerCommissions": "PrimaryBrokerCommissions"})
    )

    # ---- Product gap flags
    g = gaps.copy()
    g["Life_f"] = as_flag(g["Life"])
    g["STD_f"] = as_flag(g["STD"])
    g["LTD_f"] = as_flag(g["LTD"])
    g["Dis_f"] = g["STD_f"] | g["LTD_f"]
    g["Bundled_f"] = g["Life_f"] & g["Dis_f"]
    g["LifeOnly_NoDis_f"] = g["Life_f"] & (~g["Dis_f"])
    g["DisOnly_NoLife_f"] = g["Dis_f"] & (~g["Life_f"])
    g["MissingAny_f"] = (~g["Life_f"]) | (~g["STD_f"]) | (~g["LTD_f"])

    # ---- Geo, with state normalization
    geo_use = geo.copy()
    for col in ["State", "City", "ZIP", "EIN"]:
        if col not in geo_use.columns:
            geo_use[col] = pd.NA
    geo_use["StateNorm"] = geo_use["State"].apply(normalize_state)
    geo_use.loc[~geo_use["StateNorm"].isin(list(VALID_STATES)), "StateNorm"] = pd.NA
    # ZIP/EIN are identifiers, not quantities - keep them as text so Excel does not
    # render 07030 as 7030 or add thousands separators.
    for idcol in ["ZIP", "EIN"]:
        geo_use[idcol] = (
            geo_use[idcol].astype(str).str.replace(r"\.0$", "", regex=True).replace({"nan": "", "<NA>": "", "None": ""})
        )
    geo_use = geo_use[["Employer", "StateNorm", "City", "ZIP", "EIN"]].drop_duplicates("Employer")

    emp = (
        g.merge(emp_lives, on="Employer", how="left")
         .merge(carriers, on="Employer", how="left")
         .merge(top_carrier, on="Employer", how="left")
         .merge(emp_comm, on="Employer", how="left")
         .merge(primary_broker, on="Employer", how="left")
         .merge(geo_use, on="Employer", how="left")
    )

    emp["CoveredLives"] = to_numeric(emp["CoveredLives"]).fillna(0)
    emp["TotalCommissions"] = to_numeric(emp["TotalCommissions"]).fillna(0)
    emp["PrimaryBrokerCommissions"] = to_numeric(emp["PrimaryBrokerCommissions"]).fillna(0)
    emp["BrokerCount"] = to_numeric(emp["BrokerCount"]).fillna(0).astype(int)
    emp["CarrierCount"] = to_numeric(emp["CarrierCount"]).fillna(0).astype(int)
    emp["PrimaryBroker"] = emp["PrimaryBroker"].fillna("UNKNOWN")
    emp["TopCarrier"] = emp["TopCarrier"].fillna("UNKNOWN")

    # ---- Company-name matching
    emp["PrimaryBrokerNorm"] = emp["PrimaryBroker"].apply(norm)
    emp["BrokerFamily"] = emp["PrimaryBroker"].apply(broker_family)

    # ---- Broker rollup + tiering (unfiltered market view)
    broker_agg = (
        emp.groupby(["PrimaryBroker", "BrokerFamily"], as_index=False)
           .agg(
               Employers=("Employer", "nunique"),
               CoveredLives=("CoveredLives", "sum"),
               TotalCommissions=("TotalCommissions", "sum"),
               MedianCommissionPerEmployer=("TotalCommissions", "median"),
               MeanCommissionPerEmployer=("TotalCommissions", "mean"),
           )
    )
    broker_agg = assign_tiers(broker_agg, tier2_pct=tier2_pct)

    total_lives = float(broker_agg["CoveredLives"].sum())
    total_emps = int(emp["Employer"].nunique())
    broker_agg["LivesShare%"] = (broker_agg["CoveredLives"] / total_lives * 100.0) if total_lives else 0.0
    broker_agg["EmployerShare%"] = (broker_agg["Employers"] / total_emps * 100.0) if total_emps else 0.0
    broker_agg["PrimaryBrokerNorm"] = broker_agg["PrimaryBroker"].apply(norm)
    broker_agg["TierLabel"] = broker_agg["Tier"].map(TIER_LABEL)
    broker_agg = broker_agg.sort_values("CoveredLives", ascending=False)

    # Tier back onto employers
    tier_map = broker_agg.set_index("PrimaryBroker")["Tier"].to_dict()
    emp["BrokerTier"] = emp["PrimaryBroker"].map(tier_map).fillna("Tier3")

    # ---- Readable product columns
    # NB: avoid the literal "None" as a sentinel - pandas reads it back as NaN.
    def products_held(r):
        held = [p for p, f in [("Life", r["Life_f"]), ("STD", r["STD_f"]), ("LTD", r["LTD_f"])] if f]
        return " + ".join(held) if held else "(no products)"

    def products_missing(r):
        miss = [p for p, f in [("Life", r["Life_f"]), ("STD", r["STD_f"]), ("LTD", r["LTD_f"])] if not f]
        return " + ".join(miss) if miss else "(complete - all 3)"

    emp["ProductsHeld"] = emp.apply(products_held, axis=1)
    emp["ProductsMissing"] = emp.apply(products_missing, axis=1)
    emp["LivesLog"] = emp["CoveredLives"].apply(lambda x: math.log(float(x) + 1.0))

    employers = emp.rename(columns={
        "StateNorm": "State",
        "Life_f": "Has_Life",
        "STD_f": "Has_STD",
        "LTD_f": "Has_LTD",
        "Bundled_f": "Has_Life_And_Disability",
        "LifeOnly_NoDis_f": "LifeOnly_NoDisability",
        "DisOnly_NoLife_f": "DisabilityOnly_NoLife",
        "MissingAny_f": "MissingAnyProduct",
    })[[
        "Employer", "State", "City", "ZIP", "EIN",
        "PrimaryBroker", "PrimaryBrokerNorm", "BrokerFamily", "BrokerTier",
        "BrokerCount", "PrimaryBrokerCommissions", "TotalCommissions",
        "CoveredLives", "CarrierCount", "TopCarrier",
        "Has_Life", "Has_STD", "Has_LTD",
        "ProductsHeld", "ProductsMissing",
        "Has_Life_And_Disability", "LifeOnly_NoDisability", "DisabilityOnly_NoLife", "MissingAnyProduct",
    ]].sort_values("CoveredLives", ascending=False)
    employers["BrokerTier"] = employers["BrokerTier"].map(TIER_LABEL).fillna(employers["BrokerTier"])

    # ---- Broker name mapping (audit trail for the company-name matching)
    name_map = (
        ebc.assign(_norm=lambda d: d["Broker"].apply(norm))
           .groupby(["Broker", "_norm"], as_index=False)
           .agg(EmployerFilings=("Employer", "nunique"), Commissions=("total_commissions", "sum"))
           .rename(columns={"Broker": "BrokerNameAsFiled", "_norm": "NormalizedName"})
    )
    name_map["MatchedFamily"] = name_map["BrokerNameAsFiled"].apply(broker_family)
    name_map["MatchRule"] = name_map["BrokerNameAsFiled"].apply(
        lambda b: "AON composite (prefix match on AON roots)" if is_aon_composite(b)
        else ("Tier1 major (keyword match)" if broker_family(b) != "OTHER" else "No match - OTHER")
    )
    name_map = name_map.sort_values("Commissions", ascending=False)

    # ---- State summary (AON vs competitors)
    st_base = employers.dropna(subset=["State"]).copy()
    st_base["IsAON"] = st_base["BrokerFamily"].eq("AON")
    state_summary = (
        st_base.groupby("State", as_index=False)
               .agg(
                   Employers=("Employer", "nunique"),
                   Brokers=("PrimaryBroker", "nunique"),
                   CoveredLives=("CoveredLives", "sum"),
                   TotalCommissions=("TotalCommissions", "sum"),
                   MissingAnyProductRate=("MissingAnyProduct", "mean"),
               )
    )
    state_summary["MissingAnyProduct%"] = state_summary["MissingAnyProductRate"] * 100.0
    aon_state = (
        st_base[st_base["IsAON"]].groupby("State", as_index=False)
              .agg(AON_Employers=("Employer", "nunique"), AON_Lives=("CoveredLives", "sum"))
    )
    state_summary = state_summary.merge(aon_state, on="State", how="left").fillna({"AON_Employers": 0, "AON_Lives": 0})
    state_summary["AONLivesShare%"] = state_summary.apply(
        lambda r: (r["AON_Lives"] / r["CoveredLives"] * 100.0) if r["CoveredLives"] else 0.0, axis=1
    )
    state_summary["CompetitorLivesShare%"] = 100.0 - state_summary["AONLivesShare%"]
    state_summary["UnderIndexGap%"] = state_summary["CompetitorLivesShare%"] - state_summary["AONLivesShare%"]
    state_summary["BrokerFragRatio"] = state_summary.apply(
        lambda r: (r["Brokers"] / r["Employers"]) if r["Employers"] else 0.0, axis=1
    )
    state_summary = state_summary.sort_values("CoveredLives", ascending=False)

    # ---- Carrier x product summary
    carrier_summary = (
        epc.groupby(["Carrier", "Product"], as_index=False)
           .agg(Employers=("Employer", "nunique"), CoveredLives=("Covered_Lives", "sum"))
           .sort_values(["Product", "CoveredLives"], ascending=[True, False])
    )

    # ---- Product whitespace summary
    whitespace = (
        employers.groupby("ProductsHeld", as_index=False)
                 .agg(Employers=("Employer", "nunique"), CoveredLives=("CoveredLives", "sum"),
                      TotalCommissions=("TotalCommissions", "sum"))
                 .sort_values("Employers", ascending=False)
    )
    whitespace["Employers%"] = whitespace["Employers"] / whitespace["Employers"].sum() * 100.0

    # ---- Detail sheets
    detail_broker = (
        ebc[["Employer", "Broker", "total_commissions", "ACK_ID"]]
        .rename(columns={"total_commissions": "CommissionsPaid", "ACK_ID": "FilingID"})
        .assign(BrokerFamily=lambda d: d["Broker"].apply(broker_family))
        .sort_values("CommissionsPaid", ascending=False)
    )
    detail_carrier = (
        epc[["Employer", "Product", "Carrier", "Covered_Lives", "ACK_ID"]]
        .rename(columns={"Covered_Lives": "CoveredLives", "ACK_ID": "FilingID"})
        .sort_values("CoveredLives", ascending=False)
    )

    return {
        "employers": employers,
        "broker_summary": broker_agg[[
            "PrimaryBroker", "PrimaryBrokerNorm", "BrokerFamily", "TierLabel", "Employers", "CoveredLives",
            "LivesShare%", "EmployerShare%", "TotalCommissions",
            "MedianCommissionPerEmployer", "MeanCommissionPerEmployer",
        ]],
        "name_map": name_map,
        "state_summary": state_summary[[
            "State", "Employers", "Brokers", "BrokerFragRatio", "CoveredLives", "TotalCommissions",
            "AON_Employers", "AON_Lives", "AONLivesShare%", "CompetitorLivesShare%", "UnderIndexGap%",
            "MissingAnyProduct%",
        ]],
        "carrier_summary": carrier_summary,
        "whitespace": whitespace,
        "data_quality_flags": flags,
        "detail_broker": detail_broker,
        "detail_carrier": detail_carrier,
    }, dq


# =========================================================
# Excel writing
# =========================================================
MAX_XLSX_ROWS = 1_048_575  # minus header


def write_sheet(writer, df: pd.DataFrame, sheet: str, pct_cols=(), money_cols=(), int_cols=(), ratio_cols=()):
    if len(df) > MAX_XLSX_ROWS:
        log(f"WARNING: '{sheet}' has {len(df):,} rows - truncating to Excel's limit")
        df = df.head(MAX_XLSX_ROWS)

    df.to_excel(writer, sheet_name=sheet, index=False, startrow=1, header=False)
    wb, ws = writer.book, writer.sheets[sheet]

    hdr = wb.add_format({"bold": True, "bg_color": "#1F3B57", "font_color": "white",
                         "border": 1, "text_wrap": True, "valign": "vcenter"})
    f_money = wb.add_format({"num_format": "$#,##0"})
    f_int = wb.add_format({"num_format": "#,##0"})
    f_pct = wb.add_format({"num_format": "0.0"})
    f_ratio = wb.add_format({"num_format": "0.00"})

    for i, col in enumerate(df.columns):
        ws.write(0, i, col, hdr)
        width = max(len(str(col)) + 2, 12)
        if df[col].dtype == "object" and len(df):
            width = max(width, min(int(df[col].astype(str).str.len().quantile(0.95)) + 2, 46))
        fmt = None
        if col in money_cols:
            fmt = f_money
        elif col in int_cols:
            fmt = f_int
        elif col in pct_cols:
            fmt = f_pct
        elif col in ratio_cols:
            fmt = f_ratio
        ws.set_column(i, i, width, fmt)

    ws.freeze_panes(1, 1)
    if len(df):
        ws.autofilter(0, 0, len(df), len(df.columns) - 1)


def write_readme(writer, counts: dict, dq: dict, tier2_pct: float, with_detail: bool):
    wb = writer.book
    ws = wb.add_worksheet("README")
    writer.sheets["README"] = ws

    title = wb.add_format({"bold": True, "font_size": 16})
    h2 = wb.add_format({"bold": True, "font_size": 12, "bg_color": "#EAF0F6"})
    wrap = wb.add_format({"text_wrap": True, "valign": "top"})
    bold = wb.add_format({"bold": True, "valign": "top"})

    ws.set_column(0, 0, 34)
    ws.set_column(1, 1, 96)

    r = 0
    ws.write(r, 0, "KAPI Pilot - Cleaned Dataset", title); r += 2
    ws.write(r, 0, "Generated", bold); ws.write(r, 1, datetime.now().strftime("%Y-%m-%d %H:%M")); r += 1
    ws.write(r, 0, "Source", bold)
    ws.write(r, 1, "DOL Form 5500 + Schedule A, 2024 latest filings (F_5500, F_SCH_A, F_SCH_A_PART1)", wrap); r += 2

    ws.write(r, 0, "Sheets", h2); r += 1
    sheets = [
        ("Employers", f"{counts['employers']:,} rows. One row per employer - the master cleaned table. "
                      "Product coverage, covered lives, commissions, primary broker + matched family/tier, geography."),
        ("Broker_Summary", f"{counts['broker_summary']:,} rows. One row per broker (as primary/incumbent) with "
                           "employer count, covered lives, market share and commission stats."),
        ("Broker_Name_Matching", f"{counts['name_map']:,} rows. Audit trail for the company-name matching: every broker "
                                 "name as filed, its normalized form, the family it matched, and which rule fired."),
        ("State_Summary", f"{counts['state_summary']:,} rows. Per-state totals, AON vs competitor share, "
                          "under-index gap and broker fragmentation."),
        ("Carrier_Product_Summary", f"{counts['carrier_summary']:,} rows. Carrier x product footprint by employers and lives."),
        ("Product_Whitespace", f"{counts['whitespace']:,} rows. Employer counts by product combination held (gap analysis)."),
        ("Data_Quality_Flags", f"{counts['data_quality_flags']:,} rows. Every source row excluded as an implausible "
                               "filer keying error, with its reported value - read this before quoting any total."),
    ]
    if with_detail:
        sheets += [
            ("Detail_Employer_Broker", f"{counts['detail_broker']:,} rows. Every employer-broker commission row."),
            ("Detail_Employer_Carrier", f"{counts['detail_carrier']:,} rows. Every employer-product-carrier row."),
        ]
    for name, desc in sheets:
        ws.write(r, 0, name, bold); ws.write(r, 1, desc, wrap); r += 1
    r += 1

    ws.write(r, 0, "Cleaning rules applied", h2); r += 1
    rules = [
        ("Broker name matching",
         "Names are uppercased, punctuation stripped, whitespace collapsed. AON is matched as a composite: a name "
         "counts as AON only if it STARTS WITH one of AON CORPORATION / AON RISK SERVICES / AON HEWITT / "
         "AON CONSULTING / AON SOLUTIONS - this deliberately excludes false positives like DATAONLINE. "
         "Tier1 majors (Marsh/MMC, WTW/Willis/Towers, Gallagher, Brown & Brown) are matched on word-boundary keywords. "
         "Everything else is OTHER. See the Broker_Name_Matching sheet for the full mapping."),
        ("Broker tiering",
         f"Tier0 = AON. Tier1 = named global majors. Tier2 = top {tier2_pct:.0%} of OTHER brokers by covered lives. "
         "Tier3 = the rest."),
        ("Primary broker",
         "An employer can file multiple brokers. The primary broker is the one with the largest commissions on that "
         "employer's filings - used as the incumbent proxy for all broker rollups."),
        ("Covered lives",
         "Taken as the MAX across an employer's product/carrier rows, not the sum, to avoid double-counting the same "
         "population across Life/STD/LTD lines."),
        ("Commissions",
         "Summed per employer across brokers. Commissions come from Schedule A Part 1 and are NOT joined to "
         "carrier/product rows - joining them would multiply the amounts."),
        ("State normalization",
         "Full state names mapped to 2-letter codes; anything outside the 50 states + DC + PR is blanked rather than "
         "counted as its own state."),
        ("Outlier removal (IMPORTANT)",
         f"Form 5500 is self-reported and a few filings contain keying errors large enough to swamp every total. "
         f"As filed, total commissions came to ${dq['comm_raw_total']:,.0f} - because ONE row reported $563 trillion. "
         f"Rows above ${dq['comm_cap']:,.0f} in commissions ({dq['comm_rows_excluded']} rows) or "
         f"{dq['lives_cap']:,.0f} covered lives ({dq['lives_rows_excluded']} rows, max reported was "
         f"{dq['lives_raw_max']:,.0f} by a small employer) are excluded from all rollups. "
         f"Cleaned total commissions: ${dq['comm_clean_total']:,.0f}. "
         "Every excluded row is listed on the Data_Quality_Flags sheet - re-include them there if you disagree "
         "with the thresholds."),
    ]
    for name, desc in rules:
        ws.write(r, 0, name, bold); ws.write(r, 1, desc, wrap); r += 1
    r += 1

    ws.write(r, 0, "Caveats", h2); r += 1
    caveats = [
        "Employers are keyed on the sponsor name as filed - separate legal entities of the same parent appear as "
        "separate rows. No parent-company roll-up has been applied.",
        "Only Life, STD and LTD lines are in scope; medical/dental/vision are excluded.",
        "2024 filings only - a plan that did not file in 2024 will not appear.",
        "Commissions are as reported on Schedule A and vary in completeness by filer.",
        "PrimaryBroker = UNKNOWN means the employer has carrier/product coverage but no broker commission record on "
        "Schedule A Part 1 - not that the employer has no broker. Those rows still carry valid covered-lives data.",
    ]
    for c in caveats:
        ws.write(r, 1, "- " + c, wrap); r += 1


def export(out_path: Path, tier2_pct: float, with_detail: bool, comm_cap: float, lives_cap: float):
    d, dq = build(tier2_pct=tier2_pct, comm_cap=comm_cap, lives_cap=lives_cap)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {k: len(v) for k, v in d.items()}
    log(f"writing {out_path}")

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        write_readme(writer, counts, dq, tier2_pct, with_detail)

        write_sheet(
            writer, d["employers"], "Employers",
            money_cols=("TotalCommissions", "PrimaryBrokerCommissions"),
            int_cols=("CoveredLives", "BrokerCount", "CarrierCount"),
        )
        write_sheet(
            writer, d["broker_summary"], "Broker_Summary",
            money_cols=("TotalCommissions", "MedianCommissionPerEmployer", "MeanCommissionPerEmployer"),
            int_cols=("Employers", "CoveredLives"),
            pct_cols=("LivesShare%", "EmployerShare%"),
        )
        write_sheet(
            writer, d["name_map"], "Broker_Name_Matching",
            money_cols=("Commissions",), int_cols=("EmployerFilings",),
        )
        write_sheet(
            writer, d["state_summary"], "State_Summary",
            money_cols=("TotalCommissions",),
            int_cols=("Employers", "Brokers", "CoveredLives", "AON_Employers", "AON_Lives"),
            pct_cols=("AONLivesShare%", "CompetitorLivesShare%", "UnderIndexGap%", "MissingAnyProduct%"),
            ratio_cols=("BrokerFragRatio",),
        )
        write_sheet(
            writer, d["carrier_summary"], "Carrier_Product_Summary",
            int_cols=("Employers", "CoveredLives"),
        )
        write_sheet(
            writer, d["whitespace"], "Product_Whitespace",
            money_cols=("TotalCommissions",), int_cols=("Employers", "CoveredLives"),
            pct_cols=("Employers%",),
        )
        write_sheet(
            writer, d["data_quality_flags"], "Data_Quality_Flags",
            int_cols=("ReportedValue",),
        )

        if with_detail:
            write_sheet(writer, d["detail_broker"], "Detail_Employer_Broker",
                        money_cols=("CommissionsPaid",))
            write_sheet(writer, d["detail_carrier"], "Detail_Employer_Carrier",
                        int_cols=("CoveredLives",))

    size_mb = out_path.stat().st_size / 1024 / 1024
    log(f"done - {size_mb:.1f} MB")
    for k, v in counts.items():
        log(f"  {k}: {v:,} rows")


def parse_args():
    p = argparse.ArgumentParser(description="Export the cleaned pilot dataset to Excel.")
    p.add_argument("--out", default=str(REPO_ROOT / "exports" / "kapi_pilot_dataset.xlsx"),
                   help="Output .xlsx path")
    p.add_argument("--tier2-pct", type=float, default=0.10,
                   help="Tier2 cutoff: top X%% of 'Other' brokers by covered lives (default 0.10)")
    p.add_argument("--with-detail", action="store_true",
                   help="Include the full employer-broker and employer-carrier detail sheets (much larger file)")
    p.add_argument("--comm-cap", type=float, default=10_000_000.0,
                   help="Drop employer-broker commission rows above this amount as filer errors (default 10,000,000)")
    p.add_argument("--lives-cap", type=float, default=1_500_000.0,
                   help="Drop employer-carrier rows above this many covered lives as filer errors (default 1,500,000)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    export(Path(a.out).expanduser().resolve(), a.tier2_pct, a.with_detail, a.comm_cap, a.lives_cap)
