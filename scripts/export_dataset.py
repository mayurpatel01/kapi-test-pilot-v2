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
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "marts"

sys.path.insert(0, str(REPO_ROOT / "etl"))
from benefits import ALL_PRODUCTS, PRODUCT_GROUP, VB_TRIO, VOLUNTARY_PRODUCTS, column_suffix
from brokers import (
    TIER1_PATTERNS, TIER_LABEL, assign_tiers, broker_family, is_aon_composite,
    match_rule, norm,
)


def log(msg: str):
    print(f"[export] {msg}")


# =========================================================
# Cleaning helpers (mirrors app.py)
# =========================================================
# norm() is imported from etl/brokers.py -- see the import block above.


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


# Broker name matching and tiering live in etl/brokers.py so the dashboard and
# this export cannot drift apart.


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
        contracts = rd("employer_contract")
    except FileNotFoundError:
        log("WARNING: employer_contract.parquet missing - premium columns will be zero. "
            "Re-run etl/build_marts.py to generate it.")
        contracts = pd.DataFrame(columns=["ContractRowID", "ACK_ID", "Employer", "Carrier",
                                          "Covered_Lives", "Premium", "PremiumSource",
                                          "RetainedCommission", "Products", "ProductCount"])
    try:
        geo = rd("employer_geo")
    except FileNotFoundError:
        log("WARNING: employer_geo.parquet missing - State/City/ZIP/EIN will be blank")
        geo = pd.DataFrame(columns=["Employer", "State", "City", "ZIP", "EIN"])
    return epc, ebc, gaps, contracts, geo


def build(tier2_pct: float = 0.10, comm_cap: float = 10_000_000.0, lives_cap: float = 1_500_000.0,
          premium_cap: float = 500_000_000.0):
    epc, ebc, gaps, contracts, geo = load_marts()

    epc["Covered_Lives"] = to_numeric(epc["Covered_Lives"]).fillna(0)
    epc["Premium"] = to_numeric(epc.get("Premium", 0)).fillna(0)
    ebc["total_commissions"] = to_numeric(ebc["total_commissions"]).fillna(0)
    contracts["Premium"] = to_numeric(contracts["Premium"]).fillna(0)

    # ---- Data-quality guard -------------------------------------------------
    # Form 5500 is self-reported and a handful of filings carry obvious keying
    # errors (e.g. a single row reporting $563 TRILLION in commissions, or a
    # small employer reporting 6M covered lives). Left in, one row swamps every
    # sum in the workbook. Rows above the caps are removed from all rollups and
    # listed in full on the Data_Quality_Flags sheet - nothing is dropped silently.
    comm_raw_total = float(ebc["total_commissions"].sum())
    lives_raw_max = float(epc["Covered_Lives"].max()) if len(epc) else 0.0

    premium_raw_total = float(contracts["Premium"].sum())

    bad_comm = ebc[ebc["total_commissions"] > comm_cap].copy()
    bad_lives = epc[epc["Covered_Lives"] > lives_cap].copy()
    bad_prem = contracts[contracts["Premium"] > premium_cap].copy()

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
        bad_prem.assign(
            Field="Premium",
            Counterparty=bad_prem["Carrier"] + " (" + bad_prem["Products"].astype(str) + ")",
            ReportedValue=bad_prem["Premium"],
            Reason=lambda d: f"Exceeds plausibility cap of ${premium_cap:,.0f} for a single contract",
        )[["Employer", "Field", "Counterparty", "ACK_ID", "ReportedValue", "Reason"]],
    ], ignore_index=True).rename(columns={"ACK_ID": "FilingID"}).sort_values("ReportedValue", ascending=False)

    if len(flags):
        log(f"data-quality: excluding {len(bad_comm)} commission row(s), {len(bad_lives)} covered-lives row(s) "
            f"and {len(bad_prem)} premium row(s)")

    ebc = ebc[ebc["total_commissions"] <= comm_cap].copy()
    epc = epc[epc["Covered_Lives"] <= lives_cap].copy()
    contracts = contracts[contracts["Premium"] <= premium_cap].copy()

    dq = {
        "comm_raw_total": comm_raw_total,
        "comm_clean_total": float(ebc["total_commissions"].sum()),
        "lives_raw_max": lives_raw_max,
        "comm_rows_excluded": int(len(bad_comm)),
        "lives_rows_excluded": int(len(bad_lives)),
        "comm_cap": comm_cap,
        "lives_cap": lives_cap,
        "premium_raw_total": premium_raw_total,
        "premium_clean_total": float(contracts["Premium"].sum()),
        "premium_rows_excluded": int(len(bad_prem)),
        "premium_cap": premium_cap,
        "premium_fill_pct": float((contracts["Premium"] > 0).mean() * 100.0) if len(contracts) else 0.0,
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

    # ---- Premium per employer.
    # Summed over DISTINCT contracts, not over the product-exploded table: a
    # contract covering life + STD + LTD appears three times there, and summing
    # it would triple the premium. Different contracts with the same carrier are
    # genuinely separate and do add up.
    emp_premium = (
        contracts.groupby("Employer", as_index=False)
                 .agg(TotalPremium=("Premium", "sum"),
                      ContractCount=("ContractRowID", "nunique"))
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

    # ---- Voluntary benefit flags (parsed from Schedule A OTHER free text).
    # Driven off benefits.VOLUNTARY_PRODUCTS so adding a product there flows
    # through the flags, penetration, targets and carrier sheets automatically.
    for prod in ALL_PRODUCTS:
        g[f"{prod}_f"] = as_flag(g[prod]) if prod in g.columns else False

    g["AnyVB_f"] = False
    g["VBCount"] = 0
    for prod in VOLUNTARY_PRODUCTS:
        g["AnyVB_f"] = g["AnyVB_f"] | g[f"{prod}_f"]
        g["VBCount"] = g["VBCount"] + g[f"{prod}_f"].astype(int)

    # The classic worksite trio sold together.
    g["VBTrio_f"] = True
    for prod in VB_TRIO:
        g["VBTrio_f"] = g["VBTrio_f"] & g[f"{prod}_f"]
    # Core-but-no-VB is the cross-sell target: an established group benefits
    # relationship with none of the voluntary line attached.
    g["CoreNoVB_f"] = (g["Life_f"] | g["Dis_f"]) & (~g["AnyVB_f"])

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
         .merge(emp_premium, on="Employer", how="left")
         .merge(primary_broker, on="Employer", how="left")
         .merge(geo_use, on="Employer", how="left")
    )

    emp["CoveredLives"] = to_numeric(emp["CoveredLives"]).fillna(0)
    emp["TotalPremium"] = to_numeric(emp["TotalPremium"]).fillna(0)
    emp["ContractCount"] = to_numeric(emp["ContractCount"]).fillna(0).astype(int)
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
               TotalPremium=("TotalPremium", "sum"),
               MedianCommissionPerEmployer=("TotalCommissions", "median"),
               MeanCommissionPerEmployer=("TotalCommissions", "mean"),
               MedianPremiumPerEmployer=("TotalPremium", "median"),
           )
    )
    broker_agg["CommissionPctOfPremium"] = broker_agg.apply(
        lambda r: (r["TotalCommissions"] / r["TotalPremium"] * 100.0) if r["TotalPremium"] > 0 else pd.NA, axis=1
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

    def vb_held(r):
        held = [p for p in VOLUNTARY_PRODUCTS if r[f"{p}_f"]]
        return " + ".join(held) if held else "(none)"

    def vb_missing(r):
        miss = [p for p in VOLUNTARY_PRODUCTS if not r[f"{p}_f"]]
        return " + ".join(miss) if miss else "(complete - all 4)"

    emp["ProductsHeld"] = emp.apply(products_held, axis=1)
    emp["ProductsMissing"] = emp.apply(products_missing, axis=1)
    emp["VBHeld"] = emp.apply(vb_held, axis=1)
    emp["VBMissing"] = emp.apply(vb_missing, axis=1)
    emp["LivesLog"] = emp["CoveredLives"].apply(lambda x: math.log(float(x) + 1.0))

    # Derived money ratios. Guarded against divide-by-zero, and left blank rather
    # than zero where the denominator is missing so a gap reads as "unknown"
    # instead of "$0 per life".
    emp["PremiumPerLife"] = emp.apply(
        lambda r: (r["TotalPremium"] / r["CoveredLives"]) if r["CoveredLives"] > 0 else pd.NA, axis=1
    )
    emp["CommissionPctOfPremium"] = emp.apply(
        lambda r: (r["TotalCommissions"] / r["TotalPremium"] * 100.0) if r["TotalPremium"] > 0 else pd.NA, axis=1
    )

    rename_map = {
        "StateNorm": "State",
        "Life_f": "Has_Life",
        "STD_f": "Has_STD",
        "LTD_f": "Has_LTD",
        "Bundled_f": "Has_Life_And_Disability",
        "LifeOnly_NoDis_f": "LifeOnly_NoDisability",
        "DisOnly_NoLife_f": "DisabilityOnly_NoLife",
        "MissingAny_f": "MissingAnyProduct",
        "AD&D_f": "Has_ADandD",
        "AnyVB_f": "Has_AnyVoluntary",
        "VBTrio_f": "Has_VoluntaryTrio",
        "CoreNoVB_f": "CoreButNoVoluntary",
    }
    vb_has_cols = []
    for prod in VOLUNTARY_PRODUCTS:
        col = f"Has_{column_suffix(prod)}"
        rename_map[f"{prod}_f"] = col
        vb_has_cols.append(col)

    employers = emp.rename(columns=rename_map)[[
        "Employer", "State", "City", "ZIP", "EIN",
        "PrimaryBroker", "PrimaryBrokerNorm", "BrokerFamily", "BrokerTier",
        "BrokerCount", "PrimaryBrokerCommissions", "TotalCommissions",
        "TotalPremium", "PremiumPerLife", "CommissionPctOfPremium", "ContractCount",
        "CoveredLives", "CarrierCount", "TopCarrier",
        "Has_Life", "Has_STD", "Has_LTD",
        "ProductsHeld", "ProductsMissing",
        "Has_Life_And_Disability", "LifeOnly_NoDisability", "DisabilityOnly_NoLife", "MissingAnyProduct",
    ] + vb_has_cols + [
        "VBCount", "VBHeld", "VBMissing",
        "Has_AnyVoluntary", "Has_VoluntaryTrio", "CoreButNoVoluntary",
        "Has_ADandD",
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
    name_map["MatchRule"] = name_map["BrokerNameAsFiled"].apply(match_rule)
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
                   TotalPremium=("TotalPremium", "sum"),
                   MissingAnyProductRate=("MissingAnyProduct", "mean"),
                   AnyVBRate=("Has_AnyVoluntary", "mean"),
               )
    )
    state_summary["AnyVoluntary%"] = state_summary.pop("AnyVBRate") * 100.0
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
    group_col = ["ProductGroup"] if "ProductGroup" in epc.columns else []
    carrier_summary = (
        epc.groupby(["Carrier", "Product"] + group_col, as_index=False)
           .agg(Employers=("Employer", "nunique"), CoveredLives=("Covered_Lives", "sum"),
                Premium=("Premium", "sum"))
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

    # ---- Employer x product detail: ONE ROW PER PRODUCT PER EMPLOYER.
    # The "where do we stand, line by line" view. Money columns are named for
    # exactly what they are, because Form 5500 does not report all of them at
    # product grain:
    #   CoveredLives          - real, per product (MAX across carriers, as elsewhere)
    #   PremiumOnContracts    - real, but OVERLAPPING: a life+STD+LTD contract
    #                           reports one premium and it appears on all three rows
    #   SoleProductPremium    - real and unambiguous: contracts covering only this product
    #   EmployerCommissions   - real, but EMPLOYER-level, repeated on every row
    #   EstCommissionForProduct - ESTIMATE: employer commissions split by the
    #                           product's share of that employer's premium
    prod_lives = (
        epc.groupby(["Employer", "Product", "ProductGroup"], as_index=False)
           .agg(CoveredLives=("Covered_Lives", "max"),
                Carriers=("Carrier", "nunique"),
                Contracts=("ContractRowID", "nunique"))
    )
    prod_premium = (
        epc.drop_duplicates(["Employer", "Product", "ContractRowID"])
           .groupby(["Employer", "Product"], as_index=False)
           .agg(PremiumOnContracts=("Premium", "sum"))
    )
    sole_ids = set(contracts.loc[contracts["ProductCount"] == 1, "ContractRowID"])
    prod_sole = (
        epc[epc["ContractRowID"].isin(sole_ids)]
           .drop_duplicates(["Employer", "Product", "ContractRowID"])
           .groupby(["Employer", "Product"], as_index=False)
           .agg(SoleProductPremium=("Premium", "sum"))
    )
    prod_top_carrier = (
        epc.groupby(["Employer", "Product", "Carrier"], as_index=False)
           .agg(_l=("Covered_Lives", "max"))
           .sort_values(["Employer", "Product", "_l"], ascending=[True, True, False])
           .drop_duplicates(["Employer", "Product"])[["Employer", "Product", "Carrier"]]
           .rename(columns={"Carrier": "TopCarrier"})
    )

    emp_cols = ["Employer", "EIN", "State", "City", "PrimaryBroker", "BrokerFamily",
                "BrokerTier", "TotalCommissions", "TotalPremium", "CoveredLives"]
    product_detail = (
        prod_lives
        .merge(prod_premium, on=["Employer", "Product"], how="left")
        .merge(prod_sole, on=["Employer", "Product"], how="left")
        .merge(prod_top_carrier, on=["Employer", "Product"], how="left")
        .merge(employers[emp_cols].rename(columns={
            "TotalCommissions": "EmployerCommissions",
            "TotalPremium": "EmployerPremium",
            "CoveredLives": "EmployerCoveredLives",
        }), on="Employer", how="left")
    )
    product_detail["PremiumOnContracts"] = product_detail["PremiumOnContracts"].fillna(0)
    product_detail["SoleProductPremium"] = product_detail["SoleProductPremium"].fillna(0)
    product_detail["EmployerCommissions"] = product_detail["EmployerCommissions"].fillna(0)

    # Pro-rata commission estimate. Shares sum to 1 per employer by construction,
    # so the estimates reconstitute the employer total even though the underlying
    # premium figures overlap.
    share_base = product_detail.groupby("Employer")["PremiumOnContracts"].transform("sum")
    product_detail["ProductPremiumShare%"] = (
        product_detail["PremiumOnContracts"] / share_base.replace(0, pd.NA) * 100.0
    )
    product_detail["EstCommissionForProduct"] = (
        product_detail["EmployerCommissions"]
        * product_detail["PremiumOnContracts"] / share_base.replace(0, pd.NA)
    )
    # An employer who reported commissions but no premium anywhere has nothing to
    # split the commission across, so the estimate is blank rather than zero.
    # Flagged instead of hidden: summing EstCommissionForProduct will fall short
    # of total commissions by exactly this much.
    product_detail["CommissionAllocatable"] = share_base > 0
    _unalloc = product_detail.loc[
        (~product_detail["CommissionAllocatable"]) & (product_detail["EmployerCommissions"] > 0)
    ].drop_duplicates("Employer")
    if len(_unalloc):
        log(f"commission estimate unavailable for {len(_unalloc):,} employers "
            f"(${_unalloc['EmployerCommissions'].sum():,.0f}) - they report commissions but no premium")

    # EIN is an identifier: keep it text so Excel neither strips a leading zero
    # nor renders it in scientific notation.
    product_detail["EIN"] = (
        product_detail["EIN"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .replace({"nan": "", "<NA>": "", "None": ""})
    )

    # The filter the whole page exists for.
    product_detail["AON_Is_Broker"] = product_detail["BrokerFamily"].eq("AON")
    product_detail["BrokerStatus"] = product_detail["AON_Is_Broker"].map(
        {True: "AON is broker of record", False: "NOT AON - opportunity"}
    )
    product_detail["PremiumPerLife"] = (
        product_detail["PremiumOnContracts"] / product_detail["CoveredLives"].replace(0, pd.NA)
    )

    product_detail = product_detail[[
        "Employer", "EIN", "State", "City",
        "Product", "ProductGroup",
        "CoveredLives", "PremiumOnContracts", "PremiumPerLife", "SoleProductPremium",
        "ProductPremiumShare%", "EstCommissionForProduct", "CommissionAllocatable",
        "BrokerStatus", "AON_Is_Broker", "PrimaryBroker", "BrokerFamily", "BrokerTier",
        "EmployerCommissions", "EmployerPremium", "EmployerCoveredLives",
        "TopCarrier", "Carriers", "Contracts",
    ]].sort_values(["PremiumOnContracts", "Employer", "Product"], ascending=[False, True, True])

    log(f"product detail: {len(product_detail):,} employer-product rows "
        f"({int((~product_detail['AON_Is_Broker']).sum()):,} not on the AON book)")

    # ---- Premium by product.
    # A contract covering several benefits reports ONE premium, so these figures
    # OVERLAP and must not be summed down the column. ContractsIncluding /
    # SoleProductPremium make that explicit: the sole-product column is the only
    # premium unambiguously attributable to a single product.
    prem_rows = []
    contract_products = contracts.set_index("ContractRowID")["Products"].to_dict() if len(contracts) else {}
    epc_in = epc[epc["ContractRowID"].isin(contracts["ContractRowID"])] if "ContractRowID" in epc.columns else epc
    for prod in ALL_PRODUCTS:
        sel = epc_in[epc_in["Product"] == prod]
        if not len(sel):
            continue
        by_contract = sel.drop_duplicates("ContractRowID") if "ContractRowID" in sel.columns else sel
        sole = by_contract[by_contract["ContractRowID"].map(
            lambda cid: contract_products.get(cid, "").count("+") == 0)] \
            if "ContractRowID" in by_contract.columns else by_contract.iloc[0:0]
        prem_rows.append({
            "Product": prod,
            "ProductGroup": PRODUCT_GROUP.get(prod, "Other"),
            "Employers": sel["Employer"].nunique(),
            "ContractsIncluding": by_contract["ContractRowID"].nunique() if "ContractRowID" in by_contract.columns else len(by_contract),
            "PremiumOnThoseContracts": by_contract["Premium"].sum(),
            "SoleProductContracts": len(sole),
            "SoleProductPremium": sole["Premium"].sum(),
            "CoveredLives": by_contract["Covered_Lives"].sum(),
        })
    premium_by_product = pd.DataFrame(prem_rows).sort_values("PremiumOnThoseContracts", ascending=False)

    # ---- Voluntary benefits: penetration by product
    vb_rows = []
    n_emp = len(employers)
    rollups = [("--- ANY voluntary product ---", "Has_AnyVoluntary"),
               (f"--- ALL of {' + '.join(VB_TRIO)} ---", "Has_VoluntaryTrio"),
               ("AD&D (life rider, NOT counted as voluntary)", "Has_ADandD")]
    for prod, col in [(p, f"Has_{column_suffix(p)}") for p in VOLUNTARY_PRODUCTS] + rollups:
        sel = employers[employers[col]]
        vb_rows.append({
            "Product": prod,
            "Employers": len(sel),
            "Employers%": len(sel) / n_emp * 100.0 if n_emp else 0.0,
            "CoveredLives": sel["CoveredLives"].sum(),
            "TotalPremium": sel["TotalPremium"].sum(),
            "TotalCommissions": sel["TotalCommissions"].sum(),
            "MedianLives": sel["CoveredLives"].median() if len(sel) else 0.0,
            "MedianPremium": sel["TotalPremium"].median() if len(sel) else 0.0,
        })
    vb_penetration = pd.DataFrame(vb_rows)

    # ---- Voluntary benefits: attach-rate gap by broker tier
    # This is the sheet that answers "who is leaving VB on the table".
    tier_aggs = {f"{column_suffix(p)}Rate": (f"Has_{column_suffix(p)}", "mean")
                 for p in VOLUNTARY_PRODUCTS}
    vb_by_tier = (
        employers.groupby("BrokerTier", as_index=False)
                 .agg(Employers=("Employer", "nunique"), CoveredLives=("CoveredLives", "sum"),
                      AnyVBRate=("Has_AnyVoluntary", "mean"),
                      CoreNoVBRate=("CoreButNoVoluntary", "mean"),
                      **tier_aggs)
                 .sort_values("CoveredLives", ascending=False)
    )
    for c in [c for c in vb_by_tier.columns if c.endswith("Rate")]:
        vb_by_tier[c[:-4] + "%"] = vb_by_tier.pop(c) * 100.0

    # ---- Voluntary benefits: the cross-sell target list
    # Employers with an established core relationship and zero voluntary attached.
    vb_whitespace = (
        employers[employers["CoreButNoVoluntary"]]
        [["Employer", "State", "CoveredLives", "TotalPremium", "PremiumPerLife",
          "TotalCommissions", "PrimaryBroker", "BrokerFamily", "BrokerTier",
          "ProductsHeld", "TopCarrier"]]
        .sort_values("TotalPremium", ascending=False)
    )

    # ---- Per-product target lists: who to sell EACH voluntary product to.
    # An employer qualifies for a product when they have an established benefits
    # relationship (core coverage) but do not hold that specific product. Holding
    # some other voluntary product is a positive signal, not a disqualifier -- it
    # proves the group already buys worksite benefits.
    vb_target_cols = ["Employer", "State", "CoveredLives", "TotalPremium", "PremiumPerLife",
                      "TotalCommissions", "PrimaryBroker", "BrokerFamily", "BrokerTier",
                      "TopCarrier", "ProductsHeld", "VBHeld"]
    vb_targets = {}
    has_core = employers["Has_Life"] | employers["Has_STD"] | employers["Has_LTD"]
    for prod in VOLUNTARY_PRODUCTS:
        col = f"Has_{column_suffix(prod)}"
        tgt = employers[has_core & ~employers[col]].copy()
        # Already buys voluntary, just not this one -- the warmest subset.
        tgt["AlreadyBuysVoluntary"] = tgt["Has_AnyVoluntary"]
        vb_targets[prod] = (
            tgt[vb_target_cols + ["AlreadyBuysVoluntary"]]
            .sort_values(["AlreadyBuysVoluntary", "CoveredLives"], ascending=[False, False])
        )

    # ---- Per-product opportunity summary: sizes each product's gap in one view.
    opp_rows = []
    for prod in VOLUNTARY_PRODUCTS:
        col = f"Has_{column_suffix(prod)}"
        tgt = vb_targets[prod]
        warm = tgt[tgt["AlreadyBuysVoluntary"]]
        opp_rows.append({
            "Product": prod,
            "EmployersHolding": int(employers[col].sum()),
            "Penetration%": employers[col].mean() * 100.0,
            "TargetEmployers": len(tgt),
            "TargetLives": tgt["CoveredLives"].sum(),
            "TargetPremiumInPlace": tgt["TotalPremium"].sum(),
            "WarmTargets": len(warm),
            "WarmTargetLives": warm["CoveredLives"].sum(),
            "WarmTargetPremiumInPlace": warm["TotalPremium"].sum(),
            "TargetsOnAONBook": int(tgt["BrokerFamily"].eq("AON").sum()),
            "TargetsOnCompetitorBook": int((~tgt["BrokerFamily"].eq("AON")).sum()),
        })
    vb_opportunity = pd.DataFrame(opp_rows).sort_values("TargetLives", ascending=False)

    # A dedicated target sheet is only worth a tab where enough employers hold the
    # product for the gap to mean anything. Identity theft and pet are counted in
    # every rollup but are too thin to justify a ~50k-row target list, so they are
    # skipped here -- and said out loud rather than silently dropped.
    TARGET_SHEET_MIN_HOLDERS = 500
    target_products = [p for p in VOLUNTARY_PRODUCTS
                       if int(employers[f"Has_{column_suffix(p)}"].sum()) >= TARGET_SHEET_MIN_HOLDERS]
    skipped = [p for p in VOLUNTARY_PRODUCTS if p not in target_products]
    if skipped:
        log(f"no target sheet for {', '.join(skipped)} "
            f"(<{TARGET_SHEET_MIN_HOLDERS} employers hold them; still counted in all rollups)")

    # ---- Voluntary benefits: carrier league table
    vb_carriers = (
        carrier_summary[carrier_summary["Product"].isin(VOLUNTARY_PRODUCTS)]
        .pivot_table(index="Carrier", columns="Product", values="Employers", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    for p in VOLUNTARY_PRODUCTS:
        if p not in vb_carriers.columns:
            vb_carriers[p] = 0
    vb_carriers["TotalVBEmployers"] = vb_carriers[VOLUNTARY_PRODUCTS].sum(axis=1)
    vb_carriers = vb_carriers[["Carrier"] + VOLUNTARY_PRODUCTS + ["TotalVBEmployers"]] \
        .sort_values("TotalVBEmployers", ascending=False)

    # ---- Detail sheets
    detail_broker = (
        ebc[["Employer", "Broker", "total_commissions", "ACK_ID"]]
        .rename(columns={"total_commissions": "CommissionsPaid", "ACK_ID": "FilingID"})
        .assign(BrokerFamily=lambda d: d["Broker"].apply(broker_family))
        .sort_values("CommissionsPaid", ascending=False)
    )
    detail_carrier = (
        epc[["Employer", "Product"] + group_col + ["Carrier", "Covered_Lives", "ACK_ID"]]
        .rename(columns={"Covered_Lives": "CoveredLives", "ACK_ID": "FilingID"})
        .sort_values("CoveredLives", ascending=False)
    )

    return {
        "employers": employers,
        "product_detail": product_detail,
        "premium_by_product": premium_by_product,
        "vb_penetration": vb_penetration,
        "vb_opportunity": vb_opportunity,
        "vb_by_tier": vb_by_tier,
        "vb_whitespace": vb_whitespace,
        "vb_carriers": vb_carriers,
        **{f"vb_target_{column_suffix(p).lower()}": vb_targets[p] for p in target_products},
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
    }, {**dq, "target_products": target_products}


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
                      "Product coverage, covered lives, premium, premium per life, commissions, commission as a "
                      "percent of premium, primary broker + matched family/tier, geography."),
        ("Employer_Product_Detail", f"{counts['product_detail']:,} rows. ONE ROW PER PRODUCT PER EMPLOYER - the "
                                    "line-by-line view. Lives, premium, estimated commission, EIN, incumbent broker "
                                    "and a BrokerStatus column that splits 'AON is broker of record' from "
                                    "'NOT AON - opportunity'. Filter BrokerStatus to see the whitespace. "
                                    "Read the money-column notes in Caveats before quoting these figures."),
        ("Premium_By_Product", f"{counts['premium_by_product']:,} rows. Premium and contract counts per product. "
                               "NOTE: PremiumOnThoseContracts figures overlap and must not be summed - "
                               "SoleProductPremium is the unambiguous single-product figure."),
        ("Broker_Summary", f"{counts['broker_summary']:,} rows. One row per broker (as primary/incumbent) with "
                           "employer count, covered lives, market share and commission stats."),
        ("Broker_Name_Matching", f"{counts['name_map']:,} rows. Audit trail for the company-name matching: every broker "
                                 "name as filed, its normalized form, the family it matched, and which rule fired."),
        ("State_Summary", f"{counts['state_summary']:,} rows. Per-state totals, AON vs competitor share, "
                          "under-index gap and broker fragmentation."),
        ("VB_Penetration", f"{counts['vb_penetration']:,} rows. Voluntary benefit take-up: employers, lives and "
                           "commissions for accident, critical illness, hospital indemnity and cancer."),
        ("VB_Opportunity_By_Product", f"{counts['vb_opportunity']:,} rows. Sizes the sell-in gap for each voluntary "
                                      "product: who holds it today, how many employers are addressable, and how many "
                                      "of those already buy some other voluntary product (the warm subset)."),
    ]
    for prod in dq["target_products"]:
        n = counts[f"vb_target_{column_suffix(prod).lower()}"]
        sheets.append((
            f"Target_{column_suffix(prod)}",
            f"{n:,} rows. Named employers to sell {prod.upper()} to - core coverage in place, no "
            f"{prod.lower()} product. Sorted warm targets (already buy some voluntary) first, then by size.",
        ))
    sheets += [
        ("VB_By_Broker_Tier", f"{counts['vb_by_tier']:,} rows. Voluntary attach rates by broker tier - shows which "
                              "tiers are leaving the voluntary line unsold."),
        ("VB_Crosssell_Targets", f"{counts['vb_whitespace']:,} rows. Employers holding core products (life and/or "
                                 "disability) with NO voluntary benefit attached - the cross-sell list."),
        ("VB_Carriers", f"{counts['vb_carriers']:,} rows. Carrier league table for voluntary products only."),
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
         "Names are uppercased, punctuation stripped, whitespace collapsed. AON is matched as a composite on three "
         "rules, all requiring AON as a WHOLE WORD: (a) the name starts with AON, (b) the name declares itself an AON "
         "subsidiary ('... AN AON COMPANY', '... (AON)'), or (c) it names an AON operating unit after a person or "
         "office prefix. The word boundary is what excludes the real false positives, which contain the letters 'aon' "
         "inside another word: SAMMAONS COMPANY LP, CHERYL LYNN GAONA, DATAONLINE. "
         "Note DOL truncates filed broker names at 35 characters, so subsidiary suffixes arrive mangled ('AN AON "
         "COMP') and the rules tolerate that. "
         "Tier1 majors (Marsh/MMC, WTW/Willis/Towers, Gallagher, Brown & Brown) are matched on word-boundary keywords. "
         "Everything else is OTHER. See the Broker_Name_Matching sheet for the full mapping and which rule fired."),
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
        ("Voluntary benefit derivation (IMPORTANT)",
         "Schedule A has a checkbox for life, STD, LTD, health, dental, vision etc. but NOT for voluntary products. "
         "Critical illness, accident and hospital indemnity are all filed under the OTHER checkbox with a free-text "
         "label (WLFR_TYPE_BNFT_OTH_TEXT), which is why OTHER is the most-used box on the form. Those labels are "
         "comma-separated lists ('ACCIDENT, CRITICAL ILLNESS, HOSPITAL'), so the text is split into individual "
         "benefits before matching, and each product is counted once per filing. "
         "AD&D IS DELIBERATELY EXCLUDED FROM 'ACCIDENT': 'ACCIDENTAL DEATH AND DISMEMBERMENT' appears in ~46,000 "
         "filings, nearly double the entire real voluntary universe, and folding it in would roughly double apparent "
         "market size. It is reported as its own product on VB_Penetration for contrast. Free text that matches no "
         "rule (EAP, telehealth, supplemental life, wellness) is not counted as a product."),
        ("Outlier removal (IMPORTANT)",
         f"Form 5500 is self-reported and a few filings contain keying errors large enough to swamp every total. "
         f"As filed, total commissions came to ${dq['comm_raw_total']:,.0f} - because ONE row reported $563 trillion. "
         f"Rows above ${dq['comm_cap']:,.0f} in commissions ({dq['comm_rows_excluded']} rows) or "
         f"{dq['lives_cap']:,.0f} covered lives ({dq['lives_rows_excluded']} rows, max reported was "
         f"{dq['lives_raw_max']:,.0f} by a small employer) are excluded from all rollups. "
         f"Cleaned total commissions: ${dq['comm_clean_total']:,.0f}. "
         "Every excluded row is listed on the Data_Quality_Flags sheet - re-include them there if you disagree "
         "with the thresholds."),
        ("Premium",
         f"Premium is populated on {dq['premium_fill_pct']:.1f}% of in-scope contracts, totalling "
         f"${dq['premium_clean_total']:,.0f} across {dq['premium_rows_excluded'] + 0} excluded outlier(s) removed "
         f"at a ${dq['premium_cap']:,.0f} per-contract cap. Unlike commissions, premium needed almost no cleaning: "
         "the trillion-dollar keying errors in this file sit on health contracts, which are out of scope here. "
         "It is summed over distinct contracts - see the Caveats section for why that matters."),
    ]
    for name, desc in rules:
        ws.write(r, 0, name, bold); ws.write(r, 1, desc, wrap); r += 1
    r += 1

    ws.write(r, 0, "Caveats", h2); r += 1
    caveats = [
        "Employers are keyed on the sponsor name as filed - separate legal entities of the same parent appear as "
        "separate rows. No parent-company roll-up has been applied.",
        "AON does not sell medical, dental or vision, so those lines are excluded from the pipeline entirely - "
        "their Schedule A checkboxes are never read and the free-text parser drops their labels. In scope are the "
        "core group lines (Life, STD, LTD) and the voluntary products: accident, critical illness, hospital "
        "indemnity, cancer, legal, long term care, identity theft and pet.",
        "AD&D is NOT counted as a voluntary product. It is a life rider that ~87% of groups already carry; counting "
        "it would put voluntary penetration at ~91% and shrink the cross-sell list from ~26,000 employers to ~4,400, "
        "hiding the real opportunity. It is reported separately on VB_Penetration for contrast.",
        "Voluntary products are identified from free text, so coverage depends on how each filer described the plan. "
        "A filer who wrote nothing in the OTHER box will look like they have no voluntary benefit even if they do - "
        "treat the voluntary counts as a floor, not an exact census.",
        "Covered lives cannot be split by benefit within a filing. A Schedule A row listing 'ACCIDENT, CRITICAL "
        "ILLNESS, HOSPITAL' reports one covered-lives figure, so the same population is attributed to all three. "
        "Use it to answer 'who holds this product', not to sum lives across products.",
        "Commissions cannot be attributed to voluntary products specifically - they sit at filing/broker grain with "
        "no benefit dimension. Commission figures on the VB sheets are the employer's TOTAL commissions, not the "
        "voluntary portion. Schedule A does carry a retained-commission field at contract grain, but it is populated "
        "on under 5% of rows and overlaps voluntary filings only ~1,900 times, so it cannot fill the gap either.",
        "PREMIUM is summed over distinct CONTRACTS, never over the product table. A contract covering life + STD + "
        "LTD appears three times in the product view and reports one premium; summing there would treble it. For the "
        "same reason premium cannot be split by product - see the Premium_By_Product sheet, where "
        "'PremiumOnThoseContracts' figures OVERLAP and must not be added down the column. 'SoleProductPremium' is "
        "the only premium unambiguously attributable to one product.",
        "ON THE Employer_Product_Detail SHEET, the money columns are not interchangeable. CoveredLives is real per "
        "product. PremiumOnContracts is real but OVERLAPS across products on a shared contract, so it must not be "
        "summed down the column for one employer. SoleProductPremium is the unambiguous single-product figure. "
        "EmployerCommissions is the employer's TOTAL, repeated on every one of that employer's rows - summing it "
        "multiplies by the product count. EstCommissionForProduct is an ESTIMATE that splits employer commissions by "
        "each product's share of premium; the shares sum to 1 per employer, so the estimates do add back to the "
        "employer total. Use it for relative sizing, not as a reported figure. Where an employer reported "
        "commissions but no premium at all there is nothing to split across, so the estimate is BLANK and "
        "CommissionAllocatable is FALSE - 267 employers and $15.0M of commissions sit in that bucket, which is "
        "exactly how far a sum of EstCommissionForProduct will fall short of total commissions.",
        "Premium is a coalesce of two Schedule A fields: WLFR_PREMIUM_RCVD_AMT (the experience-rated section, ~7% "
        "populated) falling back to WLFR_TOT_CHARGES_PAID_AMT ('total charges paid for this contract', ~78% "
        "populated). Together they cover ~84% of Schedule A rows. Employers with no premium figure are not "
        "premium-free - the filer simply left both boxes empty, so PremiumPerLife is left BLANK rather than zero.",
        "2024 filings only - a plan that did not file in 2024 will not appear.",
        "Commissions are as reported on Schedule A and vary in completeness by filer.",
        "PrimaryBroker = UNKNOWN means the employer has carrier/product coverage but no broker commission record on "
        "Schedule A Part 1 - not that the employer has no broker. Those rows still carry valid covered-lives data.",
    ]
    for c in caveats:
        ws.write(r, 1, "- " + c, wrap); r += 1


def export(out_path: Path, tier2_pct: float, with_detail: bool, comm_cap: float, lives_cap: float,
           premium_cap: float = 500_000_000.0):
    d, dq = build(tier2_pct=tier2_pct, comm_cap=comm_cap, lives_cap=lives_cap, premium_cap=premium_cap)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {k: len(v) for k, v in d.items()}
    log(f"writing {out_path}")

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        write_readme(writer, counts, dq, tier2_pct, with_detail)

        write_sheet(
            writer, d["employers"], "Employers",
            money_cols=("TotalCommissions", "PrimaryBrokerCommissions", "TotalPremium", "PremiumPerLife"),
            int_cols=("CoveredLives", "BrokerCount", "CarrierCount", "ContractCount"),
            pct_cols=("CommissionPctOfPremium",),
        )
        write_sheet(
            writer, d["product_detail"], "Employer_Product_Detail",
            money_cols=("PremiumOnContracts", "PremiumPerLife", "SoleProductPremium",
                        "EstCommissionForProduct", "EmployerCommissions", "EmployerPremium"),
            int_cols=("CoveredLives", "EmployerCoveredLives", "Carriers", "Contracts"),
            pct_cols=("ProductPremiumShare%",),
        )
        write_sheet(
            writer, d["premium_by_product"], "Premium_By_Product",
            money_cols=("PremiumOnThoseContracts", "SoleProductPremium"),
            int_cols=("Employers", "ContractsIncluding", "SoleProductContracts", "CoveredLives"),
        )
        write_sheet(
            writer, d["broker_summary"], "Broker_Summary",
            money_cols=("TotalCommissions", "MedianCommissionPerEmployer", "MeanCommissionPerEmployer",
                        "TotalPremium", "MedianPremiumPerEmployer"),
            int_cols=("Employers", "CoveredLives"),
            pct_cols=("LivesShare%", "EmployerShare%", "CommissionPctOfPremium"),
        )
        write_sheet(
            writer, d["vb_penetration"], "VB_Penetration",
            money_cols=("TotalCommissions", "TotalPremium", "MedianPremium"),
            int_cols=("Employers", "CoveredLives", "MedianLives"),
            pct_cols=("Employers%",),
        )
        write_sheet(
            writer, d["vb_opportunity"], "VB_Opportunity_By_Product",
            money_cols=("TargetPremiumInPlace", "WarmTargetPremiumInPlace"),
            int_cols=("EmployersHolding", "TargetEmployers", "TargetLives", "WarmTargets",
                      "WarmTargetLives", "TargetsOnAONBook", "TargetsOnCompetitorBook"),
            pct_cols=("Penetration%",),
        )
        for prod in dq["target_products"]:
            write_sheet(
                writer, d[f"vb_target_{column_suffix(prod).lower()}"],
                f"Target_{column_suffix(prod)}",
                money_cols=("TotalCommissions", "TotalPremium", "PremiumPerLife"),
                int_cols=("CoveredLives",),
            )
        write_sheet(
            writer, d["vb_by_tier"], "VB_By_Broker_Tier",
            int_cols=("Employers", "CoveredLives"),
            pct_cols=tuple(f"{column_suffix(p)}%" for p in VOLUNTARY_PRODUCTS) + ("AnyVB%", "CoreNoVB%"),
        )
        write_sheet(
            writer, d["vb_whitespace"], "VB_Crosssell_Targets",
            money_cols=("TotalCommissions", "TotalPremium", "PremiumPerLife"),
            int_cols=("CoveredLives",),
        )
        write_sheet(
            writer, d["vb_carriers"], "VB_Carriers",
            int_cols=tuple(VOLUNTARY_PRODUCTS) + ("TotalVBEmployers",),
        )
        write_sheet(
            writer, d["name_map"], "Broker_Name_Matching",
            money_cols=("Commissions",), int_cols=("EmployerFilings",),
        )
        write_sheet(
            writer, d["state_summary"], "State_Summary",
            money_cols=("TotalCommissions", "TotalPremium"),
            int_cols=("Employers", "Brokers", "CoveredLives", "AON_Employers", "AON_Lives"),
            pct_cols=("AONLivesShare%", "CompetitorLivesShare%", "UnderIndexGap%", "MissingAnyProduct%",
                      "AnyVoluntary%"),
            ratio_cols=("BrokerFragRatio",),
        )
        write_sheet(
            writer, d["carrier_summary"], "Carrier_Product_Summary",
            money_cols=("Premium",), int_cols=("Employers", "CoveredLives"),
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
                        money_cols=("Premium",), int_cols=("CoveredLives",))

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
    p.add_argument("--premium-cap", type=float, default=500_000_000.0,
                   help="Drop contracts with premium above this amount as filer errors (default 500,000,000)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    export(Path(a.out).expanduser().resolve(), a.tier2_pct, a.with_detail, a.comm_cap, a.lives_cap,
           a.premium_cap)
