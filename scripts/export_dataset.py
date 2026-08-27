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

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "marts"

sys.path.insert(0, str(REPO_ROOT / "etl"))
from benefits import (
    ALL_PRODUCTS, CORE_PRODUCTS, PRODUCT_GROUP, VB_TRIO, VOLUNTARY_PRODUCTS, column_suffix,
)
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

    # Commission at product grain, via the (ACK_ID, FORM_ID) contract join in the
    # ETL. Exact where a contract lists one product, split evenly where it lists
    # several. Absent from older marts, so default rather than KeyError.
    HAS_PRODUCT_COMM = "ProductCommission" in epc.columns
    if HAS_PRODUCT_COMM:
        epc["ProductCommission"] = to_numeric(epc["ProductCommission"]).fillna(0)
        epc["ContractCommission"] = to_numeric(epc["ContractCommission"]).fillna(0)
        contracts["Commission"] = to_numeric(contracts.get("Commission", 0)).fillna(0)
    else:
        log("WARNING: marts predate contract-level commission - per-product commission "
            "will be absent. Re-run etl/build_marts.py.")
        epc["ProductCommission"] = 0.0
        epc["ContractCommission"] = 0.0
        epc["CommissionIsExact"] = False
        contracts["Commission"] = 0.0

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

    # The same keying errors reach product grain through the contract join, so the
    # commission cap applies there too - otherwise the $563T row lands on whatever
    # product its contract happens to list.
    _bad_contract_comm = int((epc["ContractCommission"] > comm_cap).sum())
    epc.loc[epc["ContractCommission"] > comm_cap, ["ProductCommission", "ContractCommission"]] = 0.0
    contracts.loc[contracts["Commission"] > comm_cap, "Commission"] = 0.0
    if _bad_contract_comm:
        log(f"data-quality: zeroed product commission on {_bad_contract_comm} row(s) "
            f"whose contract commission exceeded ${comm_cap:,.0f}")

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
    # Real commission per product, summed over that employer's contracts.
    prod_comm = (
        epc.drop_duplicates(["Employer", "Product", "ContractRowID"])
           .groupby(["Employer", "Product"], as_index=False)
           .agg(ProductCommission=("ProductCommission", "sum"),
                ExactCommission=("ProductCommission",
                                 lambda s: 0.0))  # replaced below; placeholder keeps column order
    )
    _exact_only = (
        epc[epc["CommissionIsExact"]]
        .drop_duplicates(["Employer", "Product", "ContractRowID"])
        .groupby(["Employer", "Product"], as_index=False)
        .agg(ExactCommission=("ProductCommission", "sum"))
    )
    prod_comm = prod_comm.drop(columns=["ExactCommission"]).merge(
        _exact_only, on=["Employer", "Product"], how="left")
    prod_comm["ExactCommission"] = prod_comm["ExactCommission"].fillna(0)
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
        .merge(prod_comm, on=["Employer", "Product"], how="left")
        .merge(prod_top_carrier, on=["Employer", "Product"], how="left")
        .merge(employers[emp_cols].rename(columns={
            "TotalCommissions": "EmployerCommissions",
            "TotalPremium": "EmployerPremium",
            "CoveredLives": "EmployerCoveredLives",
        }), on="Employer", how="left")
    )
    for c in ["PremiumOnContracts", "SoleProductPremium", "EmployerCommissions",
              "ProductCommission", "ExactCommission"]:
        product_detail[c] = product_detail[c].fillna(0)

    product_detail["ProductPremiumShare%"] = (
        product_detail["PremiumOnContracts"]
        / product_detail.groupby("Employer")["PremiumOnContracts"].transform("sum").replace(0, pd.NA)
        * 100.0
    )
    # How much of this product's commission is reported rather than split. 100%
    # means every contract behind it listed only this product.
    product_detail["CommissionExact%"] = (
        product_detail["ExactCommission"]
        / product_detail["ProductCommission"].replace(0, pd.NA) * 100.0
    )

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
        "CoveredLives", "ProductCommission", "ExactCommission", "CommissionExact%",
        "PremiumOnContracts", "PremiumPerLife", "SoleProductPremium", "ProductPremiumShare%",
        "BrokerStatus", "AON_Is_Broker", "PrimaryBroker", "BrokerFamily", "BrokerTier",
        "EmployerCommissions", "EmployerPremium", "EmployerCoveredLives",
        "TopCarrier", "Carriers", "Contracts",
    ]].sort_values(["ProductCommission", "Employer", "Product"], ascending=[False, True, True])

    _pc = product_detail["ProductCommission"]
    log(f"product detail: {len(product_detail):,} employer-product rows "
        f"({int((~product_detail['AON_Is_Broker']).sum()):,} not on the AON book)")
    log(f"  commission by product: ${_pc.sum():,.0f} total, "
        f"${product_detail['ExactCommission'].sum():,.0f} of it reported exactly "
        f"({product_detail['ExactCommission'].sum() / _pc.sum() * 100:.1f}%)")

    # ---- Company x product matrix: ONE ROW PER COMPANY, one column per product.
    # The long detail table only has rows for products a company HOLDS, so a gap
    # is invisible there - the row simply is not present. Pivoting makes the gap
    # explicit: a blank cell means that product was never sold to that company.
    wide_comm = (
        product_detail.pivot_table(index="Employer", columns="Product",
                                   values="ProductCommission", aggfunc="sum")
        .reindex(columns=ALL_PRODUCTS)
    )
    wide_comm.columns = [f"Comm_{column_suffix(c)}" for c in wide_comm.columns]
    wide_lives = (
        product_detail.pivot_table(index="Employer", columns="Product",
                                   values="CoveredLives", aggfunc="max")
        .reindex(columns=ALL_PRODUCTS)
    )

    # ---- Sizing the gap: what would a missing product be worth on this account?
    #
    # Two corrections matter here, and getting either wrong inflates the answer
    # several times over:
    #
    # 1. Benchmark ONLY from single-product contracts. On a bundled contract the
    #    premium and commission cover every product on it, mostly life and
    #    disability, so deriving a voluntary rate from bundled rows attributes
    #    core-line money to the voluntary line. Measured on this file, critical
    #    illness looks like $43.51 per life on clean contracts against $23.65 from
    #    bundled ones - and the bundled figure is the WRONG one despite looking
    #    conservative, because its denominator is the whole group.
    #
    # 2. Apply participation. Voluntary products are employee-paid and elective,
    #    so only part of a group takes them up: measured medians here run from
    #    0.99 for accident down to 0.11 for long term care. Multiplying a whole
    #    group's lives by a per-participant rate assumes universal take-up.
    #
    # Commission per life is banded by group size because small groups pay
    # materially more per life; medians throughout so one contract cannot move a
    # band.
    BANDS = [(0, 100), (100, 500), (500, 1_000), (1_000, 5_000),
             (5_000, 20_000), (20_000, float("inf"))]

    def band_of(lives):
        for i, (lo, hi) in enumerate(BANDS):
            if lo <= lives < hi:
                return i
        return len(BANDS) - 1

    bench_src = product_detail[
        (product_detail["ExactCommission"] > 0) & (product_detail["CoveredLives"] > 0)
    ].copy()
    bench_src["_band"] = bench_src["CoveredLives"].map(band_of)
    bench_src["_cpl"] = bench_src["ExactCommission"] / bench_src["CoveredLives"]

    # Commission per life falls steeply with group size - accident runs $30.49 for
    # groups under 100 against $0.61 above 20,000, a 50x spread - so a thin band
    # must fall back to the NEAREST band, never to the overall median. The overall
    # median is dominated by small groups, and using it for a large employer
    # overstates the gap by an order of magnitude.
    MIN_BENCH_N = 30
    MIN_HOLDERS_TO_MODEL = 200

    _banded = bench_src.groupby(["Product", "_band"]).agg(cpl=("_cpl", "median"), n=("Employer", "size"))
    solid = {(p, int(b)): r["cpl"] for (p, b), r in _banded.iterrows() if r["n"] >= MIN_BENCH_N}
    counts = {(p, int(b)): int(r["n"]) for (p, b), r in _banded.iterrows()}

    def nearest_band_cpl(prod, band):
        """Walk outward to the closest band that has enough observations."""
        for dist in range(len(BANDS)):
            for cand in (band - dist, band + dist):
                if 0 <= cand < len(BANDS) and (prod, cand) in solid:
                    return solid[(prod, cand)], (dist == 0)
        return np.nan, False

    # A product almost nobody buys cannot be benchmarked - Pet and Identity Theft
    # have a single clean contract each, and extrapolating from one contract made
    # Pet the "biggest gap" for 47,195 companies in an earlier build.
    holders = product_detail.groupby("Product")["Employer"].nunique()
    modellable = {p for p in ALL_PRODUCTS if holders.get(p, 0) >= MIN_HOLDERS_TO_MODEL}
    _skipped = [p for p in ALL_PRODUCTS if p not in modellable]
    if _skipped:
        log(f"opportunity not modelled for {', '.join(_skipped)} "
            f"(<{MIN_HOLDERS_TO_MODEL} holders - too thin to benchmark)")

    # Participation = this product's covered lives against the group's core lives.
    core_lives = (
        product_detail[product_detail["Product"].isin(CORE_PRODUCTS)]
        .groupby("Employer")["CoveredLives"].max()
    )
    participation = {}
    for prod in ALL_PRODUCTS:
        sub = (product_detail[product_detail["Product"] == prod]
               .set_index("Employer")["CoveredLives"])
        j = pd.concat([sub.rename("p"), core_lives.rename("c")], axis=1).dropna()
        j = j[(j["c"] > 0) & (j["p"] > 0)]
        participation[prod] = float((j["p"] / j["c"]).clip(upper=1.0).median()) if len(j) >= 20 else 1.0

    emp_core = employers.set_index("Employer")["CoveredLives"]
    emp_band = emp_core.map(band_of)

    opp = pd.DataFrame(index=employers["Employer"])
    _band_exact = pd.DataFrame(index=employers["Employer"])
    for prod in ALL_PRODUCTS:
        col = f"Opp_{column_suffix(prod)}"
        if prod not in modellable:
            opp[col] = np.nan
            continue
        held = wide_comm.get(f"Comm_{column_suffix(prod)}")
        held = held.reindex(opp.index) if held is not None else pd.Series(index=opp.index, dtype=float)
        _lookup = {b: nearest_band_cpl(prod, b) for b in range(len(BANDS))}
        cpl = emp_band.map(lambda b: _lookup[b][0])
        _band_exact[col] = emp_band.map(lambda b: _lookup[b][1])
        est = emp_core * participation[prod] * cpl
        # Only a gap is an opportunity: blank out anything already sold.
        opp[col] = est.where(held.isna() & (emp_core > 0))

    VB_OPP_COLS = [f"Opp_{column_suffix(p)}" for p in VOLUNTARY_PRODUCTS if p in modellable]
    opp["VoluntaryOpportunity"] = opp[VB_OPP_COLS].sum(axis=1, min_count=1)
    opp["TotalOpportunity"] = opp[[c for c in opp.columns if c.startswith("Opp_")]].sum(axis=1, min_count=1)
    opp["BiggestGap"] = (
        opp[VB_OPP_COLS].idxmax(axis=1).str.replace("Opp_", "", regex=False)
        .where(opp[VB_OPP_COLS].notna().any(axis=1))
    )
    # Whether the benchmark came from this company's own size band or had to be
    # borrowed from a neighbouring one. Large employers mostly land on "Indicative"
    # because few groups above 20,000 lives buy voluntary on a standalone contract.
    _exact_share = _band_exact[[c for c in VB_OPP_COLS if c in _band_exact.columns]]
    opp["OpportunityConfidence"] = np.where(
        _exact_share.mean(axis=1) >= 0.5, "Benchmarked in own size band", "Indicative - borrowed band")

    key_cols = ["Employer", "EIN", "State", "CoveredLives", "PrimaryBroker", "BrokerFamily",
                "BrokerTier", "TotalCommissions", "TotalPremium"]
    company_matrix = (
        employers[key_cols]
        .merge(wide_comm.reset_index(), on="Employer", how="left")
        .merge(opp.reset_index(), on="Employer", how="left")
    )
    company_matrix["AON_Is_Broker"] = company_matrix["BrokerFamily"].eq("AON")
    company_matrix["BrokerStatus"] = np.where(
        company_matrix["AON_Is_Broker"], "AON is broker of record", "NOT AON - opportunity")
    comm_cols = [f"Comm_{column_suffix(p)}" for p in ALL_PRODUCTS]
    company_matrix["ProductsHeld"] = company_matrix[comm_cols].notna().sum(axis=1)
    company_matrix["CommissionTotal"] = company_matrix[comm_cols].sum(axis=1, min_count=1)

    company_matrix = company_matrix[
        key_cols[:3] + ["BrokerStatus", "PrimaryBroker", "BrokerFamily", "BrokerTier",
                        "CoveredLives", "ProductsHeld", "CommissionTotal"]
        + comm_cols
        + ["VoluntaryOpportunity", "BiggestGap", "OpportunityConfidence", "TotalOpportunity"]
        + [f"Opp_{column_suffix(p)}" for p in ALL_PRODUCTS]
        + ["AON_Is_Broker"]
    ].sort_values("VoluntaryOpportunity", ascending=False)

    log(f"company matrix: {len(company_matrix):,} companies, "
        f"voluntary opportunity ${company_matrix['VoluntaryOpportunity'].sum():,.0f} "
        f"(${company_matrix.loc[~company_matrix['AON_Is_Broker'], 'VoluntaryOpportunity'].sum():,.0f} "
        "on non-AON accounts)")

    # ---- Commission by product: the headline "what is each line worth" table.
    commission_by_product = (
        product_detail.groupby(["ProductGroup", "Product"], as_index=False)
        .agg(Employers=("Employer", "nunique"),
             CoveredLives=("CoveredLives", "sum"),
             Contracts=("Contracts", "sum"),
             Commission=("ProductCommission", "sum"),
             ExactCommission=("ExactCommission", "sum"),
             Premium=("PremiumOnContracts", "sum"))
    )
    commission_by_product["SplitCommission"] = (
        commission_by_product["Commission"] - commission_by_product["ExactCommission"]
    )
    commission_by_product["Exact%"] = (
        commission_by_product["ExactCommission"]
        / commission_by_product["Commission"].replace(0, pd.NA) * 100.0
    )
    commission_by_product["CommissionRate%"] = (
        commission_by_product["Commission"]
        / commission_by_product["Premium"].replace(0, pd.NA) * 100.0
    )
    commission_by_product = commission_by_product.sort_values("Commission", ascending=False)

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
        "company_matrix": company_matrix,
        "product_detail": product_detail,
        "commission_by_product": commission_by_product,
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
    ws.write(r, 1, "DOL Form 5500 + Schedule A (F_5500, F_SCH_A, F_SCH_A_PART1), 2024 'Latest' release", wrap); r += 1
    ws.write(r, 0, "Period covered", bold)
    ws.write(r, 1, "PLAN YEAR 2024. 96.2% of filings have a plan year beginning in 2024; the rest are late or "
                   "amended filings for earlier years. These were SUBMITTED to DOL during 2025 (97.9% received "
                   "in calendar 2025, the remainder by January 2026), which is why the coverage year and the "
                   "filing year differ. 'Latest' means the most recent version of each filing, so amendments "
                   "supersede originals.", wrap); r += 1
    ws.write(r, 0, "Why not 2025?", bold)
    ws.write(r, 1, "Plan year 2025 exists but is only about 26% filed as of August 2026 (57,161 filings against "
                   "223,028 for a complete year). Only ~36% of a year files by the on-time deadline and the "
                   "mid-October extension deadline carries ~37% more, so 2025 is not comparable to a complete year "
                   "until around November 2026. The gap is also biased: large complex plans take extensions, so an "
                   "early pull holds 40% of filings but only 34% of covered lives. Trending 2024 against a partial "
                   "2025 would show a false decline concentrated in the largest accounts.", wrap); r += 2

    ws.write(r, 0, "Sheets", h2); r += 1
    sheets = [
        ("Employers", f"{counts['employers']:,} rows. One row per employer - the master cleaned table. "
                      "Product coverage, covered lives, premium, premium per life, commissions, commission as a "
                      "percent of premium, primary broker + matched family/tier, geography."),
        ("Company_Product_Matrix", f"{counts['company_matrix']:,} rows. ONE ROW PER COMPANY, one column per product. "
                                   "Comm_* holds the commission that product actually earned on that account; a BLANK "
                                   "means the product was never sold to them. Opp_* is the mirror image - a modelled "
                                   "estimate of what the gap would be worth, blank where the product is already held. "
                                   "VoluntaryOpportunity totals the voluntary gaps, BiggestGap names the largest one. "
                                   "Sort by VoluntaryOpportunity and filter BrokerStatus to 'NOT AON' for the "
                                   "target list. Opp_* figures are MODELLED, not reported - see Caveats."),
        ("Employer_Product_Detail", f"{counts['product_detail']:,} rows. ONE ROW PER PRODUCT PER EMPLOYER - the "
                                    "line-by-line view. Commission, lives, premium, EIN, incumbent broker and a "
                                    "BrokerStatus column that splits 'AON is broker of record' from "
                                    "'NOT AON - opportunity'. Filter BrokerStatus to see the whitespace. "
                                    "CommissionExact% tells you how much of each row's commission was reported "
                                    "against that product alone rather than split within a bundled contract."),
        ("Commission_By_Product", f"{counts['commission_by_product']:,} rows. Commission, premium and the implied "
                                  "commission rate for each product. Exact% is the share reported against a "
                                  "single-product contract; the remainder is split evenly inside bundled contracts."),
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
        "COMMISSION IS NOW REPORTED PER PRODUCT. Schedule A Part 1 carries FORM_ID alongside ACK_ID, and "
        "(ACK_ID, FORM_ID) identifies a single Schedule A contract, so every broker commission row resolves to one "
        "contract and therefore to that contract's products - all 380,007 Part 1 rows join, 100%. Where a contract "
        "lists one product the commission is exact. Where it bundles several, the contract reports a single figure "
        "with no per-benefit breakdown, so it is split evenly across them; CommissionExact% and the Exact% column on "
        "Commission_By_Product show how much of any figure is reported versus split. Note the VB_* sheets still "
        "carry EMPLOYER-level commissions - use Employer_Product_Detail for per-product money.",
        "PREMIUM is summed over distinct CONTRACTS, never over the product table. A contract covering life + STD + "
        "LTD appears three times in the product view and reports one premium; summing there would treble it. For the "
        "same reason premium cannot be split by product - see the Premium_By_Product sheet, where "
        "'PremiumOnThoseContracts' figures OVERLAP and must not be added down the column. 'SoleProductPremium' is "
        "the only premium unambiguously attributable to one product.",
        "THE Opp_* COLUMNS ON Company_Product_Matrix ARE MODELLED, NOT REPORTED. Nothing in Form 5500 says what an "
        "unsold product would have earned. Each estimate is the company's core covered lives x that product's median "
        "participation rate x median commission per participating life, benchmarked among companies of similar size "
        "(six bands, since small groups pay materially more per life; bands under 30 observations fall back to the "
        "product's overall median). Benchmarks come ONLY from single-product contracts, where the commission is "
        "unambiguously that product's - deriving them from bundled contracts attributes life and disability money to "
        "the voluntary line and roughly doubles every figure.",
        "USE THE Opp_* COLUMNS TO RANK ACCOUNTS, NOT TO FORECAST REVENUE. Summed across all companies they describe "
        "a fully-saturated market in which every employer buys every product - roughly 6x today's voluntary "
        "commission - which is a ceiling, not a pipeline. The per-account figures are what matter: a $50k gap and a "
        "$500 gap are reliably different, and that ordering is the point. Real pricing depends on industry, "
        "demographics, take-up and plan design, none of which are in this data. Blank where a company reports no "
        "covered lives, or where the product is already held.",
        "ON THE Employer_Product_Detail SHEET, the money columns are not interchangeable. ProductCommission and "
        "CoveredLives are safe to sum down the column - both are real per product. PremiumOnContracts is real but "
        "OVERLAPS across products on a bundled contract, so it must not be summed for a single employer; "
        "SoleProductPremium is the unambiguous single-product figure. EmployerCommissions and EmployerPremium are "
        "employer TOTALS repeated on every one of that employer's rows - summing either multiplies by the product "
        "count. They are there for reconciliation, not aggregation.",
        "Premium is a coalesce of two Schedule A fields: WLFR_PREMIUM_RCVD_AMT (the experience-rated section, ~7% "
        "populated) falling back to WLFR_TOT_CHARGES_PAID_AMT ('total charges paid for this contract', ~78% "
        "populated). Together they cover ~84% of Schedule A rows. Employers with no premium figure are not "
        "premium-free - the filer simply left both boxes empty, so PremiumPerLife is left BLANK rather than zero.",
        "Plan year 2024 only - a plan that did not file for 2024 will not appear. Note the filings themselves were "
        "submitted during 2025, so this is the most recent complete year available, not a 2025 view of the market.",
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
            writer, d["company_matrix"], "Company_Product_Matrix",
            money_cols=tuple([f"Comm_{column_suffix(p)}" for p in ALL_PRODUCTS]
                             + [f"Opp_{column_suffix(p)}" for p in ALL_PRODUCTS]
                             + ["CommissionTotal", "VoluntaryOpportunity", "TotalOpportunity"]),
            int_cols=("CoveredLives", "ProductsHeld"),
        )
        write_sheet(
            writer, d["product_detail"], "Employer_Product_Detail",
            money_cols=("ProductCommission", "ExactCommission", "PremiumOnContracts",
                        "PremiumPerLife", "SoleProductPremium", "EmployerCommissions",
                        "EmployerPremium"),
            int_cols=("CoveredLives", "EmployerCoveredLives", "Carriers", "Contracts"),
            pct_cols=("ProductPremiumShare%", "CommissionExact%"),
        )
        write_sheet(
            writer, d["commission_by_product"], "Commission_By_Product",
            money_cols=("Commission", "ExactCommission", "SplitCommission", "Premium"),
            int_cols=("Employers", "CoveredLives", "Contracts"),
            pct_cols=("Exact%", "CommissionRate%"),
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
