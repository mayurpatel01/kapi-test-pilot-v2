# app.py
# Voluntary Benefits Intelligence Insights (Form 5500)
# AON vs Competitors: Market Share + Diagnostics + Opportunity Scoring + AI Q&A
#
# Data marts expected in data/marts:
# - employer_product_carrier.parquet: ['ACK_ID','Carrier','Covered_Lives','Product','Employer_ID','Employer']
# - employer_broker_commissions.parquet: ['ACK_ID','Broker','total_commissions','Employer_ID','Employer']
# - employer_product_matrix.parquet: ['Employer','Life','STD','LTD','Accident','Critical Illness',
#     'Hospital Indemnity','Cancer','AD&D','Long Term Care','Legal','Identity Theft','Pet']
# - employer_geo.parquet (optional, downloaded via EMPLOYER_GEO_URL): ['Employer','State','ZIP','City','EIN'] (we use State/City)
#
# Key assumptions:
# - Covered lives per Employer = max(Covered_Lives) to avoid double-counting across Product/Carrier rows
# - Commission comparisons default to median per employer, with toggle mean vs median + histograms (log)
# - States are sponsor HQ (Form 5500 sponsor location), not employee residence

import os
import re
import sys
import math
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent / "etl"))
from benefits import ALL_PRODUCTS, CORE_PRODUCTS, VB_TRIO, VOLUNTARY_PRODUCTS  # noqa: E402
from quality import flag_contracts, summarise as summarise_quality  # noqa: E402
from brokers import (  # noqa: E402
    TIER1_PATTERNS, assign_tiers, broker_family, is_aon_composite, match_any, norm,
)


# =========================
# Page config / Branding
# =========================
st.set_page_config(
    page_title="Voluntary Benefits Intelligence Insights (Form 5500)",
    page_icon="📍",
    layout="wide",
)

st.title("Voluntary Benefits Intelligence Insights (Form 5500)")
st.caption(
    "AON vs competitors decision engine: market share, whitespace, robust diagnostics, target scoring "
    "and per-product commissions from DOL Form 5500. Pick a plan year in the sidebar — DOL names each "
    "release for the plan year, and filings arrive over roughly the following two years."
)

DATA_DIR = Path("data/marts")


# =========================
# Auth
# =========================
def require_password():
    try:
        pw = st.secrets.get("APP_PASSWORD", "")
    except FileNotFoundError:
        pw = os.getenv("APP_PASSWORD", "")

    if not pw:
        st.error("APP_PASSWORD not set. Add it to Streamlit Secrets (or env var APP_PASSWORD).")
        st.stop()

    if "authed" not in st.session_state:
        st.session_state.authed = False

    if st.session_state.authed:
        return

    with st.sidebar:
        st.markdown("### Sign in")
        entered = st.text_input("Password", type="password")
        if st.button("Sign in"):
            if entered == pw:
                st.session_state.authed = True
                st.success("Signed in.")
            else:
                st.error("Incorrect password.")

    if not st.session_state.authed:
        st.stop()


require_password()


# =========================
# Helpers
# =========================
def ensure_str(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str)
    return out


def ensure_mart_exists(filename: str) -> None:
    """
    For Cloud: download missing marts that are excluded from git.
    """
    path = DATA_DIR / filename
    if path.exists():
        return

    if filename == "employer_geo.parquet":
        try:
            url = st.secrets.get("EMPLOYER_GEO_URL", "")
        except FileNotFoundError:
            url = os.getenv("EMPLOYER_GEO_URL", "")

        if not url:
            st.warning(
                "Geo mart missing (employer_geo.parquet) and EMPLOYER_GEO_URL not set. "
                "State/City charts will be disabled until provided."
            )
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("Downloading geo mart..."):
            urllib.request.urlretrieve(url, path)


@st.cache_data(show_spinner=False)
def available_years() -> list:
    """Plan years with a built mart, newest first. Empty means the older
    single-directory layout is in use."""
    if not DATA_DIR.exists():
        return []
    years = []
    for p in DATA_DIR.iterdir():
        if p.is_dir() and p.name.isdigit() and (p / "employer_product_carrier.parquet").exists():
            years.append(int(p.name))
    return sorted(years, reverse=True)


@st.cache_data(show_spinner=False)
def load_parquet(filename: str, year: int | None = None) -> pd.DataFrame:
    """Load a mart, from the year partition when one is selected."""
    if year is not None:
        path = DATA_DIR / str(year) / filename
        if path.exists():
            return pd.read_parquet(path)
        # employer_geo and other shared marts may only exist at the root.
        root = DATA_DIR / filename
        if root.exists():
            return pd.read_parquet(root)
        raise FileNotFoundError(f"Missing mart: {path}")

    ensure_mart_exists(filename)
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing mart: {path}")
    return pd.read_parquet(path)


# norm() is imported from etl/brokers.py -- see the import block at the top.


def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def as_flag(series: pd.Series) -> pd.Series:
    s = series.fillna(0)
    if s.dtype == "object":
        s = s.astype(str).str.strip()
        return s.isin(["1", "TRUE", "True", "true", "Y", "YES", "Yes", "yes"])
    return s.astype(int).fillna(0) == 1


# =========================
# State normalization (avoid bogus "55 states")
# =========================
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

def normalize_state(x: str) -> str:
    s = "" if x is None else str(x).strip().upper()
    s = s.replace(".", "")
    if s in VALID_STATES:
        return s
    return STATE_NAME_TO_ABBR.get(s, s)


# =========================
# Load marts
# =========================
# ---- Plan year selection.
# Marts are partitioned by plan year so only the year being viewed is held in
# memory. A DOL release is filed over about two years, so the most recent year
# present is normally still accumulating and must not be compared to a complete
# one -- the default lands on the newest COMPLETE year rather than the newest.
YEARS = available_years()
SELECTED_YEAR = None
INCOMPLETE_YEARS = set()

if YEARS:
    try:
        _trend = load_parquet("trend_summary.parquet")
        _emp_by_year = _trend.groupby("PlanYear")["TotalEmployers"].first()
        _peak = _emp_by_year.max()
        # Under 70% of the largest year means the filing cycle is still open.
        INCOMPLETE_YEARS = set(_emp_by_year[_emp_by_year < _peak * 0.7].index.astype(int))
    except FileNotFoundError:
        _trend = None

    _complete = [y for y in YEARS if y not in INCOMPLETE_YEARS]
    _default = _complete[0] if _complete else YEARS[0]

    with st.sidebar:
        st.markdown("## Plan year")
        SELECTED_YEAR = st.selectbox(
            "Plan year",
            options=YEARS,
            index=YEARS.index(_default),
            format_func=lambda y: f"{y}  (partial)" if y in INCOMPLETE_YEARS else str(y),
            help="DOL names each release for the plan year, not the filing year. "
                 "Filings arrive over roughly two years, so the newest year is "
                 "incomplete until its October extension deadline has passed.",
        )
        if SELECTED_YEAR in INCOMPLETE_YEARS:
            st.warning(
                f"Plan year {SELECTED_YEAR} is still being filed. Large, complex plans "
                "take extensions, so what is here under-represents big employers. "
                "Do not compare it to a complete year."
            )

with st.spinner("Loading data marts..."):
    epc = ensure_str(load_parquet("employer_product_carrier.parquet", SELECTED_YEAR))
    ebc = ensure_str(load_parquet("employer_broker_commissions.parquet", SELECTED_YEAR))
    gaps = ensure_str(load_parquet("employer_product_matrix.parquet", SELECTED_YEAR))

    # Contract-level mart. Premium MUST be summed from here rather than from epc:
    # a contract covering life + STD + LTD appears three times in the product view
    # and carries one premium, so summing there inflates it ~2.5x.
    try:
        contracts = ensure_str(load_parquet("employer_contract.parquet", SELECTED_YEAR))
    except FileNotFoundError:
        contracts = pd.DataFrame(columns=["ContractRowID", "Employer", "Carrier", "Premium"])

    try:
        geo = ensure_str(load_parquet("employer_geo.parquet", SELECTED_YEAR))
    except FileNotFoundError:
        geo = pd.DataFrame(columns=["Employer", "State", "City", "ZIP", "EIN"])


# =========================
# Validate expected columns
# =========================
required_epc = {"Employer", "Product", "Carrier", "Covered_Lives"}
required_ebc = {"Employer", "Broker", "total_commissions"}
required_gaps = {"Employer", "Life", "STD", "LTD"}

missing = []
if not required_epc.issubset(set(epc.columns)): missing.append(f"EPC missing: {sorted(list(required_epc - set(epc.columns)))}")
if not required_ebc.issubset(set(ebc.columns)): missing.append(f"EBC missing: {sorted(list(required_ebc - set(ebc.columns)))}")
if not required_gaps.issubset(set(gaps.columns)): missing.append(f"GAPS missing: {sorted(list(required_gaps - set(gaps.columns)))}")
if missing:
    st.error("Schema mismatch:\n- " + "\n- ".join(missing))
    st.stop()


# =========================
# Normalize numeric fields
# =========================
epc["Covered_Lives"] = to_numeric(epc["Covered_Lives"]).fillna(0)
epc["Premium"] = to_numeric(epc["Premium"]).fillna(0) if "Premium" in epc.columns else 0.0
# ProductPremium is the contract premium divided across the products on that
# contract, so it can be summed. Premium repeats the whole contract's figure on
# every product row. Older marts lack it, so fall back rather than KeyError.
if "ProductPremium" not in epc.columns:
    epc["ProductPremium"] = epc["Premium"] / epc.get("ProductsOnContract", 1)
epc["ProductPremium"] = to_numeric(epc["ProductPremium"]).fillna(0)
for _c in ("ProductCommission", "ContractCommission"):
    epc[_c] = to_numeric(epc[_c]).fillna(0) if _c in epc.columns else 0.0
if "CommissionIsExact" not in epc.columns:
    epc["CommissionIsExact"] = False
ebc["total_commissions"] = to_numeric(ebc["total_commissions"]).fillna(0)
if len(contracts) and "Premium" in contracts.columns:
    contracts["Premium"] = to_numeric(contracts["Premium"]).fillna(0)


# =========================
# Data-quality guard
# =========================
# Form 5500 is self-reported and a few filings carry keying errors large enough
# to swamp every total: one row reports $563 TRILLION in commissions, and two
# small employers report 5-6M covered lives. The Excel export has always capped
# these; the dashboard did not, so every commission figure here silently carried
# the $563T row. Same thresholds as scripts/export_dataset.py so the two agree.
COMM_CAP = 10_000_000.0
LIVES_CAP = 1_500_000.0
PREMIUM_CAP = 500_000_000.0

_comm_raw = float(ebc["total_commissions"].sum())
_lives_raw_max = float(epc["Covered_Lives"].max()) if len(epc) else 0.0
_prem_raw = float(contracts["Premium"].sum()) if len(contracts) else 0.0

# Keep the excluded rows rather than just counting them - the Data Quality tab
# lists every one with its reported value, so nothing disappears silently.
DQ_ROWS = []


def _flag(df, field, value_col, cap, counterparty=None, reason=None):
    bad = df[df[value_col] > cap]
    if not len(bad):
        return
    out = pd.DataFrame({
        "Employer": bad.get("Employer", pd.Series(index=bad.index, dtype=object)),
        "EIN": bad.get("EIN", pd.Series(index=bad.index, dtype=object)),
        "Field": field,
        "Counterparty": counterparty(bad) if counterparty else "",
        "ReportedValue": bad[value_col],
        "Cap": cap,
        "Reason": reason or f"Above the plausibility cap of {cap:,.0f}",
    })
    DQ_ROWS.append(out)


_flag(ebc, "Commissions paid", "total_commissions", COMM_CAP,
      counterparty=lambda d: d["Broker"],
      reason=f"Employer-broker commission above ${COMM_CAP:,.0f}. Form 5500 is self-reported "
             "and a single keying error here swamps every total in the app.")
_flag(epc, "Covered lives", "Covered_Lives", LIVES_CAP,
      counterparty=lambda d: d["Carrier"] + " (" + d["Product"] + ")",
      reason=f"More than {LIVES_CAP:,.0f} covered lives on one employer-carrier row.")
if len(contracts):
    _flag(contracts, "Premium", "Premium", PREMIUM_CAP,
          counterparty=lambda d: d["Carrier"] + " (" + d.get("Products", "") + ")",
          reason=f"Single contract premium above ${PREMIUM_CAP:,.0f}.")
_flag(epc.drop_duplicates("ContractRowID"), "Contract commission", "ContractCommission", COMM_CAP,
      counterparty=lambda d: d["Carrier"],
      reason=f"Contract commission above ${COMM_CAP:,.0f}. Zeroed at product grain so the "
             "error cannot land on whichever products that contract happens to list.")

_n_comm = int((ebc["total_commissions"] > COMM_CAP).sum())
_n_lives = int((epc["Covered_Lives"] > LIVES_CAP).sum())
_n_prem = int((contracts["Premium"] > PREMIUM_CAP).sum()) if len(contracts) else 0
_n_ccomm = int((epc.drop_duplicates("ContractRowID")["ContractCommission"] > COMM_CAP).sum())

ebc = ebc[ebc["total_commissions"] <= COMM_CAP].copy()
epc = epc[epc["Covered_Lives"] <= LIVES_CAP].copy()
if len(contracts):
    contracts = contracts[contracts["Premium"] <= PREMIUM_CAP].copy()

# The same keying errors reach product grain through the contract join, so the
# cap has to be applied there too. The export has always done this; the app did
# not, which left a $6.9 BILLION commission sitting on a 170-life employer.
_over = epc["ContractCommission"] > COMM_CAP
epc.loc[_over, ["ProductCommission", "ContractCommission"]] = 0.0
# The contract table needs the same treatment, or the $563 trillion row lands in
# the quality summary and swamps it exactly as it swamped the totals.
if len(contracts) and "Commission" in contracts.columns:
    contracts.loc[contracts["Commission"] > COMM_CAP, "Commission"] = 0.0

DQ_EXCLUDED = (pd.concat(DQ_ROWS, ignore_index=True).sort_values("ReportedValue", ascending=False)
               if DQ_ROWS else pd.DataFrame(columns=["Employer", "EIN", "Field", "Counterparty",
                                                     "ReportedValue", "Cap", "Reason"]))

# Second tier: implausible but not destructive, so these stay in every total and
# are surfaced for a human to judge rather than dropped. Auto-removing them would
# be the worse error - most are real money with one bad field.
if len(contracts):
    DQ_FLAGGED = flag_contracts(contracts)
    DQ_FLAG_SUMMARY = summarise_quality(DQ_FLAGGED)
else:
    DQ_FLAGGED = pd.DataFrame(columns=["Employer", "Carrier", "Covered_Lives", "Premium",
                                       "Commission", "QualityFlag", "AnyQualityFlag"])
    DQ_FLAG_SUMMARY = pd.DataFrame()

DQ_NOTE = (
    f"Excluded {_n_comm} commission row(s) over ${COMM_CAP:,.0f}, {_n_lives} row(s) over "
    f"{LIVES_CAP:,.0f} covered lives, {_n_prem} contract(s) over ${PREMIUM_CAP:,.0f} premium, "
    f"and zeroed product commission on {_n_ccomm} contract(s) over ${COMM_CAP:,.0f}. "
    f"Commissions as filed totalled ${_comm_raw:,.0f} because one row reports $563 trillion; "
    f"cleaned they total ${ebc['total_commissions'].sum():,.0f}."
)

# Normalize products a bit (defensive)
epc["ProductNorm"] = epc["Product"].apply(norm)


# =========================
# AON Composite + Competitor Tiers
# =========================
# Matching rules live in etl/brokers.py, shared with scripts/export_dataset.py
# so the dashboard and the exported workbook cannot disagree. AON is matched on
# AON as a whole word (prefix, declared subsidiary, or named operating unit),
# which keeps out SAMMAONS / GAONA / DATAONLINE while catching the ~97 AON
# entities the old prefix-only rule scored as competitors.


# =========================
# Build Employer-level base tables
# =========================
# Covered lives per Employer: MAX (avoid double counting)
emp_lives = (
    epc.groupby("Employer", as_index=False)
       .agg(CoveredLives=("Covered_Lives", "max"))
)

# Commission per Employer: sum (then use median/mean in rollups)
emp_comm = (
    ebc.groupby(["Employer"], as_index=False)
       .agg(TotalCommissions=("total_commissions", "sum"))
)

# Primary broker per Employer: broker with max commissions (proxy for incumbent)
ebc_b = ebc.copy()
ebc_b["BrokerNorm"] = ebc_b["Broker"].apply(norm)
primary = (
    ebc_b.groupby(["Employer", "Broker"], as_index=False)
         .agg(BrokerCommissions=("total_commissions", "sum"))
         .sort_values(["Employer", "BrokerCommissions"], ascending=[True, False])
)
primary_broker = primary.drop_duplicates("Employer")[["Employer", "Broker"]].rename(columns={"Broker": "PrimaryBroker"})

# Gaps flags
g = gaps.copy()
g["Life_f"] = as_flag(g["Life"])
g["STD_f"] = as_flag(g["STD"])
g["LTD_f"] = as_flag(g["LTD"])
g["Dis_f"] = g["STD_f"] | g["LTD_f"]
g["Bundled_f"] = g["Life_f"] & g["Dis_f"]
g["LifeOnly_NoDis_f"] = g["Life_f"] & (~g["Dis_f"])
g["DisOnly_NoLife_f"] = g["Dis_f"] & (~g["Life_f"])
g["MissingAny_f"] = (~g["Life_f"]) | (~g["STD_f"]) | (~g["LTD_f"])

# Voluntary benefit flags. These come from the Schedule A OTHER free text rather
# than a checkbox - see etl/benefits.py. AD&D is kept separate from Accident on
# purpose; folding it in roughly doubles apparent VB penetration.
VB_PRODUCTS = list(VOLUNTARY_PRODUCTS)
for prod in ALL_PRODUCTS:
    g[f"{prod}_f"] = as_flag(g[prod]) if prod in g.columns else False

g["AnyVB_f"] = False
for _p in VOLUNTARY_PRODUCTS:
    g["AnyVB_f"] = g["AnyVB_f"] | g[f"{_p}_f"]

g["VBTrio_f"] = True
for _p in VB_TRIO:
    g["VBTrio_f"] = g["VBTrio_f"] & g[f"{_p}_f"]

g["CoreNoVB_f"] = (g["Life_f"] | g["Dis_f"]) & (~g["AnyVB_f"])

# Readable product columns so every table can show what a group actually holds,
# not just a row of booleans.
_vb_flags = g[[f"{p}_f" for p in VOLUNTARY_PRODUCTS]].to_numpy()
g["VBHeld"] = [
    " + ".join([p for p, f in zip(VOLUNTARY_PRODUCTS, row) if f]) or "(none)"
    for row in _vb_flags
]
g["VBMissing"] = [
    " + ".join([p for p, f in zip(VOLUNTARY_PRODUCTS, row) if not f]) or "(complete)"
    for row in _vb_flags
]
g["VBCount"] = _vb_flags.sum(axis=1)
g["CoreHeld"] = [
    " + ".join([p for p, f in zip(CORE_PRODUCTS, row) if f]) or "(none)"
    for row in g[[f"{p}_f" for p in CORE_PRODUCTS]].to_numpy()
]

# Geo merge (State/City)
geo_use = geo.copy()
for col in ["State", "City", "EIN"]:
    if col not in geo_use.columns:
        geo_use[col] = np.nan
geo_use["StateNorm"] = geo_use["State"].apply(normalize_state) if "State" in geo_use.columns else np.nan
geo_use.loc[~geo_use["StateNorm"].isin(list(VALID_STATES)), "StateNorm"] = np.nan
# EIN is an identifier, not a quantity - keep it text so 07-1234567 style values
# do not lose their leading zero or pick up thousands separators.
geo_use["EIN"] = (
    geo_use["EIN"].astype(str).str.replace(r"\.0$", "", regex=True)
    .replace({"nan": "", "<NA>": "", "None": ""})
)

# Premium per employer, summed over DISTINCT contracts (see the load block above).
if len(contracts) and "Premium" in contracts.columns:
    contracts["Premium"] = to_numeric(contracts["Premium"]).fillna(0)
    emp_premium = (
        contracts.groupby("Employer", as_index=False)
                 .agg(TotalPremium=("Premium", "sum"), ContractCount=("ContractRowID", "nunique"))
    )
else:
    emp_premium = pd.DataFrame(columns=["Employer", "TotalPremium", "ContractCount"])

emp = (
    g.merge(emp_lives, on="Employer", how="left")
     .merge(emp_comm, on="Employer", how="left")
     .merge(emp_premium, on="Employer", how="left")
     .merge(primary_broker, on="Employer", how="left")
     .merge(geo_use[["Employer", "StateNorm", "City", "EIN"]], on="Employer", how="left")
)

emp["CoveredLives"] = to_numeric(emp["CoveredLives"]).fillna(0)
emp["TotalCommissions"] = to_numeric(emp["TotalCommissions"]).fillna(0)
emp["TotalPremium"] = to_numeric(emp["TotalPremium"]).fillna(0)
emp["ContractCount"] = to_numeric(emp["ContractCount"]).fillna(0).astype(int)
emp["PrimaryBroker"] = emp["PrimaryBroker"].fillna("UNKNOWN")
emp["BrokerFamily"] = emp["PrimaryBroker"].apply(broker_family)

# Left as NaN, not 0, where the denominator is missing: a filer who left the
# premium box empty is unknown, not free.
emp["PremiumPerLife"] = np.where(
    emp["CoveredLives"] > 0, emp["TotalPremium"] / emp["CoveredLives"].replace(0, np.nan), np.nan
)
emp["CommissionPctOfPremium"] = np.where(
    emp["TotalPremium"] > 0, emp["TotalCommissions"] / emp["TotalPremium"].replace(0, np.nan) * 100.0, np.nan
)


# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.markdown("## Filters")

    aon_view = st.toggle("AON Composite Focus (recommended)", value=True)
    # Even in market view, we still compute all; this toggle controls framing and some tables

    employer_search = st.text_input("Employer search", value="").strip()

    st.divider()
    st.markdown("## Tier settings")
    tier2_pct = st.slider("Tier2 cutoff (top % of 'Other' brokers by lives)", 0.05, 0.30, 0.10, 0.01)

    st.divider()
    st.markdown("## Robustness")
    metric_mode = st.radio("Commission aggregation", ["Median (recommended)", "Mean"], index=0, horizontal=False)
    log_hist = st.toggle("Log scale histograms", value=True)

    st.divider()
    st.markdown("## Opportunity model")
    score_mode = st.radio(
        "Target mode",
        ["Competitive takeout (AON vs competitors)", "AON cross-sell (inside AON book)"],
        index=0
    )

    w_whitespace = st.slider("Weight: Product whitespace", 0.0, 5.0, 2.0, 0.1)
    w_lives = st.slider("Weight: Covered lives (log)", 0.0, 5.0, 2.5, 0.1)
    w_tier = st.slider("Weight: Competitor tier factor", 0.0, 5.0, 1.5, 0.1)
    w_state_gap = st.slider("Weight: State under-index factor", 0.0, 5.0, 1.0, 0.1)
    w_frag = st.slider("Weight: Broker fragmentation", 0.0, 5.0, 0.7, 0.1)

    st.caption("Tip: Start with defaults, then adjust to reflect sales strategy (Tier2/Tier3 emphasis vs Tier1).")

    st.divider()
    st.markdown("## Products")
    product_mode = st.radio(
        "Product filter",
        ["No product filter",
         "HOLDS selected products",
         "MISSING selected products (sell-in targets)"],
        index=0,
        help="Filters every tab, not just this one. 'Missing' is the sell-in view: "
             "employers who do not hold the selected product(s).",
    )
    product_filter = st.multiselect(
        "Products",
        options=ALL_PRODUCTS,
        default=[],
        help="Voluntary products (accident, critical illness, hospital indemnity, cancer) are parsed "
             "from the Schedule A OTHER free text, not a checkbox. AD&D is listed separately because "
             "it is a life rider, not a worksite product.",
    )
    product_match_all = st.toggle(
        "Match ALL selected (instead of ANY)", value=False,
        help="ANY: holds/misses at least one of the selected products. "
             "ALL: holds/misses every one of them.",
    )
    require_core = st.toggle(
        "Sell-in targets must hold a core product", value=True,
        help="Restricts 'missing' results to employers with an existing life or disability "
             "relationship - i.e. a real cross-sell conversation, not a cold open.",
    )

    st.divider()
    st.markdown("## Geography")
    available_states = sorted([s for s in emp["StateNorm"].dropna().unique().tolist() if s in VALID_STATES])
    state_filter = st.multiselect("State filter", options=available_states, default=[])


# Apply employer search + state filter to employer table (base view)
emp_view = emp.copy()
if employer_search:
    pat = norm(employer_search)
    emp_view = emp_view[emp_view["Employer"].apply(lambda x: pat in norm(x))].copy()

if state_filter:
    emp_view = emp_view[emp_view["StateNorm"].isin(state_filter)].copy()

# Product filter. Applied to the base view so every downstream tab - market share,
# whitespace, opportunity scoring, target lists - reflects the selection.
product_filter_note = ""
if product_filter and product_mode != "No product filter":
    flags = [emp_view[f"{p}_f"] for p in product_filter]

    held_any = flags[0].copy()
    held_all = flags[0].copy()
    for s in flags[1:]:
        held_any = held_any | s
        held_all = held_all & s

    joiner = " AND " if product_match_all else " or "
    names = joiner.join(product_filter)

    if product_mode.startswith("HOLDS"):
        # ALL -> holds every selected product; ANY -> holds at least one.
        mask = held_all if product_match_all else held_any
        product_filter_note = f"Holds {names}"
    else:
        # De Morgan: missing EVERY selected product is NOT(holds any);
        # missing AT LEAST ONE is NOT(holds all).
        mask = (~held_any) if product_match_all else (~held_all)
        product_filter_note = f"Missing {names}"

    emp_view = emp_view[mask].copy()

    if product_mode.startswith("MISSING") and require_core:
        emp_view = emp_view[
            emp_view["Life_f"] | emp_view["STD_f"] | emp_view["LTD_f"]
        ].copy()
        product_filter_note += " (with core coverage)"


# =========================
# Build Broker aggregates + tiers
# =========================
# Broker rollups use PrimaryBroker assignment (clean “incumbent” proxy)
broker_agg = (
    emp_view.groupby(["PrimaryBroker", "BrokerFamily"], as_index=False)
            .agg(
                Employers=("Employer", "nunique"),
                CoveredLives=("CoveredLives", "sum"),
                # commission per employer distribution used later; store sum for reference
                TotalCommissions=("TotalCommissions", "sum")
            )
)

broker_agg = assign_tiers(broker_agg, tier2_pct=tier2_pct)

# Add commission metric (median or mean per employer) computed from employer-level
if metric_mode.startswith("Median"):
    comm_metric = (
        emp_view.groupby(["PrimaryBroker"], as_index=False)
                .agg(CommissionMetric=("TotalCommissions", "median"))
    )
else:
    comm_metric = (
        emp_view.groupby(["PrimaryBroker"], as_index=False)
                .agg(CommissionMetric=("TotalCommissions", "mean"))
    )

broker_agg = broker_agg.merge(comm_metric, on="PrimaryBroker", how="left")
broker_agg["CommissionMetric"] = broker_agg["CommissionMetric"].fillna(0)

# Market totals
TOTAL_LIVES = float(broker_agg["CoveredLives"].sum()) if len(broker_agg) else 0.0
TOTAL_EMPS = int(emp_view["Employer"].nunique())

broker_agg["LivesShare"] = (broker_agg["CoveredLives"] / TOTAL_LIVES) if TOTAL_LIVES > 0 else 0.0
broker_agg["EmployerShare"] = (broker_agg["Employers"] / TOTAL_EMPS) if TOTAL_EMPS > 0 else 0.0

# AON vs Everyone (family-level)
family_agg = (
    broker_agg.assign(Group=lambda d: np.where(d["BrokerFamily"].eq("AON"), "AON", "Competitors"))
             .groupby("Group", as_index=False)
             .agg(Employers=("Employers", "sum"), CoveredLives=("CoveredLives", "sum"), TotalCommissions=("TotalCommissions", "sum"))
)
family_agg["LivesShare"] = family_agg["CoveredLives"] / family_agg["CoveredLives"].sum() if family_agg["CoveredLives"].sum() > 0 else 0.0


# =========================
# Fragmentation measures (by state)
# =========================
# Broker fragmentation proxy: distinct brokers per employer count in a state (using PrimaryBroker)
state_frag = (
    emp_view.dropna(subset=["StateNorm"])
            .groupby("StateNorm", as_index=False)
            .agg(
                Employers=("Employer", "nunique"),
                Brokers=("PrimaryBroker", "nunique"),
                Lives=("CoveredLives", "sum")
            )
)
state_frag["FragRatio"] = state_frag.apply(lambda r: (r["Brokers"] / r["Employers"]) if r["Employers"] else 0.0, axis=1)


# =========================
# Competitive state share + under-index factor
# =========================
# Compute AON vs competitor lives share by state
state_share = (
    emp_view.dropna(subset=["StateNorm"])
            .assign(IsAON=lambda d: d["BrokerFamily"].eq("AON"))
            .groupby(["StateNorm", "IsAON"], as_index=False)
            .agg(Lives=("CoveredLives", "sum"), Employers=("Employer", "nunique"))
)

# Pivot to get AON/Competitor lives by state
pivot = state_share.pivot_table(index="StateNorm", columns="IsAON", values="Lives", aggfunc="sum").fillna(0)
pivot.columns = ["CompetitorLives" if c is False else "AONLives" for c in pivot.columns]
pivot = pivot.reset_index()

pivot["TotalLives"] = pivot.get("AONLives", 0) + pivot.get("CompetitorLives", 0)
pivot["AONShare"] = pivot.apply(lambda r: (r["AONLives"] / r["TotalLives"]) if r["TotalLives"] else 0.0, axis=1)
pivot["CompetitorShare"] = pivot.apply(lambda r: (r["CompetitorLives"] / r["TotalLives"]) if r["TotalLives"] else 0.0, axis=1)
pivot["UnderIndexGap"] = pivot["CompetitorShare"] - pivot["AONShare"]  # >0 means competitors dominate

# State under-index factor for scoring (clip to [0,1])
pivot["UnderIndexFactor"] = pivot["UnderIndexGap"].clip(lower=0, upper=1)

# merge into employer rows
emp_view = emp_view.merge(pivot[["StateNorm", "UnderIndexFactor"]], on="StateNorm", how="left")
emp_view["UnderIndexFactor"] = emp_view["UnderIndexFactor"].fillna(0)


# =========================
# Tier factors for scoring (editable logic)
# =========================
TIER_FACTOR = {
    "Tier0": 0.0,  # AON itself for takeout mode
    "Tier1": 0.6,  # hard to steal from top global majors
    "Tier2": 1.0,  # prime targets
    "Tier3": 1.3,  # easiest displacement targets
}

# map primary broker to tier
tier_map = broker_agg.set_index("PrimaryBroker")["Tier"].to_dict()
emp_view["BrokerTier"] = emp_view["PrimaryBroker"].map(tier_map).fillna("Tier3")
emp_view["TierFactor"] = emp_view["BrokerTier"].map(TIER_FACTOR).fillna(1.0)

# Fragmentation factor at state level
frag_map = state_frag.set_index("StateNorm")["FragRatio"].to_dict()
emp_view["FragRatio"] = emp_view["StateNorm"].map(frag_map).fillna(0.0)


# =========================
# Opportunity Scoring
# =========================
def compute_whitespace_score(row: pd.Series) -> float:
    # Simple whitespace: missing any of Life/STD/LTD
    # You could refine to prioritize specific missing product combinations later
    return 1.0 if bool(row.get("MissingAny_f", False)) else 0.0

emp_view["WhitespaceScore"] = emp_view.apply(compute_whitespace_score, axis=1)

# Lives component (log transform)
emp_view["LivesLog"] = emp_view["CoveredLives"].apply(lambda x: math.log(float(x) + 1.0))

# Target eligibility based on mode
if score_mode.startswith("Competitive"):
    # Target competitors only (exclude AON-incumbent employers)
    emp_view["IsTargetEligible"] = ~emp_view["BrokerFamily"].eq("AON")
else:
    # Cross-sell inside AON book only
    emp_view["IsTargetEligible"] = emp_view["BrokerFamily"].eq("AON")

# Score formula
emp_view["OpportunityScore"] = (
    w_whitespace * emp_view["WhitespaceScore"]
    + w_lives * emp_view["LivesLog"]
    + w_tier * emp_view["TierFactor"]
    + w_state_gap * emp_view["UnderIndexFactor"]
    + w_frag * emp_view["FragRatio"]
)

emp_view.loc[~emp_view["IsTargetEligible"], "OpportunityScore"] = 0.0

# State-level score: sum of top employer opportunities and weighted by lives
state_score = (
    emp_view.dropna(subset=["StateNorm"])
            .groupby("StateNorm", as_index=False)
            .agg(
                StateEmployers=("Employer", "nunique"),
                StateLives=("CoveredLives", "sum"),
                TotalOpportunity=("OpportunityScore", "sum"),
                AvgOpportunity=("OpportunityScore", "mean"),
                UnderIndexFactor=("UnderIndexFactor", "mean")
            )
)
state_score["OpportunityPerLife"] = state_score.apply(
    lambda r: (r["TotalOpportunity"] / r["StateLives"]) if r["StateLives"] else 0.0, axis=1
)
state_score = state_score.sort_values("TotalOpportunity", ascending=False)


# =========================
# Header KPIs
# =========================
# Make an active product filter impossible to miss - every number below it is
# scoped to the selection, not to the whole market.
if product_filter_note:
    st.info(
        f"**Product filter active — {product_filter_note}.** "
        f"{int(emp_view['Employer'].nunique()):,} of {int(emp['Employer'].nunique()):,} employers in view. "
        "Every tab below reflects this filter."
    )

with st.expander("Data quality: which rows are excluded and why"):
    st.write(DQ_NOTE)

def _money(v):
    return f"${v/1e9:.2f}B" if abs(v) >= 1e9 else f"${v/1e6:,.1f}M"


# Voluntary-only totals, computed at product grain over the employers in view.
# ProductCommission and ProductPremium are both already divided across the
# products on a bundled contract, so these are additive and do not double count.
_vb_rows = epc[epc["ProductGroup"].eq("Voluntary")
               & epc["Employer"].isin(set(emp_view["Employer"]))]
_vb_by_contract = _vb_rows.drop_duplicates(["Employer", "Product", "ContractRowID"])
VB_COMM = float(_vb_by_contract["ProductCommission"].sum())
VB_PREM = float(_vb_by_contract["ProductPremium"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Employers in view", f"{int(emp_view['Employer'].nunique()):,}")
k2.metric("Total covered lives", f"{int(emp_view['CoveredLives'].sum()):,}")
_prem = float(emp_view["TotalPremium"].sum())
k3.metric("Total premium", _money(_prem))
_comm = float(emp_view["TotalCommissions"].sum())
k4.metric("Total commissions", _money(_comm),
          delta=f"{_comm/_prem*100:.1f}% of premium" if _prem > 0 else None,
          delta_color="off")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Total VB commission", _money(VB_COMM),
          delta=f"{VB_COMM/_comm*100:.1f}% of all commission" if _comm > 0 else None,
          delta_color="off")
k6.metric("Total VB premium", _money(VB_PREM),
          delta=f"{VB_COMM/VB_PREM*100:.2f}% commission rate" if VB_PREM > 0 else None,
          delta_color="off")

aon_lives = float(family_agg.loc[family_agg["Group"] == "AON", "CoveredLives"].sum())
comp_lives = float(family_agg.loc[family_agg["Group"] == "Competitors", "CoveredLives"].sum())
k7.metric("AON lives share (view)", f"{(aon_lives / (aon_lives + comp_lives) * 100.0):.1f}%" if (aon_lives + comp_lives) else "—")
k8.metric("Targets eligible", f"{int(emp_view['IsTargetEligible'].sum()):,}")

st.caption(
    "VB = the voluntary line only (accident, critical illness, hospital indemnity, cancer, "
    "legal, long term care, identity theft, pet). Both VB figures are split across the "
    "products on a bundled contract, so they are additive; the total premium above is the "
    "employer-level figure summed over distinct contracts."
)

st.divider()


# =========================
# Tabs
# =========================
(tab_product, tab_trend, tab_dq, tab_overview, tab_comp, tab_whitespace, tab_scoring,
 tab_diag, tab_report, tab_ai, tab_raw) = st.tabs(
    [
        "Product Detail",
        "Trends",
        "Data Quality",
        "Market Overview",
        "Competitive Share",
        "Product Whitespace",
        "Opportunity Scoring",
        "Diagnostics",
        "Industry Intelligence Report",
        "AI Analyst",
        "Raw Tables",
    ]
)


# =========================
# Product Detail - one row per employer per product
# =========================
with tab_product:
    st.subheader("Every product, every employer - one row each")
    st.caption(
        "Commission and premium are both reported per contract, then divided across the products "
        "on that contract, so every money column here is additive - summing a column gives the "
        "real total, not a multiple of it. Where a contract lists a single product the figure is "
        "exact; where it bundles several it is split evenly, which the Exact% column makes "
        "visible. Covered lives cannot be split the same way (the same people are covered by "
        "each benefit), so lives are the real per-product figure and should not be summed across "
        "products for one employer."
    )

    @st.cache_data(show_spinner="Building product detail...")
    def build_product_detail(_epc: pd.DataFrame, emp_slim: pd.DataFrame) -> pd.DataFrame:
        base = (
            _epc.groupby(["Employer", "Product", "ProductGroup"], as_index=False)
                .agg(CoveredLives=("Covered_Lives", "max"),
                     Carriers=("Carrier", "nunique"),
                     Contracts=("ContractRowID", "nunique"))
        )
        by_contract = _epc.drop_duplicates(["Employer", "Product", "ContractRowID"])
        prem = (
            by_contract.groupby(["Employer", "Product"], as_index=False)
                       .agg(Premium=("ProductPremium", "sum"),
                            ContractPremium=("Premium", "sum"),
                            Commission=("ProductCommission", "sum"))
        )
        exact = (
            by_contract[by_contract["CommissionIsExact"]]
            .groupby(["Employer", "Product"], as_index=False)
            .agg(ExactCommission=("ProductCommission", "sum"))
        )
        prem = prem.merge(exact, on=["Employer", "Product"], how="left")
        prem["ExactCommission"] = prem["ExactCommission"].fillna(0)
        topc = (
            _epc.groupby(["Employer", "Product", "Carrier"], as_index=False)
                .agg(_l=("Covered_Lives", "max"))
                .sort_values(["Employer", "Product", "_l"], ascending=[True, True, False])
                .drop_duplicates(["Employer", "Product"])[["Employer", "Product", "Carrier"]]
                .rename(columns={"Carrier": "TopCarrier"})
        )
        out = (base.merge(prem, on=["Employer", "Product"], how="left")
                   .merge(topc, on=["Employer", "Product"], how="left")
                   .merge(emp_slim, on="Employer", how="left"))
        for c in ["Premium", "Commission", "ExactCommission", "TotalCommissions"]:
            out[c] = out[c].fillna(0)

        out["CommissionExact%"] = out["ExactCommission"] / out["Commission"].replace(0, np.nan) * 100.0
        out["CommissionRate%"] = out["Commission"] / out["Premium"].replace(0, np.nan) * 100.0
        out["PremiumPerLife"] = out["Premium"] / out["CoveredLives"].replace(0, np.nan)
        out["AON_Is_Broker"] = out["BrokerFamily"].eq("AON")
        out["BrokerStatus"] = np.where(out["AON_Is_Broker"], "AON is broker of record",
                                       "NOT AON - opportunity")
        out["EIN"] = (out["EIN"].astype(str).str.replace(r"\.0$", "", regex=True)
                        .replace({"nan": "", "<NA>": "", "None": ""}))
        return out

    _emp_slim = emp_view[["Employer", "EIN", "StateNorm", "City", "PrimaryBroker",
                          "BrokerFamily", "BrokerTier", "TotalCommissions"]].drop_duplicates("Employer")
    pdet = build_product_detail(epc, _emp_slim)
    pdet = pdet[pdet["Employer"].isin(set(emp_view["Employer"]))]

    f1, f2, f3 = st.columns([1.2, 1.4, 1.4])
    with f1:
        broker_view = st.radio(
            "Broker of record",
            ["Everyone", "NOT AON (opportunity)", "AON only"],
            index=0,
            help="AON includes its owned brands - Custom Benefit Programs, Univers Workplace, "
                 "Cammack Health - not just names starting with 'Aon'.",
        )
    with f2:
        prod_pick = st.multiselect("Products", options=ALL_PRODUCTS, default=[],
                                   help="Leave empty for all products.")
    with f3:
        group_pick = st.multiselect("Product group", options=["Core", "Voluntary", "Adjacent"],
                                    default=[])

    view = pdet
    if broker_view.startswith("NOT AON"):
        view = view[~view["AON_Is_Broker"]]
    elif broker_view.startswith("AON only"):
        view = view[view["AON_Is_Broker"]]
    if prod_pick:
        view = view[view["Product"].isin(prod_pick)]
    if group_pick:
        view = view[view["ProductGroup"].isin(group_pick)]

    view = view.sort_values("Commission", ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows in view", f"{len(view):,}")
    c2.metric("Employers", f"{view['Employer'].nunique():,}")
    _c = float(view["Commission"].sum())
    _ex = float(view["ExactCommission"].sum())
    c3.metric("Commission", f"${_c/1e9:.2f}B" if _c >= 1e9 else f"${_c/1e6:,.1f}M",
              delta=f"{_ex/_c*100:.0f}% reported exactly" if _c > 0 else None, delta_color="off")
    _p = float(view["Premium"].sum())
    c4.metric("Premium", f"${_p/1e9:.2f}B" if _p >= 1e9 else f"${_p/1e6:,.0f}M",
              delta=f"{_c/_p*100:.2f}% commission rate" if _p > 0 else None, delta_color="off")

    st.markdown("#### By product")
    roll = (view.groupby(["ProductGroup", "Product"], as_index=False)
                .agg(Employers=("Employer", "nunique"), Lives=("CoveredLives", "sum"),
                     Commission=("Commission", "sum"), Exact=("ExactCommission", "sum"),
                     Premium=("Premium", "sum"))
                .sort_values("Commission", ascending=False))
    roll["Exact%"] = roll["Exact"] / roll["Commission"].replace(0, np.nan) * 100.0
    roll["Rate%"] = roll["Commission"] / roll["Premium"].replace(0, np.nan) * 100.0
    st.dataframe(
        roll[["ProductGroup", "Product", "Employers", "Lives", "Commission", "Exact%",
              "Premium", "Rate%"]],
        use_container_width=True, hide_index=True,
        column_config={
            "Lives": st.column_config.NumberColumn(format="%,d"),
            "Commission": st.column_config.NumberColumn(format="$%,.0f"),
            "Exact%": st.column_config.NumberColumn("Exact%", format="%.0f%%",
                                                    help="Share reported against a single-product contract."),
            "Premium": st.column_config.NumberColumn(format="$%,.0f"),
            "Rate%": st.column_config.NumberColumn("Comm rate", format="%.2f%%"),
        },
    )

    st.markdown("#### Company view — one row per company, one column per product")
    st.caption("A blank cell means that product was never sold to that company. "
               "The Company_Product_Matrix sheet in the Excel export adds a modelled "
               "value for each gap.")
    wide = (view.pivot_table(index="Employer", columns="Product",
                             values="Commission", aggfunc="sum")
                .reindex(columns=[p for p in ALL_PRODUCTS if p in set(view["Product"])]))
    wide_keys = (view[["Employer", "EIN", "StateNorm", "CoveredLives", "PrimaryBroker",
                       "BrokerStatus"]]
                 .drop_duplicates("Employer").set_index("Employer"))
    wide = wide_keys.join(wide, how="right")
    wide["Total"] = wide[[c for c in wide.columns if c in ALL_PRODUCTS]].sum(axis=1, min_count=1)
    wide["Held"] = wide[[c for c in wide.columns if c in ALL_PRODUCTS]].notna().sum(axis=1)
    wide = wide.sort_values("Total", ascending=False)

    st.dataframe(
        wide.head(1000).reset_index(), use_container_width=True, hide_index=True,
        column_config={
            "StateNorm": st.column_config.TextColumn("State"),
            "CoveredLives": st.column_config.NumberColumn("Lives", format="%,d"),
            "Total": st.column_config.NumberColumn("Total comm", format="$%,.0f"),
            **{p: st.column_config.NumberColumn(p, format="$%,.0f")
               for p in ALL_PRODUCTS if p in wide.columns},
        },
    )
    st.download_button(
        "Download company matrix as CSV",
        data=wide.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="company_product_matrix.csv",
        mime="text/csv",
        key="dl_wide",
        help="Every filtered company, not just the 1,000 shown.",
    )

    ROW_CAP = 5000
    st.markdown(f"#### Detail rows")
    if len(view) > ROW_CAP:
        st.caption(f"Showing the top {ROW_CAP:,} of {len(view):,} rows by premium. "
                   "Narrow the filters, or use the download for the full set.")
    st.dataframe(
        view.head(ROW_CAP)[[
            "Employer", "EIN", "StateNorm", "Product", "ProductGroup", "CoveredLives",
            "Commission", "CommissionExact%", "CommissionRate%", "Premium", "PremiumPerLife",
            "BrokerStatus", "PrimaryBroker", "BrokerFamily", "BrokerTier", "TopCarrier",
        ]].reset_index(drop=True),
        use_container_width=True, hide_index=True,
        column_config={
            "StateNorm": st.column_config.TextColumn("State"),
            "CoveredLives": st.column_config.NumberColumn("Lives", format="%,d"),
            "Commission": st.column_config.NumberColumn("Commission", format="$%,.0f"),
            "CommissionExact%": st.column_config.NumberColumn("Exact%", format="%.0f%%"),
            "CommissionRate%": st.column_config.NumberColumn("Comm rate", format="%.2f%%"),
            "Premium": st.column_config.NumberColumn(format="$%,.0f"),
            "PremiumPerLife": st.column_config.NumberColumn("Prem/life", format="$%,.0f"),
        },
    )

    st.download_button(
        "Download this view as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="product_detail.csv",
        mime="text/csv",
        help="Downloads every filtered row, not just the ones displayed above.",
    )


# =========================
# Trends across plan years
# =========================
with tab_trend:
    st.subheader("Product trends across plan years")
    try:
        trend = load_parquet("trend_summary.parquet")
    except FileNotFoundError:
        trend = None

    if trend is None or not len(trend):
        st.info("No trend summary yet. Build one with:  python etl/build_all_years.py")
    else:
        partial = sorted(INCOMPLETE_YEARS)
        if partial:
            st.warning(
                f"Plan year{'s' if len(partial) > 1 else ''} {', '.join(map(str, partial))} "
                "still being filed and excluded from the charts below. Large plans take "
                "extensions, so including a partial year shows a decline that is an artefact "
                "of the filing calendar rather than the market."
            )
        full = trend[~trend["PlanYear"].isin(INCOMPLETE_YEARS)].copy()

        if full["PlanYear"].nunique() < 2:
            st.info("At least two complete plan years are needed for a trend. "
                    "Build more with:  python etl/build_all_years.py")
        else:
            group = st.radio("Product group", ["Voluntary", "Core", "Adjacent"],
                             index=0, horizontal=True)
            sel = full[full["ProductGroup"] == group]

            metric = st.radio("Metric", ["Employers holding", "Commission", "Premium"],
                              index=0, horizontal=True)
            col = {"Employers holding": "Employers", "Commission": "Commission",
                   "Premium": "Premium"}[metric]

            fig = px.line(sel.sort_values("PlanYear"), x="PlanYear", y=col, color="Product",
                          markers=True)
            fig.update_layout(xaxis_title=None,
                              xaxis=dict(tickmode="array",
                                         tickvals=sorted(sel["PlanYear"].unique())))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Year on year")
            years_sorted = [int(y) for y in sorted(sel["PlanYear"].unique())]
            first, last = years_sorted[0], years_sorted[-1]
            piv = sel.pivot_table(index="Product", columns="PlanYear", values=col, aggfunc="sum")
            # Year columns arrive as numpy ints. Streamlit serialises column_config
            # to JSON and a numpy int key is not serialisable, so the labels have
            # to be real Python strings before they reach st.dataframe.
            piv.columns = [str(int(c)) for c in piv.columns]
            fy, ly = str(first), str(last)
            if fy in piv.columns and ly in piv.columns:
                piv["Change"] = piv[ly] - piv[fy]
                piv["Change%"] = (piv[ly] / piv[fy].replace(0, np.nan) - 1) * 100
            money = col in ("Commission", "Premium")
            num_fmt = "$%,.0f" if money else "%,d"
            st.dataframe(
                piv.reset_index().sort_values("Change%", ascending=False),
                use_container_width=True, hide_index=True,
                column_config={
                    **{str(y): st.column_config.NumberColumn(str(y), format=num_fmt)
                       for y in years_sorted},
                    "Change": st.column_config.NumberColumn(f"{first} to {last}", format=num_fmt),
                    "Change%": st.column_config.NumberColumn("Change %", format="%.1f%%"),
                },
            )
            st.caption(
                f"Comparing complete plan years {first} and {last}. Employer counts are "
                "distinct EINs, so a company that changed its filed name is still counted once."
            )


# =========================
# Data Quality - everything cleaned, omitted or transformed, and why
# =========================
with tab_dq:
    st.subheader("What this app cleans, omits or transforms")
    st.caption(
        "Form 5500 is self-reported and unaudited. Nothing below is hidden - every excluded "
        "row is listed with its reported value so you can judge the thresholds yourself."
    )

    d1, d2, d3 = st.columns(3)
    d1.metric("Rows excluded as filer errors", f"{len(DQ_EXCLUDED):,}")
    d2.metric("Commissions as filed", f"${_comm_raw/1e12:,.1f}T" if _comm_raw >= 1e12
              else _money(_comm_raw))
    d3.metric("Commissions after cleaning", _money(float(ebc['total_commissions'].sum())))

    st.markdown("### 1. Rows excluded as implausible")
    st.markdown(
        "A handful of filings carry keying errors large enough to swamp every total. "
        "One row reports **$563 trillion** in commissions; another puts **$6.9 billion** of "
        "commission on a 170-life employer. These are removed from all rollups and listed here."
    )
    if len(DQ_EXCLUDED):
        st.dataframe(
            DQ_EXCLUDED.reset_index(drop=True), use_container_width=True, hide_index=True,
            column_config={
                "ReportedValue": st.column_config.NumberColumn("Reported value", format="%,.0f"),
                "Cap": st.column_config.NumberColumn("Threshold", format="%,.0f"),
            },
        )
        st.download_button("Download excluded rows as CSV",
                           data=DQ_EXCLUDED.to_csv(index=False).encode("utf-8"),
                           file_name="excluded_rows.csv", mime="text/csv", key="dl_dq")
    else:
        st.success("No rows exceeded the thresholds in this view.")

    st.markdown("### 2. Flagged but KEPT — implausible figures still in every total")
    st.markdown(
        "These are not large enough to destroy a total, so removing them silently would be "
        "the bigger error: most are real money with one bad field. They stay in the numbers "
        "and are listed here so you can judge them."
    )
    if len(DQ_FLAG_SUMMARY):
        _fl = DQ_FLAGGED[DQ_FLAGGED["AnyQualityFlag"]]
        q1, q2, q3 = st.columns(3)
        q1.metric("Contracts flagged", f"{len(_fl):,}",
                  delta=f"{len(_fl)/len(DQ_FLAGGED)*100:.2f}% of all contracts",
                  delta_color="off")
        q2.metric("Premium involved", _money(float(_fl["Premium"].sum())),
                  delta=f"{_fl['Premium'].sum()/DQ_FLAGGED['Premium'].sum()*100:.2f}% of premium",
                  delta_color="off")
        q3.metric("Commission involved", _money(float(_fl["Commission"].sum())),
                  delta=f"{_fl['Commission'].sum()/DQ_FLAGGED['Commission'].sum()*100:.2f}% of commission",
                  delta_color="off")

        st.dataframe(
            DQ_FLAG_SUMMARY, use_container_width=True, hide_index=True,
            column_config={
                "Contracts": st.column_config.NumberColumn(format="%,d"),
                "Employers": st.column_config.NumberColumn(format="%,d"),
                "Premium": st.column_config.NumberColumn(format="$%,.0f"),
                "Commission": st.column_config.NumberColumn(format="$%,.0f"),
                "What it means": st.column_config.TextColumn(width="large"),
            },
        )

        pick = st.selectbox("Inspect a check", ["(all)"] + DQ_FLAG_SUMMARY["Check"].tolist())
        show = _fl if pick == "(all)" else _fl[_fl["QualityFlag"] == pick]
        cols = [c for c in ["Employer", "Carrier", "Products", "Covered_Lives", "Premium",
                            "Commission", "PremiumPerLife", "QualityFlag"] if c in show.columns]
        st.dataframe(
            show.nlargest(500, "Premium")[cols].reset_index(drop=True),
            use_container_width=True, hide_index=True,
            column_config={
                "Covered_Lives": st.column_config.NumberColumn("Lives", format="%,d"),
                "Premium": st.column_config.NumberColumn(format="$%,.0f"),
                "Commission": st.column_config.NumberColumn(format="$%,.0f"),
                "PremiumPerLife": st.column_config.NumberColumn("Prem/life", format="$%,.0f"),
            },
        )
        st.caption(f"Showing the largest 500 of {len(show):,} by premium.")
        st.download_button("Download flagged contracts as CSV",
                           data=show[cols].to_csv(index=False).encode("utf-8"),
                           file_name="flagged_contracts.csv", mime="text/csv", key="dl_flag")
    else:
        st.info("No contract-level quality flags in this view.")

    st.markdown("### 3. What is deliberately out of scope")
    st.table(pd.DataFrame([
        {"Excluded": "Medical, dental, vision",
         "Why": "AON does not sell them. Their Schedule A checkboxes are never read and the "
                "free-text parser drops their labels, so they never enter the pipeline."},
        {"Excluded": "Plan years other than the one selected",
         "Why": "A DOL release is only ~96% the year it is named for. The late and amended "
                "tail also appears in its own release, so keeping it would double-count "
                "across years."},
        {"Excluded": "AD&D from the voluntary line",
         "Why": "It is a life rider ~87% of groups already carry. Counting it as voluntary "
                "puts penetration near 91% and hides the real opportunity. Reported "
                "separately, never inside VB totals."},
        {"Excluded": "Free text matching no product rule",
         "Why": "EAP, telehealth, wellness, supplemental life and similar. Counted as no "
                "product rather than guessed at."},
    ]))

    st.markdown("### 4. Where a single figure is split, and why")
    st.markdown(
        "Schedule A reports money **per contract**, not per benefit. A contract covering "
        "life + STD + LTD reports one premium and one commission. Left alone, those repeat "
        "on every product row and sums inflate by the number of products on the contract."
    )
    _mult = epc[epc["ProductsOnContract"] > 1] if "ProductsOnContract" in epc.columns else epc.iloc[0:0]
    _raw_prem = float(epc.drop_duplicates(["Employer", "Product", "ContractRowID"])["Premium"].sum())
    _split_prem = float(epc.drop_duplicates(["Employer", "Product", "ContractRowID"])["ProductPremium"].sum())
    st.table(pd.DataFrame([
        {"Measure": "Premium", "Raw (repeated)": f"${_raw_prem:,.0f}",
         "Split (additive)": f"${_split_prem:,.0f}",
         "Inflation avoided": f"{_raw_prem/_split_prem:.2f}x" if _split_prem else "-"},
        {"Measure": "Commission",
         "Raw (repeated)": f"${float(epc.drop_duplicates('ContractRowID')['ContractCommission'].sum()):,.0f}",
         "Split (additive)": f"${float(epc.drop_duplicates(['Employer','Product','ContractRowID'])['ProductCommission'].sum()):,.0f}",
         "Inflation avoided": "split across products on each contract"},
    ]))
    st.caption(
        f"{_mult['ContractRowID'].nunique():,} contracts in this year cover more than one product. "
        "Covered lives cannot be split the same way - the same people are covered by each "
        "benefit - so lives use MAX per employer rather than SUM."
    )

    st.markdown("### 5. Identity and matching rules")
    st.table(pd.DataFrame([
        {"Rule": "Employer identity",
         "Detail": "Keyed on EIN, not filed name. 5.9% of companies change their filed name "
                   "between years and 1,719 names map to several EINs within one year. "
                   "Display name is the one that company filed most often, with the EIN "
                   "appended where two companies share a name."},
        {"Rule": "AON composite",
         "Detail": "Matched on AON as a whole word, plus declared subsidiaries and owned "
                   "brands: Custom Benefit Programs, Univers Workplace, Cammack Health and "
                   "NFP. Excludes SAMMAONS, GAONA, DATAONLINE and 'not for profit'."},
        {"Rule": "Voluntary products",
         "Detail": "Parsed from Schedule A free text, since there is no checkbox for them. "
                   "Counts are a floor: a filer who left the box empty looks like they hold "
                   "nothing."},
        {"Rule": "Primary broker",
         "Detail": "The broker with the largest commission on that employer's filings. "
                   "UNKNOWN means no broker commission record, not no broker."},
    ]))

    st.markdown("### 6. Known limits")
    for line in [
        "Premium is populated on ~84% of Schedule A rows. A blank is an unreported figure, "
        "not a zero, so premium-per-life is left empty rather than showing $0.",
        "Commission is exact where a contract lists one product and split evenly where it "
        "bundles several. The Exact% column shows which is which on every row.",
        "The newest plan year is always partially filed. Large complex plans take extensions, "
        "so an early pull under-represents big employers.",
        "DOL keeps adding late and amended filings to closed years, so these figures are a "
        "snapshot rather than a fixed truth.",
    ]:
        st.markdown(f"- {line}")


# =========================
# Market Overview
# =========================
with tab_overview:
    st.subheader("Market Overview (AON vs Competitors)")

    c1, c2 = st.columns([1.1, 0.9])

    with c1:
        st.markdown("### Market share by covered lives (AON vs Competitors)")
        fam_plot = family_agg.copy()
        fam_plot["Share"] = (fam_plot["LivesShare"] * 100.0).round(1)
        fig = px.bar(fam_plot, x="Group", y="Share", text="Share")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Broker tier mix (by covered lives)")
        tier_mix = broker_agg.groupby("Tier", as_index=False).agg(Lives=("CoveredLives", "sum"))
        fig2 = px.pie(tier_mix, names="Tier", values="Lives")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### State footprint validation")
    if emp_view["StateNorm"].notna().any():
        uniq_states = sorted([s for s in emp_view["StateNorm"].dropna().unique().tolist() if s in VALID_STATES])
        st.write(f"Unique normalized states in view: **{len(uniq_states)}**")
        with st.expander("Show state list"):
            st.write(uniq_states)
    else:
        st.info("State not available. Provide employer_geo.parquet to enable state analysis.")


# =========================
# Competitive Share
# =========================
with tab_comp:
    st.subheader("Competitive Share (AON vs broker tiers)")

    st.markdown("### Top brokers by covered lives")
    top_brokers = broker_agg.sort_values("CoveredLives", ascending=False).head(25).copy()
    top_brokers["LivesShare%"] = (top_brokers["LivesShare"] * 100.0).round(2)
    top_brokers["EmployerShare%"] = (top_brokers["EmployerShare"] * 100.0).round(2)

    fig = px.bar(top_brokers, x="PrimaryBroker", y="CoveredLives", color="Tier")
    fig.update_layout(xaxis_title="Broker", yaxis_title="Covered Lives")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        top_brokers[
            ["PrimaryBroker", "BrokerFamily", "Tier", "Employers", "CoveredLives", "LivesShare%", "EmployerShare%", "CommissionMetric"]
        ].reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("### Market share comparison (employers vs lives)")
    comp_plot = broker_agg.copy()
    comp_plot = comp_plot.sort_values("CoveredLives", ascending=False).head(15)
    fig2 = px.scatter(
        comp_plot,
        x="EmployerShare",
        y="LivesShare",
        size="CoveredLives",
        color="Tier",
        hover_name="PrimaryBroker",
    )
    fig2.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### AON under-index heatmap by state (competitor share - AON share)")
    if len(pivot) and pivot["StateNorm"].notna().any():
        heat = pivot.copy()
        heat = heat[heat["StateNorm"].isin(list(VALID_STATES))]
        fig3 = px.choropleth(
            heat,
            locations="StateNorm",
            locationmode="USA-states",
            color="UnderIndexGap",
            scope="usa",
            hover_data=["AONShare", "CompetitorShare", "TotalLives"],
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(
            heat.sort_values("UnderIndexGap", ascending=False)
                .head(25)[["StateNorm", "UnderIndexGap", "AONShare", "CompetitorShare", "TotalLives"]]
                .reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.info("State heatmap unavailable (geo missing).")


# =========================
# Product Whitespace
# =========================
with tab_whitespace:
    st.subheader("Product Whitespace (Life / STD / LTD)")

    # Attach rates by Tier
    tmp = emp_view.copy()
    tier_attach = (
        tmp.groupby("BrokerTier", as_index=False)
           .agg(
               Employers=("Employer", "nunique"),
               Lives=("CoveredLives", "sum"),
               LifeRate=("Life_f", "mean"),
               STDRate=("STD_f", "mean"),
               LTDRate=("LTD_f", "mean"),
               BundledRate=("Bundled_f", "mean"),
               MissingAnyRate=("MissingAny_f", "mean")
           )
           .sort_values("Lives", ascending=False)
    )

    st.markdown("### Attach rates by broker tier")
    fig = px.bar(
        tier_attach.melt(id_vars=["BrokerTier"], value_vars=["LifeRate", "STDRate", "LTDRate", "BundledRate", "MissingAnyRate"]),
        x="BrokerTier",
        y="value",
        color="variable",
        barmode="group",
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(tier_attach.reset_index(drop=True), use_container_width=True)

    # ---------------------------------------------------------
    # Voluntary benefits
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("Voluntary Benefits")
    st.caption(
        "Voluntary products have no Schedule A checkbox - they are parsed out of the OTHER free-text field, "
        "so these counts are a floor rather than an exact census. AD&D is shown for contrast but does NOT count "
        "as voluntary: it is a life rider ~87% of groups already carry, and including it would put penetration "
        "near 91% and hide the real opportunity."
    )

    vb_rows = []
    n_emp = tmp["Employer"].nunique()
    rollups = [("Any voluntary product", "AnyVB_f"),
               (" + ".join(VB_TRIO) + " (all)", "VBTrio_f"),
               ("AD&D (not counted as voluntary)", "AD&D_f")]
    for label, col in [(p, f"{p}_f") for p in VOLUNTARY_PRODUCTS] + rollups:
        sel = tmp[tmp[col]]
        vb_rows.append({
            "Product": label,
            "Employers": sel["Employer"].nunique(),
            "Penetration": (sel["Employer"].nunique() / n_emp) if n_emp else 0.0,
            "Lives": sel["CoveredLives"].sum(),
        })
    vb_pen = pd.DataFrame(vb_rows)

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("#### Penetration")
        figv = px.bar(vb_pen[vb_pen["Product"].isin(VOLUNTARY_PRODUCTS)], x="Product", y="Penetration")
        figv.update_layout(yaxis_tickformat=".0%", xaxis_title=None)
        st.plotly_chart(figv, use_container_width=True)
    with c2:
        st.markdown("#### Voluntary attach rate by broker tier")
        # Chart the trio plus the any-VB rollup; the thin products would be flat lines.
        tier_aggs = {p.replace(" ", ""): (f"{p}_f", "mean") for p in VB_TRIO}
        vb_tier = (
            tmp.groupby("BrokerTier", as_index=False)
               .agg(AnyVB=("AnyVB_f", "mean"), **tier_aggs)
        )
        figt = px.bar(
            vb_tier.melt(id_vars=["BrokerTier"],
                         value_vars=[p.replace(" ", "") for p in VB_TRIO] + ["AnyVB"]),
            x="BrokerTier", y="value", color="variable", barmode="group",
        )
        figt.update_layout(yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
        st.plotly_chart(figt, use_container_width=True)

    st.dataframe(vb_pen, use_container_width=True)

    st.markdown("### Voluntary cross-sell targets")
    st.caption("Employers holding life and/or disability with no voluntary product attached.")
    vb_targets = (
        tmp[tmp["CoreNoVB_f"]][
            ["Employer", "StateNorm", "CoveredLives", "TotalPremium", "PremiumPerLife",
             "TotalCommissions", "CoreHeld", "PrimaryBroker", "BrokerFamily", "BrokerTier"]
        ].sort_values("TotalPremium", ascending=False)
    )
    m1, m2 = st.columns(2)
    m1.metric("Employers with core products but zero voluntary", f"{len(vb_targets):,}")
    m2.metric("Premium already in place with those employers",
              f"${vb_targets['TotalPremium'].sum()/1e9:.2f}B")
    st.caption("Premium in place is what they already spend on life/disability - the size of the "
               "relationship, not the voluntary revenue on offer.")
    st.dataframe(
        vb_targets.head(300).reset_index(drop=True), use_container_width=True,
        column_config={
            "TotalPremium": st.column_config.NumberColumn("Premium", format="$%,.0f"),
            "PremiumPerLife": st.column_config.NumberColumn("Prem/life", format="$%,.0f"),
            "TotalCommissions": st.column_config.NumberColumn("Commissions", format="$%,.0f"),
            "CoveredLives": st.column_config.NumberColumn("Lives", format="%,d"),
        },
    )

    st.markdown("---")
    st.markdown("### Whitespace distribution by state")
    if tmp["StateNorm"].notna().any():
        state_ws = (
            tmp.dropna(subset=["StateNorm"])
               .groupby("StateNorm", as_index=False)
               .agg(
                   Employers=("Employer", "nunique"),
                   Lives=("CoveredLives", "sum"),
                   MissingAnyRate=("MissingAny_f", "mean"),
                   BundledRate=("Bundled_f", "mean"),
               )
        )
        fig2 = px.choropleth(
            state_ws,
            locations="StateNorm",
            locationmode="USA-states",
            color="MissingAnyRate",
            scope="usa",
            hover_data=["Employers", "Lives", "BundledRate"],
        )
        fig2.update_layout(coloraxis_colorbar=dict(tickformat=".0%"))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("State whitespace map unavailable (geo missing).")


# =========================
# Opportunity Scoring
# =========================
with tab_scoring:
    st.subheader("Opportunity Scoring")

    st.markdown("### Opportunity score distribution")
    fig = px.histogram(emp_view[emp_view["OpportunityScore"] > 0], x="OpportunityScore", nbins=40)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top employer targets")
    topN = 50
    t = emp_view[emp_view["OpportunityScore"] > 0].copy()
    t = t.sort_values(["OpportunityScore", "CoveredLives"], ascending=False).head(topN)

    # Build “Why” explanation deterministic (no AI required)
    def why_row(r: pd.Series) -> str:
        reasons = []
        if r.get("MissingAny_f", False):
            missing = [p for p in CORE_PRODUCTS if not bool(r.get(f"{p}_f", False))]
            if missing:
                reasons.append(f"Missing core: {', '.join(missing)}")
        # Voluntary gap is the whole point of the pilot, so it is called out
        # explicitly rather than left for the reader to infer from the flags.
        if not bool(r.get("AnyVB_f", False)):
            reasons.append("No voluntary products")
        elif not bool(r.get("VBTrio_f", False)):
            gap = [p for p in VB_TRIO if not bool(r.get(f"{p}_f", False))]
            if gap:
                reasons.append(f"Voluntary gap: {', '.join(gap)}")
        if float(r.get("CoveredLives", 0)) > 0:
            reasons.append(f"Lives: {int(r['CoveredLives']):,}")
        if str(r.get("BrokerTier","")):
            reasons.append(f"Tier: {r['BrokerTier']}")
        if float(r.get("UnderIndexFactor", 0)) > 0.15:
            reasons.append("Under-indexed state")
        if float(r.get("FragRatio", 0)) > 0.6:
            reasons.append("High broker fragmentation")
        return " | ".join(reasons) if reasons else "—"

    t["Why"] = t.apply(why_row, axis=1)

    show_cols = ["Employer", "StateNorm", "City", "CoveredLives", "TotalPremium", "PremiumPerLife",
                 "TotalCommissions", "CommissionPctOfPremium", "PrimaryBroker", "BrokerFamily", "BrokerTier",
                 "CoreHeld", "VBHeld", "VBMissing", "OpportunityScore", "Why"]
    st.dataframe(
        t[show_cols].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "TotalPremium": st.column_config.NumberColumn("Premium", format="$%,.0f"),
            "PremiumPerLife": st.column_config.NumberColumn("Prem/life", format="$%,.0f"),
            "TotalCommissions": st.column_config.NumberColumn("Commissions", format="$%,.0f"),
            "CommissionPctOfPremium": st.column_config.NumberColumn("Comm % of prem", format="%.1f%%"),
            "CoveredLives": st.column_config.NumberColumn("Lives", format="%,d"),
        },
    )

    st.markdown("### State opportunity ranking")
    st.dataframe(
        state_score.head(30).reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("### State opportunity heatmap")
    if state_score["StateNorm"].notna().any():
        fig2 = px.choropleth(
            state_score,
            locations="StateNorm",
            locationmode="USA-states",
            color="TotalOpportunity",
            scope="usa",
            hover_data=["StateEmployers", "StateLives", "OpportunityPerLife", "UnderIndexFactor"],
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("State opportunity heatmap unavailable (geo missing).")


# =========================
# Diagnostics (trust layer)
# =========================
with tab_diag:
    st.subheader("Diagnostics (Skew / Outliers / Trust)")

    st.markdown("### Commission histogram (per employer)")
    data = emp_view["TotalCommissions"].copy()
    data = data[data > 0]

    if len(data) == 0:
        st.info("No commission values > 0 in current view.")
    else:
        if log_hist:
            # log1p transform
            fig = px.histogram(np.log1p(data), nbins=50)
            fig.update_layout(xaxis_title="log(1 + TotalCommissions)", yaxis_title="Count")
        else:
            fig = px.histogram(data, nbins=50)
            fig.update_layout(xaxis_title="TotalCommissions", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Commission by broker tier (box plot)")
        tmp = emp_view[emp_view["TotalCommissions"] > 0].copy()
        if log_hist:
            tmp["CommShown"] = np.log1p(tmp["TotalCommissions"])
            y_title = "log(1 + TotalCommissions)"
        else:
            tmp["CommShown"] = tmp["TotalCommissions"]
            y_title = "TotalCommissions"
        fig2 = px.box(tmp, x="BrokerTier", y="CommShown", points="outliers")
        fig2.update_layout(yaxis_title=y_title)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Covered lives distribution by broker tier (box plot)")
    tmp2 = emp_view[emp_view["CoveredLives"] > 0].copy()
    if len(tmp2):
        if log_hist:
            tmp2["LivesShown"] = np.log1p(tmp2["CoveredLives"])
            y_title = "log(1 + CoveredLives)"
        else:
            tmp2["LivesShown"] = tmp2["CoveredLives"]
            y_title = "CoveredLives"
        fig3 = px.box(tmp2, x="BrokerTier", y="LivesShown", points="outliers")
        fig3.update_layout(yaxis_title=y_title)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No covered lives > 0 in current view.")

    st.markdown("### State count validation")
    if emp_view["StateNorm"].notna().any():
        uniq_states = sorted([s for s in emp_view["StateNorm"].dropna().unique().tolist() if s in VALID_STATES])
        st.write(f"Normalized states in view: **{len(uniq_states)}** (includes PR if present)")
        st.dataframe(pd.DataFrame({"StateNorm": uniq_states}), use_container_width=True)
    else:
        st.info("State validation unavailable (geo missing).")


# =========================
# Industry Intelligence Report (deterministic “study” tab)
# =========================
with tab_report:
    st.subheader("Industry Intelligence Report (Draft)")
    st.caption("This is a deterministic study draft generated from current view filters—safe, auditable, and not LLM-dependent.")

    # Core bullets
    aon_share = float(family_agg.loc[family_agg["Group"] == "AON", "LivesShare"].sum())
    comp_share = 1 - aon_share if (aon_share <= 1) else 0
    top_comp_states = pivot.sort_values("UnderIndexGap", ascending=False).head(5) if len(pivot) else pd.DataFrame()
    top_targets = emp_view[emp_view["OpportunityScore"] > 0].sort_values("OpportunityScore", ascending=False).head(10)

    st.markdown("### Executive summary (auto)")
    bullets = []
    bullets.append(f"- Current view contains **{int(emp_view['Employer'].nunique()):,}** employers and **{int(emp_view['CoveredLives'].sum()):,}** covered lives.")
    bullets.append(f"- AON covered-lives share in this view is **{aon_share*100:.1f}%** (competitors: **{comp_share*100:.1f}%**).")
    bullets.append(f"- Commission reporting is **right-skewed**; use **{metric_mode.lower()}** for comparisons and rely on histograms/box plots for validation.")
    bullets.append(f"- Opportunity scoring mode: **{score_mode}** with adjustable weights (whitespace, lives, tier, state gap, fragmentation).")
    for b in bullets:
        st.write(b)

    st.markdown("### Key findings")
    if len(top_comp_states):
        st.write("- States where competitors most under-index AON (by lives share):")
        st.dataframe(
            top_comp_states[["StateNorm", "UnderIndexGap", "AONShare", "CompetitorShare", "TotalLives"]].reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.write("- State under-index view not available without geo mart.")

    st.write("- Top target employers (from scoring):")
    if len(top_targets):
        st.dataframe(
            top_targets[["Employer", "StateNorm", "CoveredLives", "PrimaryBroker", "BrokerTier", "OpportunityScore"]]
            .reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.write("  - No eligible targets found under current filters/weights.")

    st.markdown("### Suggested “study” outputs you can publish")
    st.write("- Market share by lives and employers (AON vs competitors + tier segmentation)")
    st.write("- Regional footprint: where AON is under-indexed and why")
    st.write("- Product whitespace rates by tier and region")
    st.write("- Robust methodology: median vs mean, skew diagnostics, and outlier policy")


# =========================
# AI Analyst (keep an AI piece)
# =========================
with tab_ai:
    st.subheader("AI Analyst (Q&A)")
    st.caption("This assistant answers questions based on the computed tables (not raw aggregation).")

    try:
        openai_key = st.secrets.get("OPENAI_API_KEY", "")
        model_name = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    except FileNotFoundError:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Prepare compact context (keep it small and factual)
    top_b = broker_agg.sort_values("CoveredLives", ascending=False).head(15)[
        ["PrimaryBroker", "BrokerFamily", "Tier", "Employers", "CoveredLives", "LivesShare", "CommissionMetric"]
    ].copy()
    top_s = state_score.head(15)[["StateNorm", "StateEmployers", "StateLives", "TotalOpportunity", "UnderIndexFactor"]].copy()
    top_e = emp_view[emp_view["OpportunityScore"] > 0].sort_values("OpportunityScore", ascending=False).head(20)[
        ["Employer", "StateNorm", "CoveredLives", "PrimaryBroker", "BrokerTier", "OpportunityScore"]
    ].copy()

    context = {
        "filters": {
            "metric_mode": metric_mode,
            "score_mode": score_mode,
            "tier2_pct": tier2_pct,
            "state_filter": state_filter,
            "employer_search": employer_search,
        },
        "summary": {
            "employers": int(emp_view["Employer"].nunique()),
            "covered_lives": int(emp_view["CoveredLives"].sum()),
            "aon_lives_share": float(family_agg.loc[family_agg["Group"] == "AON", "LivesShare"].sum()) if len(family_agg) else 0.0,
        },
        "top_brokers": top_b.to_dict(orient="records"),
        "top_states": top_s.to_dict(orient="records"),
        "top_employer_targets": top_e.to_dict(orient="records"),
    }

    q = st.text_area(
        "Ask a question (examples: 'Where is AON most under-indexed?', 'Why are top targets scoring high?', 'What does skew imply for commission metrics?')",
        height=100,
    )

    colA, colB = st.columns([0.25, 0.75])
    with colA:
        ask = st.button("Ask AI")

    if not openai_key:
        st.info("AI is disabled because OPENAI_API_KEY is not set in Streamlit Secrets (or environment).")
    else:
        if ask and q.strip():
            with st.spinner("Thinking..."):
                # Use OpenAI client if available; fallback to a clear error otherwise
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=openai_key)

                    system = (
                        "You are an analytical assistant for an AON competitive intelligence dashboard.\n"
                        "Rules:\n"
                        "- Do not invent numbers.\n"
                        "- Use only the provided context tables.\n"
                        "- If data is insufficient, say what is missing.\n"
                        "- Be concise, executive-friendly, and focus on AON vs competitors.\n"
                        "- Explain skew/outliers when discussing commissions.\n"
                    )

                    user = (
                        f"Question: {q}\n\n"
                        f"Context (JSON): {context}\n"
                    )

                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0.2,
                    )
                    st.markdown(resp.choices[0].message.content)
                except Exception as e:
                    st.error(f"AI call failed: {e}")
        elif ask:
            st.warning("Type a question first.")


# =========================
# Raw tables
# =========================
with tab_raw:
    st.subheader("Raw Tables (for validation)")

    st.markdown("#### Employer table (modeled)")
    st.caption("One boolean column per product. Voluntary products come from the Schedule A OTHER free text; "
               "AD&D is shown but is not counted as voluntary.")
    st.dataframe(
        emp_view[
            ["Employer", "StateNorm", "City", "CoveredLives", "TotalPremium", "PremiumPerLife",
             "TotalCommissions", "CommissionPctOfPremium", "ContractCount",
             "PrimaryBroker", "BrokerFamily", "BrokerTier",
             "Life_f", "STD_f", "LTD_f", "MissingAny_f"]
            + [f"{p}_f" for p in VOLUNTARY_PRODUCTS]
            + ["AD&D_f", "VBCount", "VBHeld", "AnyVB_f", "CoreNoVB_f", "OpportunityScore"]
        ].head(500).reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("#### Broker aggregate table")
    st.dataframe(broker_agg.sort_values("CoveredLives", ascending=False).head(200).reset_index(drop=True), use_container_width=True)

    st.markdown("#### State score table")
    if len(state_score):
        st.dataframe(state_score.head(200).reset_index(drop=True), use_container_width=True)
    else:
        st.info("No state scoring available (geo missing).")