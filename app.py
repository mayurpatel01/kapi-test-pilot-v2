# app.py
# Voluntary Benefits Intelligence Insights (Form 5500)
# AON vs Competitors: Market Share + Diagnostics + Opportunity Scoring + AI Q&A
#
# Data marts expected in data/marts:
# - employer_product_carrier.parquet: ['ACK_ID','Carrier','Covered_Lives','Product','Employer_ID','Employer']
# - employer_broker_commissions.parquet: ['ACK_ID','Broker','total_commissions','Employer_ID','Employer']
# - employer_product_matrix.parquet: ['Employer','Life','STD','LTD']
# - employer_geo.parquet (optional, downloaded via EMPLOYER_GEO_URL): ['Employer','State','ZIP','City','EIN'] (we use State/City)
#
# Key assumptions:
# - Covered lives per Employer = max(Covered_Lives) to avoid double-counting across Product/Carrier rows
# - Commission comparisons default to median per employer, with toggle mean vs median + histograms (log)
# - States are sponsor HQ (Form 5500 sponsor location), not employee residence

import os
import re
import math
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


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
    "AON vs competitors decision engine: market share, whitespace, robust diagnostics, and target scoring from Form 5500."
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
def load_parquet(filename: str) -> pd.DataFrame:
    ensure_mart_exists(filename)
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing mart: {path}")
    return pd.read_parquet(path)


def norm(s: str) -> str:
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
with st.spinner("Loading data marts..."):
    epc = ensure_str(load_parquet("employer_product_carrier.parquet"))
    ebc = ensure_str(load_parquet("employer_broker_commissions.parquet"))
    gaps = ensure_str(load_parquet("employer_product_matrix.parquet"))

    try:
        geo = ensure_str(load_parquet("employer_geo.parquet"))
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
ebc["total_commissions"] = to_numeric(ebc["total_commissions"]).fillna(0)

# Normalize products a bit (defensive)
epc["ProductNorm"] = epc["Product"].apply(norm)


# =========================
# AON Composite + Competitor Tiers
# =========================
AON_ROOTS = [
    "AON CORPORATION",
    "AON RISK SERVICES",
    "AON HEWITT",
    "AON CONSULTING",
    "AON SOLUTIONS",
]

def is_aon_composite(broker_name: str) -> bool:
    n = norm(broker_name)
    for root in AON_ROOTS:
        r = norm(root)
        # Must start with root to avoid DATAONLINE false positives
        if re.match(rf"^{re.escape(r)}(\s+.*)?$", n):
            return True
    return False

# Tier 1 majors: keep simple name matching (robust enough for strategy view)
TIER1_PATTERNS = {
    "MARSH": [r"\bMARSH\b", r"\bMARSH\s+MCLENNAN\b", r"\bMMC\b"],
    "WTW": [r"\bWILLIS\b", r"\bTOWERS\b", r"\bWTW\b", r"\bWILLIS\s+TOWERS\s+WATSON\b"],
    "GALLAGHER": [r"\bGALLAGHER\b", r"\bARTHUR\s+J\s+GALLAGHER\b"],
    "BROWN & BROWN": [r"\bBROWN\b.*\bBROWN\b", r"\bBROWN\s*&\s*BROWN\b"],
}

def match_any(patterns, text):
    for p in patterns:
        if re.search(p, text):
            return True
    return False

def broker_family(broker_name: str) -> str:
    n = norm(broker_name)
    if not n:
        return "UNKNOWN"

    if is_aon_composite(n):
        return "AON"

    # Tier1 named
    for fam, pats in TIER1_PATTERNS.items():
        if match_any(pats, n):
            return fam

    return "OTHER"

def assign_tiers(broker_agg: pd.DataFrame, tier2_pct: float = 0.10) -> pd.DataFrame:
    """
    Tier rules:
    - AON -> Tier0
    - Named majors -> Tier1
    - Remaining OTHER -> Tier2 if top X% by Covered Lives; else Tier3
    """
    out = broker_agg.copy()
    out["Tier"] = "Tier3"

    out.loc[out["BrokerFamily"] == "AON", "Tier"] = "Tier0"
    out.loc[out["BrokerFamily"].isin(list(TIER1_PATTERNS.keys())), "Tier"] = "Tier1"

    mask_other = out["BrokerFamily"].eq("OTHER")
    other = out[mask_other].copy()

    if len(other) > 0:
        thresh = other["CoveredLives"].quantile(1 - tier2_pct)
        out.loc[mask_other & (out["CoveredLives"] >= thresh), "Tier"] = "Tier2"
        out.loc[mask_other & (out["CoveredLives"] < thresh), "Tier"] = "Tier3"

    return out


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

# Geo merge (State/City)
geo_use = geo.copy()
for col in ["State", "City"]:
    if col not in geo_use.columns:
        geo_use[col] = np.nan
geo_use["StateNorm"] = geo_use["State"].apply(normalize_state) if "State" in geo_use.columns else np.nan
geo_use.loc[~geo_use["StateNorm"].isin(list(VALID_STATES)), "StateNorm"] = np.nan

emp = (
    g.merge(emp_lives, on="Employer", how="left")
     .merge(emp_comm, on="Employer", how="left")
     .merge(primary_broker, on="Employer", how="left")
     .merge(geo_use[["Employer", "StateNorm", "City"]], on="Employer", how="left")
)

emp["CoveredLives"] = to_numeric(emp["CoveredLives"]).fillna(0)
emp["TotalCommissions"] = to_numeric(emp["TotalCommissions"]).fillna(0)
emp["PrimaryBroker"] = emp["PrimaryBroker"].fillna("UNKNOWN")
emp["BrokerFamily"] = emp["PrimaryBroker"].apply(broker_family)


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
k1, k2, k3, k4 = st.columns(4)
k1.metric("Employers in view", f"{int(emp_view['Employer'].nunique()):,}")
k2.metric("Total covered lives", f"{int(emp_view['CoveredLives'].sum()):,}")
aon_lives = float(family_agg.loc[family_agg["Group"] == "AON", "CoveredLives"].sum())
comp_lives = float(family_agg.loc[family_agg["Group"] == "Competitors", "CoveredLives"].sum())
k3.metric("AON lives share (view)", f"{(aon_lives / (aon_lives + comp_lives) * 100.0):.1f}%" if (aon_lives + comp_lives) else "—")
k4.metric("Targets eligible", f"{int(emp_view['IsTargetEligible'].sum()):,}")

st.divider()


# =========================
# Tabs
# =========================
tab_overview, tab_comp, tab_whitespace, tab_scoring, tab_diag, tab_report, tab_ai, tab_raw = st.tabs(
    [
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
            missing = []
            if not bool(r.get("Life_f", False)): missing.append("Life")
            if not bool(r.get("STD_f", False)): missing.append("STD")
            if not bool(r.get("LTD_f", False)): missing.append("LTD")
            if missing:
                reasons.append(f"Missing: {', '.join(missing)}")
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

    show_cols = ["Employer", "StateNorm", "City", "CoveredLives", "PrimaryBroker", "BrokerFamily", "BrokerTier",
                 "TotalCommissions", "OpportunityScore", "Why"]
    st.dataframe(t[show_cols].reset_index(drop=True), use_container_width=True)

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
    st.dataframe(
        emp_view[
            ["Employer", "StateNorm", "City", "CoveredLives", "TotalCommissions", "PrimaryBroker", "BrokerFamily", "BrokerTier",
             "Life_f", "STD_f", "LTD_f", "MissingAny_f", "OpportunityScore"]
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