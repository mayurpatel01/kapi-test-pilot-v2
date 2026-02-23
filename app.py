# app.py
# Voluntary Benefits Intelligence Insights (Form 5500)
# Streamlit dashboard + AON Composite filtering + Geo heatmap + ZIP intelligence

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px


# -----------------------------
# Page config + Branding (NO "CEO" language)
# -----------------------------
st.set_page_config(
    page_title="Voluntary Benefits Intelligence Insights (Form 5500)",
    page_icon="📍",
    layout="wide",
)

st.title("Voluntary Benefits Intelligence Insights (Form 5500)")
st.caption(
    "Strategic signals from public Form 5500 data: portfolio footprint, cross-sell patterns, and regional opportunity structure."
)

# -----------------------------
# Password gate (works locally + Streamlit Cloud)
# -----------------------------
def require_password():
    """
    Uses Streamlit Secrets if available.
    Falls back to env var APP_PASSWORD if secrets file is missing in some environments.
    """
    try:
        pw = st.secrets.get("APP_PASSWORD", "")
    except FileNotFoundError:
        pw = os.getenv("APP_PASSWORD", "")

    if not pw:
        st.error("APP_PASSWORD not set in Streamlit Secrets.")
        st.stop()

    # session state
    if "authed" not in st.session_state:
        st.session_state.authed = False

    if st.session_state.authed:
        return

    with st.sidebar:
        st.markdown("### Access")
        entered = st.text_input("Password", type="password")
        if st.button("Unlock"):
            if entered == pw:
                st.session_state.authed = True
                st.success("Unlocked.")
            else:
                st.error("Incorrect password.")

    if not st.session_state.authed:
        st.stop()


require_password()

# -----------------------------
# Data loading helpers
# -----------------------------
DATA_DIR = Path("data/marts")

def ensure_str(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all object columns are strings (helps stable merges/filters)."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str)
    return out

@st.cache_data(show_spinner=False)
def load_parquet(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing mart: {path}")
    return pd.read_parquet(path)

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first existing column in df that matches one of candidates (case-sensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def find_col_contains(df: pd.DataFrame, needle: str) -> Optional[str]:
    """Find first column whose lowercase name contains needle."""
    needle = needle.lower()
    for c in df.columns:
        if needle in c.lower():
            return c
    return None


# -----------------------------
# Load marts
# -----------------------------
with st.spinner("Loading marts..."):
    # Required marts (as you described)
    epc = ensure_str(load_parquet("employer_product_carrier.parquet"))      # Employer x Product x Carrier (+ Covered_Lives)
    ebc = ensure_str(load_parquet("employer_broker_commissions.parquet"))   # Employer x Broker (+ total_commissions)
    gaps = ensure_str(load_parquet("employer_product_matrix.parquet"))      # Employer x Life/STD/LTD flags

    # Geo mart (you built this)
    geo = ensure_str(load_parquet("employer_geo.parquet"))                 # Employer x State/ZIP/City/EIN (optional)

# Keep needed geo columns if present
geo_cols = [c for c in ["Employer", "State", "ZIP", "City", "EIN"] if c in geo.columns]
geo = geo[geo_cols].copy()

# Merge geo into all marts so any view can use it
for name, df in [("gaps", gaps), ("epc", epc), ("ebc", ebc)]:
    if "Employer" in df.columns and "Employer" in geo.columns:
        merged = df.merge(geo, on="Employer", how="left")
        if name == "gaps":
            gaps = merged
        elif name == "epc":
            epc = merged
        else:
            ebc = merged


# -----------------------------
# AON Composite filter (STRICT, avoids DATAONLINE false positives)
# -----------------------------
AON_ENTITIES = [
    "AON CORPORATION",
    "AON RISK SERVICES",
    "AON HEWITT",
    "AON CONSULTING",
    "AON SOLUTIONS",
]

def norm_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_aon_composite(broker_name: str) -> bool:
    """
    Strict match: broker/service-provider name must START with one of the approved AON roots.
    Allows suffixes (INC, LLC, GROUP, etc.) and extra tokens.
    """
    n = norm_name(broker_name)
    for root in AON_ENTITIES:
        r = norm_name(root)
        if re.match(rf"^{re.escape(r)}(\s+.*)?$", n):
            return True
    return False

def get_aon_employers_from_ebc(ebc_df: pd.DataFrame) -> Tuple[set, str]:
    # Identify broker column
    broker_col = find_col(ebc_df, ["Broker", "BROKER", "broker"])
    if not broker_col:
        broker_col = find_col_contains(ebc_df, "broker")
    if not broker_col:
        raise ValueError(f"No broker column found in employer_broker_commissions. Columns: {list(ebc_df.columns)}")

    aon_rows = ebc_df[ebc_df[broker_col].apply(is_aon_composite)]
    aon_employers = set(aon_rows["Employer"].dropna().unique())
    return aon_employers, broker_col


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.markdown("## Filters")
    aon_only = st.toggle("AON Composite Only", value=False)

    # Optional state filter
    all_states = sorted([s for s in gaps["State"].dropna().unique().tolist() if str(s).strip()])
    state_filter = st.multiselect("State", options=all_states, default=[])

    # ZIP search
    st.markdown("## ZIP Intelligence")
    zip_query = st.text_input("Enter ZIP code (5-digit)", value="").strip()
    zip_query = re.sub(r"\D", "", zip_query)[:5]

# Apply AON filter (by employers tied to AON brokers in ebc)
broker_col_used = None
aon_employers = None
if aon_only:
    aon_employers, broker_col_used = get_aon_employers_from_ebc(ebc)
    gaps = gaps[gaps["Employer"].isin(aon_employers)].copy()
    epc  = epc[epc["Employer"].isin(aon_employers)].copy()
    ebc  = ebc[ebc["Employer"].isin(aon_employers)].copy()

# Apply state filter
if state_filter:
    gaps = gaps[gaps["State"].isin(state_filter)].copy()
    epc  = epc[epc["State"].isin(state_filter)].copy() if "State" in epc.columns else epc
    ebc  = ebc[ebc["State"].isin(state_filter)].copy() if "State" in ebc.columns else ebc


# -----------------------------
# Product columns (defensive mapping)
# -----------------------------
# Your gaps mart is described as Employer x Life/STD/LTD flags.
# Column names vary across builds, so we map robustly.

life_col = find_col(gaps, ["Life", "LIFE", "life"])
std_col  = find_col(gaps, ["STD", "Std", "std", "ShortTermDisability", "SHORT_TERM_DISABILITY"])
ltd_col  = find_col(gaps, ["LTD", "Ltd", "ltd", "LongTermDisability", "LONG_TERM_DISABILITY"])

# If not found, try contains
if not life_col:
    life_col = find_col_contains(gaps, "life")
if not std_col:
    # match "std" but avoid "state"
    for c in gaps.columns:
        if c.lower() == "std":
            std_col = c
            break
    if not std_col:
        std_col = find_col_contains(gaps, "short")
if not ltd_col:
    ltd_col = find_col_contains(gaps, "long")

def as_flag(series: pd.Series) -> pd.Series:
    # Normalize to boolean (accept 1/0, True/False, "1"/"0")
    s = series.fillna(0)
    if s.dtype == "object":
        s = s.astype(str).str.strip()
        return s.isin(["1", "TRUE", "True", "true", "Y", "YES", "Yes", "yes"])
    return s.astype(int).fillna(0) == 1


# -----------------------------
# Tabs layout (keeps your app organized; prevents "losing tables")
# -----------------------------
tab_overview, tab_heatmap, tab_zip, tab_tables = st.tabs(
    ["Overview", "Heatmap", "ZIP Intelligence", "Data Tables"]
)

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)

    employers_n = gaps["Employer"].nunique() if "Employer" in gaps.columns else 0
    states_n = gaps["State"].nunique() if "State" in gaps.columns else 0
    cities_n = gaps["City"].nunique() if "City" in gaps.columns else 0

    col1.metric("Employers", f"{employers_n:,}")
    col2.metric("States", f"{states_n:,}")
    col3.metric("Cities", f"{cities_n:,}")

    if aon_only and aon_employers is not None:
        col4.metric("AON Employers (Composite)", f"{len(aon_employers):,}")
    else:
        col4.metric("View", "Market Benchmark")

    st.markdown("### Notes")
    st.write(
        "- This dashboard is positioned as **Intelligence Insights** (no role-specific framing).\n"
        "- Use **AON Composite Only** to analyze AON-linked employers (based on broker/service-provider naming in filings).\n"
        "- ZIP intelligence provides local opportunity signals using portfolio/product/carrier/broker structure."
    )

    # Quick AON validation list
    if aon_only and broker_col_used:
        with st.expander("AON entity matching (validation)"):
            st.write("Broker column used:", broker_col_used)
            st.write("Sample matched broker names:")
            st.write(sorted(ebc[broker_col_used].dropna().unique().tolist())[:25])


# -----------------------------
# Heatmap (State choropleth + Top Cities)
# -----------------------------
with tab_heatmap:
    st.subheader("Employer Footprint by State")

    if "State" not in gaps.columns:
        st.warning("State column not found. Ensure employer_geo.parquet merged correctly.")
    else:
        state_counts = (
            gaps.dropna(subset=["State"])
                .groupby("State", as_index=False)
                .agg(Employers=("Employer", "nunique"))
        )

        fig = px.choropleth(
            state_counts,
            locations="State",
            locationmode="USA-states",
            color="Employers",
            scope="usa",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Cities by Employer Count")
    if "City" not in gaps.columns or "State" not in gaps.columns:
        st.warning("City/State columns not found. Ensure employer_geo.parquet merged correctly.")
    else:
        city_counts = (
            gaps.dropna(subset=["City", "State"])
                .assign(CityState=lambda d: d["City"].astype(str).str.title() + ", " + d["State"].astype(str))
                .groupby("CityState", as_index=False)
                .agg(Employers=("Employer", "nunique"))
                .sort_values("Employers", ascending=False)
                .head(25)
        )
        st.dataframe(city_counts, use_container_width=True)

    # Optional: Upsell density by State (if product columns exist)
    st.subheader("Cross-sell Signal by State (Life → Disability)")
    if life_col and (std_col or ltd_col) and "State" in gaps.columns:
        life_flag = as_flag(gaps[life_col])
        std_flag = as_flag(gaps[std_col]) if std_col else pd.Series(False, index=gaps.index)
        ltd_flag = as_flag(gaps[ltd_col]) if ltd_col else pd.Series(False, index=gaps.index)

        # "Disability" considered present if either STD or LTD present
        dis_flag = std_flag | ltd_flag

        gaps_tmp = gaps.copy()
        gaps_tmp["LifeOnly_NoDisability"] = life_flag & (~dis_flag)

        by_state = (
            gaps_tmp.dropna(subset=["State"])
                .groupby("State", as_index=False)
                .agg(
                    Employers=("Employer", "nunique"),
                    LifeOnlyNoDis=("LifeOnly_NoDisability", "sum")
                )
        )
        by_state["LifeOnlyNoDis_Rate"] = (by_state["LifeOnlyNoDis"] / by_state["Employers"]).fillna(0)

        fig2 = px.choropleth(
            by_state,
            locations="State",
            locationmode="USA-states",
            color="LifeOnlyNoDis_Rate",
            scope="usa",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Interpretation: higher values indicate a denser cluster of Life-only employers lacking disability (STD/LTD), "
            "suggesting structured bundling/cross-sell signal."
        )
    else:
        st.info(
            "Cross-sell heatmap requires Life and STD/LTD flags in employer_product_matrix.parquet. "
            f"Detected: Life={life_col}, STD={std_col}, LTD={ltd_col}"
        )


# -----------------------------
# ZIP Intelligence (rule-based “AI-style” insights)
# -----------------------------
with tab_zip:
    st.subheader("ZIP Intelligence: Opportunity & Risk Signals")

    if not zip_query:
        st.info("Enter a 5-digit ZIP code in the sidebar to generate local intelligence insights.")
    else:
        if "ZIP" not in gaps.columns:
            st.error("ZIP column not found in gaps. Ensure employer_geo.parquet merged correctly.")
        else:
            # Filter all marts to the ZIP
            gaps_z = gaps[gaps["ZIP"].astype(str) == zip_query].copy()
            epc_z  = epc[epc["ZIP"].astype(str) == zip_query].copy() if "ZIP" in epc.columns else epc.iloc[0:0].copy()
            ebc_z  = ebc[ebc["ZIP"].astype(str) == zip_query].copy() if "ZIP" in ebc.columns else ebc.iloc[0:0].copy()

            st.markdown(f"### ZIP: **{zip_query}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Employers in ZIP", f"{gaps_z['Employer'].nunique():,}")
            c2.metric("Cities represented", f"{gaps_z['City'].nunique() if 'City' in gaps_z.columns else 0:,}")
            c3.metric("State", (gaps_z["State"].dropna().iloc[0] if len(gaps_z["State"].dropna()) else "—"))
            c4.metric("View", "AON Composite" if aon_only else "Market Benchmark")

            if gaps_z.empty:
                st.warning("No employers found for this ZIP in the current filtered view.")
            else:
                # Opportunity signals (Life → Disability)
                opp_notes = []

                if life_col:
                    life_flag = as_flag(gaps_z[life_col])
                else:
                    life_flag = pd.Series(False, index=gaps_z.index)

                std_flag = as_flag(gaps_z[std_col]) if std_col else pd.Series(False, index=gaps_z.index)
                ltd_flag = as_flag(gaps_z[ltd_col]) if ltd_col else pd.Series(False, index=gaps_z.index)
                dis_flag = std_flag | ltd_flag

                gaps_z["LifeOnly_NoDisability"] = life_flag & (~dis_flag)
                gaps_z["HasDisability_NoLife"] = dis_flag & (~life_flag)

                life_only_no_dis = int(gaps_z["LifeOnly_NoDisability"].sum())
                dis_only_no_life = int(gaps_z["HasDisability_NoLife"].sum())

                # Broker fragmentation risk (proxy)
                broker_col = None
                if not ebc_z.empty:
                    broker_col = find_col(ebc_z, ["Broker", "BROKER", "broker"]) or find_col_contains(ebc_z, "broker")

                brokers_n = int(ebc_z[broker_col].nunique()) if (broker_col and broker_col in ebc_z.columns) else 0
                employers_n_zip = int(gaps_z["Employer"].nunique())
                fragmentation = (brokers_n / employers_n_zip) if employers_n_zip else 0

                # Carrier concentration (proxy) from epc
                carrier_col = find_col(epc_z, ["Carrier", "CARRIER", "carrier"]) or find_col_contains(epc_z, "carrier")
                carriers_n = int(epc_z[carrier_col].nunique()) if (carrier_col and carrier_col in epc_z.columns) else 0

                # Metrics cards
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Life-only (no STD/LTD)", f"{life_only_no_dis:,}")
                k2.metric("Disability-only (no Life)", f"{dis_only_no_life:,}")
                k3.metric("Distinct brokers (proxy)", f"{brokers_n:,}")
                k4.metric("Distinct carriers (proxy)", f"{carriers_n:,}")

                # “AI-style” insight summary (rule-based, explainable)
                st.markdown("### Intelligence Summary")
                insights = []

                # Opportunity logic
                if life_only_no_dis > 0:
                    insights.append(
                        f"**Cross-sell signal:** {life_only_no_dis:,} employers show **Life-only** coverage without disability (STD/LTD). "
                        "In many markets, disability is a natural bundle/adjacent product."
                    )
                if dis_only_no_life > 0:
                    insights.append(
                        f"**Cross-sell signal:** {dis_only_no_life:,} employers show **Disability-only** coverage without Life. "
                        "Life can be an adjacent bundle candidate depending on size/industry mix."
                    )

                # Risk / structure logic
                if employers_n_zip >= 30 and fragmentation >= 0.6:
                    insights.append(
                        "**Market structure risk/opportunity:** High broker dispersion suggests a fragmented environment. "
                        "This can indicate competitive intensity, but also potential for structured consolidation or standardized packaging."
                    )
                elif employers_n_zip >= 30 and brokers_n <= max(3, int(employers_n_zip * 0.1)):
                    insights.append(
                        "**Market structure:** Broker concentration appears higher (fewer distinct brokers relative to employers). "
                        "This can imply stronger incumbency; focus may need to be account-specific rather than broad."
                    )

                if carriers_n > 0 and carriers_n <= 3 and employers_n_zip >= 20:
                    insights.append(
                        "**Carrier concentration:** Few carriers present (proxy). In some regions, this can shape bundling norms and pricing leverage."
                    )

                if not insights:
                    insights.append(
                        "Not enough signal density in this ZIP (or current filters too narrow) to generate strong opportunity/risk flags. "
                        "Try widening filters or switching Market Benchmark view."
                    )

                for s in insights:
                    st.write(f"- {s}")

                # Show top employers in ZIP (small table)
                with st.expander("Employers in ZIP (sample)"):
                    cols = ["Employer", "City", "State", "ZIP"]
                    cols = [c for c in cols if c in gaps_z.columns]
                    st.dataframe(
                        gaps_z[cols].drop_duplicates().head(50),
                        use_container_width=True
                    )

                # Optional: show broker summary
                if broker_col and broker_col in ebc_z.columns and not ebc_z.empty:
                    with st.expander("Broker presence in ZIP (proxy from filings)"):
                        bro = (
                            ebc_z.groupby(broker_col, as_index=False)
                                .agg(Employers=("Employer", "nunique"))
                                .sort_values("Employers", ascending=False)
                                .head(25)
                        )
                        st.dataframe(bro, use_container_width=True)

                # Optional: show carrier summary
                if carrier_col and carrier_col in epc_z.columns and not epc_z.empty:
                    with st.expander("Carrier presence in ZIP (proxy from filings)"):
                        car = (
                            epc_z.groupby(carrier_col, as_index=False)
                                .agg(Employers=("Employer", "nunique"))
                                .sort_values("Employers", ascending=False)
                                .head(25)
                        )
                        st.dataframe(car, use_container_width=True)


# -----------------------------
# Data tables (keeps everything accessible, so nothing “disappears”)
# -----------------------------
with tab_tables:
    st.subheader("Data Tables")

    st.markdown("#### Employer Product Matrix (gaps)")
    st.dataframe(gaps.head(500), use_container_width=True)

    st.markdown("#### Employer × Product × Carrier (epc)")
    st.dataframe(epc.head(500), use_container_width=True)

    st.markdown("#### Employer × Broker Commissions (ebc)")
    st.dataframe(ebc.head(500), use_container_width=True)

    st.caption(
        "Tip: Keep this tab for validation during builds. The other tabs are the executive-facing intelligence views."
    )