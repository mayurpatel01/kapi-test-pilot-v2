# app.py
# Voluntary Benefits Intelligence Insights (Form 5500)
# Streamlit: AON Composite filtering + heatmaps + ZIP intelligence + gap matrix

import os
import re
import urllib.request
from pathlib import Path

import pandas as pd
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
    "Strategy-ready signals from public Form 5500 data: portfolio footprint, cross-sell whitespace, and regional opportunity structure."
)

DATA_DIR = Path("data/marts")


# =========================
# Auth (local + cloud)
# =========================
def require_password():
    """
    Uses Streamlit Secrets if available. Falls back to env var APP_PASSWORD.
    Does not crash with missing secrets file; provides clear message.
    """
    try:
        pw = st.secrets.get("APP_PASSWORD", "")
    except FileNotFoundError:
        pw = os.getenv("APP_PASSWORD", "")

    if not pw:
        st.error("APP_PASSWORD not set. Add it to Streamlit Secrets (or set env var APP_PASSWORD).")
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
    Currently used for employer_geo.parquet.
    """
    path = DATA_DIR / filename
    if path.exists():
        return

    # Only geo is expected to be externalized for now.
    if filename == "employer_geo.parquet":
        try:
            url = st.secrets.get("EMPLOYER_GEO_URL", "")
        except FileNotFoundError:
            url = os.getenv("EMPLOYER_GEO_URL", "")

        if not url:
            # Don't crash the entire app: show warning and continue without geo.
            # Features that need geo will degrade gracefully.
            st.warning(
                "Geo mart missing (employer_geo.parquet) and EMPLOYER_GEO_URL is not set in Streamlit Secrets. "
                "State/ZIP/City features will be disabled until provided."
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


# =========================
# Load marts
# =========================
with st.spinner("Loading data marts..."):
    # Required marts
    epc = ensure_str(load_parquet("employer_product_carrier.parquet"))      # Employer x Product x Carrier (+ Covered_Lives)
    ebc = ensure_str(load_parquet("employer_broker_commissions.parquet"))   # Employer x Broker (+ total_commissions)
    gaps = ensure_str(load_parquet("employer_product_matrix.parquet"))      # Employer x Life/STD/LTD flags

    # Optional geo mart (downloaded on cloud if EMPLOYER_GEO_URL is set)
    try:
        geo = ensure_str(load_parquet("employer_geo.parquet"))
    except FileNotFoundError:
        geo = pd.DataFrame(columns=["Employer", "State", "ZIP", "City", "EIN"])


# =========================
# Merge geo into marts (if available)
# =========================
geo_cols = [c for c in ["Employer", "State", "ZIP", "City", "EIN"] if c in geo.columns]
geo = geo[geo_cols].copy() if len(geo_cols) else pd.DataFrame(columns=["Employer", "State", "ZIP", "City", "EIN"])

if "Employer" in geo.columns and not geo.empty:
    gaps = gaps.merge(geo, on="Employer", how="left")
    epc = epc.merge(geo, on="Employer", how="left")
    ebc = ebc.merge(geo, on="Employer", how="left")


# =========================
# Product columns (confirmed)
# =========================
life_col, std_col, ltd_col = "Life", "STD", "LTD"

def as_flag(series: pd.Series) -> pd.Series:
    s = series.fillna(0)
    if s.dtype == "object":
        s = s.astype(str).str.strip()
        return s.isin(["1", "TRUE", "True", "true", "Y", "YES", "Yes", "yes"])
    return s.astype(int).fillna(0) == 1


# =========================
# AON Composite filter (STRICT, broker-based)
# =========================
AON_ENTITIES = [
    "AON CORPORATION",
    "AON RISK SERVICES",
    "AON HEWITT",
    "AON CONSULTING",
    "AON SOLUTIONS",
]

def is_aon_composite(broker_name: str) -> bool:
    n = norm(broker_name)
    for root in AON_ENTITIES:
        r = norm(root)
        # Must START with the approved root, avoids false matches like DATAONLINE
        if re.match(rf"^{re.escape(r)}(\s+.*)?$", n):
            return True
    return False

def aon_employer_set(ebc_df: pd.DataFrame) -> set:
    if "Broker" not in ebc_df.columns:
        return set()
    aon_rows = ebc_df[ebc_df["Broker"].apply(is_aon_composite)]
    return set(aon_rows["Employer"].dropna().unique())


# =========================
# Global controls (visible regardless of tab)
# =========================
with st.sidebar:
    st.markdown("## Global controls")
    aon_only = st.toggle("AON Composite Only", value=True)
    employer_search = st.text_input("Employer search", value="").strip()

    zip_query = st.text_input("ZIP search (5-digit)", value="").strip()
    zip_query = re.sub(r"\D", "", zip_query)[:5]

    st.divider()
    st.markdown("## Geography")
    if "State" in gaps.columns:
        states = sorted([s for s in gaps["State"].dropna().unique().tolist() if str(s).strip()])
    else:
        states = []
    state_filter = st.multiselect("State filter", options=states, default=[])

# Apply AON filter (consistent across marts)
if aon_only:
    aon_emps = aon_employer_set(ebc)
    gaps = gaps[gaps["Employer"].isin(aon_emps)].copy()
    epc = epc[epc["Employer"].isin(aon_emps)].copy()
    ebc = ebc[ebc["Employer"].isin(aon_emps)].copy()
else:
    aon_emps = None

# Apply Employer search (global)
if employer_search:
    pat = norm(employer_search)
    gaps = gaps[gaps["Employer"].apply(lambda x: pat in norm(x))].copy()
    epc = epc[epc["Employer"].apply(lambda x: pat in norm(x))].copy()
    ebc = ebc[ebc["Employer"].apply(lambda x: pat in norm(x))].copy()

# Apply State filter (if geo present)
if state_filter and "State" in gaps.columns:
    gaps = gaps[gaps["State"].isin(state_filter)].copy()
    if "State" in epc.columns:
        epc = epc[epc["State"].isin(state_filter)].copy()
    if "State" in ebc.columns:
        ebc = ebc[ebc["State"].isin(state_filter)].copy()


# =========================
# Main “always visible” header row (not buried in tabs)
# =========================
top_left, top_mid, top_right = st.columns([1.2, 1.2, 1.6])

with top_left:
    st.markdown("### View")
    st.write("AON Composite" if aon_only else "Market benchmark")
    st.write(f"Employers: **{gaps['Employer'].nunique():,}**")

with top_mid:
    st.markdown("### Searches")
    st.write(f"Employer query: **{employer_search or '—'}**")
    st.write(f"ZIP query: **{zip_query or '—'}**")

with top_right:
    st.markdown("### Quick product mix (filtered view)")
    lf = as_flag(gaps[life_col]) if life_col in gaps.columns else pd.Series(False, index=gaps.index)
    sf = as_flag(gaps[std_col]) if std_col in gaps.columns else pd.Series(False, index=gaps.index)
    tf = as_flag(gaps[ltd_col]) if ltd_col in gaps.columns else pd.Series(False, index=gaps.index)
    dis = sf | tf

    life_only = int((lf & ~dis).sum())
    dis_only = int((dis & ~lf).sum())
    both = int((lf & dis).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Life-only", f"{life_only:,}")
    c2.metric("Disability-only", f"{dis_only:,}")
    c3.metric("Bundled", f"{both:,}")

st.divider()


# =========================
# ZIP Intelligence panel (always accessible)
# =========================
def zip_intelligence(zip_code: str):
    if not zip_code:
        st.info("Enter a ZIP code in the left sidebar to generate local opportunity/risk signals.")
        return
    if "ZIP" not in gaps.columns:
        st.warning("ZIP not available (geo mart missing). Set EMPLOYER_GEO_URL in Streamlit Secrets to enable ZIP features.")
        return

    gz = gaps[gaps["ZIP"].astype(str) == zip_code].copy()
    ez = ebc[ebc["ZIP"].astype(str) == zip_code].copy() if "ZIP" in ebc.columns else ebc.iloc[0:0].copy()
    pz = epc[epc["ZIP"].astype(str) == zip_code].copy() if "ZIP" in epc.columns else epc.iloc[0:0].copy()

    st.markdown(f"## ZIP Intelligence — **{zip_code}**")
    if gz.empty:
        st.warning("No employers found for this ZIP in the current filtered view.")
        return

    # Signals
    lf = as_flag(gz[life_col])
    sf = as_flag(gz[std_col])
    tf = as_flag(gz[ltd_col])
    dis = sf | tf

    gz["LifeOnly_NoDisability"] = lf & (~dis)
    gz["DisOnly_NoLife"] = dis & (~lf)

    n_emps = int(gz["Employer"].nunique())
    n_life_only = int(gz["LifeOnly_NoDisability"].sum())
    n_dis_only = int(gz["DisOnly_NoLife"].sum())
    n_bundled = int((lf & dis).sum())

    brokers_n = int(ez["Broker"].nunique()) if ("Broker" in ez.columns and not ez.empty) else 0
    frag = (brokers_n / n_emps) if n_emps else 0.0

    total_comm = float(pd.to_numeric(ez["total_commissions"], errors="coerce").fillna(0).sum()) if ("total_commissions" in ez.columns and not ez.empty) else 0.0
    comm_per_emp = (total_comm / n_emps) if n_emps else 0.0

    carrier_col = None
    for c in pz.columns:
        if c.lower() == "carrier" or "carrier" in c.lower():
            carrier_col = c
            break
    carriers_n = int(pz[carrier_col].nunique()) if (carrier_col and not pz.empty) else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Employers", f"{n_emps:,}")
    k2.metric("Life-only (no dis)", f"{n_life_only:,}")
    k3.metric("Dis-only (no life)", f"{n_dis_only:,}")
    k4.metric("Distinct brokers", f"{brokers_n:,}")
    k5.metric("Total commissions", f"${total_comm:,.0f}")

    # Memo-style rules
    st.markdown("### Opportunity & Risk Signals")
    signals = []

    if n_emps < 10:
        signals.append("**Low signal density:** small employer count in this ZIP. Treat as directional, not definitive.")

    if n_life_only >= max(3, int(0.15 * n_emps)):
        signals.append(
            f"**Bundling whitespace (Life → Disability):** {n_life_only:,} Life-only employers lack STD/LTD. "
            "Strong candidate pool for disability attach strategies."
        )

    if n_dis_only >= max(3, int(0.15 * n_emps)):
        signals.append(
            f"**Adjacency whitespace (Disability → Life):** {n_dis_only:,} disability-only employers lack Life. "
            "Candidate pool for Life attach depending on size/industry mix."
        )

    if n_bundled >= max(5, int(0.30 * n_emps)):
        signals.append(
            f"**Bundling norm present:** {n_bundled:,} employers already bundle Life+Disability. "
            "This suggests a market where packaged offers are accepted."
        )

    if n_emps >= 25 and frag >= 0.6:
        signals.append(
            f"**Broker fragmentation:** {brokers_n:,} brokers across {n_emps:,} employers (ratio {frag:.2f}). "
            "Fragmented structure can favor standardized packaging, consolidation plays, or channel partnerships."
        )

    if n_emps >= 25 and brokers_n <= 3:
        signals.append(
            f"**Incumbency risk:** only {brokers_n:,} broker(s) appear in filings. "
            "Likely entrenched relationships; prioritize targeted displacement, co-sell, or specialty positioning."
        )

    if comm_per_emp > 0:
        signals.append(
            f"**Commission density:** approx. ${comm_per_emp:,.0f} per employer (proxy). "
            "Higher values indicate stronger monetization potential; prioritize account-based motion."
        )

    if carriers_n and n_emps >= 20 and carriers_n <= 3:
        signals.append(
            f"**Carrier concentration (proxy):** only {carriers_n:,} carriers appear. "
            "May influence pricing leverage and packaging norms; align carrier strategy accordingly."
        )

    if not signals:
        signals.append("No strong signals fired under current filters. Try widening to Market benchmark or removing State filter.")

    for s in signals:
        st.write(f"- {s}")

    with st.expander("Employers in this ZIP (sample)"):
        cols = [c for c in ["Employer", "City", "State", "ZIP"] if c in gz.columns]
        st.dataframe(gz[cols].drop_duplicates().head(100), use_container_width=True)

    if not ez.empty and "Broker" in ez.columns:
        with st.expander("Brokers in this ZIP (proxy)"):
            bro = (
                ez.groupby("Broker", as_index=False)
                .agg(Employers=("Employer", "nunique"), TotalCommissions=("total_commissions", "sum"))
                .sort_values(["Employers", "TotalCommissions"], ascending=False)
                .head(25)
            )
            st.dataframe(bro, use_container_width=True)


# Show ZIP intelligence panel at top
zip_intelligence(zip_query)
st.divider()


# =========================
# Tabs (keep your other dashboards + tables)
# =========================
tab_lens, tab_radar, tab_gaps, tab_tables = st.tabs(
    ["Lens Dashboard", "Opportunity Radar", "Gaps Matrix", "Tables"]
)

# ---- Lens Dashboard ----
with tab_lens:
    st.subheader("Lens Dashboard (filtered view)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employers", f"{gaps['Employer'].nunique():,}")
    c2.metric("States", f"{gaps['State'].nunique():,}" if "State" in gaps.columns else "—")
    c3.metric("Cities", f"{gaps['City'].nunique():,}" if "City" in gaps.columns else "—")
    c4.metric("AON Footprint", "ON" if aon_only else "OFF")

    if "State" in gaps.columns and gaps["State"].notna().any():
        st.markdown("### Employer footprint by State")
        by_state = (
            gaps.dropna(subset=["State"])
            .groupby("State", as_index=False)
            .agg(Employers=("Employer", "nunique"))
        )
        fig = px.choropleth(by_state, locations="State", locationmode="USA-states", color="Employers", scope="usa")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Top Cities")
        if "City" in gaps.columns:
            by_city = (
                gaps.dropna(subset=["City", "State"])
                .assign(CityState=lambda d: d["City"].str.title() + ", " + d["State"])
                .groupby("CityState", as_index=False)
                .agg(Employers=("Employer", "nunique"))
                .sort_values("Employers", ascending=False)
                .head(25)
            )
            st.dataframe(by_city, use_container_width=True)
    else:
        st.info("Geo not available (State/City). Set EMPLOYER_GEO_URL in Streamlit Secrets to enable maps.")

# ---- Opportunity Radar ----
with tab_radar:
    st.subheader("Opportunity Radar")

    lf = as_flag(gaps[life_col])
    sf = as_flag(gaps[std_col])
    tf = as_flag(gaps[ltd_col])
    dis = sf | tf

    tmp = gaps.copy()
    tmp["LifeOnly_NoDisability"] = lf & (~dis)
    tmp["DisOnly_NoLife"] = dis & (~lf)

    st.markdown("### Cross-sell whitespace summary (filtered view)")
    r1, r2, r3 = st.columns(3)
    r1.metric("Life-only (no STD/LTD)", f"{int(tmp['LifeOnly_NoDisability'].sum()):,}")
    r2.metric("Dis-only (no Life)", f"{int(tmp['DisOnly_NoLife'].sum()):,}")
    r3.metric("Bundled (Life + dis)", f"{int((lf & dis).sum()):,}")

    if "State" in tmp.columns and tmp["State"].notna().any():
        st.markdown("### Life-only whitespace rate by State")
        agg = (
            tmp.dropna(subset=["State"])
            .groupby("State", as_index=False)
            .agg(Employers=("Employer", "nunique"), LifeOnlyNoDis=("LifeOnly_NoDisability", "sum"))
        )
        agg["LifeOnlyNoDis_Rate"] = (agg["LifeOnlyNoDis"] / agg["Employers"]).fillna(0)
        fig = px.choropleth(agg, locations="State", locationmode="USA-states", color="LifeOnlyNoDis_Rate", scope="usa")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("State not available. Provide geo mart to enable state opportunity radar.")

    st.markdown("### Broker + commission structure (proxy)")
    if "Broker" in ebc.columns and not ebc.empty:
        ebc2 = ebc.copy()
        ebc2["total_commissions"] = pd.to_numeric(ebc2["total_commissions"], errors="coerce").fillna(0)
        by_broker = (
            ebc2.groupby("Broker", as_index=False)
            .agg(Employers=("Employer", "nunique"), TotalCommissions=("total_commissions", "sum"))
            .sort_values("TotalCommissions", ascending=False)
            .head(25)
        )
        st.dataframe(by_broker, use_container_width=True)
    else:
        st.info("Broker/commission table not available in current filter context.")

# ---- Gaps Matrix (green/red) ----
with tab_gaps:
    st.subheader("Gaps Matrix (Life / STD / LTD)")
    st.caption("Green = present (1), Red = missing (0). Use filters/search above to narrow the view.")

    view = gaps[["Employer", life_col, std_col, ltd_col] + ([c for c in ["City", "State", "ZIP"] if c in gaps.columns])].copy()
    # keep reasonable row count for UI
    view = view.drop_duplicates(subset=["Employer"]).head(5000)

    # style function
    def style_flag(val):
        try:
            v = int(val)
        except Exception:
            v = 0
        if v == 1:
            return "background-color: rgba(0, 200, 0, 0.25); font-weight: 600;"
        return "background-color: rgba(255, 0, 0, 0.20);"

    styled = (
        view.style
        .applymap(style_flag, subset=[life_col, std_col, ltd_col])
    )
    st.dataframe(styled, use_container_width=True)

    with st.expander("Download current gaps view (CSV)"):
        st.download_button(
            "Download",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="gaps_view.csv",
            mime="text/csv",
        )

# ---- Tables (raw) ----
with tab_tables:
    st.subheader("Tables (raw, for validation)")

    st.markdown("#### employer_product_matrix.parquet (gaps)")
    st.dataframe(gaps.head(500), use_container_width=True)

    st.markdown("#### employer_broker_commissions.parquet (ebc)")
    st.dataframe(ebc.head(500), use_container_width=True)

    st.markdown("#### employer_product_carrier.parquet (epc)")
    st.dataframe(epc.head(500), use_container_width=True)

    if geo is not None and not geo.empty:
        st.markdown("#### employer_geo.parquet (geo)")
        st.dataframe(geo.head(500), use_container_width=True)
    else:
        st.info("Geo mart not loaded in this environment.")