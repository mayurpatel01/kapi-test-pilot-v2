# app.py
# Voluntary Benefits Intelligence Insights (Form 5500)
# Streamlit: AON Composite filter + heatmaps + ZIP intelligence + gap matrix
# FIX: Removed st.stop() inside tabs (ZIP tab no longer prevents other tabs from rendering)

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

    if filename == "employer_geo.parquet":
        try:
            url = st.secrets.get("EMPLOYER_GEO_URL", "")
        except FileNotFoundError:
            url = os.getenv("EMPLOYER_GEO_URL", "")

        if not url:
            st.warning(
                "Geo mart missing (employer_geo.parquet) and EMPLOYER_GEO_URL not set. "
                "State/ZIP/City features are disabled until provided."
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
# State normalization (fix blank choropleth)
# =========================
VALID_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME",
    "MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI",
    "SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","PR"
}

STATE_NAME_TO_ABBR = {
    "ALABAMA":"AL","ALASKA":"AK","ARIZONA":"AZ","ARKANSAS":"AR","CALIFORNIA":"CA","COLORADO":"CO","CONNECTICUT":"CT",
    "DELAWARE":"DE","DISTRICT OF COLUMBIA":"DC","WASHINGTON DC":"DC","WASHINGTON, DC":"DC",
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
# AON Composite filter (STRICT broker-based)
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
        if re.match(rf"^{re.escape(r)}(\s+.*)?$", n):
            return True
    return False

def aon_employer_set(ebc_df: pd.DataFrame) -> set:
    if "Broker" not in ebc_df.columns:
        return set()
    aon_rows = ebc_df[ebc_df["Broker"].apply(is_aon_composite)]
    return set(aon_rows["Employer"].dropna().unique())


# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.markdown("## Global controls")
    aon_only = st.toggle("AON Composite Only", value=True)

    employer_search = st.text_input("Employer search", value="").strip()

    st.divider()
    st.markdown("## ZIP Intelligence")
    zip_query = st.text_input("ZIP (5-digit)", value="").strip()
    zip_query = re.sub(r"\D", "", zip_query)[:5]
    apply_zip_to_all = st.toggle("Apply ZIP filter to all dashboards", value=False)

    st.divider()
    st.markdown("## Geography")
    state_filter = []
    if "State" in gaps.columns and gaps["State"].notna().any():
        all_states = sorted({normalize_state(s) for s in gaps["State"].dropna().tolist()})
        all_states = [s for s in all_states if s in VALID_STATES]
        state_filter = st.multiselect("State filter", options=all_states, default=[])


# =========================
# Build base filtered view (AON + employer search + state)
# =========================
gaps_base, epc_base, ebc_base = gaps.copy(), epc.copy(), ebc.copy()

if aon_only:
    aon_emps = aon_employer_set(ebc_base)
    gaps_base = gaps_base[gaps_base["Employer"].isin(aon_emps)].copy()
    epc_base = epc_base[epc_base["Employer"].isin(aon_emps)].copy()
    ebc_base = ebc_base[ebc_base["Employer"].isin(aon_emps)].copy()

if employer_search:
    pat = norm(employer_search)
    gaps_base = gaps_base[gaps_base["Employer"].apply(lambda x: pat in norm(x))].copy()
    epc_base = epc_base[epc_base["Employer"].apply(lambda x: pat in norm(x))].copy()
    ebc_base = ebc_base[ebc_base["Employer"].apply(lambda x: pat in norm(x))].copy()

if state_filter and "State" in gaps_base.columns:
    gaps_base["StateNorm"] = gaps_base["State"].apply(normalize_state)
    gaps_base = gaps_base[gaps_base["StateNorm"].isin(state_filter)].copy()

    if "State" in epc_base.columns:
        epc_base["StateNorm"] = epc_base["State"].apply(normalize_state)
        epc_base = epc_base[epc_base["StateNorm"].isin(state_filter)].copy()

    if "State" in ebc_base.columns:
        ebc_base["StateNorm"] = ebc_base["State"].apply(normalize_state)
        ebc_base = ebc_base[ebc_base["StateNorm"].isin(state_filter)].copy()


# =========================
# ZIP-filtered view (optional for entire dashboard)
# =========================
gaps_view, epc_view, ebc_view = gaps_base, epc_base, ebc_base

if apply_zip_to_all and zip_query and "ZIP" in gaps_base.columns:
    gaps_view = gaps_base[gaps_base["ZIP"].astype(str) == zip_query].copy()
    if "ZIP" in epc_base.columns:
        epc_view = epc_base[epc_base["ZIP"].astype(str) == zip_query].copy()
    if "ZIP" in ebc_base.columns:
        ebc_view = ebc_base[ebc_base["ZIP"].astype(str) == zip_query].copy()


# =========================
# Top header row (always visible)
# =========================
top_left, top_mid, top_right = st.columns([1.2, 1.4, 1.6])

with top_left:
    st.markdown("### View")
    st.write("AON Composite" if aon_only else "Market benchmark")
    st.write(f"Employers: **{gaps_view['Employer'].nunique():,}**")

with top_mid:
    st.markdown("### Filters")
    st.write(f"Employer query: **{employer_search or '—'}**")
    st.write(f"ZIP: **{zip_query or '—'}**  | Apply to all: **{'ON' if apply_zip_to_all else 'OFF'}**")

with top_right:
    st.markdown("### Quick product mix")
    lf = as_flag(gaps_view[life_col]) if life_col in gaps_view.columns else pd.Series(False, index=gaps_view.index)
    sf = as_flag(gaps_view[std_col]) if std_col in gaps_view.columns else pd.Series(False, index=gaps_view.index)
    tf = as_flag(gaps_view[ltd_col]) if ltd_col in gaps_view.columns else pd.Series(False, index=gaps_view.index)
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
# Tabs
# =========================
tab_lens, tab_radar, tab_zip, tab_gaps, tab_tables = st.tabs(
    ["Lens Dashboard", "Opportunity Radar", "ZIP Intelligence", "Gaps Matrix", "Tables"]
)


# =========================
# Lens Dashboard
# =========================
with tab_lens:
    st.subheader("Lens Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employers", f"{gaps_view['Employer'].nunique():,}")
    c2.metric("States", f"{gaps_view['State'].nunique():,}" if "State" in gaps_view.columns else "—")
    c3.metric("Cities", f"{gaps_view['City'].nunique():,}" if "City" in gaps_view.columns else "—")
    c4.metric("AON Footprint", "ON" if aon_only else "OFF")

    st.markdown("### Employer footprint by State")
    if "State" in gaps_view.columns and gaps_view["State"].notna().any():
        tmp = gaps_view.copy()
        tmp["StateNorm"] = tmp["State"].apply(normalize_state)

        by_state = (
            tmp[tmp["StateNorm"].isin(VALID_STATES)]
            .groupby("StateNorm", as_index=False)
            .agg(Employers=("Employer", "nunique"))
        )

        fig = px.choropleth(
            by_state,
            locations="StateNorm",
            locationmode="USA-states",
            color="Employers",
            scope="usa",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("State not available (geo missing). Set EMPLOYER_GEO_URL in Secrets to enable maps.")

    st.markdown("### Top Cities")
    if "City" in gaps_view.columns and "State" in gaps_view.columns:
        city_counts = (
            gaps_view.dropna(subset=["City", "State"])
            .assign(CityState=lambda d: d["City"].astype(str).str.title() + ", " + d["State"].apply(normalize_state))
            .groupby("CityState", as_index=False)
            .agg(Employers=("Employer", "nunique"))
            .sort_values("Employers", ascending=False)
            .head(25)
            .reset_index(drop=True)
        )
        st.dataframe(city_counts, use_container_width=True)
    else:
        st.info("City/State not available (geo missing).")


# =========================
# Opportunity Radar
# =========================
with tab_radar:
    st.subheader("Opportunity Radar")

    lf = as_flag(gaps_view[life_col])
    sf = as_flag(gaps_view[std_col])
    tf = as_flag(gaps_view[ltd_col])
    dis = sf | tf

    tmp = gaps_view.copy()
    tmp["LifeOnly_NoDisability"] = lf & (~dis)
    tmp["DisOnly_NoLife"] = dis & (~lf)

    st.markdown("### Cross-sell whitespace summary")
    r1, r2, r3 = st.columns(3)
    r1.metric("Life-only (no STD/LTD)", f"{int(tmp['LifeOnly_NoDisability'].sum()):,}")
    r2.metric("Dis-only (no Life)", f"{int(tmp['DisOnly_NoLife'].sum()):,}")
    r3.metric("Bundled (Life + dis)", f"{int((lf & dis).sum()):,}")

    if "State" in tmp.columns and tmp["State"].notna().any():
        st.markdown("### Life-only whitespace rate by State")
        tmp2 = tmp.copy()
        tmp2["StateNorm"] = tmp2["State"].apply(normalize_state)
        agg = (
            tmp2[tmp2["StateNorm"].isin(VALID_STATES)]
            .groupby("StateNorm", as_index=False)
            .agg(Employers=("Employer", "nunique"), LifeOnlyNoDis=("LifeOnly_NoDisability", "sum"))
        )
        agg["LifeOnlyNoDis_Rate"] = (agg["LifeOnlyNoDis"] / agg["Employers"]).fillna(0)
        fig = px.choropleth(
            agg,
            locations="StateNorm",
            locationmode="USA-states",
            color="LifeOnlyNoDis_Rate",
            scope="usa",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("State not available. Provide geo mart to enable state radar.")

    st.markdown("### Broker + commission structure (proxy)")
    if "Broker" in ebc_view.columns and not ebc_view.empty:
        ebc2 = ebc_view.copy()
        ebc2["total_commissions"] = pd.to_numeric(ebc2["total_commissions"], errors="coerce").fillna(0)
        by_broker = (
            ebc2.groupby("Broker", as_index=False)
            .agg(Employers=("Employer", "nunique"), TotalCommissions=("total_commissions", "sum"))
            .sort_values("TotalCommissions", ascending=False)
            .head(25)
            .reset_index(drop=True)
        )
        st.dataframe(by_broker, use_container_width=True)
    else:
        st.info("Broker/commission data not available in current view.")


# =========================
# ZIP Intelligence (no st.stop; does not block other tabs)
# =========================
with tab_zip:
    st.subheader("ZIP Intelligence")
    st.caption(
        "This tab evaluates the ZIP you entered. Turn ON 'Apply ZIP filter to all dashboards' if you want Lens/Radar/Gaps to also filter to the ZIP."
    )

    if not zip_query:
        st.info("Enter a 5-digit ZIP in the sidebar to generate intelligence.")
    elif "ZIP" not in gaps_base.columns:
        st.warning("ZIP not available (geo missing). Set EMPLOYER_GEO_URL in Secrets.")
    else:
        gz = gaps_base[gaps_base["ZIP"].astype(str) == zip_query].copy()
        ez = ebc_base[ebc_base["ZIP"].astype(str) == zip_query].copy() if "ZIP" in ebc_base.columns else ebc_base.iloc[0:0].copy()
        pz = epc_base[epc_base["ZIP"].astype(str) == zip_query].copy() if "ZIP" in epc_base.columns else epc_base.iloc[0:0].copy()

        if gz.empty:
            st.warning("No employers found for this ZIP under the current base filters (AON toggle / employer search / state filter).")
        else:
            lf = as_flag(gz[life_col])
            sf = as_flag(gz[std_col])
            tf = as_flag(gz[ltd_col])
            dis = sf | tf

            gz["Expandable_Life_to_Disability"] = lf & (~dis)
            gz["Expandable_Disability_to_Life"] = dis & (~lf)
            gz["Bundled"] = lf & dis

            n_emps = int(gz["Employer"].nunique())
            n_life_to_dis = int(gz["Expandable_Life_to_Disability"].sum())
            n_dis_to_life = int(gz["Expandable_Disability_to_Life"].sum())
            n_bundled = int(gz["Bundled"].sum())

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

            st.markdown(f"### ZIP: **{zip_query}**")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Employers", f"{n_emps:,}")
            k2.metric("Expandable: Life → Dis", f"{n_life_to_dis:,}")
            k3.metric("Expandable: Dis → Life", f"{n_dis_to_life:,}")
            k4.metric("Distinct brokers", f"{brokers_n:,}")
            k5.metric("Total commissions", f"${total_comm:,.0f}")

            st.markdown("### Opportunity & Risk Signals")
            signals = []

            if n_emps < 10:
                signals.append("**Low signal density:** small employer count in this ZIP. Treat as directional, not definitive.")

            if n_life_to_dis >= max(3, int(0.15 * n_emps)):
                signals.append(
                    f"**Bundling whitespace (Life → Disability):** {n_life_to_dis:,} employers have Life without STD/LTD. "
                    "Clean candidate pool for disability attach strategies."
                )

            if n_dis_to_life >= max(3, int(0.15 * n_emps)):
                signals.append(
                    f"**Adjacency whitespace (Disability → Life):** {n_dis_to_life:,} employers have disability without Life. "
                    "Candidate pool for Life attach depending on segment fit."
                )

            if n_bundled >= max(5, int(0.30 * n_emps)):
                signals.append(
                    f"**Bundling norm present:** {n_bundled:,} employers already bundle Life+Disability. "
                    "Suggests packaged offers are accepted in this local market."
                )

            if n_emps >= 25 and frag >= 0.6:
                signals.append(
                    f"**Broker fragmentation:** {brokers_n:,} brokers across {n_emps:,} employers (ratio {frag:.2f}). "
                    "Fragmented structure can favor standardization, packaging, or consolidation plays."
                )

            if n_emps >= 25 and brokers_n <= 3:
                signals.append(
                    f"**Incumbency risk:** only {brokers_n:,} broker(s) appear in filings. "
                    "Likely entrenched relationships; prioritize targeted displacement or specialty positioning."
                )

            if comm_per_emp > 0:
                signals.append(
                    f"**Commission density (proxy):** approx. ${comm_per_emp:,.0f} per employer. "
                    "Higher values suggest stronger monetization; prioritize account-based motion."
                )

            if carriers_n and n_emps >= 20 and carriers_n <= 3:
                signals.append(
                    f"**Carrier concentration (proxy):** only {carriers_n:,} carriers appear. "
                    "May influence pricing leverage and bundling norms; align carrier strategy accordingly."
                )

            if not signals:
                signals.append("No strong signals fired under current filters. Try Market benchmark view or remove State filter.")

            for s in signals:
                st.write(f"- {s}")

            st.markdown("### Expandable Employers in this ZIP")
            show_cols = [c for c in ["Employer", "City", "State", "ZIP"] if c in gz.columns]
            out = gz[show_cols + ["Expandable_Life_to_Disability", "Expandable_Disability_to_Life", "Bundled"]].copy()
            out = out.drop_duplicates(subset=["Employer"]).reset_index(drop=True)

            def highlight_expand(row):
                if bool(row.get("Expandable_Life_to_Disability", False)) or bool(row.get("Expandable_Disability_to_Life", False)):
                    return ["background-color: rgba(0, 200, 0, 0.14)"] * len(row)
                return [""] * len(row)

            st.dataframe(out.style.apply(highlight_expand, axis=1), use_container_width=True)

            if not ez.empty and "Broker" in ez.columns:
                st.markdown("### Brokers in this ZIP (proxy)")
                bro = ez.copy()
                bro["total_commissions"] = pd.to_numeric(bro["total_commissions"], errors="coerce").fillna(0)
                bro["Is_AON_Composite"] = bro["Broker"].apply(is_aon_composite)

                bro_agg = (
                    bro.groupby(["Broker", "Is_AON_Composite"], as_index=False)
                    .agg(Employers=("Employer", "nunique"), TotalCommissions=("total_commissions", "sum"))
                    .sort_values(["Is_AON_Composite", "Employers", "TotalCommissions"], ascending=[False, False, False])
                    .head(50)
                    .reset_index(drop=True)
                )

                def highlight_aon(row):
                    if bool(row.get("Is_AON_Composite", False)):
                        return ["background-color: rgba(0, 140, 255, 0.12); font-weight: 600;"] * len(row)
                    return [""] * len(row)

                st.dataframe(bro_agg.style.apply(highlight_aon, axis=1), use_container_width=True)


# =========================
# Gaps Matrix (green/red)
# =========================
with tab_gaps:
    st.subheader("Gaps Matrix (Life / STD / LTD)")
    st.caption("Green = present (1), Red = missing (0). Uses the current dashboard view filters.")

    cols = ["Employer", life_col, std_col, ltd_col] + ([c for c in ["City", "State", "ZIP"] if c in gaps_view.columns])
    view = gaps_view[cols].drop_duplicates(subset=["Employer"]).head(5000).reset_index(drop=True)

    def style_flag(val):
        try:
            v = int(val)
        except Exception:
            v = 0
        if v == 1:
            return "background-color: rgba(0, 200, 0, 0.25); font-weight: 600;"
        return "background-color: rgba(255, 0, 0, 0.20);"

    styled = view.style.applymap(style_flag, subset=[life_col, std_col, ltd_col])
    st.dataframe(styled, use_container_width=True)

    with st.expander("Download current gaps view (CSV)"):
        st.download_button(
            "Download",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="gaps_view.csv",
            mime="text/csv",
        )


# =========================
# Tables (raw validation)
# =========================
with tab_tables:
    st.subheader("Tables (raw)")
    st.markdown("#### employer_product_matrix.parquet (gaps)")
    st.dataframe(gaps_view.head(500).reset_index(drop=True), use_container_width=True)

    st.markdown("#### employer_broker_commissions.parquet (ebc)")
    st.dataframe(ebc_view.head(500).reset_index(drop=True), use_container_width=True)

    st.markdown("#### employer_product_carrier.parquet (epc)")
    st.dataframe(epc_view.head(500).reset_index(drop=True), use_container_width=True)

    if geo is not None and not geo.empty:
        st.markdown("#### employer_geo.parquet (geo)")
        st.dataframe(geo.head(500).reset_index(drop=True), use_container_width=True)
    else:
        st.info("Geo mart not loaded in this environment.")