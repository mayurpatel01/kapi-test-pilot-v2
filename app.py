import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="Kapi Test Pilot",
    layout="wide",
    page_icon="📊",
)

# ---------------------------
# Password gate (uses Streamlit Secrets)
# ---------------------------
def require_password():
    pw = st.secrets.get("APP_PASSWORD", "")
    if not pw:
        st.warning("APP_PASSWORD not set in Streamlit Secrets.")
        st.stop()

    if "authed" not in st.session_state:
        st.session_state.authed = False

    if not st.session_state.authed:
        st.title("Kapi Test Pilot")
        st.caption("Enter the password to access the dashboard.")
        entered = st.text_input("Password", type="password")
        if st.button("Login"):
            if entered == pw:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("Wrong password")
        st.stop()


require_password()

# ---------------------------
# Helpers: load marts if present, otherwise sample data
# ---------------------------
DATA_DIR = Path("data") / "marts"

@st.cache_data(show_spinner=False)
def load_sample_facts() -> pd.DataFrame:
    # Simple demo “fact table”
    return pd.DataFrame(
        {
            "Product": ["Life", "Life", "STD", "STD", "LTD", "LTD", "Life", "STD", "LTD"],
            "Broker":  ["Broker A", "Broker B", "Broker A", "Broker C", "Broker B", "Broker C", "Broker A", "Broker B", "Broker C"],
            "Employer": ["E1", "E2", "E3", "E4", "E2", "E5", "E6", "E6", "E6"],
            "Commissions Paid": [250000, 150000, 110000, 90000, 70000, 65000, 40000, 30000, 20000],
            "Carrier": ["Carrier X", "Carrier Y", "Carrier X", "Carrier Z", "Carrier Y", "Carrier Z", "Carrier X", "Carrier Y", "Carrier Z"],
        }
    )

@st.cache_data(show_spinner=False)
def load_sample_gap_matrix() -> pd.DataFrame:
    # Demo “gap” matrix
    return pd.DataFrame(
        {
            "Employer": ["E1", "E2", "E3", "E4", "E5", "E6"],
            "Life": [1, 1, 0, 1, 0, 1],
            "STD":  [1, 0, 1, 0, 0, 1],
            "LTD":  [0, 1, 1, 0, 1, 1],
        }
    )

@st.cache_data(show_spinner=False)
def try_load_parquet(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None

@st.cache_data(show_spinner=False)
def load_data():
    """
    If Parquet marts exist, use them.
    Otherwise fall back to sample data so the app never breaks.
    """
    facts = try_load_parquet(DATA_DIR / "employer_broker_carrier.parquet")
    gaps = try_load_parquet(DATA_DIR / "employer_product_matrix.parquet")

    using_real = facts is not None and gaps is not None

    if not using_real:
        facts = load_sample_facts()
        gaps = load_sample_gap_matrix()

    return facts, gaps, using_real


facts_df, gaps_df, using_real_data = load_data()

# ---------------------------
# UI Header
# ---------------------------
st.title("📊 Kapi Test Pilot — Form 5500 Market Intelligence")

if using_real_data:
    st.success("Real Form 5500 marts detected ✅")
else:
    st.info("Using sample data for now. Next step: generate Parquet marts from Form 5500 datasets.")

st.caption("Filter by product, broker, and carrier. Review gaps, then use AI Insights to ask questions.")

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Filters")

product_choices = ["Life", "STD", "LTD"]
product = st.sidebar.selectbox("Product", product_choices, index=0)

broker_choices = ["All"] + sorted(facts_df["Broker"].dropna().unique().tolist()) if "Broker" in facts_df.columns else ["All"]
carrier_choices = ["All"] + sorted(facts_df["Carrier"].dropna().unique().tolist()) if "Carrier" in facts_df.columns else ["All"]

broker = st.sidebar.selectbox("Broker", broker_choices, index=0)
carrier = st.sidebar.selectbox("Carrier", carrier_choices, index=0)

employer_search = st.sidebar.text_input("Employer contains (optional)", value="").strip()

# Apply filters to facts
filtered = facts_df.copy()
if "Product" in filtered.columns:
    filtered = filtered[filtered["Product"] == product]
if broker != "All" and "Broker" in filtered.columns:
    filtered = filtered[filtered["Broker"] == broker]
if carrier != "All" and "Carrier" in filtered.columns:
    filtered = filtered[filtered["Carrier"] == carrier]
if employer_search and "Employer" in filtered.columns:
    filtered = filtered[filtered["Employer"].astype(str).str.contains(employer_search, case=False, na=False)]

# ---------------------------
# KPI row
# ---------------------------
k1, k2, k3, k4 = st.columns(4)

unique_employers = filtered["Employer"].nunique() if "Employer" in filtered.columns else 0
unique_brokers = filtered["Broker"].nunique() if "Broker" in filtered.columns else 0
unique_carriers = filtered["Carrier"].nunique() if "Carrier" in filtered.columns else 0
total_comm = filtered["Commissions Paid"].sum() if "Commissions Paid" in filtered.columns else 0

k1.metric("Unique Employers", f"{unique_employers:,}")
k2.metric("Unique Brokers", f"{unique_brokers:,}")
k3.metric("Unique Carriers", f"{unique_carriers:,}")
k4.metric("Total Commissions", f"${total_comm:,.0f}")

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(["Overview", "Brokers", "Carriers", "Gaps", "AI Insights"])

# ---------------------------
# Tab: Overview
# ---------------------------
with tabs[0]:
    st.subheader("Overview")

    if filtered.empty:
        st.warning("No rows match your filters.")
    else:
        left, right = st.columns([1, 1])

        with left:
            st.markdown("**Commissions by Broker**")
            by_broker = (
                filtered.groupby("Broker", as_index=False)
                .agg(commissions=("Commissions Paid", "sum"), employers=("Employer", "nunique"))
                .sort_values("commissions", ascending=False)
            )
            fig = px.bar(by_broker, x="Broker", y="commissions", title="Commissions by Broker")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("**Commissions by Carrier**")
            by_carrier = (
                filtered.groupby("Carrier", as_index=False)
                .agg(commissions=("Commissions Paid", "sum"), employers=("Employer", "nunique"))
                .sort_values("commissions", ascending=False)
            )
            fig2 = px.bar(by_carrier, x="Carrier", y="commissions", title="Commissions by Carrier")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Filtered rows**")
        st.dataframe(filtered, use_container_width=True, height=300)

# ---------------------------
# Tab: Brokers
# ---------------------------
with tabs[1]:
    st.subheader("Brokers")

    if filtered.empty:
        st.warning("No rows match your filters.")
    else:
        summary = (
            filtered.groupby("Broker", as_index=False)
            .agg(
                unique_employers=("Employer", "nunique"),
                total_commissions=("Commissions Paid", "sum"),
                unique_carriers=("Carrier", "nunique"),
            )
            .sort_values("total_commissions", ascending=False)
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.bar(summary, x="Broker", y="unique_employers", title="Unique Employers by Broker")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(summary, x="Broker", y="total_commissions", title="Total Commissions by Broker")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(summary, use_container_width=True)

# ---------------------------
# Tab: Carriers
# ---------------------------
with tabs[2]:
    st.subheader("Carriers")

    if filtered.empty:
        st.warning("No rows match your filters.")
    else:
        summary = (
            filtered.groupby("Carrier", as_index=False)
            .agg(
                unique_employers=("Employer", "nunique"),
                total_commissions=("Commissions Paid", "sum"),
                unique_brokers=("Broker", "nunique"),
            )
            .sort_values("total_commissions", ascending=False)
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.bar(summary, x="Carrier", y="unique_employers", title="Unique Employers by Carrier")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(summary, x="Carrier", y="total_commissions", title="Total Commissions by Carrier")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(summary, use_container_width=True)

# ---------------------------
# Tab: Gaps (red/green matrix)
# ---------------------------
with tabs[3]:
    st.subheader("Gaps")

    if gaps_df.empty:
        st.warning("Gap matrix is empty.")
    else:
        # If your real marts use different column names later, we’ll adjust.
        expected_cols = ["Employer", "Life", "STD", "LTD"]
        missing = [c for c in expected_cols if c not in gaps_df.columns]
        if missing:
            st.error(f"Gap matrix is missing expected columns: {missing}")
        else:
            # optional employer filter
            g = gaps_df.copy()
            if employer_search:
                g = g[g["Employer"].astype(str).str.contains(employer_search, case=False, na=False)]

            # Show only top N to keep things fast
            max_rows = st.slider("Max rows to display", min_value=25, max_value=500, value=150, step=25)
            g = g.head(max_rows)

            def color_cell(val):
                # Green = covered (1), Red = gap (0)
                try:
                    v = int(val)
                except Exception:
                    v = 0
                return "background-color: #d4f8d4; font-weight: 600;" if v == 1 else "background-color: #ffd6d6; font-weight: 600;"

            styled = g.style.applymap(color_cell, subset=["Life", "STD", "LTD"])
            st.caption("Green = covered, Red = gap (missing product).")
            st.dataframe(styled, use_container_width=True, height=420)

# ---------------------------
# Tab: AI Insights (pilot mode, no external AI keys needed yet)
# ---------------------------
with tabs[4]:
    st.subheader("AI Insights (Pilot Mode)")
    st.caption(
        "Ask questions like: "
        "“Which carrier has the biggest LTD gap?” "
        "“Top brokers by commissions in Life” "
        "“Employers missing STD”"
    )

    q = st.text_input("Ask a question", placeholder="e.g., Which broker has the highest Life commissions?").strip()
    go = st.button("Answer")

    if go and q:
        ql = q.lower()

        # Basic “AI-ish” answers using current filtered data (works now; becomes powerful once marts are real)
        if "top" in ql and "broker" in ql and ("commission" in ql or "commissions" in ql):
            ans = (
                filtered.groupby("Broker", as_index=False)["Commissions Paid"]
                .sum()
                .sort_values("Commissions Paid", ascending=False)
                .head(10)
            )
            st.write(f"Top brokers by commissions for **{product}** (with your current filters):")
            st.dataframe(ans, use_container_width=True)

        elif "top" in ql and "carrier" in ql and ("commission" in ql or "commissions" in ql):
            ans = (
                filtered.groupby("Carrier", as_index=False)["Commissions Paid"]
                .sum()
                .sort_values("Commissions Paid", ascending=False)
                .head(10)
            )
            st.write(f"Top carriers by commissions for **{product}** (with your current filters):")
            st.dataframe(ans, use_container_width=True)

        elif "missing" in ql or "gap" in ql:
            # Simple gap queries: "employers missing std", etc.
            target = None
            if "std" in ql:
                target = "STD"
            elif "ltd" in ql:
                target = "LTD"
            elif "life" in ql:
                target = "Life"

            if target and target in gaps_df.columns:
                gap_list = gaps_df[gaps_df[target] == 0][["Employer", "Life", "STD", "LTD"]].head(50)
                st.write(f"Sample employers with a **{target} gap** (showing up to 50):")
                st.dataframe(gap_list, use_container_width=True)
                st.info("Once real marts are loaded, we’ll add broker/carrier attribution to these gaps.")
            else:
                st.write("Tell me which product gap you mean (Life, STD, or LTD). Example: “Employers missing LTD”.")

        else:
            st.write(
                "Pilot answer: I can answer ranking and gap questions now. "
                "Try: “Top brokers by commissions”, “Top carriers by commissions”, or “Employers missing LTD”."
            )

    st.divider()
    st.caption("Next upgrade: connect this panel to real marts + optional LLM (Claude/OpenAI) for natural language strategy answers.")