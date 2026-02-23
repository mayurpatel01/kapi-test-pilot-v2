from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px


# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Kapi Test Pilot", layout="wide", page_icon="📊")


# ---------------------------
# Password gate (Streamlit Secrets)
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
# Load marts (preferred) or fail loudly
# ---------------------------
DATA_DIR = Path("data") / "marts"

@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Missing mart: {p}. Run etl/build_marts.py and commit data/marts.")
    return pd.read_parquet(p)

epc = load_parquet("employer_product_carrier.parquet")  # Employer x Product x Carrier (+ Covered_Lives)
ebc = load_parquet("employer_broker_commissions.parquet")  # Employer x Broker (+ total_commissions)
gaps = load_parquet("employer_product_matrix.parquet")  # Employer x Life/STD/LTD flags

st.title("📊 Kapi Test Pilot — Form 5500 Market Intelligence")
st.success("Real Form 5500 marts detected ✅")
st.caption("Carrier views use covered lives & employer counts. Broker views use commissions (not duplicated).")


# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Filters")

product = st.sidebar.selectbox("Product", ["Life", "STD", "LTD"], index=0)

# Carrier filter applies to carrier/product mart
carrier_choices = ["All"] + sorted(epc["Carrier"].dropna().unique().tolist())
carrier = st.sidebar.selectbox("Carrier", carrier_choices, index=0)

# Broker filter applies to broker/comm mart
broker_choices = ["All"] + sorted(ebc["Broker"].dropna().unique().tolist())
broker = st.sidebar.selectbox("Broker", broker_choices, index=0)

employer_search = st.sidebar.text_input("Employer contains (optional)", value="").strip()

top_n = st.sidebar.slider("Top N for charts", min_value=10, max_value=200, value=25, step=5)

# Filter EPC
epc_f = epc[epc["Product"] == product].copy()
if carrier != "All":
    epc_f = epc_f[epc_f["Carrier"] == carrier]
if employer_search:
    epc_f = epc_f[epc_f["Employer"].astype(str).str.contains(employer_search, case=False, na=False)]

# Filter EBC
ebc_f = ebc.copy()
if broker != "All":
    ebc_f = ebc_f[ebc_f["Broker"] == broker]
if employer_search:
    ebc_f = ebc_f[ebc_f["Employer"].astype(str).str.contains(employer_search, case=False, na=False)]


# ---------------------------
# KPI row (sane)
# ---------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Unique Employers (product)", f"{epc_f['Employer'].nunique():,}")
k2.metric("Unique Carriers (product)", f"{epc_f['Carrier'].nunique():,}")
k3.metric("Covered Lives (product)", f"{int(epc_f['Covered_Lives'].sum()):,}")
k4.metric("Total Broker Commissions (filtered)", f"${ebc_f['total_commissions'].sum():,.0f}")


tabs = st.tabs(["Overview", "Brokers", "Carriers", "Gaps", "AI Insights"])

# ---------------------------
# Overview
# ---------------------------
with tabs[0]:
    st.subheader("Overview (Top N)")

    left, right = st.columns(2)

    with left:
        st.markdown("**Top Carriers by Covered Lives**")
        csum = (
            epc_f.groupby("Carrier", as_index=False)
            .agg(covered_lives=("Covered_Lives", "sum"), employers=("Employer", "nunique"))
            .sort_values("covered_lives", ascending=False)
            .head(top_n)
        )
        fig = px.bar(csum, x="Carrier", y="covered_lives", title=f"Top {top_n} Carriers by Covered Lives ({product})")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(csum, use_container_width=True)

    with right:
        st.markdown("**Top Brokers by Commissions**")
        bsum = (
            ebc_f.groupby("Broker", as_index=False)
            .agg(commissions=("total_commissions", "sum"), employers=("Employer", "nunique"))
            .sort_values("commissions", ascending=False)
            .head(top_n)
        )
        fig2 = px.bar(bsum, x="Broker", y="commissions", title=f"Top {top_n} Brokers by Commissions (All Products)")
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(bsum, use_container_width=True)

# ---------------------------
# Brokers tab (commissions)
# ---------------------------
with tabs[1]:
    st.subheader("Brokers (Commissions)")

    bsum = (
        ebc_f.groupby("Broker", as_index=False)
        .agg(commissions=("total_commissions", "sum"), employers=("Employer", "nunique"))
        .sort_values("commissions", ascending=False)
        .head(top_n)
    )
    fig = px.bar(bsum, x="Broker", y="commissions", title=f"Top {top_n} Brokers by Commissions")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(bsum, use_container_width=True)

# ---------------------------
# Carriers tab (presence + lives)
# ---------------------------
with tabs[2]:
    st.subheader("Carriers (Covered Lives + Employers)")

    csum = (
        epc_f.groupby("Carrier", as_index=False)
        .agg(covered_lives=("Covered_Lives", "sum"), employers=("Employer", "nunique"))
        .sort_values("covered_lives", ascending=False)
        .head(top_n)
    )

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(csum, x="Carrier", y="covered_lives", title=f"Top {top_n} Carriers by Covered Lives ({product})")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.bar(csum, x="Carrier", y="employers", title=f"Top {top_n} Carriers by Employers ({product})")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(csum, use_container_width=True)

# ---------------------------
# Gaps tab (red/green matrix)
# ---------------------------
with tabs[3]:
    st.subheader("Gaps (Employer Product Matrix)")

    g = gaps.copy()
    if employer_search:
        g = g[g["Employer"].astype(str).str.contains(employer_search, case=False, na=False)]

    max_rows = st.slider("Max rows to display", min_value=25, max_value=500, value=150, step=25)
    g = g.head(max_rows)

    def color_cell(val):
        try:
            v = int(val)
        except Exception:
            v = 0
        return "background-color: #d4f8d4; font-weight: 600;" if v == 1 else "background-color: #ffd6d6; font-weight: 600;"

    styled = g.style.applymap(color_cell, subset=["Life", "STD", "LTD"])
    st.caption("Green = covered, Red = gap (missing product).")
    st.dataframe(styled, use_container_width=True, height=420)

# ---------------------------
# AI Insights tab (real answers, no external AI key needed)
# ---------------------------
with tabs[4]:
    st.subheader("AI Insights (Pilot Analytics Q&A)")
    st.caption(
        "Examples: "
        "“Top carriers by covered lives in Life”, "
        "“Top brokers by commissions”, "
        "“Employers missing STD”, "
        "“How many employers have Life but no LTD?”"
    )

    q = st.text_input("Ask a question", placeholder="e.g., Employers missing STD").strip()
    go = st.button("Answer")

    if go and q:
        ql = q.lower()

        if "top" in ql and "carrier" in ql:
            csum = (
                epc_f.groupby("Carrier", as_index=False)
                .agg(covered_lives=("Covered_Lives", "sum"), employers=("Employer", "nunique"))
                .sort_values("covered_lives", ascending=False)
                .head(25)
            )
            st.write(f"Top carriers for **{product}** (using your current filters):")
            st.dataframe(csum, use_container_width=True)

        elif "top" in ql and "broker" in ql:
            bsum = (
                ebc_f.groupby("Broker", as_index=False)
                .agg(commissions=("total_commissions", "sum"), employers=("Employer", "nunique"))
                .sort_values("commissions", ascending=False)
                .head(25)
            )
            st.write("Top brokers by commissions (with your current filters):")
            st.dataframe(bsum, use_container_width=True)

        elif "missing" in ql or "gap" in ql:
            target = None
            if "std" in ql:
                target = "STD"
            elif "ltd" in ql:
                target = "LTD"
            elif "life" in ql:
                target = "Life"

            if target:
                missing_df = gaps[gaps[target] == 0][["Employer", "Life", "STD", "LTD"]]
                st.write(f"Employers missing **{target}** (showing top 50):")
                st.dataframe(missing_df.head(50), use_container_width=True)

                count_missing = int((gaps[target] == 0).sum())
                st.success(f"Total employers missing {target}: {count_missing:,}")
            else:
                st.write("Ask like: “Employers missing STD” or “Employers missing LTD”.")

        elif "life but" in ql and "no" in ql:
            # Example: "Life but no LTD" / "Life but no STD"
            if "ltd" in ql:
                subset = gaps[(gaps["Life"] == 1) & (gaps["LTD"] == 0)]
                st.success(f"Employers with Life but no LTD: {len(subset):,}")
                st.dataframe(subset.head(50), use_container_width=True)
            elif "std" in ql:
                subset = gaps[(gaps["Life"] == 1) & (gaps["STD"] == 0)]
                st.success(f"Employers with Life but no STD: {len(subset):,}")
                st.dataframe(subset.head(50), use_container_width=True)
            else:
                st.write("Try: “How many employers have Life but no LTD?”")

        else:
            st.write(
                "Try one of these:\n"
                "- Top carriers by covered lives\n"
                "- Top brokers by commissions\n"
                "- Employers missing STD\n"
                "- Employers missing LTD\n"
                "- How many employers have Life but no LTD?"
            )