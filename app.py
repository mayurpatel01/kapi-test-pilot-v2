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
# Load marts
# ---------------------------
DATA_DIR = Path("data") / "marts"


@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"Missing mart: {p}. Make sure data/marts is committed to GitHub and deployed."
        )
    return pd.read_parquet(p)


epc = load_parquet("employer_product_carrier.parquet")          # Employer x Product x Carrier (+ Covered_Lives)
ebc = load_parquet("employer_broker_commissions.parquet")      # Employer x Broker (+ total_commissions)
gaps = load_parquet("employer_product_matrix.parquet")         # Employer x Life/STD/LTD flags

# Basic cleaning
for df in (epc, ebc, gaps):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

# Ensure numeric
if "Covered_Lives" in epc.columns:
    epc["Covered_Lives"] = pd.to_numeric(epc["Covered_Lives"], errors="coerce").fillna(0.0)
if "total_commissions" in ebc.columns:
    ebc["total_commissions"] = pd.to_numeric(ebc["total_commissions"], errors="coerce").fillna(0.0)
for col in ["Life", "STD", "LTD"]:
    if col in gaps.columns:
        gaps[col] = pd.to_numeric(gaps[col], errors="coerce").fillna(0).astype("int8")

# ---------------------------
# Header
# ---------------------------
st.title("📊 Kapi Test Pilot — Form 5500 Market Intelligence")
st.success("Real Form 5500 marts detected ✅")
st.caption("This pilot normalizes commission outliers and focuses on opportunity & market gaps.")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Controls")

lens = st.sidebar.radio("AI Lens", ["Broker Lens", "Carrier Lens"], index=0)

product = st.sidebar.selectbox("Product", ["Life", "STD", "LTD"], index=0)
employer_search = st.sidebar.text_input("Employer contains (optional)", value="").strip()

top_n = st.sidebar.slider("Top N for charts/tables", min_value=10, max_value=200, value=25, step=5)

st.sidebar.divider()
st.sidebar.subheader("Commission normalization")

cap_mode = st.sidebar.selectbox(
    "Cap commissions at",
    ["None (raw)", "95th percentile", "99th percentile"],
    index=2,
)

# Precompute caps (global across ebc)
q95 = float(ebc["total_commissions"].quantile(0.95)) if len(ebc) else 0.0
q99 = float(ebc["total_commissions"].quantile(0.99)) if len(ebc) else 0.0

if cap_mode == "95th percentile":
    cap_value = q95
elif cap_mode == "99th percentile":
    cap_value = q99
else:
    cap_value = None

show_cap_value = f"${cap_value:,.0f}" if cap_value else "None"
st.sidebar.caption(f"95th: ${q95:,.0f} • 99th: ${q99:,.0f} • Cap: {show_cap_value}")

# Apply employer search filter (shared)
if employer_search:
    epc_f = epc[epc["Employer"].str.contains(employer_search, case=False, na=False)].copy()
    ebc_f = ebc[ebc["Employer"].str.contains(employer_search, case=False, na=False)].copy()
    gaps_f = gaps[gaps["Employer"].str.contains(employer_search, case=False, na=False)].copy()
else:
    epc_f, ebc_f, gaps_f = epc.copy(), ebc.copy(), gaps.copy()

# Filter EPC by product for carrier/product views
epc_p = epc_f[epc_f["Product"] == product].copy()

# Commission normalization (ONLY affects analytics, does not overwrite raw)
ebc_f["comm_norm"] = ebc_f["total_commissions"]
if cap_value is not None:
    ebc_f["comm_norm"] = ebc_f["comm_norm"].clip(upper=cap_value)

# ---------------------------
# Global KPIs (sane)
# ---------------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("Unique Employers", f"{gaps_f['Employer'].nunique():,}")
k2.metric("Unique Brokers", f"{ebc_f['Broker'].nunique():,}")
k3.metric("Unique Carriers", f"{epc_p['Carrier'].nunique():,}")
k4.metric("Median Comm (Employer↔Broker)", f"${ebc_f['total_commissions'].median():,.0f}")

k5, k6, k7, k8 = st.columns(4)
k5.metric(f"Covered Lives ({product})", f"{int(epc_p['Covered_Lives'].sum()):,}")
k6.metric("95th Comm", f"${q95:,.0f}")
k7.metric("99th Comm", f"${q99:,.0f}")
k8.metric("Normalized Comm Sum", f"${ebc_f['comm_norm'].sum():,.0f}")

tabs = st.tabs(["Overview", "Lens Dashboard", "Gaps", "AI Insights", "Data QA"])

# ---------------------------
# Overview
# ---------------------------
with tabs[0]:
    st.subheader("Overview (Top N, readable)")

    left, right = st.columns(2)

    with left:
        st.markdown(f"**Top Carriers by Covered Lives — {product}**")
        csum = (
            epc_p.groupby("Carrier", as_index=False)
            .agg(covered_lives=("Covered_Lives", "sum"), employers=("Employer", "nunique"))
            .sort_values("covered_lives", ascending=False)
            .head(top_n)
        )
        fig = px.bar(csum, x="Carrier", y="covered_lives", title=f"Top {top_n} Carriers by Covered Lives ({product})")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(csum, use_container_width=True, height=280)

    with right:
        st.markdown("**Top Brokers by Normalized Commissions (All Products)**")
        bsum = (
            ebc_f.groupby("Broker", as_index=False)
            .agg(commissions=("comm_norm", "sum"), employers=("Employer", "nunique"))
            .sort_values("commissions", ascending=False)
            .head(top_n)
        )
        fig2 = px.bar(bsum, x="Broker", y="commissions", title=f"Top {top_n} Brokers by Normalized Commissions")
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(bsum, use_container_width=True, height=280)

    st.caption("Note: Commission totals are normalized to reduce distortion from extreme outliers.")

# ---------------------------
# Lens Dashboard
# ---------------------------
with tabs[1]:
    if lens == "Broker Lens":
        st.subheader("Broker Lens — Cross-sell Opportunity Intelligence")

        # Build employer-level gap flags for joins
        # We'll compute opportunity based on missing products for employers in the broker's book.
        g = gaps_f[["Employer", "Life", "STD", "LTD"]].copy()

        # Broker book = employers per broker + normalized commissions
        broker_book = ebc_f.groupby(["Broker", "Employer"], as_index=False).agg(
            comm_norm=("comm_norm", "sum"),
            comm_raw=("total_commissions", "sum"),
        )

        # Attach gaps to each broker-employer relation
        broker_book = broker_book.merge(g, on="Employer", how="left").fillna(0)

        # Define opportunity: life-only missing STD/LTD, or STD missing LTD, etc.
        # Simple scoring (pilot):
        # - If employer has Life but missing STD: +1
        # - If employer has Life but missing LTD: +1
        # - If employer has STD but missing LTD: +0.5
        broker_book["opp_life_no_std"] = ((broker_book["Life"] == 1) & (broker_book["STD"] == 0)).astype(int)
        broker_book["opp_life_no_ltd"] = ((broker_book["Life"] == 1) & (broker_book["LTD"] == 0)).astype(int)
        broker_book["opp_std_no_ltd"] = ((broker_book["STD"] == 1) & (broker_book["LTD"] == 0)).astype(int)

       # CEO-friendly weighting (higher weight on LTD white space = higher LTV / stickiness)
        broker_book["opportunity_score"] = (
            broker_book["opp_life_no_ltd"] * 1.5
            + broker_book["opp_life_no_std"] * 1.0
            + broker_book["opp_std_no_ltd"] * 1.0
)

        # Aggregate broker-level intelligence
        broker_ai = (
            broker_book.groupby("Broker", as_index=False)
            .agg(
                employers=("Employer", "nunique"),
                comm_norm_sum=("comm_norm", "sum"),
                comm_norm_median=("comm_norm", "median"),
                opp_score=("opportunity_score", "sum"),
                life_no_std=("opp_life_no_std", "sum"),
                life_no_ltd=("opp_life_no_ltd", "sum"),
                std_no_ltd=("opp_std_no_ltd", "sum"),
            )
        )

        # Normalize opportunity by employer count for a "rate"
        broker_ai["opp_per_100_employers"] = (broker_ai["opp_score"] / broker_ai["employers"].clip(lower=1)) * 100.0

        # Rankers (two useful perspectives)
        left, right = st.columns(2)

        with left:
            st.markdown("**Top Brokers by Cross-sell Opportunity Score**")
            top_opp = broker_ai.sort_values("opp_score", ascending=False).head(top_n)
            fig = px.bar(top_opp, x="Broker", y="opp_score", title=f"Top {top_n} Brokers — Opportunity Score")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top_opp, use_container_width=True)

        with right:
            st.markdown("**Top Brokers by Opportunity Rate (per 100 employers)**")
            top_rate = broker_ai.sort_values("opp_per_100_employers", ascending=False).head(top_n)
            fig2 = px.bar(top_rate, x="Broker", y="opp_per_100_employers", title=f"Top {top_n} Brokers — Opportunity Rate")
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(top_rate, use_container_width=True)

        st.divider()
        st.markdown("### Broker Drilldown")
        selected = st.selectbox("Select a broker to drill down", options=["(Select)"] + sorted(broker_ai["Broker"].tolist()))
        if selected != "(Select)":
            bb = broker_book[broker_book["Broker"] == selected].copy()
            st.write(f"**{selected}** — Employers in book: {bb['Employer'].nunique():,}")
            st.write("Top opportunities (sample employers):")
            show = bb.sort_values(["opportunity_score", "comm_norm"], ascending=[False, False]).head(50)[
                ["Employer", "Life", "STD", "LTD", "opportunity_score", "comm_norm"]
            ]
            st.dataframe(show, use_container_width=True, height=350)

            st.caption("Pilot scoring is intentionally simple. Next version can weight by lives, carrier concentration, and premium proxies.")

    else:
        st.subheader("Carrier Lens — Market Presence + White-space (Imbalance)")

        # Build carrier/product aggregates across all products (not just selected)
        epc_all = epc_f.copy()

        carrier_prod = (
            epc_all.groupby(["Carrier", "Product"], as_index=False)
            .agg(
                employers=("Employer", "nunique"),
                covered_lives=("Covered_Lives", "sum"),
            )
        )

        # Pivot to get per-carrier per-product employers/lives
        pivot_emp = carrier_prod.pivot_table(index="Carrier", columns="Product", values="employers", aggfunc="sum", fill_value=0).reset_index()
        pivot_lives = carrier_prod.pivot_table(index="Carrier", columns="Product", values="covered_lives", aggfunc="sum", fill_value=0).reset_index()

        # Ensure columns exist
        for col in ["Life", "STD", "LTD"]:
            if col not in pivot_emp.columns:
                pivot_emp[col] = 0
            if col not in pivot_lives.columns:
                pivot_lives[col] = 0.0

        carrier_ai = pivot_emp.merge(pivot_lives, on="Carrier", suffixes=("_emp", "_lives"))

        # Simple “white-space” / imbalance metrics:
        # - If strong Life but weak STD/LTD => high imbalance
        # Use employer counts to avoid lives reporting noise.
        carrier_ai["total_emp"] = carrier_ai["Life_emp"] + carrier_ai["STD_emp"] + carrier_ai["LTD_emp"]
        carrier_ai["life_share"] = carrier_ai["Life_emp"] / carrier_ai["total_emp"].clip(lower=1)
        carrier_ai["std_share"] = carrier_ai["STD_emp"] / carrier_ai["total_emp"].clip(lower=1)
        carrier_ai["ltd_share"] = carrier_ai["LTD_emp"] / carrier_ai["total_emp"].clip(lower=1)

        # Imbalance Score (pilot):
        # If Life share is high and STD/LTD shares are low => larger score.
        carrier_ai["imbalance_score"] = (
            carrier_ai["life_share"] * (1.0 - carrier_ai["std_share"]) * 50
            + carrier_ai["life_share"] * (1.0 - carrier_ai["ltd_share"]) * 50
        )

        left, right = st.columns(2)

        with left:
            st.markdown("**Top Carriers by Imbalance Score (Life-heavy white space)**")
            top_imb = carrier_ai.sort_values("imbalance_score", ascending=False).head(top_n)
            fig = px.bar(top_imb, x="Carrier", y="imbalance_score", title=f"Top {top_n} Carriers — Imbalance Score")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top_imb[[
                "Carrier", "Life_emp", "STD_emp", "LTD_emp", "total_emp",
                "life_share", "std_share", "ltd_share", "imbalance_score"
            ]], use_container_width=True)

        with right:
            st.markdown(f"**Top Carriers by Covered Lives — {product}**")
            csum = (
                epc_p.groupby("Carrier", as_index=False)
                .agg(covered_lives=("Covered_Lives", "sum"), employers=("Employer", "nunique"))
                .sort_values("covered_lives", ascending=False)
                .head(top_n)
            )
            fig2 = px.bar(csum, x="Carrier", y="covered_lives", title=f"Top {top_n} Carriers by Covered Lives ({product})")
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(csum, use_container_width=True)

        st.divider()
        st.markdown("### Carrier Drilldown")
        selected = st.selectbox("Select a carrier to drill down", options=["(Select)"] + sorted(carrier_ai["Carrier"].tolist()))
        if selected != "(Select)":
            row = carrier_ai[carrier_ai["Carrier"] == selected].iloc[0]
            st.write(f"**{selected}** — Employers: {int(row['total_emp']):,}")
            st.write(
                f"Life_emp: {int(row['Life_emp']):,} • STD_emp: {int(row['STD_emp']):,} • LTD_emp: {int(row['LTD_emp']):,}"
            )
            st.write(
                f"Life share: {row['life_share']:.2%} • STD share: {row['std_share']:.2%} • LTD share: {row['ltd_share']:.2%}"
            )
            st.info("Interpretation: Higher imbalance score suggests a carrier is Life-heavy and may have disability product white space.")

# ---------------------------
# Gaps tab (matrix)
# ---------------------------
with tabs[2]:
    st.subheader("Gaps (Employer Product Matrix)")

    g = gaps_f.copy()

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
# AI Insights tab (real Q&A over marts)
# ---------------------------
with tabs[3]:
    st.subheader("AI Insights (Pilot) — Ask Strategy Questions")
    st.caption(
        "Examples:\n"
        "• Which brokers have the biggest cross-sell opportunity?\n"
        "• Which carriers are Life-heavy with LTD white space?\n"
        "• How many employers have Life but no STD?\n"
        "• Top carriers by covered lives in LTD"
    )

    q = st.text_input("Ask a question", placeholder="e.g., Which brokers have the biggest cross-sell opportunity?").strip()
    go = st.button("Answer")

    if go and q:
        ql = q.lower()

        # Helpful precomputed views
        # Broker opp view (lightweight recompute)
        g = gaps_f[["Employer", "Life", "STD", "LTD"]].copy()
        broker_book = ebc_f.groupby(["Broker", "Employer"], as_index=False).agg(comm_norm=("comm_norm", "sum"))
        broker_book = broker_book.merge(g, on="Employer", how="left").fillna(0)
        # CEO-friendly weighting (LTD white space weighted highest)
        broker_book["opp_score"] = (
            ((broker_book["Life"] == 1) & (broker_book["LTD"] == 0)).astype(int) * 1.5
            + ((broker_book["Life"] == 1) & (broker_book["STD"] == 0)).astype(int) * 1.0
            + ((broker_book["STD"] == 1) & (broker_book["LTD"] == 0)).astype(int) * 1.0
)
        broker_ai = (
            broker_book.groupby("Broker", as_index=False)
            .agg(employers=("Employer", "nunique"), opp_score=("opp_score", "sum"), comm_norm=("comm_norm", "sum"))
        )
        broker_ai["opp_per_100"] = (broker_ai["opp_score"] / broker_ai["employers"].clip(lower=1)) * 100.0

        # Carrier imbalance view (lightweight recompute)
        epc_all = epc_f.copy()
        carrier_prod = (
            epc_all.groupby(["Carrier", "Product"], as_index=False)
            .agg(employers=("Employer", "nunique"), covered_lives=("Covered_Lives", "sum"))
        )
        piv = carrier_prod.pivot_table(index="Carrier", columns="Product", values="employers", aggfunc="sum", fill_value=0).reset_index()
        for col in ["Life", "STD", "LTD"]:
            if col not in piv.columns:
                piv[col] = 0
        piv["total_emp"] = piv["Life"] + piv["STD"] + piv["LTD"]
        piv["life_share"] = piv["Life"] / piv["total_emp"].clip(lower=1)
        piv["std_share"] = piv["STD"] / piv["total_emp"].clip(lower=1)
        piv["ltd_share"] = piv["LTD"] / piv["total_emp"].clip(lower=1)
        piv["imbalance_score"] = (piv["life_share"] * (1.0 - piv["std_share"]) * 50 + piv["life_share"] * (1.0 - piv["ltd_share"]) * 50)

        # Question routing
        if "broker" in ql and ("opportunity" in ql or "cross" in ql or "gap" in ql):
            st.write("Top brokers by **cross-sell opportunity** (pilot scoring):")
            out = broker_ai.sort_values("opp_score", ascending=False).head(25)
            st.dataframe(out, use_container_width=True)
            st.info("Interpretation: higher opp_score means more employers in that broker’s book are missing STD/LTD coverage.")

        elif "broker" in ql and ("rate" in ql or "per 100" in ql):
            st.write("Top brokers by **opportunity rate (per 100 employers)**:")
            out = broker_ai.sort_values("opp_per_100", ascending=False).head(25)
            st.dataframe(out, use_container_width=True)

        elif "carrier" in ql and ("white space" in ql or "imbalance" in ql or "gap" in ql):
            st.write("Top carriers by **Life-heavy imbalance score** (white space indicator):")
            out = piv.sort_values("imbalance_score", ascending=False).head(25)
            st.dataframe(out, use_container_width=True)
            st.info("Interpretation: carriers with high Life share but low STD/LTD presence may have disability white space.")

        elif "top carriers" in ql and ("ltd" in ql or "std" in ql or "life" in ql or "covered" in ql):
            # Choose product from question if present
            p = "LTD" if "ltd" in ql else ("STD" if "std" in ql else ("Life" if "life" in ql else product))
            tmp = epc_f[epc_f["Product"] == p].copy()
            out = (
                tmp.groupby("Carrier", as_index=False)
                .agg(covered_lives=("Covered_Lives", "sum"), employers=("Employer", "nunique"))
                .sort_values("covered_lives", ascending=False)
                .head(25)
            )
            st.write(f"Top carriers by covered lives for **{p}**:")
            st.dataframe(out, use_container_width=True)

        elif "how many" in ql and ("missing" in ql or "gap" in ql):
            target = "STD" if "std" in ql else ("LTD" if "ltd" in ql else ("Life" if "life" in ql else None))
            if target:
                cnt = int((gaps_f[target] == 0).sum())
                st.success(f"Employers missing {target}: {cnt:,}")
            else:
                st.write("Try: “How many employers are missing STD?” or “…missing LTD?”")

        elif "life but no" in ql:
            if "std" in ql:
                subset = gaps_f[(gaps_f["Life"] == 1) & (gaps_f["STD"] == 0)]
                st.success(f"Employers with Life but no STD: {len(subset):,}")
                st.dataframe(subset.head(50), use_container_width=True)
            elif "ltd" in ql:
                subset = gaps_f[(gaps_f["Life"] == 1) & (gaps_f["LTD"] == 0)]
                st.success(f"Employers with Life but no LTD: {len(subset):,}")
                st.dataframe(subset.head(50), use_container_width=True)
            else:
                st.write("Try: “How many employers have Life but no LTD?”")

        else:
            st.write(
                "Try one of these:\n"
                "- Which brokers have the biggest cross-sell opportunity?\n"
                "- Which carriers have the most LTD white space?\n"
                "- Top carriers by covered lives in LTD\n"
                "- How many employers are missing STD?\n"
                "- Employers with Life but no LTD"
            )

# ---------------------------
# Data QA tab (transparency + trust)
# ---------------------------
with tabs[4]:
    st.subheader("Data QA (Pilot Trust Panel)")

    st.markdown("### Commission distribution sanity")
    st.write(f"Median: **${ebc_f['total_commissions'].median():,.0f}**")
    st.write(f"95th percentile: **${q95:,.0f}**")
    st.write(f"99th percentile: **${q99:,.0f}**")
    st.write(f"Max (raw): **${ebc_f['total_commissions'].max():,.0f}**")
    st.caption("We normalize/cap commissions for rankings because extreme outliers distort sums and means.")

    st.markdown("### Quick counts")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows: EPC", f"{len(epc_f):,}")
    c2.metric("Rows: EBC", f"{len(ebc_f):,}")
    c3.metric("Rows: GAPS", f"{len(gaps_f):,}")

    st.markdown("### Sample rows (for debugging)")
    st.write("Employer↔Broker commissions (sample):")
    st.dataframe(ebc_f.head(20), use_container_width=True)

    st.write("Employer↔Product↔Carrier (sample):")
    st.dataframe(epc_p.head(20), use_container_width=True)