from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 0) App setup
# ============================================================
st.set_page_config(page_title="Test Pilot", layout="wide", page_icon="📊")


# ============================================================
# 1) Password gate (Streamlit Secrets)
# ============================================================
def require_password():
    pw = st.secrets.get("APP_PASSWORD", "")
    if not pw:
        st.warning("APP_PASSWORD not set in Streamlit Secrets.")
        st.stop()

    if "authed" not in st.session_state:
        st.session_state.authed = False

    if not st.session_state.authed:
        st.title("Test Pilot")
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


# ============================================================
# 2) Load marts
# ============================================================
DATA_DIR = Path("data") / "marts"


@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"Missing mart: {p}. Make sure data/marts is committed to GitHub and deployed."
        )
    return pd.read_parquet(p)


# Required marts from your revised ETL
epc = load_parquet("employer_product_carrier.parquet")          # Employer x Product x Carrier (+ Covered_Lives)
ebc = load_parquet("employer_broker_commissions.parquet")      # Employer x Broker (+ total_commissions)
gaps = load_parquet("employer_product_matrix.parquet")         # Employer x Life/STD/LTD flags


# Basic cleaning
def _to_str(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str)
    return df


epc = _to_str(epc)
ebc = _to_str(ebc)
gaps = _to_str(gaps)

# Ensure numeric fields
if "Covered_Lives" in epc.columns:
    epc["Covered_Lives"] = pd.to_numeric(epc["Covered_Lives"], errors="coerce").fillna(0.0)

if "total_commissions" in ebc.columns:
    ebc["total_commissions"] = pd.to_numeric(ebc["total_commissions"], errors="coerce").fillna(0.0)

for col in ["Life", "STD", "LTD"]:
    if col in gaps.columns:
        gaps[col] = pd.to_numeric(gaps[col], errors="coerce").fillna(0).astype("int8")


# ============================================================
# 3) Sidebar controls (CEO-friendly)
# ============================================================
st.title("📊 Test Pilot — CEO Opportunity Intelligence (Form 5500)")

st.success("Real Form 5500 marts detected ✅")
st.caption(
    "This pilot normalizes outliers and surfaces cross-sell, white space, risk, and M&A signals. "
    "All scores are transparent and adjustable in the sidebar."
)

st.sidebar.header("CEO Controls")

DEFAULT_LENS = "Broker Lens"  # CEO-first default (tactical revenue / distribution)
lens = st.sidebar.radio("Lens", ["Broker Lens", "Carrier Lens", "M&A Radar"], index=0)

product = st.sidebar.selectbox("Product view", ["Life", "STD", "LTD"], index=0)
top_n = st.sidebar.slider("Top N", min_value=10, max_value=200, value=25, step=5)

employer_search = st.sidebar.text_input("Employer contains (optional)", value="").strip()

st.sidebar.divider()
st.sidebar.subheader("Normalization & Scenarios")

cap_mode = st.sidebar.selectbox(
    "Commission cap (reduces outlier distortion)",
    ["99th percentile (recommended)", "95th percentile", "None (raw)"],
    index=0,
)

# CEO scenario knobs
conversion_rate = st.sidebar.slider("Conversion rate (wins from opportunities)", 0.0, 0.5, 0.10, 0.01)
uplift_multiplier = st.sidebar.slider(
    "Commission uplift per win (vs median)",
    0.5, 3.0, 1.0, 0.1,
    help="Pilot assumption: each converted gap produces ~median commission * uplift."
)

# Opportunity weighting (CEO LTV logic)
w_life_no_ltd = st.sidebar.slider("Weight: Life but missing LTD (highest LTV)", 0.5, 3.0, 1.5, 0.1)
w_life_no_std = st.sidebar.slider("Weight: Life but missing STD", 0.5, 3.0, 1.0, 0.1)
w_std_no_ltd = st.sidebar.slider("Weight: STD but missing LTD", 0.0, 3.0, 1.0, 0.1)

# Risk / concentration sensitivity
concentration_penalty = st.sidebar.slider(
    "Concentration penalty strength",
    0.0, 2.0, 0.7, 0.1,
    help="Higher = penalize brokers/carriers that rely on few accounts."
)




# ============================================================
# 4) Filters (lightweight)
# ============================================================
def filter_by_employer(df: pd.DataFrame) -> pd.DataFrame:
    if employer_search:
        if "Employer" in df.columns:
            return df[df["Employer"].str.contains(employer_search, case=False, na=False)].copy()
    return df.copy()


epc_f = filter_by_employer(epc)
ebc_f = filter_by_employer(ebc)
gaps_f = filter_by_employer(gaps)

# Product-specific view for EPC
epc_p = epc_f[epc_f["Product"] == product].copy()


# ============================================================
# 5) Normalization helpers
# ============================================================
@st.cache_data(show_spinner=False)
def compute_caps(series: pd.Series) -> Tuple[float, float]:
    if len(series) == 0:
        return 0.0, 0.0
    q95 = float(series.quantile(0.95))
    q99 = float(series.quantile(0.99))
    return q95, q99


q95, q99 = compute_caps(ebc_f["total_commissions"])

if cap_mode.startswith("99"):
    cap_value = q99
elif cap_mode.startswith("95"):
    cap_value = q95
else:
    cap_value = None

ebc_f["comm_norm"] = ebc_f["total_commissions"]
if cap_value is not None and cap_value > 0:
    ebc_f["comm_norm"] = ebc_f["comm_norm"].clip(upper=cap_value)

median_comm_norm = float(ebc_f["comm_norm"].median()) if len(ebc_f) else 0.0
median_comm_raw = float(ebc_f["total_commissions"].median()) if len(ebc_f) else 0.0

def money(x: float) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "$0"
    
def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = float(s.std(ddof=0)) if len(s) else 0.0
    if sd == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - float(s.mean())) / sd


def hhi_from_shares(shares: np.ndarray) -> float:
    # Herfindahl-Hirschman Index (0..1). Higher = more concentrated.
    if shares.size == 0:
        return 0.0
    return float(np.sum(np.square(shares)))


# ============================================================
# 6) CEO KPIs (avoid misleading totals)
# ============================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Unique Employers", f"{gaps_f['Employer'].nunique():,}")
k2.metric("Unique Brokers", f"{ebc_f['Broker'].nunique():,}")
k3.metric("Unique Carriers", f"{epc_p['Carrier'].nunique():,}")
k4.metric("Median Commission (Employer↔Broker, normalized)", f"${median_comm_norm:,.0f}")

k5, k6, k7, k8 = st.columns(4)
k5.metric(f"Covered Lives ({product})", f"{int(epc_p['Covered_Lives'].sum()):,}")
k6.metric("Commission cap", f"${cap_value:,.0f}" if cap_value else "None")
k7.metric("95th / 99th", f"${q95:,.0f} / ${q99:,.0f}")
k8.metric("Max (raw)", f"${float(ebc_f['total_commissions'].max()):,.0f}" if len(ebc_f) else "$0")


tabs = st.tabs([
    "Executive Overview",
    "Opportunity Radar",
    "Lens Dashboard",
    "AI Analyst Chat",
    "Gaps Matrix",
    "Data QA / Methods",
])

# ============================================================
# MAIN PAGE AI ANALYST (always visible)
# ============================================================
st.markdown("## 🤖 AI Analyst (always available)")
st.caption("Ask: 'Top 10 recommended actions', 'Which brokers have biggest LTD white space?', 'Carrier LTD white space targets', 'M&A roll-up targets'.")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Show last messages (compact)
with st.expander("Open AI Analyst Chat", expanded=True):
    for msg in st.session_state.chat[-8:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask the AI analyst… (always available)")
    if user_q:
        st.session_state.chat.append({"role": "user", "content": user_q})

        ql = user_q.lower()
        response_lines = []

        # helper
        def _money(x: float) -> str:
            try:
                return f"${float(x):,.0f}"
            except Exception:
                return "$0"

        # ROUTES (same logic as your AI Analyst tab, condensed)
        if "recommended actions" in ql or ("top 10" in ql and "action" in ql):
            top10 = broker_ai.head(10)
            response_lines.append("### Top 10 Recommended Actions (CEO view)")
            for i, r in enumerate(top10.itertuples(index=False), start=1):
                response_lines.append(
                    f"**{i}. Target {r.Broker}** — {int(r.employers):,} employers • "
                    f"Opp: **{float(r.opp_score):,.1f}** • "
                    f"Impact: **{_money(float(r.rev_impact_low))} – {_money(float(r.rev_impact_high))}** "
                    f"(base {_money(float(r.rev_impact_base))})"
                )
            response_lines.append(
                f"\nModel: wins = opp_score × conversion ({conversion_rate:.0%}); "
                f"per-win ≈ median_comm_norm × uplift ({uplift_multiplier:.1f})."
            )

        elif "broker" in ql and ("opportunity" in ql or "white space" in ql or "gap" in ql):
            view = broker_ai.sort_values("opp_score", ascending=False).head(10)[
                ["Broker", "employers", "opp_score", "opp_per_100_employers", "rev_impact_base", "hhi_concentration"]
            ]
            response_lines.append("### Top brokers by cross-sell opportunity")
            response_lines.append(view.to_markdown(index=False))
            response_lines.append("\nInterpretation: target high opp + reasonable concentration (lower HHI).")

        elif "carrier" in ql and ("ltd" in ql or "white space" in ql or "imbalance" in ql):
            view = carrier_ai.sort_values("whitespace_ltd", ascending=False).head(10)[
                ["Carrier", "total_emp", "Life_emp", "LTD_emp", "whitespace_ltd", "imbalance_score"]
            ]
            response_lines.append("### Top carriers by LTD white space (Life-heavy, low LTD)")
            response_lines.append(view.to_markdown(index=False))

        elif "m&a" in ql or "acquisition" in ql or "roll-up" in ql:
            broker_ma = broker_ai.copy()
            broker_ma["ma_score"] = (
                zscore(broker_ma["comm_norm_sum"]) * 0.40
                + zscore(broker_ma["employers"]) * 0.25
                + zscore(broker_ma["opp_score"]) * 0.35
                - zscore(broker_ma["hhi_concentration"]) * 0.40
            )
            view = broker_ma.sort_values("ma_score", ascending=False).head(10)[
                ["Broker", "employers", "comm_norm_sum", "opp_score", "hhi_concentration", "ma_score"]
            ]
            response_lines.append("### Broker roll-up targets (pilot heuristic)")
            response_lines.append(view.to_markdown(index=False))

        else:
            response_lines.append(
                "Try:\n"
                "- **Top 10 recommended actions**\n"
                "- **Top brokers by LTD white space**\n"
                "- **Top carriers by LTD white space**\n"
                "- **M&A roll-up targets**"
            )

        assistant_content = "\n\n".join(response_lines)
        st.session_state.chat.append({"role": "assistant", "content": assistant_content})
        st.rerun()
# ============================================================
# 7) Core computed views (cached)
# ============================================================
@st.cache_data(show_spinner=False)
def build_broker_book(ebc_df: pd.DataFrame, gaps_df: pd.DataFrame) -> pd.DataFrame:
    g = gaps_df[["Employer", "Life", "STD", "LTD"]].copy()
    bb = ebc_df.groupby(["Broker", "Employer"], as_index=False).agg(
        comm_norm=("comm_norm", "sum"),
        comm_raw=("total_commissions", "sum"),
    )
    bb = bb.merge(g, on="Employer", how="left").fillna(0)
    return bb


@st.cache_data(show_spinner=False)
def build_broker_ai(
    broker_book: pd.DataFrame,
    w1: float, w2: float, w3: float,
    conc_penalty: float,
    median_comm: float,
    conversion: float,
    uplift: float,
) -> pd.DataFrame:
    bb = broker_book.copy()

    bb["opp_life_no_ltd"] = ((bb["Life"] == 1) & (bb["LTD"] == 0)).astype(int)
    bb["opp_life_no_std"] = ((bb["Life"] == 1) & (bb["STD"] == 0)).astype(int)
    bb["opp_std_no_ltd"] = ((bb["STD"] == 1) & (bb["LTD"] == 0)).astype(int)

    bb["opportunity_score"] = (
        bb["opp_life_no_ltd"] * w1
        + bb["opp_life_no_std"] * w2
        + bb["opp_std_no_ltd"] * w3
    )

    # Broker-level aggregates
    ai = (
        bb.groupby("Broker", as_index=False)
        .agg(
            employers=("Employer", "nunique"),
            comm_norm_sum=("comm_norm", "sum"),
            comm_norm_median=("comm_norm", "median"),
            opp_score=("opportunity_score", "sum"),
            life_no_ltd=("opp_life_no_ltd", "sum"),
            life_no_std=("opp_life_no_std", "sum"),
            std_no_ltd=("opp_std_no_ltd", "sum"),
        )
    )

    ai["opp_per_100_employers"] = (ai["opp_score"] / ai["employers"].clip(lower=1)) * 100.0

    # Concentration risk (HHI) based on comm_norm distribution across employers within each broker
    # Higher HHI => broker depends on few accounts => risk.
    # Compute HHI cheaply by taking top employer comm shares per broker.
    hhi_map = {}
    for broker, grp in bb.groupby("Broker"):
        comms = grp.groupby("Employer", as_index=False)["comm_norm"].sum()["comm_norm"].to_numpy()
        total = comms.sum()
        if total <= 0:
            hhi_map[broker] = 0.0
            continue
        shares = comms / total
        hhi_map[broker] = hhi_from_shares(shares)

    ai["hhi_concentration"] = ai["Broker"].map(hhi_map).fillna(0.0)

    # CEO-friendly estimated impact model (range)
    # Expected wins = opp_score * conversion_rate
    # Expected commission per win ≈ median_comm_norm * uplift_multiplier
    # Impact = wins * per-win commission
    ai["expected_wins"] = ai["opp_score"] * float(conversion)
    ai["rev_impact_base"] = ai["expected_wins"] * float(median_comm) * float(uplift)
    ai["rev_impact_low"] = ai["rev_impact_base"] * 0.6
    ai["rev_impact_high"] = ai["rev_impact_base"] * 1.4

    # Composite CEO score (actionability):
    # - reward impact
    # - reward opp rate
    # - penalize concentration (risk)
    ai["score_ceo"] = (
        zscore(ai["rev_impact_base"]) * 0.55
        + zscore(ai["opp_per_100_employers"]) * 0.30
        + zscore(ai["comm_norm_sum"]) * 0.25
        - zscore(ai["hhi_concentration"]) * float(conc_penalty) * 0.35
    )

    return ai.sort_values("score_ceo", ascending=False)


@st.cache_data(show_spinner=False)
def build_carrier_ai(epc_df: pd.DataFrame, conc_penalty: float) -> pd.DataFrame:
    # Aggregate employer counts and covered lives by carrier/product
    cp = (
        epc_df.groupby(["Carrier", "Product"], as_index=False)
        .agg(
            employers=("Employer", "nunique"),
            covered_lives=("Covered_Lives", "sum"),
        )
    )

    emp = cp.pivot_table(index="Carrier", columns="Product", values="employers", aggfunc="sum", fill_value=0).reset_index()
    lives = cp.pivot_table(index="Carrier", columns="Product", values="covered_lives", aggfunc="sum", fill_value=0).reset_index()

    for col in ["Life", "STD", "LTD"]:
        if col not in emp.columns:
            emp[col] = 0
        if col not in lives.columns:
            lives[col] = 0.0

    ai = emp.merge(lives, on="Carrier", suffixes=("_emp", "_lives"))

    # Shares on employer footprint (robust to reporting noise)
    ai["total_emp"] = ai["Life_emp"] + ai["STD_emp"] + ai["LTD_emp"]
    ai["life_share"] = ai["Life_emp"] / ai["total_emp"].clip(lower=1)
    ai["std_share"] = ai["STD_emp"] / ai["total_emp"].clip(lower=1)
    ai["ltd_share"] = ai["LTD_emp"] / ai["total_emp"].clip(lower=1)

    # White-space / imbalance score:
    # Life-heavy carriers with low STD/LTD are prime up-sell targets.
    ai["whitespace_ltd"] = ai["life_share"] * (1.0 - ai["ltd_share"])
    ai["whitespace_std"] = ai["life_share"] * (1.0 - ai["std_share"])
    ai["imbalance_score"] = (ai["whitespace_ltd"] * 60) + (ai["whitespace_std"] * 40)

    # Concentration risk for carriers based on employer distribution (by covered lives or employers)
    # We'll use employer counts here for stability: share of employers by product across the carrier.
    # (If the carrier is entirely one product, that’s a business concentration risk flag.)
    ai["product_concentration"] = (ai["life_share"] ** 2 + ai["std_share"] ** 2 + ai["ltd_share"] ** 2)

    # CEO score for carriers:
    # - reward total employers and total lives
    # - reward imbalance (whitespace)
    # - penalize product concentration
    ai["total_lives"] = ai["Life_lives"] + ai["STD_lives"] + ai["LTD_lives"]
    ai["score_ceo"] = (
        zscore(ai["total_emp"]) * 0.35
        + zscore(ai["total_lives"]) * 0.25
        + zscore(ai["imbalance_score"]) * 0.55
        - zscore(ai["product_concentration"]) * float(conc_penalty) * 0.35
    )

    return ai.sort_values("score_ceo", ascending=False)


# Build views
broker_book = build_broker_book(ebc_f, gaps_f)
broker_ai = build_broker_ai(
    broker_book=broker_book,
    w1=w_life_no_ltd,
    w2=w_life_no_std,
    w3=w_std_no_ltd,
    conc_penalty=concentration_penalty,
    median_comm=median_comm_norm,
    conversion=conversion_rate,
    uplift=uplift_multiplier,
)

carrier_ai = build_carrier_ai(epc_f, concentration_penalty)


# ============================================================
# 8) Executive Overview tab (CEO summary story)
# ============================================================
with tabs[0]:
    st.subheader("Executive Overview")

    # ============================================================
# STRATEGIC BRIEF (Decision Engine Layer)
# ============================================================
st.markdown("### 🧠 Strategic Brief (Auto-generated)")

top_b = broker_ai.head(1).iloc[0]
top_c = carrier_ai.head(1).iloc[0]

# Portfolio-wide opportunity sizing (pilot)
total_opp = float(broker_ai["opp_score"].sum())
expected_wins_total = total_opp * float(conversion_rate)
impact_total_base = expected_wins_total * float(median_comm_norm) * float(uplift_multiplier)

brief = f"""
**What we’re seeing**
- The biggest growth lever is **disability expansion (especially LTD)** inside broker books that already place Life.
- Commission data contains extreme outliers, so rankings use **normalized commissions (capped at {cap_mode})**.

**What it means**
- The most actionable path is targeting brokers with: **high LTD gap density**, **large employer footprint**, and **low concentration risk**.
- Carrier strategy: the highest white-space carriers show **Life-heavy footprint with low LTD penetration**.

**What to do next (next 30 days)**
1) Start with **{top_b['Broker']}** (highest CEO score): focus on employers missing LTD/STD; estimated base upside **${top_b['rev_impact_base']:,.0f}**.
2) Replicate playbook across the **Top 10 brokers** (ranked by CEO score) to scale impact.
3) For carrier partnerships, prioritize **{top_c['Carrier']}** (highest CEO score) where LTD white space is highest.

**Scenario-based upside (portfolio)**
- Total weighted opportunity score: **{total_opp:,.0f}**
- Expected wins (score × conversion): **{expected_wins_total:,.0f}**
- Estimated incremental commissions (base): **${impact_total_base:,.0f}**
"""

st.info(brief)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Top 10 CEO Targets (Brokers)")
        top10 = broker_ai.head(10).copy()
        show_cols = [
            "Broker", "employers", "opp_score", "opp_per_100_employers",
            "rev_impact_low", "rev_impact_base", "rev_impact_high",
            "hhi_concentration", "score_ceo",
        ]
        top10_display = top10[show_cols].copy()
        st.dataframe(top10_display, use_container_width=True, height=320)

        st.caption(
            "Interpretation: ranked by CEO composite score (impact + opportunity rate + book size − concentration risk)."
        )

    with right:
        st.markdown("### Top 10 White-space Carriers (Life-heavy imbalance)")
        topc = carrier_ai.head(10).copy()
        show_cols_c = [
            "Carrier", "total_emp", "total_lives",
            "Life_emp", "STD_emp", "LTD_emp",
            "life_share", "std_share", "ltd_share",
            "imbalance_score", "product_concentration", "score_ceo",
        ]
        st.dataframe(topc[show_cols_c], use_container_width=True, height=320)

        st.caption("Interpretation: high imbalance score suggests strong Life presence with STD/LTD white space.")

    st.divider()

    # CEO “Market Map” scatterplots (fast + informative)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Broker Market Map (Opportunity vs Book Size)")
        scatter = broker_ai.head(max(300, top_n)).copy()
        fig = px.scatter(
            scatter,
            x="employers",
            y="opp_per_100_employers",
            size="rev_impact_base",
            hover_name="Broker",
            title="Brokers: Employers vs Opportunity Rate (bubble = estimated impact)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Carrier Market Map (Imbalance vs Total Employers)")
        cs = carrier_ai.head(max(300, top_n)).copy()
        fig2 = px.scatter(
            cs,
            x="total_emp",
            y="imbalance_score",
            size="total_lives",
            hover_name="Carrier",
            title="Carriers: Footprint vs Imbalance (bubble = total lives)",
        )
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# 9) Opportunity Radar (CEO “next opportunities” panel)
#    10–15 CEO-relevant “AI” signals surfaced as cards + ranked lists
# ============================================================
with tabs[1]:
    st.subheader("Opportunity Radar (CEO Signals)")

    # Signal 1: Biggest gap counts (market-level)
    total_employers = gaps_f["Employer"].nunique()
    missing_std = int((gaps_f["STD"] == 0).sum())
    missing_ltd = int((gaps_f["LTD"] == 0).sum())
    missing_life = int((gaps_f["Life"] == 0).sum())

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Employers missing STD", f"{missing_std:,}", f"{(missing_std / max(total_employers,1)):.1%}")
    s2.metric("Employers missing LTD", f"{missing_ltd:,}", f"{(missing_ltd / max(total_employers,1)):.1%}")
    s3.metric("Employers missing Life", f"{missing_life:,}", f"{(missing_life / max(total_employers,1)):.1%}")
    s4.metric("Median comm (norm)", f"${median_comm_norm:,.0f}")

    st.divider()

    # Signal 2–5: Broker targets by different CEO angles
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Signal: Top brokers by expected revenue impact")
        tbl = broker_ai.sort_values("rev_impact_base", ascending=False).head(top_n)
        st.dataframe(tbl[[
            "Broker", "employers", "opp_score", "expected_wins",
            "rev_impact_low", "rev_impact_base", "rev_impact_high",
            "hhi_concentration"
        ]], use_container_width=True)

    with c2:
        st.markdown("### Signal: Best efficiency (opportunity rate, low concentration)")
        tbl2 = broker_ai.sort_values(["opp_per_100_employers", "hhi_concentration"], ascending=[False, True]).head(top_n)
        st.dataframe(tbl2[[
            "Broker", "employers", "opp_per_100_employers", "opp_score",
            "rev_impact_base", "hhi_concentration"
        ]], use_container_width=True)

    st.divider()

    # Signal 6–8: Carrier white-space + stability
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("### Signal: Top carriers by LTD white space (Life-heavy with low LTD)")
        tbl3 = carrier_ai.sort_values("whitespace_ltd", ascending=False).head(top_n)
        st.dataframe(tbl3[[
            "Carrier", "total_emp", "Life_emp", "LTD_emp", "whitespace_ltd", "imbalance_score"
        ]], use_container_width=True)

    with c4:
        st.markdown("### Signal: Largest carriers with high imbalance (strategic partnerships)")
        tbl4 = carrier_ai.sort_values(["imbalance_score", "total_emp"], ascending=[False, False]).head(top_n)
        st.dataframe(tbl4[[
            "Carrier", "total_emp", "total_lives", "imbalance_score", "product_concentration"
        ]], use_container_width=True)

    st.divider()

    # Signal 9–12: “Deal-like” insights (M&A / roll-up style heuristics)
    # NOTE: This is heuristic; we clearly label as pilot.
    st.markdown("## M&A Signals (Pilot Heuristics)")

    # Broker M&A thesis:
    # - Large employer book
    # - High normalized commissions
    # - Low concentration risk (more diversified)
    # - High cross-sell opportunity (post-acquisition growth)
    broker_ma = broker_ai.copy()
    broker_ma["ma_score"] = (
        zscore(broker_ma["comm_norm_sum"]) * 0.40
        + zscore(broker_ma["employers"]) * 0.25
        + zscore(broker_ma["opp_score"]) * 0.35
        - zscore(broker_ma["hhi_concentration"]) * 0.40
    )
    top_ma = broker_ma.sort_values("ma_score", ascending=False).head(top_n)

    st.markdown("### Signal: Broker roll-up targets (size + diversification + growth headroom)")
    st.dataframe(top_ma[[
        "Broker", "employers", "comm_norm_sum", "opp_score", "opp_per_100_employers",
        "hhi_concentration", "ma_score"
    ]], use_container_width=True)

    # Carrier M&A / partnership thesis:
    # - Big footprint (employers/lives)
    # - High imbalance (white space for product expansion)
    # - Lower product concentration is generally healthier (but depends on strategy)
    carrier_ma = carrier_ai.copy()
    carrier_ma["ma_score"] = (
        zscore(carrier_ma["total_emp"]) * 0.35
        + zscore(carrier_ma["total_lives"]) * 0.20
        + zscore(carrier_ma["imbalance_score"]) * 0.45
        - zscore(carrier_ma["product_concentration"]) * 0.35
    )
    top_cma = carrier_ma.sort_values("ma_score", ascending=False).head(top_n)

    st.markdown("### Signal: Carrier targets (footprint + imbalance + product expansion thesis)")
    st.dataframe(top_cma[[
        "Carrier", "total_emp", "total_lives",
        "Life_emp", "STD_emp", "LTD_emp",
        "imbalance_score", "product_concentration", "ma_score"
    ]], use_container_width=True)

    st.caption(
        "M&A signals are heuristics intended for prioritization. Due diligence would validate premium, persistency, margin, and retention."
    )


# ============================================================
# 10) Lens Dashboard (deep dive + drilldowns)
# ============================================================
with tabs[2]:
    if lens == "Broker Lens":
        st.subheader("Broker Lens — Cross-sell & Growth Engine")

        left, right = st.columns([1, 1])

        with left:
            st.markdown("### Top brokers by CEO score")
            view = broker_ai.head(top_n).copy()
            fig = px.bar(view, x="Broker", y="score_ceo", title=f"Top {top_n} Brokers by CEO Score")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(view[[
                "Broker", "score_ceo", "employers", "opp_score", "opp_per_100_employers",
                "rev_impact_base", "hhi_concentration"
            ]], use_container_width=True, height=330)

        with right:
            st.markdown("### Top brokers by expected revenue impact")
            view2 = broker_ai.sort_values("rev_impact_base", ascending=False).head(top_n)
            fig2 = px.bar(view2, x="Broker", y="rev_impact_base", title=f"Top {top_n} Brokers by Estimated Impact (Base)")
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(view2[[
                "Broker", "employers", "expected_wins",
                "rev_impact_low", "rev_impact_base", "rev_impact_high",
                "opp_score", "hhi_concentration"
            ]], use_container_width=True, height=330)

        st.divider()

        st.markdown("## Broker Drilldown")
        broker_list = ["(Select)"] + broker_ai["Broker"].head(5000).tolist()
        selected = st.selectbox("Select a broker", broker_list)

        if selected != "(Select)":
            bb = broker_book[broker_book["Broker"] == selected].copy()

            bb["opp_life_no_ltd"] = ((bb["Life"] == 1) & (bb["LTD"] == 0)).astype(int)
            bb["opp_life_no_std"] = ((bb["Life"] == 1) & (bb["STD"] == 0)).astype(int)
            bb["opp_std_no_ltd"] = ((bb["STD"] == 1) & (bb["LTD"] == 0)).astype(int)

            bb["opportunity_score"] = (
                bb["opp_life_no_ltd"] * w_life_no_ltd
                + bb["opp_life_no_std"] * w_life_no_std
                + bb["opp_std_no_ltd"] * w_std_no_ltd
            )

            st.write(f"**{selected}**")
            st.write(f"Employers: **{bb['Employer'].nunique():,}** • Median comm (norm): **${bb['comm_norm'].median():,.0f}**")

            # Top employer opportunity list
            show = bb.sort_values(["opportunity_score", "comm_norm"], ascending=[False, False]).head(100)[
                ["Employer", "Life", "STD", "LTD", "opportunity_score", "comm_norm"]
            ]
            st.markdown("### Highest-value employer targets (sample)")
            st.dataframe(show, use_container_width=True, height=380)

            # Recommended actions panel (AI-consultant style)
            st.divider()
            st.markdown("## Top Recommended Actions (for this broker)")

            opp_total = float(bb["opportunity_score"].sum())
            exp_wins = opp_total * conversion_rate
            per_win = median_comm_norm * uplift_multiplier
            impact_base = exp_wins * per_win

            st.markdown(
                f"""
**Action Plan Summary**
- **Why this broker:** high weighted white-space in LTD/STD relative to their book.
- **Opportunity score:** **{opp_total:,.1f}**
- **Expected wins (scenario):** **{exp_wins:,.1f}** (score × conversion rate)
- **Estimated incremental commissions (base):** **${impact_base:,.0f}**
"""
            )

            st.caption(
                "Pilot: This is a prioritization model. A production model would incorporate premium proxies, persistency, renewal dates, and carrier appetite."
            )

    elif lens == "Carrier Lens":
        st.subheader("Carrier Lens — White Space, Footprint, Product Strategy")

        left, right = st.columns([1, 1])

        with left:
            st.markdown("### Top carriers by CEO score")
            view = carrier_ai.head(top_n).copy()
            fig = px.bar(view, x="Carrier", y="score_ceo", title=f"Top {top_n} Carriers by CEO Score")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(view[[
                "Carrier", "score_ceo", "total_emp", "total_lives",
                "imbalance_score", "product_concentration"
            ]], use_container_width=True, height=330)

        with right:
            st.markdown("### Top carriers by imbalance (Life-heavy white space)")
            view2 = carrier_ai.sort_values("imbalance_score", ascending=False).head(top_n)
            fig2 = px.bar(view2, x="Carrier", y="imbalance_score", title=f"Top {top_n} Carriers by Imbalance Score")
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(view2[[
                "Carrier", "total_emp", "Life_emp", "STD_emp", "LTD_emp",
                "life_share", "std_share", "ltd_share", "imbalance_score"
            ]], use_container_width=True, height=330)

        st.divider()

        st.markdown("## Carrier Drilldown")
        carrier_list = ["(Select)"] + carrier_ai["Carrier"].head(5000).tolist()
        selected = st.selectbox("Select a carrier", carrier_list)

        if selected != "(Select)":
            row = carrier_ai[carrier_ai["Carrier"] == selected].iloc[0]
            st.write(f"**{selected}**")
            st.write(f"Employers: **{int(row['total_emp']):,}** • Lives: **{int(row['total_lives']):,}**")
            st.write(
                f"Shares — Life: **{row['life_share']:.1%}** • STD: **{row['std_share']:.1%}** • LTD: **{row['ltd_share']:.1%}**"
            )
            st.info(
                "Interpretation: higher imbalance suggests a Life-heavy footprint with disability expansion headroom."
            )

            st.markdown("### Sample employer footprint (filtered by product dropdown)")
            tmp = epc_f[(epc_f["Carrier"] == selected) & (epc_f["Product"] == product)].copy()
            sample = (
                tmp.groupby("Employer", as_index=False)
                .agg(covered_lives=("Covered_Lives", "sum"))
                .sort_values("covered_lives", ascending=False)
                .head(100)
            )
            st.dataframe(sample, use_container_width=True, height=360)

    else:
        st.subheader("M&A Radar — Roll-up + Partnership Targets")

        st.markdown("### Broker Roll-up Targets (heuristic)")
        broker_ma = broker_ai.copy()
        broker_ma["ma_score"] = (
            zscore(broker_ma["comm_norm_sum"]) * 0.40
            + zscore(broker_ma["employers"]) * 0.25
            + zscore(broker_ma["opp_score"]) * 0.35
            - zscore(broker_ma["hhi_concentration"]) * 0.40
        )
        st.dataframe(
            broker_ma.sort_values("ma_score", ascending=False).head(top_n)[
                ["Broker", "employers", "comm_norm_sum", "opp_score", "opp_per_100_employers", "hhi_concentration", "ma_score"]
            ],
            use_container_width=True
        )

        st.markdown("### Carrier Targets (heuristic)")
        carrier_ma = carrier_ai.copy()
        carrier_ma["ma_score"] = (
            zscore(carrier_ma["total_emp"]) * 0.35
            + zscore(carrier_ma["total_lives"]) * 0.20
            + zscore(carrier_ma["imbalance_score"]) * 0.45
            - zscore(carrier_ma["product_concentration"]) * 0.35
        )
        st.dataframe(
            carrier_ma.sort_values("ma_score", ascending=False).head(top_n)[
                ["Carrier", "total_emp", "total_lives", "imbalance_score", "product_concentration", "ma_score"]
            ],
            use_container_width=True
        )

        st.caption(
            "Pilot note: these are prioritization signals only. M&A diligence would validate premium, margin, retention, and contractual transferability."
        )


# ============================================================
# 11) AI Analyst Chat (typed Q&A)
# ============================================================
with tabs[3]:
    st.subheader("AI Analyst (Pilot Chat)")
    st.caption(
        "Type questions like:\n"
        "• Which brokers have the biggest LTD white space?\n"
        "• Give me top 10 recommended actions.\n"
        "• Which carriers are Life-heavy but weak in LTD?\n"
        "• How many employers have Life but no STD?\n"
        "• List broker roll-up targets with diversification.\n"
    )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Render recent history
    for msg in st.session_state.chat[-12:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask the AI analyst…")
    if user_q:
        st.session_state.chat.append({"role": "user", "content": user_q})
        ql = user_q.lower()

        # Deterministic “AI analyst” responses (fast, no external keys)
        response_lines = []

        def money(x: float) -> str:
            return f"${x:,.0f}"

        # Routing
        if "recommended actions" in ql or ("top 10" in ql and "action" in ql):
            top10 = broker_ai.head(10)
            response_lines.append("### Top 10 Recommended Actions (CEO view)")
            for i, r in enumerate(top10.itertuples(index=False), start=1):
                response_lines.append(
                    f"**{i}. Target {r.Broker}** — "
                    f"{int(r.employers):,} employers • "
                    f"Opp score: **{float(r.opp_score):,.1f}** • "
                    f"Impact: **{money(float(r.rev_impact_low))} – {money(float(r.rev_impact_high))}** "
                    f"(base {money(float(r.rev_impact_base))})"
                )
            response_lines.append(
                f"\n**Model note:** impact is scenario-based: wins = opp_score × conversion ({conversion_rate:.0%}); "
                f"commission per win ≈ median_comm_norm × uplift ({uplift_multiplier:.1f})."
            )

        elif "broker" in ql and ("white space" in ql or "opportunity" in ql or "gap" in ql):
            response_lines.append("### Top brokers by cross-sell opportunity (weighted)")
            view = broker_ai.sort_values("opp_score", ascending=False).head(10)[
                ["Broker", "employers", "opp_score", "opp_per_100_employers", "rev_impact_base", "hhi_concentration"]
            ]
            response_lines.append(view.to_markdown(index=False))

            response_lines.append(
                "\n**Interpretation:** prioritize brokers with high opp score AND reasonable concentration risk. "
                "These are the fastest routes to revenue expansion."
            )

        elif "carrier" in ql and ("life-heavy" in ql or "weak" in ql or "imbalance" in ql or "white space" in ql):
            response_lines.append("### Top carriers by LTD white space (Life footprint with low LTD)")
            view = carrier_ai.sort_values("whitespace_ltd", ascending=False).head(10)[
                ["Carrier", "total_emp", "Life_emp", "LTD_emp", "whitespace_ltd", "imbalance_score"]
            ]
            response_lines.append(view.to_markdown(index=False))
            response_lines.append(
                "\n**Interpretation:** strong Life presence + low LTD penetration suggests a disability expansion thesis "
                "(partnering, targeted broker alignment, or product bundling)."
            )

        elif "how many" in ql and ("missing" in ql or "gap" in ql):
            target = "STD" if "std" in ql else ("LTD" if "ltd" in ql else ("Life" if "life" in ql else None))
            if target:
                cnt = int((gaps_f[target] == 0).sum())
                response_lines.append(f"**Employers missing {target}: {cnt:,}**")
            else:
                response_lines.append("Ask: “How many employers are missing STD/LTD/Life?”")

        elif "life but no" in ql:
            if "std" in ql:
                subset = gaps_f[(gaps_f["Life"] == 1) & (gaps_f["STD"] == 0)]
                response_lines.append(f"**Employers with Life but no STD: {len(subset):,}** (showing sample 20)")
                response_lines.append(subset.head(20).to_markdown(index=False))
            elif "ltd" in ql:
                subset = gaps_f[(gaps_f["Life"] == 1) & (gaps_f["LTD"] == 0)]
                response_lines.append(f"**Employers with Life but no LTD: {len(subset):,}** (showing sample 20)")
                response_lines.append(subset.head(20).to_markdown(index=False))
            else:
                response_lines.append("Try: “How many employers have Life but no LTD?”")

        elif "m&a" in ql or "acquisition" in ql or "roll-up" in ql:
            broker_ma = broker_ai.copy()
            broker_ma["ma_score"] = (
                zscore(broker_ma["comm_norm_sum"]) * 0.40
                + zscore(broker_ma["employers"]) * 0.25
                + zscore(broker_ma["opp_score"]) * 0.35
                - zscore(broker_ma["hhi_concentration"]) * 0.40
            )
            top = broker_ma.sort_values("ma_score", ascending=False).head(10)[
                ["Broker", "employers", "comm_norm_sum", "opp_score", "hhi_concentration", "ma_score"]
            ]
            response_lines.append("### Broker roll-up targets (pilot heuristic)")
            response_lines.append(top.to_markdown(index=False))
            response_lines.append(
                "\n**Interpretation:** prioritizes diversified brokers with sizable books and strong post-acquisition growth headroom."
            )

        else:
            response_lines.append(
                "I can help with:\n"
                "- Broker opportunity / white space\n"
                "- Carrier imbalance / product strategy\n"
                "- Employer gap sizing\n"
                "- M&A prioritization heuristics\n\n"
                "Try: **“Give me top 10 recommended actions”** or **“Which carriers are Life-heavy but weak in LTD?”**"
            )

        assistant_content = "\n\n".join(response_lines)

        st.session_state.chat.append({"role": "assistant", "content": assistant_content})
        st.rerun()


# ============================================================
# 12) Gaps Matrix (red/green)
# ============================================================
with tabs[4]:
    st.subheader("Gaps Matrix (Employer Product Coverage)")

    g = gaps_f.copy()
    max_rows = st.slider("Max rows to display", min_value=25, max_value=500, value=200, step=25)
    g = g.head(max_rows)

    def color_cell(val):
        try:
            v = int(val)
        except Exception:
            v = 0
        return "background-color: #d4f8d4; font-weight: 600;" if v == 1 else "background-color: #ffd6d6; font-weight: 600;"

    styled = g.style.applymap(color_cell, subset=["Life", "STD", "LTD"])
    st.caption("Green = covered, Red = gap (missing product).")
    st.dataframe(styled, use_container_width=True, height=450)

    st.divider()
    st.markdown("### Quick Gap Counts")
    c1, c2, c3 = st.columns(3)
    c1.metric("Life covered", f"{int((gaps_f['Life'] == 1).sum()):,}")
    c2.metric("STD covered", f"{int((gaps_f['STD'] == 1).sum()):,}")
    c3.metric("LTD covered", f"{int((gaps_f['LTD'] == 1).sum()):,}")


# ============================================================
# 13) Data QA / Methods (for demo credibility)
# ============================================================
with tabs[5]:
    st.subheader("Data QA / Methods (Demo Transparency Panel)")

    st.markdown("### Key distribution checks (commissions)")
    st.write(f"Median (raw): **${median_comm_raw:,.0f}**")
    st.write(f"Median (normalized): **${median_comm_norm:,.0f}**")
    st.write(f"95th percentile: **${q95:,.0f}**")
    st.write(f"99th percentile: **${q99:,.0f}**")
    st.write(f"Max (raw): **${float(ebc_f['total_commissions'].max()):,.0f}**")

    st.caption(
        "We cap commissions by percentile to prevent extreme outliers from distorting totals/means. "
        "The median reflects typical employer↔broker commission relationships."
    )

    st.markdown("### Scenario assumptions used in Estimated Revenue Impact")
    st.write(f"- Conversion rate: **{conversion_rate:.0%}**")
    st.write(f"- Per-win commission proxy: **median_comm_norm × uplift = {money(median_comm_norm)} × {uplift_multiplier:.1f}**")
    st.write(
        "- Estimated wins: **opp_score × conversion_rate**\n"
        "- Estimated impact: **wins × (median_comm_norm × uplift)**\n"
        "- Low/high bands: **±40%** around base (pilot uncertainty band)."
    )

    st.markdown("### Opportunity score (weighted gaps)")
    st.write(
        f"- Life but missing LTD: **{w_life_no_ltd:.1f}**\n"
        f"- Life but missing STD: **{w_life_no_std:.1f}**\n"
        f"- STD but missing LTD: **{w_std_no_ltd:.1f}**"
    )
    st.caption(
        "Weights are CEO-oriented: LTD is typically higher LTV / stickier; hence a higher weight by default."
    )

    st.markdown("### Concentration risk")
    st.write(
        "- Broker concentration uses HHI on employer commission shares within broker.\n"
        "- Higher HHI = more reliant on a few accounts.\n"
        f"- Penalty strength = **{concentration_penalty:.1f}** (adjustable)."
    )

    st.markdown("### Row counts (sanity)")
    qa1, qa2, qa3 = st.columns(3)
    qa1.metric("Rows: EPC", f"{len(epc_f):,}")
    qa2.metric("Rows: EBC", f"{len(ebc_f):,}")
    qa3.metric("Rows: GAPS", f"{len(gaps_f):,}")

    st.markdown("### Sample data (first 10 rows)")
    st.write("Employer↔Broker commissions (EBC):")
    st.dataframe(ebc_f.head(10), use_container_width=True)
    st.write("Employer↔Product↔Carrier (EPC):")
    st.dataframe(epc_p.head(10), use_container_width=True)