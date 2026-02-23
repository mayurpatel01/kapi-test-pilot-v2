from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 0) Page setup
# ============================================================
st.set_page_config(page_title="Test Pilot", layout="wide", page_icon="🧭")


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
# 2) Helpers
# ============================================================
def money(x: float) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "$0"


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    if len(s) == 0:
        return pd.Series([], dtype=float)
    sd = float(s.std(ddof=0))
    if sd == 0:
        return pd.Series([0.0] * len(s), index=s.index, dtype=float)
    return (s - float(s.mean())) / sd


def hhi_from_shares(shares: np.ndarray) -> float:
    # Herfindahl-Hirschman Index (0..1). Higher = more concentrated.
    if shares.size == 0:
        return 0.0
    return float(np.sum(np.square(shares)))


# ============================================================
# 3) Load marts
# ============================================================
DATA_DIR = Path("data") / "marts"


@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"Missing mart: {p}. Ensure data/marts exists in your repo and redeploy."
        )
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def compute_caps(series: pd.Series) -> Tuple[float, float]:
    series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(series) == 0:
        return 0.0, 0.0
    return float(series.quantile(0.95)), float(series.quantile(0.99))


def ensure_str(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str)
    return df


# Required marts
epc = ensure_str(load_parquet("employer_product_carrier.parquet"))      # Employer x Product x Carrier (+ Covered_Lives)
ebc = ensure_str(load_parquet("employer_broker_commissions.parquet"))   # Employer x Broker (+ total_commissions)
gaps = ensure_str(load_parquet("employer_product_matrix.parquet"))      # Employer x Life/STD/LTD flags

# Load geo mart
geo = ensure_str(load_parquet("employer_geo.parquet"))

# Keep only needed geo fields
geo = geo[["Employer", "State", "ZIP", "City"]]

# Merge geo into gaps dataset (primary modeling table)
gaps = gaps.merge(geo, on="Employer", how="left")

# (Optional but recommended) also merge into epc and ebc for flexibility later
epc = epc.merge(geo, on="Employer", how="left")
ebc = ebc.merge(geo, on="Employer", how="left")

# Numeric cleanup
if "Covered_Lives" in epc.columns:
    epc["Covered_Lives"] = pd.to_numeric(epc["Covered_Lives"], errors="coerce").fillna(0.0)

if "total_commissions" in ebc.columns:
    ebc["total_commissions"] = pd.to_numeric(ebc["total_commissions"], errors="coerce").fillna(0.0)

for col in ["Life", "STD", "LTD"]:
    if col in gaps.columns:
        gaps[col] = pd.to_numeric(gaps[col], errors="coerce").fillna(0).astype("int8")


# ============================================================
# 4) Sidebar controls (CEO controls)
# ============================================================
st.title("🧭 Test Pilot — CEO Opportunity Intelligence (Form 5500)")
st.caption(
    "Decision-engine style prioritization built from Form 5500 marts: "
    "cross-sell gaps, carrier white space, and M&A heuristics. "
    "All assumptions are visible in the Methods tab."
)

st.sidebar.header("CEO Controls")

lens = st.sidebar.radio("Default lens", ["Broker Lens", "Carrier Lens", "M&A Radar"], index=0)

product_view = st.sidebar.selectbox("Product view (charts)", ["Life", "STD", "LTD"], index=0)
top_n = st.sidebar.slider("Top N", 10, 200, 25, 5)

employer_search = st.sidebar.text_input("Employer contains (optional)", value="").strip()

st.sidebar.divider()
st.sidebar.subheader("Normalization")

cap_mode = st.sidebar.selectbox(
    "Commission cap",
    ["99th percentile (recommended)", "95th percentile", "None (raw)"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Scenario (Estimated Revenue Impact)")

conversion_rate = st.sidebar.slider(
    "Conversion rate (wins from opportunities)",
    0.0, 0.50, 0.10, 0.01,
    help="Pilot: percent of scored opportunities assumed to convert."
)
uplift_multiplier = st.sidebar.slider(
    "Commission per win (vs median)",
    0.5, 3.0, 1.0, 0.1,
    help="Pilot proxy: commission per win ≈ median_normalized_commission × uplift."
)

st.sidebar.divider()
st.sidebar.subheader("Opportunity weights (CEO LTV logic)")

w_life_no_ltd = st.sidebar.slider("Life but missing LTD (highest LTV)", 0.5, 3.0, 1.5, 0.1)
w_life_no_std = st.sidebar.slider("Life but missing STD", 0.5, 3.0, 1.0, 0.1)
w_std_no_ltd = st.sidebar.slider("STD but missing LTD", 0.0, 3.0, 1.0, 0.1)

st.sidebar.divider()
st.sidebar.subheader("Risk")

concentration_penalty = st.sidebar.slider(
    "Concentration penalty strength",
    0.0, 2.0, 0.7, 0.1,
    help="Penalizes brokers/carriers concentrated in a few accounts/products."
)


# ============================================================
# 5) Filters
# ============================================================
def filter_by_employer(df: pd.DataFrame) -> pd.DataFrame:
    if employer_search and "Employer" in df.columns:
        return df[df["Employer"].str.contains(employer_search, case=False, na=False)].copy()
    return df.copy()


epc_f = filter_by_employer(epc)
ebc_f = filter_by_employer(ebc)
gaps_f = filter_by_employer(gaps)

epc_p = epc_f[epc_f["Product"] == product_view].copy()

import plotly.express as px

st.subheader("Employer Footprint by State")

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

city_counts = (
    gaps.dropna(subset=["City", "State"])
        .assign(CityState=lambda d: d["City"].str.title() + ", " + d["State"])
        .groupby("CityState", as_index=False)
        .agg(Employers=("Employer", "nunique"))
        .sort_values("Employers", ascending=False)
        .head(25)
)

st.dataframe(city_counts, use_container_width=True)


# ============================================================
# 6) Commission normalization
# ============================================================
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

median_comm_norm = float(pd.to_numeric(ebc_f["comm_norm"], errors="coerce").fillna(0.0).median()) if len(ebc_f) else 0.0
median_comm_raw = float(pd.to_numeric(ebc_f["total_commissions"], errors="coerce").fillna(0.0).median()) if len(ebc_f) else 0.0
max_comm_raw = float(pd.to_numeric(ebc_f["total_commissions"], errors="coerce").fillna(0.0).max()) if len(ebc_f) else 0.0


# ============================================================
# 7) Build analytic views (Broker / Carrier)
# ============================================================
@st.cache_data(show_spinner=False)
def build_broker_book(ebc_df: pd.DataFrame, gaps_df: pd.DataFrame) -> pd.DataFrame:
    g = gaps_df[["Employer", "Life", "STD", "LTD"]].copy()
    bb = ebc_df.groupby(["Broker", "Employer"], as_index=False).agg(
        comm_norm=("comm_norm", "sum"),
        comm_raw=("total_commissions", "sum"),
    )
    bb = bb.merge(g, on="Employer", how="left").fillna(0)
    # Ensure ints for flags
    for c in ["Life", "STD", "LTD"]:
        bb[c] = pd.to_numeric(bb[c], errors="coerce").fillna(0).astype("int8")
    return bb


@st.cache_data(show_spinner=False)
def build_broker_ai(
    broker_book: pd.DataFrame,
    w1: float, w2: float, w3: float,
    median_comm: float,
    conversion: float,
    uplift: float,
    conc_penalty: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    bb = broker_book.copy()

    bb["opp_life_no_ltd"] = ((bb["Life"] == 1) & (bb["LTD"] == 0)).astype(int)
    bb["opp_life_no_std"] = ((bb["Life"] == 1) & (bb["STD"] == 0)).astype(int)
    bb["opp_std_no_ltd"] = ((bb["STD"] == 1) & (bb["LTD"] == 0)).astype(int)

    bb["opportunity_score"] = (
        bb["opp_life_no_ltd"] * float(w1)
        + bb["opp_life_no_std"] * float(w2)
        + bb["opp_std_no_ltd"] * float(w3)
    )

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

    # Broker concentration risk (HHI over employer commission shares)
    hhi_map: Dict[str, float] = {}
    for broker, grp in bb.groupby("Broker"):
        comms = grp.groupby("Employer", as_index=False)["comm_norm"].sum()["comm_norm"].to_numpy()
        total = comms.sum()
        if total <= 0:
            hhi_map[broker] = 0.0
        else:
            shares = comms / total
            hhi_map[broker] = hhi_from_shares(shares)

    ai["hhi_concentration"] = ai["Broker"].map(hhi_map).fillna(0.0)

    # Estimated revenue impact (scenario-based)
    ai["expected_wins"] = ai["opp_score"] * float(conversion)
    per_win = float(median_comm) * float(uplift)
    ai["rev_impact_base"] = ai["expected_wins"] * per_win
    ai["rev_impact_low"] = ai["rev_impact_base"] * 0.6
    ai["rev_impact_high"] = ai["rev_impact_base"] * 1.4

    # CEO composite score
    ai["score_ceo"] = (
        zscore(ai["rev_impact_base"]) * 0.55
        + zscore(ai["opp_per_100_employers"]) * 0.30
        + zscore(ai["comm_norm_sum"]) * 0.25
        - zscore(ai["hhi_concentration"]) * float(conc_penalty) * 0.35
    )

    # Portfolio stats
    stats = {
        "total_opp_score": float(ai["opp_score"].sum()),
        "total_expected_wins": float((ai["opp_score"].sum()) * float(conversion)),
        "portfolio_impact_base": float((ai["opp_score"].sum()) * float(conversion) * per_win),
        "per_win_commission_proxy": float(per_win),
    }

    ai = ai.sort_values("score_ceo", ascending=False)
    return ai, stats


@st.cache_data(show_spinner=False)
def build_carrier_ai(epc_df: pd.DataFrame, conc_penalty: float) -> pd.DataFrame:
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

    ai["total_emp"] = ai["Life_emp"] + ai["STD_emp"] + ai["LTD_emp"]
    ai["total_lives"] = ai["Life_lives"] + ai["STD_lives"] + ai["LTD_lives"]

    ai["life_share"] = ai["Life_emp"] / ai["total_emp"].clip(lower=1)
    ai["std_share"] = ai["STD_emp"] / ai["total_emp"].clip(lower=1)
    ai["ltd_share"] = ai["LTD_emp"] / ai["total_emp"].clip(lower=1)

    ai["whitespace_ltd"] = ai["life_share"] * (1.0 - ai["ltd_share"])
    ai["whitespace_std"] = ai["life_share"] * (1.0 - ai["std_share"])
    ai["imbalance_score"] = (ai["whitespace_ltd"] * 60) + (ai["whitespace_std"] * 40)

    # Product concentration (higher => concentrated in 1 product)
    ai["product_concentration"] = (ai["life_share"] ** 2 + ai["std_share"] ** 2 + ai["ltd_share"] ** 2)

    ai["score_ceo"] = (
        zscore(ai["total_emp"]) * 0.35
        + zscore(ai["total_lives"]) * 0.25
        + zscore(ai["imbalance_score"]) * 0.55
        - zscore(ai["product_concentration"]) * float(conc_penalty) * 0.35
    )

    return ai.sort_values("score_ceo", ascending=False)


broker_book = build_broker_book(ebc_f, gaps_f)
broker_ai, broker_stats = build_broker_ai(
    broker_book=broker_book,
    w1=w_life_no_ltd,
    w2=w_life_no_std,
    w3=w_std_no_ltd,
    median_comm=median_comm_norm,
    conversion=conversion_rate,
    uplift=uplift_multiplier,
    conc_penalty=concentration_penalty,
)
carrier_ai = build_carrier_ai(epc_f, concentration_penalty)


# ============================================================
# 8) Top KPIs (CEO-friendly)
# ============================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Unique Employers", f"{gaps_f['Employer'].nunique():,}")
k2.metric("Unique Brokers", f"{ebc_f['Broker'].nunique():,}")
k3.metric("Unique Carriers", f"{epc_p['Carrier'].nunique():,}")
k4.metric("Median Commission (normalized)", money(median_comm_norm))

k5, k6, k7, k8 = st.columns(4)
k5.metric(f"Covered Lives ({product_view})", f"{int(epc_p['Covered_Lives'].sum()):,}")
k6.metric("Commission cap", money(cap_value) if cap_value else "None")
k7.metric("95th / 99th", f"{money(q95)} / {money(q99)}")
k8.metric("Max (raw)", money(max_comm_raw))


# ============================================================
# 9) MAIN PAGE AI ANALYST (always visible, single chat_input)
# ============================================================
st.markdown("## 🤖 AI Analyst (always available)")
st.caption(
    "Try: “Top 10 recommended actions”, “Top brokers by LTD white space”, "
    "“Top carriers by LTD white space”, “M&A roll-up targets”."
)

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.expander("Open AI Analyst Chat", expanded=True):
    # Show last N messages
    for msg in st.session_state.chat[-10:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask the AI analyst…")
    if user_q:
        st.session_state.chat.append({"role": "user", "content": user_q})
        ql = user_q.lower()
        lines = []

        # ---- Response routing (deterministic “AI”) ----
        if "recommended" in ql and "action" in ql or ("top 10" in ql and "action" in ql):
            top10 = broker_ai.head(10)
            lines.append("### Top 10 Recommended Actions (CEO view)")
            for i, r in enumerate(top10.itertuples(index=False), start=1):
                lines.append(
                    f"**{i}. Target {r.Broker}** — {int(r.employers):,} employers • "
                    f"Opp **{float(r.opp_score):,.1f}** • "
                    f"Impact **{money(float(r.rev_impact_low))}–{money(float(r.rev_impact_high))}** "
                    f"(base {money(float(r.rev_impact_base))})"
                )
            lines.append(
                f"\nModel: wins = opp_score × conversion ({conversion_rate:.0%}); "
                f"per-win ≈ median_norm_comm × uplift = {money(median_comm_norm)} × {uplift_multiplier:.1f}."
            )

        elif "broker" in ql and ("ltd" in ql or "white space" in ql or "gap" in ql or "opportunity" in ql):
            view = broker_ai.sort_values("opp_score", ascending=False).head(10)
            lines.append("### Top brokers by weighted opportunity (white space)")
            for i, r in enumerate(view.itertuples(index=False), start=1):
                lines.append(
                    f"**{i}. {r.Broker}** — employers {int(r.employers):,} • "
                    f"opp {float(r.opp_score):,.1f} • rate {float(r.opp_per_100_employers):.1f}/100 • "
                    f"HHI {float(r.hhi_concentration):.3f}"
                )
            lines.append("\nInterpretation: higher opp + lower HHI tends to be the best near-term revenue target.")

        elif "carrier" in ql and ("ltd" in ql or "white space" in ql or "imbalance" in ql or "gap" in ql):
            view = carrier_ai.sort_values("whitespace_ltd", ascending=False).head(10)
            lines.append("### Top carriers by LTD white space (Life-heavy, low LTD)")
            for i, r in enumerate(view.itertuples(index=False), start=1):
                lines.append(
                    f"**{i}. {r.Carrier}** — total employers {int(r.total_emp):,} • "
                    f"Life_emp {int(r.Life_emp):,} • LTD_emp {int(r.LTD_emp):,} • "
                    f"whitespace_ltd {float(r.whitespace_ltd):.3f}"
                )
            lines.append("\nInterpretation: Life footprint + low LTD suggests a disability expansion thesis.")

        elif "m&a" in ql or "acquisition" in ql or "roll-up" in ql:
            # Broker M&A heuristic: size + diversification + growth headroom
            tmp = broker_ai.copy()
            tmp["ma_score"] = (
                zscore(tmp["comm_norm_sum"]) * 0.40
                + zscore(tmp["employers"]) * 0.25
                + zscore(tmp["opp_score"]) * 0.35
                - zscore(tmp["hhi_concentration"]) * 0.40
            )
            view = tmp.sort_values("ma_score", ascending=False).head(10)
            lines.append("### Broker roll-up targets (pilot heuristic)")
            for i, r in enumerate(view.itertuples(index=False), start=1):
                lines.append(
                    f"**{i}. {r.Broker}** — employers {int(r.employers):,} • "
                    f"comm_norm_sum {money(float(r.comm_norm_sum))} • opp {float(r.opp_score):,.1f} • "
                    f"HHI {float(r.hhi_concentration):.3f} • ma_score {float(r.ma_score):.2f}"
                )
            lines.append("\nInterpretation: diversified brokers with sizable books + post-acquisition cross-sell headroom.")

        elif "how many" in ql and ("missing" in ql or "gap" in ql):
            target = "STD" if "std" in ql else ("LTD" if "ltd" in ql else ("Life" if "life" in ql else None))
            if target:
                cnt = int((gaps_f[target] == 0).sum())
                lines.append(f"**Employers missing {target}: {cnt:,}**")
            else:
                lines.append("Ask: “How many employers are missing STD/LTD/Life?”")

        else:
            lines.append(
                "Try:\n"
                "- **Top 10 recommended actions**\n"
                "- **Top brokers by LTD white space**\n"
                "- **Top carriers by LTD white space**\n"
                "- **M&A roll-up targets**\n"
                "- **How many employers are missing LTD?**"
            )

        assistant = "\n\n".join(lines)
        st.session_state.chat.append({"role": "assistant", "content": assistant})
        st.rerun()


# ============================================================
# 10) Tabs (supporting evidence + drilldowns)
# ============================================================
tabs = st.tabs([
    "Executive Brief",
    "Opportunity Radar",
    "Lens Dashboard",
    "Gaps Matrix",
    "Data QA / Methods",
])


# ----------------------------
# Tab 1: Executive Brief
# ----------------------------
with tabs[0]:
    st.subheader("Executive Brief (Decision Engine Layer)")

    if len(broker_ai) > 0:
        top_b = broker_ai.iloc[0]
    else:
        top_b = None

    if len(carrier_ai) > 0:
        top_c = carrier_ai.iloc[0]
    else:
        top_c = None

    total_opp = broker_stats.get("total_opp_score", 0.0)
    total_wins = broker_stats.get("total_expected_wins", 0.0)
    portfolio_impact = broker_stats.get("portfolio_impact_base", 0.0)

    brief_lines = []
    brief_lines.append("**What we’re seeing**")
    brief_lines.append("- Fastest growth lever is **disability expansion (especially LTD)** where Life is already present.")
    brief_lines.append(f"- We normalize commissions to reduce outlier distortion (cap: **{cap_mode}**).")
    brief_lines.append("")
    brief_lines.append("**What it means**")
    brief_lines.append("- Priority = brokers with **high LTD gap density**, meaningful footprint, and manageable concentration risk.")
    brief_lines.append("- Secondary = carriers that are **Life-heavy but weak in LTD/STD** (white space).")
    brief_lines.append("")
    brief_lines.append("**What to do next (next 30 days)**")
    if top_b is not None:
        brief_lines.append(
            f"1) Start with **{top_b['Broker']}** (highest CEO score). "
            f"Estimated base upside: **{money(float(top_b['rev_impact_base']))}**."
        )
    brief_lines.append("2) Replicate playbook across **Top 10 brokers** by CEO score (scale the outreach engine).")
    if top_c is not None:
        brief_lines.append(
            f"3) For carrier strategy, prioritize **{top_c['Carrier']}** (high white space) for disability expansion thesis."
        )
    brief_lines.append("")
    brief_lines.append("**Scenario-based portfolio sizing**")
    brief_lines.append(f"- Total opportunity score: **{total_opp:,.0f}**")
    brief_lines.append(f"- Expected wins (score × conversion): **{total_wins:,.0f}**")
    brief_lines.append(f"- Estimated incremental commissions (base): **{money(portfolio_impact)}**")
    brief_lines.append(
        f"- Per-win commission proxy: **median_norm_comm × uplift = {money(median_comm_norm)} × {uplift_multiplier:.1f}**"
    )

    st.info("\n".join(brief_lines))

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("### Top 10 CEO Targets (Brokers)")
        st.dataframe(
            broker_ai.head(10)[[
                "Broker", "score_ceo", "employers", "opp_score", "opp_per_100_employers",
                "rev_impact_low", "rev_impact_base", "rev_impact_high", "hhi_concentration"
            ]],
            use_container_width=True
        )

    with right:
        st.markdown("### Top 10 White-space Carriers")
        st.dataframe(
            carrier_ai.head(10)[[
                "Carrier", "score_ceo", "total_emp", "total_lives",
                "Life_emp", "STD_emp", "LTD_emp",
                "imbalance_score", "product_concentration"
            ]],
            use_container_width=True
        )


# ----------------------------
# Tab 2: Opportunity Radar
# ----------------------------
with tabs[1]:
    st.subheader("Opportunity Radar (Signals)")

    total_employers = int(gaps_f["Employer"].nunique())
    missing_std = int((gaps_f["STD"] == 0).sum())
    missing_ltd = int((gaps_f["LTD"] == 0).sum())
    missing_life = int((gaps_f["Life"] == 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employers missing STD", f"{missing_std:,}", f"{(missing_std / max(total_employers,1)):.1%}")
    c2.metric("Employers missing LTD", f"{missing_ltd:,}", f"{(missing_ltd / max(total_employers,1)):.1%}")
    c3.metric("Employers missing Life", f"{missing_life:,}", f"{(missing_life / max(total_employers,1)):.1%}")
    c4.metric("Median commission (norm)", money(median_comm_norm))

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("### Signal: Top brokers by estimated revenue impact")
        tbl = broker_ai.sort_values("rev_impact_base", ascending=False).head(top_n)
        st.dataframe(tbl[[
            "Broker", "employers", "opp_score", "expected_wins",
            "rev_impact_low", "rev_impact_base", "rev_impact_high", "hhi_concentration"
        ]], use_container_width=True)

    with right:
        st.markdown("### Signal: Best efficiency (high rate, low concentration)")
        tbl2 = broker_ai.sort_values(["opp_per_100_employers", "hhi_concentration"], ascending=[False, True]).head(top_n)
        st.dataframe(tbl2[[
            "Broker", "employers", "opp_per_100_employers", "opp_score",
            "rev_impact_base", "hhi_concentration"
        ]], use_container_width=True)

    st.divider()
    left2, right2 = st.columns(2)

    with left2:
        st.markdown("### Signal: Top carriers by LTD white space")
        tbl3 = carrier_ai.sort_values("whitespace_ltd", ascending=False).head(top_n)
        st.dataframe(tbl3[[
            "Carrier", "total_emp", "Life_emp", "LTD_emp", "whitespace_ltd", "imbalance_score"
        ]], use_container_width=True)

    with right2:
        st.markdown(f"### Signal: Top carriers by covered lives ({product_view})")
        csum = (
            epc_p.groupby("Carrier", as_index=False)
            .agg(covered_lives=("Covered_Lives", "sum"), employers=("Employer", "nunique"))
            .sort_values("covered_lives", ascending=False)
            .head(top_n)
        )
        st.dataframe(csum, use_container_width=True)


# ----------------------------
# Tab 3: Lens Dashboard
# ----------------------------
with tabs[2]:
    if lens == "Broker Lens":
        st.subheader("Broker Lens — Cross-sell & Growth Engine")

        left, right = st.columns(2)

        with left:
            st.markdown("### Top brokers by CEO score")
            view = broker_ai.head(top_n).copy()
            fig = px.bar(view, x="Broker", y="score_ceo", title=f"Top {top_n} Brokers by CEO Score")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(view[[
                "Broker", "score_ceo", "employers", "opp_score", "opp_per_100_employers",
                "rev_impact_base", "hhi_concentration"
            ]], use_container_width=True)

        with right:
            st.markdown("### Top 10 recommended actions (consultant-style)")
            top10 = broker_ai.head(10)
            for i, r in enumerate(top10.itertuples(index=False), start=1):
                st.markdown(
                    f"**{i}. Target {r.Broker}** — "
                    f"{int(r.employers):,} employers • "
                    f"Opp **{float(r.opp_score):,.1f}** • "
                    f"Impact **{money(float(r.rev_impact_low))}–{money(float(r.rev_impact_high))}** "
                    f"(base {money(float(r.rev_impact_base))})"
                )
                st.caption("Play: lead with LTD (highest LTV), bundle STD, then align carrier appetite + underwriting.")

        st.divider()
        st.markdown("## Broker Drilldown")
        sel = st.selectbox("Select a broker", ["(Select)"] + broker_ai["Broker"].head(5000).tolist())

        if sel != "(Select)":
            bb = broker_book[broker_book["Broker"] == sel].copy()

            bb["opp_life_no_ltd"] = ((bb["Life"] == 1) & (bb["LTD"] == 0)).astype(int)
            bb["opp_life_no_std"] = ((bb["Life"] == 1) & (bb["STD"] == 0)).astype(int)
            bb["opp_std_no_ltd"] = ((bb["STD"] == 1) & (bb["LTD"] == 0)).astype(int)
            bb["opportunity_score"] = (
                bb["opp_life_no_ltd"] * w_life_no_ltd
                + bb["opp_life_no_std"] * w_life_no_std
                + bb["opp_std_no_ltd"] * w_std_no_ltd
            )

            st.write(f"**{sel}** — employers: **{bb['Employer'].nunique():,}** • median comm (norm): **{money(bb['comm_norm'].median())}**")

            show = bb.sort_values(["opportunity_score", "comm_norm"], ascending=[False, False]).head(100)[
                ["Employer", "Life", "STD", "LTD", "opportunity_score", "comm_norm"]
            ]
            st.dataframe(show, use_container_width=True, height=380)

    elif lens == "Carrier Lens":
        st.subheader("Carrier Lens — White Space & Product Strategy")

        left, right = st.columns(2)

        with left:
            st.markdown("### Top carriers by CEO score")
            view = carrier_ai.head(top_n)
            fig = px.bar(view, x="Carrier", y="score_ceo", title=f"Top {top_n} Carriers by CEO Score")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(view[[
                "Carrier", "score_ceo", "total_emp", "total_lives",
                "imbalance_score", "product_concentration"
            ]], use_container_width=True)

        with right:
            st.markdown("### Top carriers by LTD white space")
            view2 = carrier_ai.sort_values("whitespace_ltd", ascending=False).head(top_n)
            fig2 = px.bar(view2, x="Carrier", y="whitespace_ltd", title=f"Top {top_n} Carriers by LTD White Space")
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(view2[[
                "Carrier", "total_emp", "Life_emp", "STD_emp", "LTD_emp",
                "life_share", "std_share", "ltd_share", "whitespace_ltd"
            ]], use_container_width=True)

        st.divider()
        st.markdown("## Carrier Drilldown")
        sel = st.selectbox("Select a carrier", ["(Select)"] + carrier_ai["Carrier"].head(5000).tolist())
        if sel != "(Select)":
            row = carrier_ai[carrier_ai["Carrier"] == sel].iloc[0]
            st.write(f"**{sel}** — employers: **{int(row.total_emp):,}** • lives: **{int(row.total_lives):,}**")
            st.write(
                f"Shares — Life **{row.life_share:.1%}**, STD **{row.std_share:.1%}**, LTD **{row.ltd_share:.1%}**"
            )

            tmp = epc_f[(epc_f["Carrier"] == sel) & (epc_f["Product"] == product_view)].copy()
            sample = (
                tmp.groupby("Employer", as_index=False)
                .agg(covered_lives=("Covered_Lives", "sum"))
                .sort_values("covered_lives", ascending=False)
                .head(100)
            )
            st.markdown(f"### Sample employers (by covered lives) — {product_view}")
            st.dataframe(sample, use_container_width=True, height=360)

    else:
        st.subheader("M&A Radar — Roll-up + Partnership Targets (Heuristic)")

        st.markdown("### Broker roll-up targets (size + diversification + growth headroom)")
        tmp = broker_ai.copy()
        tmp["ma_score"] = (
            zscore(tmp["comm_norm_sum"]) * 0.40
            + zscore(tmp["employers"]) * 0.25
            + zscore(tmp["opp_score"]) * 0.35
            - zscore(tmp["hhi_concentration"]) * 0.40
        )
        st.dataframe(
            tmp.sort_values("ma_score", ascending=False).head(top_n)[[
                "Broker", "employers", "comm_norm_sum", "opp_score", "hhi_concentration", "ma_score"
            ]],
            use_container_width=True
        )

        st.markdown("### Carrier targets (footprint + imbalance thesis)")
        tmpc = carrier_ai.copy()
        tmpc["ma_score"] = (
            zscore(tmpc["total_emp"]) * 0.35
            + zscore(tmpc["total_lives"]) * 0.20
            + zscore(tmpc["imbalance_score"]) * 0.45
            - zscore(tmpc["product_concentration"]) * 0.35
        )
        st.dataframe(
            tmpc.sort_values("ma_score", ascending=False).head(top_n)[[
                "Carrier", "total_emp", "total_lives", "imbalance_score", "product_concentration", "ma_score"
            ]],
            use_container_width=True
        )

        st.caption("Pilot note: these are prioritization heuristics—not diligence. Validate premium, margin, retention, and transferability.")


# ----------------------------
# Tab 4: Gaps Matrix
# ----------------------------
with tabs[3]:
    st.subheader("Gaps Matrix (Employer Product Coverage)")

    max_rows = st.slider("Max rows to display", 25, 500, 200, 25)
    g = gaps_f.head(max_rows).copy()

    def color_cell(v):
        try:
            v = int(v)
        except Exception:
            v = 0
        return "background-color: #d4f8d4; font-weight: 600;" if v == 1 else "background-color: #ffd6d6; font-weight: 600;"

    styled = g.style.applymap(color_cell, subset=["Life", "STD", "LTD"])
    st.caption("Green = covered, Red = gap (missing product).")
    st.dataframe(styled, use_container_width=True, height=450)


# ----------------------------
# Tab 5: Data QA / Methods
# ----------------------------
with tabs[4]:
    st.subheader("Data QA / Methods (Assumptions & Calculations)")

    st.markdown("### Commission sanity checks")
    st.write(f"- Median (raw): **{money(median_comm_raw)}**")
    st.write(f"- Median (normalized): **{money(median_comm_norm)}**")
    st.write(f"- 95th percentile: **{money(q95)}**")
    st.write(f"- 99th percentile: **{money(q99)}**")
    st.write(f"- Max (raw): **{money(max_comm_raw)}**")
    st.caption(
        "We cap commissions (default 99th percentile) to reduce extreme outlier distortion. "
        "We do not use raw national totals as a headline KPI."
    )

    st.markdown("### Opportunity scoring (weighted gaps)")
    st.write("- Gap definitions (per employer):")
    st.write("  - Life but missing LTD")
    st.write("  - Life but missing STD")
    st.write("  - STD but missing LTD")
    st.write("- Weights (adjustable):")
    st.write(f"  - Life missing LTD: **{w_life_no_ltd:.1f}**")
    st.write(f"  - Life missing STD: **{w_life_no_std:.1f}**")
    st.write(f"  - STD missing LTD: **{w_std_no_ltd:.1f}**")
    st.caption("CEO logic: LTD is often higher LTV / stickier, so it defaults to the highest weight.")

    st.markdown("### Estimated revenue impact (scenario-based)")
    st.write(f"- Conversion rate: **{conversion_rate:.0%}**")
    st.write(f"- Per-win commission proxy: **median_norm_comm × uplift = {money(median_comm_norm)} × {uplift_multiplier:.1f} = {money(broker_stats.get('per_win_commission_proxy', 0.0))}**")
    st.write("- Expected wins = opp_score × conversion_rate")
    st.write("- Impact (base) = expected_wins × per_win_commission_proxy")
    st.write("- Low/high bands = base × 0.6 / 1.4 (pilot uncertainty band)")
    st.caption("This is a prioritization estimate—not GAAP revenue.")

    st.markdown("### Risk & concentration")
    st.write("- Broker concentration uses HHI over employer commission shares within broker.")
    st.write(f"- Concentration penalty strength: **{concentration_penalty:.1f}**")
    st.caption("Higher HHI = more reliance on a small number of accounts (risk flag).")

    st.markdown("### Row counts")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows: EPC", f"{len(epc_f):,}")
    c2.metric("Rows: EBC", f"{len(ebc_f):,}")
    c3.metric("Rows: GAPS", f"{len(gaps_f):,}")

    st.markdown("### Samples (first 10 rows)")
    st.write("EBC sample:")
    st.dataframe(ebc_f.head(10), use_container_width=True)
    st.write("EPC sample (filtered by product_view):")
    st.dataframe(epc_p.head(10), use_container_width=True)