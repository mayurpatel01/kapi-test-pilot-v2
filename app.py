import streamlit as st
import pandas as pd
import plotly.express as px

import streamlit as st

def require_password():
    pw = st.secrets.get("APP_PASSWORD", "")
    if not pw:
        st.warning("APP_PASSWORD not set in Streamlit Secrets.")
        st.stop()

    if "authed" not in st.session_state:
        st.session_state.authed = False

    if not st.session_state.authed:
        st.title("Kapi Test Pilot")
        entered = st.text_input("Enter password", type="password")
        if st.button("Login"):
            if entered == pw:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("Wrong password")
        st.stop()

require_password()

st.set_page_config(page_title="Kapi Test Pilot", layout="wide")

st.title("Kapi Test Pilot — Form 5500 Pilot Dashboard")
st.caption("Starter app (sample data). Next we will wire in Form 5500 aggregated marts.")

# Sample pilot data (we'll replace with real Form 5500 marts later)
df = pd.DataFrame({
    "Product": ["Life", "Life", "STD", "STD", "LTD", "LTD"],
    "Broker": ["Broker A", "Broker B", "Broker A", "Broker C", "Broker B", "Broker C"],
    "Unique Employers": [120, 80, 60, 40, 25, 35],
    "Commissions Paid": [250000, 150000, 110000, 90000, 70000, 65000],
    "Carrier": ["Carrier X", "Carrier Y", "Carrier X", "Carrier Z", "Carrier Y", "Carrier Z"]
})

col1, col2, col3 = st.columns([1,1,1])
product = col1.selectbox("Product", ["Life", "STD", "LTD"])
broker = col2.selectbox("Broker (optional)", ["All"] + sorted(df["Broker"].unique().tolist()))
carrier = col3.selectbox("Carrier (optional)", ["All"] + sorted(df["Carrier"].unique().tolist()))

filtered = df[df["Product"] == product]
if broker != "All":
    filtered = filtered[filtered["Broker"] == broker]
if carrier != "All":
    filtered = filtered[filtered["Carrier"] == carrier]

st.subheader("Broker / Employers")
fig1 = px.bar(filtered, x="Broker", y="Unique Employers", title="Unique Employers by Broker")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Broker / Commissions Paid")
fig2 = px.bar(filtered, x="Broker", y="Commissions Paid", title="Commissions Paid by Broker")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Table")
st.dataframe(filtered, use_container_width=True)