import streamlit as st
import pandas as pd
import numpy as np
import os
import streamlit.components.v1 as components
from PIL import Image

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & CACHED DATA LOADING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Maharashtra Bio-Ethanol Decision Support System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

ASSET_DIR = "assets"

@st.cache_data
def load_data():
    data = {}
    files = {
        "dispatch": "Optimal_Bagasse_Dispatch_Schedule.csv",
        "pareto": "Pareto_Frontier_Scenarios.csv",
        "audit": "Regional_Plant_Audit_Kolhapur_Sangli.csv",
        "summary": "Model_Performance_Summary.csv",
        "predictions": "Pooled_Holdout_Predictions.csv"
    }
    for key, filename in files.items():
        path = os.path.join(ASSET_DIR, filename)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = pd.DataFrame()
    return data

data = load_data()
df_dispatch, df_pareto, df_audit, df_summary, df_predictions = data["dispatch"], data["pareto"], data["audit"], data["summary"], data["predictions"]

# ---------------------------------------------------------
# 2. SIDEBAR CONFIGURATION
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/biofuel.png", width=70)
st.sidebar.title("Bio-Ethanol DSS")
st.sidebar.markdown("**State-Wide Industrial Decision Support System**  \n*Physics-Informed ML & Multi-Objective Spatial Optimization*")
st.sidebar.divider()

navigation = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏛️ Executive Dashboard",
        "🔬 ML Model Diagnostics & XAI",
        "🗺️ Geospatial Network Explorer",
        "🚚 Dispatch & Logistics Matrix",
        "⚙️ Physics-Informed What-If Simulator"
    ]
)
st.sidebar.divider()
st.sidebar.caption("Data Source: Maharashtra Sugar Commissionerate & Industry Baselines")

# ---------------------------------------------------------
# TAB 1: EXECUTIVE DASHBOARD
# ---------------------------------------------------------
if navigation == "🏛️ Executive Dashboard":
    st.title("🌾 Maharashtra Sugar & Bio-Ethanol Integrated Platform")
    st.markdown("End-to-end framework coupling physics-constrained machine learning throughput forecasting with mathematical supply chain optimization.")

    col1, col2, col3, col4 = st.columns(4)
    total_cane = df_predictions["Pred_Cane_MT"].sum() if not df_predictions.empty else 0
    total_bagasse = df_predictions["Pred_Bagasse_MT"].sum() if not df_predictions.empty else 0
    total_surplus = df_predictions["Pred_Surplus_Bagasse_MT"].sum() if not df_predictions.empty else 0
    total_sugar = df_predictions["Pred_Sugar_Qtl"].sum() if not df_predictions.empty else 0

    col1.metric("Forecasted Cane Crushing", f"{total_cane/1e6:.2f} M MT", "Pooled Seasons")
    col2.metric("Total Bagasse (28%)", f"{total_bagasse/1e6:.2f} M MT", "Mass Conserved")
    col3.metric("Surplus Bagasse for 2G", f"{total_surplus/1e3:.1f} k MT", "8% Net Surplus")
    col4.metric("Total Sugar Output", f"{total_sugar/1e6:.2f} M Qtl", "Bounded Recovery")

    st.divider()
    st.subheader("📊 Pipeline vs. Decoupled ML Baseline Performance")
    if not df_summary.empty:
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

    st.subheader("📈 System Architecture")
    st.markdown("""
    * **Physics Layer 1 (Cane Crushing):** HistGradientBoosting with Poisson loss objective capturing non-negative, right-skewed throughput.
    * **Physics Layer 2 (Sucrose Yield):** L1-regularized chemical recovery regression agronomically bounded to [7.5%, 13.5%].
    * **Deterministic Mass-Balance:** 100% mass conservation multiplier (0.280000) generating downstream metrics without variance drift.
    * **Spatial Logistics Layer:** PuLP mixed-integer optimization minimizing haul distance across 63 distilleries subject to 100 km economic radii.
    """)

# ---------------------------------------------------------
# TAB 2: ML MODEL DIAGNOSTICS & XAI
# ---------------------------------------------------------
elif navigation == "🔬 ML Model Diagnostics & XAI":
    st.title("🔬 Physics-Informed ML Diagnostics & Explainability")
    diag_view = st.selectbox("Select Diagnostic Suite", ["4-Target Master Pipeline Suite", "Cane Crushed Throughput (Poisson)", "Sucrose Recovery Chemistry (Bounded)"])

    img_map = {
        "4-Target Master Pipeline Suite": "Master_Pipeline_Diagnostics.png",
        "Cane Crushed Throughput (Poisson)": "Model_Diagnostics_Dashboard.png",
        "Sucrose Recovery Chemistry (Bounded)": "Recovery_Diagnostics_Dashboard.png"
    }
    
    img_path = os.path.join(ASSET_DIR, img_map[diag_view])
    if os.path.exists(img_path):
        st.image(Image.open(img_path), use_container_width=True)
    else:
        st.warning(f"Image not found at {img_path}")

    st.divider()
    st.subheader("📋 Mill-Level Holdout Validation Table")
    if not df_predictions.empty:
        st.dataframe(df_predictions, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: GEOSPATIAL NETWORK EXPLORER
# ---------------------------------------------------------
elif navigation == "🗺️ Geospatial Network Explorer":
    st.title("🗺️ Geospatial Supply Chain & Network Topology")
    map_choice = st.radio("Select Network Scope:", ["State-Wide (All 63 Ethanol Plants)", "Regional Cluster (Kolhapur & Sangli Catchment)"], horizontal=True)

    if map_choice == "State-Wide (All 63 Ethanol Plants)":
        map_path = os.path.join(ASSET_DIR, "All_Ethanol_Plants_Supply_Network.html")
        if os.path.exists(map_path):
            with open(map_path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=650, scrolling=True)
    else:
        col_map, col_stat = st.columns([1.2, 0.8])
        with col_map:
            map_path = os.path.join(ASSET_DIR, "Kolhapur_Sangli_Ethanol_Supply_Network.html")
            if os.path.exists(map_path):
                with open(map_path, 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=600, scrolling=True)
        with col_stat:
            st.subheader("Regional Plant Utilization Audit")
            if not df_audit.empty:
                st.dataframe(df_audit[["Distillery Plant", "Capacity (KLPD)", "Capacity Met (%)", "Avg Haul Distance (km)"]], height=550, hide_index=True)

        st.divider()
        st.subheader("Regional Catchment & Logistics Analytics")
        img_path = os.path.join(ASSET_DIR, "Kolhapur_Sangli_Comprehensive_Logistics_Suite.png")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), use_container_width=True)

# ---------------------------------------------------------
# TAB 4: DISPATCH & LOGISTICS MATRIX
# ---------------------------------------------------------
elif navigation == "🚚 Dispatch & Logistics Matrix":
    st.title("🚚 Optimized Biomass Dispatch & Pareto Frontier")
    if not df_pareto.empty:
        st.subheader("⚡ Pareto Optimal Trade-Off Scenarios (Volume vs. Haul Distance)")
        st.dataframe(df_pareto, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔍 Mill & Plant Routing Explorer")
    if not df_dispatch.empty:
        col_f1, col_f2 = st.columns(2)
        all_mills = ["All Mills"] + sorted(df_dispatch["Sugar Mill Source"].unique().tolist())
        all_plants = ["All Plants"] + sorted(df_dispatch["Ethanol Plant Destination"].unique().tolist())

        selected_mill = col_f1.selectbox("Filter by Sugar Mill Source:", all_mills)
        selected_plant = col_f2.selectbox("Filter by Ethanol Plant Destination:", all_plants)

        df_filtered = df_dispatch.copy()
        if selected_mill != "All Mills":
            df_filtered = df_filtered[df_filtered["Sugar Mill Source"] == selected_mill]
        if selected_plant != "All Plants":
            df_filtered = df_filtered[df_filtered["Ethanol Plant Destination"] == selected_plant]

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Active Routes", len(df_filtered))
        col_m2.metric("Total Bagasse Allocated", f"{df_filtered['Bagasse Allocated (MT)'].sum():,.2f} MT")
        col_m3.metric("Total Logistical Work", f"{df_filtered['Ton-Kilometers'].sum():,.0f} Ton-KM")

        st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True)
        st.download_button("📥 Download Filtered Dispatch Schedule (CSV)", df_filtered.to_csv(index=False).encode('utf-8'), "Selected_Bagasse_Dispatch_Schedule.csv", "text/csv")

# ---------------------------------------------------------
# TAB 5: WHAT-IF SIMULATOR
# ---------------------------------------------------------
elif navigation == "⚙️ Physics-Informed What-If Simulator":
    st.title("⚙️ Interactive Bio-Industrial Simulation Engine")
    col_inp, col_out = st.columns([1, 1])

    with col_inp:
        st.subheader("Operational Parameters")
        sim_capacity = st.slider("Mill Crushing Capacity (TCD/Day)", 1000, 15000, 5000, step=250)
        sim_days = st.slider("Total Operational Days", 60, 200, 130, step=5)
        sim_util = st.slider("Capacity Utilization Factor", 0.70, 1.10, 0.95, step=0.05)
        sim_rec = st.slider("Sucrose Recovery Rate (%)", 7.5, 13.5, 10.5, step=0.1)
        sim_surplus_bagasse_pct = st.slider("Surplus Bagasse Allocated to 2G (%)", 4.0, 15.0, 8.0, step=0.5) / 100.0
        sim_surplus_sugar_pct = st.slider("Surplus Sugar Policy Allocation (%)", 20.0, 40.0, 32.0, step=1.0) / 100.0

    with col_out:
        st.subheader("Physical Yield Projections")
        sim_cane = sim_capacity * sim_days * sim_util
        sim_total_bagasse = sim_cane * 0.28
        sim_surplus_bagasse = sim_total_bagasse * sim_surplus_bagasse_pct
        sim_total_sugar = sim_cane * (sim_rec / 10.0)

        st.metric("Estimated Cane Crushed", f"{sim_cane:,.0f} MT")
        st.metric("Total Bagasse Generated (28%)", f"{sim_total_bagasse:,.0f} MT")
        st.metric("Net Surplus Bagasse for 2G", f"{sim_surplus_bagasse:,.0f} MT", delta=f"{(sim_surplus_bagasse * 200.0)/1e6:.2f}M Liters Ethanol")
        st.metric("Total Sugar Production", f"{sim_total_sugar:,.0f} Qtl")
        st.metric("Surplus Commercial Sugar", f"{sim_total_sugar * sim_surplus_sugar_pct:,.0f} Qtl")