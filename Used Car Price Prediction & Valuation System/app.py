import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime

# ── Page Configuration ──
st.set_page_config(
    page_title="Used Car Valuation System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global Theme & CSS Styling ──
st.markdown("""
    <style>
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2a5298;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .insight-card {
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

# ── API Connection & Health Check ──
api_online = False
try:
    r_health = requests.get(f"{API_URL}/health", timeout=2)
    if r_health.status_code == 200 and r_health.json().get("status") == "healthy":
        api_online = True
except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
    pass

# ── Sidebar Layout & Navigation ──
st.sidebar.markdown("<h2 style='text-align: center;'>🚗 Navigation</h2>", unsafe_allow_html=True)

# API Status Widget
if api_online:
    st.sidebar.success("🟢 API Server: Connected")
else:
    st.sidebar.error("🔴 API Server: Offline")
    st.sidebar.warning("⚠️ Please start the FastAPI backend server to run valuations.")

page = st.sidebar.radio(
    "Go To:",
    ["🚗 Valuation Calculator", "📋 Valuation Log History", "💡 Valuation Economics Guide"]
)

# ── Initialize Session Log State ──
if "history" not in st.session_state:
    st.session_state.history = []

# ── Load Preprocessor Artifacts ──
artifacts = None
if os.path.exists('preprocessor_artifacts.pkl'):
    try:
        with open('preprocessor_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading preprocessor_artifacts.pkl: {e}")
else:
    st.error("⚠️ Preprocessor artifacts file 'preprocessor_artifacts.pkl' not found. Please train the model and save artifacts first!")

# ── PAGE 1: VALUATION CALCULATOR ──
if page == "🚗 Valuation Calculator":
    st.markdown("<h1 class='main-title'>Used Car Price Valuation Calculator</h1>", unsafe_allow_html=True)
    st.write("Enter raw vehicle characteristics below to retrieve real-time market value estimation.")
    st.markdown("---")

    if artifacts is None:
        st.stop()

    # dynamic model filtering based on selected brand
    brand_models = artifacts.get('brand_models', {})
    brands_avail = sorted(list(brand_models.keys()))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏷️ Identity & Specs")
        oem_sel = st.selectbox("Brand (OEM)", [b.title() for b in brands_avail])
        oem_clean = oem_sel.lower().strip()

        # Filter models based on selected brand
        models_avail = brand_models.get(oem_clean, [])
        model_sel = st.selectbox("Model", [m.title() for m in models_avail], key=f"model_{oem_clean}")
        model_clean = model_sel.lower().strip()

        myear = st.slider("Manufacturing Year", min_value=2000, max_value=2026, value=2018)
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    with col2:
        st.subheader("📈 Usage & Power")
        km = st.number_input("Kilometers Driven (km)", min_value=0.0, max_value=500000.0, value=50000.0, step=5000.0)
        owner = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth", "Unregistered Car"])
        engine_cc = st.number_input("Engine capacity (cc)", min_value=500.0, max_value=6000.0, value=1197.0, step=100.0)
        power = st.number_input("Max Power (BHP)", min_value=30.0, max_value=600.0, value=83.0, step=5.0)
        torque = st.number_input("Max Torque (Nm)", min_value=30.0, max_value=900.0, value=113.0, step=10.0)

    with col3:
        st.subheader("📏 Weight & Dimensions")
        kerb_weight = st.number_input("Kerb Weight (kg)", min_value=500.0, max_value=3500.0, value=960.0, step=50.0)
        length = st.number_input("Length (mm)", min_value=2000.0, max_value=6000.0, value=3840.0, step=50.0)
        width = st.number_input("Width (mm)", min_value=1000.0, max_value=2500.0, value=1735.0, step=10.0)
        height = st.number_input("Height (mm)", min_value=1000.0, max_value=2500.0, value=1530.0, step=10.0)
        cylinders = st.number_input("Number of Cylinders", min_value=2, max_value=12, value=4, step=1)

    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.button("🔍 Predict Price", use_container_width=True)

    if submit_btn:
        if not api_online:
            st.error("🔴 Server is offline! Cannot execute valuation. Please start the FastAPI backend.")
        else:
            payload = {
                "oem": oem_clean,
                "model": model_clean,
                "fuel": fuel,
                "transmission": transmission,
                "owner_type": owner,
                "myear": int(myear),
                "km": float(km),
                "engine_cc": float(engine_cc),
                "max_power_bhp": float(power),
                "max_torque_nm": float(torque),
                "kerb_weight": float(kerb_weight),
                "length": float(length),
                "width": float(width),
                "height": float(height),
                "no_of_cylinder": int(cylinders)
            }

            with st.spinner("⌛ Communicating with backend Valuation API and running model..."):
                try:
                    r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                    if r.status_code == 200:
                        res = r.json()
                        pred_price = res["predicted_price"]
                        range_str = res["price_range"]

                        # Determine pricing category based on price thresholds
                        if pred_price < 400000:
                            category = "Budget (Entry-Level)"
                            cat_color = "#27ae60"
                        elif pred_price < 1000000:
                            category = "Mid-Range (Mass-Market)"
                            cat_color = "#2980b9"
                        elif pred_price < 2500000:
                            category = "Premium Segment"
                            cat_color = "#8e44ad"
                        else:
                            category = "Luxury Segment"
                            cat_color = "#d35400"

                        st.balloons()
                        st.success("🎉 Valuation successfully computed!")

                        # ── Results Render ──
                        res_col1, res_col2 = st.columns([1, 1])
                        with res_col1:
                            st.markdown(f"""
                                <div class='metric-container'>
                                    <p style='margin: 0; color: #555; font-size: 14px; text-transform: uppercase; font-weight: bold;'>Estimated Market Value</p>
                                    <h1 style='margin: 5px 0 0 0; color: #1e3c72; font-size: 42px;'>₹{pred_price:,.0f}</h1>
                                </div>
                            """, unsafe_allow_html=True)

                        with res_col2:
                            st.markdown(f"""
                                <div class='metric-container' style='border-left-color: {cat_color};'>
                                    <p style='margin: 0; color: #555; font-size: 14px; text-transform: uppercase; font-weight: bold;'>Price Segment</p>
                                    <h2 style='margin: 5px 0 0 0; color: {cat_color}; font-size: 26px;'>{category}</h2>
                                    <p style='margin: 5px 0 0 0; font-weight: bold; font-size: 16px; color: #333;'>Expected Range: {range_str}</p>
                                </div>
                            """, unsafe_allow_html=True)

                        # ── Valuation Insights ──
                        st.markdown("<br><h3>💡 AI Valuation Insights</h3>", unsafe_allow_html=True)

                        car_age = 2026 - myear
                        LUXURY_BRANDS = ['bmw', 'audi', 'mercedes-benz', 'jaguar', 'volvo', 'land rover', 'porsche', 'bentley', 'rolls-royce']
                        PREMIUM_BRANDS = ['honda', 'toyota', 'hyundai', 'volkswagen', 'skoda', 'kia']

                        # Age Insight
                        if car_age <= 5:
                            st.success(f"✓ **Low vehicle age ({car_age} years)** increases resale value. Minimal depreciation occurred.")
                        elif car_age > 10:
                            st.warning(f"⚠️ **High vehicle age ({car_age} years)** introduces steep depreciation penalties.")

                        # Mileage Insight
                        if km <= 50000:
                            st.success(f"✓ **Low mileage ({km:,.0f} km)** indicates lower mechanical wear and tear, boosting value.")
                        elif km > 120000:
                            st.warning(f"⚠️ **High mileage ({km:,.0f} km)** suggests heavy usage, leading to value discounts.")

                        # Brand Flag Insight
                        if oem_clean in LUXURY_BRANDS:
                            st.info(f"✓ **Luxury Brand equity ({oem_sel})** commands a significant market premium.")
                        elif oem_clean in PREMIUM_BRANDS:
                            st.info(f"✓ **Premium brand positioning ({oem_sel})** contributes positively to valuation.")

                        # Performance Insight
                        if power >= 120:
                            st.success(f"✓ **High power output ({power:.0f} BHP)** indicates performance engine, driving demand.")

                        # Add request to Log History
                        st.session_state.history.append({
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Brand": oem_sel,
                            "Model": model_sel,
                            "Year": myear,
                            "Km Driven": km,
                            "Fuel": fuel,
                            "Transmission": transmission,
                            "Predicted Price": f"₹{pred_price:,.0f}",
                            "Range": range_str,
                            "Category": category
                        })
                    else:
                        st.error(f"❌ Server returned an error: {r.status_code} - {r.text}")
                except Exception as e:
                    st.error(f"❌ Network request failed: {e}")

# ── PAGE 2: VALUATION LOG HISTORY ──
elif page == "📋 Valuation Log History":
    st.markdown("<h1 class='main-title'>Valuation Log History</h1>", unsafe_allow_html=True)
    st.write("View and export predictions executed during the current session.")
    st.markdown("---")

    if not st.session_state.history:
        st.info("ℹ️ No valuations computed yet. Go back to the calculator and perform a prediction!")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)

        # CSV Export
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction Report as CSV",
            data=csv,
            file_name="used_car_valuation_report.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("🗑️ Clear Log History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ── PAGE 3: VALUATION ECONOMICS GUIDE ──
elif page == "💡 Valuation Economics Guide":
    st.markdown("<h1 class='main-title'>Used Car Valuation Economics</h1>", unsafe_allow_html=True)
    st.markdown("""
    This section outlines how used car prices are determined in the used car market and how our **Tuned XGBoost Regressor** models them.

    ### Major Valuation Depreciation Drivers
    1. **Time-Based Depreciation (`car_age`)**
       - Vehicles lose a significant portion of their value the moment they leave the showroom.
       - Standard passenger cars experience the steepest depreciation curve in the first 3-5 years (losing up to 40-50%). after 8-10 years, the depreciation curve flattens out towards a salvage value floor.
    2. **Usage wear (`km` / `KM_PER_YEAR`)**
       - Average annual wear is approximately 10,000–12,000 km. Vehicle mileage scales directly with parts wear (brakes, engine rings, suspension).
       - High mileage driven over a short period of time is heavily penalized by the valuation model.
    3. **Brand Positioning & Equity**
       - Brands classified as **Luxury** segment (e.g. BMW, Audi, Mercedes-Benz) command substantial premiums, but their absolute depreciation rate in terms of rupees is much higher than budget models.
       - **Premium** brands (e.g., Honda, Toyota) hold their value extremely well due to perceived reliability and high availability of spare parts.
    4. **Performance & Volume**
       - **Bigger engines (Engine CC)**, **Number of Cylinders**, and **Max Power (BHP)** increase vehicle pricing. Larger physical sizes (measured through **Car Volume**) represent bigger utility segments (SUVs, premium sedans) which sell for higher initial prices.
    """)
    st.info("💡 Tip: Low vehicle age + Low ownership changes + Low KM driven creates the highest valuation premium in the resale market.")
