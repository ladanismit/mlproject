import streamlit as st
import requests
import json

# --- Section 10: Dashboard Styling ---
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    layout="wide"
)

# FastAPI Endpoint settings
API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"

# --- Section 1: Introduction ---
st.title("🏦 Loan Default Risk Prediction System")
st.markdown("Predict whether a loan applicant is likely to default using a trained XGBoost model.")
st.markdown("**Model:** XGBoost | **Version:** 1.0.0")

# --- Section 8: API Health Check ---
st.sidebar.header("System Status")
try:
    health_response = requests.get(HEALTH_URL, timeout=2)
    if health_response.status_code == 200:
        st.sidebar.success("API Status: Healthy")
    else:
        st.sidebar.error("API Status: Offline")
except requests.exceptions.RequestException:
    st.sidebar.error("API Status: Offline")

# --- Section 2: Applicant Information Form ---
st.header("Applicant Information")

with st.form("prediction_form"):
    st.subheader("Loan Information")
    col1, col2 = st.columns(2)
    with col1:
        income_total = st.number_input("Income Total", min_value=0.0, value=150000.0, step=1000.0)
        credit_amount = st.number_input("Credit Amount", min_value=0.0, value=500000.0, step=1000.0)
    with col2:
        annuity_amount = st.number_input("Annuity Amount", min_value=0.0, value=25000.0, step=500.0)
        goods_price = st.number_input("Goods Price", min_value=0.0, value=450000.0, step=1000.0)

    st.subheader("Personal Information")
    col3, col4 = st.columns(2)
    with col3:
        gender = st.selectbox("Gender", ["F", "M", "XNA"])
        own_car = st.selectbox("Own Car", ["Y", "N"])
        children_count = st.number_input("Children Count", min_value=0, max_value=20, value=0, step=1)
        own_realty = st.selectbox("Own Realty", ["Y", "N"])
    with col4:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        years_employed = st.number_input("Years Employed", min_value=0.0, max_value=60.0, value=5.0, step=0.5)

    st.subheader("Contract Information")
    col5, col6, col7 = st.columns(3)
    with col5:
        contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
        income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"])
        family_status = st.selectbox("Family Status", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow", "Unknown"])
    with col6:
        education_type = st.selectbox("Education Type", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
        housing_type = st.selectbox("Housing Type", ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"])
        occupation_type = st.selectbox("Occupation Type", ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff", "Accountants", "Medicine staff", "Security staff", "Cooking staff", "Cleaning staff", "Private service staff", "Low-skill Laborers", "Waiters/barmen staff", "Secretaries", "Realty agents", "HR staff", "IT staff"])
    with col7:
        name_type_suite = st.selectbox("Name Type Suite", ["Unaccompanied", "Family", "Spouse, partner", "Children", "Other_B", "Other_A", "Group of people"])
        organization_type = st.selectbox("Organization Type", ["Business Entity Type 3", "XNA", "Self-employed", "Other", "Medicine", "Business Entity Type 2", "Government", "School", "Trade: type 7", "Kindergarten", "Construction", "Business Entity Type 1", "Transport: type 4", "Trade: type 3", "Industry: type 9", "Industry: type 3", "Security", "Housing", "Industry: type 11", "Military", "Bank", "Agriculture", "Police", "Transport: type 2", "Postal", "Security Ministries", "Trade: type 2", "Restaurant", "Services", "University", "Industry: type 7", "Transport: type 3", "Industry: type 1", "Hotel", "Electricity", "Industry: type 4", "Trade: type 6", "Industry: type 5", "Insurance", "Telecom", "Emergency", "Industry: type 2", "Advertising", "Realtor", "Culture", "Industry: type 12", "Trade: type 1", "Mobile", "Legal Services", "Cleaning", "Transport: type 1", "Industry: type 6", "Industry: type 10", "Religion", "Industry: type 13", "Trade: type 4", "Trade: type 5", "Industry: type 8"])

    st.subheader("External Credit Scores")
    col8, col9, col10 = st.columns(3)
    with col8:
        ext_source_1 = st.slider("EXT_SOURCE_1", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    with col9:
        ext_source_2 = st.slider("EXT_SOURCE_2", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    with col10:
        ext_source_3 = st.slider("EXT_SOURCE_3", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # --- Section 3: Predict Button ---
    submit_button = st.form_submit_button("Predict Risk")

if submit_button:
    # Prepare payload to send to FastAPI
    payload = {
        "income_total": income_total,
        "credit_amount": credit_amount,
        "annuity_amount": annuity_amount,
        "goods_price": goods_price,
        "gender": gender,
        "children_count": children_count,
        "age": age,
        "years_employed": years_employed,
        "name_contract_type": contract_type,
        "own_car": own_car,
        "own_realty": own_realty,
        "income_type": income_type,
        "family_status": family_status,
        "education_type": education_type,
        "housing_type": housing_type,
        "occupation_type": occupation_type,
        "name_type_suite": name_type_suite,
        "organization_type": organization_type,
        "ext_source_1": ext_source_1,
        "ext_source_2": ext_source_2,
        "ext_source_3": ext_source_3
    }

    st.divider()

    # --- Section 9: Error Handling ---
    with st.spinner("Processing prediction request..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=5)

            if response.status_code == 200:
                result = response.json()

                # Extracting values from API response
                # Assuming response format: {"default_probability": 0.638, "prediction": 1, "risk_level": "High Risk", "model_version": "1.0.0"}
                prob = result.get("default_probability", 0.0)
                pred = result.get("prediction", 0)
                risk_level = result.get("risk_level", "Unknown")

                st.header("Prediction Results")

                # --- Section 4: Prediction Results ---
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric(label="Default Probability", value=f"{prob*100:.1f}%")
                with col_b:
                    prediction_text = "Default" if pred == 1 else "Non Default"
                    st.metric(label="Prediction", value=prediction_text)
                with col_c:
                    st.metric(label="Risk Level", value=risk_level)

                # --- Section 5: Visual Risk Gauge ---
                st.subheader("Visual Risk Gauge")
                # Normalize the progress value to be strictly between 0.0 and 1.0
                progress_val = min(max(prob, 0.0), 1.0)
                st.progress(progress_val)
                if prob <= 0.3:
                    st.markdown("**10% → Safe**")
                elif prob <= 0.6:
                    st.markdown("**50% → Moderate**")
                else:
                    st.markdown("**80% → Risky**")

                # --- Section 6: Business Interpretation ---
                st.subheader("Business Interpretation")
                if risk_level == "Low Risk":
                    st.success("🟢 **Low Risk**: Applicant appears financially stable. Loan approval can proceed to standard verification.")
                elif risk_level == "Medium Risk":
                    st.warning("🟡 **Medium Risk**: Additional financial review is recommended.")
                elif risk_level == "High Risk":
                    st.error("🔴 **High Risk**: High probability of default. Consider enhanced risk assessment before approval.")
                else:
                    st.info("Risk assessment logic undefined for this output.")

                # --- Section 7: Feature Engineering Preview ---
                # Provide calculated values
                engineered_features = result.get("engineered_features", {})
                if not engineered_features:
                    # Mocking calculation for display if API doesn't return them directly
                    engineered_features = {
                        "Credit Income Ratio": credit_amount / income_total if income_total else 0,
                        "Annuity Income Ratio": annuity_amount / income_total if income_total else 0,
                        "Credit Term": credit_amount / annuity_amount if annuity_amount else 0,
                        "EXT_SOURCE_MEAN": (ext_source_1 + ext_source_2 + ext_source_3) / 3.0
                    }

                with st.expander("Engineered Features"):
                    st.write("Calculated values for this applicant:")
                    for k, v in engineered_features.items():
                        st.write(f"- **{k}**: `{v:.4f}`")

            else:
                st.error(f"Invalid response from API (Status Code: {response.status_code})")
                st.write(response.text)

        except requests.exceptions.ConnectionError:
            st.error("API Unavailable: Could not connect to the FastAPI service. Is it running?")
        except requests.exceptions.Timeout:
            st.error("Network Timeout: The API request took too long to complete.")
        except ValueError:
            st.error("Invalid response format: Failed to parse JSON.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
