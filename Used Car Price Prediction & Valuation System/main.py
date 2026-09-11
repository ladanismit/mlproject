from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
import os
import uvicorn

app = FastAPI(
    title="Used Car Price Valuation API",
    description="Production REST API for Used Car Price Prediction & Valuation System.",
    version="1.0.0"
)

# Global states
model = None
artifacts = None

class CarInputs(BaseModel):
    oem: str = Field(..., example="Maruti", description="Brand / OEM name")
    model: str = Field(..., example="Swift", description="Model name")
    fuel: str = Field(..., example="Petrol", description="Fuel type (Petrol, Diesel, CNG, LPG, Electric)")
    transmission: str = Field(..., example="Manual", description="Transmission (Manual, Automatic)")
    owner_type: str = Field(..., example="First", description="Ownership (First, Second, Third, Fourth, Unregistered Car)")
    myear: int = Field(..., example=2018, ge=1980, le=2026, description="Manufacturing Year")
    km: float = Field(..., example=50000.0, ge=0.0, description="Total kilometers driven")
    engine_cc: float = Field(..., example=1197.0, ge=0.0, description="Engine capacity in cc")
    max_power_bhp: float = Field(..., example=83.0, ge=0.0, description="Max power in bhp")
    max_torque_nm: float = Field(..., example=113.0, ge=0.0, description="Max torque in Nm")
    kerb_weight: float = Field(..., example=960.0, ge=0.0, description="Kerb weight in kg")
    length: float = Field(..., example=3840.0, ge=0.0, description="Vehicle length in mm")
    width: float = Field(..., example=1735.0, ge=0.0, description="Vehicle width in mm")
    height: float = Field(..., example=1530.0, ge=0.0, description="Vehicle height in mm")
    no_of_cylinder: int = Field(..., example=4, ge=1, description="Number of cylinders")

def preprocess_raw_inputs(inputs: CarInputs, artifacts: dict) -> pd.DataFrame:
    features = artifacts['defaults'].copy()

    features['km'] = inputs.km
    features['Length'] = inputs.length
    features['Width'] = inputs.width
    features['Height'] = inputs.height
    features['Kerb Weight'] = inputs.kerb_weight
    features['No of Cylinder'] = float(inputs.no_of_cylinder)
    features['Max Power Delivered'] = inputs.max_power_bhp
    features['Max Torque Delivered'] = inputs.max_torque_nm

    clean_oem = inputs.oem.lower().strip()
    clean_model = inputs.model.lower().strip()

    le_oem = artifacts['label_encoders']['oem']
    le_model = artifacts['label_encoders']['model']

    features['oem'] = le_oem.transform([clean_oem])[0] if clean_oem in le_oem.classes_ else 0
    features['model'] = le_model.transform([clean_model])[0] if clean_model in le_model.classes_ else 0

    features['transmission_manual'] = 1 if inputs.transmission.lower().strip() == 'manual' else 0

    clean_fuel = inputs.fuel.lower().strip()
    features['fuel_diesel'] = 1 if clean_fuel == 'diesel' else 0
    features['fuel_electric'] = 1 if clean_fuel == 'electric' else 0
    features['fuel_lpg'] = 1 if clean_fuel == 'lpg' else 0
    features['fuel_petrol'] = 1 if clean_fuel == 'petrol' else 0

    clean_owner = inputs.owner_type.lower().strip()
    features['owner_type_first'] = 1 if clean_owner == 'first' else 0
    features['owner_type_second'] = 1 if clean_owner == 'second' else 0
    features['owner_type_third'] = 1 if clean_owner == 'third' else 0
    features['owner_type_fourth'] = 1 if clean_owner == 'fourth' else 0
    features['owner_type_unregistered car'] = 1 if clean_owner in ['unregistered', 'unregistered car'] else 0

    car_age = 2026 - inputs.myear
    features['car_age'] = float(car_age)
    features['KM_PER_YEAR'] = inputs.km / (car_age + 1.0)
    features['POWER_TO_WEIGHT'] = inputs.max_power_bhp / (inputs.kerb_weight + 1.0)
    features['TORQUE_TO_POWER'] = inputs.max_torque_nm / (inputs.max_power_bhp + 1.0)
    features['ENGINE_PER_CYLINDER'] = inputs.max_power_bhp / (float(inputs.no_of_cylinder) + 1.0)
    features['CAR_VOLUME'] = (inputs.length * inputs.width * inputs.height) / 1e6

    LUXURY_BRANDS = ['bmw', 'audi', 'mercedes-benz', 'jaguar', 'volvo', 'land rover',
                     'porsche', 'bentley', 'lamborghini', 'maserati', 'rolls-royce']
    PREMIUM_BRANDS = ['honda', 'toyota', 'hyundai', 'volkswagen', 'skoda', 'kia']

    features['LUXURY_BRAND_FLAG'] = 1 if clean_oem in LUXURY_BRANDS else 0
    features['PREMIUM_BRAND_FLAG'] = 1 if clean_oem in PREMIUM_BRANDS else 0

    features['HIGH_MILEAGE_FLAG'] = 1 if inputs.km > artifacts['median_km'] else 0
    features['OLD_CAR_FLAG'] = 1 if car_age > artifacts['median_car_age'] else 0

    df_vec = pd.DataFrame([features])
    df_vec = df_vec[artifacts['feature_names']]
    return df_vec

@app.on_event("startup")
def startup_event():
    global model, artifacts
    model = xgb.XGBRegressor()
    model.load_model('xgb_model.json')
    with open('preprocessor_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    print("Startup: XGBoost model & Preprocessor loaded successfully.")

@app.get("/", tags=["Status"])
def root():
    return {
        "api_name": "Used Car Price Valuation API",
        "status": "online",
        "version": "1.0.0",
        "documentation_url": "/docs"
    }

@app.get("/health", tags=["Status"])
def health():
    is_healthy = (model is not None) and (artifacts is not None)
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": artifacts is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", tags=["Valuation"], status_code=status.HTTP_200_OK)
def predict(payload: CarInputs):
    if model is None or artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server model is not loaded yet."
        )
    try:
        df_input = preprocess_raw_inputs(payload, artifacts)
        pred_raw = float(model.predict(df_input)[0])
        predicted_price = max(0.0, round(pred_raw))
        lower_bound = predicted_price * 0.95
        upper_bound = predicted_price * 1.05

        lower_lakhs = lower_bound / 100000
        upper_lakhs = upper_bound / 100000
        price_range = f"₹{lower_lakhs:.1f}L - ₹{upper_lakhs:.1f}L"

        return {
            "predicted_price": predicted_price,
            "price_range": price_range,
            "model_version": "1.0.0",
            "prediction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logging.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference valuation error occurred: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
