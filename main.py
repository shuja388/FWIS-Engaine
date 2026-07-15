import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --- 1. Define Request & Response Schemas ---
class ShipmentManifest(BaseModel):
    fruit: str = Field(..., description="Fruit variety (e.g., Mango, Apple, Banana, Strawberry, Orange)")
    packaging_type: str = Field(...,
                                description="Packaging style (e.g., Plastic Vent, Corrugated Carton, Heavy Wooden, Mesh Bag)")
    transit_temp_c: float = Field(..., ge=0.0, le=50.0, description="Average transit temperature in Celsius")
    rh_pct: float = Field(..., ge=0.0, le=100.0, description="Relative Humidity percentage")
    distance_km: float = Field(..., ge=0.0, description="Transit distance in kilometers")
    delay_days: int = Field(..., ge=0, description="Warehouse or customs delay in days")
    transit_days: float = Field(..., ge=0.1, description="Total transit days (including delays)")


class DiagnosticResponse(BaseModel):
    predicted_waste_pct: float = Field(..., description="Expected percentage of batch waste")
    remaining_shelf_life_days: float = Field(..., description="Predicted remaining shelf life in days")
    primary_degradation_risk: str = Field(..., description="Identified major driver of quality loss")
    action_directive: str = Field(..., description="Recommended warehouse routing instruction")


# --- 2. Define ML Assets Dictionary ---
MODELS = {}


# --- 3. Define Modern Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This block executes ON STARTUP
    required_files = {
        "model_waste": "model_waste.pkl",
        "model_rsl": "model_rsl.pkl",
        "model_reason": "model_reason.pkl",
        "columns_layout": "columns_layout.pkl"
    }

    # Check if all files exist in the current working directory
    for key, file_name in required_files.items():
        if not os.path.exists(file_name):
            raise RuntimeError(
                f"Missing required model file: '{file_name}'. Ensure this file is placed "
                f"in the same directory as main.py before running."
            )
        MODELS[key] = joblib.load(file_name)

    yield  # Handover control to the FastAPI application

    # Optional: Put cleanup/shutdown code here (executes ON SHUTDOWN)
    MODELS.clear()


# --- 4. Initialize FastAPI App with Lifespan ---
app = FastAPI(
    title="🍏 FWIS - Food Waste Intelligence System API",
    description="Inbound shipment prediction and optimization engine API.",
    version="1.0.0",
    lifespan=lifespan  # Register the lifespan handler here
)


# --- 5. Define API Endpoints ---
@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Food Waste Intelligence System (FWIS)",
        "docs_url": "/docs"
    }


@app.post("/predict", response_model=DiagnosticResponse)
def predict_shipment(manifest: ShipmentManifest):
    if not MODELS:
        raise HTTPException(status_code=500, detail="Models are not initialized.")

    try:
        # 1. Recreate Feature Engineering (Consistent with training)
        degree_days = manifest.transit_temp_c * manifest.transit_days
        vpd_proxy = (100.0 - manifest.rh_pct) * manifest.transit_temp_c

        # 2. Re-create the pre-encoded feature dictionary
        raw_input = {
            'Transit_Temp_C': manifest.transit_temp_c,
            'RH_Pct': manifest.rh_pct,
            'Distance_km': manifest.distance_km,
            'Delay_Days': manifest.delay_days,
            'Transit_Days': manifest.transit_days,
            'Degree_Days': degree_days,
            'VPD_Proxy': vpd_proxy
        }

        # Dynamic One-Hot Encoding (matching df_fwis schema drop_first structure)
        # Fruits ('Apple' is drop_first reference)
        fruits_list = ['Banana', 'Mango', 'Orange', 'Strawberry']
        for f in fruits_list:
            raw_input[f'Fruit_{f}'] = 1 if manifest.fruit.strip().title() == f else 0

        # Packaging ('Corrugated Carton' is drop_first reference)
        pkg_list = ['Heavy Wooden', 'Mesh Bag', 'Plastic Vent']
        for p in pkg_list:
            raw_input[f'Packaging_Type_{p}'] = 1 if manifest.packaging_type.strip().title() == p else 0

        # 3. Align DataFrame Columns exactly to match training schema
        input_df = pd.DataFrame([raw_input])
        columns_layout = MODELS["columns_layout"]
        input_df = input_df[columns_layout]

        # 4. Generate Predictions
        predicted_w = MODELS["model_waste"].predict(input_df)[0]
        predicted_r = MODELS["model_rsl"].predict(input_df)[0]
        predicted_msg = MODELS["model_reason"].predict(input_df)[0]

        # 5. Execute Action Directive Router logic
        if predicted_w > 25.0 or predicted_r < 3.0:
            directive = "🚨 ACTION REQUIRED: Route instantly to local discount processing or juice processing. Do NOT export."
        elif predicted_r <= 7.0:
            directive = "⚡ ACTION REQUIRED: 'First-Expired, First-Out' (FEFO) override triggered. Route to nearest regional market."
        else:
            directive = "✅ ACTION REQUIRED: Safe for standard cold storage or long-distance redistribution chains."

        return DiagnosticResponse(
            predicted_waste_pct=round(float(predicted_w), 1),
            remaining_shelf_life_days=round(float(predicted_r), 1),
            primary_degradation_risk=predicted_msg,
            action_directive=directive
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

