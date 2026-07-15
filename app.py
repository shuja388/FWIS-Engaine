import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="FWIS - Food Waste Intelligence System",
    page_icon="🍏",
    layout="centered"
)

# Load Models and Feature Layout
@st.cache_resource
def load_assets():
    try:
        model_waste = joblib.load('model_waste.pkl')
        model_rsl = joblib.load('model_rsl.pkl')
        model_reason = joblib.load('model_reason.pkl')
        columns_layout = joblib.load('columns_layout.pkl')
        return model_waste, model_rsl, model_reason, columns_layout
    except FileNotFoundError as e:
        st.error(f"Could not load model files. Please ensure you have downloaded and placed "
                 f"all model files (`model_waste.pkl`, `model_rsl.pkl`, `model_reason.pkl`, `columns_layout.pkl`) "
                 f"in the same directory as this script. Error: {e}")
        return None, None, None, None

model_waste, model_rsl, model_reason, columns_layout = load_assets()

# --- STREAMLIT UI ---
st.title("🍏 Food Waste Intelligence System (FWIS)")
st.write("Inbound shipment prediction and optimization engine.")

if model_waste is not None:
    st.markdown("---")
    st.subheader("📋 Inbound Shipment Manifest Details")

    col1, col2 = st.columns(2)

    with col1:
        fruit = st.selectbox(
            "Fruit Variety",
            ['Mango', 'Apple', 'Banana', 'Strawberry', 'Orange']
        )
        packaging = st.selectbox(
            "Packaging Type",
            ['Plastic Vent', 'Corrugated Carton', 'Heavy Wooden', 'Mesh Bag']
        )
        transit_temp = st.slider("Average Transit Temp (°C)", 0.0, 30.0, 15.0, step=0.1)
        rh_pct = st.slider("Relative Humidity (%)", 40.0, 100.0, 75.0, step=0.1)

    with col2:
        distance = st.number_input("Distance Traveled (km)", min_value=10, max_value=10000, value=1500, step=50)
        delay_days = st.number_input("Warehouse/Customs Delay (Days)", min_value=0, max_value=15, value=2, step=1)
        
        # Calculate Base transit (consistent with training logic: dist / 450)
        base_transit = distance / 450.0
        transit_days = st.number_input("Total Transit Days", min_value=0.1, value=round(base_transit + delay_days, 1), step=0.1)

    # Trigger Prediction
    if st.button("🚀 Run FWIS Diagnostic", type="primary"):
        # 1. Feature Engineering (match the training setup)
        degree_days = transit_temp * transit_days
        vpd_proxy = (100.0 - rh_pct) * transit_temp

        # Assemble row matching df_fwis schema (pre-encoded)
        raw_input = {
            'Transit_Temp_C': transit_temp,
            'RH_Pct': rh_pct,
            'Distance_km': float(distance),
            'Delay_Days': delay_days,
            'Transit_Days': transit_days,
            'Degree_Days': degree_days,
            'VPD_Proxy': vpd_proxy
        }

        # One-hot encode Categoricals dynamically to align with training features
        # Fruits (with drop_first context)
        fruits_list = ['Banana', 'Mango', 'Orange', 'Strawberry'] # 'Apple' dropped first
        for f in fruits_list:
            raw_input[f'Fruit_{f}'] = 1 if fruit == f else 0

        # Packaging (with drop_first context)
        # 'Corrugated Carton' dropped first
        pkg_list = ['Heavy Wooden', 'Mesh Bag', 'Plastic Vent']
        for p in pkg_list:
            raw_input[f'Packaging_Type_{p}'] = 1 if packaging == p else 0

        # Build DataFrame and enforce exact column sequencing
        input_df = pd.DataFrame([raw_input])
        input_df = input_df[columns_layout]

        # 2. Run Inference
        predicted_w = model_waste.predict(input_df)[0]
        predicted_r = model_rsl.predict(input_df)[0]
        predicted_msg = model_reason.predict(input_df)[0]

        # 3. Present Results & Intelligence Alerts
        st.markdown("---")
        st.subheader("🔮 Intelligence Dashboard Results")

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Predicted Batch Waste", f"{predicted_w:.1f}%")
        res_col2.metric("Remaining Shelf Life", f"{predicted_r:.1f} Days")
        res_col3.metric("Primary Degradation Risk", predicted_msg)

        # 4. Warehouse Action Engine
        st.markdown("### ⚠️ Action Directive")
        if predicted_w > 25.0 or predicted_r < 3.0:
            st.error(
                "🚨 **ACTION REQUIRED:** Route instantly to local discount processing or juice processing. **Do NOT export.**"
            )
        elif predicted_r <= 7.0:
            st.warning(
                "⚡ **ACTION REQUIRED:** 'First-Expired, First-Out' (FEFO) override triggered. **Route to nearest regional market.**"
            )
        else:
            st.success(
                "✅ **ACTION REQUIRED:** Safe for standard cold storage or long-distance redistribution chains."
            )
