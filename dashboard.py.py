import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="M&V Dashboard - RETROFIT-LAT", layout="wide")

st.title("🏠 AI-based Measurement & Verification (M&V) Dashboard")
st.markdown("*Energy savings prediction for residential building retrofits using Random Forest*")

# Load model and features
@st.cache_resource
def load_model():
    # Try multiple paths
    model_paths = [
        'models/thesis_mv_random_forest.pkl',
        'thesis_mv_random_forest.pkl',
        'model.pkl'
    ]
    
    model = None
    model_path_used = None
    
    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            model_path_used = path
            break
    
    if model is None:
        st.error("❌ Model file not found!")
        return None, None
    
    # Try to load features
    features_paths = [
        'models/thesis_mv_features.txt',
        'thesis_mv_features.txt',
        'features.txt'
    ]
    
    features = None
    for path in features_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                features = [line.strip() for line in f.readlines()]
            break
    
    # If features file not found, try to infer from model
    if features is None:
        st.warning("⚠️ Features file not found. Using default features.")
        # Try to get feature names from model if available
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
        else:
            features = ['floor_area', 'building_age', 'floors', 'retrofit']
    
    return model, features

model, FEATURES = load_model()

if model is None:
    st.stop()

st.success(f"✅ Model loaded! Features: {len(FEATURES)}")

# Display features for debugging
with st.expander("🔧 Model Info (Debug)"):
    st.write(f"**Features expected ({len(FEATURES)}):** {FEATURES}")
    if hasattr(model, 'n_features_in_'):
        st.write(f"**Model expects:** {model.n_features_in_} features")

# ==========================================
# SIDEBAR INPUTS
# ==========================================
st.sidebar.header("📋 Building Parameters")

# Initialize input values
input_values = {}

# Dynamically create inputs based on features
for feat in FEATURES:
    if feat == 'floor_area':
        input_values[feat] = st.sidebar.number_input("🏠 Floor Area (m²)", min_value=30, max_value=300, value=90, step=5)
    elif feat == 'building_age':
        input_values[feat] = st.sidebar.number_input("📅 Building Age (years)", min_value=0, max_value=150, value=20, step=5)
    elif feat == 'floors':
        input_values[feat] = st.sidebar.selectbox("🏢 Number of Floors", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    elif feat == 'retrofit':
        input_values[feat] = st.sidebar.selectbox("🔧 Retrofit Status", [0, 1], format_func=lambda x: "✅ Yes (Retrofitted)" if x == 1 else "❌ No (Baseline)")
    elif feat == 'region_code':
        region_options = {0: "Riga", 1: "Liepaja", 2: "Ventspils", 3: "Jelgava", 4: "Jurmala", 5: "Other"}
        input_values[feat] = st.sidebar.selectbox("📍 Region", list(region_options.keys()), format_func=lambda x: region_options[x])
    elif feat == 'class_code':
        class_options = {0: "A (Best)", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G (Worst)"}
        input_values[feat] = st.sidebar.selectbox("📊 Energy Class", list(class_options.keys()), format_func=lambda x: class_options[x])
    else:
        # Default to number input for unknown features
        input_values[feat] = st.sidebar.number_input(f"📊 {feat}", value=0, step=1)

# ==========================================
# PREDICTION
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔮 Predict Energy Consumption", type="primary", use_container_width=True):
        # Prepare features in correct order
        features_list = [input_values[feat] for feat in FEATURES]
        features_df = pd.DataFrame([features_list], columns=FEATURES)
        
        prediction = model.predict(features_df)[0]
        
        st.metric("⚡ Predicted Energy Consumption", f"{prediction:.2f} kWh/m²")
        
        # Calculate savings
        if 'retrofit' in FEATURES and input_values['retrofit'] == 1:
            # Create baseline
            baseline_values = features_list.copy()
            retrofit_idx = FEATURES.index('retrofit')
            baseline_values[retrofit_idx] = 0
            baseline_df = pd.DataFrame([baseline_values], columns=FEATURES)
            baseline_pred = model.predict(baseline_df)[0]
            
            savings = baseline_pred - prediction
            savings_pct = (savings / baseline_pred) * 100
            
            st.success(f"💡 **Retrofit Savings:** {savings:.2f} kWh/m² ({savings_pct:.1f}% reduction)")
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            categories = ['Baseline\n(No Retrofit)', 'Retrofitted']
            values = [baseline_pred, prediction]
            
            bars = ax.bar(categories, values, color=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}', ha='center', fontweight='bold')
            
            ax.set_ylabel('Energy Consumption (kWh/m²)')
            ax.set_title('Retrofit Impact', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            
        elif 'retrofit' in FEATURES and input_values['retrofit'] == 0:
            # Show potential
            retrofit_values = features_list.copy()
            retrofit_idx = FEATURES.index('retrofit')
            retrofit_values[retrofit_idx] = 1
            retrofit_df = pd.DataFrame([retrofit_values], columns=FEATURES)
            retrofit_pred = model.predict(retrofit_df)[0]
            potential_savings = prediction - retrofit_pred
            potential_pct = (potential_savings / prediction) * 100
            st.info(f"💡 **If retrofitted:** Would save ~{potential_savings:.2f} kWh/m² ({potential_pct:.1f}%)")

with col2:
    st.info("""
    **📖 About this M&V System**
    
    - **Model:** Random Forest Regressor
    - **Data:** RETROFIT-LAT dataset
    - **Target:** Energy (kWh/m²)
    
    **Interpretation:**
    - Lower energy = more efficient
    - Savings > 20% = good retrofit
    """)

st.markdown("---")
st.caption("🎓 AI-based Measurement & Verification System | Thesis Project")
