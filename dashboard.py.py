import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="M&V Dashboard - RETROFIT-LAT", layout="wide")

# Title
st.title("🏠 AI-based Measurement & Verification (M&V) Dashboard")
st.markdown("*Energy savings prediction for residential building retrofits using Random Forest*")
st.markdown("**Dataset:** RETROFIT-LAT (1,010 building projects, before & after retrofit)")

# Load model
@st.cache_resource
def load_model():
    # Try multiple possible paths
    possible_paths = [
        'models/thesis_mv_random_forest.pkl',
        'thesis_mv_random_forest.pkl',
        'model.pkl'
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        st.error("❌ Model file not found!")
        st.info("""
        **Troubleshooting:**
        1. Check that `models/thesis_mv_random_forest.pkl` exists in GitHub
        2. Or upload the model file manually
        3. Make sure the file is not zipped
        """)
        return None, None
    
    # Load features
    features_path = 'models/thesis_mv_features.txt'
    if not os.path.exists(features_path):
        features_path = 'thesis_mv_features.txt'
    
    model = joblib.load(model_path)
    
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = [line.strip() for line in f.readlines()]
    else:
        # Default features
        features = ['floor_area', 'building_age', 'floors', 'retrofit', 'region_code', 'class_code']
    
    return model, features

model, FEATURES = load_model()

if model is None:
    st.stop()

st.success("✅ Model loaded successfully!")

# ==========================================
# SIDEBAR INPUTS
# ==========================================
st.sidebar.header("📋 Building Parameters")

floor_area = st.sidebar.number_input("🏠 Floor Area (m²)", min_value=30, max_value=300, value=90, step=5)
building_age = st.sidebar.number_input("📅 Building Age (years)", min_value=0, max_value=150, value=20, step=5)
floors = st.sidebar.selectbox("🏢 Number of Floors", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
retrofit = st.sidebar.selectbox("🔧 Retrofit Status", [0, 1], format_func=lambda x: "✅ Yes (Retrofitted)" if x == 1 else "❌ No (Baseline)")

# Region codes (from RETROFIT-LAT dataset)
region_options = {
    0: "Riga",
    1: "Liepaja", 
    2: "Ventspils",
    3: "Jelgava",
    4: "Jurmala",
    5: "Other"
}
region_code = st.sidebar.selectbox("📍 Region", list(region_options.keys()), format_func=lambda x: region_options[x])

# Energy class (A=best, G=worst)
class_options = {
    0: "A (Best)",
    1: "B",
    2: "C", 
    3: "D",
    4: "E",
    5: "F",
    6: "G (Worst)"
}
class_code = st.sidebar.selectbox("📊 Energy Class", list(class_options.keys()), format_func=lambda x: class_options[x])

# ==========================================
# PREDICTION
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔮 Predict Energy Consumption", type="primary", use_container_width=True):
        # Prepare features
        features_df = pd.DataFrame([[
            floor_area, building_age, floors, retrofit, region_code, class_code
        ]], columns=FEATURES)
        
        prediction = model.predict(features_df)[0]
        
        st.metric("⚡ Predicted Energy Consumption", f"{prediction:.2f} kWh/m²")
        
        # Calculate savings if retrofitted
        if retrofit == 1:
            # Baseline (without retrofit)
            baseline_df = features_df.copy()
            baseline_df['retrofit'] = 0
            baseline_pred = model.predict(baseline_df)[0]
            
            savings = baseline_pred - prediction
            savings_pct = (savings / baseline_pred) * 100
            
            st.success(f"💡 **Retrofit Savings:** {savings:.2f} kWh/m² ({savings_pct:.1f}% reduction)")
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            categories = ['Baseline\n(No Retrofit)', 'Retrofitted']
            values = [baseline_pred, prediction]
            colors = ['#e74c3c', '#2ecc71']
            
            bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f} kWh/m²', ha='center', fontweight='bold')
            
            ax.set_ylabel('Energy Consumption (kWh/m²)', fontsize=12)
            ax.set_title('Retrofit Impact on Energy Consumption', fontweight='bold', fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            
            st.pyplot(fig)
            
        else:
            # Show potential savings if retrofitted
            retrofit_df = features_df.copy()
            retrofit_df['retrofit'] = 1
            retrofit_pred = model.predict(retrofit_df)[0]
            potential_savings = prediction - retrofit_pred
            potential_pct = (potential_savings / prediction) * 100
            
            st.info(f"💡 **If retrofitted:** Would save ~{potential_savings:.2f} kWh/m² ({potential_pct:.1f}%)")
            st.caption("👉 Try selecting 'Retrofitted' above to see actual savings")

with col2:
    st.info("""
    **📖 About this M&V System**
    
    | Item | Details |
    |------|---------|
    | **Model** | Random Forest Regressor |
    | **Dataset** | RETROFIT-LAT |
    | **Samples** | 1,010 buildings |
    | **Features** | 6 parameters |
    
    **Features used:**
    - Floor area (m²)
    - Building age (years)
    - Number of floors
    - Retrofit status
    - Region
    - Energy class
    
    **Target:** Energy consumption (kWh/m²)
    """)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("🎓 AI-based Measurement & Verification (M&V) System | Thesis Project | Data: RETROFIT-LAT (Zenodo)")
