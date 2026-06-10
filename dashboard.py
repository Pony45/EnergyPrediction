import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="M&V Dashboard", layout="wide")

st.title("🏠 AI-based Measurement & Verification (M&V) Dashboard")
st.markdown("*Energy savings prediction for residential building retrofits using Random Forest*")

# Load model
@st.cache_resource
def load_model():
    model_path = 'models/thesis_mv_random_forest.pkl'
    features_path = 'models/thesis_mv_features.txt'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        return None, None
    
    model = joblib.load(model_path)
    
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = [line.strip() for line in f.readlines()]
    else:
        features = ['floor_area', 'building_age', 'floors', 'retrofit', 'region_code', 'class_code']
    
    return model, features

model, FEATURES = load_model()

if model is None:
    st.stop()

st.success("✅ Model loaded successfully!")

# Load model performance metrics
import json

@st.cache_resource
def load_metrics():
    metrics_path = 'models/model_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

metrics = load_metrics()

# ==========================================
# SIDEBAR INPUTS
# ==========================================
st.sidebar.header("📋 Building Parameters")

floor_area = st.sidebar.number_input("🏠 Floor Area (m²)", min_value=30, max_value=300, value=90, step=5)
building_age = st.sidebar.number_input("📅 Building Age (years)", min_value=0, max_value=150, value=20, step=5)
floors = st.sidebar.selectbox("🏢 Number of Floors", [1, 2, 3, 4, 5])
retrofit = st.sidebar.selectbox("🔧 Retrofit Status", [0, 1], format_func=lambda x: "✅ Yes (Retrofitted)" if x == 1 else "❌ No (Baseline)")
region_code = st.sidebar.selectbox("📍 Region", [0, 1, 2, 3, 4, 5], format_func=lambda x: ["Riga", "Liepaja", "Ventspils", "Jelgava", "Jurmala", "Other"][x])
class_code = st.sidebar.selectbox("📊 Energy Class", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["A (Best)", "B", "C", "D", "E", "F", "G (Worst)"][x])

# ==========================================
# PREDICTION BUTTON
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔮 Predict Energy Consumption", type="primary", use_container_width=True):
        
        # Prepare features
        data = [[floor_area, building_age, floors, retrofit, region_code, class_code]]
        df_input = pd.DataFrame(data, columns=FEATURES)
        prediction = model.predict(df_input)[0]
        
        # Total energy calculations
        total_per_year = prediction * floor_area
        total_per_month = total_per_year / 12
        total_per_day = total_per_year / 365
        
        # Display metrics
        st.subheader("📊 Prediction Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⚡ Per m² per Year", f"{prediction:.1f} kWh/m²/yr")
        m2.metric("🏠 Total per Year", f"{total_per_year:.0f} kWh/yr")
        m3.metric("📅 Total per Month", f"{total_per_month:.0f} kWh/month")
        m4.metric("🌙 Total per Day", f"{total_per_day:.1f} kWh/day")
        
        # ==========================================
        # GRAPH 1: BASELINE vs RETROFIT (Bar Chart)
        # ==========================================
        if retrofit == 1:
            # Calculate baseline (without retrofit)
            data_base = [[floor_area, building_age, floors, 0, region_code, class_code]]
            baseline_pred = model.predict(pd.DataFrame(data_base, columns=FEATURES))[0]
            baseline_total = baseline_pred * floor_area
            
            savings = baseline_pred - prediction
            savings_pct = (savings / baseline_pred) * 100
            savings_total = savings * floor_area
            
            st.subheader("📊 Energy Savings Analysis")
            
            # Bar chart
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            categories = ['Baseline\n(No Retrofit)', 'Retrofitted\n(With Retrofit)']
            values = [baseline_pred, prediction]
            colors = ['#e74c3c', '#2ecc71']
            
            bars = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
            
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f} kWh/m²/yr', ha='center', fontweight='bold', fontsize=11)
            
            # Add savings annotation
            ax1.annotate(f'💡 Savings: {savings:.1f} kWh/m²/yr\n({savings_pct:.1f}%)',
                        xy=(1, prediction + savings/2),
                        xytext=(1.3, prediction + savings/2 + 5),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
            
            ax1.set_ylabel('Energy Consumption (kWh/m²/year)', fontsize=12)
            ax1.set_title('Retrofit Impact on Energy Consumption', fontweight='bold', fontsize=14)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            st.pyplot(fig1)
            
            # ==========================================
            # GRAPH 2: SAVINGS GAUGE (Meter)
            # ==========================================
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            
            # Gauge colors
            if savings_pct < 10:
                color = '#e74c3c'  # Red - poor
                label = 'Low Savings'
            elif savings_pct < 20:
                color = '#f39c12'  # Orange - medium
                label = 'Medium Savings'
            else:
                color = '#2ecc71'  # Green - good
                label = 'High Savings'
            
            ax2.barh([0], [savings_pct], color=color, height=0.4, edgecolor='black')
            ax2.barh([0], [100], color='lightgray', height=0.4, alpha=0.3)
            ax2.set_xlim(0, 100)
            ax2.set_yticks([])
            ax2.set_xlabel('Energy Savings (%)', fontsize=12)
            ax2.set_title(f'Retrofit Efficiency: {label} ({savings_pct:.1f}% savings)', fontweight='bold')
            ax2.text(savings_pct + 2, 0, f'{savings_pct:.1f}%', va='center', fontweight='bold', fontsize=12)
            
            # Add threshold lines
            ax2.axvline(x=10, color='orange', linestyle='--', alpha=0.5)
            ax2.axvline(x=20, color='green', linestyle='--', alpha=0.5)
            ax2.text(5, -0.3, 'Poor', ha='center', fontsize=9)
            ax2.text(15, -0.3, 'Medium', ha='center', fontsize=9)
            ax2.text(60, -0.3, 'Good', ha='center', fontsize=9)
            
            st.pyplot(fig2)
            
            # ==========================================
            # GRAPH 3: TOTAL ENERGY COMPARISON (kWh/year)
            # ==========================================
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            
            categories_total = ['Baseline\n(No Retrofit)', 'Retrofitted\n(With Retrofit)']
            values_total = [baseline_total, total_per_year]
            
            bars3 = ax3.bar(categories_total, values_total, color=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.5)
            
            for bar, val in zip(bars3, values_total):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'{val:,.0f} kWh', ha='center', fontweight='bold', fontsize=11)
            
            ax3.set_ylabel('Total Energy Consumption (kWh/year)', fontsize=12)
            ax3.set_title('Total Annual Energy Consumption', fontweight='bold', fontsize=14)
            ax3.grid(axis='y', alpha=0.3, linestyle='--')
            
            st.pyplot(fig3)
            
            # Savings summary in a nice box
            st.info(f"""
            ### 💰 **Retrofit Savings Summary**
            
            | Metric | Value |
            |--------|-------|
            | **Energy Savings** | {savings:.1f} kWh/m²/year |
            | **Percentage Reduction** | {savings_pct:.1f}% |
            | **Total Annual Savings** | {savings_total:,.0f} kWh/year |
            | **Monthly Savings** | {savings_total/12:,.0f} kWh/month |
            | **Daily Savings** | {savings_total/365:.1f} kWh/day |
            """)
            
        else:
            # Show potential savings if not retrofitted
            data_retro = [[floor_area, building_age, floors, 1, region_code, class_code]]
            retrofit_pred = model.predict(pd.DataFrame(data_retro, columns=FEATURES))[0]
            potential_savings = prediction - retrofit_pred
            potential_pct = (potential_savings / prediction) * 100
            
            st.info(f"""
            💡 **If you retrofit this building:**
            - Would save ~{potential_savings:.1f} kWh/m²/year ({potential_pct:.1f}%)
            - Total annual savings: ~{potential_savings * floor_area:,.0f} kWh/year
            - **Try selecting 'Retrofitted' above to see full analysis!**
            """)
            
            # Simple comparison chart
            fig_simple, ax_simple = plt.subplots(figsize=(8, 5))
            ax_simple.bar(['Current\n(No Retrofit)', 'If Retrofitted'], 
                         [prediction, retrofit_pred],
                         color=['#e74c3c', '#2ecc71'], edgecolor='black')
            ax_simple.set_ylabel('Energy (kWh/m²/year)')
            ax_simple.set_title('Potential Retrofit Impact')
            st.pyplot(fig_simple)
        
        # ==========================================
        # GRAPH 4: FEATURE IMPORTANCE (For thesis)
        # ==========================================
        st.subheader("📊 Feature Importance Analysis")
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': ['Floor Area', 'Building Age', 'Floors', 'Retrofit', 'Region', 'Energy Class'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            colors_imp = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
            bars4 = ax4.barh(importance_df['Feature'], importance_df['Importance'], color=colors_imp, edgecolor='black')
            
            for bar, val in zip(bars4, importance_df['Importance']):
                ax4.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f} ({val*100:.1f}%)', 
                        va='center', fontsize=10)
            
            ax4.set_xlabel('Feature Importance Score', fontsize=12)
            ax4.set_title('Random Forest - Feature Importance', fontweight='bold', fontsize=14)
            ax4.invert_yaxis()
            ax4.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig4)
            
            st.caption("📝 **Interpretation:** Retrofit status is the most important factor, followed by floor area. This confirms that retrofit interventions have significant impact on energy consumption.")

with col2:
    st.info("""
    **📖 About this M&V System**
    
    | Item | Details |
    |------|---------|
    | **Model** | Random Forest Regressor |
    | **Dataset** | RETROFIT-LAT |
    | **Samples** | 1,010 buildings |
    | **Target** | kWh/m²/year |
    
    **Graphs Interpretation:**
    - **Bar Chart:** Compare baseline vs retrofit
    - **Gauge:** Savings efficiency (0-100%)
    - **Feature Importance:** Key drivers of energy use
    """)

    # Show model performance
if metrics:
    with st.sidebar.expander("📊 Model Performance", expanded=True):
        st.metric("R² Score", f"{metrics['r2_score']:.4f}", 
                  help="Higher is better (1.0 = perfect)")
        st.metric("MAE", f"{metrics['mae']:.2f} kWh/m²/yr",
                  help="Lower is better")
        st.caption(f"RMSE: {metrics['rmse']:.2f} kWh/m²/yr")
        st.progress(metrics['r2_score'], text="Model Accuracy")
        
    st.markdown("---")
    st.caption("🎓 AI-based Measurement & Verification System | Thesis Project")
