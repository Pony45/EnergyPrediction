import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="M&V Dashboard", layout="wide")
st.title("🏠 Energy Savings M&V Dashboard")

@st.cache_resource
def load_model():
    model = joblib.load('models/thesis_mv_random_forest.pkl')
    with open('models/thesis_mv_features.txt', 'r') as f:
        features = [line.strip() for line in f.readlines()]
    return model, features

model, FEATURES = load_model()
st.success("✅ Model loaded!")

st.sidebar.header("Building Parameters")

# Inputs
floor_area = st.sidebar.number_input("Floor Area (m²)", 30, 300, 90)
building_age = st.sidebar.number_input("Building Age (years)", 0, 150, 20)
floors = st.sidebar.selectbox("Floors", [1,2,3,4,5])
retrofit = st.sidebar.selectbox("Retrofit", [0,1], format_func=lambda x: "Yes" if x else "No")
region_code = st.sidebar.selectbox("Region", [0,1,2,3,4,5])
class_code = st.sidebar.selectbox("Energy Class", [0,1,2,3,4,5,6])

if st.button("Predict"):
    data = [[floor_area, building_age, floors, retrofit, region_code, class_code]]
    df_input = pd.DataFrame(data, columns=FEATURES)
    pred = model.predict(df_input)[0]
    st.metric("Predicted Energy", f"{pred:.2f} kWh/m²")
    
    if retrofit == 1:
        data_base = [[floor_area, building_age, floors, 0, region_code, class_code]]
        base_pred = model.predict(pd.DataFrame(data_base, columns=FEATURES))[0]
        savings = base_pred - pred
        st.success(f"Savings: {savings:.2f} kWh/m² ({savings/base_pred*100:.1f}%)")
