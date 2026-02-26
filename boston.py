import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from joblib import dump
import os
from sklearn.linear_model import LinearRegression
from pathlib import Path
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
model = LinearRegression()

dump(model, "boston_house_model.joblib")
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ------------------------------------
# Load model + scaler (if present)
# ------------------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "boston_house_model.joblib"
    if not model_path.exists():
        return None, model_path
    model = load(model_path)
    return model, model_path

model, model_path = load_model()

if model is None:
    st.error(f"Model file not found: {model_path}")
    st.stop()
# ------------------------------------
# Title
# ------------------------------------
st.title("🏠 Boston House Price Prediction")
st.write("Predict house prices using Machine Learning")
st.divider()

# ------------------------------------
# Sidebar inputs
# ------------------------------------
st.sidebar.header("Enter House Details")

CRIM = st.sidebar.number_input("Crime Rate (CRIM)", 0.0, 100.0, 3.6)
ZN = st.sidebar.number_input("Residential Zoning (ZN)", 0.0, 100.0, 11.3)
INDUS = st.sidebar.number_input("Industrial Area (INDUS)", 0.0, 30.0, 11.1)
CHAS = st.sidebar.selectbox("Near River (CHAS)", [0, 1])
NOX = st.sidebar.number_input("Nitric Oxide (NOX)", 0.0, 1.0, 0.55)
RM = st.sidebar.number_input("Rooms (RM)", 1.0, 10.0, 6.2)
AGE = st.sidebar.number_input("Age (AGE)", 0.0, 100.0, 68.5)
DIS = st.sidebar.number_input("Distance (DIS)", 0.0, 15.0, 3.8)
RAD = st.sidebar.number_input("Highway Access (RAD)", 1, 24, 9)
TAX = st.sidebar.number_input("Tax (TAX)", 100.0, 800.0, 408.0)
PTRATIO = st.sidebar.number_input("Pupil Ratio (PTRATIO)", 10.0, 30.0, 18.4)
B = st.sidebar.number_input("B Value", 0.0, 400.0, 356.6)
LSTAT = st.sidebar.number_input("Lower Status (LSTAT)", 0.0, 50.0, 12.6)

# ------------------------------------
# Prediction
# ------------------------------------
if st.button("🔍 Predict Price"):

    data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE,
                      DIS, RAD, TAX, PTRATIO, B, LSTAT]])

    # Apply scaler if available
    scaler = None
    use_scaling = True   # or False


    if use_scaling:
        scaler = StandardScaler()

    if scaler is not None:
        try:
            data = scaler.transform(data)
        except Exception:
            # if scaler exists but transform fails, fall back to raw
            pass

    prediction = model.predict(data)[0]

    st.success("Prediction Completed!")
    st.subheader("💰 Estimated House Price")
    st.markdown(f"### ${prediction:.2f}K")
    st.caption("Price is in thousands of dollars")

# ------------------------------------
# Footer
# ------------------------------------
st.divider()
st.caption("Boston Housing ML Project | ShadowFox Internship")
dump(model, "boston_house_model.joblib")



