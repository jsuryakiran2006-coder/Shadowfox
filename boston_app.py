import streamlit as st
import numpy as np
from joblib import load
from pathlib import Path

# ------------------------------------
# Page Config
# ------------------------------------
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ------------------------------------
# Load Model + Scaler
# ------------------------------------
@st.cache_resource
def load_artifacts():
    base_path = Path(__file__).parent

    model = load(base_path / "boston_house_model.joblib")
    scaler = load(base_path / "scaler.joblib")

    return model, scaler

model, scaler = load_artifacts()

# ------------------------------------
# Title
# ------------------------------------
st.title("🏠 Boston House Price Prediction")
st.write("Predict house prices using Machine Learning")
st.divider()

# ------------------------------------
# Sidebar Inputs
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

    # Apply trained scaler
    scaled_data = scaler.transform(data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    st.success("Prediction Completed!")
    st.subheader("💰 Estimated House Price")
    st.markdown(f"### ${prediction:.2f}K")
    st.caption("Price is in thousands of dollars")

# ------------------------------------
# Footer
# ------------------------------------
st.divider()
st.caption("Boston Housing ML Project | ShadowFox Internship")