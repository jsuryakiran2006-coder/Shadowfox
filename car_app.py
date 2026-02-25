# ==============================
# CAR PRICE PREDICTION WEB APP
# ==============================

import streamlit as st
import pickle
import numpy as np
import datetime

# Load trained model
model = pickle.load(open("car_price_model.pkl", "rb"))

st.title("🚗 Car Selling Price Prediction App")

st.write("Enter car details below to estimate selling price.")

# ------------------------------
# User Inputs
# ------------------------------

present_price = st.number_input("Showroom Price (in Lakhs)", min_value=0.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0)
owner = st.selectbox("Number of Previous Owners", [0,1,2,3])

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

year = st.number_input("Manufacturing Year", 1990, datetime.datetime.now().year)

# ------------------------------
# Feature Engineering
# ------------------------------

years_service = datetime.datetime.now().year - year

# Encoding
fuel_diesel = 1 if fuel_type == "Diesel" else 0
fuel_petrol = 1 if fuel_type == "Petrol" else 0

seller_individual = 1 if seller_type == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

# ------------------------------
# Prediction
# ------------------------------

if st.button("Predict Selling Price"):

    input_data = np.array([[
        present_price,
        kms_driven,
        owner,
        years_service,
        fuel_diesel,
        fuel_petrol,
        seller_individual,
        transmission_manual
    ]])

    prediction = model.predict(input_data)

    st.success(f"Estimated Selling Price: ₹ {round(prediction[0],2)} Lakhs")