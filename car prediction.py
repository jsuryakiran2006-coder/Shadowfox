# ==============================
# CAR PRICE PREDICTION TRAINING
# ==============================

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------
# 1. Load Dataset
# ------------------------------

df = pd.read_csv("car.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ------------------------------
# 2. Feature Engineering
# ------------------------------

# Create Years of Service
df['Years_Service'] = 2026 - df['Year']

# Drop unnecessary columns
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# ------------------------------
# 3. One-Hot Encoding
# ------------------------------

df = pd.get_dummies(df, drop_first=True)

# ------------------------------
# 4. Split Features and Target
# ------------------------------

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# ------------------------------
# 5. Train-Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 6. Hyperparameter Tuning
# ------------------------------

rf = RandomForestRegressor()

params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=params,
    n_iter=20,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

print("Best Parameters:", random_search.best_params_)

# ------------------------------
# 7. Model Evaluation
# ------------------------------

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ------------------------------
# 8. Feature Importance
# ------------------------------

importances = best_model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ------------------------------
# 9. Save Model
# ------------------------------

pickle.dump(best_model, open("car_price_model.pkl", "wb"))

print("\nModel Saved Successfully as car_price_model.pkl")
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