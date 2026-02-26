import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load Boston dataset
data = fetch_openml(name="boston", version=1, as_frame=True)
X = data.data
y = data.target.astype(float)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
model = LinearRegression()
model.fit(X_scaled, y)

# Save BOTH
dump(model, "boston_house_model.joblib")
dump(scaler, "scaler.joblib")

print("Model trained and saved.")