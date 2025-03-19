import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Sample dataset (Replace with real data)
data = {
    "origin_lat": [12.9716, 12.2958, 13.0827],
    "origin_lon": [77.5946, 76.6394, 80.2707],
    "dest_lat": [13.0827, 12.9716, 12.2958],
    "dest_lon": [80.2707, 77.5946, 76.6394],
    "traffic_condition": [2, 1, 3],  # (1=Light, 2=Moderate, 3=Heavy)
    "battery_level": [80, 50, 90],
    "travel_time": [40, 60, 35]  # Target variable (in minutes)
}

df = pd.DataFrame(data)

# Split data
X = df.drop(columns=["travel_time"])
y = df["travel_time"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "route_prediction_model.pkl")
print("✅ Model trained & saved!")




