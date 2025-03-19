import os
import streamlit as st
import folium
from streamlit_folium import folium_static
import openrouteservice
import pandas as pd
import joblib
import requests

# Define paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "route_prediction_model.pkl")

# Load the trained ML model safely
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("⚠️ Model file not found! Please train the model first.")
    st.stop()  # Stop execution if model is missing

# API Keys (Consider storing these in environment variables for security)
ORS_API_KEY = os.getenv("ORS_API_KEY", "5b3ce3597851110001cf624816dc8a1f16c041d38d03a54a468112b6")
OPEN_CHARGE_MAP_API_KEY = os.getenv("OPEN_CHARGE_MAP_API_KEY", "5ddf3c4d-3eac-48b2-ad9f-a6333d957582")

# Initialize OpenRouteService Client
ors_client = openrouteservice.Client(key=ORS_API_KEY)

st.title("⚡ EV Smart Route & Charging Finder 🚗🔋")

# User Input
origin = st.text_input("Enter Origin (lat,lon)", "12.9716,77.5946")
destination = st.text_input("Enter Destination (lat,lon)", "13.0827,80.2707")
traffic_condition = st.selectbox("Traffic Condition", [1, 2, 3], format_func=lambda x: ["Light", "Moderate", "Heavy"][x - 1])
battery_level = st.slider("Battery Level (%)", 0, 100, 80)

if st.button("🔍 Predict Best Route"):
    try:
        # Convert input to float values
        origin_coords = tuple(map(float, origin.split(",")))
        destination_coords = tuple(map(float, destination.split(",")))

        # Predict Travel Time using ML Model
        df_input = pd.DataFrame([[origin_coords[0], origin_coords[1], destination_coords[0], destination_coords[1], traffic_condition, battery_level]],
                                columns=['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon', 'traffic_condition', 'battery_level'])

        predicted_time = model.predict(df_input)[0]

        # Fetch Route from OpenRouteService
        route = ors_client.directions(
            coordinates=[origin_coords, destination_coords],
            profile="driving-car",
            format="geojson"
        )

        route_coords = [(point[1], point[0]) for point in route["features"][0]["geometry"]["coordinates"]]

        # Fetch Charging Stations near Origin
        charging_stations_url = f"https://api.openchargemap.io/v3/poi/?output=json&latitude={origin_coords[0]}&longitude={origin_coords[1]}&maxresults=10&key={OPEN_CHARGE_MAP_API_KEY}"
        charging_stations = requests.get(charging_stations_url).json()

        # Create Map
        m = folium.Map(location=origin_coords, zoom_start=7)

        # Add Route to Map
        folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.7).add_to(m)

        # Fit map to the route to avoid excessive zooming
        m.fit_bounds(route_coords)

        # Add Charging Stations to Map
        for station in charging_stations:
            lat = station["AddressInfo"]["Latitude"]
            lon = station["AddressInfo"]["Longitude"]
            name = station["AddressInfo"]["Title"]

            folium.Marker(
                [lat, lon], popup=f"⚡ {name}", icon=folium.Icon(color="green", icon="bolt")
            ).add_to(m)

        # Display Map
        folium_static(m)
        st.success(f"🚀 Estimated Travel Time: {predicted_time:.2f} minutes")

    except openrouteservice.exceptions.ApiError as api_err:
        st.error(f"⚠️ Route Fetching Failed: {api_err}")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
