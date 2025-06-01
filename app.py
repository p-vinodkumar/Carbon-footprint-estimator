from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import joblib
import requests
import os
import pandas as pd

app = Flask(__name__)

# Load model and preprocessor
class EmissionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load the trained model and preprocessor
preprocessor = joblib.load('preprocessor.pkl')
input_dim = preprocessor.transformers_[0][1].n_features_in_ + preprocessor.transformers_[1][1].categories_[0].shape[0] - 1  # numeric features + onehot minus first dropped
model = EmissionModel(input_dim=input_dim)
model.load_state_dict(torch.load('emission_model_final.pth', map_location=torch.device('cpu')))
model.eval()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "AIzaSyDE1MV0UmquDjMKUH7UH3GlgHgQjKTEj0s")

feature_columns = [
    'route_distance',
    'avg_speed',
    'fuel_consumption',
    'cargo_weight',
    'temperature',
    'wind_speed',
    'traffic_level'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'source' in request.form and 'destination' in request.form:
            source = request.form['source']
            destination = request.form['destination']

            print(f"Source: {source}, Destination: {destination}")  # Debug input

            routes = get_route_options(source, destination)
            print(f"Number of routes received: {len(routes)}")  # Debug routes count

            if not routes:
                return render_template('index.html', prediction_text="No routes found or error fetching routes from Google Maps API.")

            route_results = []

            for route in routes:
                features_dict = extract_features_from_route(route)
                if features_dict is None:
                    print("Skipping route due to feature extraction failure")
                    continue

                df_features = pd.DataFrame([features_dict], columns=feature_columns)
                features_scaled = preprocessor.transform(df_features)
                input_tensor = torch.tensor(features_scaled, dtype=torch.float32)

                with torch.no_grad():
                    log_pred = model(input_tensor).item()

                pred_emission = np.expm1(log_pred)
                route_results.append((route['summary'], pred_emission))

            if not route_results:
                return render_template('index.html', prediction_text="No valid routes to predict emissions.")

            best_route = min(route_results, key=lambda x: x[1])
            return render_template('index.html', prediction_text=f"Best Route: {best_route[0]} with estimated emissions of {best_route[1]:.2f} kg CO2")

        else:
            # Manual input
            features_dict = {
                'route_distance': float(request.form['route_distance']),
                'avg_speed': float(request.form['avg_speed']),
                'fuel_consumption': float(request.form['fuel_consumption']),
                'cargo_weight': float(request.form['cargo_weight']),
                'temperature': float(request.form['temperature']),
                'wind_speed': float(request.form['wind_speed']),
                'traffic_level': request.form['traffic_level']
            }

            df_features = pd.DataFrame([features_dict], columns=feature_columns)
            features_scaled = preprocessor.transform(df_features)
            input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            with torch.no_grad():
                log_pred = model(input_tensor).item()
            pred_emission = np.expm1(log_pred)

            return render_template('index.html', prediction_text=f"Estimated Carbon Emission: {pred_emission:.2f} kg CO2")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

def get_route_options(source, destination):
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={source}&destination={destination}&alternatives=true&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    print("Google Maps API response status:", data.get("status"))
    print("Google Maps API response:", data)  # DEBUG: print full response
    return data.get("routes", [])

def extract_features_from_route(route):
    try:
        distance_km = route['legs'][0]['distance']['value'] / 1000
        duration_hr = route['legs'][0]['duration']['value'] / 3600
        avg_speed = distance_km / duration_hr if duration_hr > 0 else 40

        fuel_consumption = distance_km * 0.08
        cargo_weight = 1000

        temperature = 25
        wind_speed = 10
        traffic_level = 'medium'

        return {
            'route_distance': distance_km,
            'avg_speed': avg_speed,
            'fuel_consumption': fuel_consumption,
            'cargo_weight': cargo_weight,
            'temperature': temperature,
            'wind_speed': wind_speed,
            'traffic_level': traffic_level
        }
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True)
