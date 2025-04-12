# Library Imports
from flask import Flask, render_template, request
import pandas as pd
import os
import openrouteservice
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
ORS_API_KEY = os.getenv('ORS_API_KEY')
client = openrouteservice.Client(key=ORS_API_KEY)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Geocodes Address Using OpenRouteService API
def geocodeAddress(address):
    result = client.pelias_search(text=address)
    features = result.get('features', [])
    if features:
        return features[0]['geometry']['coordinates']
    return None

# Retrieves Driving Distance and Duration Between 2 coordinate Addresses Using OpenRouteService API
def getRouteInfo(startCoords, endCoords):
    route = client.directions(
        coordinates=[startCoords, endCoords],
        profile='driving-car',
        format='geojson'
    )
    segment = route['features'][0]['properties']['segments'][0]
    return segment['distance'], segment['duration']

# Renders Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Process Route to Handle Inputted Data
@app.route('/process', methods=['POST'])
def process():

    # Check for Request File or manually Inputted Addresses
    startAddress = request.form.get('startingAddress', '').strip()
    file = request.files.get('file')
    manualInput = request.form.get('manualAddresses')
    if not startAddress:
        return "Missing starting address", 400

    # Geocode Starting Address
    startCoords = geocodeAddress(startAddress)
    if not startCoords:
        return "Could not geocode starting address.", 400
    startLng, startLat = startCoords

    rawPoints = []

    # Manual Input
    if manualInput and not file:
        addresses = [line.strip() for line in manualInput.split('\n') if line.strip()]
        for addr in addresses:
            coords = geocodeAddress(addr)
            if coords:
                lng, lat = coords
                rawPoints.append({'lat': lat, 'lng': lng, 'label': addr})

    # CSV File Input
    elif file and file.filename.endswith('.csv'):
        filePath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filePath)
        try:
            df = pd.read_csv(filePath)
            requiredCols = {'Address', 'City', 'Observed Latitude', 'Observed Longitude'}
            if not requiredCols.issubset(df.columns):
                return "CSV must contain 'Address', 'City', 'Observed Latitude', and 'Observed Longitude' columns.", 400
            df = df.dropna(subset=['Observed Latitude', 'Observed Longitude'])
            df['fullAddress'] = df['Address'].astype(str) + ", " + df['City'].astype(str)
            for _, row in df.iterrows():
                lat, lng = row['Observed Latitude'], row['Observed Longitude']
                label = row['fullAddress']
                rawPoints.append({'lat': lat, 'lng': lng, 'label': label})
        except Exception as e:
            return f"Error processing CSV file: {str(e)}", 500
    else:
        return "Please either upload a valid CSV file or enter manual addresses.", 400

    # Cluster Points Using KMeans
    coordsArray = np.array([[p['lat'], p['lng']] for p in rawPoints])
    kmeans = KMeans(n_clusters=4, random_state=42).fit(coordsArray)

    # Group Points by Cluster
    clusters = defaultdict(list)
    for i, point in enumerate(rawPoints):
        clusterId = kmeans.labels_[i]
        clusters[clusterId].append(point)

    driverRoutes = []
    mapPoints = []

    # Compute Routes for Each Cluster
    for clusterId in sorted(clusters.keys()):

        points = clusters[clusterId]
        points.sort(key=lambda p: (p['lat'] - startLat) ** 2 + (p['lng'] - startLng) ** 2)
        total_distance_km = 0
        total_duration_min = 0
        route_points = []
        prev_coords = [startLng, startLat]

        for p in points:
            curr_coords = [p['lng'], p['lat']]
            distance, duration = getRouteInfo(prev_coords, curr_coords)
            route_points.append({
                'lat': p['lat'],
                'lng': p['lng'],
                'label': p['label'],
                'distance_km': round(distance / 1000, 2),
                'duration_min': round(duration / 60, 1)
            })
            total_distance_km += distance / 1000
            total_duration_min += duration / 60
            prev_coords = curr_coords
        
        driverRoutes.append({
            'driver': f"Driver {clusterId + 1}",
            'points': route_points,
            'total_distance_km': round(total_distance_km, 2),
            'total_duration_min': round(total_duration_min, 1)
        })
        
        mapPoints.extend([{**p, 'driver': int(clusterId + 1)} for p in route_points])

    # Render Results Page with Route Summaries and Map
    return render_template('results.html',
                           start=startAddress,
                           driverRoutes=driverRoutes,
                           mapPoints=mapPoints,
                           startLat=startLat,
                           startLng=startLng)

# Run Flask Development Server
if __name__ == '__main__':
    app.run(debug=True)

