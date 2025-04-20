# Library Imports
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import openrouteservice
from openrouteservice.exceptions import ApiError
import warnings
# Supress OpenRouteService Warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openrouteservice")

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

# Retrieves Driving Distance and Duration Between 2 Coordinate Addresses Using OpenRouteService API
def getRouteInfo(startCoords, endCoords):
    try:
        route = client.directions(
            coordinates=[startCoords, endCoords],
            profile='driving-car',
            format='geojson'
        )
        segment = route['features'][0]['properties']['segments'][0]
        return segment['distance'], segment['duration']
    except ApiError as e:
        print(f"Failed to get route between {startCoords} and {endCoords}: {e}")
        return None, None

# Renders Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Process Route to Handle Inputted Data
@app.route('/process', methods=['POST'])
def process():

    # Check for Request File or Manually Inputted Addresses
    startAddress = request.form.get('startingAddress', '').strip()
    file = request.files.get('file')
    manualInput = request.form.get('manualAddresses')
    if not startAddress:
        return "Missing starting address", 400
    numDrivers = int(request.form.get('numDrivers', 4))

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

    # Filter Routable Points From Start Address
    routablePoints = []
    skippedCount = 0
    for p in rawPoints:
        curr_coords = [p['lng'], p['lat']]
        distance, duration = getRouteInfo([startLng, startLat], curr_coords)
        if distance is not None and duration is not None:
            p['start_distance'] = distance
            p['start_duration'] = duration
            routablePoints.append(p)
        else:
            print(f"Skipping unroutable point before clustering: {p['label']}")
            skippedCount += 1

    if len(routablePoints) < numDrivers:
        return f"Not enough routable points to assign {numDrivers} drivers.", 400

    # Cluster Points Using KMeans
    coordsArray = np.array([[p['lat'], p['lng']] for p in routablePoints])
    kmeans = KMeans(n_clusters=numDrivers, random_state=42).fit(coordsArray)

    # Group Points by Cluster
    clusters = defaultdict(list)
    for i, point in enumerate(routablePoints):
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
            if distance is None or duration is None:
                print(f"Skipping Unroutable Point: {p['label']}")
                continue
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

