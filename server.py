#!/usr/bin/env python3
"""
synthetic_adsb_server.py

A Flask-based HTTP server that serves synthetic ADS-B data in tar1090 format.
Simulates one aircraft flying in a circular pattern around Mount Lofty (–34.9810, 138.7081),
at a constant barometric altitude, updating once per second.

Endpoint:
  GET /data/aircraft.json

Response schema:
{
  "now": <epoch seconds float>,
  "aircraft": [
    {
      "hex": <string>,         # ICAO hex address
      "lat": <float>,          # degrees
      "lon": <float>,          # degrees
      "alt_baro": <int>,       # feet
      "alt_geom": <int>,        # feet
      "flight": <string>,      # Synthetic flight number with padding
      "seen_pos": 0
    },
    … (more aircraft)
  ]
}
"""

import time
import math
import threading
import os
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import uuid

load_dotenv()


def require_env_var(var_name):
    value = os.environ.get(var_name)
    if value is None or value == "":
        raise EnvironmentError(
            f"Required environment variable '{var_name}' is missing or empty."
        )
    return value


REQUIRED_ENV_VARS = [
    "TX_LAT",
    "TX_LON",
    "TX_ALT",
    "FC_MHZ",
    "RADIUS_DEG",
    "ANGULAR_SPEED",
    "ALT_BARO_FT",
    "ICAO_HEX",
    "HOST",
    "PORT",
    "RADARS",
]
for var in REQUIRED_ENV_VARS:
    require_env_var(var)

TX_LAT = float(os.environ.get("TX_LAT"))
TX_LON = float(os.environ.get("TX_LON"))
TX_ALT = int(os.environ.get("TX_ALT"))
FC_MHZ = float(os.environ.get("FC_MHZ"))

RADIUS_DEG = float(os.environ.get("RADIUS_DEG"))
ANGULAR_SPEED = float(os.environ.get("ANGULAR_SPEED"))
ALT_BARO_FT = int(os.environ.get("ALT_BARO_FT"))
ICAO_HEX = os.environ.get("ICAO_HEX")

HOST = os.environ.get("HOST")
PORT = int(os.environ.get("PORT"))

try:
    RADARS = json.loads(os.environ.get("RADARS"))
except json.JSONDecodeError:
    raise EnvironmentError("Failed to parse RADARS from environment variable.")

app = Flask(__name__)
CORS(app)

radar_configs = {}
for radar in RADARS:
    if "port" not in radar:
        raise EnvironmentError(f"Radar {radar.get('id', 'unknown')} missing 'port' field in RADARS config")
    port = radar["port"]
    radar_configs[port] = {
        "id": radar["id"],
        "lat": radar["lat"],
        "lon": radar["lon"],
        "alt": radar["alt"],
        "frequency": FC_MHZ * 1e6
    }

def calculate_bistatic_range(aircraft_lat, aircraft_lon, aircraft_alt, tx_lat, tx_lon, tx_alt, rx_lat, rx_lon, rx_alt):
    """Calculate bistatic range (distance from tx to aircraft to rx)."""
    # Simplified calculation using great circle distance
    def distance(lat1, lon1, alt1, lat2, lon2, alt2):
        # Convert to radians
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
        
        # Haversine formula for surface distance
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
        surface_dist = 2 * 6371000 * math.asin(math.sqrt(a))  # Earth radius in meters
        
        # Add altitude difference
        alt_diff = alt2 - alt1
        return math.sqrt(surface_dist**2 + alt_diff**2)
    
    tx_to_aircraft = distance(tx_lat, tx_lon, tx_alt, aircraft_lat, aircraft_lon, aircraft_alt)
    aircraft_to_rx = distance(aircraft_lat, aircraft_lon, aircraft_alt, rx_lat, rx_lon, rx_alt)
    
    return tx_to_aircraft + aircraft_to_rx

def generate_synthetic_detection(aircraft_data, radar_config):
    """Generate synthetic radar detection for given aircraft and radar."""
    if not aircraft_data:
        return None
        
    aircraft = aircraft_data[0]  # Single aircraft
    
    # Calculate bistatic range
    bistatic_range = calculate_bistatic_range(
        aircraft["lat"], aircraft["lon"], aircraft["alt_geom"] * 0.3048,  # Convert feet to meters
        TX_LAT, TX_LON, TX_ALT,
        radar_config["lat"], radar_config["lon"], radar_config["alt"]
    )
    
    # Calculate approximate doppler (simplified)
    # This is a rough approximation for circular motion
    now = time.time()
    theta = (now * ANGULAR_SPEED) % (2 * math.pi)
    velocity_magnitude = RADIUS_DEG * 111320 * ANGULAR_SPEED  # Convert to m/s
    doppler_shift = velocity_magnitude * math.cos(theta) * 2 * radar_config["frequency"] / 299792458
    
    return {
        "detection_id": str(uuid.uuid4()),
        "timestamp": now,
        "bistatic_range_m": round(bistatic_range, 2),
        "doppler_hz": round(doppler_shift, 2),
        "snr_db": 15.0 + (hash(aircraft["hex"]) % 10),  # Synthetic SNR
        "radar_id": radar_config["id"],
        "frequency_hz": radar_config["frequency"]
    }


@app.route("/data/aircraft.json")
def serve_synthetic_adsb():
    """
    Generate one circular-flight aircraft at the current time.
    """
    now = time.time()
    theta = (now * ANGULAR_SPEED) % (2 * math.pi)

    lat = TX_LAT + RADIUS_DEG * math.cos(theta)
    lon = TX_LON + RADIUS_DEG * math.sin(theta)

    dlat_dt = -RADIUS_DEG * ANGULAR_SPEED * math.sin(theta)
    dlon_dt = RADIUS_DEG * ANGULAR_SPEED * math.cos(theta)

    dlat_dt_m = dlat_dt * 111320
    dlon_dt_m = dlon_dt * 111320 * math.cos(math.radians(lat))

    speed_ms = math.sqrt(dlat_dt_m**2 + dlon_dt_m**2)
    gs_knots = speed_ms * 1.94384

    track_rad = math.atan2(dlon_dt_m, dlat_dt_m)
    track_deg = math.degrees(track_rad) % 360

    aircraft = {
        "hex": ICAO_HEX,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "alt_baro": ALT_BARO_FT,
        "alt_geom": ALT_BARO_FT
        + 100,
        "gs": round(gs_knots, 1),
        "track": round(track_deg, 2),
        "true_heading": round(track_deg, 2),
        "flight": "SYN001  ",
        "seen_pos": 0,
    }

    return jsonify({"now": now, "aircraft": [aircraft]})


@app.route("/api/detection")
def radar_detection():
    """Generate synthetic radar detection data in blah2 format."""
    port = request.environ.get('SERVER_PORT', 5001)
    try:
        port = int(port)
    except:
        port = 5001
    
    # Get radar config for this port
    radar_config = radar_configs.get(port)
    if not radar_config:
        # Default config if port not found
        radar_config = radar_configs[49158]
    
    # Get current aircraft data
    aircraft_response = serve_synthetic_adsb()
    aircraft_data = aircraft_response.get_json()["aircraft"]
    
    delays = []
    dopplers = []
    
    if aircraft_data:
        detection = generate_synthetic_detection(aircraft_data, radar_config)
        if detection:
            # Convert bistatic range to delay (distance/speed of light)
            delay_seconds = detection["bistatic_range_m"] / 299792458
            delays.append(delay_seconds)
            dopplers.append(detection["doppler_hz"])
    
    current_time = time.time()
    return jsonify({
        "timestamp": int(current_time * 1000),  # milliseconds for blah2 format
        "delay": delays,
        "doppler": dopplers,
        "snr": [15.0] * len(delays) if delays else [],
        "status": "active"
    })


@app.route("/api/config")
def radar_config():
    """Return radar configuration in blah2 format."""
    port = request.environ.get('SERVER_PORT', 5001)
    try:
        port = int(port)
    except:
        port = 5001
        
    config = radar_configs.get(port, radar_configs[49158])
    
    return jsonify({
        "location": {
            "rx": {
                "latitude": config["lat"],
                "longitude": config["lon"],
                "altitude": config["alt"]
            },
            "tx": {
                "latitude": TX_LAT,
                "longitude": TX_LON,
                "altitude": TX_ALT
            }
        },
        "capture": {
            "fc": config["frequency"]
        },
        "truth": {
            "adsb": {
                "tar1090": f"http://synthetic-adsb-test:5001"
            }
        },
        "radar_id": config["id"],
        "status": "operational",
        "timestamp": time.time()
    })


@app.route("/radar1")
def radar1_detection():
    """Radar 1 detection endpoint."""
    os.environ['SERVER_PORT'] = '49158'
    return radar_detection()


@app.route("/radar2") 
def radar2_detection():
    """Radar 2 detection endpoint."""
    os.environ['SERVER_PORT'] = '49159'
    return radar_detection()


@app.route("/radar3")
def radar3_detection():
    """Radar 3 detection endpoint."""
    os.environ['SERVER_PORT'] = '49160'
    return radar_detection()


def run_server():
    app.run(host=HOST, port=PORT, threaded=True)


def create_radar_app(port):
    """Create a separate Flask app for radar detection on a specific port."""
    radar_app = Flask(f'radar_{port}')
    CORS(radar_app)
    
    @radar_app.route("/api/detection")
    def radar_detection():
        """Generate synthetic radar detection data in blah2 format."""
        radar_config = radar_configs.get(port, radar_configs[49158])
        
        # Get current aircraft data
        aircraft_response = serve_synthetic_adsb()
        aircraft_data = aircraft_response.get_json()["aircraft"]
        
        delays = []
        dopplers = []
        
        if aircraft_data:
            detection = generate_synthetic_detection(aircraft_data, radar_config)
            if detection:
                # Convert bistatic range from meters to kilometers
                delay_km = detection["bistatic_range_m"] / 1000.0
                
                # Validation: Ensure realistic range values (10-200 km typical for aircraft)
                if delay_km < 5.0:
                    print(f"WARNING: Unrealistically small bistatic range: {delay_km:.3f} km")
                elif delay_km > 300.0:
                    print(f"WARNING: Unrealistically large bistatic range: {delay_km:.3f} km")
                
                delays.append(delay_km)
                dopplers.append(detection["doppler_hz"])
        
        current_time = time.time()
        return jsonify({
            "timestamp": int(current_time * 1000),
            "delay": delays,
            "doppler": dopplers,
            "snr": [15.0] * len(delays) if delays else [],
            "status": "active"
        })
    
    @radar_app.route("/api/config")
    def radar_config_endpoint():
        """Return radar configuration in blah2 format."""
        config = radar_configs.get(port, radar_configs[49158])
        return jsonify({
            "location": {
                "rx": {
                    "latitude": config["lat"],
                    "longitude": config["lon"],
                    "altitude": config["alt"]
                },
                "tx": {
                    "latitude": TX_LAT,
                    "longitude": TX_LON,
                    "altitude": TX_ALT
                }
            },
            "capture": {
                "fc": config["frequency"]
            },
            "truth": {
                "adsb": {
                    "tar1090": f"http://localhost:5001"
                }
            },
            "radar_id": config["id"],
            "status": "operational",
            "timestamp": time.time()
        })
    
    return radar_app

def run_radar_server(port):
    """Run additional radar servers on different ports."""
    radar_app = create_radar_app(port)
    radar_app.run(host=HOST, port=port, threaded=True)


if __name__ == "__main__":
    print(
        f"[synthetic_adsb_server] starting on http://{HOST}:{PORT}/data/aircraft.json"
    )
    
    # Start main ADS-B server
    threading.Thread(target=run_server, daemon=True).start()
    
    # Always start radar detection servers on their respective ports
    radar_ports = sorted(radar_configs.keys())
    print(f"[synthetic_adsb_server] starting radar APIs on ports {', '.join(map(str, radar_ports))}")
    for port in radar_ports:
        threading.Thread(target=lambda p=port: run_radar_server(p), daemon=True).start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[synthetic_adsb_server] shutting down.")
