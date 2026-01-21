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
import random
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import uuid
from motion_patterns import (
    MotionPattern,
    CircularMotion,
    SupersonicLinearMotion,
    InstantDirectionChangeMotion,
    InstantAccelerationMotion,
)

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

# FREEZE_TIMESTAMP: if set to "true", json.now will stay constant
FREEZE_TIMESTAMP = os.environ.get("FREEZE_TIMESTAMP", "false").lower() == "true"
FROZEN_TIMESTAMP = time.time() if FREEZE_TIMESTAMP else None
if FREEZE_TIMESTAMP:
    print(f"[synthetic_adsb_server] FREEZE_TIMESTAMP enabled - json.now frozen at {FROZEN_TIMESTAMP}")

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

ENABLE_ANOMALIES = os.environ.get("ENABLE_ANOMALIES", "false").lower() == "true"
NORMAL_AIRCRAFT_COUNT = int(os.environ.get("NORMAL_AIRCRAFT_COUNT", "1"))
ANOMALOUS_AIRCRAFT_COUNT = int(os.environ.get("ANOMALOUS_AIRCRAFT_COUNT", "0")) if ENABLE_ANOMALIES else 0
ANOMALY_ADSB_PROBABILITY = float(os.environ.get("ANOMALY_ADSB_PROBABILITY", "0.1"))

SUPERSONIC_MACH_MIN = float(os.environ.get("SUPERSONIC_MACH_MIN", "2.0"))
SUPERSONIC_MACH_MAX = float(os.environ.get("SUPERSONIC_MACH_MAX", "5.0"))
SUPERSONIC_ALTITUDE_FT = int(os.environ.get("SUPERSONIC_ALTITUDE_FT", "50000"))

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


class Aircraft:
    """Represents a single aircraft with its motion pattern and properties."""

    def __init__(
        self,
        icao_hex,
        motion_pattern,
        altitude_ft,
        flight_number,
        is_anomalous=False,
        has_adsb=True,
        adsb_accurate=True,
        adsb_gs_override=None,
        adsb_track_override=None,
    ):
        self.icao_hex = icao_hex
        self.motion_pattern = motion_pattern
        self.altitude_ft = altitude_ft
        self.flight_number = flight_number
        self.is_anomalous = is_anomalous
        self.has_adsb = has_adsb
        self.adsb_accurate = adsb_accurate
        self.adsb_gs_override = adsb_gs_override
        self.adsb_track_override = adsb_track_override

    def get_data(self, current_time, timestamp_for_json):
        lat, lon = self.motion_pattern.get_position(current_time)
        actual_gs_knots = self.motion_pattern.get_velocity(current_time)
        actual_track_deg = self.motion_pattern.get_heading(current_time)

        if not self.has_adsb:
            return None

        if self.adsb_accurate:
            gs_knots = actual_gs_knots
            track_deg = actual_track_deg
        else:
            gs_knots = self.adsb_gs_override if self.adsb_gs_override is not None else actual_gs_knots
            track_deg = self.adsb_track_override if self.adsb_track_override is not None else actual_track_deg

        return {
            "hex": self.icao_hex,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "alt_baro": self.altitude_ft,
            "alt_geom": self.altitude_ft + 100,
            "gs": round(gs_knots, 1),
            "track": round(track_deg, 2),
            "true_heading": round(track_deg, 2),
            "flight": self.flight_number,
            "seen_pos": 0,
        }


class AircraftManager:
    """Manages multiple aircraft with different motion patterns."""

    def __init__(self):
        self.aircraft_list = []
        self._initialize_aircraft()

    def _generate_icao_hex(self, index):
        base_hex = int(ICAO_HEX, 16) if ICAO_HEX else 0xAEF123
        return f"{(base_hex + index):06X}"

    def _initialize_aircraft(self):
        aircraft_index = 0

        for i in range(NORMAL_AIRCRAFT_COUNT):
            icao_hex = self._generate_icao_hex(aircraft_index)
            motion = CircularMotion(TX_LAT, TX_LON, RADIUS_DEG, ANGULAR_SPEED)
            flight_number = f"SYN{aircraft_index + 1:03d}  "

            aircraft = Aircraft(
                icao_hex=icao_hex,
                motion_pattern=motion,
                altitude_ft=ALT_BARO_FT,
                flight_number=flight_number,
                is_anomalous=False,
            )
            self.aircraft_list.append(aircraft)
            aircraft_index += 1

        for i in range(ANOMALOUS_AIRCRAFT_COUNT):
            icao_hex = self._generate_icao_hex(aircraft_index)
            direction = random.uniform(0, 360)
            start_lat = TX_LAT + random.uniform(-0.1, 0.1)
            start_lon = TX_LON + random.uniform(-0.1, 0.1)

            anomaly_type = random.choice(["supersonic", "direction_change", "acceleration"])

            if anomaly_type == "supersonic":
                mach = random.uniform(SUPERSONIC_MACH_MIN, SUPERSONIC_MACH_MAX)
                motion = SupersonicLinearMotion(
                    start_lat=start_lat,
                    start_lon=start_lon,
                    mach_number=mach,
                    direction_deg=direction,
                )
                altitude_ft = SUPERSONIC_ALTITUDE_FT

            elif anomaly_type == "direction_change":
                velocity_knots = random.uniform(400, 600)
                change_interval = random.uniform(3.0, 7.0)
                motion = InstantDirectionChangeMotion(
                    start_lat=start_lat,
                    start_lon=start_lon,
                    velocity_knots=velocity_knots,
                    initial_direction_deg=direction,
                    change_interval_sec=change_interval,
                )
                altitude_ft = random.randint(20000, 40000)

            else:
                speed_profile = [
                    (3.0, random.uniform(300, 500)),
                    (2.0, 0),
                    (3.0, random.uniform(500, 700)),
                    (4.0, random.uniform(200, 400)),
                ]
                motion = InstantAccelerationMotion(
                    start_lat=start_lat,
                    start_lon=start_lon,
                    direction_deg=direction,
                    speed_profile=speed_profile,
                )
                altitude_ft = random.randint(15000, 35000)

            flight_number = f"ANOM{i + 1:02d}   "

            has_adsb = random.random() < ANOMALY_ADSB_PROBABILITY
            adsb_accurate = True
            adsb_gs_override = None
            adsb_track_override = None

            if has_adsb and anomaly_type == "supersonic":
                adsb_accurate = random.random() < 0.3
                if not adsb_accurate:
                    adsb_gs_override = random.uniform(300, 500)
                    adsb_track_override = direction + random.uniform(-30, 30)

            aircraft = Aircraft(
                icao_hex=icao_hex,
                motion_pattern=motion,
                altitude_ft=altitude_ft,
                flight_number=flight_number,
                is_anomalous=True,
                has_adsb=has_adsb,
                adsb_accurate=adsb_accurate,
                adsb_gs_override=adsb_gs_override,
                adsb_track_override=adsb_track_override,
            )
            self.aircraft_list.append(aircraft)
            aircraft_index += 1

    def get_all_aircraft_data(self, current_time, timestamp_for_json):
        data = [aircraft.get_data(current_time, timestamp_for_json) for aircraft in self.aircraft_list]
        return [d for d in data if d is not None]

    def get_all_aircraft_for_radar(self, current_time):
        aircraft_data = []
        for aircraft in self.aircraft_list:
            lat, lon = aircraft.motion_pattern.get_position(current_time)
            gs_knots = aircraft.motion_pattern.get_velocity(current_time)
            track_deg = aircraft.motion_pattern.get_heading(current_time)

            aircraft_data.append({
                "hex": aircraft.icao_hex,
                "lat": lat,
                "lon": lon,
                "alt_geom": aircraft.altitude_ft + 100,
                "gs": gs_knots,
                "track": track_deg,
            })
        return aircraft_data


aircraft_manager = AircraftManager()

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

def generate_synthetic_detections(aircraft_data_list, radar_config):
    """Generate synthetic radar detections for all aircraft and given radar."""
    if not aircraft_data_list:
        return []

    detections = []
    now = time.time()

    for aircraft in aircraft_data_list:
        bistatic_range = calculate_bistatic_range(
            aircraft["lat"],
            aircraft["lon"],
            aircraft["alt_geom"] * 0.3048,
            TX_LAT,
            TX_LON,
            TX_ALT,
            radar_config["lat"],
            radar_config["lon"],
            radar_config["alt"],
        )

        velocity_ms = aircraft["gs"] * 0.514444
        doppler_shift = velocity_ms * 2 * radar_config["frequency"] / 299792458

        detection = {
            "detection_id": str(uuid.uuid4()),
            "timestamp": now,
            "bistatic_range_m": round(bistatic_range, 2),
            "doppler_hz": round(doppler_shift, 2),
            "snr_db": 15.0 + (hash(aircraft["hex"]) % 10),
            "radar_id": radar_config["id"],
            "frequency_hz": radar_config["frequency"],
            "icao_hex": aircraft["hex"],
        }
        detections.append(detection)

    return detections


@app.route("/data/aircraft.json")
def serve_synthetic_adsb():
    """
    Generate aircraft data for all configured aircraft (normal + anomalous).
    """
    timestamp_for_json = FROZEN_TIMESTAMP if FREEZE_TIMESTAMP else time.time()
    now = time.time()

    aircraft_data = aircraft_manager.get_all_aircraft_data(now, timestamp_for_json)

    return jsonify({"now": timestamp_for_json, "aircraft": aircraft_data})


@app.route("/api/detection")
def radar_detection():
    """Generate synthetic radar detection data in blah2 format."""
    port = request.environ.get("SERVER_PORT", 5001)
    try:
        port = int(port)
    except:
        port = 5001

    radar_config = radar_configs.get(port)
    if not radar_config:
        radar_config = radar_configs[49158]

    current_time = time.time()
    aircraft_data = aircraft_manager.get_all_aircraft_for_radar(current_time)

    delays = []
    dopplers = []
    snrs = []

    if aircraft_data:
        detections = generate_synthetic_detections(aircraft_data, radar_config)
        for detection in detections:
            delay_seconds = detection["bistatic_range_m"] / 299792458
            delays.append(delay_seconds)
            dopplers.append(detection["doppler_hz"])
            snrs.append(detection["snr_db"])

    current_time = time.time()
    return jsonify(
        {
            "timestamp": int(current_time * 1000),
            "delay": delays,
            "doppler": dopplers,
            "snr": snrs,
            "status": "active",
        }
    )


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

        current_time = time.time()
        aircraft_data = aircraft_manager.get_all_aircraft_for_radar(current_time)

        delays = []
        dopplers = []
        snrs = []

        if aircraft_data:
            detections = generate_synthetic_detections(aircraft_data, radar_config)
            for detection in detections:
                delay_km = detection["bistatic_range_m"] / 1000.0

                if delay_km < 5.0:
                    print(f"WARNING: Unrealistically small bistatic range: {delay_km:.3f} km")
                elif delay_km > 300.0:
                    print(f"WARNING: Unrealistically large bistatic range: {delay_km:.3f} km")

                delays.append(delay_km)
                dopplers.append(detection["doppler_hz"])
                snrs.append(detection["snr_db"])

        current_time = time.time()
        return jsonify(
            {
                "timestamp": int(current_time * 1000),
                "delay": delays,
                "doppler": dopplers,
                "snr": snrs,
                "status": "active",
            }
        )
    
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
