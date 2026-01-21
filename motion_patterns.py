#!/usr/bin/env python3
"""
motion_patterns.py

Abstract motion pattern classes for synthetic aircraft generation.
Supports various flight behaviors including normal circular flight and anomalous patterns.
"""

import math
import time
from abc import ABC, abstractmethod


class MotionPattern(ABC):
    """Abstract base class for aircraft motion patterns."""

    @abstractmethod
    def get_position(self, current_time):
        """
        Calculate aircraft position at given time.

        Returns:
            tuple: (lat, lon) in degrees
        """
        pass

    @abstractmethod
    def get_velocity(self, current_time):
        """
        Calculate aircraft velocity at given time.

        Returns:
            float: Ground speed in knots
        """
        pass

    @abstractmethod
    def get_heading(self, current_time):
        """
        Calculate aircraft heading at given time.

        Returns:
            float: True heading in degrees (0-360)
        """
        pass


class CircularMotion(MotionPattern):
    """Circular flight pattern around a center point."""

    def __init__(self, center_lat, center_lon, radius_deg, angular_speed):
        """
        Initialize circular motion pattern.

        Args:
            center_lat: Center latitude in degrees
            center_lon: Center longitude in degrees
            radius_deg: Radius in degrees
            angular_speed: Angular velocity in radians/second
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius_deg = radius_deg
        self.angular_speed = angular_speed

    def get_position(self, current_time):
        theta = (current_time * self.angular_speed) % (2 * math.pi)
        lat = self.center_lat + self.radius_deg * math.cos(theta)
        lon = self.center_lon + self.radius_deg * math.sin(theta)
        return lat, lon

    def get_velocity(self, current_time):
        theta = (current_time * self.angular_speed) % (2 * math.pi)
        lat, _ = self.get_position(current_time)

        dlat_dt = -self.radius_deg * self.angular_speed * math.sin(theta)
        dlon_dt = self.radius_deg * self.angular_speed * math.cos(theta)

        dlat_dt_m = dlat_dt * 111320
        dlon_dt_m = dlon_dt * 111320 * math.cos(math.radians(lat))

        speed_ms = math.sqrt(dlat_dt_m**2 + dlon_dt_m**2)
        return speed_ms * 1.94384

    def get_heading(self, current_time):
        theta = (current_time * self.angular_speed) % (2 * math.pi)
        lat, _ = self.get_position(current_time)

        dlat_dt = -self.radius_deg * self.angular_speed * math.sin(theta)
        dlon_dt = self.radius_deg * self.angular_speed * math.cos(theta)

        dlat_dt_m = dlat_dt * 111320
        dlon_dt_m = dlon_dt * 111320 * math.cos(math.radians(lat))

        track_rad = math.atan2(dlon_dt_m, dlat_dt_m)
        return math.degrees(track_rad) % 360


class SupersonicLinearMotion(MotionPattern):
    """Supersonic linear flight pattern (Mach 2-5)."""

    def __init__(self, start_lat, start_lon, mach_number, direction_deg, start_time=None):
        """
        Initialize supersonic linear motion pattern.

        Args:
            start_lat: Starting latitude in degrees
            start_lon: Starting longitude in degrees
            mach_number: Mach number (2.0-5.0 for supersonic)
            direction_deg: Flight direction in degrees (0=north, 90=east)
            start_time: Start time (defaults to current time)
        """
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.mach_number = mach_number
        self.direction_rad = math.radians(direction_deg)
        self.start_time = start_time if start_time is not None else time.time()

        self.velocity_ms = mach_number * 343.0

    def get_position(self, current_time):
        dt = current_time - self.start_time

        dlat = (self.velocity_ms * dt / 111320) * math.cos(self.direction_rad)
        dlon = (self.velocity_ms * dt / 111320) * math.sin(self.direction_rad)

        lat = self.start_lat + dlat
        lon = self.start_lon + (dlon / math.cos(math.radians(lat)))

        return lat, lon

    def get_velocity(self, current_time):
        return self.velocity_ms * 1.94384

    def get_heading(self, current_time):
        return math.degrees(self.direction_rad) % 360


class InstantDirectionChangeMotion(MotionPattern):
    """Linear motion with instant direction changes at specified intervals."""

    def __init__(self, start_lat, start_lon, velocity_knots, initial_direction_deg, change_interval_sec=5.0, start_time=None):
        """
        Initialize instant direction change motion pattern.

        Args:
            start_lat: Starting latitude in degrees
            start_lon: Starting longitude in degrees
            velocity_knots: Ground speed in knots
            initial_direction_deg: Initial flight direction in degrees (0=north, 90=east)
            change_interval_sec: Time between direction changes in seconds
            start_time: Start time (defaults to current time)
        """
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.velocity_knots = velocity_knots
        self.velocity_ms = velocity_knots * 0.514444
        self.initial_direction_rad = math.radians(initial_direction_deg)
        self.change_interval = change_interval_sec
        self.start_time = start_time if start_time is not None else time.time()

        self.segment_positions = []
        self._compute_segments()

    def _compute_segments(self):
        """Pre-compute position at each direction change."""
        current_lat = self.start_lat
        current_lon = self.start_lon
        current_dir = self.initial_direction_rad
        current_time = 0

        self.segment_positions.append({
            'time': current_time,
            'lat': current_lat,
            'lon': current_lon,
            'direction': current_dir
        })

        for i in range(10):
            current_time += self.change_interval
            dt = self.change_interval

            dlat = (self.velocity_ms * dt / 111320) * math.cos(current_dir)
            dlon = (self.velocity_ms * dt / 111320) * math.sin(current_dir)

            current_lat += dlat
            current_lon += dlon / math.cos(math.radians(current_lat))

            current_dir = math.radians((math.degrees(current_dir) + 90) % 360)

            self.segment_positions.append({
                'time': current_time,
                'lat': current_lat,
                'lon': current_lon,
                'direction': current_dir
            })

    def get_position(self, current_time):
        dt = current_time - self.start_time

        for i in range(len(self.segment_positions) - 1):
            seg = self.segment_positions[i]
            next_seg = self.segment_positions[i + 1]

            if seg['time'] <= dt < next_seg['time']:
                time_in_segment = dt - seg['time']
                direction = seg['direction']

                dlat = (self.velocity_ms * time_in_segment / 111320) * math.cos(direction)
                dlon = (self.velocity_ms * time_in_segment / 111320) * math.sin(direction)

                lat = seg['lat'] + dlat
                lon = seg['lon'] + dlon / math.cos(math.radians(lat))

                return lat, lon

        last_seg = self.segment_positions[-1]
        time_since_last = dt - last_seg['time']
        direction = last_seg['direction']

        dlat = (self.velocity_ms * time_since_last / 111320) * math.cos(direction)
        dlon = (self.velocity_ms * time_since_last / 111320) * math.sin(direction)

        lat = last_seg['lat'] + dlat
        lon = last_seg['lon'] + dlon / math.cos(math.radians(lat))

        return lat, lon

    def get_velocity(self, current_time):
        return self.velocity_knots

    def get_heading(self, current_time):
        dt = current_time - self.start_time

        for i in range(len(self.segment_positions) - 1):
            seg = self.segment_positions[i]
            next_seg = self.segment_positions[i + 1]

            if seg['time'] <= dt < next_seg['time']:
                return math.degrees(seg['direction']) % 360

        return math.degrees(self.segment_positions[-1]['direction']) % 360


class InstantAccelerationMotion(MotionPattern):
    """Linear motion with instant speed changes including standstill periods."""

    def __init__(self, start_lat, start_lon, direction_deg, speed_profile, start_time=None):
        """
        Initialize instant acceleration/deceleration motion pattern.

        Args:
            start_lat: Starting latitude in degrees
            start_lon: Starting longitude in degrees
            direction_deg: Flight direction in degrees (0=north, 90=east)
            speed_profile: List of (duration_sec, speed_knots) tuples
            start_time: Start time (defaults to current time)
        """
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.direction_rad = math.radians(direction_deg)
        self.speed_profile = speed_profile
        self.start_time = start_time if start_time is not None else time.time()

        self.segments = []
        self._compute_segments()

    def _compute_segments(self):
        """Pre-compute position at each speed change."""
        current_lat = self.start_lat
        current_lon = self.start_lon
        current_time = 0

        for duration_sec, speed_knots in self.speed_profile:
            speed_ms = speed_knots * 0.514444

            self.segments.append({
                'start_time': current_time,
                'end_time': current_time + duration_sec,
                'start_lat': current_lat,
                'start_lon': current_lon,
                'speed_knots': speed_knots,
                'speed_ms': speed_ms
            })

            dlat = (speed_ms * duration_sec / 111320) * math.cos(self.direction_rad)
            dlon = (speed_ms * duration_sec / 111320) * math.sin(self.direction_rad)

            current_lat += dlat
            current_lon += dlon / math.cos(math.radians(current_lat))
            current_time += duration_sec

    def get_position(self, current_time):
        dt = current_time - self.start_time

        for seg in self.segments:
            if seg['start_time'] <= dt < seg['end_time']:
                time_in_segment = dt - seg['start_time']

                dlat = (seg['speed_ms'] * time_in_segment / 111320) * math.cos(self.direction_rad)
                dlon = (seg['speed_ms'] * time_in_segment / 111320) * math.sin(self.direction_rad)

                lat = seg['start_lat'] + dlat
                lon = seg['start_lon'] + dlon / math.cos(math.radians(lat))

                return lat, lon

        last_seg = self.segments[-1]
        time_since_last = dt - last_seg['end_time']

        dlat = (last_seg['speed_ms'] * time_since_last / 111320) * math.cos(self.direction_rad)
        dlon = (last_seg['speed_ms'] * time_since_last / 111320) * math.sin(self.direction_rad)

        lat = last_seg['start_lat'] + dlat
        lon = last_seg['start_lon'] + dlon / math.cos(math.radians(lat))

        return lat, lon

    def get_velocity(self, current_time):
        dt = current_time - self.start_time

        for seg in self.segments:
            if seg['start_time'] <= dt < seg['end_time']:
                return seg['speed_knots']

        return self.segments[-1]['speed_knots']

    def get_heading(self, current_time):
        return math.degrees(self.direction_rad) % 360
