# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import json
import os
from pathlib import Path
from typing import Union

import numpy as np
import requests

# Cache file path for persistent storage
_cache_file_path = Path.home() / ".mechaphlowers" / "elevation_cache.json"

# Cache for elevation data to avoid duplicate API requests
_elevation_cache = {}


def _load_elevation_cache():
    """
    Load elevation cache from file.
    """
    global _elevation_cache
    try:
        if _cache_file_path.exists():
            with open(_cache_file_path, "r") as f:
                _elevation_cache = json.load(f)
        else:
            _elevation_cache = {}
    except (json.JSONDecodeError, IOError) as e:
        print(
            f"Warning: Could not load elevation cache from {_cache_file_path}: {e}"
        )
        _elevation_cache = {}


def _save_elevation_cache():
    """
    Save elevation cache to file.
    """
    try:
        _cache_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_cache_file_path, "w") as f:
            json.dump(_elevation_cache, f)
    except IOError as e:
        print(
            f"Warning: Could not save elevation cache to {_cache_file_path}: {e}"
        )


def gps_to_lambert93(
    latitude: Union[np.float64, np.ndarray],
    longitude: Union[np.float64, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert GPS coordinates (WGS84) to Lambert 93 coordinates.

    Args:
        latitude: Latitude in decimal degrees (WGS84). Can be scalar or numpy array.
        longitude: Longitude in decimal degrees (WGS84). Can be scalar or numpy array.

    Returns:
        tuple: (X, Y) coordinates in Lambert 93 projection (in meters)
    """

    # Convert inputs to numpy arrays if they aren't already
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)

    # Lambert 93 projection parameters
    # Semi-major axis of the GRS80 ellipsoid
    a = 6378137.0
    # Flattening of the GRS80 ellipsoid
    f = 1 / 298.257222101

    # Derived parameters
    e2 = 2 * f - f * f  # First eccentricity squared

    # Lambert 93 specific parameters
    phi0 = np.radians(46.5)  # Latitude of origin
    phi1 = np.radians(44.0)  # First standard parallel
    phi2 = np.radians(49.0)  # Second standard parallel
    lambda0 = np.radians(3.0)  # Central meridian
    X0 = 700000.0  # False easting
    Y0 = 6600000.0  # False northing

    # Convert input coordinates to radians
    phi = np.radians(latitude)
    lambda_deg = np.radians(longitude)

    # Calculate auxiliary functions
    def m(phi):
        return np.cos(phi) / np.sqrt(1 - e2 * np.sin(phi) ** 2)

    def t(phi):
        return np.tan(np.pi / 4 - phi / 2) / (
            (1 - np.sqrt(e2) * np.sin(phi)) / (1 + np.sqrt(e2) * np.sin(phi))
        ) ** (np.sqrt(e2) / 2)

    # Calculate projection constants
    m1 = m(phi1)
    m2 = m(phi2)

    t0 = t(phi0)
    t1 = t(phi1)
    t2 = t(phi2)

    # Calculate n and F
    n = (np.log(m1) - np.log(m2)) / (np.log(t1) - np.log(t2))
    F = m1 / (n * t1**n)

    # Calculate rho0 (radius at origin)
    rho0 = a * F * t0**n

    # Calculate rho and theta for the point
    t_phi = t(phi)
    rho = a * F * t_phi**n
    theta = n * (lambda_deg - lambda0)

    # Calculate Lambert 93 coordinates
    X = X0 + rho * np.sin(theta)
    Y = Y0 + rho0 - rho * np.cos(theta)

    return (X, Y)


def lambert93_to_gps(
    lambert_e: Union[np.float64, np.ndarray],
    lambert_n: Union[np.float64, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Lambert 93 coordinates to WGS84 (longitude, latitude)

    Args:
        lambert_e: Lambert 93 Easting coordinate. Can be scalar or numpy array.
        lambert_n: Lambert 93 Northing coordinate. Can be scalar or numpy array.

    Returns:
        tuple: (latitude, longitude) in decimal degrees
    """

    # Convert inputs to numpy arrays if they aren't already
    lambert_e = np.asarray(lambert_e, dtype=np.float64)
    lambert_n = np.asarray(lambert_n, dtype=np.float64)

    constantes = {
        'GRS80E': 0.081819191042816,
        'LONG_0': 3,
        'XS': 700000,
        'YS': 12655612.0499,
        'n': 0.7256077650532670,
        'C': 11754255.4261,
    }

    del_x = lambert_e - constantes['XS']
    del_y = lambert_n - constantes['YS']
    gamma = np.arctan(-del_x / del_y)
    r = np.sqrt(del_x * del_x + del_y * del_y)
    latiso = np.log(constantes['C'] / r) / constantes['n']

    # Iterative calculation for sinPhiit
    sin_phi_it0 = np.tanh(
        latiso
        + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * np.sin(1))
    )
    sin_phi_it1 = np.tanh(
        latiso
        + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * sin_phi_it0)
    )
    sin_phi_it2 = np.tanh(
        latiso
        + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * sin_phi_it1)
    )
    sin_phi_it3 = np.tanh(
        latiso
        + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * sin_phi_it2)
    )
    sin_phi_it4 = np.tanh(
        latiso
        + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * sin_phi_it3)
    )
    sin_phi_it5 = np.tanh(
        latiso
        + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * sin_phi_it4)
    )
    sin_phi_it6 = np.tanh(
        latiso
        + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * sin_phi_it5)
    )

    long_rad = np.arcsin(sin_phi_it6)
    lat_rad = gamma / constantes['n'] + constantes['LONG_0'] / 180 * np.pi

    longitude = lat_rad / np.pi * 180
    latitude = long_rad / np.pi * 180

    return (latitude, longitude)


def gps_to_elevation(
    lat: np.typing.NDArray[np.float64], lon: np.typing.NDArray[np.float64]
) -> np.typing.NDArray[np.float64]:
    """
    Fetch elevation data for a list of locations using Open-Elevation API.

    This function caches results to avoid making duplicate API requests for the same
    coordinates. The cache persists across program runs by storing data in a file
    located at ~/.mechaphlowers/elevation_cache.json.

    Args:
        lat (np.typing.NDArray[np.float64]): Latitude of the location in degrees
        lon (np.typing.NDArray[np.float64]): Longitude of the location in degrees

    Returns:
        np.typing.NDArray[np.float64]: Elevation in meters
    """
    import hashlib

    url = "https://api.open-elevation.com/api/v1/lookup"

    _load_elevation_cache()

    # Convert inputs to numpy arrays if they aren't already
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Create locations list for the API
    locations = [
        {"latitude": float(lat.item(i)), "longitude": float(lon.item(i))}
        for i in range(lat.size)
    ]
    payload = {"locations": locations}

    # Create a cache key from the payload
    # Convert to JSON string for consistent hashing
    # hash the cache key
    cache_key = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()

    # Check if we have cached results
    if cache_key in _elevation_cache and not os.environ.get('PYTEST_VERSION'):
        return _elevation_cache[cache_key]

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        elevations = [result["elevation"] for result in data["results"]]

        # Cache the results
        _elevation_cache[cache_key] = elevations

        # Save cache to file for persistence
        _save_elevation_cache()

        return np.array(elevations)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching elevation data: {e}")
        return np.zeros(lat.size)  # Return zeros if request fails


def reverse_haversine(
    lat: np.typing.NDArray[np.float64],
    lon: np.typing.NDArray[np.float64],
    bearing: np.typing.NDArray[np.float64],
    distance: np.typing.NDArray[np.float64],
) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    """
    Implementation of the reverse of Haversine formula. Takes one set of
    latitude/longitude as a start point, a bearing, and a distance, and
    returns the resultant lat/long pair.

    Args:
        lat (np.typing.NDArray[np.float64]): Starting latitude in degrees
        lon (np.typing.NDArray[np.float64]): Starting longitude in degrees
        bearing (np.typing.NDArray[np.float64]): Bearing in degrees
        distance (np.typing.NDArray[np.float64]): Distance in kilometers

    Returns:
        tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]: Tuple containing the latitude and longitude of the result
    """
    R = 6378.137  # Radius of Earth in km

    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    angdist = distance / R
    theta = np.radians(bearing)

    lat2 = np.degrees(
        np.arcsin(
            np.sin(lat1) * np.cos(angdist)
            + np.cos(lat1) * np.sin(angdist) * np.cos(theta)
        )
    )

    lon2 = np.degrees(
        lon1
        + np.arctan2(
            np.sin(theta) * np.sin(angdist) * np.cos(lat1),
            np.cos(angdist) - np.sin(lat1) * np.sin(np.radians(lat2)),
        )
    )

    return (lat2, lon2)


def haversine(
    lat1: np.typing.NDArray[np.float64],
    lon1: np.typing.NDArray[np.float64],
    lat2: np.typing.NDArray[np.float64],
    lon2: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Args:
        lat1 (np.typing.NDArray[np.float64]): Latitude of point A in degrees
        lon1 (np.typing.NDArray[np.float64]): Longitude of point A in degrees
        lat2 (np.typing.NDArray[np.float64]): Latitude of point B in degrees
        lon2 (np.typing.NDArray[np.float64]): Longitude of point B in degrees

    Returns:
        np.typing.NDArray[np.float64]: Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def gps_to_bearing(
    lat1: np.typing.NDArray[np.float64],
    lon1: np.typing.NDArray[np.float64],
    lat2: np.typing.NDArray[np.float64],
    lon2: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    """
    Calculate the bearing between two points
    Returns bearing in degrees from north (0-360)
    Args:
        lat1 (np.typing.NDArray[np.float64]): Latitude of point A in degrees
        lon1 (np.typing.NDArray[np.float64]): Longitude of point A in degrees
        lat2 (np.typing.NDArray[np.float64]): Latitude of point B in degrees
        lon2 (np.typing.NDArray[np.float64]): Longitude of point B in degrees

    Returns:
        np.typing.NDArray[np.float64]: Bearing angle in degrees from north (0-360)
    """
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(
        dlon
    )
    bearing = np.arctan2(y, x)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360
    return bearing


def bearing_to_direction(bearing: np.typing.NDArray[np.float64]) -> np.ndarray:
    """
    Convert bearing angle to cardinal direction name
    Args:
        bearing (np.typing.NDArray[np.float64]): Bearing angle in degrees

    Returns:
        np.ndarray: Array of cardinal direction names
    """
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = np.round(bearing / 45) % 8
    return np.array([directions[int(i)] for i in index])


def distances_to_gps(
    lat_a: np.typing.NDArray[np.float64],
    lon_a: np.typing.NDArray[np.float64],
    x_meters: np.typing.NDArray[np.float64],
    y_meters: np.typing.NDArray[np.float64],
) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    """
    Calculate GPS coordinates of point B given point A's coordinates and x,y distances in meters.

    Args:
        lat_a (np.typing.NDArray[np.float64]): Latitude of point A in degrees
        lon_a (np.typing.NDArray[np.float64]): Longitude of point A in degrees
        x_meters (np.typing.NDArray[np.float64]): Distance from west to east in meters (positive = east, negative = west)
        y_meters (np.typing.NDArray[np.float64]): Distance from south to north in meters (positive = north, negative = south)

    Returns:
        tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]: Tuple containing the latitude and longitude of point B
    """
    # Convert distances to degrees
    # 1 degree of latitude is approximately 111,111 meters
    lat_change = y_meters / 111111.0

    # 1 degree of longitude varies with latitude
    # At the equator, 1 degree is about 111,111 meters
    # At other latitudes, multiply by cos(latitude)
    lon_change = x_meters / (111111.0 * np.cos(np.radians(lat_a)))

    # Calculate new coordinates
    lat_b = lat_a + lat_change
    lon_b = lon_a + lon_change

    return lat_b, lon_b
