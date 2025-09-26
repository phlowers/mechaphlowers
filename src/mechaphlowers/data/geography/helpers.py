# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Union

import numpy as np


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
        tuple[np.ndarray, np.ndarray]: (X, Y) coordinates in Lambert 93 projection (in meters)
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
        tuple[np.ndarray, np.ndarray]: (latitude, longitude) in decimal degrees
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


def reverse_haversine(
    lat: np.ndarray,
    lon: np.ndarray,
    bearing: np.ndarray,
    distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the reverse of Haversine formula. Takes one set of
    latitude/longitude as a start point, a bearing, and a distance, and
    returns the resultant lat/long pair.

    Args:
        lat (np.ndarray): Starting latitude in decimal degrees
        lon (np.ndarray): Starting longitude in decimal degrees
        bearing (np.ndarray): Bearing in decimal degrees
        distance (np.ndarray): Distance in meters

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the latitude and longitude of the result
    """
    R = 6378137  # Radius of Earth in meters

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
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Args:
        lat1 (np.ndarray): Latitude of point A in decimal degrees
        lon1 (np.ndarray): Longitude of point A in decimal degrees
        lat2 (np.ndarray): Latitude of point B in decimal degrees
        lon2 (np.ndarray): Longitude of point B in decimal degrees

    Returns:
        np.ndarray: Distance in meters
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
    r = 6371000  # Radius of earth in meters
    return c * r


def gps_to_bearing(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Calculate the bearing between two points
    Returns bearing in degrees from north (0-360)
    Args:
        lat1 (np.ndarray): Latitude of point A in decimal degrees
        lon1 (np.ndarray): Longitude of point A in decimal degrees
        lat2 (np.ndarray): Latitude of point B in decimal degrees
        lon2 (np.ndarray): Longitude of point B in decimal degrees

    Returns:
        np.ndarray: Bearing angle in degrees from north (0-360)
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


def bearing_to_direction(
    bearing: np.ndarray,
) -> np.ndarray:
    """
    Convert bearing angle to cardinal direction name
    Args:
        bearing (np.ndarray): Bearing angle in decimal degrees

    Returns:
        np.ndarray: Cardinal direction name
    """
    directions = np.array(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    index = np.round(bearing / 45) % 8
    return directions[index.astype(int)]


def distances_to_gps(
    lat_a: np.ndarray,
    lon_a: np.ndarray,
    x_meters: np.ndarray,
    y_meters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate GPS coordinates of point B given point A's coordinates and x,y distances in meters.

    Args:
        lat_a (np.ndarray): Latitude of point A in decimal degrees
        lon_a (np.ndarray): Longitude of point A in decimal degrees
        x_meters (np.ndarray): Distance from west to east in meters (positive = east, negative = west)
        y_meters (np.ndarray): Distance from south to north in meters (positive = north, negative = south)

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the latitude and longitude of point B
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


def support_distances_to_gps(
    support_bases_x_meters: np.ndarray,
    support_bases_y_meters: np.ndarray,
    first_support_lat: np.float64,
    first_support_lon: np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse support distances to calculate gps coordinates in the form:
    Args:
        support_bases_x_meters (np.typing.NDArray[np.float64]): Array of x distances from first support base to support bases excluding the first support base
        support_bases_y_meters (np.typing.NDArray[np.float64]): Array of y distances from first support base to support bases excluding the first support base
        first_support_lat (np.float64): Latitude of the first support base
        first_support_lon (np.float64): Longitude of the first support base

    Returns:
        tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]: Tuple containing the latitude and longitude of the support bases
    """
    # Calculate GPS coordinates for additional support bases
    additional_lats, additional_lons = distances_to_gps(
        np.full(
            len(support_bases_x_meters), first_support_lat, dtype=np.float64
        ),
        np.full(
            len(support_bases_x_meters), first_support_lon, dtype=np.float64
        ),
        support_bases_x_meters,
        support_bases_y_meters,
    )

    # Concatenate additional coordinates with first support coordinates
    all_lats = np.concatenate(
        [np.array([first_support_lat], dtype=np.float64), additional_lats]
    )
    all_lons = np.concatenate(
        [np.array([first_support_lon], dtype=np.float64), additional_lons]
    )

    return all_lats, all_lons
