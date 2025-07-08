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
