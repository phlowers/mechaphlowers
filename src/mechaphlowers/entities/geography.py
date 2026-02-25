# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import TypedDict

import numpy as np

from mechaphlowers.data.geography.helpers import (
    bearing_to_direction,
    gps_to_bearing,
    gps_to_lambert93,
    haversine,
    reverse_haversine_float,
)


class SupportGeoInfo(TypedDict):
    latitude: np.ndarray
    longitude: np.ndarray
    elevation: np.ndarray
    distance_to_next: np.ndarray
    bearing_to_next: np.ndarray
    direction_to_next: np.ndarray
    lambert_93: tuple[np.ndarray, np.ndarray]


def geo_info_from_gps(lats: np.ndarray, lons: np.ndarray) -> SupportGeoInfo:
    """
    Create a list of support geo info from a list of gps points.
    Args:
        gps_points: A list of gps points.
    Returns:
        A list of support geo info.
    """
    from mechaphlowers.data.geography.elevation import gps_to_elevation

    elevations = gps_to_elevation(lats, lons)
    lambert_93_tuples = gps_to_lambert93(lats, lons)
    distances = haversine(lats[:-1], lons[:-1], lats[1:], lons[1:], unit="deg")
    bearings = gps_to_bearing(
        lats[:-1], lons[:-1], lats[1:], lons[1:], unit="deg"
    )
    directions = bearing_to_direction(bearings)

    return SupportGeoInfo(
        latitude=lats,
        longitude=lons,
        elevation=elevations,
        distance_to_next=distances,
        bearing_to_next=bearings,
        direction_to_next=directions,
        lambert_93=lambert_93_tuples,
    )


def get_dist_and_angles_from_gps(
    latitudes: np.ndarray, longitudes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute distances and angles between supports using latitudes and longitudes.

    Args:
        latitudes (np.ndarray): array of latitudes in decimal degrees
        longitudes (np.ndarray): array of longitudes in decimal degrees

    Returns:
        tuple[np.ndarray, np.ndarray]: tuple (distance, angles) in meters and geometrical degrees
    """
    lats_rolled_rad = np.radians(latitudes[1:])
    lons_rolled_rad = np.radians(longitudes[1:])

    lats_rad = np.radians(latitudes[:-1])
    lons_rad = np.radians(longitudes[:-1])
    distances = haversine(lats_rad, lons_rad, lats_rolled_rad, lons_rolled_rad)
    distances = np.append(distances, np.nan)

    # first and last angles are not computed
    angles = gps_to_bearing(
        lats_rad, lons_rad, lats_rolled_rad, lons_rolled_rad
    )
    # convert bearing to angles relative between supports
    angles = np.diff(angles)
    angles = np.insert(angles, 0, 0)
    angles = np.append(angles, 0)

    return distances, np.degrees(angles)


def get_gps_coordinates(
    start_lat: float,
    start_lon: float,
    azimuth: float,
    line_angles_degrees: np.ndarray,
    span_length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gets arrays of line angle and span length, and starting point data.

    Builds iteratively all the gps points using the input arrays.

    Input and output are in degrees. This function converts in radians in order to use reverse_haversine_float()

    Args:
        start_lat (float): latitude of the first point
        start_lon (float): longitude of the first point
        azimuth (float): azimuth of the first span in geometrical degrees
        line_angles_degrees (np.ndarray): line angle array (data from SectionArray), in geometrical degrees
        span_length (np.ndarray): span length array (data from SectionArray)

    Returns:
        tuple[np.ndarray, np.ndarray]: (lat, lon) two arrays of angles in degrees
    """
    current_lat = np.radians(start_lat)
    current_lon = np.radians(start_lon)
    lat_array = [current_lat]
    lon_array = [current_lon]
    bearings_deg = np.cumsum(line_angles_degrees) + azimuth
    bearings_rad = np.radians(bearings_deg)
    for index in range(len(line_angles_degrees) - 1):
        current_lat, current_lon = reverse_haversine_float(
            current_lat, current_lon, bearings_rad[index], span_length[index]
        )
        lat_array.append(current_lat)
        lon_array.append(current_lon)
    return np.degrees(lat_array), np.degrees(lon_array)
