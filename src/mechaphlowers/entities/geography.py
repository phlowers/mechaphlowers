# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import TypedDict

import numpy as np
from typing_extensions import Literal

from mechaphlowers.data.geography.helpers import (
    bearing_to_direction,
    gps_to_bearing,
    gps_to_lambert93,
    haversine,
    lambert93_to_gps,
    reverse_haversine_float,
)
from mechaphlowers.data.units import Q_
from mechaphlowers.entities.errors import GpsNoDataAvailable
from mechaphlowers.utils import convert_angle_unsigned_to_signed


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
    directions = bearing_to_direction(-bearings)

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
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    unit_output_angles: Literal["rad", "deg", "grad"] = "deg",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute distances and angles between supports using latitudes and longitudes.

    Args:
        latitudes_deg (np.ndarray): array of latitudes in decimal degrees
        longitudes_deg (np.ndarray): array of longitudes in decimal degrees
        unit_output_angles (Literal["rad", "deg", "grad"], optional): unit of the output angles. Defaults to "deg".

    Raises:
        ValueError: If latitudes and longitudes arrays have different lengths, or unit_output_angles is not valid.

    Returns:
        tuple[np.ndarray, np.ndarray]: tuple (distance, angles) distance is in meters and angles is anti-clockwise
    """
    if len(latitudes_deg) != len(longitudes_deg):
        raise ValueError("latitudes and longitudes must have the same length")
    if unit_output_angles not in ["rad", "deg", "grad"]:
        raise ValueError(
            "unit_output_angles must be one of 'rad', 'deg', 'grad'"
        )
    lats_rolled_rad = np.radians(latitudes_deg[1:])
    lons_rolled_rad = np.radians(longitudes_deg[1:])

    lats_rad = np.radians(latitudes_deg[:-1])
    lons_rad = np.radians(longitudes_deg[:-1])
    distances = haversine(lats_rad, lons_rad, lats_rolled_rad, lons_rolled_rad)
    distances = np.append(distances, np.nan)

    # first and last angles are not computed
    bearings_rad = gps_to_bearing(
        lats_rad, lons_rad, lats_rolled_rad, lons_rolled_rad
    )
    # convert bearing to angles relative between supports
    angles_rad = np.diff(bearings_rad)
    angles_rad = convert_angle_unsigned_to_signed(angles_rad)
    angles_rad = np.concatenate(([0], angles_rad, [0]))

    angles_correct_unit = (
        Q_(angles_rad, "rad").to(unit_output_angles).magnitude
    )
    return distances, angles_correct_unit


def get_gps_from_arrays(
    start_lat_deg: float,
    start_lon_deg: float,
    azimuth_deg: float,
    line_angles_degrees: np.ndarray,
    span_length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gets arrays of line angle and span length, and starting point data.

    Builds iteratively all the gps points using the input arrays.

    Input and output are in degrees. This function converts in radians in order to use reverse_haversine_float()

    span_length and line_angles_degrees come from SectionArray. Their data is based on support view.
    Therefore:
    - last value of span_length is np.nan
    - first and last value are only relevant for support orientation, and not considered for this computation

    Args:
        start_lat_deg (float): latitude of the first point
        start_lon_deg (float): longitude of the first point
        azimuth_deg (float): azimuth of the first span in degrees, anti-clockwise. 0 means North, 90 means West.
        line_angles_degrees (np.ndarray): line angle array (data from SectionArray), in degrees, anti-clockwise
        span_length (np.ndarray): span length array (data from SectionArray)

    Returns:
        tuple[np.ndarray, np.ndarray]: (lat, lon) two arrays of GPS coordinates. Angles in degrees
    """
    current_lat_rad = np.radians(start_lat_deg)
    current_lon_rad = np.radians(start_lon_deg)
    lat_array_rad = [current_lat_rad]
    lon_array_rad = [current_lon_rad]
    # first value of line_angles is set to 0 to avoid unexpected behaviour.
    # now azimuth is truly the orientation of the first span
    line_angles_degrees[0] = 0.0
    bearings_deg = np.cumsum(line_angles_degrees) + azimuth_deg
    bearings_rad = np.radians(bearings_deg)
    # Deliberate choice to not take into account the last angle: refers to the angle with the next section
    for index in range(len(line_angles_degrees) - 1):
        # Build the current point using the previous one, length and angle
        current_lat_rad, current_lon_rad = reverse_haversine_float(
            current_lat_rad,
            current_lon_rad,
            bearings_rad[index],
            span_length[index],
        )
        lat_array_rad.append(current_lat_rad)
        lon_array_rad.append(current_lon_rad)
    return np.degrees(lat_array_rad), np.degrees(lon_array_rad)


class GeoLocator:
    """Stores a starting GPS point and azimuth, then computes pylon GPS/Lambert93 coordinates on demand.

    The starting point must be set via set_starting_gps() or set_starting_lambert93() before
    calling get_gps() or get_lambert93(). No computed arrays are cached; every call to
    get_gps() / get_lambert93() recomputes from the stored starting point and the provided arrays.
    """

    def __init__(self) -> None:
        self._latitude_0: float | None = None
        self._longitude_0: float | None = None
        self._azimuth_0: float | None = None

    def set_starting_gps(
        self,
        latitude_0: float,
        longitude_0: float,
        azimuth_0: float,
    ) -> None:
        """Set the starting GPS point and azimuth for the section.

        Args:
            latitude_0 (float): Latitude of the first support in decimal degrees.
            longitude_0 (float): Longitude of the first support in decimal degrees.
            azimuth_0 (float): Azimuth of the first span in degrees, anti-clockwise. 0 means North, 90 means West.
        """
        self._latitude_0 = latitude_0
        self._longitude_0 = longitude_0
        self._azimuth_0 = azimuth_0

    def set_starting_lambert93(
        self,
        easting: float,
        northing: float,
        azimuth_0: float,
    ) -> None:
        """Set the starting point from Lambert 93 coordinates and azimuth.

        Converts the Lambert 93 easting/northing to GPS (WGS84) then stores the result.

        Args:
            easting (float): Lambert 93 easting coordinate in meters.
            northing (float): Lambert 93 northing coordinate in meters.
            azimuth_0 (float): Azimuth of the first span in degrees, anti-clockwise. 0 means North, 90 means West.
        """
        lat, lon = lambert93_to_gps(np.float64(easting), np.float64(northing))
        self.set_starting_gps(float(lat), float(lon), azimuth_0)

    def __copy__(self) -> "GeoLocator":
        new = GeoLocator()
        new._latitude_0 = self._latitude_0
        new._longitude_0 = self._longitude_0
        new._azimuth_0 = self._azimuth_0
        return new

    def _check_gps_available(self) -> None:
        if self._latitude_0 is None:
            raise GpsNoDataAvailable(
                "GPS data is not available. Call set_starting_gps() or set_starting_lambert93() first."
            )

    def get_gps(
        self,
        line_angles_degrees: np.ndarray,
        span_length: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GPS coordinates for all pylons.

        Args:
            line_angles_degrees (np.ndarray): Line angle array in degrees, anti-clockwise.
            span_length (np.ndarray): Span length array in meters (last value is NaN).

        Returns:
            tuple[np.ndarray, np.ndarray]: (latitudes, longitudes) in decimal degrees.
        """
        self._check_gps_available()
        return get_gps_from_arrays(
            self._latitude_0,  # type: ignore[arg-type]
            self._longitude_0,  # type: ignore[arg-type]
            self._azimuth_0,  # type: ignore[arg-type]
            line_angles_degrees,
            span_length,
        )

    def get_lambert93(
        self,
        line_angles_degrees: np.ndarray,
        span_length: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Lambert 93 coordinates for all pylons.

        Args:
            line_angles_degrees (np.ndarray): Line angle array in degrees, anti-clockwise.
            span_length (np.ndarray): Span length array in meters (last value is NaN).

        Returns:
            tuple[np.ndarray, np.ndarray]: (easting, northing) in Lambert 93 meters.
        """
        lats, lons = self.get_gps(line_angles_degrees, span_length)
        return gps_to_lambert93(lats, lons)
