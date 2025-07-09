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
    distances = haversine(lats[:-1], lons[:-1], lats[1:], lons[1:])
    bearings = gps_to_bearing(lats[:-1], lons[:-1], lats[1:], lons[1:])
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
