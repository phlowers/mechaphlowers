# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.data.geography.helpers import (
    bearing_to_direction,
    distances_to_gps,
    gps_to_bearing,
    gps_to_lambert93,
    haversine,
    lambert93_to_gps,
    parse_support_distances_to_calculate_gps,
    reverse_haversine,
)


def test_reverse_haversine():
    # Test case 1: Moving north
    result = reverse_haversine(48.8566, 2.3522, 0, 100)
    assert isinstance(result, dict)
    assert 'lat' in result and 'lon' in result
    assert result['lat'] > 48.8566  # Should be north of Paris
    assert abs(result['lon'] - 2.3522) < 0.1  # Longitude should be similar

    # Test case 2: Moving east
    result = reverse_haversine(48.8566, 2.3522, 90, 100)
    assert result['lon'] > 2.3522  # Should be east of Paris
    assert abs(result['lat'] - 48.8566) < 0.1  # Latitude should be similar

    # Test case 3: Moving southwest
    result = reverse_haversine(48.8566, 2.3522, 225, 100)
    assert result['lat'] < 48.8566  # Should be south of Paris
    assert result['lon'] < 2.3522  # Should be west of Paris

def test_haversine_distance():
    # Test case 1: Same point
    assert haversine(48.8566, 2.3522, 48.8566, 2.3522) == 0

    # Test case 2: Paris to London (approximately 343 km)
    distance = haversine(48.8566, 2.3522, 51.5074, -0.1278)
    assert 340 < distance < 350

    # Test case 3: Paris to New York (approximately 5837 km)
    distance = haversine(48.8566, 2.3522, 40.7128, -74.0060)
    assert 5800 < distance < 5900

def test_gps_to_lambert93():
    # Test case 1: Paris coordinates
    x, y = gps_to_lambert93(48.866667, 2.33333)
    assert isinstance(x, float)
    assert isinstance(y, float)
    # Paris should be roughly at (652709, 6861137) in Lambert 93
    assert 650000 < x < 660000
    assert 6850000 < y < 6870000

    # Test case 2: Corner case - North Pole
    x, y = gps_to_lambert93(90, 0)
    assert isinstance(x, float)
    assert isinstance(y, float)

def test_calculate_bearing():
    # Test case 1: North bearing
    bearing = gps_to_bearing(48.8566, 2.3522, 49.8566, 2.3522)
    assert 0 <= bearing <= 360
    assert abs(bearing - 0) < 1

    # Test case 2: East bearing
    bearing = gps_to_bearing(48.8566, 2.3522, 48.8566, 3.3522)
    assert abs(bearing - 90) < 1

    # Test case 3: Southwest bearing
    bearing = gps_to_bearing(48.8566, 2.3522, 47.8566, 1.3522)
    assert 200 < bearing < 250

def test_get_direction_name():
    # Test all cardinal directions
    assert bearing_to_direction(0) == 'N'
    assert bearing_to_direction(45) == 'NE'
    assert bearing_to_direction(90) == 'E'
    assert bearing_to_direction(135) == 'SE'
    assert bearing_to_direction(180) == 'S'
    assert bearing_to_direction(225) == 'SW'
    assert bearing_to_direction(270) == 'W'
    assert bearing_to_direction(315) == 'NW'

    # Test edge cases
    assert bearing_to_direction(360) == 'N'
    assert bearing_to_direction(-45) == 'NW'

def test_parse_coordinates_and_calculate_gps():
    # Test case with simple coordinates
    coords = np.array([
        [0, 0, 0],
        [0, 0, 30],
        [0, 40, 30],
        [np.nan, np.nan, np.nan]
    ])
    first_support_gps = np.array([48.8566, 2.3522])  # Paris coordinates
    
    result = parse_support_distances_to_calculate_gps(coords, first_support_gps)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 2)
    assert not np.isnan(result[0]).any()
    assert not np.isnan(result[1]).any()
    assert not np.isnan(result[2]).any()

def test_calculate_gps_from_distances():
    # Test case 1: Moving north
    result = distances_to_gps(48.8566, 2.3522, 0, 1000)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] > 48.8566
    assert abs(result[1] - 2.3522) < 0.001

    # Test case 2: Moving east
    result = distances_to_gps(48.8566, 2.3522, 1000, 0)
    assert result[1] > 2.3522
    assert abs(result[0] - 48.8566) < 0.001

    # Test case 3: Moving southwest
    result = distances_to_gps(48.8566, 2.3522, -1000, -1000)
    assert result[0] < 48.8566
    assert result[1] < 2.3522

def test_lambert93_to_gps():
    paris_lat, paris_lon = 48.8566, 2.3522
    paris_x, paris_y = 652469.02, 6862035.26
    result_lat, result_lon = lambert93_to_gps(paris_x, paris_y)
    assert abs(result_lat - paris_lat) < 0.000001
    assert abs(result_lon - paris_lon) < 0.000001
    result_x, result_y = gps_to_lambert93(result_lat, result_lon)
    assert abs(result_x - paris_x) < 0.0001
    assert abs(result_y - paris_y) < 0.0001