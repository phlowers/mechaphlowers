# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import Mock, patch

import numpy as np

from mechaphlowers.data.geography.helpers import (
    bearing_to_direction,
    distances_to_gps,
    gps_to_bearing,
    gps_to_elevation,
    gps_to_lambert93,
    haversine,
    lambert93_to_gps,
    reverse_haversine,
)

test_gps_coors = np.array(
    [
        [48.8566, 2.3522],  # Paris
        [43.2965, 5.3698],  # Marseille
        [45.7640, 4.8357],  # Lyon
        [43.6047, 1.4442],  # Toulouse
        [48.5734, 7.7521],  # Strasbourg
        [47.2184, -1.5536],  # Nantes
        [50.6292, 3.0573],  # Lille
        [44.8378, -0.5792],  # Bordeaux
    ],
    dtype=np.float64,
)

test_lambert_coords = np.array(
    [
        [652469.02, 6862035.26],  # Paris
        [892390.22, 6247035.26],  # Marseille
        [842666.66, 6519924.37],  # Lyon
        [574357.82, 6279642.97],  # Toulouse
        [1050362.70, 6840899.65],  # Strasbourg
        [355577.80, 6689723.10],  # Nantes
        [704061.15, 7059136.59],  # Lille
        [417241.54, 6421813.35],  # Bordeaux
    ],
    dtype=np.float64,
)


def test_gps_to_lambert93():
    """Test GPS to Lambert 93 conversion with multiple coordinates using matrices."""

    latitudes = test_gps_coors[:, 0]
    longitudes = test_gps_coors[:, 1]

    x_coords, y_coords = gps_to_lambert93(latitudes, longitudes)

    assert isinstance(x_coords, np.ndarray)
    assert isinstance(y_coords, np.ndarray)
    assert x_coords.shape == (8,)
    assert y_coords.shape == (8,)
    assert x_coords.dtype == np.float64
    assert y_coords.dtype == np.float64

    assert np.allclose(x_coords, test_lambert_coords[:, 0], atol=1e-08)
    assert np.allclose(y_coords, test_lambert_coords[:, 1], atol=1e-08)


def test_lambert93_to_gps():
    """Test Lambert 93 to GPS conversion with multiple coordinates using matrices."""

    x_coords = test_lambert_coords[:, 0]
    y_coords = test_lambert_coords[:, 1]

    latitudes, longitudes = lambert93_to_gps(x_coords, y_coords)

    assert isinstance(latitudes, np.ndarray)
    assert isinstance(longitudes, np.ndarray)
    assert latitudes.shape == (8,)
    assert longitudes.shape == (8,)
    assert latitudes.dtype == np.float64
    assert longitudes.dtype == np.float64

    assert np.allclose(latitudes, test_gps_coors[:, 0], atol=1e-6)
    assert np.allclose(longitudes, test_gps_coors[:, 1], atol=1e-6)


def test_reverse_haversine():
    """Test reverse_haversine with multiple coordinate inputs."""

    # Test multiple starting points with different bearings and distances
    lat_start = np.array(
        [48.8566, 43.2965, 45.7640], dtype=np.float64
    )  # Paris, Marseille, Lyon
    lon_start = np.array([2.3522, 5.3698, 4.8357], dtype=np.float64)
    bearing = np.array(
        [0.0, 90.0, 180.0], dtype=np.float64
    )  # North, East, South
    distance = np.array(
        [50.0, 75.0, 25.0], dtype=np.float64
    )  # Different distances

    lat_end, lon_end = reverse_haversine(
        lat_start, lon_start, bearing, distance
    )

    assert isinstance(lat_end, np.ndarray)
    assert isinstance(lon_end, np.ndarray)
    assert lat_end.shape == (3,)
    assert lon_end.shape == (3,)
    assert lat_end.dtype == np.float64
    assert lon_end.dtype == np.float64

    # Check that north bearing increases latitude
    assert lat_end[0] > lat_start[0]
    # Check that east bearing increases longitude
    assert lon_end[1] > lon_start[1]
    # Check that south bearing decreases latitude
    assert lat_end[2] < lat_start[2]


def test_haversine():
    """Test haversine with known French city distances."""

    # Test distances between major French cities
    cities = {
        'Paris': (48.8566, 2.3522),
        'Marseille': (43.2965, 5.3698),
        'Lyon': (45.7640, 4.8357),
    }

    # Test a few known distances
    lat1 = np.array([cities["Paris"][0]], dtype=np.float64)
    lon1 = np.array([cities["Paris"][1]], dtype=np.float64)
    lat2 = np.array([cities["Marseille"][0]], dtype=np.float64)
    lon2 = np.array([cities["Marseille"][1]], dtype=np.float64)

    distance = haversine(lat1, lon1, lat2, lon2)
    assert np.isclose(distance[0], 660, atol=1)

    # Test Paris to Lyon
    lat2 = np.array([cities["Lyon"][0]], dtype=np.float64)
    lon2 = np.array([cities["Lyon"][1]], dtype=np.float64)

    distance = haversine(lat1, lon1, lat2, lon2)
    assert np.isclose(distance[0], 391, atol=1)


def test_gps_to_bearing_cardinal_directions():
    """Test gps_to_bearing with cardinal directions and known bearings."""

    # Test cardinal directions from Paris
    paris_lat = np.array([48.8566], dtype=np.float64)
    paris_lon = np.array([2.3522], dtype=np.float64)

    # Test North (should be close to 0 degrees)
    north_lat = np.array([49.8566], dtype=np.float64)  # 1 degree north
    north_lon = np.array([2.3522], dtype=np.float64)  # same longitude
    bearing = gps_to_bearing(paris_lat, paris_lon, north_lat, north_lon)
    assert np.isclose(bearing[0], 0.0, atol=5.0)  # Within 5 degrees of north

    # Test East (should be close to 90 degrees)
    east_lat = np.array([48.8566], dtype=np.float64)  # same latitude
    east_lon = np.array([3.3522], dtype=np.float64)  # 1 degree east
    bearing = gps_to_bearing(paris_lat, paris_lon, east_lat, east_lon)
    assert np.isclose(bearing[0], 90.0, atol=5.0)  # Within 5 degrees of east

    # Test South (should be close to 180 degrees)
    south_lat = np.array([47.8566], dtype=np.float64)  # 1 degree south
    south_lon = np.array([2.3522], dtype=np.float64)  # same longitude
    bearing = gps_to_bearing(paris_lat, paris_lon, south_lat, south_lon)
    assert np.isclose(bearing[0], 180.0, atol=5.0)  # Within 5 degrees of south

    # Test West (should be close to 270 degrees)
    west_lat = np.array([48.8566], dtype=np.float64)  # same latitude
    west_lon = np.array([1.3522], dtype=np.float64)  # 1 degree west
    bearing = gps_to_bearing(paris_lat, paris_lon, west_lat, west_lon)
    assert np.isclose(bearing[0], 270.0, atol=5.0)  # Within 5 degrees of west


def test_gps_to_bearing_cities():
    """Test gps_to_bearing with known routes between French cities."""

    # Test Paris to Marseille (should be roughly southeast)
    paris_lat = np.array([48.8566], dtype=np.float64)
    paris_lon = np.array([2.3522], dtype=np.float64)
    marseille_lat = np.array([43.2965], dtype=np.float64)
    marseille_lon = np.array([5.3698], dtype=np.float64)

    bearing = gps_to_bearing(
        paris_lat, paris_lon, marseille_lat, marseille_lon
    )
    # Paris to Marseille should be roughly southeast (135-180 degrees)
    assert np.isclose(bearing[0], 158.0, atol=1)

    # Test Paris to Lyon (should be roughly southeast)
    lyon_lat = np.array([45.7640], dtype=np.float64)
    lyon_lon = np.array([4.8357], dtype=np.float64)

    bearing = gps_to_bearing(paris_lat, paris_lon, lyon_lat, lyon_lon)
    # Paris to Lyon should be roughly southeast (135-180 degrees)
    assert np.isclose(bearing[0], 150.0, atol=1)


def test_bearing_to_direction():
    """Test bearing_to_direction with boundary values between directions."""

    test_cases = [
        (22, 'N'),
        (67, 'NE'),
        (112, 'E'),
        (157, 'SE'),
        (202, 'S'),
        (247, 'SW'),
        (292, 'W'),
        (337, 'NW'),
    ]

    for bearing, expected_direction in test_cases:
        bearing_array = np.array([bearing], dtype=np.float64)
        direction = bearing_to_direction(bearing_array)
        assert (
            direction == expected_direction
        ), f"Bearing {bearing}° should be {expected_direction}, got {direction}"


def test_distances_to_gps():
    """Test distances_to_gps."""

    lat_a = np.array(
        [48.8566, 43.2965, 45.7640], dtype=np.float64
    )  # Paris, Marseille, Lyon
    lon_a = np.array([2.3522, 5.3698, 4.8357], dtype=np.float64)
    x_meters = np.array(
        [50000.0, -30000.0, 0.0], dtype=np.float64
    )  # Different east/west movements
    y_meters = np.array(
        [100000.0, 0.0, -500000.0], dtype=np.float64
    )  # Different north/south movements

    lat_b, lon_b = distances_to_gps(lat_a, lon_a, x_meters, y_meters)

    assert isinstance(lat_b, np.ndarray)
    assert isinstance(lon_b, np.ndarray)
    assert lat_b.shape == (3,)
    assert lon_b.shape == (3,)
    assert lat_b.dtype == np.float64
    assert lon_b.dtype == np.float64

    # All results should be different from starting points
    assert not np.allclose(lat_b, lat_a)
    assert not np.allclose(lon_b, lon_a)
    assert np.allclose(lat_b, np.array([49.7, 43.3, 41.2]), atol=1e-1)
    assert np.allclose(lon_b, np.array([3.0, 4.9, 4.8]), atol=1e-1)


@patch('mechaphlowers.data.geography.helpers.requests.post')
def test_gps_to_elevation(mock_post):
    """Test gps_to_elevation."""

    # Mock successful API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "results": [
            {"elevation": 35.0},  # Paris elevation
            {"elevation": 12.0},  # Marseille elevation
            {"elevation": 173.0},  # Lyon elevation
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    # Test coordinates for French cities
    lat = np.array(
        [48.8566, 43.2965, 45.7640], dtype=np.float64
    )  # Paris, Marseille, Lyon
    lon = np.array([2.3522, 5.3698, 4.8357], dtype=np.float64)

    elevations = gps_to_elevation(lat, lon)

    # Check that the function was called with correct parameters
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.open-elevation.com/api/v1/lookup"

    # Check the payload format
    payload = call_args[1]['json']
    assert 'locations' in payload

    # Check return values
    assert isinstance(elevations, np.ndarray)
    assert elevations.shape == (3,)
    assert elevations.dtype == np.float64
    assert np.allclose(elevations, np.array([35.0, 12.0, 173.0]))
