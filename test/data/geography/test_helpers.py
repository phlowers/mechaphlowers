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
    lambert93_to_gps,
    reverse_haversine,
    support_distances_to_gps,
)

# French cities mock data
FRENCH_CITIES_GPS = {
    'Paris': (48.8566, 2.3522),
    'Marseille': (43.2965, 5.3698),
    'Lyon': (45.7640, 4.8357),
    'Toulouse': (43.6047, 1.4442),
    'Strasbourg': (48.5734, 7.7521),
    'Nantes': (47.2184, -1.5536),
    'Lille': (50.6292, 3.0573),
    'Bordeaux': (44.8378, -0.5792),
}

FRENCH_CITIES_LAMBERT93 = {
    'Paris': (652469.02, 6862035.26),
    'Marseille': (892390.22, 6247035.26),
    'Lyon': (842666.66, 6519924.37),
    'Toulouse': (574357.82, 6279642.97),
    'Strasbourg': (1050362.70, 6840899.65),
    'Nantes': (355577.80, 6689723.10),
    'Lille': (704061.15, 7059136.59),
    'Bordeaux': (417241.54, 6421813.35),
}

# GPS coordinates array for all cities
test_gps_coors = np.array(
    [
        [FRENCH_CITIES_GPS['Paris'][0], FRENCH_CITIES_GPS['Paris'][1]],
        [FRENCH_CITIES_GPS['Marseille'][0], FRENCH_CITIES_GPS['Marseille'][1]],
        [FRENCH_CITIES_GPS['Lyon'][0], FRENCH_CITIES_GPS['Lyon'][1]],
        [FRENCH_CITIES_GPS['Toulouse'][0], FRENCH_CITIES_GPS['Toulouse'][1]],
        [
            FRENCH_CITIES_GPS['Strasbourg'][0],
            FRENCH_CITIES_GPS['Strasbourg'][1],
        ],
        [FRENCH_CITIES_GPS['Nantes'][0], FRENCH_CITIES_GPS['Nantes'][1]],
        [FRENCH_CITIES_GPS['Lille'][0], FRENCH_CITIES_GPS['Lille'][1]],
        [FRENCH_CITIES_GPS['Bordeaux'][0], FRENCH_CITIES_GPS['Bordeaux'][1]],
    ],
    dtype=np.float64,
)

# Lambert 93 coordinates array for all cities
test_lambert_coords = np.array(
    [
        [
            FRENCH_CITIES_LAMBERT93['Paris'][0],
            FRENCH_CITIES_LAMBERT93['Paris'][1],
        ],
        [
            FRENCH_CITIES_LAMBERT93['Marseille'][0],
            FRENCH_CITIES_LAMBERT93['Marseille'][1],
        ],
        [
            FRENCH_CITIES_LAMBERT93['Lyon'][0],
            FRENCH_CITIES_LAMBERT93['Lyon'][1],
        ],
        [
            FRENCH_CITIES_LAMBERT93['Toulouse'][0],
            FRENCH_CITIES_LAMBERT93['Toulouse'][1],
        ],
        [
            FRENCH_CITIES_LAMBERT93['Strasbourg'][0],
            FRENCH_CITIES_LAMBERT93['Strasbourg'][1],
        ],
        [
            FRENCH_CITIES_LAMBERT93['Nantes'][0],
            FRENCH_CITIES_LAMBERT93['Nantes'][1],
        ],
        [
            FRENCH_CITIES_LAMBERT93['Lille'][0],
            FRENCH_CITIES_LAMBERT93['Lille'][1],
        ],
        [
            FRENCH_CITIES_LAMBERT93['Bordeaux'][0],
            FRENCH_CITIES_LAMBERT93['Bordeaux'][1],
        ],
    ],
    dtype=np.float64,
)

# Cardinal direction test points (relative to Paris)
CARDINAL_POINTS = {
    'north': (49.8566, 2.3522),  # 1 degree north of Paris
    'east': (48.8566, 3.3522),  # 1 degree east of Paris
    'south': (47.8566, 2.3522),  # 1 degree south of Paris
    'west': (48.8566, 1.3522),  # 1 degree west of Paris
}


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
        [
            FRENCH_CITIES_GPS['Paris'][0],
            FRENCH_CITIES_GPS['Marseille'][0],
            FRENCH_CITIES_GPS['Lyon'][0],
        ],
        dtype=np.float64,
    )
    lon_start = np.array(
        [
            FRENCH_CITIES_GPS['Paris'][1],
            FRENCH_CITIES_GPS['Marseille'][1],
            FRENCH_CITIES_GPS['Lyon'][1],
        ],
        dtype=np.float64,
    )
    bearing = np.array(
        [0.0, 90.0, 180.0], dtype=np.float64
    )  # North, East, South
    distance = np.array(
        [50000.0, 75000.0, 25000.0], dtype=np.float64
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

    assert np.allclose(
        lat_end, [49.30575764, 43.29276776, 45.53942118], atol=1e-6
    )
    assert np.allclose(lon_end, [2.3522, 6.29545998, 4.8357], atol=1e-6)


# def test_haversine():
#     """Test haversine with known French city distances."""

#     # Test distances between major French cities
#     lat1 = np.array([FRENCH_CITIES_GPS["Paris"][0]], dtype=np.float64)
#     lon1 = np.array([FRENCH_CITIES_GPS["Paris"][1]], dtype=np.float64)
#     lat2 = np.array([FRENCH_CITIES_GPS["Marseille"][0]], dtype=np.float64)
#     lon2 = np.array([FRENCH_CITIES_GPS["Marseille"][1]], dtype=np.float64)

#     distance = haversine(lat1, lon1, lat2, lon2)
#     assert np.isclose(distance[0], 660478, atol=1)

#     # Test Paris to Lyon
#     lat2 = np.array([FRENCH_CITIES_GPS["Lyon"][0]], dtype=np.float64)
#     lon2 = np.array([FRENCH_CITIES_GPS["Lyon"][1]], dtype=np.float64)

#     distance = haversine(lat1, lon1, lat2, lon2)
#     assert np.isclose(distance[0], 391498, atol=1)


def test_gps_to_bearing_cardinal_directions():
    """Test gps_to_bearing with cardinal directions and known bearings."""

    # Test cardinal directions from Paris
    paris_lat = np.array([FRENCH_CITIES_GPS['Paris'][0]], dtype=np.float64)
    paris_lon = np.array([FRENCH_CITIES_GPS['Paris'][1]], dtype=np.float64)

    # Test North (should be close to 0 degrees)
    north_lat = np.array([CARDINAL_POINTS['north'][0]], dtype=np.float64)
    north_lon = np.array([CARDINAL_POINTS['north'][1]], dtype=np.float64)
    bearing = gps_to_bearing(paris_lat, paris_lon, north_lat, north_lon)
    assert np.isclose(bearing[0], 0.0, atol=5.0)  # Within 5 degrees of north

    # Test East (should be close to 90 degrees)
    east_lat = np.array([CARDINAL_POINTS['east'][0]], dtype=np.float64)
    east_lon = np.array([CARDINAL_POINTS['east'][1]], dtype=np.float64)
    bearing = gps_to_bearing(paris_lat, paris_lon, east_lat, east_lon)
    assert np.isclose(bearing[0], 90.0, atol=5.0)  # Within 5 degrees of east

    # Test South (should be close to 180 degrees)
    south_lat = np.array([CARDINAL_POINTS['south'][0]], dtype=np.float64)
    south_lon = np.array([CARDINAL_POINTS['south'][1]], dtype=np.float64)
    bearing = gps_to_bearing(paris_lat, paris_lon, south_lat, south_lon)
    assert np.isclose(bearing[0], 180.0, atol=5.0)  # Within 5 degrees of south

    # Test West (should be close to 270 degrees)
    west_lat = np.array([CARDINAL_POINTS['west'][0]], dtype=np.float64)
    west_lon = np.array([CARDINAL_POINTS['west'][1]], dtype=np.float64)
    bearing = gps_to_bearing(paris_lat, paris_lon, west_lat, west_lon)
    assert np.isclose(bearing[0], 270.0, atol=5.0)  # Within 5 degrees of west


def test_gps_to_bearing_cities():
    """Test gps_to_bearing with known routes between French cities."""

    # Test Paris to Marseille (should be roughly southeast)
    paris_lat = np.array([FRENCH_CITIES_GPS['Paris'][0]], dtype=np.float64)
    paris_lon = np.array([FRENCH_CITIES_GPS['Paris'][1]], dtype=np.float64)
    marseille_lat = np.array(
        [FRENCH_CITIES_GPS['Marseille'][0]], dtype=np.float64
    )
    marseille_lon = np.array(
        [FRENCH_CITIES_GPS['Marseille'][1]], dtype=np.float64
    )

    bearing = gps_to_bearing(
        paris_lat, paris_lon, marseille_lat, marseille_lon
    )
    # Paris to Marseille should be roughly southeast (135-180 degrees)
    assert np.isclose(bearing[0], 158.0, atol=1)

    # Test Paris to Lyon (should be roughly southeast)
    lyon_lat = np.array([FRENCH_CITIES_GPS['Lyon'][0]], dtype=np.float64)
    lyon_lon = np.array([FRENCH_CITIES_GPS['Lyon'][1]], dtype=np.float64)

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
        [
            FRENCH_CITIES_GPS['Paris'][0],
            FRENCH_CITIES_GPS['Marseille'][0],
            FRENCH_CITIES_GPS['Lyon'][0],
        ],
        dtype=np.float64,
    )
    lon_a = np.array(
        [
            FRENCH_CITIES_GPS['Paris'][1],
            FRENCH_CITIES_GPS['Marseille'][1],
            FRENCH_CITIES_GPS['Lyon'][1],
        ],
        dtype=np.float64,
    )
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


def test_support_distances_to_gps():
    """Test support_distances_to_gps with multiple support bases."""

    # Test case: 3 additional support bases from a first support base
    support_bases_x_meters = np.array(
        [50000.0, -30000.0, 100000.0], dtype=np.float64
    )  # East, West, East movements
    support_bases_y_meters = np.array(
        [100000.0, 0.0, -50000.0], dtype=np.float64
    )  # North, No change, South movements
    first_support_lat = 48.8566  # Paris latitude
    first_support_lon = 2.3522  # Paris longitude

    expected_lats = np.array(
        [48.8566, 49.7566009000009, 48.8566, 48.40659954999955],
        dtype=np.float64,
    )
    expected_lons = np.array(
        [2.3522, 3.0361475353187672, 1.9418314788087394, 3.7200950706375346],
        dtype=np.float64,
    )

    gps_coordinates = support_distances_to_gps(
        support_bases_x_meters,
        support_bases_y_meters,
        first_support_lat,
        first_support_lon,
    )

    # Check return type and shape
    assert isinstance(gps_coordinates, tuple)
    assert len(gps_coordinates) == 2
    assert isinstance(gps_coordinates[0], np.ndarray)
    assert isinstance(gps_coordinates[1], np.ndarray)

    # Should have 4 coordinates total (3 additional + 1 first support)
    assert gps_coordinates[0].shape == (4,)
    assert gps_coordinates[1].shape == (4,)
    assert gps_coordinates[0].dtype == np.float64
    assert gps_coordinates[1].dtype == np.float64

    np.testing.assert_array_equal(gps_coordinates[0], expected_lats)
    np.testing.assert_array_equal(gps_coordinates[1], expected_lons)
