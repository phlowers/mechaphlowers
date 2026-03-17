# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.entities.errors import GpsNoDataAvailable
from mechaphlowers.entities.geography import (
    GeoLocator,
    get_dist_and_angles_from_gps,
    get_gps_from_arrays,
)


def test_section_array_to_gps_0():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["1", "2", "three", "4", "5"]),
                "suspension": np.array([False, True, True, True, False]),
                "conductor_attachment_altitude": np.array([20, 5, 10, 0, 0]),
                "crossarm_length": np.array([10, 12.1, 10, 10.1, 5]),
                "line_angle": np.array([90, 90, 90, 90, 90]),
                "insulator_length": np.array([1, 4, 3.2, 1, 1]),
                "span_length": np.array([500, 500, 500.0, 500.0, np.nan]),
                "insulator_mass": np.array(
                    [1000.0, 500.0, 500.0, 500.0, 1000.0]
                ),
            }
        )
    )
    section_array.add_units({"line_angle": "deg"})
    all_lats, all_lons = get_gps_from_arrays(
        start_lat_deg=48.8566,
        start_lon_deg=2.3522,
        azimuth_deg=0,
        line_angles_degrees=np.degrees(
            section_array.data.line_angle.to_numpy()
        ),
        span_length=section_array.data.span_length.to_numpy(),
    )

    # show_street_map(section_array, all_lats, all_lons)

    expected_lats = np.array(
        [48.8566, 48.86109661, 48.86109641, 48.8565998, 48.8565996]
    )
    expected_lons = np.array(
        [2.3522, 2.3522, 2.34536507, 2.34536507, 2.35219939]
    )
    np.testing.assert_allclose(all_lats, expected_lats, atol=1e-5)
    np.testing.assert_allclose(all_lons, expected_lons, atol=1e-5)


def test_section_array_to_gps_1():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(
                    ["support 1", "2", "three", "support 4", "5"]
                ),
                "suspension": np.array([False, True, True, True, False]),
                "conductor_attachment_altitude": np.array(
                    [2.2, 5, -0.12, 0, 0]
                ),
                "crossarm_length": np.array([10, 12.1, 10, 10.1, 5]),
                "line_angle": np.array([0, 10, 15, 20, 30]),
                "insulator_length": np.array([1, 4, 3.2, 1, 1]),
                "span_length": np.array([300, 400, 500.0, 600.0, np.nan]),
                "insulator_mass": np.array(
                    [1000.0, 500.0, 500.0, 500.0, 1000.0]
                ),
            }
        )
    )
    section_array.add_units({"line_angle": "deg"})
    all_lats, all_lons = get_gps_from_arrays(
        start_lat_deg=48.8566,
        start_lon_deg=2.3522,
        azimuth_deg=0,
        line_angles_degrees=np.degrees(
            section_array.data.line_angle.to_numpy()
        ),
        span_length=section_array.data.span_length.to_numpy(),
    )

    # show_street_map(section_array, all_lats, all_lons)

    expected_lats = np.array(
        [48.8566, 48.85929796, 48.8628406, 48.86691587, 48.87073122]
    )
    expected_lons = np.array(
        [2.3522, 2.3522, 2.35125047, 2.34836157, 2.34256082]
    )
    np.testing.assert_allclose(all_lats, expected_lats, atol=1e-5)
    np.testing.assert_allclose(all_lons, expected_lons, atol=1e-5)


def test_gps_to_section_array_1():
    lats = np.array(
        [48.8566, 48.85929796, 48.8628406, 48.86691587, 48.87073122]
    )

    lons = np.array([2.3522, 2.3522, 2.35125047, 2.34836157, 2.34256082])

    distances, angles = get_dist_and_angles_from_gps(lats, lons)
    np.testing.assert_allclose(
        distances, np.array([300, 400, 500, 600, np.nan]), atol=1e-3
    )
    np.testing.assert_allclose(angles, np.array([0, 10, 15, 20, 0]), atol=1e-3)


def test_section_array_to_gps_2():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(
                    ["support 1", "2", "three", "support 4", "5"]
                ),
                "suspension": np.array([False, True, True, True, False]),
                "conductor_attachment_altitude": np.array(
                    [2.2, 5, -0.12, 0, 0]
                ),
                "crossarm_length": np.array([10, 12.1, 10, 10.1, 5]),
                "line_angle": np.array([0, 10, -15, 20, 30]),
                "insulator_length": np.array([1, 4, 3.2, 1, 1]),
                "span_length": np.array([300, 400, 500.0, 600.0, np.nan]),
                "insulator_mass": np.array(
                    [1000.0, 500.0, 500.0, 500.0, 1000.0]
                ),
            }
        )
    )
    section_array.add_units({"line_angle": "deg"})
    all_lats, all_lons = get_gps_from_arrays(
        start_lat_deg=48.8566,  # in rads: 0.852708
        start_lon_deg=2.3522,  # in rads: 0.041053
        azimuth_deg=90,
        line_angles_degrees=np.degrees(
            section_array.data.line_angle.to_numpy()
        ),
        span_length=section_array.data.span_length.to_numpy(),
    )

    # show_street_map(section_array, all_lats, all_lons)

    expected_lats = np.array(
        [48.8566, 48.8565999273, 48.8559751397, 48.8563668445, 48.8549700038]
    )
    expected_lons = np.array(
        [2.3522, 2.3480994121, 2.3427150917, 2.3359068169, 2.3279853474]
    )
    np.testing.assert_allclose(all_lats, expected_lats, atol=1e-5)
    np.testing.assert_allclose(all_lons, expected_lons, atol=1e-5)


def test_gps_to_section_array_2():
    lats = np.array(
        [48.8566, 48.8565999273, 48.8559751397, 48.8563668445, 48.8549700038]
    )
    lons = np.array(
        [2.3522, 2.3480994121, 2.3427150917, 2.3359068169, 2.3279853474]
    )
    distances, angles = get_dist_and_angles_from_gps(lats, lons)
    np.testing.assert_allclose(
        distances, np.array([300, 400, 500, 600, np.nan]), atol=1e-3
    )
    np.testing.assert_allclose(
        angles, np.array([0, 10, -15, 20, 0]), atol=1e-3
    )


def test_section_array_to_gps_3():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(
                    ["support 1", "2", "three", "support 4", "5"]
                ),
                "suspension": np.array([False, True, True, True, False]),
                "conductor_attachment_altitude": np.array(
                    [2.2, 5, -0.12, 0, 0]
                ),
                "crossarm_length": np.array([10, 12.1, 10, 10.1, 5]),
                "line_angle": np.array([0, -10, 15, -20, 30]),
                "insulator_length": np.array([1, 4, 3.2, 1, 1]),
                "span_length": np.array([300, 400, 500.0, 600.0, np.nan]),
                "insulator_mass": np.array(
                    [1000.0, 500.0, 500.0, 500.0, 1000.0]
                ),
            }
        )
    )
    section_array.add_units({"line_angle": "deg"})
    all_lats, all_lons = get_gps_from_arrays(
        start_lat_deg=48.8566,
        start_lon_deg=2.3522,
        azimuth_deg=0,
        line_angles_degrees=np.degrees(
            section_array.data.line_angle.to_numpy()
        ),
        span_length=section_array.data.span_length.to_numpy(),
    )

    # show_street_map(section_array, all_lats, all_lons)

    expected_lats = np.array(
        [48.8566, 48.85929796, 48.8628406, 48.86732009, 48.87253214]
    )
    expected_lons = np.array(
        [2.3522, 2.3522, 2.35314953, 2.35255375, 2.35467705]
    )
    np.testing.assert_allclose(all_lats, expected_lats, atol=1e-5)
    np.testing.assert_allclose(all_lons, expected_lons, atol=1e-5)


def test_gps_to_section_array_3():
    lats = np.array(
        [48.8566, 48.85929796, 48.8628406, 48.86732009, 48.87253214]
    )

    lons = np.array([2.3522, 2.3522, 2.35314953, 2.35255375, 2.35467705])

    distances, angles = get_dist_and_angles_from_gps(lats, lons)
    np.testing.assert_allclose(
        distances, np.array([300, 400, 500, 600, np.nan]), atol=1e-3
    )
    np.testing.assert_allclose(
        angles, np.array([0, -10, 15, -20, 0]), atol=1e-3
    )


def test_get_dist_and_angles_from_gps_invalid_unit():
    """Test that ValueError is raised when unit_output_angles is invalid."""
    lats = np.array([48.8566, 48.8593, 48.8628])
    lons = np.array([2.3522, 2.3522, 2.3512])
    with pytest.raises(ValueError):
        get_dist_and_angles_from_gps(lats, lons, unit_output_angles="invalid")


def test_get_dist_and_angles_from_gps_unit_conversion_consistency():
    """Test that angle conversions between units are consistent."""
    lats = np.array([48.8566, 48.85929796, 48.8628406, 48.86691587])
    lons = np.array([2.3522, 2.3522, 2.35125047, 2.34836157])

    angles_deg = get_dist_and_angles_from_gps(
        lats, lons, unit_output_angles="deg"
    )[1]
    angles_rad = get_dist_and_angles_from_gps(
        lats, lons, unit_output_angles="rad"
    )[1]
    angles_grad = get_dist_and_angles_from_gps(
        lats, lons, unit_output_angles="grad"
    )[1]

    np.testing.assert_allclose(np.radians(angles_deg), angles_rad, atol=1e-9)
    np.testing.assert_allclose(angles_deg * 200 / 180, angles_grad, atol=1e-9)


def test_round_trip():
    # --- Forward: section geometry → GPS ---
    # Put 0 at first and last support, because inverse operation can't manage them
    line_angles = np.array([0.0, 10.0, -15.0, 20.0, 0])  # anticlockwise
    span_lengths = np.array([300.0, 400.0, 500.0, 600.0, np.nan])

    lats, lons = get_gps_from_arrays(
        start_lat_deg=48.8566,
        start_lon_deg=2.3522,
        azimuth_deg=90.0,  # first span heads West
        line_angles_degrees=line_angles,
        span_length=span_lengths,
    )

    # --- Inverse: GPS → section geometry ---
    recovered_distances, recovered_angles = get_dist_and_angles_from_gps(
        lats, lons
    )

    np.testing.assert_allclose(recovered_distances, span_lengths, atol=1e-3)
    np.testing.assert_allclose(recovered_angles, line_angles, atol=1e-3)


# --- GeoLocator tests ---

_LINE_ANGLES_DEG = np.array([0.0, 10.0, 15.0, 20.0, 0.0])
_SPAN_LENGTHS = np.array([300.0, 400.0, 500.0, 600.0, np.nan])
_START_LAT = 48.8566
_START_LON = 2.3522
_START_AZIMUTH = 0.0


def test_geolocator_gps_no_data_available():
    geolocator = GeoLocator()
    with pytest.raises(GpsNoDataAvailable):
        geolocator.get_gps(_LINE_ANGLES_DEG.copy(), _SPAN_LENGTHS.copy())


def test_geolocator_set_starting_gps():
    geolocator = GeoLocator()
    geolocator.set_starting_gps(_START_LAT, _START_LON, _START_AZIMUTH)

    lats, lons = geolocator.get_gps(
        _LINE_ANGLES_DEG.copy(), _SPAN_LENGTHS.copy()
    )

    expected_lats, expected_lons = get_gps_from_arrays(
        _START_LAT,
        _START_LON,
        _START_AZIMUTH,
        _LINE_ANGLES_DEG.copy(),
        _SPAN_LENGTHS.copy(),
    )
    np.testing.assert_allclose(lats, expected_lats, atol=1e-8)
    np.testing.assert_allclose(lons, expected_lons, atol=1e-8)


def test_geolocator_set_starting_lambert93():
    from mechaphlowers.data.geography.helpers import (
        gps_to_lambert93,
        lambert93_to_gps,
    )

    easting, northing = gps_to_lambert93(_START_LAT, _START_LON)
    lat_back, lon_back = lambert93_to_gps(easting, northing)

    geolocator = GeoLocator()
    geolocator.set_starting_lambert93(
        float(easting), float(northing), _START_AZIMUTH
    )

    lats, lons = geolocator.get_gps(
        _LINE_ANGLES_DEG.copy(), _SPAN_LENGTHS.copy()
    )

    expected_lats, expected_lons = get_gps_from_arrays(
        float(lat_back),
        float(lon_back),
        _START_AZIMUTH,
        _LINE_ANGLES_DEG.copy(),
        _SPAN_LENGTHS.copy(),
    )
    np.testing.assert_allclose(lats, expected_lats, atol=1e-6)
    np.testing.assert_allclose(lons, expected_lons, atol=1e-6)


def test_geolocator_get_lambert93():
    from mechaphlowers.data.geography.helpers import gps_to_lambert93

    geolocator = GeoLocator()
    geolocator.set_starting_gps(_START_LAT, _START_LON, _START_AZIMUTH)

    easting, northing = geolocator.get_lambert93(
        _LINE_ANGLES_DEG.copy(), _SPAN_LENGTHS.copy()
    )

    lats, lons = get_gps_from_arrays(
        _START_LAT,
        _START_LON,
        _START_AZIMUTH,
        _LINE_ANGLES_DEG.copy(),
        _SPAN_LENGTHS.copy(),
    )
    expected_easting, expected_northing = gps_to_lambert93(lats, lons)

    np.testing.assert_allclose(easting, expected_easting, atol=1e-3)
    np.testing.assert_allclose(northing, expected_northing, atol=1e-3)
