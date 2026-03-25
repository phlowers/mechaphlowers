# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from copy import copy

import numpy as np
import pandas as pd
import pandera as pa
import pytest
from numpy.testing import assert_allclose, assert_equal
from pandas.testing import assert_frame_equal

from mechaphlowers.config import options
from mechaphlowers.data.geography.helpers import gps_to_lambert93
from mechaphlowers.entities.arrays import (
    SectionArray,
)
from mechaphlowers.entities.errors import DataWarning


@pytest.fixture
def section_array_input_data() -> dict[str, list]:
    return {
        "name": ["support 1", "2", "three", "support 4"],
        "suspension": [False, True, True, False],
        "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
        "crossarm_length": [10, 12.1, 10, 10.1],
        "line_angle": [0, 360, 90.1, -90.2],
        "insulator_length": [0.01, 4, 3.2, 0.01],
        "span_length": [1, 500.2, 500.05, np.nan],
        "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
    }


@pytest.fixture
def section_array(section_array_input_data: dict[str, list]) -> SectionArray:
    df = pd.DataFrame(section_array_input_data)
    section_array = SectionArray(
        data=df, sagging_parameter=2_000, sagging_temperature=15
    )
    section_array.add_units({"line_angle": "deg"})
    return section_array


def test_create_section_array__with_floats(
    section_array_input_data: dict,
) -> None:
    input_df = pd.DataFrame(section_array_input_data)
    section_array = SectionArray(
        input_df, sagging_parameter=2_000, sagging_temperature=15
    )
    repr_section_array = section_array.__repr__()
    assert repr_section_array.startswith("SectionArray")
    assert_frame_equal(
        input_df, section_array._data, check_dtype=False, atol=1e-07
    )


def test_create_section_array__only_ints() -> None:
    input_df = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2, 5, 0, 0],
            "crossarm_length": [10, 12, 10, 10],
            "line_angle": [0, 360, 90, -90],
            "insulator_length": [0.01, 4, 3, 0.01],
            "span_length": [1, 500, 500, np.nan],
            "insulator_mass": np.array([1000.0, 500.0, 500.0, 1000.0]),
        }
    )
    section_array = SectionArray(
        input_df, sagging_parameter=2_000, sagging_temperature=15
    )

    assert_frame_equal(
        input_df, section_array._data, check_dtype=False, atol=1e-07
    )


def test_create_section_array__span_length_for_last_support(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["span_length"][-1] = 300
    input_df = pd.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


@pytest.mark.parametrize(
    "column",
    [
        "name",
        "suspension",
        "conductor_attachment_altitude",
        "crossarm_length",
        "line_angle",
        "insulator_length",
        "span_length",
        "insulator_mass",
    ],
)
def test_create_section_array__missing_column(
    section_array_input_data: dict, column: str
) -> None:
    del section_array_input_data[column]
    input_df = pd.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__extra_column(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["extra column"] = [0] * 4

    input_df = pd.DataFrame(section_array_input_data)

    section_array = SectionArray(
        input_df, sagging_parameter=2_000, sagging_temperature=15
    )

    assert "extra column" not in section_array._data.columns


@pytest.mark.parametrize(
    "column,value",
    [
        ("name", [1, 2, 3, 4]),
        ("suspension", [1, 2, 3, 4]),
        ("conductor_attachment_altitude", ["1,2"] * 4),
        ("crossarm_length", ["1,2"] * 4),
        ("line_angle", ["1,2"] * 4),
        ("insulator_length", ["1,2"] * 4),
        ("span_length", ["1,2"] * 4),
        ("insulator_mass", ["1,2"] * 4),
    ],
)
def test_create_section_array__wrong_type(
    section_array_input_data: dict, column: str, value
) -> None:
    section_array_input_data[column] = value
    input_df = pd.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_compute_elevation_difference() -> None:
    # I modify data to have more meaning here and understand results
    data = {
        "name": ["support 1", "2", "three", "support 4"],
        "suspension": [False, True, True, False],
        "conductor_attachment_altitude": [50.0, 40.0, 20.0, 10.0],
        "crossarm_length": [
            5.0,
        ]
        * 4,
        "line_angle": [
            0,
        ]
        * 4,
        "insulator_length": [0, 4.0, 3.0, 0],
        "span_length": [50, 100, 500, np.nan],
        "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
    }

    df = pd.DataFrame(data)

    section_array = SectionArray(
        df, sagging_parameter=2_000, sagging_temperature=15
    )

    elevation_difference = section_array.compute_elevation_difference()

    assert_allclose(
        elevation_difference, np.array([-10.0, -20.0, -10.0, np.nan])
    )


def test_section_array__data(section_array_input_data: dict) -> None:
    df = pd.DataFrame(section_array_input_data)
    section_array = SectionArray(
        data=df, sagging_parameter=2_000, sagging_temperature=15
    )
    section_array.add_units({"line_angle": "deg"})
    inner_data = section_array._data.copy()

    exported_data = section_array.data

    expected_data = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
            "crossarm_length": [10, 12.1, 10, 10.1],
            "line_angle": [0.0, 6.283185, 1.572542, -1.574287],
            "insulator_length": [0.01, 4, 3.2, 0.01],
            "span_length": [1, 500.2, 500.05, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
            "insulator_weight": [9810.0, 4905.0, 4905.0, 9810.0],
            "ground_altitude": [-27.8, -25.0, -30.12, -30.0],
            "elevation_difference": [2.8, -5.12, 0.12, np.nan],
            "sagging_parameter": [2_000.0, 2_000.0, 2_000.0, np.nan],
            "sagging_temperature": [15] * 4,
            "bundle_number": [1] * 4,
        },
    )

    assert_frame_equal(exported_data, expected_data, atol=1e-07)
    # section_array inner data shouldn't have been modified
    assert_frame_equal(section_array._data, inner_data)


def test_section_array__data_with_optional() -> None:
    df = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
            "crossarm_length": [10, 12.1, 10, 10.1],
            "line_angle": [0, 360, 90.1, -90.2],
            "insulator_length": [0.01, 4, 3.2, 0.01],
            "span_length": [1, 500.2, 500.05, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
            "load_mass": [500, 1000, 500, np.nan],
            "load_position": [0.2, 0.4, 0.6, np.nan],
            "ground_altitude": [0.0, 3.0, -1, 0],
        }
    )
    section_array = SectionArray(
        data=df, sagging_parameter=2_000, sagging_temperature=15
    )
    section_array.add_units({"line_angle": "deg"})

    inner_data = section_array._data.copy()

    exported_data = section_array.data

    expected_data = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
            "crossarm_length": [10, 12.1, 10, 10.1],
            "line_angle": [0.0, 6.283185, 1.572542, -1.574287],
            "insulator_length": [0.01, 4, 3.2, 0.01],
            "span_length": [1, 500.2, 500.05, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
            "insulator_weight": [9810.0, 4905.0, 4905.0, 9810.0],
            "elevation_difference": [2.8, -5.12, 0.12, np.nan],
            "ground_altitude": [0.0, 3.0, -1, 0],
            "sagging_parameter": [2_000.0, 2_000.0, 2_000.0, np.nan],
            "sagging_temperature": [15] * 4,
            "load_mass": [500, 1000, 500, np.nan],
            "load_weight": [4905.0, 9810.0, 4905.0, np.nan],
            "load_position": [0.2, 0.4, 0.6, np.nan],
            "bundle_number": [1] * 4,
        },
    )

    assert_frame_equal(
        exported_data, expected_data, atol=1e-07, check_like=True
    )
    # section_array inner data shouldn't have been modified
    assert_frame_equal(section_array._data, inner_data, check_like=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_section_array__wrong_ground_altitude() -> None:
    df = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
            "crossarm_length": [10, 12.1, 10, 10.1],
            "line_angle": [0, 360, 90.1, -90.2],
            "insulator_length": [0.01, 4, 3.2, 0.01],
            "span_length": [1, 500.2, 500.05, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
            "load_mass": [500, 1000, 500, np.nan],
            "load_position": [0.2, 0.4, 0.6, np.nan],
            "ground_altitude": [0.0, 7.0, -1, 5],
        }
    )
    section_array = SectionArray(
        data=df, sagging_parameter=2_000, sagging_temperature=15
    )
    section_array.add_units({"line_angle": "deg"})

    exported_data = section_array.data

    expected_data = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
            "crossarm_length": [10, 12.1, 10, 10.1],
            "line_angle": [0.0, 6.283185, 1.572542, -1.574287],
            "insulator_length": [0.01, 4, 3.2, 0.01],
            "span_length": [1, 500.2, 500.05, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
            "insulator_weight": [9810.0, 4905.0, 4905.0, 9810.0],
            "elevation_difference": [2.8, -5.12, 0.12, np.nan],
            "ground_altitude": [0.0, -25.0, -1.0, -30.0],
            "sagging_parameter": [2_000.0, 2_000.0, 2_000.0, np.nan],
            "sagging_temperature": [15] * 4,
            "load_mass": [500, 1000, 500, np.nan],
            "load_weight": [4905.0, 9810.0, 4905.0, np.nan],
            "load_position": [0.2, 0.4, 0.6, np.nan],
            "bundle_number": [1] * 4,
        },
    )

    assert_frame_equal(
        exported_data, expected_data, atol=1e-07, check_like=True
    )


def test_section_array__data_without_sagging_properties(
    section_array_input_data: dict,
) -> None:
    df = pd.DataFrame(section_array_input_data)

    section_array_without_temperature = SectionArray(
        data=df, sagging_parameter=2_000
    )
    np.testing.assert_allclose(
        section_array_without_temperature.data.sagging_temperature,
        options.data.sagging_temperature_default
        * np.ones(len(section_array_without_temperature.data)),
    )

    section_array_without_parameter = SectionArray(
        data=df, sagging_temperature=15
    )
    np.testing.assert_allclose(
        section_array_without_parameter.data.sagging_parameter,
        section_array_without_parameter.equivalent_span()
        * 5
        * np.array(
            [1] * (len(section_array_without_temperature.data) - 1) + [np.nan]
        ),
    )


def test_section_array__data_original(section_array_input_data: dict) -> None:
    df = pd.DataFrame(section_array_input_data)
    section_array = SectionArray(data=df)

    exported_data = section_array.data_original

    expected_data = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
            "crossarm_length": [10, 12.1, 10, 10.1],
            "line_angle": [0, 360, 90.1, -90.2],
            "insulator_length": [0.01, 4, 3.2, 0.01],
            "span_length": [1, 500.2, 500.05, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
        },
    )

    assert_frame_equal(exported_data, expected_data, atol=1e-07)


def test_section_array_to_gps():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["1", "2", "three", "4", "5"]),
                "suspension": np.array([False, True, True, True, False]),
                "conductor_attachment_altitude": np.array([20, 5, 10, 0, 0]),
                "crossarm_length": np.array([10, 12.1, 10, 10.1, 5]),
                "line_angle": np.array([90, 90, 90, 90, 90]),
                "insulator_length": np.array([0, 4, 3.2, 0, 0]),
                "span_length": np.array([500, 500, 500.0, 500.0, np.nan]),
                "insulator_mass": np.array(
                    [1000.0, 500.0, 500.0, 500.0, 1000.0]
                ),
            }
        )
    )
    section_array.set_starting_gps(
        latitude_0=48.8566,
        longitude_0=2.3522,
        azimuth_0=45,
    )
    latitude, longitude = section_array.get_gps()
    assert latitude.shape == (5,)
    assert longitude.shape == (5,)


def test_section_array_set_starting_lambert93():
    """set_starting_lambert93 converts coords to GPS and delegates correctly."""

    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["1", "2", "three", "4", "5"]),
                "suspension": np.array([False, True, True, True, False]),
                "conductor_attachment_altitude": np.array([20, 5, 10, 0, 0]),
                "crossarm_length": np.array([10, 12.1, 10, 10.1, 5]),
                "line_angle": np.array([90, 90, 90, 90, 90]),
                "insulator_length": np.array([0.1, 4, 3.2, 1, 0.1]),
                "span_length": np.array([500, 500, 500.0, 500.0, np.nan]),
                "insulator_mass": np.array(
                    [1000.0, 500.0, 500.0, 500.0, 1000.0]
                ),
            }
        )
    )
    start_lat, start_lon = 48.8566, 2.3522
    easting, northing = gps_to_lambert93(start_lat, start_lon)

    section_array.set_starting_lambert93(
        easting=float(easting),
        northing=float(northing),
        azimuth_0=45,
    )
    latitude, longitude = section_array.get_gps()

    # GPS origin set via Lambert93 should produce the same shape as via GPS
    assert latitude.shape == (5,)
    assert longitude.shape == (5,)

    # The first pylon should be very close to the original GPS origin
    # (round-trip Lambert93→GPS introduces negligible error)
    assert_allclose(latitude[0], start_lat, atol=1e-4)
    assert_allclose(longitude[0], start_lon, atol=1e-4)


def test_section_array_get_lambert93():
    """get_lambert93 returns Lambert 93 coordinates consistent with get_gps output."""

    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["1", "2", "three", "4", "5"]),
                "suspension": np.array([False, True, True, True, False]),
                "conductor_attachment_altitude": np.array([20, 5, 10, 0, 0]),
                "crossarm_length": np.array([10, 12.1, 10, 10.1, 5]),
                "line_angle": np.array([90, 90, 90, 90, 90]),
                "insulator_length": np.array([0, 4, 3.2, 0, 0]),
                "span_length": np.array([500, 500, 500.0, 500.0, np.nan]),
                "insulator_mass": np.array(
                    [1000.0, 500.0, 500.0, 500.0, 1000.0]
                ),
            }
        )
    )
    section_array.set_starting_gps(
        latitude_0=48.8566,
        longitude_0=2.3522,
        azimuth_0=45,
    )

    easting, northing = section_array.get_lambert93()
    assert easting.shape == (5,)
    assert northing.shape == (5,)

    # Values must match converting the GPS output directly
    lats, lons = section_array.get_gps()
    expected_easting, expected_northing = gps_to_lambert93(lats, lons)
    assert_allclose(easting, expected_easting, atol=1e-3)
    assert_allclose(northing, expected_northing, atol=1e-3)


def test_section_array_copy_preserves_geolocator():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["1", "2", "three", "4", "5"]),
                "suspension": np.array([False, True, True, True, False]),
                "conductor_attachment_altitude": np.array([20, 5, 10, 0, 0]),
                "crossarm_length": np.array([10, 12.1, 10, 10.1, 5]),
                "line_angle": np.array([90, 90, 90, 90, 90]),
                "insulator_length": np.array([0, 4, 3.2, 0, 0]),
                "span_length": np.array([500, 500, 500.0, 500.0, np.nan]),
                "insulator_mass": np.array(
                    [1000.0, 500.0, 500.0, 500.0, 1000.0]
                ),
            }
        )
    )
    section_array.set_starting_gps(
        latitude_0=48.8566,
        longitude_0=2.3522,
        azimuth_0=45,
    )

    section_copy = copy(section_array)
    lat_orig, lon_orig = section_array.get_gps()
    lat_copy, lon_copy = section_copy.get_gps()

    np.testing.assert_allclose(lat_copy, lat_orig, atol=1e-8)
    np.testing.assert_allclose(lon_copy, lon_orig, atol=1e-8)


def test_section_array_copy_geolocator_is_independent():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["1", "2", "three", "4", "5"]),
                "suspension": np.array([False, True, True, True, False]),
                "conductor_attachment_altitude": np.array([20, 5, 10, 0, 0]),
                "crossarm_length": np.array([10, 12.1, 10, 10.1, 5]),
                "line_angle": np.array([90, 90, 90, 90, 90]),
                "insulator_length": np.array([0, 4, 3.2, 0, 0]),
                "span_length": np.array([500, 500, 500.0, 500.0, np.nan]),
                "insulator_mass": np.array(
                    [1000.0, 500.0, 500.0, 500.0, 1000.0]
                ),
            }
        )
    )
    section_array.set_starting_gps(
        latitude_0=48.8566,
        longitude_0=2.3522,
        azimuth_0=45,
    )

    section_copy = copy(section_array)
    # Mutate the copy's geolocator
    section_copy.set_starting_gps(
        latitude_0=49.0,
        longitude_0=3.0,
        azimuth_0=90,
    )

    lat_orig, lon_orig = section_array.get_gps()
    lat_copy, lon_copy = section_copy.get_gps()

    assert not np.allclose(lat_orig, lat_copy)
    assert not np.allclose(lon_orig, lon_copy)


def test_equivalent_span(section_array) -> None:
    res = (
        section_array.data.span_length**3
    ).sum() / section_array.data.span_length.sum()

    np.testing.assert_allclose(section_array.equivalent_span() ** 2, res)


def test_correct_insulator_length(section_array: SectionArray) -> None:
    expected_lengths = np.array([0.01, 4.0, 3.2, 0.01])

    np.testing.assert_allclose(
        section_array.data.insulator_length, expected_lengths
    )
    np.testing.assert_allclose(
        section_array._data.insulator_length, expected_lengths
    )

    # test on .data property
    section_array._data.insulator_length = np.array([0.0, 4.0, 3.2, 0.0])
    np.testing.assert_allclose(
        section_array.data.insulator_length, expected_lengths
    )

    # nothing to correct
    section_array._data.insulator_length = np.array([0.01, 4.0, 3.2, 0.01])
    section_array.correct_insulator_length()
    np.testing.assert_allclose(
        section_array.data.insulator_length, expected_lengths
    )


def test_warning_on_insulator_length_correction(
    section_array: SectionArray,
) -> None:
    # Force invalid lengths and ensure they are corrected with a warning
    section_array._data.insulator_length = np.array([0.0, 4.0, 3.2, 0.0])
    with pytest.warns(DataWarning):
        section_array.correct_insulator_length()

    section_array._data.insulator_length = np.array([0.0, 4.0, 3.2, 0.0])
    with pytest.warns(DataWarning):
        section_array.data


def test_section_array__data_with_counterweight(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["counterweight_mass"] = [0, 1000, 2000, 0]
    df = pd.DataFrame(section_array_input_data)
    section_array = SectionArray(
        data=df, sagging_parameter=2_000, sagging_temperature=15
    )

    assert_equal(
        section_array.data.counterweight_mass.to_numpy(),
        np.array([0, 1000, 2000, 0]),
    )
    assert_equal(
        section_array.data.counterweight.to_numpy(),
        np.array([0, 9810.0, 19620.0, 0]),
    )


def test_section_array_angle_sense() -> None:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 30, 30, 30],
                "crossarm_length": [3, 4, 5, -6],
                "line_angle": [10, 15, -20, 25],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [100, 50, 50, 100],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "deg"})
    section_array.angles_sense = "clockwise"
    expected_line_angle = -np.radians([10, 15, -20, 25])
    np.testing.assert_allclose(
        section_array.data.line_angle, expected_line_angle
    )

    expected_crossarm_length = -np.array([3, 4, 5, -6])
    np.testing.assert_allclose(
        section_array.data.crossarm_length, expected_crossarm_length
    )


def test_create_section_bundle_number(
    section_array_input_data: dict,
) -> None:
    section_array = SectionArray(
        pd.DataFrame(section_array_input_data),
        sagging_parameter=2_000,
        sagging_temperature=15,
        bundle_number=2,
    )
    assert section_array.bundle_number == 2

    with pytest.raises(ValueError):
        section_array = SectionArray(
            pd.DataFrame(section_array_input_data),
            sagging_parameter=2_000,
            sagging_temperature=15,
            bundle_number=0,
        )
