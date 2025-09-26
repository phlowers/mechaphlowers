# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
    WeatherArray,
)
from mechaphlowers.entities.data_container import (
    DataContainer,
    factory_data_container,
)


def test_data_container__factory(
    default_section_array_three_spans: SectionArray,
    default_cable_array: CableArray,
    generic_weather_array_three_spans: WeatherArray,
):
    data_container_new = factory_data_container(
        default_section_array_three_spans,
        default_cable_array,
        generic_weather_array_three_spans,
    )
    expected_result_poly = Poly(np.array([0, 59e9, 0, 0, 0]))

    expected_result_arrays = {
        "support_name": np.array(["support 1", "2", "three", "support 4"]),
        "suspension": np.array([False, True, True, False]),
        "conductor_attachment_altitude": np.array([2.2, 5, -0.12, 0]),
        "crossarm_length": np.array([10, 12.1, 10, 10.1]),
        "line_angle": np.array([0, 360, 90.1, -90.2]),
        "insulator_length": np.array([0, 4, 3.2, 0]),
        "insulator_weight": np.array([1000, 500, 500, 1000]),
        "span_length": np.array([400, 500.2, 500.0, np.nan]),
        "ice_thickness": 1e-2 * np.array([1, 2.1, 0.0, np.nan]),
        "wind_pressure": np.array([240.12, 0.0, 12.0, np.nan]),
    }
    expected_result_floats = {
        "cable_section_area": np.float64(345.55e-6),
        "diameter": np.float64(22.4e-3),
        "linear_weight": np.float64(9.55494),
        "young_modulus": np.float64(59e9),
        "dilatation_coefficient": np.float64(23e-6),
        "temperature_reference": np.float64(15),
    }

    assert isinstance(
        data_container_new.conductor_attachment_altitude, np.ndarray
    )
    assert isinstance(data_container_new.diameter, np.float64)
    assert isinstance(data_container_new.polynomial_conductor, Poly)
    assert isinstance(data_container_new.ice_thickness, np.ndarray)

    # tests are separated because mix of array for strings/arrays of floats/floats makes it difficult to test all in one go
    for attribute, value_array in expected_result_arrays.items():
        np.testing.assert_equal(
            data_container_new.__dict__[attribute], value_array
        )
    for attribute, value_float in expected_result_floats.items():
        np.testing.assert_allclose(
            data_container_new.__dict__[attribute], value_float, rtol=1e-5
        )
    assert data_container_new.polynomial_conductor == expected_result_poly


def test_data_container__add_arrays(
    default_section_array_three_spans: SectionArray,
    default_cable_array: CableArray,
    generic_weather_array_three_spans: WeatherArray,
):
    data_container = DataContainer()
    data_container.add_section_array(default_section_array_three_spans)
    data_container.add_cable_array(default_cable_array)
    data_container.add_weather_array(generic_weather_array_three_spans)


def test_update_from_dict(default_data_container_one_span) -> None:
    data = {
        "support_name": np.array(["Support1", "Support2"]),
        "suspension": np.array([10.5, 12.3]),
        "non_existing_attribute": "ignored",
    }
    default_data_container_one_span.update_from_dict(data)
    np.testing.assert_array_equal(
        default_data_container_one_span.support_name,
        np.array(["Support1", "Support2"]),
    )
    np.testing.assert_array_equal(
        default_data_container_one_span.suspension, np.array([10.5, 12.3])
    )
    assert not hasattr(
        default_data_container_one_span, "non_existing_attribute"
    )
