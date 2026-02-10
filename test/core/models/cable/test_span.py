# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from mechaphlowers.core.models.cable.span import (
    CatenarySpan,
)
from mechaphlowers.entities.data_container import DataContainer


def test_catenary_span_model__no_error_lengths() -> None:
    a = np.array([501.3, 499.0])  # test here int and float
    b = np.array([0.0, -5.0])
    p = np.array([2_112.2, 2_112.0])

    span_model = CatenarySpan(a, b, p)
    x = np.linspace(-223.2, 245.2, 250)

    assert isinstance(span_model.z_many_points(x), np.ndarray)

    assert isinstance(span_model.compute_x_m(), np.ndarray)

    span_model.compute_x_m()  # check no error
    span_model.compute_x_n()
    span_model.L_m()
    span_model.L_n()


def test_catenary_span_model__no_errors_tensions() -> None:
    a = np.array([501.3, 499.0])  # test here int and float
    b = np.array([0.0, -5.0])
    p = np.array([2_112.2, 2_112.0])
    lambd = np.float64(16.0)

    span_model = CatenarySpan(a, b, p, linear_weight=lambd)
    x = np.array([100, 200.0])

    assert isinstance(span_model.z_many_points(x), np.ndarray)

    assert isinstance(span_model.compute_x_m(), np.ndarray)

    span_model.T_h()
    span_model.T_v(x)
    span_model.T(x)
    span_model.T_mean_m()
    span_model.T_mean_n()
    span_model.T_mean()


def test_catenary_span_model__x_m__if_no_elevation_difference() -> None:
    a = np.array([100])
    b = np.array([0])
    p = np.array([2_000])

    span_model = CatenarySpan(a, b, p)
    assert abs(span_model.compute_x_m() + 50.0) < 0.01
    assert (
        abs(49.37 + span_model.z_many_points(np.array([-50.0])) - 50.0) < 0.01
    )


def test_catenary_span_model__z__one_span() -> None:
    a = np.array([100.0, 100.0])
    b = np.array([0.0, 50.0])
    p = np.array([20.0, 1.0])

    x = np.linspace((-10.0, 6.0), (10.0, 10.0), 5)

    span_model = CatenarySpan(a, b, p)

    span_model.z_many_points(x)
    span_model.compute_x_m()


def test_catenary_span_model__elevation_impact() -> None:
    a = np.array([100.0, 100.0])
    b = np.array([0.0, 50.0])
    p = np.array([200.0, 10.0])

    span_model = CatenarySpan(a, b, p)
    x_cable = span_model.x(5)

    z_cable = span_model.z_many_points(x_cable)
    assert abs(z_cable[-1, 0] - z_cable[0, 0] - b[0]) < 0.01
    assert abs(z_cable[-1, 1] - z_cable[0, 1] - b[1]) < 0.01


def test_catenary_span_model__length_impact() -> None:
    a = np.array([50.0, 500.0])
    b = np.array([10.0, 50.0])
    p = np.array([200.0, 100.0])

    span_model = CatenarySpan(a, b, p)
    x_cable = span_model.x(5)

    z_cable = span_model.z_many_points(x_cable)
    assert abs(z_cable[-1, 0] - z_cable[0, 0] - b[0]) < 0.01
    assert abs(z_cable[-1, 1] - z_cable[0, 1] - b[1]) < 0.01


def test_catenary_span_model__no_linear_weight() -> None:
    a = np.array([100.0, 100.0])
    b = np.array([0.0, 50.0])
    p = np.array([200.0, 10.0])

    span_model = CatenarySpan(a, b, p)
    with pytest.raises(AttributeError):
        span_model.T_h()


def test_catenary_span_model__tensions__wrong_dimension_array() -> None:
    a = np.array([500, 500])
    b = np.array([0.0, 0.0])
    p = np.array([2_000, 2_000])
    k_load = np.array([1, 1])
    lambd = np.float64()

    span_model = CatenarySpan(
        a, b, p, load_coefficient=k_load, linear_weight=lambd
    )

    x = np.array([2, 3, 4])

    with pytest.raises(ValueError):
        span_model.T_v(x)
    with pytest.raises(ValueError):
        span_model.T(x)


def test_catenary_span_model__tensions__no_elevation_difference() -> None:
    a = np.array([500])
    b = np.array([0])
    p = np.array([2_000])
    lambd = np.float64(9.55494)
    k_load = np.array([1])

    span_model = CatenarySpan(
        a, b, p, load_coefficient=k_load, linear_weight=lambd
    )
    x_m = span_model.compute_x_m()

    # Data given by the prototype
    assert abs(span_model.T_h()[0] - 19109.88) < 0.01
    assert abs(span_model.T_v(x_m)[0] + 2394.96053) < 0.01
    assert abs(span_model.T_mean()[0] - 19159.78784541) < 0.01


def test_catenary_span_model__geometric_output() -> None:
    a = np.array([500])
    b = np.array([0])
    p = np.array([2_000])
    lambd = np.float64(9.55494)
    k_load = np.array([1])

    span_model = CatenarySpan(
        a, b, p, load_coefficient=k_load, linear_weight=lambd
    )

    assert (span_model.compute_x_m() + 250.0) < 0.01
    assert (span_model.compute_x_n() - 250.0) < 0.01

    assert (span_model.L_m() - 250.652) < 0.01
    assert (span_model.L_n() - 250.652) < 0.01

    assert (span_model.compute_L() - 501.303) < 0.01

    # TODO: check on a non symetrical case


def test_catenary_span_model__data_container(
    default_data_container_one_span: DataContainer,
) -> None:
    span_model = CatenarySpan(**default_data_container_one_span.__dict__)
    x = np.array([100, 200.0])

    span_model.compute_x_m()
    assert isinstance(span_model.z_many_points(x), np.ndarray)

    assert isinstance(span_model.compute_x_m(), np.ndarray)
    span_model.compute_x_n()
    span_model.L_m()
    span_model.L_n()
    span_model.T_h()
    span_model.T_v(x)
    span_model.T(x)
    span_model.T_mean_m()
    span_model.T_mean_n()
    span_model.T_mean()


def test_display_span_model__span_index() -> None:
    a = np.array([501.3, 150.0, 499.0])
    b = np.array([0.0, 10.0, -5.0])
    p = np.array([2_112.2, 1999.0, 2_112.0])
    span_index = np.array([10, 30, 20, 40])

    span_model = CatenarySpan(a, b, p, span_index=span_index)

    x = np.linspace(-223.2, 245.2, 250)

    assert isinstance(span_model.z_many_points(x), np.ndarray)

    assert isinstance(span_model.compute_x_m(), np.ndarray)

    span_index = np.array([10, 30, 20, 40, 50])
    span_model = CatenarySpan(a, b, p, span_index=span_index)
    span_model.get_coords(10)


def test_display_span_model__span_type() -> None:
    a = np.array([501.3, 150.0, 499.0])
    b = np.array([0.0, 10.0, -5.0])
    p = np.array([2_112.2, 1999.0, 2_112.0])
    span_type = np.array([1, 2, 0])

    span_model = CatenarySpan(a, b, p, span_type=span_type)

    span_model.get_coords(10)


def test_display_span_model__many_spans() -> None:
    a = np.array([200.0, 100.0, 501.3, 150.0, 50.0, 75.0, 300, 600.0, 499.0])
    b = np.array([0.0, 10.0, -5.0, 5.0, 0.0, 8.0, -7.0, 5.0, 0.0])
    p = np.array([2_000] * 9)
    span_type = np.array([1, 2, 0, 1, 2, 1, 2, 0, 0])

    span_model = CatenarySpan(a, b, p, span_type=span_type)

    span_model.get_coords(10)


def test_span_model__slope() -> None:
    a = np.array([500])
    b = np.array([0])
    p = np.array([2_000])
    lambd = np.float64(1.6)

    span_model = CatenarySpan(a, b, p, linear_weight=lambd)
    # there is a diff here because in the prototype z axis point to the down
    expected_slope_values = np.array([-7.1])  # Expected slope value in radians

    np.testing.assert_allclose(
        np.degrees(span_model.slope('left')), expected_slope_values, rtol=1e-1
    )


def test_span_model__slope_2_spans() -> None:
    a = np.array([500, 600])
    b = np.array([0, 10])
    p = np.array([2_000, 2_000])
    lambd = np.float64(1.6)

    span_model = CatenarySpan(a, b, p, linear_weight=lambd)
    expected_left_slope_values_degree = np.array([-7.1, -7.6])
    expected_right_slope_values_degree = np.array([7.1, 9.5])

    np.testing.assert_allclose(
        np.degrees(span_model.slope('left')),
        expected_left_slope_values_degree,
        rtol=1e-1,
    )
    np.testing.assert_allclose(
        np.degrees(span_model.slope('right')),
        expected_right_slope_values_degree,
        rtol=1e-1,
    )


# TODO: add test with a np.nan case
def test_copy_attributes_partial() -> None:
    a = np.array([501.3, 499.0])  # test here int and float
    b = np.array([0.0, -5.0])
    p = np.array([2_112.2, 2_112.0])
    span_model = CatenarySpan(a, b, p)
    copy_span_model = CatenarySpan(
        np.array([100]), np.array([1]), np.array([2000])
    )
    old_id = id(copy_span_model)
    copy_span_model.mirror(span_model)
    np.testing.assert_equal(copy_span_model.span_length, a)
    np.testing.assert_equal(copy_span_model.elevation_difference, b)
    np.testing.assert_equal(copy_span_model.sagging_parameter, p)
    assert old_id == id(copy_span_model)


def test_copy_attributes_full() -> None:
    a = np.array([501.3, 499.0])  # test here int and float
    b = np.array([0.0, -5.0])
    p = np.array([2_112.2, 2_112.0])
    k_load = np.array([1, 1])
    lambd = np.float64(16.0)
    span_index = np.array([0, 0])
    span_type = np.array([0, 1])

    span_model = CatenarySpan(
        a,
        b,
        p,
        k_load,
        linear_weight=lambd,
        span_index=span_index,
        span_type=span_type,
    )
    copy_span_model = CatenarySpan(
        np.array([100]), np.array([1]), np.array([2000])
    )
    old_id = id(copy_span_model)
    copy_span_model.mirror(span_model)
    np.testing.assert_equal(copy_span_model.span_length, a)
    np.testing.assert_equal(copy_span_model.elevation_difference, b)
    np.testing.assert_equal(copy_span_model.sagging_parameter, p)
    np.testing.assert_equal(copy_span_model.load_coefficient, k_load)
    np.testing.assert_equal(copy_span_model.span_index, span_index)
    np.testing.assert_equal(copy_span_model.span_type, span_type)
    assert old_id == id(copy_span_model)
