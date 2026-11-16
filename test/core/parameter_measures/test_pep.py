# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.parameter_measures.pep import compute_parameter
from mechaphlowers.core.parameter_measures.utils import (
    _catenary_equation_z,
    _lowest_point_coordinates,
)


def test_compute_parameter__array() -> None:
    p1, p2, p3, p, relative_error = compute_parameter(
        np.array([466.067, 485.144, 430.521]),
        np.array([-1.560, 7.023, -0.025]),
        np.array([198.332, 123.346, 108.903]),
        np.array([-13.288, -8.779, -8.270]),
        np.array([205.037, 202.985, 221.108]),
        np.array([-13.444, -10.631, -10.896]),
        np.array([298.816, 250.684, 284.145]),
        np.array([-12.879, -10.299, -9.803]),
    )
    np.testing.assert_allclose(p1, np.array([2105, 2115, 2121]), atol=1)
    np.testing.assert_allclose(p2, np.array([2100, 2113, 2129]), atol=1)
    np.testing.assert_allclose(p3, np.array([2106, 2113, 2127]), atol=1)
    np.testing.assert_allclose(p, np.array([2104, 2113, 2126]), atol=1)
    assert (relative_error < 0.05).all()


def test_compute_parameter__scalar() -> None:
    p1, p2, p3, p, relative_error = compute_parameter(
        466.067,
        -1.560,
        198.332,
        -13.288,
        205.037,
        -13.444,
        298.816,
        -12.879,
    )
    np.testing.assert_allclose(p1, 2105, atol=1)
    np.testing.assert_allclose(p2, 2100, atol=1)
    np.testing.assert_allclose(p3, 2106, atol=1)
    np.testing.assert_allclose(p, 2104, atol=1)
    assert relative_error < 0.05


def test_compute_parameter__scalar__parameter_far_from_initial_guess() -> None:
    # given
    span_length = 600
    elevation_difference = 0
    parameter = 500
    # Same convention as in pep.py: x, z are coordinates in the span frame;
    # _x, _z are coordinates in the cable frame
    xm, ym = _lowest_point_coordinates(
        parameter,
        span_length,
        elevation_difference,
    )
    _x1 = -30
    _x3 = 30
    x1 = xm + _x1
    x3 = xm + _x3
    _y1 = _catenary_equation_z(
        _x1,
        parameter,
    )
    _y3 = _catenary_equation_z(
        _x3,
        parameter,
    )
    y1 = _y1 + ym
    y3 = _y3 + ym

    # when
    _, _, _, p, _ = compute_parameter(
        span_length,
        elevation_difference,
        x1,
        y1,
        xm,
        ym,
        x3,
        y3,
    )

    # then
    np.testing.assert_allclose(p, parameter, atol=1)
