# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If span_length copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple

from mechaphlowers.core.numeric import BrentMethod
from mechaphlowers.core.parameter_measures.utils import (
    _catenary_equation_z,
    _lowest_point_coordinates,
)


def compute_parameter(
    span_length: float,
    elevation_difference: float,
    x1: float,
    z1: float,  # TODO: rename? z -> y
    x2: float,
    z2: float,
    x3: float,
    z3: float,
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
]:
    """
    Compute the cable parameter from the coordinates of 3 measured points, using the Pep method.

    All coordinates must be in the span frame.
    (Typically, x1, x2 and x3 are positive and y1, y2 and y3 are negative.)

    This method is only valid if there was little to no wind during the measurements.
    The 3 measured points should be close to the lowest point of the cable.

    Returns:
        A tuple with the 3 estimated parameters (one for each measured point),
        their mean, and the maximum relative error.
    """
    # We assume there is so little wind that the cable frame
    # is only span_length translation of the span frame.
    p1 = _compute_parameter_from_one_point(
        span_length,
        elevation_difference,
        x1,
        z1,
    )
    p2 = _compute_parameter_from_one_point(
        span_length,
        elevation_difference,
        x2,
        z2,
    )
    p3 = _compute_parameter_from_one_point(
        span_length,
        elevation_difference,
        x3,
        z3,
    )
    _max_relative_error = max_relative_error(p1, p2, p3)
    p = (p1 + p2 + p3) / 3
    return p1, p2, p3, p, _max_relative_error


def _compute_parameter_from_one_point(
    span_length: float,
    elevation_difference: float,
    measured_x: float,
    measured_z: float,
    bounds: tuple[float, float] = (1, 3000.0),
) -> float:
    # Convention: x, z are coordinates in the span frame;
    # _x, _z are coordinates in the cable frame

    def z(x, parameter: float) -> float:
        xm, zm = _lowest_point_coordinates(
            parameter, span_length, elevation_difference
        )
        _x = x - xm
        _z = _catenary_equation_z(_x, parameter)
        return _z + zm

    def f(parameter):
        return z(measured_x, parameter) - measured_z

    solver_method = BrentMethod()

    result = solver_method.solve(f, bounds)
    assert result.converged
    # TODO: raise error or return EstimationResult?
    return result.value


def max_relative_error(
    p1: float,
    p2: float,
    p3: float,
) -> float:
    return max(
        [
            _relative_error(p1, p2),
            _relative_error(p2, p1),
            _relative_error(p3, p1),
            _relative_error(p1, p3),
            _relative_error(p3, p2),
            _relative_error(p2, p3),
        ],
    )


def _relative_error(x: float, y: float) -> float:
    return abs((x - y) / y)
