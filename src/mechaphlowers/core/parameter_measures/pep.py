# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If span_length copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple

import numpy as np
from numpy import ndarray

from mechaphlowers.config import options
from mechaphlowers.core.parameter_measures.utils import (
    _catenary_equation_z,
    _lowest_point_coordinates,
)
from mechaphlowers.numeric.newton import newton_solver_wrapper


def compute_parameter(
    span_length: np.ndarray,
    elevation_difference: np.ndarray,
    x1: np.ndarray,
    z1: np.ndarray,
    x2: np.ndarray,
    z2: np.ndarray,
    x3: np.ndarray,
    z3: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # All point coordinates are in span frame
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
    span_length: ndarray,
    elevation_difference: ndarray,
    measured_x: ndarray,
    measured_z: ndarray,
) -> ndarray:
    # Convention: x, z are coordinates in the span frame;
    # _x, _z are coordinates in the cable frame

    def z(x, parameter: ndarray) -> ndarray:
        xm, zm = _lowest_point_coordinates(
            parameter, span_length, elevation_difference
        )
        _x = x - xm
        _z = _catenary_equation_z(_x, parameter)
        return _z + zm

    def f(parameter):
        return z(measured_x, parameter) - measured_z

    # Use newton method to find parameter such that f(parameter) = 0

    def f_prime_approx(parameter):
        zeta = options.solver.pep_zeta
        return (f(parameter + zeta) - f(parameter)) / zeta

    p0 = np.full_like(span_length, 1000)

    root = newton_solver_wrapper(
        f,
        x0=p0,
        fprime=f_prime_approx,
        args=(),
        caller_name="pep",
    )
    return root


def max_relative_error(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> np.ndarray:
    return np.max(
        [
            _relative_error(p1, p2),
            _relative_error(p2, p1),
            _relative_error(p3, p1),
            _relative_error(p1, p3),
            _relative_error(p3, p2),
            _relative_error(p2, p3),
        ],
        axis=0,
    )


def _relative_error(x: ndarray, y: ndarray) -> ndarray:
    return np.abs((x - y) / y)
