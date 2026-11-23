# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If span_length copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple

import numpy as np

from mechaphlowers.config import options
from mechaphlowers.core.models.cable.span import CatenarySpan
from mechaphlowers.core.numeric import BrentMethod
from mechaphlowers.entities.errors import (
    ConvergenceError,
    MechaphlowersValueError,
)


def compute_parameter(
    span_length: float,
    elevation_difference: float,
    x1: float,
    z1: float,
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
    Compute the cable parameter from 3 points of the cable, using the Pep method.

    (xi, zi) are the coordinates of each measured point. They must be in the span frame.
    (Typically, xi is positive and zi is negative.)

    This method is only valid if there was little to no wind during the measurements.

    The 3 measured point should be near the lowest point of the cable,
    for the computed parameter to be precise.
    This function will raise a UserWarning if some xi are too close to the supports
    See config.PepParameterConfig.forbidden_zone_percentage.

    Returns:
        A tuple with the 3 estimated parameters (one for each measured point),
        their mean, and the maximum relative error.
    """
    _check_x_is_inside_span_and_not_too_close_to_supports(
        span_length, x1, x2, x3
    )

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


def _check_x_is_inside_span_and_not_too_close_to_supports(span_length, *args):
    for i, x in enumerate(args):
        if not 0 < x < span_length:
            raise MechaphlowersValueError(
                "Measured point must be inside of span."
                f" Got {x=} for point n°{i}"
            )

    forbidden_zone = options.pep_parameter.forbidden_zone_percentage
    for i, x in enumerate(args):
        if (
            not span_length * forbidden_zone
            <= x
            <= span_length * (1 - forbidden_zone)
        ):
            raise UserWarning(
                f"PEP method: measured point n°{i} is too close to the "
                f"supports ({x=}). Returned parameter won't be precise."
            )


def _compute_parameter_from_one_point(
    span_length: float,
    elevation_difference: float,
    measured_x: float,
    measured_z: float,
    bounds: tuple[float, float] = (0, 4000.0),
) -> float:
    # We assume there is so little wind that the cable frame
    # is only a span_length translation away from the span frame.

    # Convention: x, z are coordinates in the span frame;
    # _x, _z are coordinates in the cable frame

    def z(x, parameter: float) -> float:
        # Compute the coordinates of the left hanging point
        # in the cable frame, in order to switch between span and cable frames.
        span_model = CatenarySpan(
            span_length=np.array([span_length]),
            elevation_difference=np.array([elevation_difference]),
            parameter=np.array([parameter]),
        )
        x_m_in_cable_frame = span_model._x_m  # TODO: define public accessor
        z_m_in_cable_frame = span_model.z_one_point(x_m_in_cable_frame)

        x_in_cable_frame = x + x_m_in_cable_frame
        z_in_cable_frame = span_model.z_one_point(x_in_cable_frame)
        return (z_in_cable_frame - z_m_in_cable_frame).item()

    def f(parameter):
        return z(measured_x, parameter) - measured_z

    solver_method = BrentMethod()

    result = solver_method.solve(f, bounds)
    if not result.converged:
        raise ConvergenceError(
            message="pep parameter computation failed to converged",
            details=f"inputs:\n"
            f"\t{span_length=},\n{elevation_difference=},\n"
            f"{measured_x=},\n\t{measured_z=},\n"
            f"\t{bounds=}",
        )
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


def _relative_error(value1: float, value2: float) -> float:
    return abs((value1 - value2) / value2)
