# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If span_length copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Tuple

import numpy as np
from numpy import ndarray

from mechaphlowers.core.tangential_sighting.tangential_sighting import (
    _lowest_point_coordinates,
)


def compute_parameter(
    span_length: np.ndarray,
    elevation_difference: np.ndarray,
    x1: np.ndarray,
    z1: np.ndarray,
    x2: np.ndarray,
    z2: np.ndarray,
    x3: np.ndarray,
    z3: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # All point coordinates are in span frame
    # We assume there is so little wind that the cable frame
    # is only span_length translation of the span frame.
    p1 = _compute_parameter_from_one_point(
        span_length, elevation_difference, x1, z1
    )
    p2 = _compute_parameter_from_one_point(
        span_length, elevation_difference, x2, z2
    )
    p3 = _compute_parameter_from_one_point(
        span_length, elevation_difference, x3, z3
    )
    p = p1 + p2 + p3

    p1 = np.round(p1, 0)
    p2 = np.round(p2, 0)
    p3 = np.round(p3, 0)
    p = np.round(p / 3, 0)
    return p1, p2, p3, p


def _compute_parameter_from_one_point(
    span_length: ndarray,
    elevation_difference: ndarray,
    measured_x: ndarray,
    measured_z: ndarray,
) -> ndarray:
    # Convention: x, z are coordinates in the span frame;
    # _x, _z are coordinates in the cable frame
    p_estimate = np.full_like(span_length, 1000)

    #
    def z(p: ndarray) -> ndarray:
        xm, zm = _lowest_point_coordinates(
            p, span_length, elevation_difference
        )

        _measured_x = measured_x - xm
        _measured_z = measured_z - zm

        _z_estimate = _catenary_equation_z(_measured_x, p)

        return _z_estimate + zm

    for _ in range(10):
        z_estimate = z(p_estimate)
        delta = z_estimate - measured_z
        z_estimate2 = z(p_estimate + 1)
        delta2 = z_estimate2 - measured_z
        derive = delta2 - delta
        p_estimate = p_estimate - delta / derive
    return p_estimate


def _catenary_equation_z(
    _x: ndarray,
    parameter: ndarray,
) -> ndarray:
    # In cable frame
    return parameter * (np.cosh(_x / parameter) - 1)
