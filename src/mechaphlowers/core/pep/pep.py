# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Tuple

import numpy as np
from numpy import ndarray


def compute_parameter(
    a: np.ndarray,
    b: np.ndarray,
    x1: np.ndarray,
    z1: np.ndarray,
    x2: np.ndarray,
    z2: np.ndarray,
    x3: np.ndarray,
    z3: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # All point coordinates are in span frame
    # We assume there is so little wind that the cable frame
    # is only a translation of the span frame.
    p1 = _compute_parameter_from_one_point(a, b, x1, z1)
    p2 = _compute_parameter_from_one_point(a, b, x2, z2)
    p3 = _compute_parameter_from_one_point(a, b, x3, z3)
    p = p1 + p2 + p3

    p1 = np.round(p1, 0)
    p2 = np.round(p2, 0)
    p3 = np.round(p3, 0)
    p = np.round(p / 3, 0)
    return p1, p2, p3, p


def _compute_parameter_from_one_point(
    a: ndarray,
    b: ndarray,
    measured_x: ndarray,
    measured_z: ndarray,
) -> ndarray:
    p_estimate = np.full_like(a, 1000)
    for _ in range(10):
        # xm and zm the coordinates of the lowest point of the
        # cable in the span frame
        xm = a / 2 - p_estimate * np.asinh(
            b / (2 * p_estimate * np.sinh(a / (2 * p_estimate)))
        )
        zm = -p_estimate * (np.cosh(xm / p_estimate) - 1)

        x = measured_x - xm
        z = p_estimate * (np.cosh(x / p_estimate) - 1)

        delta = (z + zm) - measured_z

        # xm2 and zm2 the coordinates of the lowest point of the
        # cable in the span frame
        xm2 = a / 2 - (p_estimate + 1) * np.asinh(
            b / (2 * (p_estimate + 1) * np.sinh(a / (2 * (p_estimate + 1))))
        )
        zm2 = -(p_estimate + 1) * (np.cosh(xm2 / (p_estimate + 1)) - 1)

        x = measured_x - xm2
        z = (p_estimate + 1) * (np.cosh(x / (p_estimate + 1)) - 1)

        delta2 = (z + zm2) - measured_z

        derive = delta2 - delta

        p_estimate = p_estimate - delta / derive
    return p_estimate
