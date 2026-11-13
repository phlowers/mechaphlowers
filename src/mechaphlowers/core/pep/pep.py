# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Tuple

import numpy as np


def compute_parameter(
    a: np.ndarray,
    b: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    x3: np.ndarray,
    y3: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # calcul point 1
    p1 = np.full_like(a, 1000)

    for _ in range(10):
        xm = -a / 2 + p1 * np.asinh(b / (2 * p1 * np.sinh(a / (2 * p1))))
        ym = p1 * (np.cosh(xm / p1) - 1)

        x = x1 + xm
        y = p1 * (np.cosh(x / p1) - 1)

        delta = (y - ym) - y1

        p1 = p1 + 1

        xm = -a / 2 + p1 * np.asinh(b / (2 * p1 * np.sinh(a / (2 * p1))))
        ym = p1 * (np.cosh(xm / p1) - 1)

        x = x1 + xm
        y = p1 * (np.cosh(x / p1) - 1)

        derive = ((y - ym) - y1) - delta

        p1 = (p1 - 1) - delta / derive

    # calcul point 2
    p2 = np.full_like(a, 1000)

    for _ in range(10):
        xm = -a / 2 + p2 * np.asinh(b / (2 * p2 * np.sinh(a / (2 * p2))))
        ym = p2 * (np.cosh(xm / p2) - 1)

        x = x2 + xm
        y = p2 * (np.cosh(x / p2) - 1)

        delta = (y - ym) - y2

        p2 = p2 + 1

        xm = -a / 2 + p2 * np.asinh(b / (2 * p2 * np.sinh(a / (2 * p2))))
        ym = p2 * (np.cosh(xm / p2) - 1)

        x = x2 + xm
        y = p2 * (np.cosh(x / p2) - 1)

        derive = ((y - ym) - y2) - delta

        p2 = (p2 - 1) - delta / derive

    # calcul point 3
    p3 = np.full_like(a, 1346)

    for _ in range(10):
        xm = -a / 2 + p3 * np.asinh(b / (2 * p3 * np.sinh(a / (2 * p3))))
        ym = p3 * (np.cosh(xm / p3) - 1)

        x = x3 + xm
        y = p3 * (np.cosh(x / p3) - 1)

        delta = (y - ym) - y3

        p3 = p3 + 0.1

        xm = -a / 2 + p3 * np.asinh(b / (2 * p3 * np.sinh(a / (2 * p3))))
        ym = p3 * (np.cosh(xm / p3) - 1)

        x = x3 + xm
        y = p3 * (np.cosh(x / p3) - 1)

        derive = (((y - ym) - y3) - delta) / 0.1

        p3 = (p3 - 0.1) - delta / derive

    p = np.round((p1 + p2 + p3) / 3, 0)
    p1 = np.round(p1, 0)
    p2 = np.round(p2, 0)
    p3 = np.round(p3, 0)

    return p1, p2, p3, p
