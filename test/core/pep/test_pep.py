# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.pep.pep import compute_parameter


def test_compute_parameter() -> None:
    p1, p2, p3, p = compute_parameter(
        np.array([466.067, 485.144, 430.521]),
        np.array([-1.560, 7.023, -0.025]),
        np.array([198.332, 123.346, 108.903]),
        np.array([-13.288, -8.779, -8.270]),
        np.array([205.037, 202.985, 221.108]),
        np.array([-13.444, -10.631, -10.896]),
        np.array([298.816, 250.684, 284.145]),
        np.array([-12.879, -10.299, -9.803]),
    )
    np.testing.assert_allclose(p1, np.array([2105, 2115, 2121]))
    np.testing.assert_allclose(p2, np.array([2100, 2113, 2129]))
    np.testing.assert_allclose(p3, np.array([2106, 2113, 2127]))
    np.testing.assert_allclose(p, np.array([2104, 2113, 2126]))
