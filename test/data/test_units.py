# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from mechaphlowers.data.units import unit


def test_grad_to_deg():
    arr = np.array([0, 90, 180, 270, 360])
    angles_deg = unit.Quantity(arr, "deg")
    angles_rad = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    angles_grad = np.array([0, 100, 200, 300, 400])
    np.testing.assert_allclose(angles_deg.to('rad').magnitude, angles_rad)
    np.testing.assert_allclose(angles_deg.to('grad').magnitude, angles_grad)