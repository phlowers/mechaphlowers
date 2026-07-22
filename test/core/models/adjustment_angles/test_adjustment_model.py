# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np

from mechaphlowers.core.models.adjustment_angles.adjustment_model import (
    compute_adjustment_angles,
)
from mechaphlowers.data.units import Q_


class TestAdjustmentAnglesResults:
    def test_compute_adjustment_angles_0(self):
        a = 500
        HG = 0
        VG = 30
        HD = 90
        VD = 50
        horizontal_distance_support = 300
        parameter = 2000
        result = compute_adjustment_angles(
            a,
            Q_(HG, "grad").to("rad").magnitude,
            Q_(VG, "grad").to("rad").magnitude,
            Q_(HD, "grad").to("rad").magnitude,
            Q_(VD, "grad").to("rad").magnitude,
            horizontal_distance_support,
            parameter,
            "left",
        )
        np.testing.assert_allclose(result[0], 55.752, atol=1e-3)
        np.testing.assert_allclose(result[1], 33.214, atol=1e-3)

    def test_compute_adjustment_angles_1(self):
        a = 500
        HG = 0
        VG = 30
        HD = 90
        VD = 50
        horizontal_distance_support = 200
        parameter = 2000
        result = compute_adjustment_angles(
            a,
            Q_(HG, "grad").to("rad").magnitude,
            Q_(VG, "grad").to("rad").magnitude,
            Q_(HD, "grad").to("rad").magnitude,
            Q_(VD, "grad").to("rad").magnitude,
            horizontal_distance_support,
            parameter,
            "right",
        )
        np.testing.assert_allclose(result[0], 23.035, atol=1e-3)
        np.testing.assert_allclose(result[1], 29.75, atol=1e-3)
