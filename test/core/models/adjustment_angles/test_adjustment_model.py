# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pytest

from mechaphlowers.core.models.adjustment_angles.adjustment_model import (
    compute_adjustment_angles,
)
from mechaphlowers.data.units import Q_


class TestAdjustmentAnglesResults:
    def test_compute_adjustment_angles_0(self):
        a = 500
        HL = Q_(0, "grad").to("rad").magnitude
        VL = Q_(30, "grad").to("rad").magnitude
        HR = Q_(90, "grad").to("rad").magnitude
        VR = Q_(50, "grad").to("rad").magnitude
        horizontal_distance_support = 300
        parameter = 2000
        result = compute_adjustment_angles(
            a,
            HL,
            VL,
            HR,
            VR,
            horizontal_distance_support,
            parameter,
            "left",
        )
        result_horizontal_angle = Q_(result[0], "rad").to("grad").magnitude
        result_vertical_angle = Q_(result[1], "rad").to("grad").magnitude
        np.testing.assert_allclose(result_horizontal_angle, 55.752, atol=1e-3)
        np.testing.assert_allclose(result_vertical_angle, 33.214, atol=1e-3)

    def test_compute_adjustment_angles_1(self):
        a = 500
        HL = Q_(0, "grad").to("rad").magnitude
        VL = Q_(30, "grad").to("rad").magnitude
        HR = Q_(90, "grad").to("rad").magnitude
        VR = Q_(50, "grad").to("rad").magnitude
        horizontal_distance_support = 200
        parameter = 2000
        result = compute_adjustment_angles(
            a,
            HL,
            VL,
            HR,
            VR,
            horizontal_distance_support,
            parameter,
            "right",
        )
        result_horizontal_angle = Q_(result[0], "rad").to("grad").magnitude
        result_vertical_angle = Q_(result[1], "rad").to("grad").magnitude
        np.testing.assert_allclose(result_horizontal_angle, 23.035, atol=1e-3)
        np.testing.assert_allclose(result_vertical_angle, 29.75, atol=1e-3)

    def test_compute_adjustment_angles_bad_input_0(self):
        a = 500
        HL = Q_(90, "grad").to("rad").magnitude
        VL = Q_(30, "grad").to("rad").magnitude
        HR = Q_(90, "grad").to("rad").magnitude
        VR = Q_(50, "grad").to("rad").magnitude
        horizontal_distance_support = 200
        parameter = 2000
        with pytest.raises(ValueError):
            compute_adjustment_angles(
                a,
                HL,
                VL,
                HR,
                VR,
                horizontal_distance_support,
                parameter,
                "right",
            )

    def test_compute_adjustment_angles_bad_input_side(self):
        a = 500
        HL = Q_(0, "grad").to("rad").magnitude
        VL = Q_(30, "grad").to("rad").magnitude
        HR = Q_(90, "grad").to("rad").magnitude
        VR = Q_(50, "grad").to("rad").magnitude
        horizontal_distance_support = 300
        parameter = 2000
        with pytest.raises(ValueError):
            compute_adjustment_angles(
                a,
                HL,
                VL,
                HR,
                VR,
                horizontal_distance_support,
                parameter,
                "wrong_side",
            )

    def test_compute_adjustment_array_0(self):
        a = np.array([500])
        HL = np.array([0])
        VL = np.array([30])
        HR = np.array([90])
        VR = np.array([50])
        horizontal_distance_support = np.array([200])
        parameter = np.array([2000])
        result = compute_adjustment_angles(
            a,
            Q_(HL, "grad").to("rad").magnitude,
            Q_(VL, "grad").to("rad").magnitude,
            Q_(HR, "grad").to("rad").magnitude,
            Q_(VR, "grad").to("rad").magnitude,
            horizontal_distance_support,
            parameter,
            "right",
        )
        assert len(result[0]) == 1
        assert len(result[1]) == 1

    def test_compute_adjustment_array_1(self):
        a = np.array([500, 500])
        HL = np.array([0, 0])
        VL = np.array([30, 30])
        HR = np.array([90, 90])
        VR = np.array([50, 50])
        horizontal_distance_support = np.array([200, 200])
        parameter = np.array([2000, 2000])
        result = compute_adjustment_angles(
            a,
            Q_(HL, "grad").to("rad").magnitude,
            Q_(VL, "grad").to("rad").magnitude,
            Q_(HR, "grad").to("rad").magnitude,
            Q_(VR, "grad").to("rad").magnitude,
            horizontal_distance_support,
            parameter,
            "right",
        )
        assert len(result[0]) == 2
        assert len(result[1]) == 2
