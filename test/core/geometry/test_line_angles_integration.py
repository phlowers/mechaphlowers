# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.geometry.line_angles import (
    CablePlane,
)
from mechaphlowers.entities.arrays import SectionArray


@pytest.mark.parametrize(
    "section_array_dict,expected_a_prime,expected_b_prime",
    [
        (
            {
                "name": np.array(["one", "two", "three", "four"]),
                "suspension": np.array([False, True, True, False]),
                "conductor_attachment_altitude": np.array([30, 40, 60, 75]),
                "crossarm_length": np.array([40, 20, 30, 50]),
                "line_angle": np.array([0, -45, 9, -27]),
                "insulator_length": np.array([0, 3, 8, 0]),
                "span_length": np.array([500, 460, 520, np.nan]),
            },
            np.array([508.10969425, 465.44026071, 529.64910093, np.nan]),
            np.array([10, 20, 15, np.nan]),
        ),
        (
            {
                "name": np.array(["one", "two", "three", "four"]),
                "suspension": np.array([False, True, True, False]),
                "conductor_attachment_altitude": np.array([30, 40, 60, 75]),
                "crossarm_length": np.array([-40, -20, -30, -50]),
                "line_angle": np.array([0, -18, 9, -27]),
                "insulator_length": np.array([0, 3, 8, 0]),
                "span_length": np.array([500, 460, 520, np.nan]),
            },
            np.array([497.28363069, 459.33732276, 511.02416757, np.nan]),
            np.array([10, 20, 15, np.nan]),
        ),
        (
            {
                "name": np.array(["one", "two", "three", "four"]),
                "suspension": np.array([False, True, True, False]),
                "conductor_attachment_altitude": np.array([30, 40, 60, 75]),
                "crossarm_length": np.array([-40, -20, -30, -50]),
                "line_angle": np.array([-13.5, 18, -9, 27]),
                "insulator_length": np.array([0, 3, 8, 0]),
                "span_length": np.array([500, 460, 520, np.nan]),
            },
            np.array([498.82705114, 460.88677819, 529.64910093, np.nan]),
            np.array([10, 20, 15, np.nan]),
        ),
    ],
)
def test_span_lengths_values(
    section_array_dict: dict,
    expected_a_prime: np.ndarray,
    expected_b_prime: np.ndarray,
):
    section_array = SectionArray(pd.DataFrame(section_array_dict))
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    span_length = section_array.data.span_length.to_numpy()
    line_angle = section_array.data.line_angle.to_numpy()
    conductor_attachment_altitude = (
        section_array.data.conductor_attachment_altitude.to_numpy()
    )
    crossarm_length = section_array.data.crossarm_length.to_numpy()
    insulator_length = section_array.data.insulator_length.to_numpy()

    cable_plane = CablePlane(
        span_length=span_length,
        conductor_attachment_altitude=conductor_attachment_altitude,
        crossarm_length=crossarm_length,
        insulator_length=insulator_length,
        line_angle=line_angle,
    )

    a_prime = cable_plane.a_prime
    b_prime = cable_plane.b_prime
    np.testing.assert_allclose(a_prime, expected_a_prime, rtol=0, atol=1e-6)
    np.testing.assert_allclose(b_prime, expected_b_prime)
