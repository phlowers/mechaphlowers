# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.geometry.points import SectionPoints
from mechaphlowers.core.models.cable.span import CatenarySpan
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import SectionArray


def create_default_displacement_vector(
    insulator_length: np.ndarray,
) -> np.ndarray:
    displacement_vector = np.zeros((insulator_length.size, 3))
    displacement_vector[1:-1, 2] = -insulator_length[1:-1]
    displacement_vector[0, 0] = insulator_length[0]
    displacement_vector[-1:, 0] = -insulator_length[-1]
    return displacement_vector


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
                "insulator_weight": [1000.0, 500.0, 500.0, 1000.0],
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
                "insulator_weight": [1000.0, 500.0, 500.0, 1000.0],
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
                "insulator_weight": [1000.0, 500.0, 500.0, 1000.0],
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
    elevation_difference = section_array.data.elevation_difference.to_numpy()
    insulator_length = section_array.data.insulator_length.to_numpy()
    sagging_parameter = section_array.data.sagging_parameter.to_numpy()

    mock_span_model = CatenarySpan(
        span_length, elevation_difference, sagging_parameter
    )
    # arbitrary values, but unused
    mock_cable_loads = CableLoads(
        np.float64(1.0), np.float64(1.0), np.zeros(span_length.shape), np.zeros(span_length.shape)
    )

    def get_displacement():
        return create_default_displacement_vector(insulator_length)

    section_points = SectionPoints(
        section_array, mock_span_model, mock_cable_loads, get_displacement
    )

    a_chain = section_points.plane.a_chain
    b_chain = section_points.plane.b_chain
    np.testing.assert_allclose(a_chain, expected_a_prime, rtol=0, atol=1e-6)
    np.testing.assert_allclose(b_chain, expected_b_prime)
