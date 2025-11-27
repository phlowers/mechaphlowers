# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd

from mechaphlowers.data.measures import param_15_deg
from mechaphlowers.entities.arrays import CableArray, SectionArray


def test_parameter_15_deg(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": [0, 10, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [100, 50, 50, 100],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        )
    )

    param_0 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=0
    )
    np.testing.assert_allclose(param_0, 2543.57334, atol=1e-1)

    param_1 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=1
    )
    np.testing.assert_allclose(param_1, 2573.74948, atol=1e-1)

    param_2 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=2
    )
    np.testing.assert_allclose(param_2, 2578.5492, atol=1e-1)


def test_parameter_15_deg_no_anchor(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": [0, 10, 0, 0],
                "insulator_length": [0.001, 3, 3, 0.001],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [00, 50, 50, 00],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        )
    )
    param_0 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=0
    )
    np.testing.assert_allclose(param_0, 2548.2168, atol=1e-1)

    param_1 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=1
    )
    np.testing.assert_allclose(param_1, 2578.4119, atol=1e-1)

    param_2 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=2
    )
    np.testing.assert_allclose(param_2, 2583.5351, atol=1e-1)


def test_parameter_15_deg_simple_example(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [50, 50, 50, 50],
                "crossarm_length": [0, 0, 0, 0],
                "line_angle": [0, 0, 0, 0],
                "insulator_length": [0.001, 3, 3, 0.001],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [00, 0, 0, 00],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        )
    )

    param_0 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=0
    )
    np.testing.assert_allclose(param_0, 2566.2089, atol=1e-1)

    param_1 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=1
    )
    np.testing.assert_allclose(param_1, 2596.6778, atol=1e-1)

    param_2 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=2
    )
    np.testing.assert_allclose(param_2, 2601.3522, atol=1e-1)


def test_parameter_15_deg_elevation_diff(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 0, 0, 0],
                "line_angle": [0, 0, 0, 0],
                "insulator_length": [0.001, 3, 3, 0.001],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [100, 50, 50, 100],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        )
    )

    param_0 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=0
    )
    np.testing.assert_allclose(param_0, 2565.5791, atol=1e-1)

    param_1 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=1
    )
    np.testing.assert_allclose(param_1, 2597.9831, atol=1e-1)

    param_2 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=2
    )
    np.testing.assert_allclose(param_2, 2603.1956, atol=1e-1)
