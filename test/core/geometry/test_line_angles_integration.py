# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd

from mechaphlowers.core.geometry.line_angles import (
    CablePlane,
)
from mechaphlowers.entities.arrays import SectionArray


def test_attachments_coords_0():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["support 1", "2", "three", "support 4"]),
                "suspension": np.array([False, True, True, False]),
                "conductor_attachment_altitude": np.array([30, 40, 60, 70]),
                "crossarm_length": np.array([40, 20, 30, 50]),
                "line_angle": np.array([0, -45, 9, -27]),
                "insulator_length": np.array([0, 0, 0, 0]),
                "span_length": np.array([500, 460, 520, np.nan]),
            }
        )
    )
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
    a_prime_expected = np.array(
        [508.10969425, 465.44026071, 529.64910093, np.nan]
    )
    np.testing.assert_allclose(a_prime, a_prime_expected)
