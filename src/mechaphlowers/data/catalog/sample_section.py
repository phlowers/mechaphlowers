# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np


def section_factory_sample_data(size_section: int = 5, seed: int = 1) -> dict:
    """Create sample data for a section DataFrame

    Args:
        size_section (int, optional): Number of sections to create. Defaults to 5.
        seed (int, optional): Random seed for reproducibility. Defaults to 1.

    Returns:
        dict: sample data for a section DataFrame
    """
    np.random.seed(seed)

    name = []
    suspension = []
    altitude = []
    crossarm_length = []
    line_angle = []
    insulator_length = []
    span_length = []
    insulator_mass = []

    for i in range(size_section):
        name.append(f"Section {i}")
        if i == 0 or i == size_section - 1:
            suspension.append(False)
            insulator_mass.append(1000.0)
        else:
            suspension.append(True)
            insulator_mass.append(500.0)
        altitude.append(np.random.uniform(20, 150))
        crossarm_length.append(
            float(np.random.choice(np.array([-1, 1])))
            * float(np.random.uniform(1, 10))
        )
        insulator_length.append(float(np.random.randint(2, 10)))

        if i == size_section - 1:
            span_length.append(np.nan)
        else:
            span_length.append(float(np.random.uniform(80, 1000)))
        line_angle.append(np.random.uniform(-30, 30))

    section_data = {
        "name": name,
        "suspension": suspension,
        "conductor_attachment_altitude": altitude,
        "crossarm_length": crossarm_length,
        "insulator_length": insulator_length,
        "span_length": span_length,
        "line_angle": line_angle,
        "insulator_mass": insulator_mass,
        "load_mass": [0] * (size_section - 1) + [np.nan],
        "load_position": [0] * (size_section - 1) + [np.nan],
    }
    return section_data
