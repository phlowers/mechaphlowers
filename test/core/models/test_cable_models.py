# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.models.cable_models import CatenaryCableModel


def test_catenary_cable_model() -> None:
    a = np.array([501.3, 499])  # test here int and float
    b = np.array([0, -5.0])
    p = np.array([2_112.2, 2_112])

    cable_model = CatenaryCableModel(a, b, p)
    x = np.linspace(-223.2, 245.2, 250)

    assert isinstance(cable_model.z(x), np.ndarray)

    cable_model.x_m()  # check no error

    cable_model.x_n()


def test_catenary_cable_model__x_m__if_no_elevation_difference() -> None:
    a = np.array([100])
    b = np.array([0])
    p = np.array([2_000])

    cable_model = CatenaryCableModel(a, b, p)
    assert abs(cable_model.x_m() + 50.0) < 0.01