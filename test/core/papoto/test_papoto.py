# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np

from mechaphlowers.core.papoto.papoto_model import (
    convert_grad_to_rad,
    papoto_2_points,
    papoto_3_points,
    papoto_validity,
)


def test_papoto_function_0():
    a = np.array([460.296269689673, np.nan])
    HL = np.array([0.0, np.nan])
    VL = np.array([95.3282575966228, np.nan])
    HR = np.array([165.995104445107, np.nan])
    VR = np.array([76.6693939496654, np.nan])
    H1 = np.array([4.0612735796946, np.nan])
    V1 = np.array([94.4241159093392, np.nan])
    H2 = np.array([15.2623371798393, np.nan])
    V2 = np.array([88.8639691159579, np.nan])
    parameter = papoto_2_points(
        a=a,
        HL=convert_grad_to_rad(HL),
        VL=convert_grad_to_rad(VL),
        HR=convert_grad_to_rad(HR),
        VR=convert_grad_to_rad(VR),
        H1=convert_grad_to_rad(H1),
        V1=convert_grad_to_rad(V1),
        H2=convert_grad_to_rad(H2),
        V2=convert_grad_to_rad(V2),
    )
    np.testing.assert_allclose(parameter, np.array([2000, np.nan]), atol=1.0)


def test_papoto_function_3_points():
    a = np.array([498.565922913587, np.nan])
    HL = np.array([0.0, np.nan])
    VL = np.array([97.4327311161033, np.nan])
    HR = np.array([162.614599621714, np.nan])
    VR = np.array([88.6907631859419, np.nan])
    H1 = np.array([5.1134354937127, np.nan])
    V1 = np.array([98.4518011880176, np.nan])
    H2 = np.array([19.6314054626454, np.nan])
    V2 = np.array([97.6289296721015, np.nan])
    H3 = np.array([97.1475339907774, np.nan])
    V3 = np.array([87.9335010245142, np.nan])
    parameter = papoto_3_points(
        a=a,
        HL=convert_grad_to_rad(HL),
        VL=convert_grad_to_rad(VL),
        HR=convert_grad_to_rad(HR),
        VR=convert_grad_to_rad(VR),
        H1=convert_grad_to_rad(H1),
        V1=convert_grad_to_rad(V1),
        H2=convert_grad_to_rad(H2),
        V2=convert_grad_to_rad(V2),
        H3=convert_grad_to_rad(H3),
        V3=convert_grad_to_rad(V3),
    )
    np.testing.assert_allclose(parameter, np.array([2000, np.nan]), atol=1.0)


def test_validity():
    parameter_1_2 = np.array([999.0, np.nan])
    parameter_2_3 = np.array([1000.0, np.nan])
    parameter_1_3 = np.array([1001.0, np.nan])
    validity = papoto_validity(parameter_1_2, parameter_2_3, parameter_1_3)
    np.testing.assert_allclose(validity, np.array([1e-03, np.nan]), atol=1e-5)
