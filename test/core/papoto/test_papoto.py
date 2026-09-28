# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np

from mechaphlowers.core.papoto.papoto_model import (
    papoto_2_points,
    papoto_3_points,
    papoto_validity,
)
from mechaphlowers.data.units import Q_

NON_CONVERGENT_INPUTS = {
    "a": 100.0,
    "HL": 0.0,
    "VL": 0.001,
    "HR": 3.14,
    "VR": 3.13,
    "H1": 1.57,
    "V1": 0.001,
    "H2": 2.0,
    "V2": 3.13,
}
_NON_CONVERGENT_INPUT_THIRD_POINT = {"H3": 2.5, "V3": 3.13}


def _repeat(values: dict, n: int) -> dict:
    return {key: np.full_like(value, n) for key, value in values.items()}


def test_papoto_function_0() -> None:
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
        HL=Q_(HL, "grad").to("rad").magnitude,
        VL=Q_(VL, "grad").to("rad").magnitude,
        HR=Q_(HR, "grad").to("rad").magnitude,
        VR=Q_(VR, "grad").to("rad").magnitude,
        H1=Q_(H1, "grad").to("rad").magnitude,
        V1=Q_(V1, "grad").to("rad").magnitude,
        H2=Q_(H2, "grad").to("rad").magnitude,
        V2=Q_(V2, "grad").to("rad").magnitude,
    )
    np.testing.assert_allclose(parameter, np.array([2000, np.nan]), atol=1.0)


def test_papoto_function_3_points() -> None:
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
        HL=Q_(HL, "grad").to("rad").magnitude,
        VL=Q_(VL, "grad").to("rad").magnitude,
        HR=Q_(HR, "grad").to("rad").magnitude,
        VR=Q_(VR, "grad").to("rad").magnitude,
        H1=Q_(H1, "grad").to("rad").magnitude,
        V1=Q_(V1, "grad").to("rad").magnitude,
        H2=Q_(H2, "grad").to("rad").magnitude,
        V2=Q_(V2, "grad").to("rad").magnitude,
        H3=Q_(H3, "grad").to("rad").magnitude,
        V3=Q_(V3, "grad").to("rad").magnitude,
    )
    np.testing.assert_allclose(parameter, np.array([2000, np.nan]), atol=1.0)


def test_validity():
    parameter_1_2 = np.array([999.0, np.nan])
    parameter_2_3 = np.array([1000.0, np.nan])
    parameter_1_3 = np.array([1001.0, np.nan])
    validity = papoto_validity(parameter_1_2, parameter_2_3, parameter_1_3)
    np.testing.assert_allclose(validity, np.array([1e-03, np.nan]), atol=1e-5)


# Also tests with single element arrays inputs, because scipy handles them
# differently.


def test_validity_single_element_array() -> None:
    parameter_1_2 = np.array([999.0])
    parameter_2_3 = np.array([1000.0])
    parameter_1_3 = np.array([1001.0])
    validity = papoto_validity(parameter_1_2, parameter_2_3, parameter_1_3)
    np.testing.assert_allclose(validity, np.array([1e-03]), atol=1e-5)


def test_papoto_2_points_single_element_array() -> None:
    a = np.array([460.296269689673])
    HL = np.array([0.0])
    VL = np.array([95.3282575966228])
    HR = np.array([165.995104445107])
    VR = np.array([76.6693939496654])
    H1 = np.array([4.0612735796946])
    V1 = np.array([94.4241159093392])
    H2 = np.array([15.2623371798393])
    V2 = np.array([88.8639691159579])
    result = papoto_2_points(
        a=a,
        HL=Q_(HL, "grad").to("rad").magnitude,
        VL=Q_(VL, "grad").to("rad").magnitude,
        HR=Q_(HR, "grad").to("rad").magnitude,
        VR=Q_(VR, "grad").to("rad").magnitude,
        H1=Q_(H1, "grad").to("rad").magnitude,
        V1=Q_(V1, "grad").to("rad").magnitude,
        H2=Q_(H2, "grad").to("rad").magnitude,
        V2=Q_(V2, "grad").to("rad").magnitude,
    )
    assert len(result) == 1


def test_papoto_3_points_single_element_array_returns_array() -> None:
    a = np.array([498.565922913587])
    HL = np.array([0.0])
    VL = np.array([97.4327311161033])
    HR = np.array([162.614599621714])
    VR = np.array([88.6907631859419])
    H1 = np.array([5.1134354937127])
    V1 = np.array([98.4518011880176])
    H2 = np.array([19.6314054626454])
    V2 = np.array([97.6289296721015])
    H3 = np.array([97.1475339907774])
    V3 = np.array([87.9335010245142])
    result = papoto_3_points(
        a=a,
        HL=Q_(HL, "grad").to("rad").magnitude,
        VL=Q_(VL, "grad").to("rad").magnitude,
        HR=Q_(HR, "grad").to("rad").magnitude,
        VR=Q_(VR, "grad").to("rad").magnitude,
        H1=Q_(H1, "grad").to("rad").magnitude,
        V1=Q_(V1, "grad").to("rad").magnitude,
        H2=Q_(H2, "grad").to("rad").magnitude,
        V2=Q_(V2, "grad").to("rad").magnitude,
        H3=Q_(H3, "grad").to("rad").magnitude,
        V3=Q_(V3, "grad").to("rad").magnitude,
    )
    assert len(result) == 1
