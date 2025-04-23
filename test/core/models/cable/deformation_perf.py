# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import time
from typing import TypedDict

import numpy as np
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.core.models.cable.deformation import (
    DeformationRte,
)
from mechaphlowers.core.models.cable.span import (
    CatenarySpan,
)
from mechaphlowers.entities.data_container import DataCable


class DeformationInputDict(TypedDict, total=False):
    cable_section_area: np.float64
    linear_weight: np.float64
    young_modulus: np.float64
    dilatation_coefficient: np.float64
    temperature_reference: np.float64
    polynomial_conductor: Poly
    polynomial_heart: Poly


def test_solve_polynom_perf() -> None:
    spans_number = 10_000

    polynomial_conductor = Poly(
        [0, 1e9 * 100, 1e9 * -24_000, 1e9 * 2_440_000, 1e9 * -90_000_000]
    )
    polynomial_heart = Poly([0, 0, 0, 0, 0])
    input_dict: DeformationInputDict = {
        "cable_section_area": np.float64(345.5),
        "linear_weight": np.float64(9.6),
        "young_modulus": np.float64(59),
        "dilatation_coefficient": np.float64(23),
        "temperature_reference": np.float64(15),
        "polynomial_conductor": polynomial_conductor,
        "polynomial_heart": polynomial_heart,
    }

    a = np.array([500] * spans_number)
    b = np.array([0.0] * spans_number)
    p = np.array([2_000] * spans_number)
    lambd = np.float64(9.6)
    m = np.array([1] * spans_number)

    span_model = CatenarySpan(a, b, p, load_coefficient=m, linear_weight=lambd)
    tension_mean = span_model.T_mean()
    cable_length = span_model.L()
    deformation_model = DeformationRte(
        data_cable=DataCable(**input_dict),
        tension_mean=tension_mean,
        cable_length=cable_length,
    )

    start_time = time.time()

    deformation_model.max_stress = np.array([1e8] * (spans_number - 1) + [100])
    deformation_model.epsilon()
    exec_time = time.time() - start_time
    print(f"{spans_number} spans execution time : {exec_time}")


test_solve_polynom_perf()
