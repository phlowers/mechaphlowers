# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import time

import numpy as np
from pandera.typing import pandas as pdt

from mechaphlowers.core.models.cable.deformation import (
	PolynomialDeformation,
)
from mechaphlowers.core.models.cable.span import (
	CatenarySpan,
)
from mechaphlowers.entities.arrays import CableArray
from mechaphlowers.entities.schemas import CableArrayInput


def test_solve_polynom_perf() -> None:
	spans_number = 10_000

	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		{
			"section": [345.5],
			"diameter": [22.4],
			"linear_weight": [9.6],
			"young_modulus": [59],
			"dilatation_coefficient": [23],
			"temperature_reference": [15],
			"a0": [0],
			"a1": [100],
			"a2": [-24_000],
			"a3": [2_440_000],
			"a4": [-90_000_000],
		}
	)

	cable_array = CableArray(
		input_df.loc[input_df.index.repeat(spans_number)].reset_index(
			drop=True
		)
	)

	a = np.array([500] * spans_number)
	b = np.array([0.0] * spans_number)
	p = np.array([2_000] * spans_number)
	lambd = np.array([9.6] * spans_number)
	m = np.array([1] * spans_number)

	span_model = CatenarySpan(a, b, p, load_coefficient=m, linear_weight=lambd)
	tension_mean = span_model.T_mean()
	polynomial_deformation_model = PolynomialDeformation(
		cable_array, tension_mean
	)

	start_time = time.time()

	polynomial_deformation_model.max_stress = np.array(
		[1e8] * (spans_number - 1) + [100]
	)
	polynomial_deformation_model.epsilon_mecha()
	exec_time = time.time() - start_time
	print(f"{spans_number} spans execution time : {exec_time}")


test_solve_polynom_perf()
