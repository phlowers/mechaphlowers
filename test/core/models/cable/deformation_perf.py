# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import time

import numpy as np
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.core.models.cable.deformation import (
	DeformationRTE,
)
from mechaphlowers.core.models.cable.span import (
	CatenarySpan,
)


def test_solve_polynom_perf() -> None:
	spans_number = 10_000

	polynomial = Poly(
		[0, 1e9 * 100, 1e9 * -24_000, 1e9 * 2_440_000, 1e9 * -90_000_000]
	)
	input_dict = {
		"section": np.array([345.5] * spans_number),
		"diameter": np.array([22.4] * spans_number),
		"linear_weight": np.array([9.6] * spans_number),
		"young_modulus": np.array([59] * spans_number),
		"dilatation_coefficient": np.array([23] * spans_number),
		"temperature_reference": np.array([15] * spans_number),
		"polynomial_conductor": np.array([polynomial] * spans_number),
	}

	a = np.array([500] * spans_number)
	b = np.array([0.0] * spans_number)
	p = np.array([2_000] * spans_number)
	lambd = np.array([9.6] * spans_number)
	m = np.array([1] * spans_number)

	span_model = CatenarySpan(a, b, p, load_coefficient=m, linear_weight=lambd)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()
	deformation_model = DeformationRTE(
		**input_dict, tension_mean=tension_mean, cable_length=cable_length
	)

	start_time = time.time()

	deformation_model.max_stress = np.array([1e8] * (spans_number - 1) + [100])
	deformation_model.epsilon_mecha()
	exec_time = time.time() - start_time
	print(f"{spans_number} spans execution time : {exec_time}")


test_solve_polynom_perf()
