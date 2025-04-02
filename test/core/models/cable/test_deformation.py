# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.core.models.cable.deformation import DeformationRte
from mechaphlowers.core.models.cable.span import (
	CatenarySpan,
)
from mechaphlowers.entities.data_container import DataContainer


def test_deformation_impl(
	default_data_container_one_span: DataContainer,
) -> None:
	span_model = CatenarySpan(**default_data_container_one_span.__dict__)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()

	deformation_model = DeformationRte(
		**default_data_container_one_span.__dict__,
		tension_mean=tension_mean,
		cable_length=cable_length,
	)
	current_temperature = np.array([15, 15])
	deformation_model.epsilon_mecha()
	deformation_model.epsilon_therm(current_temperature)
	deformation_model.epsilon(current_temperature)
	deformation_model.L_ref(current_temperature)


def test_deformation_values__default_data(
	default_data_container_one_span: DataContainer,
) -> None:
	span_model = CatenarySpan(**default_data_container_one_span.__dict__)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()

	deformation_model = DeformationRte(
		**default_data_container_one_span.__dict__,
		tension_mean=tension_mean,
		cable_length=cable_length,
	)
	current_temperature = np.array([30, 30])

	eps_mecha = deformation_model.epsilon_mecha()
	eps_therm = deformation_model.epsilon_therm(current_temperature)
	L_ref = deformation_model.L_ref(current_temperature)

	# Data given by the prototype
	np.testing.assert_allclose(
		eps_mecha,
		np.array([0.00093978, np.nan]),
		atol=1e-6,
	)
	np.testing.assert_allclose(
		eps_therm,
		np.array([0.000345, 0.000345]),
		atol=1e-6,
	)
	# our method L_ref returns L_15 but proto returns L_0 so that's why 480.6392123 is not the displayed value if you are using proto
	np.testing.assert_allclose(
		L_ref,
		np.array([480.6392123, np.nan]),
		atol=1e-6,
	)


def test_poly_deformation__degree_three(
	default_data_container_one_span: DataContainer,
) -> None:
	new_poly = Poly([0, 1e9 * 50, 1e9 * -3_000, 1e9 * 44_000, 0])
	default_data_container_one_span.polynomial_conductor = new_poly

	span_model = CatenarySpan(**default_data_container_one_span.__dict__)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()

	deformation_model = DeformationRte(
		**default_data_container_one_span.__dict__,
		tension_mean=tension_mean,
		cable_length=cable_length,
	)
	current_temperature = np.array([15, 15])

	constraint = (
		tension_mean / default_data_container_one_span.cable_section_area
	)
	constraint = np.fmax(constraint, np.array([0, 0]))
	deformation_model.resolve_stress_strain_equation(
		constraint,
		default_data_container_one_span.polynomial_conductor,
	)
	deformation_model.epsilon_mecha()

	deformation_model.epsilon(current_temperature)


def test_poly_deformation__degree_four(
	default_data_container_one_span: DataContainer,
) -> None:
	new_poly = Poly(
		[0, 1e9 * 100, 1e9 * -24_000, 1e9 * 2_440_000, 1e9 * -90_000_000]
	)
	default_data_container_one_span.polynomial_conductor = new_poly

	span_model = CatenarySpan(**default_data_container_one_span.__dict__)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()

	deformation_model = DeformationRte(
		**default_data_container_one_span.__dict__,
		tension_mean=tension_mean,
		cable_length=cable_length,
	)
	current_temperature = np.array([15, 15])

	constraint = (
		tension_mean / default_data_container_one_span.cable_section_area
	)
	constraint = np.fmax(constraint, np.array([0, 0]))
	deformation_model.resolve_stress_strain_equation(
		constraint,
		default_data_container_one_span.polynomial_conductor,
	)
	deformation_model.epsilon_mecha()

	deformation_model.epsilon(current_temperature)


def test_poly_deformation__degree_four__with_max_stress(
	default_data_container_one_span: DataContainer,
) -> None:
	new_poly = Poly(
		[0, 1e9 * 100, 1e9 * -24_000, 1e9 * 2_440_000, 1e9 * -90_000_000]
	)
	default_data_container_one_span.polynomial_conductor = new_poly

	span_model = CatenarySpan(**default_data_container_one_span.__dict__)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()

	deformation_model = DeformationRte(
		**default_data_container_one_span.__dict__,
		tension_mean=tension_mean,
		cable_length=cable_length,
	)
	current_temperature = np.array([15, 15])

	constraint = (
		tension_mean / default_data_container_one_span.cable_section_area
	)
	constraint = np.fmax(constraint, np.array([0, 0]))
	deformation_model.max_stress = np.array([1000, 1e8])
	deformation_model.epsilon_mecha()
	deformation_model.epsilon(current_temperature)


def test_poly_deformation__no_solutions(
	default_data_container_one_span: DataContainer,
) -> None:
	new_poly = Poly(
		[0, 1e9 * 100, 1e9 * -24_000, 1e9 * 2_440_000, 1e9 * -90_000_000]
	)
	default_data_container_one_span.polynomial_conductor = new_poly

	span_model = CatenarySpan(**default_data_container_one_span.__dict__)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()

	deformation_model = DeformationRte(
		**default_data_container_one_span.__dict__,
		tension_mean=tension_mean,
		cable_length=cable_length,
	)

	deformation_model.max_stress = np.array([1000, 1e10])
	with pytest.raises(ValueError):
		deformation_model.epsilon_mecha()


def test_deformation__data_container(
	default_data_container_one_span: DataContainer,
) -> None:
	span_model = CatenarySpan(**default_data_container_one_span.__dict__)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()

	deformation_model = DeformationRte(
		**default_data_container_one_span.__dict__,
		tension_mean=tension_mean,
		cable_length=cable_length,
	)
	current_temperature = np.array([15, 15])
	deformation_model.epsilon_mecha()
	deformation_model.epsilon_therm(current_temperature)
	deformation_model.epsilon(current_temperature)
	deformation_model.L_ref(current_temperature)
