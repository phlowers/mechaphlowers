# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

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
	np.testing.assert_allclose(
		L_ref,
		np.array([480.6392123, np.nan]),
		atol=1e-6,
	)
	# Z - narcisse
	# first: L0 = 480.659
	# CRA 50% L0 = 480.649
	# récup epsilon plutôt?


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
