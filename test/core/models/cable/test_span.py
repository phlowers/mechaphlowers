# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from mechaphlowers.core.models.cable.span import (
	CatenarySpan,
)


def test_catenary_span_model__no_error_lengths() -> None:
	a = np.array([501.3, 499.0])  # test here int and float
	b = np.array([0.0, -5.0])
	p = np.array([2_112.2, 2_112.0])

	span_model = CatenarySpan(a, b, p)
	x = np.linspace(-223.2, 245.2, 250)

	assert isinstance(span_model.z(x), np.ndarray)

	assert isinstance(span_model.x_m(), np.ndarray)

	span_model.x_m()  # check no error
	span_model.x_n()
	span_model.L_m()
	span_model.L_n()


def test_catenary_span_model__no_errors_tensions() -> None:
	a = np.array([501.3, 499.0])  # test here int and float
	b = np.array([0.0, -5.0])
	p = np.array([2_112.2, 2_112.0])
	lambd = np.array([16, 16.1])

	span_model = CatenarySpan(a, b, p, linear_weight=lambd)
	x = np.array([100, 200.0])

	assert isinstance(span_model.z(x), np.ndarray)

	assert isinstance(span_model.x_m(), np.ndarray)

	span_model.T_h()
	span_model.T_v(x)
	span_model.T_max(x)
	span_model.T_mean_m()
	span_model.T_mean_n()
	span_model.T_mean()


def test_catenary_span_model__x_m__if_no_elevation_difference() -> None:
	a = np.array([100])
	b = np.array([0])
	p = np.array([2_000])

	span_model = CatenarySpan(a, b, p)
	assert abs(span_model.x_m() + 50.0) < 0.01
	assert abs(49.37 + span_model.z(np.array([-50.0])) - 50.0) < 0.01


def test_catenary_span_model__z__two_spans() -> None:
	a = np.array([100.0, 100.0])
	b = np.array([0.0, 50.0])
	p = np.array([20.0, 1.0])

	x = np.linspace((-10.0, 6.0), (10.0, 10.0), 5)

	span_model = CatenarySpan(a, b, p)

	span_model.z(x)
	span_model.x_m()


def test_catenary_span_model__elevation_impact() -> None:
	a = np.array([100.0, 100.0])
	b = np.array([0.0, 50.0])
	p = np.array([200.0, 10.0])

	span_model = CatenarySpan(a, b, p)
	x_cable = span_model.x(5)

	z_cable = span_model.z(x_cable)
	assert abs(z_cable[-1, 0] - z_cable[0, 0] - b[0]) < 0.01
	assert abs(z_cable[-1, 1] - z_cable[0, 1] - b[1]) < 0.01


def test_catenary_span_model__length_impact() -> None:
	a = np.array([50.0, 500.0])
	b = np.array([10.0, 50.0])
	p = np.array([200.0, 100.0])

	span_model = CatenarySpan(a, b, p)
	x_cable = span_model.x(5)

	z_cable = span_model.z(x_cable)
	assert abs(z_cable[-1, 0] - z_cable[0, 0] - b[0]) < 0.01
	assert abs(z_cable[-1, 1] - z_cable[0, 1] - b[1]) < 0.01


def test_catenary_span_model__no_linear_weight() -> None:
	a = np.array([100.0, 100.0])
	b = np.array([0.0, 50.0])
	p = np.array([200.0, 10.0])

	span_model = CatenarySpan(a, b, p)
	with pytest.raises(AttributeError):
		span_model.T_h()


def test_catenary_span_model__tensions__wrong_dimension_array() -> None:
	a = np.array([500, 500])
	b = np.array([0.0, 0.0])
	p = np.array([2_000, 2_000])
	m = np.array([1, 1])
	lambd = np.array([9.6, 9.6])

	span_model = CatenarySpan(a, b, p, load_coefficient=m, linear_weight=lambd)

	x = np.array([2, 3, 4])

	with pytest.raises(ValueError):
		span_model.T_v(x)
	with pytest.raises(ValueError):
		span_model.T_max(x)


def test_catenary_span_model__tensions__no_elevation_difference() -> None:
	a = np.array([500])
	b = np.array([0])
	p = np.array([2_000])
	lambd = np.array([9.55494])
	m = np.array([1])

	span_model = CatenarySpan(a, b, p, load_coefficient=m, linear_weight=lambd)
	x_m = span_model.x_m()

	# Data given by the prototype
	assert abs(span_model.T_h()[0] - 19109.88) < 0.01
	assert abs(span_model.T_v(x_m)[0] + 2394.96053) < 0.01
	assert abs(span_model.T_mean()[0] - 19159.78784541) < 0.01

def test_catenary_span_model__geometric_output():
    
	a = np.array([500])
	b = np.array([0])
	p = np.array([2_000])
	lambd = np.array([9.55494])
	m = np.array([1])

	span_model = CatenarySpan(a, b, p, load_coefficient=m, linear_weight=lambd)

	assert (span_model.x_m() + 250.0) < 0.01
	assert (span_model.x_n() - 250.0) < 0.01

	assert (span_model.L_m() - 250.652) < 0.01
	assert (span_model.L_n() - 250.652) < 0.01

	assert (span_model.L() - 501.303) < 0.01

	# TODO: check on a non symetrical case