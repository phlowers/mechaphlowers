# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.models.cable_models import CatenaryCableModel


def test_catenary_cable_model() -> None:
	a = np.array([501.3, 499.0])  # test here int and float
	b = np.array([0.0, -5.0])
	p = np.array([2_112.2, 2_112.0])
	lambd = np.array([16, 16.1])
	m = np.array([1, 1.1])

	cable_model = CatenaryCableModel(a, b, p, lambd, m)
	x = np.linspace(-223.2, 245.2, 250)

	assert isinstance(cable_model.z(x), np.ndarray)

	assert isinstance(cable_model.x_m(), np.ndarray)

	cable_model.x_m()  # check no error
	cable_model.x_n()
	cable_model.L_m()
	cable_model.L_n()
	cable_model.T_h()
	cable_model.T_v(x)
	cable_model.T_max(x)
	cable_model.T_mean_m()
	cable_model.T_mean_n()
	cable_model.T_mean()



def test_catenary_cable_model__x_m__if_no_elevation_difference() -> None:
	a = np.array([100])
	b = np.array([0])
	p = np.array([2_000])

	cable_model = CatenaryCableModel(a, b, p)
	assert abs(cable_model.x_m() + 50.0) < 0.01
	assert abs(49.37 + cable_model.z(np.array([-50.0])) - 50.0) < 0.01


def test_catenary_cable_model__z__two_spans() -> None:
	a = np.array([100.0, 100.0])
	b = np.array([0.0, 50.0])
	p = np.array([20.0, 1.0])

	x = np.linspace((-10.0, 6.0), (10.0, 10.0), 5)

	cable_model = CatenaryCableModel(a, b, p)

	cable_model.z(x)
	cable_model.x_m()


def test_catenary_cable_model__elevation_impact() -> None:
	a = np.array([100.0, 100.0])
	b = np.array([0.0, 50.0])
	p = np.array([200.0, 10.0])

	cable_model = CatenaryCableModel(a, b, p)
	x_cable = cable_model.x(5)

	z_cable = cable_model.z(x_cable)
	assert abs(z_cable[-1, 0] - z_cable[0, 0] - b[0]) < 0.01
	assert abs(z_cable[-1, 1] - z_cable[0, 1] - b[1]) < 0.01


def test_catenary_cable_model__length_impact() -> None:
	a = np.array([50.0, 500.0])
	b = np.array([10.0, 50.0])
	p = np.array([200.0, 100.0])

	cable_model = CatenaryCableModel(a, b, p)
	x_cable = cable_model.x(5)

	z_cable = cable_model.z(x_cable)
	assert abs(z_cable[-1, 0] - z_cable[0, 0] - b[0]) < 0.01
	assert abs(z_cable[-1, 1] - z_cable[0, 1] - b[1]) < 0.01


def test_catenary_cable_model__tensions__if_no_elevation_difference() -> None:
	a = np.array([100])
	b = np.array([0])
	p = np.array([2_000])
	lambd = np.array([5])
	m = np.array([2])

	x = np.array([50])

	cable_model = CatenaryCableModel(a, b, p, lambd, m)
	assert abs(cable_model.T_h() - 20_000) < 0.01
	assert abs(cable_model.T_v(x)[0] - 500.0521) < 0.01


def test_catenary_cable_model__tensions__two_spans() -> None:
	a = np.array([100.0, 100.0])
	b = np.array([0.0, 50.0])
	p = np.array([200.0, 10.0])

	x = np.linspace((-10.0, 6.0), (10.0, 10.0), 5)

	cable_model = CatenaryCableModel(a, b, p)

	cable_model.z(x)
	cable_model.x_m()
