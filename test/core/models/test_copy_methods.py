# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from copy import copy

import numpy as np

from mechaphlowers.core.models.balance.engine import BalanceEngine


class TestCableLoadsCopy:
    def test_copy_produces_independent_arrays(
        self, balance_engine_base_test: BalanceEngine
    ):
        original = balance_engine_base_test.cable_loads
        original.wind_pressure = np.array([100.0, 200.0, 300.0, np.nan])
        original.ice_thickness = np.array([0.01, 0.02, 0.03, np.nan])

        copied = copy(original)

        # Mutate original — copy must not change
        original.wind_pressure[0] = 999.0
        original.ice_thickness[0] = 999.0

        assert copied.wind_pressure[0] != 999.0
        assert copied.ice_thickness[0] != 999.0

    def test_copy_preserves_scalars(
        self, balance_engine_base_test: BalanceEngine
    ):
        original = balance_engine_base_test.cable_loads
        copied = copy(original)

        assert copied.diameter == original.diameter
        assert copied.linear_weight == original.linear_weight
        assert copied.ice_density == original.ice_density


class TestCatenarySpanCopy:
    def test_copy_produces_independent_arrays(
        self, balance_engine_base_test: BalanceEngine
    ):
        original = balance_engine_base_test.span_model
        copied = copy(original)

        # The arrays should be different objects
        assert copied.parameter is not original.parameter
        assert copied.span_length is not original.span_length
        assert copied.elevation_difference is not original.elevation_difference

    def test_copy_recomputes_cache(
        self, balance_engine_base_test: BalanceEngine
    ):
        original = balance_engine_base_test.span_model
        original.compute_values()
        copied = copy(original)

        # Cache values should match (recomputed from same input arrays)
        np.testing.assert_array_almost_equal(copied.x_m, original.x_m)
        np.testing.assert_array_almost_equal(copied.x_n, original.x_n)
        np.testing.assert_array_almost_equal(copied.L, original.L)

    def test_copy_preserves_scalar(
        self, balance_engine_base_test: BalanceEngine
    ):
        original = balance_engine_base_test.span_model
        copied = copy(original)
        assert copied.linear_weight == original.linear_weight


class TestDeformationRteCopy:
    def test_copy_produces_independent_arrays(
        self, balance_engine_base_test: BalanceEngine
    ):
        original = balance_engine_base_test.deformation_model
        copied = copy(original)

        # The arrays should be different objects
        assert copied.current_temperature is not original.current_temperature
        assert copied.tension_mean is not original.tension_mean
        assert copied.cable_length is not original.cable_length

    def test_copy_preserves_scalars(
        self, balance_engine_base_test: BalanceEngine
    ):
        original = balance_engine_base_test.deformation_model
        copied = copy(original)

        assert copied.young_modulus == original.young_modulus
        assert copied.cable_section_area == original.cable_section_area
        assert copied.dilatation_coefficient == original.dilatation_coefficient


class TestNodesCopy:
    def test_copy_produces_independent_dxdydz(
        self, balance_engine_base_test: BalanceEngine
    ):
        balance_engine_base_test.solve_adjustment()
        original = balance_engine_base_test.balance_model.nodes
        copied = copy(original)

        old_dxdydz = copied.dxdydz.copy()
        original.dxdydz[0, 0] = 9999.0
        np.testing.assert_array_equal(copied.dxdydz, old_dxdydz)

    def test_copy_preserves_geometry(
        self, balance_engine_base_test: BalanceEngine
    ):
        original = balance_engine_base_test.balance_model.nodes
        copied = copy(original)

        np.testing.assert_array_equal(
            copied.insulator_length, original.insulator_length
        )
        np.testing.assert_array_equal(copied.line_angle, original.line_angle)
