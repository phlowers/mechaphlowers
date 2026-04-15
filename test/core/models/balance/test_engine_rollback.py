# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import patch

import numpy as np
import pytest

from mechaphlowers.api.section_study import SectionStudy
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.entities.errors import SolverError


class TestSolveAdjustmentRollback:
    def test_rollback_on_solver_error(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        engine = study.balance_engine
        dxdydz_before = engine.balance_model.nodes.dxdydz.copy()
        param_before = engine.span_model.sagging_parameter.copy()

        with patch.object(
            engine.solver_adjustment,
            "solve",
            side_effect=SolverError("mock failure"),
        ):
            with pytest.raises(SolverError):
                study.solve_adjustment()

        # State must be unchanged
        np.testing.assert_array_equal(
            engine.balance_model.nodes.dxdydz, dxdydz_before
        )
        np.testing.assert_array_equal(
            engine.span_model.sagging_parameter, param_before
        )


class TestSolveChangeStateRollback:
    def test_rollback_on_solver_error(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        engine = study.balance_engine
        study.solve_adjustment()

        dxdydz_before = engine.balance_model.nodes.dxdydz.copy()
        param_before = engine.span_model.sagging_parameter.copy()
        L_ref_before = engine.L_ref.copy()

        with patch.object(
            engine.solver_change_state,
            "solve",
            side_effect=SolverError("mock failure"),
        ):
            with pytest.raises(SolverError):
                study.solve_change_state(wind_pressure=200, new_temperature=90)

        # State must be unchanged
        np.testing.assert_array_equal(
            engine.balance_model.nodes.dxdydz, dxdydz_before
        )
        np.testing.assert_array_equal(
            engine.span_model.sagging_parameter, param_before
        )
        np.testing.assert_array_equal(engine.L_ref, L_ref_before)


class TestIntermediateState:
    def test_intermediate_memento_stored(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        study.solve_adjustment()

        # Solve with non-default params triggers intermediate step
        study.solve_change_state(wind_pressure=200, new_temperature=90)
        assert study.intermediate_memento is not None

    def test_default_params_skip_intermediate(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        study.solve_adjustment()

        # Solve with defaults — no intermediate
        study.solve_change_state()
        assert study.intermediate_memento is None
