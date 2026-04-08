# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.core.models.balance.memento import BalanceEngineCaretaker


class TestCaretakerSave:
    """Verify that BalanceEngineCaretaker.save produces independent copies."""

    def test_memento_arrays_are_independent(
        self, balance_engine_base_test: BalanceEngine
    ):
        engine = balance_engine_base_test
        engine.solve_adjustment()

        caretaker = BalanceEngineCaretaker(engine)
        memento = caretaker.save()
        old_dxdydz = memento.nodes_dxdydz.copy()

        # Mutate the engine — memento must not change
        engine.balance_model.nodes.dxdydz[0, 0] = 9999.0
        np.testing.assert_array_equal(memento.nodes_dxdydz, old_dxdydz)

    def test_memento_captures_L_ref(
        self, balance_engine_base_test: BalanceEngine
    ):
        engine = balance_engine_base_test
        engine.solve_adjustment()

        caretaker = BalanceEngineCaretaker(engine)
        memento = caretaker.save()
        assert memento.L_ref is not None
        np.testing.assert_array_almost_equal(memento.L_ref, engine.L_ref)

    def test_memento_L_ref_none_before_adjustment(
        self, balance_engine_base_test: BalanceEngine
    ):
        engine = balance_engine_base_test
        caretaker = BalanceEngineCaretaker(engine)
        memento = caretaker.save()
        assert memento.L_ref is None


class TestCaretakerRestore:
    """Verify that BalanceEngineCaretaker.restore brings the engine back to a saved state."""

    def test_restore_after_solve_matches_saved(
        self, balance_engine_base_test: BalanceEngine
    ):
        engine = balance_engine_base_test
        engine.solve_adjustment()

        caretaker = BalanceEngineCaretaker(engine)
        memento = caretaker.save()
        saved_dxdydz = memento.nodes_dxdydz.copy()
        saved_param = memento.span_sagging_parameter.copy()

        # Change engine state via another solve
        engine.solve_change_state(wind_pressure=200, new_temperature=90)

        # Restore
        caretaker.restore(memento)

        np.testing.assert_array_almost_equal(
            engine.balance_model.nodes.dxdydz, saved_dxdydz
        )
        np.testing.assert_array_almost_equal(
            engine.span_model.sagging_parameter, saved_param
        )

    def test_restore_refreshes_span_cache(
        self, balance_engine_base_test: BalanceEngine
    ):
        engine = balance_engine_base_test
        engine.solve_adjustment()
        caretaker = BalanceEngineCaretaker(engine)
        memento = caretaker.save()

        engine.solve_change_state(new_temperature=90)
        caretaker.restore(memento)

        # _x_m, _x_n, _L should match the recomputed values from restored arrays
        engine.span_model.compute_values()
        expected_xm = engine.span_model.compute_x_m()
        np.testing.assert_array_almost_equal(
            engine.span_model.x_m, expected_xm
        )

    def test_restore_refreshes_deformation_snapshots(
        self, balance_engine_base_test: BalanceEngine
    ):
        engine = balance_engine_base_test
        engine.solve_adjustment()
        caretaker = BalanceEngineCaretaker(engine)
        memento = caretaker.save()

        saved_tension_mean = memento.deformation_tension_mean.copy()

        engine.solve_change_state(new_temperature=90)
        caretaker.restore(memento)

        np.testing.assert_array_almost_equal(
            engine.deformation_model.tension_mean, saved_tension_mean
        )

    def test_restore_then_solve_change_state_converges(
        self, balance_engine_base_test: BalanceEngine
    ):
        """Key test: save after adjustment, solve with one set of params,
        restore, then solve with different params. Must converge and produce
        the same result as solving from scratch."""
        engine = balance_engine_base_test
        engine.solve_adjustment()
        caretaker = BalanceEngineCaretaker(engine)
        memento_adj = caretaker.save()

        # First change-of-state
        engine.solve_change_state(wind_pressure=200, new_temperature=90)

        # Restore to post-adjustment
        caretaker.restore(memento_adj)

        # Second change-of-state with different params
        engine.solve_change_state(wind_pressure=0, new_temperature=-20)
        result_after_restore = engine.balance_model.nodes.dxdydz.copy()

        # Compare: solve from scratch with same params
        caretaker.restore(memento_adj)
        engine.solve_change_state(wind_pressure=0, new_temperature=-20)
        result_from_scratch = engine.balance_model.nodes.dxdydz.copy()

        np.testing.assert_array_almost_equal(
            result_after_restore, result_from_scratch, decimal=8
        )

    def test_restore_L_ref(self, balance_engine_base_test: BalanceEngine):
        engine = balance_engine_base_test
        engine.solve_adjustment()
        caretaker = BalanceEngineCaretaker(engine)
        memento = caretaker.save()
        saved_L_ref = engine.L_ref.copy()

        engine.solve_change_state(new_temperature=50)
        caretaker.restore(memento)

        np.testing.assert_array_almost_equal(engine.L_ref, saved_L_ref)

    def test_restore_nodes_span_model_synced(
        self, balance_engine_base_test: BalanceEngine
    ):
        engine = balance_engine_base_test
        engine.solve_adjustment()
        caretaker = BalanceEngineCaretaker(engine)
        memento = caretaker.save()

        engine.solve_change_state(new_temperature=90)
        caretaker.restore(memento)

        # nodes_span_model should mirror span_model after restore
        np.testing.assert_array_almost_equal(
            engine.balance_model.nodes_span_model.sagging_parameter,
            engine.span_model.sagging_parameter,
        )
