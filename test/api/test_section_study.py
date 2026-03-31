# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.api.section_study import SectionStudy
from mechaphlowers.core.models.balance.engine import BalanceEngine


class TestSectionStudyLifecycle:
    def test_solve_and_access_results(
        self, balance_engine_base_test: BalanceEngine
    ):
        """Use SectionStudy for a full solve cycle."""
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        study.solve_adjustment()
        study.solve_change_state(wind_pressure=200, new_temperature=90)

        points = study.get_supports_points()
        assert points.shape[1] == 3  # xyz coordinates

        data = study.get_data_spans()
        assert "parameter" in data
        assert "T_h" in data

    def test_save_restore_through_facade(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        study.solve_adjustment()
        memento = study.save_state()

        study.solve_change_state(new_temperature=90)
        study.restore_state(memento)

        np.testing.assert_array_almost_equal(
            study.balance_engine.balance_model.nodes.dxdydz,
            memento.nodes_dxdydz,
        )

    def test_restore_notifies_position_engine(
        self, balance_engine_base_test: BalanceEngine
    ):
        """After restore_state, PositionEngine should have fresh data."""
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        study.solve_adjustment()
        memento = study.save_state()
        points_adj = study.get_supports_points().copy()

        study.solve_change_state(wind_pressure=200, new_temperature=90)
        study.restore_state(memento)
        points_restored = study.get_supports_points()

        np.testing.assert_array_almost_equal(points_adj, points_restored)


class TestSectionStudyLazyEngines:
    def test_plot_engine_not_created_eagerly(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        assert study._plot_engine is None

    def test_plot_engine_created_on_access(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        _ = study.plot_engine
        assert study._plot_engine is not None

    def test_thermal_engine_created_on_access(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        _ = study.thermal_engine
        assert study._thermal_engine is not None

    def test_guying_created_on_access(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        _ = study.guying
        assert study._guying is not None


class TestSectionStudySubEngines:
    def test_balance_engine_property(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        assert isinstance(study.balance_engine, BalanceEngine)

    def test_position_engine_observes_balance(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        # PositionEngine should be registered as observer
        assert study.position_engine in study.balance_engine._observers
