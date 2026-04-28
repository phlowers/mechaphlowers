# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest
from plotly import graph_objects as go

from mechaphlowers.api.section_study import SectionStudy
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.plotting.plot import PlotEngine
from test.conftest import show_figures


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
            memento.dxdydz,
        )

    def test_save_restore_angles(self, cable_array_AM600: CableArray):
        section_array = SectionArray(
            pd.DataFrame(
                {
                    "name": ["1", "2", "3", "4"],
                    "suspension": [False, True, True, False],
                    "conductor_attachment_altitude": [50, 60, 40, 50],
                    "crossarm_length": [10, 10, 10, 10],
                    "line_angle": [0, 10, 15, 5],
                    "insulator_length": [3, 3, 3, 3],
                    "span_length": [500, 500, 500, np.nan],
                    "insulator_mass": [100.0, 50.0, 50.0, 100.0],
                    "load_mass": [0, 50, 50, 0],
                    "load_position": [0, 0.4, 0.6, 0],
                }
            ),
            sagging_parameter=2000,
            sagging_temperature=15,
            bundle_number=2,
        )
        section_array.add_units({"line_angle": "grad"})
        study = SectionStudy(
            cable_array=cable_array_AM600,
            section_array=section_array,
        )
        study.solve_adjustment()
        study.solve_change_state(wind_pressure=200)
        vhl_before_save = study.vhl_under_chain()
        memento = study.save_state()

        # test that using restore reverts correcty
        study.solve_change_state(wind_pressure=0, ice_thickness=2e-2)
        study.restore_state(memento)
        vhl_after_restore = study.vhl_under_chain()

        np.testing.assert_array_almost_equal(
            vhl_before_save.vhl_matrix.value(),
            vhl_after_restore.vhl_matrix.value(),
        )

        # test that solve_change_state in the same state will give same results
        study.solve_change_state(wind_pressure=200)
        vhl_after_change_state = study.vhl_under_chain()

        np.testing.assert_array_almost_equal(
            vhl_before_save.vhl_matrix.value(),
            vhl_after_change_state.vhl_matrix.value(),
        )

    @pytest.mark.skip(reason="Fix this later when refacto add_loads")
    def test_save_restore_loads(self, cable_array_AM600: CableArray):
        section_array = SectionArray(
            pd.DataFrame(
                {
                    "name": ["1", "2", "3", "4"],
                    "suspension": [False, True, True, False],
                    "conductor_attachment_altitude": [50, 60, 40, 50],
                    "crossarm_length": [10, 10, 10, 10],
                    "line_angle": [0, 10, 15, 5],
                    "insulator_length": [3, 3, 3, 3],
                    "span_length": [500, 500, 500, np.nan],
                    "insulator_mass": [100.0, 50.0, 50.0, 100.0],
                    "load_mass": [0, 0, 0, 0],
                    "load_position": [0, 0, 0, 0],
                }
            ),
            sagging_parameter=2000,
            sagging_temperature=15,
        )
        section_array.add_units({"line_angle": "grad"})
        study = SectionStudy(
            cable_array=cable_array_AM600,
            section_array=section_array,
        )
        study.solve_adjustment()
        study.solve_change_state()
        vhl_before_save = study.vhl_under_chain()
        memento = study.save_state()

        # test that using restore reverts correcty
        study.add_loads(
            load_position_distance=np.array([0, 300, 0, 0]),
            load_mass=np.array([0, 100, 0, 0]),
        )
        study.solve_change_state()
        study.restore_state(memento)
        study.solve_change_state()
        vhl_after_restore = study.vhl_under_chain()

        np.testing.assert_array_almost_equal(
            vhl_before_save.vhl_matrix.value(),
            vhl_after_restore.vhl_matrix.value(),
        )

    # TODO: test that changing state does not affect memento

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


class TestSectionStudyPlotEngine:
    def test_link_plot_engine(self, balance_engine_base_test: BalanceEngine):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        study.solve_adjustment()
        study.solve_change_state(new_temperature=90)
        fig_study = go.Figure()
        study.solve_change_state(wind_pressure=500, new_temperature=90)
        study.plot_engine.preview_line3d(fig_study)

        balance_engine_base_test.solve_adjustment()
        balance_engine_base_test.solve_change_state(new_temperature=90)
        fig_engine = go.Figure()
        plot_engine = PlotEngine(balance_engine_base_test)
        balance_engine_base_test.solve_change_state(
            wind_pressure=500, new_temperature=90
        )
        plot_engine.preview_line3d(fig_engine)

        if show_figures:
            fig_study.show()
            fig_engine.show()


class TestStudyErrorAndRestoreState:
    def test_error_back_to_default_state(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        study.solve_adjustment()
        study.solve_change_state()
        basic_state = study.save_state()
        with pytest.raises(Exception):
            study.solve_change_state(ice_thickness=7)
        np.testing.assert_array_equal(
            basic_state.dxdydz, study.chain_displacement().T
        )

    def test_change_state_error_ice(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        study.solve_adjustment()
        study.solve_change_state()
        with pytest.raises(Exception):
            study.solve_change_state(ice_thickness=7)

        # check solver runs correctly after error
        study.solve_change_state(ice_thickness=2e-2)

    def test_change_state_error_wind(
        self, balance_engine_base_test: BalanceEngine
    ):
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        study.solve_adjustment()
        study.solve_change_state()
        with pytest.raises(Exception):
            study.solve_change_state(wind_pressure=-99999)

        # check solver runs correctly after error
        study.solve_change_state(wind_pressure=400)


class TestSectionStudyDelegates:
    """Verify that SectionStudy delegate methods return the same values as
    the underlying BalanceEngine / PositionEngine."""

    @pytest.fixture()
    def solved_study(
        self, balance_engine_base_test: BalanceEngine
    ) -> SectionStudy:
        study = SectionStudy(
            cable_array=balance_engine_base_test.cable_array,
            section_array=balance_engine_base_test.section_array,
        )
        study.solve_adjustment()
        study.solve_change_state(new_temperature=30)
        return study

    def test_get_data_spans(self, solved_study: SectionStudy):
        result = solved_study.get_data_spans()
        expected = solved_study.balance_engine.get_data_spans()
        assert result == expected

    def test_chain_displacement(self, solved_study: SectionStudy):
        result = solved_study.chain_displacement()
        expected = (
            solved_study.balance_engine.balance_model.chain_displacement()
        )
        np.testing.assert_array_equal(result, expected)

    def test_vhl_under_chain(self, solved_study: SectionStudy):
        result = solved_study.vhl_under_chain()
        expected = solved_study.balance_engine.balance_model.vhl_under_chain()
        np.testing.assert_array_equal(
            result.vhl_matrix.value(), expected.vhl_matrix.value()
        )

    def test_vhl_under_console(self, solved_study: SectionStudy):
        result = solved_study.vhl_under_console()
        expected = (
            solved_study.balance_engine.balance_model.vhl_under_console()
        )
        np.testing.assert_array_equal(
            result.vhl_matrix.value(), expected.vhl_matrix.value()
        )

    def test_supports_number(self, solved_study: SectionStudy):
        result = solved_study.supports_number()
        expected = solved_study.balance_engine.support_number
        assert result == expected

    def test_get_points_for_plot(self, solved_study: SectionStudy):
        result_span, _, _ = solved_study.get_points_for_plot()
        expected_span, _, _ = solved_study.position_engine.get_points_for_plot(
            project=False, frame_index=0
        )
        np.testing.assert_array_equal(result_span.coords, expected_span.coords)
