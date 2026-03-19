# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Tests for PositionEngine — standalone geometry computation without Plotly.
"""

from unittest.mock import patch

import numpy as np
import pytest

from mechaphlowers.config import options as cfg
from mechaphlowers.core.geometry.distances import DistanceResult
from mechaphlowers.core.geometry.position_engine import PositionEngine
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.entities.reactivity import Notifier, Observer


# ── Helpers ───────────────────────────────────────────────────────────────────


class ConcreteObserver(Observer):
    """Minimal concrete Observer to verify downstream notification."""

    def __init__(self):
        self.call_count = 0
        self.last_notifier = None

    def update(self, notifier: Notifier, *args, **kwargs) -> None:
        self.call_count += 1
        self.last_notifier = notifier


# ── Construction & registration ───────────────────────────────────────────────


class TestPositionEngineConstruction:
    def test_registers_with_balance_engine_on_construction(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        assert pos_engine in balance_engine_base_test._observers

    def test_exposes_span_model_referencing_nodes_span_model(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        assert (
            pos_engine.span_model
            is balance_engine_base_test.balance_model.nodes_span_model
        )

    def test_exposes_cable_loads(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        assert pos_engine.cable_loads is balance_engine_base_test.cable_loads

    def test_exposes_section_array(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        assert (
            pos_engine.section_array is balance_engine_base_test.section_array
        )

    def test_is_also_a_notifier(self, balance_engine_base_test: BalanceEngine):
        pos_engine = PositionEngine(balance_engine_base_test)
        assert isinstance(pos_engine, Notifier)

    def test_downstream_observer_can_register(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        downstream = ConcreteObserver()
        pos_engine.bind_to(downstream)
        assert downstream in pos_engine._observers


# ── Reactivity: PositionEngine as Observer ────────────────────────────────────


class TestPositionEngineReactivity:
    def test_reset_called_when_balance_engine_notifies(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        with patch.object(
            pos_engine, "reset", wraps=pos_engine.reset
        ) as mock_reset:
            balance_engine_base_test.notify()
            mock_reset.assert_called_once_with(
                balance_engine=balance_engine_base_test
            )

    def test_update_ignores_non_balance_engine_notifier(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        with patch.object(
            pos_engine, "reset", wraps=pos_engine.reset
        ) as mock_reset:
            pos_engine.update(Notifier())
            mock_reset.assert_not_called()

    def test_solve_adjustment_does_not_trigger_reset(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        with patch.object(
            pos_engine, "reset", wraps=pos_engine.reset
        ) as mock_reset:
            balance_engine_base_test.solve_adjustment()
            mock_reset.assert_not_called()

    def test_solve_change_state_does_not_trigger_reset(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        balance_engine_base_test.solve_adjustment()
        with patch.object(
            pos_engine, "reset", wraps=pos_engine.reset
        ) as mock_reset:
            balance_engine_base_test.solve_change_state(new_temperature=50)
            mock_reset.assert_not_called()

    def test_add_loads_triggers_reset(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        with patch.object(
            pos_engine, "reset", wraps=pos_engine.reset
        ) as mock_reset:
            balance_engine_base_test.add_loads(
                load_position_distance=np.array([0, 0, 0, np.nan]),
                load_mass=np.array([0, 0, 0, np.nan]),
            )
            mock_reset.assert_called_once_with(
                balance_engine=balance_engine_base_test
            )

    def test_add_loads_notifies_downstream_observer(
        self, balance_engine_base_test: BalanceEngine
    ):
        """PositionEngine notifies its own observers after updating."""
        pos_engine = PositionEngine(balance_engine_base_test)
        downstream = ConcreteObserver()
        pos_engine.bind_to(downstream)

        balance_engine_base_test.add_loads(
            load_position_distance=np.array([0, 0, 0, np.nan]),
            load_mass=np.array([0, 0, 0, np.nan]),
        )

        assert downstream.call_count == 1
        assert downstream.last_notifier is pos_engine

    def test_section_pts_reset_called_on_notify(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        with patch.object(
            pos_engine.section_pts, "reset", wraps=pos_engine.section_pts.reset
        ) as mock_reset:
            balance_engine_base_test.add_loads(
                load_position_distance=np.array([0, 0, 0, np.nan]),
                load_mass=np.array([0, 0, 0, np.nan]),
            )
            mock_reset.assert_called_once()


# ── Reference integrity ───────────────────────────────────────────────────────


class TestPositionEngineReferenceIntegrity:
    def test_span_model_same_object_as_nodes_span_model(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        assert (
            pos_engine.span_model
            is balance_engine_base_test.balance_model.nodes_span_model
        )

    def test_span_model_identity_preserved_after_add_loads(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        original_id = id(pos_engine.span_model)

        balance_engine_base_test.add_loads(
            load_position_distance=np.array([0, 0, 0, np.nan]),
            load_mass=np.array([0, 0, 0, np.nan]),
        )

        assert id(pos_engine.span_model) == original_id
        assert (
            pos_engine.span_model
            is balance_engine_base_test.balance_model.nodes_span_model
        )

    def test_x_cable_consistent_with_span_model_after_observer_chain(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)

        balance_engine_base_test.add_loads(
            load_position_distance=np.array([0, 0, 0, np.nan]),
            load_mass=np.array([0, 0, 0, np.nan]),
        )

        x_expected, _ = pos_engine.span_model.get_coords(
            cfg.graphics.resolution
        )
        np.testing.assert_array_equal(
            pos_engine.section_pts.x_cable, x_expected
        )


# ── Data retrieval (standalone — no Plotly import needed) ─────────────────────


class TestPositionEngineDataRetrieval:
    def test_get_supports_points_returns_array(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        result = pos_engine.get_supports_points()
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 3

    def test_get_insulators_points_returns_array(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        result = pos_engine.get_insulators_points()
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 3

    def test_get_spans_points_section_frame(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        result = pos_engine.get_spans_points(frame="section")
        assert isinstance(result, np.ndarray)

    def test_get_loads_coords_empty_before_solve(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        assert pos_engine.get_loads_coords() == {}

    def test_get_points_for_plot_returns_three_points_objects(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        result = pos_engine.get_points_for_plot()
        assert len(result) == 3

    def test_beta_property(self, balance_engine_base_test: BalanceEngine):
        pos_engine = PositionEngine(balance_engine_base_test)
        assert isinstance(pos_engine.beta, np.ndarray)


# ── Distance computation ─────────────────────────────────────────────────────


class TestPositionEngineDistance:
    def test_point_distance_returns_distance_result(
        self, balance_engine_base_test: BalanceEngine
    ):
        balance_engine_base_test.solve_adjustment()
        balance_engine_base_test.solve_change_state(new_temperature=15)
        pos_engine = PositionEngine(balance_engine_base_test)

        dr = pos_engine.point_distance(
            span_index=0, point=np.array([250.0, 0.0, 30.0])
        )

        assert isinstance(dr, DistanceResult)
        assert dr.distance_3d > 0

    def test_point_distance_invalid_span_index_raises(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        with pytest.raises(IndexError):
            pos_engine.point_distance(
                span_index=999, point=np.array([0.0, 0.0, 0.0])
            )

    def test_point_distance_invalid_point_shape_raises(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        with pytest.raises(ValueError):
            pos_engine.point_distance(
                span_index=0,
                point=np.array([0.0, 0.0]),  # 2D — invalid
            )


# ── String representations ────────────────────────────────────────────────────


class TestPositionEngineRepr:
    def test_str_contains_number_of_supports(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        assert "number of supports" in str(pos_engine)

    def test_repr_contains_class_name(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        assert "PositionEngine" in repr(pos_engine)
