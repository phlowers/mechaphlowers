# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import patch

import numpy as np

from mechaphlowers.config import options as cfg
from mechaphlowers.core.geometry.position_engine import PositionEngine
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.entities.reactivity import Notifier, Observer
from mechaphlowers.plotting.plot import PlotEngine

# ── Helpers ──────────────────────────────────────────────────────────────────


class ConcreteObserver(Observer):
    """Minimal concrete Observer for Notifier unit tests."""

    def __init__(self):
        self.call_count = 0
        self.last_notifier = None

    def update(self, notifier: Notifier, *args, **kwargs) -> None:
        self.call_count += 1
        self.last_notifier = notifier


# ── Section 1: Notifier / Observer infrastructure (unit tests) ────────────────


class TestNotifierInfrastructure:
    def test_initial_state_has_empty_observers_list(self):
        n = Notifier()
        assert n._observers == []

    def test_bind_to_registers_observer(self):
        n = Notifier()
        obs = ConcreteObserver()
        n.bind_to(obs)
        assert obs in n._observers

    def test_bind_to_multiple_observers_registers_all(self):
        n = Notifier()
        obs1, obs2 = ConcreteObserver(), ConcreteObserver()
        n.bind_to(obs1)
        n.bind_to(obs2)
        assert len(n._observers) == 2
        assert obs1 in n._observers
        assert obs2 in n._observers

    def test_notify_with_no_observers_does_not_raise(self):
        n = Notifier()
        n.notify()  # must not raise

    def test_notify_calls_each_observer_update_exactly_once(self):
        n = Notifier()
        obs1, obs2 = ConcreteObserver(), ConcreteObserver()
        n.bind_to(obs1)
        n.bind_to(obs2)
        n.notify()
        assert obs1.call_count == 1
        assert obs2.call_count == 1

    def test_notify_passes_notifier_reference_to_observer(self):
        n = Notifier()
        obs = ConcreteObserver()
        n.bind_to(obs)
        n.notify()
        assert obs.last_notifier is n

    def test_notify_called_twice_calls_observer_twice(self):
        n = Notifier()
        obs = ConcreteObserver()
        n.bind_to(obs)
        n.notify()
        n.notify()
        assert obs.call_count == 2


# ── Section 2: PlotEngine / PositionEngine observer registration ──────────────


class TestPlotEngineRegistration:
    def test_position_engine_registers_with_balance_engine_on_construction(
        self, balance_engine_base_test: BalanceEngine
    ):
        """PlotEngine auto-creates a PositionEngine that registers with BalanceEngine."""
        plt_engine = PlotEngine(balance_engine_base_test)
        assert (
            plt_engine.position_engine in balance_engine_base_test._observers
        )

    def test_plot_engine_registers_with_position_engine_on_construction(
        self, balance_engine_base_test: BalanceEngine
    ):
        """PlotEngine registers itself as an observer of the PositionEngine."""
        plt_engine = PlotEngine(balance_engine_base_test)
        assert plt_engine in plt_engine.position_engine._observers

    def test_two_plot_engines_have_independent_position_engines(
        self, balance_engine_base_test: BalanceEngine
    ):
        """Two PlotEngines each create their own PositionEngine observer."""
        pe1 = PlotEngine(balance_engine_base_test)
        pe2 = PlotEngine(balance_engine_base_test)
        assert pe1.position_engine in balance_engine_base_test._observers
        assert pe2.position_engine in balance_engine_base_test._observers
        assert len(balance_engine_base_test._observers) == 2


# ── Section 3: Notification triggers — two-hop chain ─────────────────────────
# Chain: BalanceEngine.notify() → PositionEngine.reset() → PositionEngine.notify()
#        → PlotEngine.update()


class TestNotificationTriggers:
    def test_position_engine_reset_called_on_direct_notify(
        self, balance_engine_base_test: BalanceEngine
    ):
        plt_engine = PlotEngine(balance_engine_base_test)
        with patch.object(
            plt_engine.position_engine,
            "reset",
            wraps=plt_engine.position_engine.reset,
        ) as mock_reset:
            balance_engine_base_test.notify()

            mock_reset.assert_called_once_with(
                balance_engine=balance_engine_base_test
            )

    def test_plot_engine_update_called_after_position_engine_reset(
        self, balance_engine_base_test: BalanceEngine
    ):
        """PlotEngine.update() is called once per PositionEngine notification."""
        plt_engine = PlotEngine(balance_engine_base_test)
        with patch.object(
            plt_engine, "update", wraps=plt_engine.update
        ) as mock_update:
            balance_engine_base_test.notify()
            mock_update.assert_called_once()

    def test_add_loads_triggers_position_engine_reset(
        self, balance_engine_base_test: BalanceEngine
    ):
        plt_engine = PlotEngine(balance_engine_base_test)
        with patch.object(
            plt_engine.position_engine,
            "reset",
            wraps=plt_engine.position_engine.reset,
        ) as mock_reset:
            balance_engine_base_test.add_loads(
                load_position_distance=np.array([0, 0, 0, np.nan]),
                load_mass=np.array([0, 0, 0, np.nan]),
            )

            mock_reset.assert_called_once_with(
                balance_engine=balance_engine_base_test
            )

    def test_reset_full_false_triggers_position_engine_reset(
        self, balance_engine_base_test: BalanceEngine
    ):
        plt_engine = PlotEngine(balance_engine_base_test)
        with patch.object(
            plt_engine.position_engine,
            "reset",
            wraps=plt_engine.position_engine.reset,
        ) as mock_reset:
            balance_engine_base_test.reset(full=False)

            mock_reset.assert_called_once_with(
                balance_engine=balance_engine_base_test
            )

    def test_multiple_plot_engines_all_notified_on_add_loads(
        self, balance_engine_base_test: BalanceEngine
    ):
        """Each PlotEngine has its own PositionEngine observer; all are notified."""
        pe1 = PlotEngine(balance_engine_base_test)
        pe2 = PlotEngine(balance_engine_base_test)
        with (
            patch.object(
                pe1.position_engine,
                "reset",
                wraps=pe1.position_engine.reset,
            ) as mock1,
            patch.object(
                pe2.position_engine,
                "reset",
                wraps=pe2.position_engine.reset,
            ) as mock2,
        ):
            balance_engine_base_test.add_loads(
                load_position_distance=np.array([0, 0, 0, np.nan]),
                load_mass=np.array([0, 0, 0, np.nan]),
            )

            mock1.assert_called_once_with(
                balance_engine=balance_engine_base_test
            )
            mock2.assert_called_once_with(
                balance_engine=balance_engine_base_test
            )

    def test_section_pts_reset_called_through_full_observer_chain(
        self, balance_engine_base_test: BalanceEngine
    ):
        """section_pts.reset() is the terminal step of the observer chain."""
        plt_engine = PlotEngine(balance_engine_base_test)
        with patch.object(
            plt_engine.section_pts, "reset", wraps=plt_engine.section_pts.reset
        ) as mock_reset:
            balance_engine_base_test.add_loads(
                load_position_distance=np.array([0, 0, 0, np.nan]),
                load_mass=np.array([0, 0, 0, np.nan]),
            )

            mock_reset.assert_called_once()


# ── Section 4: Solve methods must NOT trigger notification ────────────────────


class TestSolveMethodsDoNotNotify:
    def test_solve_adjustment_does_not_notify_observers(
        self, balance_engine_base_test: BalanceEngine
    ):
        plt_engine = PlotEngine(balance_engine_base_test)
        with patch.object(
            plt_engine.position_engine,
            "reset",
            wraps=plt_engine.position_engine.reset,
        ) as mock_reset:
            balance_engine_base_test.solve_adjustment()

            mock_reset.assert_not_called()

    def test_solve_change_state_does_not_notify_observers(
        self, balance_engine_base_test: BalanceEngine
    ):
        plt_engine = PlotEngine(balance_engine_base_test)
        balance_engine_base_test.solve_adjustment()
        with patch.object(
            plt_engine.position_engine,
            "reset",
            wraps=plt_engine.position_engine.reset,
        ) as mock_reset:
            balance_engine_base_test.solve_change_state(new_temperature=50)

            mock_reset.assert_not_called()


# ── Section 5: Reference integrity after partial reset ───────────────────────


class TestReferenceIntegrity:
    def test_plot_engine_span_model_is_same_object_as_nodes_span_model(
        self, balance_engine_base_test: BalanceEngine
    ):
        """plt_engine.span_model (delegating property) must reference the exact
        same object as nodes_span_model so live updates are immediately visible."""
        plt_engine = PlotEngine(balance_engine_base_test)
        assert (
            plt_engine.span_model
            is balance_engine_base_test.balance_model.nodes_span_model
        )

    def test_spans_reference_preserved_after_add_loads(
        self, balance_engine_base_test: BalanceEngine
    ):
        """reset(full=False) uses mirror() so nodes_span_model keeps the same
        Python identity — PlotEngine.span_model does NOT become a dangling
        reference after the observer chain fires."""
        plt_engine = PlotEngine(balance_engine_base_test)
        original_id = id(plt_engine.span_model)

        balance_engine_base_test.add_loads(
            load_position_distance=np.array([0, 0, 0, np.nan]),
            load_mass=np.array([0, 0, 0, np.nan]),
        )

        assert id(plt_engine.span_model) == original_id
        assert (
            plt_engine.span_model
            is balance_engine_base_test.balance_model.nodes_span_model
        )

    def test_spans_reference_preserved_after_reset_full_false(
        self, balance_engine_base_test: BalanceEngine
    ):
        plt_engine = PlotEngine(balance_engine_base_test)
        original_id = id(plt_engine.span_model)

        balance_engine_base_test.reset(full=False)

        assert id(plt_engine.span_model) == original_id


# ── Section 6: reset(full=True) breaks the observer link ─────────────────────


class TestFullResetBehavior:
    def test_full_reset_clears_balance_engine_observer_registry(
        self, balance_engine_base_test: BalanceEngine
    ):
        """reset(full=True) calls super().__init__() which resets _observers=[].
        PositionEngine is no longer registered; callers must re-create PlotEngine."""
        plt_engine = PlotEngine(balance_engine_base_test)
        assert (
            plt_engine.position_engine in balance_engine_base_test._observers
        )

        balance_engine_base_test.reset(full=True)

        assert balance_engine_base_test._observers == []
        assert (
            plt_engine.position_engine
            not in balance_engine_base_test._observers
        )

    def test_position_engine_reset_calls_initialize_engine_when_not_initialized(
        self, balance_engine_base_test: BalanceEngine
    ):
        """PositionEngine.reset() calls initialize_engine() when
        balance_engine.initialized is False, re-establishing fresh references."""
        plt_engine = PlotEngine(balance_engine_base_test)
        with patch.object(
            plt_engine.position_engine,
            "initialize_engine",
            wraps=plt_engine.position_engine.initialize_engine,
        ) as mock_init:
            balance_engine_base_test.initialized = False
            plt_engine.position_engine.reset(
                balance_engine=balance_engine_base_test
            )

            mock_init.assert_called_once_with(balance_engine_base_test)


# ── Section 7: Observer selectivity ──────────────────────────────────────────


class TestObserverSelectivity:
    def test_position_engine_update_ignores_non_balance_engine_notifier(
        self, balance_engine_base_test: BalanceEngine
    ):
        """PositionEngine.update() is a no-op when the notifier is not a
        BalanceEngine, making it safe to compose with other Notifier subclasses."""
        plt_engine = PlotEngine(balance_engine_base_test)
        with patch.object(
            plt_engine.position_engine,
            "reset",
            wraps=plt_engine.position_engine.reset,
        ) as mock_reset:
            plt_engine.position_engine.update(Notifier())
            mock_reset.assert_not_called()

    def test_plot_engine_update_any_notifier_does_not_raise(
        self, balance_engine_base_test: BalanceEngine
    ):
        """PlotEngine.update() accepts any notifier without raising."""
        plt_engine = PlotEngine(balance_engine_base_test)
        plt_engine.update(Notifier())
        plt_engine.update(plt_engine.position_engine)


# ── Section 8: End-to-end coordinate coherence ───────────────────────────────


class TestCoordCoherence:
    def test_x_cable_consistent_with_span_model_after_observer_chain(
        self, balance_engine_base_test: BalanceEngine
    ):
        """After the observer chain fires, section_pts.x_cable must exactly match
        nodes_span_model.get_coords(), proving set_cable_coordinates() was called."""
        plt_engine = PlotEngine(balance_engine_base_test)

        balance_engine_base_test.add_loads(
            load_position_distance=np.array([0, 0, 0, np.nan]),
            load_mass=np.array([0, 0, 0, np.nan]),
        )

        x_expected, _ = plt_engine.span_model.get_coords(
            cfg.graphics.resolution
        )
        np.testing.assert_array_equal(
            plt_engine.section_pts.x_cable, x_expected
        )

    def test_cable_coords_stale_after_solve_then_refreshed_by_observer_chain(
        self, balance_engine_base_test: BalanceEngine
    ):
        """This test captures the core value of the reactivity feature:

        1. After construction, x_cable is computed once (initial sagging_parameter).
        2. solve_adjustment + solve_change_state update nodes_span_model.sagging_parameter
           in-place — but solve does NOT call notify(), so x_cable remains stale.
        3. add_loads() calls reset(full=False) → BalanceEngine.notify()
           → PositionEngine.reset() → section_pts.reset() → set_cable_coordinates()
           recomputes x_cable → PositionEngine.notify() → PlotEngine.update().
        4. x_cable now matches the post-solve sagging_parameter and differs from
           the initial cached value.
        """
        plt_engine = PlotEngine(balance_engine_base_test)
        x_cable_initial = plt_engine.section_pts.x_cable.copy()

        # Solve — updates nodes_span_model.sagging_parameter in-place via mirror,
        # but does NOT trigger an observer notification.
        balance_engine_base_test.solve_adjustment()
        balance_engine_base_test.solve_change_state(new_temperature=50)

        # x_cable is stale: solve did not notify observers.
        np.testing.assert_array_equal(
            plt_engine.section_pts.x_cable, x_cable_initial
        )

        # Trigger the observer chain via add_loads.
        balance_engine_base_test.add_loads(
            load_position_distance=np.array([0, 0, 0, np.nan]),
            load_mass=np.array([0, 0, 0, np.nan]),
        )

        # x_cable is now consistent with the (post-solve) span model.
        x_expected, _ = plt_engine.span_model.get_coords(
            cfg.graphics.resolution
        )
        np.testing.assert_array_equal(
            plt_engine.section_pts.x_cable, x_expected
        )

        # And it differs from the initial cached value —
        # confirming sagging_parameter changed after solve_change_state.
        assert not np.array_equal(
            plt_engine.section_pts.x_cable, x_cable_initial
        )
