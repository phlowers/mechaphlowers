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
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mechaphlowers.config import options as cfg
from mechaphlowers.core.geometry import (
    position_engine as position_engine_module,
)
from mechaphlowers.core.geometry.distances import (
    DistanceResult,
    ObstacleDistanceResult,
)
from mechaphlowers.core.geometry.position_engine import PositionEngine
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.data.units import convert_weight_to_mass
from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.entities.errors import NoIntersectionPlaneWarning
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

    def test_set_loads_triggers_reset(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        with patch.object(
            pos_engine, "reset", wraps=pos_engine.reset
        ) as mock_reset:
            balance_engine_base_test.set_loads(
                load_position_distance=np.array([0, 0, 0]),
                load_mass=np.array([0, 0, 0]),
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

        balance_engine_base_test.set_loads(
            load_position_distance=np.array([0, 0, 0]),
            load_mass=np.array([0, 0, 0]),
        )

        assert downstream.call_count == 1
        assert downstream.last_notifier is pos_engine

    def test_coords_calculator_reset_called_on_notify(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        with patch.object(
            pos_engine.coords_calculator,
            "reset",
            wraps=pos_engine.coords_calculator.reset,
        ) as mock_reset:
            balance_engine_base_test.set_loads(
                load_position_distance=np.array([0, 0, 0]),
                load_mass=np.array([0, 0, 0]),
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

        balance_engine_base_test.set_loads(
            load_position_distance=np.array([0, 0, 0]),
            load_mass=np.array([0, 0, 0]),
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

        balance_engine_base_test.set_loads(
            load_position_distance=np.array([0, 0, 0]),
            load_mass=np.array([0, 0, 0]),
        )

        x_expected, _ = pos_engine.span_model.get_coords(
            cfg.graphics.resolution
        )
        np.testing.assert_array_equal(
            pos_engine.coords_calculator.x_cable, x_expected
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

    def test_get_loads_coords_gropup_points_empty_before_solve(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        assert pos_engine.get_loads_coords_group_points() == {}

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


class TestPositionEngineObstacleArray:
    def test_obstacle_array_init(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        assert hasattr(pos_engine, "obstacle_array")

    def test_create_obstacle(self, balance_engine_base_test: BalanceEngine):
        pos_engine = PositionEngine(balance_engine_base_test)
        pos_engine.add_obstacle(
            name="obs_1",
            span_index=1,
            coords=np.array(
                [[50, 0, 0], [100, 0, 10], [150, 10, 0], [200, 0, 0]]
            ),
            support_reference='left',
        )
        expected_df = pd.DataFrame(
            {
                'name': ['obs_1', 'obs_1', 'obs_1', 'obs_1'],
                'point_index': [0, 1, 2, 3],
                'span_index': [1, 1, 1, 1],
                'x': [50.0, 100.0, 150.0, 200.0],
                'y': [0.0, 0.0, 10.0, 0.0],
                'z': [0.0, 10.0, 0.0, 0.0],
                'object_type': [
                    'ground',
                    'ground',
                    'ground',
                    'ground',
                ],
            }
        )
        assert_frame_equal(
            pos_engine.obstacle_array.data, expected_df, check_like=True
        )

    def test_add_many_obstacle(self, balance_engine_base_test: BalanceEngine):
        pos_engine = PositionEngine(balance_engine_base_test)
        pos_engine.add_obstacle(
            name="obs_0",
            span_index=0,
            coords=np.array([[100, 0, 0], [200, 0, 10]]),
            support_reference='left',
        )
        pos_engine.add_obstacle(
            name="obs_1",
            span_index=1,
            coords=np.array(
                [[50, 0, 0], [100, 0, 10], [150, 10, 0], [200, 0, 0]]
            ),
            support_reference='left',
        )
        pos_engine.add_obstacle(
            name="obs_2",
            span_index=1,
            coords=np.array([[35, 0, 0], [100, 0, 10]]),
            support_reference='right',
            span_length=np.array([500, 400, 450, np.nan]),
        )

        expected_df = pd.DataFrame(
            {
                "name": [
                    "obs_0",
                    "obs_0",
                    "obs_1",
                    "obs_1",
                    "obs_1",
                    "obs_1",
                    "obs_2",
                    "obs_2",
                ],
                "point_index": [0, 1, 0, 1, 2, 3, 0, 1],
                "span_index": [0, 0, 1, 1, 1, 1, 1, 1],
                "x": [100.0, 200.0, 50.0, 100.0, 150.0, 200.0, 365.0, 300.0],
                "y": [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                "z": [0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
                "object_type": [
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                ],
            }
        )
        assert_frame_equal(
            pos_engine.obstacle_array.data, expected_df, check_like=True
        )

    def test_refresh_absolute_coordinates(
        self, balance_engine_base_test: BalanceEngine
    ):
        # test that refresh_obstacles method keeps
        pos_engine = PositionEngine(balance_engine_base_test)
        pos_engine.add_obstacle(
            name="obs_1",
            span_index=1,
            coords=np.array(
                [[50, 0, 0], [100, 0, 10], [150, 10, 0], [200, 0, 0]]
            ),
            support_reference='left',
        )

        expected_coords = np.array(
            [
                [550.0, 0.0, 0.0],
                [600.0, 0.0, 10.0],
                [650.0, 10.0, 0.0],
                [700.0, 0.0, 0.0],
            ]
        )
        pos_engine.coords_calculator.refresh_obstacles()
        np.testing.assert_equal(
            pos_engine.coords_calculator.obstacles_points.coords,
            expected_coords,
        )

    def test_add_and_delete_obstacle(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        pos_engine.add_obstacle(
            name="obs_0",
            span_index=0,
            coords=np.array([[100, 0, 0], [200, 0, 10]]),
            support_reference='left',
        )
        pos_engine.add_obstacle(
            name="obs_1",
            span_index=1,
            coords=np.array(
                [[50, 0, 0], [100, 0, 10], [150, 10, 0], [200, 0, 0]]
            ),
            support_reference='left',
        )
        pos_engine.add_obstacle(
            name="obs_2",
            span_index=1,
            coords=np.array([[35, 0, 0], [100, 0, 10]]),
            support_reference='right',
            span_length=np.array([500, 400, 450, np.nan]),
        )
        pos_engine.delete_obstacle("obs_1")
        expected_df = pd.DataFrame(
            {
                "name": [
                    "obs_0",
                    "obs_0",
                    "obs_2",
                    "obs_2",
                ],
                "point_index": [0, 1, 0, 1],
                "span_index": [0, 0, 1, 1],
                "x": [100.0, 200.0, 365.0, 300.0],
                "y": [0.0, 0.0, 0.0, 0.0],
                "z": [0.0, 10.0, 0.0, 10.0],
                "object_type": [
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                ],
            }
        )
        assert_frame_equal(
            pos_engine.obstacle_array.data, expected_df, check_like=True
        )

    def test_delete_point(self, balance_engine_base_test: BalanceEngine):
        pos_engine = PositionEngine(balance_engine_base_test)
        pos_engine.add_obstacle(
            name="obs_0",
            span_index=0,
            coords=np.array([[100, 0, 0], [200, 0, 10]]),
            support_reference='left',
        )
        pos_engine.add_obstacle(
            name="obs_1",
            span_index=1,
            coords=np.array(
                [[50, 0, 0], [100, 0, 10], [150, 10, 0], [200, 0, 0]]
            ),
            support_reference='left',
        )
        pos_engine.add_obstacle(
            name="obs_2",
            span_index=1,
            coords=np.array([[35, 0, 0], [100, 0, 10]]),
            support_reference='right',
            span_length=np.array([500, 400, 450, np.nan]),
        )
        pos_engine.delete_point(obs_name="obs_1", point_index=2)

        expected_df = pd.DataFrame(
            {
                "name": [
                    "obs_0",
                    "obs_0",
                    "obs_1",
                    "obs_1",
                    "obs_1",
                    "obs_2",
                    "obs_2",
                ],
                "point_index": [0, 1, 0, 1, 2, 0, 1],
                "span_index": [0, 0, 1, 1, 1, 1, 1],
                "x": [100.0, 200.0, 50.0, 100.0, 200.0, 365.0, 300.0],
                "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "z": [0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 10.0],
                "object_type": [
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                ],
            }
        )
        assert_frame_equal(
            pos_engine.obstacle_array.data, expected_df, check_like=True
        )

    def test_add_and_delete_obstacle_dict(
        self, balance_engine_base_test: BalanceEngine
    ):
        pos_engine = PositionEngine(balance_engine_base_test)
        pos_engine.add_obstacle(
            name="obs_0",
            span_index=0,
            coords=np.array([[100, 0, 0], [200, 0, 10]]),
            support_reference='left',
        )
        pos_engine.add_obstacle(
            name="obs_1",
            span_index=1,
            coords=np.array(
                [[50, 0, 0], [100, 0, 10], [150, 10, 0], [200, 0, 0]]
            ),
            support_reference='left',
        )
        pos_engine.add_obstacle(
            name="obs_2",
            span_index=1,
            coords=np.array([[35, 0, 0], [100, 0, 10]]),
            support_reference='right',
            span_length=np.array([500, 400, 450, np.nan]),
        )
        pos_engine.delete_obstacle("obs_1")

        assert set(pos_engine.obstacles_dict().keys()) == {"obs_0", "obs_2"}


class TestDistancesFromObstacles:
    @pytest.fixture
    def pos_engine_with_obstacles(
        self, balance_engine_base_test: BalanceEngine
    ) -> PositionEngine:
        pos_engine = PositionEngine(balance_engine_base_test)
        pos_engine.add_obstacle(
            name="obs_0",
            span_index=0,
            coords=np.array([[100, 0, 0], [200, 0, 10]]),
            support_reference='left',
        )
        pos_engine.add_obstacle(
            name="obs_1",
            span_index=1,
            coords=np.array(
                [[50, 0, 0], [100, 0, 10], [150, 10, 0], [200, 0, 0]]
            ),
            support_reference='left',
        )
        pos_engine.add_obstacle(
            name="obs_2",
            span_index=1,
            coords=np.array([[35, 0, 0], [100, 0, 10]]),
            support_reference='right',
            span_length=np.array([500, 500, 500, np.nan]),
        )
        return pos_engine

    @pytest.fixture
    def pos_engine_with_obstacles_angles(
        self, balance_engine_angles: BalanceEngine
    ) -> PositionEngine:
        pos_engine = PositionEngine(balance_engine_angles)
        pos_engine.add_obstacle(
            name="obs_0",
            span_index=0,
            coords=np.array([[100, 0, 0], [200, 0, 10]]),
            support_reference='left',
        )
        pos_engine.add_obstacle(
            name="obs_1",
            span_index=1,
            coords=np.array(
                [[50, 0, 0], [100, 0, 10], [150, 10, 0], [200, 0, 0]]
            ),
            support_reference='left',
        )
        pos_engine.add_obstacle(
            name="obs_2",
            span_index=1,
            coords=np.array([[35, 0, 0], [100, 0, 10]]),
            support_reference='right',
            span_length=np.array([500, 500, 500, np.nan]),
        )
        return pos_engine

    def test_multiple_obstacles(
        self, pos_engine_with_obstacles: PositionEngine
    ):
        distances_dict = (
            pos_engine_with_obstacles.get_distances_from_obstacles()
        )
        assert set(distances_dict.keys()) == {"obs_0", "obs_1", "obs_2"}
        assert set(distances_dict["obs_1"].keys()) == {0, 1, 2, 3}

    def test_multiple_obstacles_angles(
        self, pos_engine_with_obstacles_angles: PositionEngine
    ):
        distances_dict = (
            pos_engine_with_obstacles_angles.get_distances_from_obstacles()
        )
        assert set(distances_dict.keys()) == {"obs_0", "obs_1", "obs_2"}
        assert set(distances_dict["obs_1"].keys()) == {0, 1, 2, 3}

    def test_no_obstacle(self, balance_engine_base_test: BalanceEngine):
        pos_engine = PositionEngine(balance_engine_base_test)

        distances_dict = pos_engine.get_distances_from_obstacles()
        assert distances_dict == {}

    def test_no_intersection_skips_point_and_warns(
        self, pos_engine_with_obstacles: PositionEngine
    ):
        original_batch = position_engine_module.batch_plane_distances
        failing_point = pos_engine_with_obstacles.coords_calculator.obstacles_points.dict_coords()[
            "obs_1"
        ][1]

        def fake_batch(points, span_indices, curves, normals):
            # obs_1 has 4 points (indices 0-3); make the 2nd one (index 1) fail.
            distances, targets = original_batch(
                points, span_indices, curves, normals
            )
            failing = (span_indices == 1) & np.all(
                points == failing_point, axis=1
            )
            distances[failing] = np.inf
            targets[failing] = np.nan
            return distances, targets

        with patch.object(
            position_engine_module,
            "batch_plane_distances",
            side_effect=fake_batch,
        ):
            with pytest.warns(NoIntersectionPlaneWarning, match="obs_1"):
                distances_dict = (
                    pos_engine_with_obstacles.get_distances_from_obstacles()
                )

        assert set(distances_dict.keys()) == {"obs_0", "obs_1", "obs_2"}
        # Point index 1 for obs_1 was skipped, the others are still present.
        assert set(distances_dict["obs_1"].keys()) == {0, 2, 3}

    def test_get_distances_equal_point_distance(
        self, pos_engine_with_obstacles: PositionEngine
    ):
        distances_dict = (
            pos_engine_with_obstacles.get_distances_from_obstacles()
        )
        obstacles_points = pos_engine_with_obstacles.coords_calculator.compute_obstacle_coords()
        checked = 0
        for row, name in enumerate(obstacles_points.object_name):
            point_index = int(obstacles_points.point_index[row])
            if point_index not in distances_dict[name]:
                continue
            batch = distances_dict[name][point_index]
            expected = pos_engine_with_obstacles.point_distance(
                int(obstacles_points.span_index[row]),
                obstacles_points.coords[row],
            )
            np.testing.assert_allclose(
                batch.distance_3d, expected.distance_3d, atol=1e-9
            )
            np.testing.assert_allclose(
                batch.point_target, expected.point_target, atol=1e-9
            )
            np.testing.assert_allclose(
                batch.distance_projection_u,
                expected.distance_projection_u,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                batch.distance_projection_v,
                expected.distance_projection_v,
                atol=1e-9,
            )
            checked += 1
        assert checked > 0

    def test_group_points_sandbox(
        self, pos_engine_with_obstacles: PositionEngine
    ):
        group_points = pos_engine_with_obstacles.get_group_points()
        group_points.change_frame(frame_index=1)
        assert True


class TestObstacleMinDistance:
    @pytest.fixture
    def pos_engine(self, balance_engine_base_test: BalanceEngine):
        return PositionEngine(balance_engine_base_test)

    def test_single_point_matches_point_distance(
        self, pos_engine: PositionEngine
    ):
        pos_engine.add_obstacle(
            name="pt", span_index=1, coords=np.array([[200, 5, 10]])
        )
        result = pos_engine.obstacle_min_distance("pt")
        point = pos_engine.obstacles_dict()["pt"][0]
        expected = pos_engine.point_distance(1, point)

        assert isinstance(result, ObstacleDistanceResult)
        assert result.span_index == 1
        assert result.segment_index == 0
        assert result.segment_parameter == 0.0
        np.testing.assert_allclose(result.distance_3d, expected.distance_3d)
        np.testing.assert_allclose(result.closest_obstacle_point, point)

    def test_minimum_between_vertices(self, pos_engine: PositionEngine):
        # Crossing line whose closest location is far from both vertices.
        pos_engine.add_obstacle(
            name="road",
            span_index=0,
            coords=np.array([[50, -40, 0], [450, 40, 0]]),
        )
        vertices = pos_engine.obstacles_dict()["road"]
        vertex_distances = [
            pos_engine.point_distance(0, vertex).distance_3d
            for vertex in vertices
        ]

        result = pos_engine.obstacle_min_distance("road", step=1.0)

        assert result is not None
        assert 0.0 < result.segment_parameter < 1.0
        assert result.distance_3d < min(vertex_distances)

        brute_force = pos_engine.obstacle_min_distance(
            "road", step=0.05, refine=False
        )
        assert brute_force is not None
        assert result.distance_3d <= brute_force.distance_3d + 1e-6
        np.testing.assert_allclose(
            result.distance_3d, brute_force.distance_3d, atol=1e-3
        )
        expected_point = vertices[0] + result.segment_parameter * (
            vertices[1] - vertices[0]
        )
        np.testing.assert_allclose(
            result.closest_obstacle_point, expected_point
        )

    def test_refine_never_worse_than_sampling(
        self, pos_engine: PositionEngine
    ):
        pos_engine.add_obstacle(
            name="polyline",
            span_index=1,
            coords=np.array(
                [[50, 0, 0], [180, 10, 20], [260, -10, 15], [400, 0, 0]]
            ),
        )
        coarse = pos_engine.obstacle_min_distance(
            "polyline", step=50.0, refine=False
        )
        refined = pos_engine.obstacle_min_distance("polyline", step=50.0)
        fine = pos_engine.obstacle_min_distance(
            "polyline", step=0.05, refine=False
        )

        assert coarse is not None and refined is not None and fine is not None
        assert refined.distance_3d <= coarse.distance_3d
        np.testing.assert_allclose(
            refined.distance_3d, fine.distance_3d, atol=1e-3
        )

    def test_partially_outside_span_warns(self, pos_engine: PositionEngine):
        pos_engine.add_obstacle(
            name="obs",
            span_index=0,
            coords=np.array([[-50, 0, 0], [100, 0, 0]]),
        )
        with pytest.warns(NoIntersectionPlaneWarning):
            result = pos_engine.obstacle_min_distance("obs")
        assert result is not None

    def test_fully_outside_span_returns_none(self, pos_engine: PositionEngine):
        pos_engine.add_obstacle(
            name="obs",
            span_index=0,
            coords=np.array([[-80, 0, 0], [-50, 0, 0]]),
        )
        with pytest.warns(NoIntersectionPlaneWarning):
            assert pos_engine.obstacle_min_distance("obs") is None

    def test_get_min_distances_from_obstacles(
        self, pos_engine: PositionEngine
    ):
        pos_engine.add_obstacle(
            name="obs_0", span_index=0, coords=np.array([[100, 0, 0]])
        )
        pos_engine.add_obstacle(
            name="obs_1",
            span_index=1,
            coords=np.array([[50, 0, 0], [100, 0, 10]]),
        )
        results = pos_engine.get_segment_distances_from_obstacles()

        assert set(results.keys()) == {"obs_0", "obs_1"}
        assert results["obs_1"] is not None
        assert results["obs_1"].span_index == 1

    def test_no_obstacle(self, pos_engine: PositionEngine):
        assert pos_engine.get_segment_distances_from_obstacles() == {}

    def test_invalid_inputs(self, pos_engine: PositionEngine):
        pos_engine.add_obstacle(
            name="obs", span_index=0, coords=np.array([[100, 0, 0]])
        )
        with pytest.raises(KeyError):
            pos_engine.obstacle_min_distance("unknown")
        with pytest.raises(ValueError):
            pos_engine.obstacle_min_distance("obs", step=0)
        with pytest.raises(ValueError):
            pos_engine.obstacle_min_distance("obs", xtol=-1)

    def test_get_min_distances_computes_spans_once(
        self, pos_engine: PositionEngine
    ):
        pos_engine.add_obstacle(
            name="obs_a", span_index=0, coords=np.array([[50, 5, 5]])
        )
        pos_engine.add_obstacle(
            name="obs_b", span_index=1, coords=np.array([[200, -5, 5]])
        )
        coords_calculator = pos_engine.coords_calculator
        with patch.object(
            coords_calculator,
            "get_spans",
            wraps=coords_calculator.get_spans,
        ) as get_spans_mock:
            results = pos_engine.get_segment_distances_from_obstacles()

        assert set(results) == {"obs_a", "obs_b"}
        assert get_spans_mock.call_count == 1

    def test_min_distances_equal_obstacle_min_distance(
        self, pos_engine: PositionEngine
    ):
        pos_engine.add_obstacle(
            name="road",
            span_index=1,
            coords=np.array([[50, 0, 0], [100, 0, 10], [150, 10, 0]]),
        )
        pos_engine.add_obstacle(
            name="pt", span_index=0, coords=np.array([[100, 3, 8]])
        )
        batch = pos_engine.get_segment_distances_from_obstacles()

        for name in ("road", "pt"):
            single = pos_engine.obstacle_min_distance(name)
            assert batch[name] is not None
            assert single is not None
            assert batch[name].span_index == single.span_index
            np.testing.assert_allclose(
                batch[name].distance_result.distance_3d,
                single.distance_result.distance_3d,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                batch[name].closest_obstacle_point,
                single.closest_obstacle_point,
                atol=1e-2,
            )

    def test_several_polylines_on_two_spans(self, pos_engine: PositionEngine):
        polylines = {
            "poly_a": (0, np.array([[50, 0, 0], [120, 5, 0], [400, -5, 10]])),
            "poly_b": (0, np.array([[0, 0, 0], [400, 0, 0]])),
            "poly_c": (1, np.array([[60, 0, 5], [150, 10, 25], [260, 20, 10]])),
            "poly_d": (1, np.array([[0, -10, 0], [450, -10, 0]])),
        }
        for name, (span_index, coords) in polylines.items():
            pos_engine.add_obstacle(
                name=name, span_index=span_index, coords=coords
            )

        results = pos_engine.get_segment_distances_from_obstacles()

        for name, (span_index, _) in polylines.items():
            result = results[name]
            assert result is not None
            assert result.span_index == span_index

            single = pos_engine.obstacle_min_distance(name)
            assert single is not None
            np.testing.assert_allclose(
                result.distance_result.distance_3d,
                single.distance_result.distance_3d,
                atol=1e-9,
            )

    def test_batch_plane_distances_call_count_independent_of_obstacles(
        self,
        pos_engine: PositionEngine,
        balance_engine_base_test: BalanceEngine,
    ):
        def count_calls(nb_obstacles: int) -> int:
            engine = PositionEngine(balance_engine_base_test)
            for i in range(nb_obstacles):
                engine.add_obstacle(
                    name=f"obs_{i}",
                    span_index=1,
                    coords=np.array([[50, 0, 0], [100, 0, 10], [150, 10, 0]]),
                )
            original = position_engine_module.batch_plane_distances
            with patch.object(
                position_engine_module,
                "batch_plane_distances",
                wraps=original,
            ) as batch_mock:
                engine.get_segment_distances_from_obstacles()
            return batch_mock.call_count

        assert count_calls(2) == count_calls(6)

    def test_ground_obstacle_min_distance_at_cable_low_point(
        self, cable_array_AM600
    ):
        section_array = SectionArray(
            pd.DataFrame(
                {
                    "name": ["1", "2"],
                    "suspension": [False, False],
                    "conductor_attachment_altitude": [30, 30],
                    "crossarm_length": [0, 0],
                    "line_angle": [0, 0],
                    "insulator_length": [0, 0],
                    "span_length": [50, np.nan],
                    "insulator_mass": convert_weight_to_mass([0.0, 0.0]),
                }
            ),
            sagging_parameter=1500,
            sagging_temperature=15,
        )
        section_array.add_units({"line_angle": "grad"})
        balance_engine = BalanceEngine(
            cable_array=cable_array_AM600, section_array=section_array
        )
        pos_engine = PositionEngine(balance_engine)
        pos_engine.add_obstacle(
            name="ground",
            span_index=0,
            coords=np.array([[0, 0, 0], [50, 0, 0]]),
        )

        result = pos_engine.get_segment_distances_from_obstacles()["ground"]
        span_coords = pos_engine.coords_calculator.get_spans(
            frame="section"
        ).coords[0]

        assert result is not None
        np.testing.assert_allclose(
            result.closest_span_point[0],
            25.,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            result.closest_obstacle_point[0], 25.0, atol=1e-3
        )



class TestPositionEngineAdditionalPointsArray:
    @pytest.fixture(autouse=True)
    def setup(self, balance_engine_base_test):
        self.pos_engine = PositionEngine(balance_engine_base_test)

    def test_additional_points_array_init(self):
        assert self.pos_engine.additional_points_array is not None
        assert self.pos_engine.additional_points_array.data.empty

    def test_create_additional_point(self):
        point_coords = np.array([[10.0, 0.0, -5.0]])
        self.pos_engine.add_additional_point(
            "pt_0", span_index=0, coords=point_coords
        )
        assert len(self.pos_engine.additional_points_array.data) == 1
        assert (
            self.pos_engine.additional_points_array.data["name"].iloc[0]
            == "pt_0"
        )

        # Test coordinate transformation (similar to obstacles)
        # get_additional_points() uses points(True) which inserts NaN separators before each group,
        # so the first actual coordinate is at index 1.
        # translate_to_absolute_frame only translates x and y; z stays as input value.
        expected_x = (
            self.pos_engine.coords_calculator.supports_ground_coords[0][0]
            + 10.0
        )
        additional_points_coords = self.pos_engine.get_additional_points()

        np.testing.assert_allclose(additional_points_coords[1, 0], expected_x)
        np.testing.assert_allclose(additional_points_coords[1, 2], -5.0)

    def test_add_and_delete_additional_point(self):
        self.pos_engine.add_additional_point(
            "pt_0", span_index=0, coords=np.array([[10.0, 0.0, -5.0]])
        )
        self.pos_engine.add_additional_point(
            "pt_1", span_index=1, coords=np.array([[10.0, 0.0, -5.0]])
        )
        assert len(self.pos_engine.additional_points_array.data) == 2

        self.pos_engine.delete_additional_point("pt_0")
        assert len(self.pos_engine.additional_points_array.data) == 1
        assert (
            self.pos_engine.additional_points_array.data["name"].iloc[0]
            == "pt_1"
        )

    def test_delete_point_index(self):
        self.pos_engine.add_additional_point(
            "pt_multi",
            span_index=0,
            coords=np.array([[10.0, 0.0, -5.0], [20.0, 0.0, -5.0]]),
        )
        assert len(self.pos_engine.additional_points_array.data) == 2

        # When creating together, pandas index handles it.
        # point_index assigned could be sequential per obstacle. Let's delete the first point_index assigned
        point_idx_to_delete = self.pos_engine.additional_points_array.data[
            "point_index"
        ].iloc[0]

        self.pos_engine.delete_additional_point_by_index(
            "pt_multi", point_idx_to_delete
        )
        assert len(self.pos_engine.additional_points_array.data) == 1
        # delete_point renumbers remaining points, so the second point (x=20) gets index 0
        assert (
            self.pos_engine.additional_points_array.data["x"].iloc[0] == 20.0
        )
