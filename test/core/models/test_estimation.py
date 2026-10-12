# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from mechaphlowers.api.section_study import SectionStudy
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.core.models.estimation import (
    EstimationEngine,
    EstimationResult,
)
from mechaphlowers.core.numeric import (
    BisectionMethod,
    BrentMethod,
    NewtonMethod,
)
from mechaphlowers.entities.errors import EstimateOnSplitSupportError


@pytest.fixture
def solved_study(balance_engine_base_test: BalanceEngine) -> SectionStudy:
    """A SectionStudy that has already been solved via solve_adjustment."""
    study = SectionStudy(
        cable_array=balance_engine_base_test.cable_array,
        section_array=balance_engine_base_test.section_array,
    )
    study.solve_adjustment()
    return study


class TestEstimationResult:
    def test_repr_converged(self):
        r = EstimationResult(
            value=42.5, residual=1e-5, iterations=10, converged=True
        )
        assert "converged" in repr(r)
        assert "42.5" in repr(r)

    def test_repr_not_converged(self):
        r = EstimationResult(
            value=10.0, residual=5.0, iterations=50, converged=False
        )
        assert "NOT converged" in repr(r)


class TestOptimizationMethods:
    """Test the optimization methods on a simple analytical function."""

    def _simple_objective(self, x: float) -> float:
        """f(x) = x^2 - 4, root at x=2."""
        return x**2 - 4

    def test_bisection_converges(self):
        method = BisectionMethod(tol=1e-6, maxiter=100)
        result = method.solve(self._simple_objective, bounds=(0.0, 5.0))
        assert result.converged
        assert abs(result.value - 2.0) < 1e-5

    def test_brent_converges(self):
        method = BrentMethod(tol=1e-6, maxiter=100)
        result = method.solve(self._simple_objective, bounds=(0.0, 5.0))
        assert result.converged
        assert abs(result.value - 2.0) < 1e-5

    def test_newton_converges(self):
        method = NewtonMethod(tol=1e-6, maxiter=50, dx=0.01)
        result = method.solve(self._simple_objective, bounds=(0.0, 5.0))
        assert result.converged
        assert abs(result.value - 2.0) < 1e-4

    def test_bisection_no_sign_change(self):
        """When bounds don't bracket a root, bisection returns best bound."""
        method = BisectionMethod(tol=1e-6, maxiter=50)
        # f(3)=5, f(5)=21, both positive
        result = method.solve(self._simple_objective, bounds=(3.0, 5.0))
        assert not result.converged
        assert result.value == 3.0


class TestEstimationEngine:
    def test_estimate_generic(self, solved_study: SectionStudy):
        """Generic estimate() works with a simple function."""
        engine = EstimationEngine(
            solved_study, method=BisectionMethod(tol=1e-4)
        )
        result = engine.estimate(
            objective=lambda x: x**2 - 9,
            bounds=(0.0, 10.0),
        )
        assert result.converged
        assert abs(result.value - 3.0) < 1e-3

    def test_method_setter(self, solved_study: SectionStudy):
        engine = EstimationEngine(solved_study)
        assert isinstance(engine.method, BrentMethod)
        engine.method = NewtonMethod(tol=1e-3)
        assert isinstance(engine.method, NewtonMethod)

    def test_estimate_temperature(self, solved_study: SectionStudy):
        """Estimate temperature: solve at known temp, use that distance as target."""
        # First, solve at a known temperature to get a reference distance
        study = solved_study
        temperature = 60.0
        study.solve_change_state(new_temperature=temperature)
        ref_distance = study.position_engine.point_distance(
            span_index=0, point=np.array([250.0, 10.0, 20.0])
        ).distance_3d

        # Restore state
        study.solve_adjustment()

        # Now estimate what temperature gives that distance
        engine = EstimationEngine(study, method=BrentMethod(tol=0.1))
        result = engine.estimate_temperature(
            span_index=0,
            obstacle_point=np.array([250.0, 10.0, 20.0]),
            target_distance=ref_distance,
            bounds=(0.0, 120.0),
        )
        assert result.converged
        assert abs(result.value - temperature) < 0.1

    def test_estimate_wind(self, solved_study: SectionStudy):
        """Estimate wind pressure using a known reference."""
        study = solved_study
        wind_pressure = 400.0
        study.solve_change_state(wind_pressure=wind_pressure)
        ref_distance = study.position_engine.point_distance(
            span_index=0, point=np.array([250.0, 10.0, 20.0])
        ).distance_3d

        # Restore state
        study.solve_adjustment()

        engine = EstimationEngine(study, method=BrentMethod(tol=0.5))
        result = engine.estimate_wind(
            span_index=0,
            obstacle_point=np.array([250.0, 10.0, 20.0]),
            target_distance=ref_distance,
            bounds=(0.0, 1000.0),
        )
        assert result.converged
        assert abs(result.value - wind_pressure) < 0.1

    def test_state_preserved_after_estimation(
        self, solved_study: SectionStudy
    ):
        """Engine state is unchanged after estimation."""
        study = solved_study
        memento_before = study.save_state()

        engine = EstimationEngine(
            study, method=BisectionMethod(tol=1.0, maxiter=5)
        )
        engine.estimate_temperature(
            span_index=0,
            obstacle_point=np.array([250.0, 10.0, 20.0]),
            target_distance=50.0,
            bounds=(0.0, 100.0),
        )

        # State should be unchanged
        np.testing.assert_array_almost_equal(
            study.balance_engine.balance_model.nodes.dxdydz,
            memento_before.dxdydz,
        )


class TestEstimationSandbox:
    obstacle = np.array([250.0, 10.0, 20.0])

    def test_estimate_load_converges(self, solved_study: SectionStudy):
        study = solved_study
        n_spans = len(study._clean_engine.section_array.data) - 1
        positions = np.zeros(n_spans)
        masses = np.zeros(n_spans)
        positions[0] = 100.0
        masses[0] = 50.0
        study.set_loads(positions, masses)
        ref_distance = study.position_engine.point_distance(
            span_index=0, point=self.obstacle
        ).distance_3d
        study.set_loads(np.zeros(n_spans), np.zeros(n_spans))

        engine = EstimationEngine(study, method=BrentMethod(tol=0.1))
        result = engine.estimate_load(
            span_index=0,
            obstacle_point=self.obstacle,
            target_distance=ref_distance,
            load_position_distance=100.0,
            bounds=(0.0, 100.0),
        )
        assert result.converged
        assert abs(result.value - 50.0) < 1.0

    def test_study_untouched_after_estimate_load(
        self, solved_study: SectionStudy
    ):
        study = solved_study
        memento_before = study.intermediate_memento
        masses_before = np.copy(study._clean_engine.span_loads.load_mass)
        engine_masses_before = np.copy(
            study.balance_engine.span_loads.load_mass
        )
        engine = EstimationEngine(
            study, method=BisectionMethod(tol=1.0, maxiter=3)
        )
        engine.estimate_load(
            span_index=0,
            obstacle_point=self.obstacle,
            target_distance=50.0,
            load_position_distance=100.0,
            bounds=(0.0, 100.0),
        )
        assert study.intermediate_memento is memento_before
        np.testing.assert_array_equal(
            study._clean_engine.span_loads.load_mass, masses_before
        )
        np.testing.assert_array_equal(
            study.balance_engine.span_loads.load_mass, engine_masses_before
        )

    virtual_support = {
        0: {
            "x": 100.0,
            "y": 0.0,
            "z": 60.0,
            "insulator_length": 3.0,
            "insulator_mass": 500.0,
            "hanging_cable_point_from_left_support": 100.0,
        },
    }

    @pytest.mark.parametrize("fixed", [{}, {"new_temperature": 40.0}])
    def test_estimate_load_with_virtual_support_and_existing_load(
        self, solved_study: SectionStudy, fixed: dict
    ):
        study = solved_study
        study.set_virtual_support(self.virtual_support)
        obstacle = np.array([750.0, 10.0, 20.0])
        positions = np.array([0.0, 100.0, 200.0])
        study.set_loads(positions, np.array([0.0, 50.0, 30.0]))
        study.solve_change_state(**fixed)
        # Clean span 1 is span 2 after the virtual support insertion
        ref_distance = study.position_engine.point_distance(
            span_index=2, point=obstacle
        ).distance_3d
        study.set_loads(positions, np.array([0.0, 0.0, 30.0]))

        engine = EstimationEngine(study, method=BrentMethod(tol=0.1))
        result = engine.estimate_load(
            span_index=1,
            obstacle_point=obstacle,
            target_distance=ref_distance,
            load_position_distance=100.0,
            bounds=(0.0, 100.0),
            **fixed,
        )
        assert result.converged
        assert abs(result.value - 50.0) < 1.0

    def test_estimate_load_on_split_span_raises(
        self, solved_study: SectionStudy
    ):
        study = solved_study
        study.set_virtual_support(self.virtual_support)
        engine = EstimationEngine(study)
        with pytest.raises(
            EstimateOnSplitSupportError,
            match="split by a virtual support",
        ):
            engine.estimate_load(
                span_index=0,
                obstacle_point=self.obstacle,
                target_distance=30.0,
                load_position_distance=50.0,
            )
