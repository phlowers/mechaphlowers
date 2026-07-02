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
    BisectionMethod,
    BrentMethod,
    EstimationEngine,
    EstimationResult,
    NewtonMethod,
)


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
    def test_lazy_property_access(self, solved_study: SectionStudy):
        """estimation_engine property creates the engine lazily."""
        engine = solved_study.estimation_engine
        assert isinstance(engine, EstimationEngine)
        # Second access returns same instance
        assert solved_study.estimation_engine is engine

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
