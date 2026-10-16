# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import sys
import warnings
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
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
)
from mechaphlowers.data.units import convert_weight_to_mass
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.entities.errors import (
    EstimateOnSplitSupportError,
    EstimationWarning,
    InvalidManipulationIndex,
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

    def test_bisection_no_sign_change(self):
        """When bounds don't bracket a root, bisection returns best bound."""
        method = BisectionMethod(tol=1e-6, maxiter=50)
        # f(3)=5, f(5)=21, both positive
        result = method.solve(self._simple_objective, bounds=(3.0, 5.0))
        assert not result.converged
        assert result.value == 3.0

    def test_bisection_maxiter_exhausted(self):
        method = BisectionMethod(tol=1e-6, maxiter=0)
        result = method.solve(self._simple_objective, bounds=(0.0, 5.0))
        assert not result.converged
        assert result.value == 2.5

    def test_brent_falls_back_without_scipy(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setitem(sys.modules, "scipy.optimize", None)
        method = BrentMethod(tol=1e-6, maxiter=100)
        result = method.solve(self._simple_objective, bounds=(0.0, 5.0))
        assert result.converged
        assert abs(result.value - 2.0) < 1e-5

    def test_brent_falls_back_without_sign_change(self):
        method = BrentMethod(tol=1e-6, maxiter=50)
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
        engine.method = BisectionMethod(tol=1e-3)
        assert isinstance(engine.method, BisectionMethod)

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
        assert abs(result.value - wind_pressure) < 5.0

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

        engine = EstimationEngine(study, method=BrentMethod(tol=5))
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

        engine = EstimationEngine(study, method=BrentMethod(tol=5))
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


N_SPANS = 4
SPAN_LENGTH = 500.0
IGNORED_LOAD_WARNING = (
    "Existing load on estimation span target: load is ignored"
)

# Virtual supports on the two first spans, so span 2 stays unsplit
VIRTUAL_SUPPORTS = {
    0: {
        "x": 100.0,
        "y": 0.0,
        "z": 60.0,
        "insulator_length": 3.0,
        "insulator_mass": 500.0,
        "hanging_cable_point_from_left_support": 100.0,
    },
    1: {
        "x": 100.0,
        "y": 0.0,
        "z": 80.0,
        "insulator_length": 3.0,
        "insulator_mass": 500.0,
        "hanging_cable_point_from_left_support": 100.0,
    },
}


@dataclass(frozen=True)
class Scenario:
    """Study configuration; span indices are the ones before manipulation."""

    span: int
    # span -> (position in m, mass in kg), may include a load on `span`
    loads: dict[int, tuple[float, float]] = field(default_factory=dict)
    # Load to retrieve with estimate_load, used to build the reference distance
    target_load: tuple[float, float] = (100.0, 50.0)
    virtual_support: dict | None = None
    modify_support: dict | None = None
    shift_support: dict | None = None
    conditions: dict = field(default_factory=dict)

    @property
    def has_load_on_span(self) -> bool:
        return self.span in self.loads


SCENARIOS = {
    "no_load_no_manipulation": Scenario(span=0),
    "one_load_estimation_on_another_span": Scenario(
        span=0, loads={1: (200.0, 40.0)}
    ),
    "two_loads_estimation_on_existing_load": Scenario(
        span=0, loads={0: (100.0, 30.0), 2: (200.0, 40.0)}
    ),
    "load_virtual_supports_shift_and_modify_support_ice": Scenario(
        span=2,
        loads={3: (200.0, 40.0)},
        target_load=(100.0, 20.0),
        virtual_support=VIRTUAL_SUPPORTS,
        modify_support={2: {"z": 1.0}},
        shift_support={2: 1.0},
        conditions={"ice_thickness": 0.01},
    ),
}


@contextmanager
def no_estimation_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", EstimationWarning)
        yield


def set_study_loads(
    study: SectionStudy, loads: dict[int, tuple[float, float]]
) -> None:
    positions = np.zeros(N_SPANS)
    masses = np.zeros(N_SPANS)
    for span, (position, mass) in loads.items():
        positions[span] = position
        masses[span] = mass
    study.set_loads(positions, masses)


def build_study(cable_array: CableArray, scenario: Scenario) -> SectionStudy:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4", "5"],
                "suspension": [False, True, True, True, False],
                "conductor_attachment_altitude": [50, 100, 50, 60, 50],
                "crossarm_length": [10, 10, 10, 10, 10],
                "line_angle": [0, 0, 0, 0, 0],
                "insulator_length": [3, 3, 3, 3, 3],
                "span_length": [SPAN_LENGTH] * N_SPANS + [np.nan],
                "insulator_mass": convert_weight_to_mass(
                    [1000.0, 500.0, 500.0, 500.0, 1000.0]
                ),
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    study = SectionStudy(cable_array=cable_array, section_array=section_array)
    study.solve_adjustment()
    if scenario.virtual_support:
        study.set_virtual_support(scenario.virtual_support)
    if scenario.modify_support:
        study.modify_support(scenario.modify_support)
    if scenario.shift_support:
        study.modify_cable(shift_support=scenario.shift_support)
    set_study_loads(study, scenario.loads)
    return study


def obstacle_point(scenario: Scenario) -> np.ndarray:
    """Absolute coordinates, below the middle of the span."""
    return np.array([SPAN_LENGTH * (scenario.span + 0.5), 10.0, 20.0])


def reference_distance(
    study: SectionStudy,
    scenario: Scenario,
    loads: dict[int, tuple[float, float]] | None = None,
    **conditions: Any,
) -> float:
    """Distance to the obstacle once solved on a copy, with the true value applied."""
    reference = study.copy()
    if loads is not None:
        set_study_loads(reference, loads)
    reference.solve_change_state(**conditions)
    return reference.position_engine.point_distance(
        engine_span_index(study, scenario), obstacle_point(scenario)
    ).distance_3d


def engine_span_index(study: SectionStudy, scenario: Scenario) -> int:
    index = study.manipulation.span_index_after_virtual_support(scenario.span)
    assert index is not None
    return index


@pytest.fixture(params=list(SCENARIOS.values()), ids=list(SCENARIOS))
def scenario(request) -> Scenario:
    return request.param


@pytest.fixture
def scenario_study(
    cable_array_AM600: CableArray, scenario: Scenario
) -> SectionStudy:
    return build_study(cable_array_AM600, scenario)


class TestEstimationScenarios:
    """Each estimation is run on every scenario; the known value must be retrieved."""

    def test_estimate_temperature(
        self, scenario_study: SectionStudy, scenario: Scenario
    ):
        temperature = 60.0
        target = reference_distance(
            scenario_study,
            scenario,
            new_temperature=temperature,
            **scenario.conditions,
        )
        engine = EstimationEngine(scenario_study, method=BrentMethod(tol=0.1))
        with no_estimation_warning():
            result = engine.estimate_temperature(
                span_index=engine_span_index(scenario_study, scenario),
                obstacle_point=obstacle_point(scenario),
                target_distance=target,
                bounds=(0.0, 120.0),
                **scenario.conditions,
            )
        assert result.converged
        assert abs(result.value - temperature) < 0.5

    def test_estimate_wind(
        self, scenario_study: SectionStudy, scenario: Scenario
    ):
        wind_pressure = 200.0
        target = reference_distance(
            scenario_study,
            scenario,
            wind_pressure=wind_pressure,
            **scenario.conditions,
        )
        engine = EstimationEngine(scenario_study, method=BrentMethod(tol=0.5))
        with no_estimation_warning():
            result = engine.estimate_wind(
                span_index=engine_span_index(scenario_study, scenario),
                obstacle_point=obstacle_point(scenario),
                target_distance=target,
                bounds=(0.0, 500.0),
                **scenario.conditions,
            )
        assert result.converged
        assert abs(result.value - wind_pressure) < 5.0

    def test_estimate_load(
        self, scenario_study: SectionStudy, scenario: Scenario
    ):
        position, mass = scenario.target_load
        target = reference_distance(
            scenario_study,
            scenario,
            loads={**scenario.loads, scenario.span: scenario.target_load},
            **scenario.conditions,
        )
        masses_before = (
            scenario_study._clean_engine.span_loads.load_mass.copy()
        )
        engine = EstimationEngine(scenario_study, method=BrentMethod(tol=5))
        expected_warning: AbstractContextManager[Any]
        if scenario.has_load_on_span:
            expected_warning = pytest.warns(
                EstimationWarning, match=IGNORED_LOAD_WARNING
            )
        else:
            expected_warning = no_estimation_warning()
        with expected_warning:
            result = engine.estimate_load(
                span_index=scenario.span,
                obstacle_point=obstacle_point(scenario),
                target_distance=target,
                load_position_distance=position,
                bounds=(0.0, 2 * mass),
                **scenario.conditions,
            )
        assert result.converged
        assert abs(result.value - mass) < 1.0
        # The study keeps its own loads
        np.testing.assert_array_equal(
            scenario_study._clean_engine.span_loads.load_mass, masses_before
        )

    def test_estimate_rope(
        self, scenario_study: SectionStudy, scenario: Scenario
    ):
        rope_length = 5.0
        # Right support of the span (index before manipulation)
        support_index = scenario.span + 1
        span_index = engine_span_index(scenario_study, scenario)
        obstacle = obstacle_point(scenario)

        reference = scenario_study.copy()
        reference.set_rope({support_index: rope_length})
        reference.solve_change_state(**scenario.conditions)
        distance = reference.position_engine.point_distance(
            span_index, obstacle
        )
        target = distance.point_target[2] - obstacle[2]

        memento_before = scenario_study.save_state()
        balance_engine_before = scenario_study.balance_engine
        engine = EstimationEngine(scenario_study, method=BrentMethod(tol=0.1))
        with no_estimation_warning():
            result = engine.estimate_rope(
                support_index=support_index,
                span_index=scenario.span,
                obstacle_point=obstacle,
                target_vertical_distance=target,
                bounds=(1.0, 10.0),
                **scenario.conditions,
            )
        assert result.converged
        assert abs(result.value - rope_length) < 0.05
        # The study is left untouched
        assert scenario_study.manipulation._rope_overlay is None
        assert scenario_study.balance_engine is balance_engine_before
        np.testing.assert_array_equal(
            scenario_study.balance_engine.balance_model.nodes.dxdydz,
            memento_before.dxdydz,
        )


def rope_target(
    study: SectionStudy,
    rope: dict[int, float],
    span_index: int,
    obstacle: np.ndarray,
) -> float:
    """Vertical distance above the obstacle once the ropes are set on a copy."""
    reference = study.copy()
    reference.set_rope(rope)
    reference.solve_change_state()
    distance = reference.position_engine.point_distance(span_index, obstacle)
    return distance.point_target[2] - obstacle[2]


class TestEstimateRope:
    obstacle = np.array([250.0, 10.0, 20.0])

    @pytest.fixture
    def study(self, cable_array_AM600: CableArray) -> SectionStudy:
        return build_study(cable_array_AM600, Scenario(span=0))

    def estimate(self, study: SectionStudy, target: float, **kwargs):
        engine = EstimationEngine(study, method=BrentMethod(tol=0.1))
        return engine.estimate_rope(
            span_index=0,
            obstacle_point=self.obstacle,
            target_vertical_distance=target,
            bounds=(1.0, 10.0),
            **kwargs,
        )

    def test_rope_on_other_support_is_kept(self, study: SectionStudy):
        target = rope_target(study, {1: 5.0, 2: 2.0}, 0, self.obstacle)
        study.set_rope({2: 2.0})
        result = self.estimate(study, target, support_index=1)
        assert result.converged
        assert abs(result.value - 5.0) < 0.05
        assert study.manipulation._rope_overlay == {2: 2.0}

    def test_rope_on_support_is_replaced(self, study: SectionStudy):
        target = rope_target(study, {1: 5.0}, 0, self.obstacle)
        study.set_rope({1: 2.0})
        result = self.estimate(study, target, support_index=1)
        assert result.converged
        assert abs(result.value - 5.0) < 0.05
        assert study.manipulation._rope_overlay == {1: 2.0}

    def test_split_span_raises(self, study: SectionStudy):
        study.set_virtual_support({0: VIRTUAL_SUPPORTS[0]})
        with pytest.raises(EstimateOnSplitSupportError):
            self.estimate(study, 30.0, support_index=1)

    @pytest.mark.parametrize("support_index", [-1, N_SPANS + 1])
    def test_out_of_range_support_raises(
        self, study: SectionStudy, support_index: int
    ):
        with pytest.raises(InvalidManipulationIndex):
            self.estimate(study, 30.0, support_index=support_index)

    def test_arm_altitude_is_kept(self, study: SectionStudy):
        target = rope_target(study, {1: 5.0}, 0, self.obstacle)
        result = self.estimate(study, target, support_index=1)
        roped = study.copy()
        roped.set_rope({1: result.value})
        roped.solve_change_state()
        np.testing.assert_allclose(
            roped.position_engine.coords_calculator.center_arm_coords[:, 2],
            study.position_engine.coords_calculator.center_arm_coords[:, 2],
        )
