# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from mechaphlowers.config import options
from mechaphlowers.core.geometry.position_engine import PositionEngine
from mechaphlowers.core.numeric import (
    BrentMethod,
    EstimationResult,
    OptimizationMethod,
)
from mechaphlowers.entities.errors import (
    EstimateOnSplitSupportError,
    EstimationWarning,
    InvalidManipulationIndex,
)
from mechaphlowers.utils import arr

if TYPE_CHECKING:
    from mechaphlowers.api.section_study import SectionStudy

logger = logging.getLogger(__name__)


class EstimationEngine:
    """Generic inverse-problem solver built on top of SectionStudy.

    Wraps a `SectionStudy` and an `OptimizationMethod` to find the value of
    a physical variable (temperature, wind, load) that produces a target
    distance to an obstacle.

    Every `estimate_*` call evaluates the objective on a deep copy of the
    study's balance engine, so the study is left untouched.

    Args:
        study: The `SectionStudy` instance (must have been solved via
            `solve_adjustment` beforehand).
        method: The optimization algorithm to use. Defaults to `BrentMethod`.

    Examples:
        >>> engine = EstimationEngine(study, method=BrentMethod(tol=0.01))
        >>> result = engine.estimate_temperature(
        ...     span_index=0,
        ...     obstacle_point=np.array([150.0, 0.0, 5.0]),
        ...     target_distance=8.0,
        ...     bounds=(0.0, 200.0),
        ... )
        >>> print(result.value, result.converged)
    """

    _warning_existing_load = (
        "Existing load on estimation span target: load is ignored"
    )

    def __init__(
        self,
        study: SectionStudy,
        method: OptimizationMethod | None = None,
    ) -> None:
        self._study = study
        self._method: OptimizationMethod = method or BrentMethod()

    @property
    def method(self) -> OptimizationMethod:
        return self._method

    @method.setter
    def method(self, value: OptimizationMethod) -> None:
        self._method = value

    def estimate(
        self,
        objective: Callable[[float], float],
        bounds: tuple[float, float],
    ) -> EstimationResult:
        """Run the optimization on a generic objective function.

        The objective must be a callable ``f(x) -> float`` where the root
        ``f(x) = 0`` corresponds to the desired solution. State management
        (save/restore) is the caller's responsibility when using this method
        directly.

        Args:
            objective: Function to zero. Signature: ``(x: float) -> float``.
            bounds: ``(lower, upper)`` search interval.

        Returns:
            EstimationResult with the solution.
        """
        return self._method.solve(objective, bounds)

    def estimate_temperature(
        self,
        span_index: int,
        obstacle_point: np.ndarray,
        target_distance: float,
        bounds: tuple[float, float] = (-50.0, 200.0),
        wind_pressure: float | None = None,
        ice_thickness: float | None = None,
    ) -> EstimationResult:
        """Find the cable temperature that yields a target distance to an obstacle.

        Args:
            span_index: Index of the span where the obstacle is located.
            obstacle_point: 3D coordinates of the obstacle point (shape ``(3,)``).
            target_distance: Desired distance between cable and obstacle (meters).
            bounds: Search interval for temperature in °C.
            wind_pressure: Fixed wind pressure in Pa (optional).
            ice_thickness: Fixed ice thickness in m (optional).

        Returns:
            EstimationResult with the temperature value.
        """

        objective = self._build_distance_objective__climatic_case(
            span_index,
            obstacle_point,
            target_distance,
            "new_temperature",
            {"wind_pressure": wind_pressure, "ice_thickness": ice_thickness},
        )

        logger.info(
            "Estimating temperature for target distance %.3f m on span %d",
            target_distance,
            span_index,
        )
        return self._method.solve(objective, bounds)

    def estimate_wind(
        self,
        span_index: int,
        obstacle_point: np.ndarray,
        target_distance: float,
        bounds: tuple[float, float] = (-1000.0, 1000.0),
        new_temperature: float | None = None,
        ice_thickness: float | None = None,
    ) -> EstimationResult:
        """Find the wind pressure that yields a target distance to an obstacle.

        Args:
            span_index: Index of the span where the obstacle is located.
            obstacle_point: 3D coordinates of the obstacle point (shape ``(3,)``).
            target_distance: Desired distance between cable and obstacle (meters).
            bounds: Search interval for wind pressure in Pa.
            new_temperature: Fixed temperature in °C (optional).
            ice_thickness: Fixed ice thickness in m (optional).

        Returns:
            EstimationResult with the wind pressure value.
        """

        objective = self._build_distance_objective__climatic_case(
            span_index,
            obstacle_point,
            target_distance,
            "wind_pressure",
            {
                "new_temperature": new_temperature,
                "ice_thickness": ice_thickness,
            },
        )

        logger.info(
            "Estimating wind pressure for target distance %.3f m on span %d",
            target_distance,
            span_index,
        )
        return self._method.solve(objective, bounds)

    def estimate_load(
        self,
        span_index: int,
        obstacle_point: np.ndarray,
        target_distance: float,
        load_position_distance: float,
        bounds: tuple[float, float] = (0.0, 2000.0),
        new_temperature: float | None = None,
        wind_pressure: float | None = None,
        ice_thickness: float | None = None,
    ) -> EstimationResult:
        """Find the load mass that yields a target distance to an obstacle.

        Existing loads on other spans are kept. The load replaces any existing
        load on ``span_index``.

        Args:
            span_index: Index of the span (before manipulation) holding the load
                and the obstacle. Must not be split by a virtual support.
            obstacle_point: 3D coordinates of the obstacle point (shape ``(3,)``).
            target_distance: Desired distance between cable and obstacle (meters).
            load_position_distance: Position of the load along the span (meters).
            bounds: Search interval for load mass in kg.
            new_temperature: Fixed temperature in °C (optional).
            wind_pressure: Fixed wind pressure in Pa (optional).
            ice_thickness: Fixed ice thickness in m (optional).

        Returns:
            EstimationResult with the load mass value.

        Raises:
            EstimateOnSplitSupportError: If ``span_index`` is split by a virtual support.
        """

        objective = self._build_distance_objective__load_case(
            span_index,
            obstacle_point,
            target_distance,
            load_position_distance,
            {
                "new_temperature": new_temperature,
                "wind_pressure": wind_pressure,
                "ice_thickness": ice_thickness,
            },
        )

        logger.info(
            "Estimating load mass for target distance %.3f m on span %d",
            target_distance,
            span_index,
        )
        return self._method.solve(objective, bounds)

    def estimate_rope(
        self,
        support_index: int,
        span_index: int,
        obstacle_point: np.ndarray,
        target_vertical_distance: float,
        bounds: tuple[float, float] = (0.01, 10.0),
        rope_lineic_mass: float | None = None,
        new_temperature: float | None = None,
        wind_pressure: float | None = None,
        ice_thickness: float | None = None,
    ) -> EstimationResult:
        """Find the rope length that yields a target vertical distance above an obstacle.

        The vertical distance is ``z_cable - z_obstacle``, where ``z_cable`` is
        the altitude of the cable point lying in the vertical plane that
        contains the obstacle and is orthogonal to the span.

        Existing manipulations, loads and ropes on other supports are kept.
        The chain on ``support_index`` is replaced by the rope with
        `SectionStudy.set_rope`: the arm stays fixed, the cable attachment moves.

        Args:
            support_index: Index of the support (before manipulation) where
                the rope is set.
            span_index: Index of the span (before manipulation) where the
                obstacle is located. Must not be split by a virtual support.
            obstacle_point: 3D coordinates of the obstacle point (shape ``(3,)``).
            target_vertical_distance: Desired vertical distance between cable
                and obstacle (meters).
            bounds: Search interval for the rope length in meters.
            rope_lineic_mass: Linear mass of the rope in kg/m. Defaults to the
                rope lineic mass already set on the study, or to
                ``options.data.rope_lineic_mass_default``. A single value
                applies to all ropes, including those on other supports.
            new_temperature: Fixed temperature in °C (optional).
            wind_pressure: Fixed wind pressure in Pa (optional).
            ice_thickness: Fixed ice thickness in m (optional).

        Returns:
            EstimationResult with the rope length value.

        Raises:
            EstimateOnSplitSupportError: If ``span_index`` is split by a virtual support.
            InvalidManipulationIndex: If ``support_index`` is out of range.
        """

        objective = self._build_vertical_distance_objective__rope_case(
            support_index,
            span_index,
            obstacle_point,
            target_vertical_distance,
            rope_lineic_mass,
            {
                "new_temperature": new_temperature,
                "wind_pressure": wind_pressure,
                "ice_thickness": ice_thickness,
            },
        )

        logger.info(
            "Estimating rope length on support %d for target vertical distance %.3f m on span %d",
            support_index,
            target_vertical_distance,
            span_index,
        )
        return self._method.solve(objective, bounds)

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_distance_objective__climatic_case(
        self,
        span_index: int,
        obstacle_point: np.ndarray,
        target_distance: float,
        variable_name: str,
        fixed_conditions: dict[str, float | None],
    ) -> Callable[[float], float]:
        """Build ``f(x) = distance(x) - target`` on a deep copy of the balance engine."""
        original_balance_engine = self._study.balance_engine
        # Drop observers so the study's downstream engines are not copied
        balance_engine = deepcopy(
            original_balance_engine,
            {id(original_balance_engine._observers): []},
        )
        position_engine = PositionEngine(balance_engine)

        def objective(x: float) -> float:
            balance_engine.solve_change_state(**fixed_conditions, **{variable_name: x})
            result = position_engine.point_distance(span_index, obstacle_point)
            return result.distance_3d - target_distance

        return objective

    def _build_vertical_distance_objective__rope_case(
        self,
        support_index: int,
        span_index: int,
        obstacle_point: np.ndarray,
        target_vertical_distance: float,
        rope_lineic_mass: float | None,
        fixed_conditions: dict[str, Any],
    ) -> Callable[[float], float]:
        """Build ``f(rope_length) = vertical_distance - target`` on a copy of the study.

        Each evaluation applies the rope with `SectionStudy.set_rope`, then runs
        `SectionStudy.solve_change_state` with the fixed conditions.
        """
        manipulation = self._study.manipulation
        engine_span_index = manipulation.span_index_after_virtual_support(
            span_index
        )
        if engine_span_index is None:
            raise EstimateOnSplitSupportError(
                f"Span {span_index} is split by a virtual support: "
                "rope estimation is not possible on this span."
            )
        n_supports = len(manipulation._section_array._data)
        if support_index < 0 or support_index >= n_supports:
            raise InvalidManipulationIndex(
                f"Support index {support_index} is out of range [0, {n_supports - 1}]"
            )
        if rope_lineic_mass is None:
            rope_lineic_mass = (
                manipulation._rope_lineic_mass
                if manipulation._rope_lineic_mass is not None
                else options.data.rope_lineic_mass_default
            )
        # set_rope replaces the whole overlay: keep the ropes on other supports
        other_ropes = {
            idx: length
            for idx, length in (manipulation._rope_overlay or {}).items()
            if idx != support_index
        }

        study = self._study.copy()

        def objective(rope_length: float) -> float:
            study.set_rope(
                {**other_ropes, support_index: rope_length}, rope_lineic_mass
            )
            study.solve_change_state(**fixed_conditions)
            # apply_manipulations rebuilds the position engine: do not cache it
            result = study.position_engine.point_distance(
                engine_span_index, obstacle_point
            )
            vertical_distance = result.point_target[2] - result.point_base[2]
            return vertical_distance - target_vertical_distance

        return objective

    def _build_distance_objective__load_case(
        self,
        span_index: int,
        obstacle_point: np.ndarray,
        target_distance: float,
        load_position_distance: float,
        fixed_conditions: dict[str, float | None],
    ) -> Callable[[float], float]:
        """Build ``f(load_mass) = distance - target`` on a deep copy of the balance engine.

        Manipulations and existing loads are kept; only the load on ``span_index``
        (index before manipulation) is replaced.
        """
        span_index_after_manip = (
            self._study.manipulation.span_index_after_virtual_support(
                span_index
            )
        )
        if span_index_after_manip is None:
            raise EstimateOnSplitSupportError(
                f"Span {span_index} is split by a virtual support: "
                "load estimation is not possible on this span."
            )
        original_balance_engine = self._study.balance_engine
        # Drop observers so the study's downstream engines are not copied
        balance_engine = deepcopy(
            original_balance_engine,
            {id(original_balance_engine._observers): []},
        )
        position_engine = PositionEngine(balance_engine)
        span_length = arr.decr(
            balance_engine.section_array.data.span_length.to_numpy()
        )
        load_positions = balance_engine.span_loads.load_position * span_length
        load_masses = balance_engine.span_loads.load_mass.copy()
        load_positions[span_index_after_manip] = load_position_distance

        if not np.isclose(
            np.nan_to_num(load_masses[span_index_after_manip]),
            0.0,
            atol=1e-6,
        ):
            logger.warning(self._warning_existing_load)
            warnings.warn(self._warning_existing_load, EstimationWarning)

        has_fixed_conditions = any(
            value is not None for value in fixed_conditions.values()
        )

        def objective(load_mass: float) -> float:
            load_masses[span_index_after_manip] = load_mass
            balance_engine.set_loads(load_positions, load_masses)
            if has_fixed_conditions:
                # set_loads resets the nodes: a cold solve with ice or wind may diverge
                balance_engine.solve_change_state()
            balance_engine.solve_change_state(**fixed_conditions)
            result = position_engine.point_distance(
                span_index_after_manip, obstacle_point
            )
            return result.distance_3d - target_distance

        return objective
