# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from mechaphlowers.core.models.estimation.methods import (
    BrentMethod,
    OptimizationMethod,
)
from mechaphlowers.core.models.estimation.result import EstimationResult

if TYPE_CHECKING:
    from mechaphlowers.api.section_study import SectionStudy

logger = logging.getLogger(__name__)


class EstimationEngine:
    """Generic inverse-problem solver built on top of SectionStudy.

    Wraps a `SectionStudy` and an `OptimizationMethod` to find the value of
    a physical variable (temperature, wind, load) that produces a target
    distance to an obstacle.

    The engine saves/restores the balance-engine state around every objective
    evaluation so that the study is left unchanged after estimation.

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
        bounds: tuple[float, float] = (0.0, 200.0),
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

        def objective(temperature: float) -> float:
            return self._distance_residual(
                span_index=span_index,
                obstacle_point=obstacle_point,
                target_distance=target_distance,
                new_temperature=temperature,
                wind_pressure=wind_pressure,
                ice_thickness=ice_thickness,
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
        bounds: tuple[float, float] = (0.0, 2000.0),
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

        def objective(wind: float) -> float:
            return self._distance_residual(
                span_index=span_index,
                obstacle_point=obstacle_point,
                target_distance=target_distance,
                wind_pressure=wind,
                new_temperature=new_temperature,
                ice_thickness=ice_thickness,
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
        bounds: tuple[float, float] = (0.0, 100.0),
        new_temperature: float | None = None,
        wind_pressure: float | None = None,
        ice_thickness: float | None = None,
    ) -> EstimationResult:
        """Find the load mass that yields a target distance to an obstacle.

        Args:
            span_index: Index of the span where the obstacle is located.
            obstacle_point: 3D coordinates of the obstacle point (shape ``(3,)``).
            target_distance: Desired distance between cable and obstacle (meters).
            load_position_distance: Position of the load along the span (meters).
            bounds: Search interval for load mass in kg.
            new_temperature: Fixed temperature in °C (optional).
            wind_pressure: Fixed wind pressure in Pa (optional).
            ice_thickness: Fixed ice thickness in m (optional).

        Returns:
            EstimationResult with the load mass value.
        """

        def objective(load_mass: float) -> float:
            return self._load_distance_residual(
                span_index=span_index,
                obstacle_point=obstacle_point,
                target_distance=target_distance,
                load_position_distance=load_position_distance,
                load_mass=load_mass,
                new_temperature=new_temperature,
                wind_pressure=wind_pressure,
                ice_thickness=ice_thickness,
            )

        logger.info(
            "Estimating load mass for target distance %.3f m on span %d",
            target_distance,
            span_index,
        )
        return self._method.solve(objective, bounds)

    # ── Private helpers ───────────────────────────────────────────────────

    def _distance_residual(
        self,
        span_index: int,
        obstacle_point: np.ndarray,
        target_distance: float,
        wind_pressure: float | None = None,
        ice_thickness: float | None = None,
        new_temperature: float | None = None,
    ) -> float:
        """Compute ``distance(x) - target`` with state save/restore.

        Solves change-state with the given parameters, computes the distance
        to the obstacle, then restores the engine to its original state.
        """
        memento = self._study.save_state()
        try:
            self._study.solve_change_state(
                wind_pressure=wind_pressure,
                ice_thickness=ice_thickness,
                new_temperature=new_temperature,
            )
            distance_result = self._study.position_engine.point_distance(
                span_index, obstacle_point
            )
            distance = distance_result.distance_3d
        finally:
            self._study.restore_state(memento)

        return distance - target_distance

    def _load_distance_residual(
        self,
        span_index: int,
        obstacle_point: np.ndarray,
        target_distance: float,
        load_position_distance: float,
        load_mass: float,
        new_temperature: float | None = None,
        wind_pressure: float | None = None,
        ice_thickness: float | None = None,
    ) -> float:
        """Compute distance residual for a given load mass with state save/restore."""
        memento = self._study.save_state()
        try:
            # Build load arrays for single point load
            n_spans = len(
                self._study.balance_engine.section_array.data.span_length
            )
            load_positions = [[] for _ in range(n_spans)]
            load_masses = [[] for _ in range(n_spans)]
            load_positions[span_index] = [load_position_distance]
            load_masses[span_index] = [load_mass]

            self._study.add_loads(load_positions, load_masses)
            self._study.solve_change_state(
                wind_pressure=wind_pressure,
                ice_thickness=ice_thickness,
                new_temperature=new_temperature,
            )
            distance_result = self._study.position_engine.point_distance(
                span_index, obstacle_point
            )
            distance = distance_result.distance_3d
        finally:
            self._study.restore_state(memento)

        return distance - target_distance
