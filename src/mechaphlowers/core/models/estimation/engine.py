# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal

import numpy as np

from mechaphlowers.config import options
from mechaphlowers.core.geometry.position_engine import PositionEngine
from mechaphlowers.core.models.estimation.distance import (
    ComputeDistance,
    DistanceType,
)
from mechaphlowers.core.numeric import (
    BrentMethod,
    BrentMinimizationMethod,
    EstimationResult,
    MinimizationMethod,
    OptimizationMethod,
)
from mechaphlowers.entities.errors import (
    EstimateOnSplitSupportError,
    EstimationWarning,
    InvalidManipulationIndex,
    SolverError,
)
from mechaphlowers.utils import arr

if TYPE_CHECKING:
    from mechaphlowers.api.section_study import SectionStudy
    from mechaphlowers.core.models.balance.engine import BalanceEngine

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
        minimizer: The minimization algorithm used by closest-point
            estimations. Defaults to `BrentMinimizationMethod`.

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
        minimizer: MinimizationMethod | None = None,
    ) -> None:
        self._study = study
        self._method: OptimizationMethod = method or BrentMethod()
        self._minimizer: MinimizationMethod = (
            minimizer or BrentMinimizationMethod()
        )

    @property
    def method(self) -> OptimizationMethod:
        return self._method

    @method.setter
    def method(self, value: OptimizationMethod) -> None:
        self._method = value

    @property
    def minimizer(self) -> MinimizationMethod:
        return self._minimizer

    @minimizer.setter
    def minimizer(self, value: MinimizationMethod) -> None:
        self._minimizer = value

    def estimate(
        self,
        objective: Callable[[float], float],
        bounds: tuple[float, float],
        tol: float | None = None,
    ) -> EstimationResult:
        """Run the optimization on a generic objective function.

        The objective must be a callable ``f(x) -> float`` where the root
        ``f(x) = 0`` corresponds to the desired solution. State management
        (save/restore) is the caller's responsibility when using this method
        directly.

        Args:
            objective: Function to zero. Signature: ``(x: float) -> float``.
            bounds: ``(lower, upper)`` search interval.

            tol: Tolerance set on the engine method before the search starts
                (restored afterwards). If None, the method's own tolerance is used.
        Returns:
            EstimationResult with the solution.
        """
        with self._set_tolerance(self._method, tol):
            return self._method.solve(objective, bounds)

    def estimate_temperature(
        self,
        span_index: int,
        obstacle_point: np.ndarray,
        target_distance: float,
        bounds: tuple[float, float] = (-50.0, 200.0),
        wind_pressure: float | None = None,
        ice_thickness: float | None = None,
        distance_type: DistanceType = "vertical",
        tol: float | None = None,
    ) -> EstimationResult:
        """Find the cable temperature that yields a target distance to an obstacle.

        Args:
            span_index: Index of the span where the obstacle is located.
            obstacle_point: 3D coordinates of the obstacle point (shape ``(3,)``).
            target_distance: Desired distance between cable and obstacle (meters).
            bounds: Search interval for temperature in °C.
            wind_pressure: Fixed wind pressure in Pa (optional).
            ice_thickness: Fixed ice thickness in m (optional).
            distance_type: Distance used as the optimization target:
                ``"distance_3d"`` (unsigned 3D distance),
                ``"horizontal"`` (signed component along the plane ``u`` axis) or
                ``"vertical"`` (signed ``z_cable - z_obstacle``).
                Defaults to ``"vertical"``.

            tol: Tolerance set on the engine method before the search starts
                (restored afterwards). If None, the method's own tolerance is used.
        Returns:
            EstimationResult with the temperature value.
        """

        objective = self._build_distance_objective__climatic_case(
            span_index,
            obstacle_point,
            target_distance,
            "new_temperature",
            {"wind_pressure": wind_pressure, "ice_thickness": ice_thickness},
            distance_type=distance_type,
        )

        logger.info(
            "Estimating temperature for target distance %.3f m on span %d",
            target_distance,
            span_index,
        )
        with self._set_tolerance(self._method, tol):
            return self._method.solve(objective, bounds)

    def estimate_wind(
        self,
        span_index: int,
        obstacle_point: np.ndarray,
        target_distance: float | None,
        bounds: tuple[float, float] = (-1000.0, 1000.0),
        new_temperature: float | None = None,
        ice_thickness: float | None = None,
        wind_direction: Literal[
            "clockwise", "anticlockwise"
        ] = "anticlockwise",
        distance_type: DistanceType = "distance_3d",
        tol: float | None = None,
    ) -> EstimationResult:
        """Find the wind pressure that yields a target distance to an obstacle.

        If ``target_distance`` is None, the closest-point mode finds the wind
        pressure within ``bounds`` that minimizes the distance to the obstacle.
        It minimizes the distance against wind pressure with the engine
        ``minimizer``, so it assumes a single local minimum of the distance
        inside ``bounds``. If the minimum lies on a bound, that bound is
        returned with ``converged=False``.

        Args:
            span_index: Index of the span where the obstacle is located.
            obstacle_point: 3D coordinates of the obstacle point (shape ``(3,)``).
            target_distance: Desired distance between cable and obstacle (meters),
                or None for the closest-point mode described above.
            bounds: Search interval for wind pressure in Pa.
            new_temperature: Fixed temperature in °C (optional).
            ice_thickness: Fixed ice thickness in m (optional).
            wind_direction: Direction of positive wind pressure: "clockwise"
                towards the user (right), "anticlockwise" away from the user
                (left). Defaults to "anticlockwise". Negative pressure reverses
                the selected direction.
            distance_type: Distance used as the optimization target:
                ``"distance_3d"`` (unsigned 3D distance),
                ``"horizontal"`` (signed component along the plane ``u`` axis) or
                ``"vertical"`` (signed ``z_cable - z_obstacle``).
                Defaults to ``"distance_3d"``. In closest-point mode, the
                absolute value of the selected distance is minimized.

            tol: Tolerance set on the engine method (or minimizer in
                closest-point mode) before the search starts
                (restored afterwards). If None, the solver's own tolerance is used.
        Returns:
            EstimationResult with the wind pressure value. ``extra["distance"]``
            holds the selected distance at that wind pressure (signed for
            ``"horizontal"`` and ``"vertical"`` in target mode; absolute value
            in closest-point mode).

        Raises:
            ValueError: If ``wind_direction`` is not "clockwise" or "anticlockwise".
        """

        fixed_conditions: dict[str, float | None] = {
            "new_temperature": new_temperature,
            "ice_thickness": ice_thickness,
        }

        if target_distance is None:
            logger.info(
                "Estimating wind pressure for closest distance to obstacle on span %d",
                span_index,
            )
            return self._estimate_closest_distance__climatic_case(
                span_index,
                obstacle_point,
                "wind_pressure",
                fixed_conditions,
                bounds,
                wind_direction=wind_direction,
                distance_type=distance_type,
                tol=tol,
            )

        distance_at = self._build_climatic_distance_function(
            span_index,
            obstacle_point,
            "wind_pressure",
            fixed_conditions,
            wind_direction=wind_direction,
            distance_type=distance_type,
        )

        def objective(x: float) -> float:
            return distance_at(x) - target_distance

        logger.info(
            "Estimating wind pressure for target distance %.3f m on span %d",
            target_distance,
            span_index,
        )
        with self._set_tolerance(self._method, tol):
            result = self._method.solve(objective, bounds)
        return replace(
            result, extra={"distance": float(distance_at(result.value))}
        )

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
        distance_type: DistanceType = "distance_3d",
        tol: float | None = None,
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
            distance_type: Distance used as the optimization target:
                ``"distance_3d"`` (unsigned 3D distance),
                ``"horizontal"`` (signed component along the plane ``u`` axis) or
                ``"vertical"`` (signed ``z_cable - z_obstacle``).
                Defaults to ``"distance_3d"``.

            tol: Tolerance set on the engine method before the search starts
                (restored afterwards). If None, the method's own tolerance is used.
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
            distance_type=distance_type,
        )

        logger.info(
            "Estimating load mass for target distance %.3f m on span %d",
            target_distance,
            span_index,
        )
        with self._set_tolerance(self._method, tol):
            return self._method.solve(objective, bounds)

    def estimate_load_relative(
        self,
        span_index: int,
        target_distance: float,
        load_position_distance: float,
        bounds: tuple[float, float] = (0.0, 2000.0),
        new_temperature: float | None = None,
        wind_pressure: float | None = None,
        ice_thickness: float | None = None,
        tol: float | None = None,
    ) -> EstimationResult:
        """Find the load mass that yields a target altitude change of the cable.

        The altitude (z) of the cable at the abscissa ``load_position_distance``
        is read on the current state of the study (initial altitude). The load
        mass is then optimized so that ``new altitude - initial altitude`` equals
        ``target_distance`` at the same abscissa. A load lowers the cable, so
        ``target_distance`` is usually negative.

        Existing loads on other spans are kept. The load replaces any existing
        load on ``span_index``.

        Args:
            span_index: Index of the span (before manipulation) holding the load.
                Must not be split by a virtual support.
            target_distance: Desired altitude change of the cable at
                ``load_position_distance`` (meters).
            load_position_distance: Position of the load along the span, from
                the left attachment (meters). It is also the abscissa where the
                altitude is measured.
            bounds: Search interval for load mass in kg.
            new_temperature: Fixed temperature in °C (optional).
            wind_pressure: Fixed wind pressure in Pa (optional).
            ice_thickness: Fixed ice thickness in m (optional).

            tol: Tolerance set on the engine method before the search starts
                (restored afterwards). If None, the method's own tolerance is used.
        Returns:
            EstimationResult with the load mass value. ``extra["x_shift"]``
            holds ``initial x_cable - final x_cable`` (meters) at the load point.

        Raises:
            EstimateOnSplitSupportError: If ``span_index`` is split by a virtual support.
        """

        # set target state

        objective, x_shift = self._build_relative_altitude_objective__load_case(
            span_index,
            target_distance,
            load_position_distance,
            {
                "new_temperature": new_temperature,
                "wind_pressure": wind_pressure,
                "ice_thickness": ice_thickness,
            },
        )

        logger.info(
            "Estimating load mass for target altitude change %.3f m on span %d",
            target_distance,
            span_index,
        )

        with self._set_tolerance(self._method, tol):
            result = self._method.solve(objective, bounds)
        # put the state at the solution before reading the final x
        objective(result.value)
        return replace(result, extra={"x_shift": float(x_shift())})

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
        distance_type: DistanceType = "vertical",
        tol: float | None = None,
    ) -> EstimationResult:
        """Find the rope length that yields a target distance above an obstacle.

        By default, the vertical distance is ``z_cable - z_obstacle``, where
        ``z_cable`` is the altitude of the cable point lying in the vertical
        plane that contains the obstacle and is orthogonal to the span. It is
        signed: a negative value means the obstacle is above the cable.

        Existing manipulations, loads and ropes on other supports are kept.
        The chain on ``support_index`` is replaced by the rope with
        `SectionStudy.set_rope_or_chain`: the arm stays fixed, the cable attachment moves.

        Args:
            support_index: Index of the support (before manipulation) where
                the rope is set.
            span_index: Index of the span (before manipulation) where the
                obstacle is located. Must not be split by a virtual support.
            obstacle_point: 3D coordinates of the obstacle point (shape ``(3,)``).
            target_vertical_distance: Desired signed vertical distance between
                cable and obstacle (meters), ``z_cable - z_obstacle`` by default.
                This value is used as the target for the selected
                ``distance_type``.
            bounds: Search interval for the rope length in meters.
            rope_lineic_mass: Linear mass of the rope in kg/m. Defaults to the
                rope lineic mass already set on the study, or to
                ``options.data.rope_lineic_mass_default``. A single value
                applies to all ropes, including those on other supports.
            new_temperature: Fixed temperature in °C (optional).
            wind_pressure: Fixed wind pressure in Pa (optional).
            ice_thickness: Fixed ice thickness in m (optional).
            distance_type: Distance used as the optimization target:
                ``"distance_3d"`` (unsigned 3D distance),
                ``"horizontal"`` (signed component along the plane ``u`` axis) or
                ``"vertical"`` (signed ``z_cable - z_obstacle``).
                Defaults to ``"vertical"``.

            tol: Tolerance set on the engine method before the search starts
                (restored afterwards). If None, the method's own tolerance is used.
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
            distance_type=distance_type,
        )

        logger.info(
            "Estimating rope length on support %d for target vertical distance %.3f m on span %d",
            support_index,
            target_vertical_distance,
            span_index,
        )
        with self._set_tolerance(self._method, tol):
            return self._method.solve(objective, bounds)

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    @contextmanager
    def _set_tolerance(solver: Any, tol: float | None) -> Generator[None]:
        """Temporarily set ``solver.tol`` before the search starts."""
        if tol is None:
            yield
            return
        if not hasattr(solver, "tol"):
            raise ValueError(
                f"{type(solver).__name__} has no 'tol' attribute: cannot set tolerance"
            )
        previous = solver.tol
        solver.tol = tol
        try:
            yield
        finally:
            solver.tol = previous

    def _build_distance_objective__climatic_case(
        self,
        span_index: int,
        obstacle_point: np.ndarray,
        target_distance: float,
        variable_name: str,
        fixed_conditions: dict[str, float | None],
        wind_direction: Literal[
            "clockwise", "anticlockwise"
        ] = "anticlockwise",
        distance_type: DistanceType = "vertical",
    ) -> Callable[[float], float]:
        """Build ``f(x) = distance(x) - target`` for the climatic variable."""
        distance_at = self._build_climatic_distance_function(
            span_index,
            obstacle_point,
            variable_name,
            fixed_conditions,
            wind_direction=wind_direction,
            distance_type=distance_type,
        )

        def residual(x: float) -> float:
            return distance_at(x) - target_distance

        return residual

    def _build_distance_objective__load_case(
        self,
        span_index: int,
        obstacle_point: np.ndarray,
        target_distance: float,
        load_position_distance: float,
        fixed_conditions: dict[str, float | None],
        distance_type: DistanceType = "distance_3d",
    ) -> Callable[[float], float]:
        """Build ``f(load_mass) = distance - target`` on a deep copy of the balance engine.

        Manipulations and existing loads are kept; only the load on ``span_index``
        (index before manipulation) is replaced.
        """
        compute_distance = ComputeDistance(distance_type)

        def residual_factory(
            position_engine: PositionEngine, span_index_after_manip: int
        ) -> Callable[[], float]:
            return compute_distance.compute_distance_factory(
                lambda: position_engine.point_distance(
                    span_index_after_manip, obstacle_point
                ),
                target_distance,
            )

        return self._build_load_objective(
            span_index,
            load_position_distance,
            fixed_conditions,
            residual_factory,
        )

    def _build_climatic_distance_function(
        self,
        span_index: int,
        obstacle_point: np.ndarray,
        variable_name: str,
        fixed_conditions: dict[str, float | None],
        wind_direction: Literal[
            "clockwise", "anticlockwise"
        ] = "anticlockwise",
        distance_type: DistanceType = "vertical",
    ) -> Callable[[float], float]:
        """Build ``d(x)``: the selected distance once the climatic case is solved.

        Each call solves the balance engine copy with ``variable_name=x``. The
        result is signed for ``"horizontal"`` and ``"vertical"``.
        """
        compute_distance = ComputeDistance(distance_type)
        original_balance_engine = self._study.balance_engine

        def copy_engines() -> tuple[BalanceEngine, PositionEngine]:
            # Drop observers so the study's downstream engines are not copied
            balance_engine = deepcopy(
                original_balance_engine,
                {id(original_balance_engine._observers): []},
            )
            return balance_engine, PositionEngine(balance_engine)

        balance_engine, position_engine = copy_engines()

        def distance_at(x: float) -> float:
            # nonlocal allows us to reassign balance_engine and position_engine when executing function several times
            nonlocal balance_engine, position_engine
            solve_kwargs: dict[str, Any] = {
                **fixed_conditions,
                variable_name: x,
                "wind_direction": wind_direction,
            }
            try:
                balance_engine.solve_change_state(**solve_kwargs)
            except SolverError:
                # A warm start from a distant state can fail: retry on a fresh copy
                # it can happen in case of high wind variations
                balance_engine, position_engine = copy_engines()
                balance_engine.solve_change_state(**solve_kwargs)
            return compute_distance.select_distance(
                position_engine.point_distance(span_index, obstacle_point)
            )

        return distance_at

    def _build_vertical_distance_objective__rope_case(
        self,
        support_index: int,
        span_index: int,
        obstacle_point: np.ndarray,
        target_vertical_distance: float,
        rope_lineic_mass: float | None,
        fixed_conditions: dict[str, Any],
        distance_type: DistanceType = "vertical",
    ) -> Callable[[float], float]:
        """Build ``f(rope_length) = selected_distance - target`` on a copy of the study.

        Each evaluation applies the rope with `SectionStudy.set_rope_or_chain`, then runs
        `SectionStudy.solve_change_state` with the fixed conditions.
        """
        compute_distance = ComputeDistance(distance_type)
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
            overlay = manipulation._rope_or_chain_overlay
            rope_lineic_mass = overlay.get(support_index, {}).get(
                "lineic_mass", options.data.rope_lineic_mass_default
            )
        other_ropes = {
            idx: other_dict
            for idx, other_dict in manipulation._rope_or_chain_overlay.items()
            if idx != support_index
        }

        study = self._study.copy()
        # apply_manipulations rebuilds the position engine: the lambda must not cache it
        residual = compute_distance.compute_distance_factory(
            lambda: study.position_engine.point_distance(
                engine_span_index, obstacle_point
            ),
            target_vertical_distance,
        )

        def objective(rope_length: float) -> float:
            study.set_rope_or_chain(
                {
                    **other_ropes,
                    support_index: {
                        "type": "rope",
                        "length": rope_length,
                        "lineic_mass": rope_lineic_mass,
                    },
                }
            )
            study.solve_change_state(**fixed_conditions)
            return residual()

        return objective

    def _build_relative_altitude_objective__load_case(
        self,
        span_index: int,
        target_distance: float,
        load_position_distance: float,
        fixed_conditions: dict[str, float | None],
    ) -> tuple[Callable[[float], float], Callable[[], float]]:
        """Build ``f(load_mass) = (z(x) - z_initial(x)) - target`` on a deep copy of the balance engine.

        ``x`` is ``load_position_distance``. ``z_initial`` is read on the current
        state of the study, before the load and the fixed conditions are applied.
        Also returns a function giving ``x_initial - x`` of the cable at the load
        point, for the state of the last objective evaluation.
        """
        x_shift_getters: list[Callable[[], float]] = []

        def residual_factory(
            position_engine: PositionEngine, span_index_after_manip: int
        ) -> Callable[[], float]:
            if not position_engine.get_loads_coords_group_points():
                raise ValueError("No load coordinates available")
            initial_altitude = position_engine.get_loads_coords_group_points()[
                span_index_after_manip
            ].copy()

            def residual() -> float:
                altitude = position_engine.get_loads_coords_group_points()[
                    span_index_after_manip
                ]
                # take z component
                return altitude[2] - initial_altitude[2] - target_distance

            def x_shift() -> float:
                altitude = position_engine.get_loads_coords_group_points()[
                    span_index_after_manip
                ]
                return initial_altitude[0] - altitude[0]

            x_shift_getters.append(x_shift)
            return residual

        objective = self._build_load_objective(
            span_index,
            load_position_distance,
            fixed_conditions,
            residual_factory,
        )
        return objective, x_shift_getters[0]

    def _build_load_objective(
        self,
        span_index: int,
        load_position_distance: float,
        fixed_conditions: dict[str, float | None],
        residual_factory: Callable[[PositionEngine, int], Callable[[], float]],
    ) -> Callable[[float], float]:
        """Build ``f(load_mass)`` on a deep copy of the balance engine.

        Manipulations and existing loads are kept; only the load on ``span_index``
        (index before manipulation) is replaced. ``residual_factory`` is called
        once, before any change of state, with the position engine and the span
        index after manipulation. It returns the function evaluated on the
        updated state.
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

        # get the current load positions and masses for the span after manipulation
        load_positions = balance_engine.span_loads.load_position * span_length
        load_masses = balance_engine.span_loads.load_mass.copy()
        load_positions[span_index_after_manip] = load_position_distance
        # in case there is no existing load, set a small default value
        load_masses[span_index_after_manip] = (
            load_masses[span_index_after_manip] or 0.01
        )

        if not np.isclose(
            np.nan_to_num(load_masses[span_index_after_manip]),
            0.01,
            atol=1e-2,
        ):
            logger.warning(self._warning_existing_load)
            warnings.warn(self._warning_existing_load, EstimationWarning)

        has_fixed_conditions = any(
            value is not None for value in fixed_conditions.values()
        )

        # set initial state for balance engine
        balance_engine.set_loads(load_positions, load_masses)
        balance_engine.solve_change_state(**fixed_conditions)

        residual = residual_factory(position_engine, span_index_after_manip)

        def objective(load_mass: float) -> float:
            load_masses[span_index_after_manip] = load_mass
            balance_engine.set_loads(load_positions, load_masses)
            if has_fixed_conditions:
                # set_loads resets the nodes: a cold solve with ice or wind may diverge
                balance_engine.solve_change_state()
            balance_engine.solve_change_state(**fixed_conditions)
            return residual()

        return objective

    def _estimate_closest_distance__climatic_case(
        self,
        span_index: int,
        obstacle_point: np.ndarray,
        variable_name: str,
        fixed_conditions: dict[str, float | None],
        bounds: tuple[float, float],
        wind_direction: Literal[
            "clockwise", "anticlockwise"
        ] = "anticlockwise",
        distance_type: DistanceType = "vertical",
        tol: float | None = None,
    ) -> EstimationResult:
        """Find the climatic value giving the closest distance to the obstacle.

        Minimizes ``|d(x)|`` on ``bounds`` with the engine ``minimizer``. If
        the minimum lies on a bound, that bound is returned, not converged.
        """
        distance_at = self._build_climatic_distance_function(
            span_index,
            obstacle_point,
            variable_name,
            fixed_conditions,
            wind_direction=wind_direction,
            distance_type=distance_type,
        )

        def absolute_distance(x: float) -> float:
            return abs(distance_at(x))

        with self._set_tolerance(self._minimizer, tol):
            result = self._minimizer.minimize(absolute_distance, bounds)
        distance = result.residual
        if not result.converged and result.value in bounds:
            logger.warning(
                "Closest distance to obstacle lies on a bound of %s: returning %s "
                "with distance %.3f m, not converged.",
                bounds,
                result.value,
                distance,
            )
        return replace(result, extra={"distance": distance})
