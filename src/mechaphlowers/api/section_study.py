# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Type

import numpy as np
from typing_extensions import Literal

from mechaphlowers.core.geometry.points import Points
from mechaphlowers.core.geometry.position_engine import PositionEngine
from mechaphlowers.core.manipulation import Manipulation
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.core.models.balance.memento import (
    BalanceEngineCaretaker,
    BalanceEngineMemento,
)
from mechaphlowers.core.models.cable.deformation import (
    DeformationRte,
    IDeformation,
)
from mechaphlowers.core.models.cable.span import CatenarySpan, ISpan
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.entities.core import VhlResult
from mechaphlowers.entities.errors import SolverError

if TYPE_CHECKING:
    from mechaphlowers.core.models.cable.thermal import ThermalEngine
    from mechaphlowers.core.models.guying import Guying
    from mechaphlowers.plotting.plot import PlotEngine

logger = logging.getLogger(__name__)


class SectionStudy:
    """User-facing facade that bundles all engines for a power-line section.

    `SectionStudy` creates a
    [`BalanceEngine`][mechaphlowers.core.models.balance.engine.BalanceEngine] and a
    [`PositionEngine`][mechaphlowers.core.geometry.position_engine.PositionEngine]
    eagerly, wiring the observer chain so that every solve automatically
    refreshes downstream geometry.  A
    [`PlotEngine`][mechaphlowers.plotting.plot.PlotEngine],
    [`ThermalEngine`][mechaphlowers.core.models.cable.thermal.ThermalEngine],
    and [`Guying`][mechaphlowers.core.models.guying.Guying] are created lazily
    on first access to avoid unnecessary dependencies (e.g. *plotly*).

    All state-management features (save / restore, rollback on solver error,
    intermediate warm-start) live here so that
    [`BalanceEngine`][mechaphlowers.core.models.balance.engine.BalanceEngine]
    is never modified.

    Args:
        cable_array (CableArray): Cable specification data.
        section_array (SectionArray): Support / section data.
        span_model_type (Type[ISpan], optional): Span model class. Defaults to CatenarySpan.
        deformation_model_type (Type[IDeformation], optional): Deformation model class. Defaults to DeformationRte.

    Examples:
        >>> study = SectionStudy(cable_array, section_array)
        >>> study.solve_adjustment()
        >>> study.solve_change_state(wind_pressure=200, new_temperature=90)
        >>> points = study.get_supports_points()
    """

    def __init__(
        self,
        cable_array: CableArray,
        section_array: SectionArray,
        span_model_type: Type[ISpan] = CatenarySpan,
        deformation_model_type: Type[IDeformation] = DeformationRte,
    ) -> None:
        self._cable_array = cable_array
        self._section_array = section_array
        self._span_model_type = span_model_type
        self._deformation_model_type = deformation_model_type
        self._manipulation = Manipulation(section_array)

        self._balance_engine = BalanceEngine(
            cable_array=cable_array,
            section_array=section_array,
            span_model_type=span_model_type,
            deformation_model_type=deformation_model_type,
        )
        self._caretaker = BalanceEngineCaretaker(self._balance_engine)
        self._position_engine = PositionEngine(self._balance_engine)
        self._plot_engine: PlotEngine | None = None
        self._thermal_engine: ThermalEngine | None = None
        self._guying: Guying | None = None
        self._intermediate_memento: BalanceEngineMemento | None = None

    # ── Sub-engine properties ─────────────────────────────────────────────

    @property
    def balance_engine(self) -> BalanceEngine:
        return self._balance_engine

    @property
    def position_engine(self) -> PositionEngine:
        return self._position_engine

    @property
    def plot_engine(self) -> PlotEngine:
        if self._plot_engine is None:
            from mechaphlowers.plotting.plot import PlotEngine as _PE

            self._plot_engine = _PE(self._position_engine)
        return self._plot_engine

    @property
    def thermal_engine(self) -> ThermalEngine:
        if self._thermal_engine is None:
            from mechaphlowers.core.models.cable.thermal import (
                ThermalEngine as _TE,
            )

            self._thermal_engine = _TE()
        return self._thermal_engine

    @property
    def guying(self) -> Guying:
        if self._guying is None:
            from mechaphlowers.core.models.guying import Guying as _G

            self._guying = _G(self._balance_engine)
        return self._guying

    @property
    def intermediate_memento(self) -> BalanceEngineMemento | None:
        """The memento captured after the intermediate warm-start solve, if any."""
        return self._intermediate_memento

    @property
    def manipulation(self) -> Manipulation:
        """The :class:`Manipulation` object storing geometric overlays."""
        return self._manipulation

    # ── Manipulation methods ──────────────────────────────────────────────

    def support_manipulation(
        self, manipulation: dict[int, dict[str, float]]
    ) -> None:
        """Apply additive offsets to support geometry.

        Delegates to
        :meth:`Manipulation.support_manipulation`.

        Args:
            manipulation: Dictionary mapping support index (0-based) to
                offsets with optional keys ``"y"`` and ``"z"``.
        """
        self._manipulation.support_manipulation(manipulation)

    def reset_manipulation(self) -> None:
        """Remove the support manipulation overlay.

        Delegates to :meth:`Manipulation.reset_manipulation`.
        """
        self._manipulation.reset_manipulation()

    def rope_manipulation(
        self,
        rope: dict[int, float],
        rope_lineic_mass: float | None = None,
    ) -> None:
        """Override insulator length and mass for specified supports with rope values.

        Delegates to :meth:`Manipulation.rope_manipulation`.

        Args:
            rope: Dictionary mapping support index (0-based) to rope length (meters).
            rope_lineic_mass: Linear mass of the rope in kg/m.
        """
        self._manipulation.rope_manipulation(rope, rope_lineic_mass)

    def reset_rope_manipulation(self) -> None:
        """Remove the rope overlay.

        Delegates to :meth:`Manipulation.reset_rope_manipulation`.
        """
        self._manipulation.reset_rope_manipulation()

    def add_virtual_support(
        self, virtual_support: dict[int, dict[str, float]]
    ) -> None:
        """Insert virtual supports.

        Delegates to :meth:`Manipulation.add_virtual_support`.

        Args:
            virtual_support: Dictionary mapping left-support index to virtual
                support parameters.
        """
        self._manipulation.add_virtual_support(virtual_support)

    def reset_virtual_support(self) -> None:
        """Remove all virtual supports.

        Delegates to :meth:`Manipulation.reset_virtual_support`.
        """
        self._manipulation.reset_virtual_support()

    def add_cable_shifting(
        self,
        shift_support: np.ndarray | list | None = None,
        shorten_span: np.ndarray | list | None = None,
    ) -> None:
        """Validate and store cable shifting values.

        Delegates to :meth:`Manipulation.add_cable_shifting`.

        Args:
            shift_support (np.ndarray | list | None): Horizontal shifting of each support, in meters.
            shorten_span (np.ndarray | list | None): Span length modification, in meters.
        """
        self._manipulation.add_cable_shifting(shift_support, shorten_span)

    def reset_cable_shifting(self) -> None:
        """Remove cable shifting.

        Delegates to :meth:`Manipulation.reset_cable_shifting`.
        """
        self._manipulation.reset_cable_shifting()

    # ── Solve methods (with rollback + intermediate) ──────────────────────

    def solve_adjustment(self) -> None:
        """Run adjustment on clean geometry, then apply manipulations if any.

        1. Build a clean engine from the original section array and solve
           adjustment to obtain ``initial_L_ref``.
        2. If manipulations are registered, call
           :meth:`Manipulation.from_section_array` to produce a manipulated
           copy, then :meth:`Manipulation.initialize_engine` to build the
           target engine with injected ``L_ref`` and blocked adjustment.
        3. Rewire downstream engines (caretaker, position, plot, guying).

        On [`SolverError`][mechaphlowers.entities.errors.SolverError], the engine
        state is restored to the snapshot taken before the solve attempt, and the
        error is re-raised.

        Raises:
            SolverError: If the solver fails to converge.
        """
        if self._manipulation.has_manipulations:
            # Phase 1 – solve on clean geometry
            clean_engine = BalanceEngine(
                cable_array=self._cable_array,
                section_array=self._section_array,
                span_model_type=self._span_model_type,
                deformation_model_type=self._deformation_model_type,
            )
            clean_engine.solve_adjustment()
            initial_L_ref = clean_engine.initial_L_ref.copy()

            # Phase 2 – build manipulated SA and target engine
            manipulated_sa = self._manipulation.from_section_array(
                self._section_array
            )
            self._balance_engine = self._manipulation.initialize_engine(
                clean_engine, manipulated_sa, initial_L_ref
            )

            # Rewire downstream engines
            self._caretaker = BalanceEngineCaretaker(self._balance_engine)
            self._position_engine = PositionEngine(self._balance_engine)
            self._plot_engine = None
            self._guying = None
        else:
            memento = self._caretaker.save()
            try:
                self._balance_engine.solve_adjustment()
            except SolverError:
                logger.error(
                    "Error during solve_adjustment, rolling back state."
                )
                self._caretaker.restore(memento)
                raise

    def solve_change_state(
        self,
        wind_pressure: np.ndarray | float | None = None,
        ice_thickness: np.ndarray | float | None = None,
        new_temperature: np.ndarray | float | None = None,
        wind_sense: Literal["clockwise", "anticlockwise"] = "anticlockwise",
    ) -> None:
        """Run [`BalanceEngine.solve_change_state`][mechaphlowers.core.models.balance.engine.BalanceEngine.solve_change_state] with automatic rollback.

        An intermediate solve at default conditions (T=15 °C, wind=0, ice=0)
        is performed first as a warm-start to improve convergence, unless the
        requested conditions already match the defaults.

        On [`SolverError`][mechaphlowers.entities.errors.SolverError], the engine
        state is restored to the snapshot taken before the solve attempt, and the
        error is re-raised.

        Args:
            wind_pressure (np.ndarray | float | None): Wind pressure in Pa. Defaults to None.
            ice_thickness (np.ndarray | float | None): Ice thickness in m. Defaults to None.
            new_temperature (np.ndarray | float | None): New temperature in °C. Defaults to None.
            wind_sense (Literal["clockwise", "anticlockwise"]): Direction of the wind. Defaults to "anticlockwise".

        Raises:
            SolverError: If the solver fails to converge.
        """
        engine = self._balance_engine
        default = engine.default_value

        span_shape = engine.section_array.data.span_length.shape

        def _to_array(val: np.ndarray | float | None, name: str) -> np.ndarray:
            if val is None:
                return np.full(span_shape, default[name])
            if isinstance(val, (int, float)):
                return np.full(span_shape, val)
            return val

        target_wind = _to_array(wind_pressure, "wind_pressure")
        target_ice = _to_array(ice_thickness, "ice_thickness")
        target_temp = _to_array(new_temperature, "new_temperature")

        is_default = (
            np.allclose(target_wind, default["wind_pressure"])
            and np.allclose(target_ice, default["ice_thickness"])
            and np.allclose(target_temp, default["new_temperature"])
        )

        memento = self._caretaker.save()
        try:
            if not is_default:
                self._solve_intermediate()

            engine.solve_change_state(
                wind_pressure=wind_pressure,
                ice_thickness=ice_thickness,
                new_temperature=new_temperature,
                wind_sense=wind_sense,
            )
        except SolverError:
            logger.error(
                "Error during solve_change_state, rolling back state."
            )
            self._caretaker.restore(memento)
            raise

    def _solve_intermediate(self) -> None:
        """Solve at default conditions (T=15°C, wind=0, ice=0) as warm-start.

        The resulting node displacements are kept as the starting point for
        the actual target solve, improving convergence for extreme conditions.
        The intermediate result is stored in
        [`intermediate_memento`][mechaphlowers.api.section_study.SectionStudy.intermediate_memento].
        """
        logger.debug("Intermediate warm-start solve at default conditions.")
        engine = self._balance_engine
        default = engine.default_value

        engine.solve_change_state(
            wind_pressure=default["wind_pressure"],
            ice_thickness=default["ice_thickness"],
            new_temperature=default["new_temperature"],
        )
        self._intermediate_memento = self._caretaker.save()

    def add_loads(
        self,
        load_position_distance: np.ndarray | list,
        load_mass: np.ndarray | list,
    ) -> None:
        """Delegate to [`BalanceEngine.add_loads`][mechaphlowers.core.models.balance.engine.BalanceEngine.add_loads].

        Args:
            load_position_distance (np.ndarray | list): Position of the loads, in meters.
            load_mass (np.ndarray | list): Mass of the loads.
        """
        self._balance_engine.add_loads(load_position_distance, load_mass)

    # ── State management ──────────────────────────────────────────────────

    def save_state(self) -> BalanceEngineMemento:
        """Create an immutable snapshot of the current engine state.

        Returns:
            BalanceEngineMemento: Independent copy of every mutable array.
        """
        return self._caretaker.save()

    def restore_state(self, memento: BalanceEngineMemento) -> None:
        """Restore state and notify observers to refresh geometry.

        Args:
            memento (BalanceEngineMemento): Snapshot previously returned by
                [`save_state`][mechaphlowers.api.section_study.SectionStudy.save_state].
        """
        self._caretaker.restore(memento)
        self._balance_engine.notify()

    # ── Data retrieval delegates ──────────────────────────────────────────

    def get_data_spans(self) -> dict[str, list]:
        """Delegate to [`BalanceEngine.get_data_spans`][mechaphlowers.core.models.balance.engine.BalanceEngine.get_data_spans].

        Returns:
            dict[str, list]: Dictionary with span data (parameter, tensions, etc.).
        """
        return self._balance_engine.get_data_spans()

    def get_spans_points(
        self, frame: Literal["section", "localsection", "cable"] = "section"
    ) -> np.ndarray:
        """Delegate to [`PositionEngine.get_spans_points`][mechaphlowers.core.geometry.position_engine.PositionEngine.get_spans_points].

        Args:
            frame (Literal["section", "localsection", "cable"]): Coordinate frame. Defaults to "section".

        Returns:
            np.ndarray: Array of span points in the requested frame.
        """
        return self._position_engine.get_spans_points(frame)

    def get_supports_points(self) -> np.ndarray:
        """Delegate to [`PositionEngine.get_supports_points`][mechaphlowers.core.geometry.position_engine.PositionEngine.get_supports_points].

        Returns:
            np.ndarray: Array of support points coordinates.
        """
        return self._position_engine.get_supports_points()

    def get_points_for_plot(
        self, project=False, frame_index=0
    ) -> tuple[Points, Points, Points]:
        """Delegate to [`PositionEngine.get_points_for_plot`][mechaphlowers.core.geometry.position_engine.PositionEngine.get_points_for_plot].

        Args:
            project: `True` to project into a support frame (2-D mode).
            frame_index: Index of the support frame for projection.

        Returns:
            Tuple of ``(spans, supports, insulators)`` as `Points`.
        """
        return self._position_engine.get_points_for_plot(project, frame_index)

    def chain_displacement(self) -> np.ndarray:
        return self._balance_engine.balance_model.chain_displacement()

    def vhl_under_chain(self) -> VhlResult:
        return self._balance_engine.balance_model.vhl_under_chain()

    def vhl_under_console(self) -> VhlResult:
        return self._balance_engine.balance_model.vhl_under_console()

    def supports_number(self) -> int:
        """Return the number of supports in the balance engine."""
        return self._balance_engine.support_number

    # ── Representation ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"SectionStudy(\n{self._balance_engine!r}\n)"

    def __str__(self) -> str:
        return repr(self)
