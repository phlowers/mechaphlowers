# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations

import logging
import warnings
from typing import Callable, Type

import numpy as np
from typing_extensions import Literal

from mechaphlowers.config import options
from mechaphlowers.core.models.balance.interfaces import IBalanceModel
from mechaphlowers.core.models.balance.models.model_ducloux import BalanceModel
from mechaphlowers.core.models.balance.solvers.balance_solver import (
    BalanceSolver,
)
from mechaphlowers.core.models.cable.deformation import (
    DeformationRte,
    IDeformation,
    deformation_model_builder,
)
from mechaphlowers.core.models.cable.span import (
    CatenarySpan,
    ISpan,
    span_model_builder,
)
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.entities.core import QuantityArray
from mechaphlowers.entities.errors import BalanceEngineWarning, SolverError
from mechaphlowers.entities.reactivity import Notifier
from mechaphlowers.utils import arr, check_time

logger = logging.getLogger(__name__)


class DisplacementResult:
    def __init__(
        self,
        dxdydz: np.ndarray,
    ):
        self.dxdydz = dxdydz


class BalanceEngine(Notifier):
    """Engine for solving insulator chains positions.

    After solving any situation, many attributes are updated in the models.

    Most interesting ones are

    * `self.L_ref` for solve_adjustment()

    * `self.balance_model.nodes.dxdydz` and `self.span_model.sagging_parameter` for solve_change_state().

    Examples:

            >>> balance_engine = BalanceEngine(cable_array, section_array)
            >>> balance_engine.solve_adjustment()
            >>> wind_pressure = np.array([...])  # in Pa
            >>> ice_thickness = np.array([...])  # in m
            >>> new_temperature = np.array([...])  # in °C
            >>> balance_engine.solve_change_state(
            ...     wind_pressure, ice_thickness, new_temperature
            ... )

    Args:
        cable_array (CableArray): Cable data
        section_array (SectionArray): Section data
        span_model_type (Type[Span], optional): Span model to use. Defaults to CatenarySpan.
        deformation_model_type (Type[IDeformation], optional): Deformation model to use. Defaults to DeformationRte.
    """

    default_value = {
        "wind_pressure": 0.0,
        "ice_thickness": 0.0,
        "new_temperature": 15.0,
    }
    _warning_no_L_ref = "L_ref is not defined. You must run solve_adjustment() before solve_change_state(). Running solve_adjustment() now."

    def __init__(
        self,
        cable_array: CableArray,
        section_array: SectionArray,
        balance_model_type: Type[IBalanceModel] = BalanceModel,
        span_model_type: Type[ISpan] = CatenarySpan,
        deformation_model_type: Type[IDeformation] = DeformationRte,
    ) -> None:
        # TODO: find a better way to initialize objects
        self.section_array = section_array
        self.cable_array = cable_array
        self.balance_model_type = balance_model_type
        self.span_model_type = span_model_type
        self.deformation_model_type = deformation_model_type
        self._shifting_distance_support: np.ndarray = np.zeros_like(
            self.section_array.data.conductor_attachment_altitude.to_numpy()
        )
        self._shortening_distance_span: np.ndarray = np.zeros(
            len(self.section_array.data.conductor_attachment_altitude) - 1
        )

        self.reset(full=True)

    def reset(self, full: bool = False) -> None:
        """Reset the balance engine to initial state.

        This method re-initializes the span model, cable loads, deformation model, balance model, and solvers.
        This method is useful when an error occurs during solving that may cause an inconsistent state with NaN values.
        """

        logger.debug("Resetting balance engine.")

        if full:
            self.initialized = False
            zeros_vector = np.zeros_like(
                self.section_array.data.conductor_attachment_altitude.to_numpy()
            )
            sagging_temperature = arr.decr(
                (self.section_array.data.sagging_temperature.to_numpy())
            )
            parameter = arr.decr(
                self.section_array.data.sagging_parameter.to_numpy()
            )
            self.span_model = span_model_builder(
                self.section_array, self.cable_array, self.span_model_type
            )
            self.cable_loads = CableLoads(
                np.float64(self.cable_array.data.diameter.iloc[0]),
                np.float64(self.cable_array.data.linear_weight.iloc[0]),
                zeros_vector,
                zeros_vector,
            )
            self.deformation_model = deformation_model_builder(
                self.cable_array,
                self.span_model,
                sagging_temperature,
                self.deformation_model_type,
            )
            super().__init__()
            self.balance_model = self.balance_model_type(
                sagging_temperature,
                parameter,
                self.section_array,
                self.cable_array,
                self.span_model,
                self.deformation_model,
                self.cable_loads,
            )
        else:
            self.balance_model.reset(
                cable_array=self.cable_array,
                span_model=self.span_model,
                deformation_model=self.deformation_model,
                cable_loads=self.cable_loads,
                full=full,
            )

        if full:
            self.solver_change_state = BalanceSolver(
                **options.solver.balance_solver_change_state_params
            )
            self.solver_adjustment = BalanceSolver(
                **options.solver.balance_solver_adjustment_params
            )
            self.L_ref: np.ndarray

        self.get_displacement: Callable[[], np.ndarray] = (
            self.balance_model.chain_displacement
        )

        self.notify()
        self.initialized = True

        logger.debug("Balance engine initialized.")

    def add_loads(
        self,
        load_position_distance: np.ndarray | list,
        load_mass: np.ndarray | list,
    ) -> None:
        """Adds loads to BalanceEngine.
        Updates load_position and load_mass fields in SectionArray.

        Input for position is a distance, and will be converted into ratio to match SectionArray.

        Expected input are arrays of size matching the number of supports. Each value refers to a span.

        Args:
            load_position_distance (np.ndarray | list): Position of the loads, in meters
            load_mass (np.ndarray | list): Mass of the loads

        Raises:
            ValueError: if load_position_distance is not in [0, span_length] for at least one span

        Examples:
            >>> load_position_distance = np.array([150, 200, 0, np.nan])  # 4 supports/3 spans
            >>> load_mass = np.array([500, 70, 0, np.nan])
            >>> engine.add_loads(load_position_distance, load_mass)
            >>> plot_engine.reset()  # necessary if plot_engine already exists
        """
        span_length = self.section_array.data["span_length"].to_numpy()
        load_position_distance = np.array(load_position_distance)
        if (
            arr.decr(load_position_distance > span_length).any()
            or arr.decr(load_position_distance < 0).any()
        ):
            raise ValueError(
                f"{load_position_distance=} should be all between 0 and {span_length=}"
            )

        # This formula for load_position_ratio may change later
        load_position_ratio = load_position_distance / span_length
        self.section_array._data["load_position"] = load_position_ratio
        self.section_array._data["load_mass"] = load_mass

        self.reset(full=False)
        debug_loads = (
            "Loads have been added. PlotEngine will be notified automatically "
            "via the observer pattern; no manual reset is required."
        )
        logger.debug(debug_loads)

    @property
    def shift_support(self) -> np.ndarray:
        """Shifting distance, in meters."""
        return self._shifting_distance_support

    @property
    def shortening_span(self) -> np.ndarray:
        """shortening distance, in meters."""
        return self._shortening_distance_span

    def add_cable_shifting(
        self,
        shift_support: np.ndarray | list | None = None,
        shorten_span: np.ndarray | list | None = None,
    ) -> None:
        """Adds cable shifting to BalanceEngine.
        Updates internal shifting and span-length modification fields.

        Expected input are arrays whose sizes match the number of supports.
        Warning: shift shorten the left span and lengthen the right span.
        So if you want to shorten the right span, you should input a negative value.

        Args:
            shift_support (np.ndarray | list | None): Horizontal shifting of each support, in meters.
                Array of length ``support_number`` (support based). The first and last values
                are enforced to 0. If ``None``, an array of zeros is used.
            shorten_span (np.ndarray | list | None): Span length modification, in meters.
                Array of length ``support_number - 1`` (span based), one value per span.
                Positive values shorten the spans, negative values shorten them. If ``None``,
                an array of zeros is used.

        Raises:
            ValueError: if input arrays don't have correct size.

        Examples:
            >>> balance_engine.add_cable_shifting(
            ...     shorten_span=np.array([1.5, 0, 0]),
            ...     shift_support=np.array([0, 1, 0.5, 0]),
            ... )
            >>> balance_engine.shift_shorten_cable()
            >>> balance_engine.L_ref
            # Output: array of shifted span lengths, in meters.
            >>> balance_engine.solve_change_state(wind_pressure=0.0, new_temperature=15.0)
            >>> balance_engine.parameter
            # Output: array of sagging parameters, updated after shifting and shortening the cable.
        """
        # Convert to numpy arrays
        shift_support = (
            np.array(shift_support, dtype=np.float64)
            if shift_support is not None
            else np.zeros(self.support_number)
        )
        shorten_span = (
            np.array(shorten_span, dtype=np.float64)
            if shorten_span is not None
            else np.zeros(self.support_number - 1)
        )

        # Check size matches number of supports
        expected_size = self.support_number
        if shift_support.size != expected_size:
            raise ValueError(
                f"shift_support has incorrect size: {expected_size} is expected, received {shift_support.size}"
            )
        expected_span_size = self.support_number - 1
        if shorten_span.size != expected_span_size:
            raise ValueError(
                f"shorten_span has incorrect size: {expected_span_size} is expected, received {shorten_span.size}"
            )

        # Enforce constraints: shifting_distance first and last are 0
        if abs(shift_support[0]) > 0.0 or abs(shift_support[-1]) > 0.0:
            logger.warning(
                "shift_support first and last values must be 0 (support based). "
                "Enforcing this constraint."
            )
            warnings.warn(
                "First and last values of shift_support have been reset to 0",
                BalanceEngineWarning,
            )
        shift_support[0] = 0.0
        shift_support[-1] = 0.0

        # Store in private attributes
        self._shifting_distance_support = shift_support
        self._shortening_distance_span = shorten_span

        # apply
        self.shift_shorten_cable()

        # Reset and notify observers
        self.reset(full=False)
        debug_msg = (
            "Cable shifting has been added. PlotEngine will be notified automatically "
            "via the observer pattern; no manual reset is required."
        )
        logger.debug(debug_msg)

    def support_manipulation(
        self, manipulation: dict[int, dict[str, float]]
    ) -> None:
        """Apply additive offsets to support geometry.

        Delegates to
        [`SectionArray.support_manipulation`][mechaphlowers.entities.arrays.SectionArray.support_manipulation],
        then rebuilds internal models while preserving observer bindings.

        Args:
            manipulation: Dictionary mapping support index (0-based) to
                offsets. Each value is a dict with optional keys:

                - `"y"`: added to `crossarm_length` (meters)
                - `"z"`: added to `conductor_attachment_altitude` (meters)

        Raises:
            ValueError: If a support index is out of range.
            ValueError: If an inner dict contains keys other than `"y"` or `"z"`.

        Examples:
            >>> balance_engine.support_manipulation({1: {"z": 2.0, "y": -1.0}})
            >>> balance_engine.solve_adjustment()
            >>> balance_engine.solve_change_state(new_temperature=15.0)
        """
        self.section_array.support_manipulation(manipulation)

    def reset_manipulation(self) -> None:
        """Restore support geometry to the original values before any manipulation.

        Delegates to
        [`SectionArray.reset_manipulation`][mechaphlowers.entities.arrays.SectionArray.reset_manipulation],
        then rebuilds internal models while preserving observer bindings.

        Examples:
            >>> balance_engine.support_manipulation({1: {"z": 5.0}})
            >>> balance_engine.reset_manipulation()
        """
        self.section_array.reset_manipulation()

    def rope_manipulation(
        self,
        rope: dict[int, float],
        rope_lineic_mass: float | None = None,
    ) -> None:
        """Override insulator length and mass for specified supports with rope values.

        Delegates to
        [`SectionArray.rope_manipulation`][mechaphlowers.entities.arrays.SectionArray.rope_manipulation],
        then resets internal models while preserving observer bindings.

        Args:
            rope: Dictionary mapping support index (0-based) to rope length (meters).
            rope_lineic_mass: Linear mass of the rope in kg/m. Defaults to
                ``options.data.rope_lineic_mass_default`` (``0.01`` kg/m).

        Raises:
            ValueError: If a support index is out of range.

        Examples:
            >>> balance_engine.rope_manipulation({1: 4.5, 2: 3.0})
            >>> balance_engine.solve_adjustment()
            >>> balance_engine.solve_change_state(new_temperature=15.0)
        """
        self.section_array.rope_manipulation(rope, rope_lineic_mass)

    def reset_rope_manipulation(self) -> None:
        """Remove the rope overlay and restore original insulator values.

        Delegates to
        [`SectionArray.reset_rope_manipulation`][mechaphlowers.entities.arrays.SectionArray.reset_rope_manipulation],
        then resets internal models while preserving observer bindings.

        Examples:
            >>> balance_engine.rope_manipulation({1: 4.5})
            >>> balance_engine.reset_rope_manipulation()
        """
        self.section_array.reset_rope_manipulation()

    def add_virtual_support(
        self, virtual_support: dict[int, dict[str, float]]
    ) -> None:
        """Insert virtual supports into the section array overlay.

        Delegates to
        [`SectionArray.add_virtual_support`][mechaphlowers.entities.arrays.SectionArray.add_virtual_support],
        then fully rebuilds all internal models while preserving observer bindings.

        Args:
            virtual_support: See
                [`SectionArray.add_virtual_support`][mechaphlowers.entities.arrays.SectionArray.add_virtual_support].

        Examples:
            >>> balance_engine.add_virtual_support(
            ...     {
            ...         1: {
            ...             "x": 200.0,
            ...             "y": 0.0,
            ...             "z": 55.0,
            ...             "insulator_length": 3.0,
            ...             "insulator_mass": 500.0,
            ...         }
            ...     }
            ... )
        """
        self.section_array.add_virtual_support(virtual_support)

    def reset_virtual_support(self) -> None:
        """Remove all virtual supports from the section array overlay.

        Delegates to
        [`SectionArray.reset_virtual_support`][mechaphlowers.entities.arrays.SectionArray.reset_virtual_support],
        then fully rebuilds all internal models while preserving observer bindings.

        Examples:
            >>> balance_engine.add_virtual_support(
            ...     {
            ...         1: {
            ...             "x": 200.0,
            ...             "y": 0.0,
            ...             "z": 55.0,
            ...             "insulator_length": 3.0,
            ...             "insulator_mass": 500.0,
            ...         }
            ...     }
            ... )
            >>> balance_engine.reset_virtual_support()
        """
        self.section_array.reset_virtual_support()

    def _virtual_support_hanging_points_vector_and_mask(
        self,
        n_spans: int | None = None,
        apply_virtual_offset: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a span-aligned hanging-point vector and impacted-span mask.

        Args:
            n_spans: Target vector size. If ``None``, uses the current
                span-model size.
            apply_virtual_offset: If ``True``, map each original span index to
                its effective index after virtual-support insertion
                (``span_idx + offset``). If ``False``, use original span
                indices from the overlay.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - A zero vector of size ``n_spans`` where virtual-support
                  hanging points are placed at their corresponding span index.
                - A boolean mask selecting spans impacted by virtual supports.
        """
        if n_spans is None:
            n_spans = int(self.span_model.span_length.shape[0])
        hanging_points = np.zeros(n_spans, dtype=np.float64)
        impacted_spans = np.zeros(n_spans, dtype=bool)

        overlay = self.section_array._virtual_support_overlay
        if not overlay:
            return hanging_points, impacted_spans

        for offset, span_idx in enumerate(sorted(overlay.keys())):
            target_idx = span_idx + offset if apply_virtual_offset else span_idx
            if target_idx >= n_spans:
                # Fallback for any temporary mismatch between overlay and
                # current span model shape.
                target_idx = span_idx
            if 0 <= target_idx < n_spans:
                hanging_points[target_idx] = float(
                    overlay[span_idx][
                        "hanging_cable_point_from_left_support"
                    ]
                )
                impacted_spans[target_idx] = True

        return hanging_points, impacted_spans

    # def _expand_l_ref_with_virtual_support(
    #     self,
    #     base_l_ref: np.ndarray,
    #     partial_l_ref: np.ndarray,
    #     impacted_spans: np.ndarray,
    # ) -> np.ndarray:
    #     """Expand base L_ref by splitting impacted spans into two semi-spans."""
    #     expanded_l_ref: list[float] = []
    #     for span_idx, l_ref in enumerate(base_l_ref):
    #         if impacted_spans[span_idx]:
    #             left_l_ref = float(partial_l_ref[span_idx])
    #             right_l_ref = float(l_ref - left_l_ref)
    #             expanded_l_ref.extend([left_l_ref, right_l_ref])
    #         else:
    #             expanded_l_ref.append(float(l_ref))
    #     return np.array(expanded_l_ref, dtype=np.float64)

    def shift_span_length(self) -> np.ndarray:
        """Transform shifting distance which is support based into L_ref shift which is span based.

        Returns:
            np.ndarray: Shifted span length, in meters.
        """
        return (
            self._shifting_distance_support[:-1]
            - self._shifting_distance_support[1:]
        )

    def shift_shorten_cable(self) -> None:
        """Shift the cable length according to the shifting and shortening distances."""
        shifted_length = (
            -self.shift_span_length() - self._shortening_distance_span
        )
        self.L_ref = self.initial_L_ref + shifted_length
        self.balance_model.L_ref = self.L_ref

    @check_time
    def solve_adjustment(self) -> None:
        """Solve the chain positions in the adjustment case, updating L_ref in the balance model.
        In this case, there is no weather, no loads, and temperature is the sagging temperature.

        After running this method, many attributes are updated.
        Most interesting ones are `L_ref`, `sagging_parameter` in Span, and `dxdydz` in Nodes.

        raises:
            SolverError: If the solver fails to converge.
        """
        logger.debug("Starting adjustment.")

        # Check if any manipulation overlay is active.
        has_overlays = (
            self.section_array._manipulation_flags.get("support")
            or self.section_array._manipulation_flags.get("rope")
            or self.section_array._manipulation_flags.get("virtual_support")
        )
        # Only virtual_support changes array dimensions and requires rebuild.
        needs_rebuild = bool(
            self.section_array._manipulation_flags.get("virtual_support")
        )

        if has_overlays:
            # Phase 1: deactivate all overlays so L_ref is computed on
            # clean geometry.
            self.section_array.deactivate_manipulation()

            if needs_rebuild:
                saved_observers = self._observers.copy()
                self.reset(full=True)
                self._observers = saved_observers

        self.balance_model.adjustment = True
        try:
            self.solver_adjustment.solve(self.balance_model)
        except SolverError as e:
            logger.error(
                "Error during solve_adjustment, resetting balance engine."
            )
            e.origin = "solve_adjustment"
            raise e

        self.initial_L_ref = self.L_ref = self.balance_model.update_L_ref()            

        logger.debug(f"Output : L_ref = {str(self.L_ref)}")

        if has_overlays:
            # Phase 2: re-enable all overlays and rebuild with manipulated
            # geometry.
            if self.section_array._manipulation_flags.get("virtual_support"):
                self.L_ref, impacted_span_mask = self.get_split_L_ref()
                # resize every span-based array in balance_model to match the new shape after virtual support insertion
                
            
            self.section_array.activate_manipulation()

            if needs_rebuild:
                saved_observers = self._observers.copy()
                self.reset(full=True)
                self._observers = saved_observers


            self.balance_model.L_ref = self.initial_L_ref
            self.notify()

        # Resize shifting/shortening arrays (may have changed due to virtual supports)
        _zeros = np.zeros_like(
            self.section_array.data.conductor_attachment_altitude.to_numpy()
        )
        self._shifting_distance_support = _zeros.copy()
        self._shortening_distance_span = np.zeros(len(_zeros) - 1)

    def get_split_L_ref(self):
        impacted_span_mask = np.zeros_like(self.initial_L_ref, dtype=bool)

        (
            hanging_points,
            impacted_span_mask,
        ) = self._virtual_support_hanging_points_vector_and_mask(
            n_spans=self.__len__(),
            apply_virtual_offset=False,
        )
        
        new_L_ref = arr.incr(self.initial_L_ref.copy())

        new_L_ref_0 = self.span_model.compute_partial_L(x=hanging_points)[
            impacted_span_mask
        ]
        new_L_ref_1 = new_L_ref[impacted_span_mask] - new_L_ref_0

        # replace new L_ref 0 values in the right positions in initial_L_ref
        new_L_ref[impacted_span_mask] = new_L_ref_1

        # insert new L_ref 1 values in the right positions in initial_L_ref, shifted by one position to the right
        
        new_L_ref = np.insert(new_L_ref, impacted_span_mask, new_L_ref_0)
        
        return new_L_ref, impacted_span_mask


    @check_time
    def solve_change_state(
        self,
        wind_pressure: np.ndarray | float | None = None,
        ice_thickness: np.ndarray | float | None = None,
        new_temperature: np.ndarray | float | None = None,
        wind_sense: Literal["clockwise", "anticlockwise"] = "anticlockwise",
    ) -> None:
        """Solve the chain positions, for a case of change of state.
        Updates weather conditions and/or sagging temperature if provided.
        Takes into account loads if any.

        Args:
            wind_pressure (np.ndarray | float | None): Wind pressure in Pa. Default to None
            ice_thickness (np.ndarray | float | None): Ice thickness in m. Default to None
            new_temperature (np.ndarray | float | None): New temperature in °C. Default to None
            wind_sense (Literal["clockwise", "anticlockwise"]): Direction of the wind: if "clockwise": towards user (right), if "anticlockwise": away from user (left). Default to "anticlockwise".

        After running this method, many attributes are updated.
        Most interesting ones are `L_ref`, `sagging_parameter` in Span, and `dxdydz` in Nodes.

        raises:
            SolverError: If the solver fails to converge.
            TypeError: If input parameters have incorrect type.
            ValueError: If input parameters have incorrect shape.
        """
        logger.debug("Starting change state.")
        logger.debug(
            f"Parameters received: \nwind_pressure {str(wind_pressure)}\nice_thickness {str(ice_thickness)}\nnew_temperature {str(new_temperature)}\nwind_sense {str(wind_sense)}"
        )

        if wind_sense not in ["clockwise", "anticlockwise"]:
            raise ValueError(
                f"wind_sense should be 'clockwise' or 'anticlockwise', received {wind_sense}"
            )

        # check if adjustment has been done before
        try:
            _ = self.initial_L_ref
            logger.debug(
                f"Adjustment has been done before, initial_L_ref before shifting: {str(self.initial_L_ref)}"
            )
        except AttributeError:
            logger.warning(self._warning_no_L_ref)
            warnings.warn(self._warning_no_L_ref, BalanceEngineWarning)
            self.solve_adjustment()

        # Use current span_model (potentially rebuilt by solve_adjustment)
        span_shape = (
            self.span_model.sagging_parameter.shape
        )  # span_model holds n-sized array (same shape as span_length)

        def validate_input(input_value, name: str):
            if input_value is None:
                input_value = np.full(span_shape, self.default_value[name])
            elif isinstance(input_value, (int, float)):
                input_value = np.full(span_shape, input_value)
            elif isinstance(input_value, np.ndarray):
                if input_value.shape != span_shape:
                    raise ValueError(
                        f"{name} has incorrect shape: {span_shape} is expected, received {input_value.shape}"
                    )
            else:
                raise TypeError(f"{name} has incorrect type")

            return input_value

        # Set model attributes after potential solve_adjustment (which may
        # rebuild models via reset(full=True)).
        validated_wind = validate_input(wind_pressure, "wind_pressure")
        if wind_sense == "clockwise":
            validated_wind = -validated_wind

        self.balance_model.cable_loads.wind_pressure = validated_wind

        # TODO: convert ice thickness from cm to m? Right now, user has to input in m
        self.balance_model.cable_loads.ice_thickness = validate_input(
            ice_thickness, "ice_thickness"
        )

        new_t = validate_input(new_temperature, "new_temperature")
        self.balance_model.sagging_temperature = arr.decr(new_t)
        self.deformation_model.current_temperature = new_t

        self.balance_model.adjustment = False

        logger.debug(f"Shifting distance: {str(self.shift_support)}")
        logger.debug(f"shortening distance: {str(self.shortening_span)}")
        logger.debug("Taking into account cable shifting and shortening.")
        self.shift_shorten_cable()
        logger.debug(f"L_ref after shifting: {str(self.L_ref)}")

        self.span_model.load_coefficient = (
            self.balance_model.cable_loads.load_coefficient
        )

        try:
            self.solver_change_state.solve(self.balance_model)
        except SolverError as e:
            logger.error(
                "Error during solve_change_state, you should reset the balance engine."
            )
            e.origin = "solve_change_state"
            raise e

        logger.debug(
            f"Output : get_displacement \n{str(self.get_displacement())}"
        )
        self.balance_model.update_nodes_span_model()

    def get_data_spans(self) -> dict[str, list]:
        """Fetch data from BalanceEngine about spans.

        This data is stored as a dictionary containing lists.

        Returns:
            dict: dictionnary contains following fields:
                <ul>
                    <li>span_length</li>
                    <li>elevation</li>
                    <li>parameter</li>
                    <li>tension_sup</li>
                    <li>tension_inf</li>
                    <li>slope_left</li>
                    <li>slope_right</li>
                    <li>L0</li>
                    <li>horizontal_distance</li>
                    <li>arc_length</li>
                    <li>T_h</li>
                </ul>
        """
        T_sup, T_inf = self.span_model.tensions_sup_inf()
        force_output_unit = options.output_units.force
        T_sup_q_array, T_inf_q_array = (
            QuantityArray(T_sup, 'N', force_output_unit),
            QuantityArray(T_inf, 'N', force_output_unit),
        )
        T_h_q_array = QuantityArray(
            self.span_model.T_h(), 'N', force_output_unit
        )
        span_slope_left = QuantityArray(
            self.span_model.slope(side="left"), 'rad', 'deg'
        )
        span_slope_right = QuantityArray(
            self.span_model.slope(side="right"), 'rad', 'deg'
        )

        result_dict = {
            "span_length": arr.decr(
                self.section_array.data["span_length"].to_numpy()
            ).tolist(),
            "elevation": arr.decr(
                self.section_array.data["elevation_difference"].to_numpy()
            ).tolist(),
            "parameter": arr.decr(self.parameter).tolist(),
            "slope_left": arr.decr(span_slope_left.value()).tolist(),
            "slope_right": arr.decr(span_slope_right.value()).tolist(),
            "tension_sup": arr.decr(T_sup_q_array.value()).tolist(),
            "tension_inf": arr.decr(T_inf_q_array.value()).tolist(),
            "L0": self.L_ref.tolist(),
            "horizontal_distance": self.balance_model.a.tolist(),
            "arc_length": arr.decr(self.span_model.compute_L()).tolist(),
            "T_h": arr.decr(T_h_q_array.value()).tolist(),
            "sag": arr.decr(self.span_model.sag()).tolist(),
            "sag_s2": arr.decr(self.span_model.sag_s2()).tolist(),
        }
        return result_dict

    @property
    def support_number(self) -> int:
        return self.section_array.data.span_length.shape[0]

    def __len__(self) -> int:
        """Return the number of supports in the balance engine."""
        return self.support_number

    def __str__(self) -> str:
        dxdydz = self.balance_model.chain_displacement().T
        return_string = (
            f"number of supports: {self.support_number}\n"
            f"parameter: {self.span_model.sagging_parameter}\n"
            f"wind: {self.balance_model.cable_loads.wind_pressure}\n"
            f"ice: {self.balance_model.cable_loads.ice_thickness}\n"
            f"temperature: {self.balance_model.sagging_temperature}\n"
            f"dx: {dxdydz[0]}\n"
            f"dy: {dxdydz[1]}\n"
            f"dz: {dxdydz[2]}\n"
        )
        if hasattr(self, "L_ref"):
            return_string += f"L_ref: {self.L_ref}\n"
        return return_string

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}\n{self.__str__()}"

    @property
    def parameter(self) -> np.ndarray:
        return self.span_model.sagging_parameter
