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
from mechaphlowers.core.models.balance.span_loads import SpanLoads
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

    * `self.balance_model.nodes.dxdydz` and `self.span_model.parameter` for solve_change_state().

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
        self._adjustment_blocked: bool = False

        self.reset(full=True)

    def reset(self, full: bool = False) -> None:
        """Reset the balance engine to initial state.

        This method re-initializes the span model, cable loads, span loads, deformation model, balance model, and solvers.
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
            self.span_loads = SpanLoads(
                arr.decr(zeros_vector),
                arr.decr(zeros_vector),
                self.section_array.data.span_length.to_numpy(),
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
                self.span_loads,
            )
        else:
            self.balance_model.reset(
                cable_array=self.cable_array,
                span_model=self.span_model,
                deformation_model=self.deformation_model,
                cable_loads=self.cable_loads,
                span_loads=self.span_loads,
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
        """Calls preferred method [`set_loads`](mechaphlowers.core.models.balance.engine.BalanceEngine.set_loads).

        Kept for compatibility.

        Expected length for load_position_distance and load_mass is the number of pylons.
        Last array elements should be nan or zero.

        Raises:
            ValueError: if at least one load_position_distance is not in [0, span_length]
                or if the arguments don't have the right lengths.
        """
        warnings.warn(
            "add_loads is deprecated, use set_loads instead."
            "Caution: expected argument length for set_loads is the number of spans.",
            category=DeprecationWarning,
        )
        load_position_distance = np.array(load_position_distance)
        load_mass = np.array(load_mass)
        self.set_loads(arr.decr(load_position_distance), arr.decr(load_mass))

    def set_loads(
        self,
        load_position_distance: np.ndarray | list,
        load_mass: np.ndarray | list,
    ) -> None:
        """Adds loads to BalanceEngine.

        Input for position is a distance, and will be converted into ratio.

        Expected input are arrays of size matching the number of spans. Each value refers to a span.

        If either load_position_distance[i] or load_mass[i] is 0 or nan, it means there is no load at span i.

        Args:
            load_position_distance (np.ndarray | list): Position of the loads, in meters
            load_mass (np.ndarray | list): Mass of the loads

        Raises:
            ValueError: if at least one load_position_distance is not in [0, span_length]
                or if the arguments don't have the right lengths.

        Examples:
            >>> load_position_distance = np.array([150, 200, 0])  # 3 spans
            >>> load_mass = np.array([500, 70, 0])
            >>> engine.set_loads(load_position_distance, load_mass)
            >>> plot_engine.reset()  # optional: only needed if cached plots must be discarded
        """
        span_length = self.section_array.data.span_length.to_numpy()

        self.span_loads.set_loads(
            load_position_distance, load_mass, span_length
        )

        self.reset(full=False)

        debug_loads = (
            "Loads have been added. PlotEngine will be notified automatically "
            "via the observer pattern; no manual reset is required."
        )
        logger.debug(debug_loads)

    @check_time
    def solve_adjustment(self) -> None:
        """Solve the chain positions in the adjustment case, updating L_ref in the balance model.
        In this case, there is no weather, no loads, and temperature is the sagging temperature.

        After running this method, many attributes are updated.
        Most interesting ones are `L_ref`, `parameter` in Span, and `dxdydz` in Nodes.

        Raises:
            SolverError: If the solver fails to converge.
            RuntimeError: If adjustment is blocked (engine built from manipulations).
        """
        if self._adjustment_blocked:
            raise RuntimeError(
                "solve_adjustment is blocked on this engine. "
                "L_ref was injected externally from a clean adjustment."
            )
        logger.debug("Starting adjustment.")

        self.balance_model.adjustment = True
        # reset parameter to sagging parameter
        sagging_parameter = (
            self.section_array.data.sagging_parameter.to_numpy()
        )
        self.span_model.set_parameter(sagging_parameter)
        # reset displacements to zero (hypothesis of adjustment)
        self.balance_model.nodes.dxdydz[:] = 0

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

    @check_time
    def solve_change_state(
        self,
        wind_pressure: np.ndarray | float | None = None,
        ice_thickness: np.ndarray | float | None = None,
        new_temperature: np.ndarray | float | None = None,
        wind_direction: Literal[
            "clockwise", "anticlockwise"
        ] = "anticlockwise",
    ) -> None:
        """Solve the chain positions, for a case of change of state.
        Updates weather conditions and/or sagging temperature if provided.
        Takes into account loads if any.

        Args:
            wind_pressure (np.ndarray | float | None): Wind pressure in Pa. Default to None
            ice_thickness (np.ndarray | float | None): Ice thickness in m. Default to None
            new_temperature (np.ndarray | float | None): New temperature in °C. Default to None
            wind_direction (Literal["clockwise", "anticlockwise"]): Direction of the wind: if "clockwise": towards user (right), if "anticlockwise": away from user (left). Default to "anticlockwise".

        After running this method, many attributes are updated.
        Most interesting ones are `L_ref`, `parameter` in Span, and `dxdydz` in Nodes.

        Raises:
            SolverError: If the solver fails to converge.
            TypeError: If input parameters have incorrect type.
            ValueError: If input parameters have incorrect shape.
        """
        logger.debug("Starting change state.")
        logger.debug(
            f"Parameters received: \nwind_pressure {str(wind_pressure)}\nice_thickness {str(ice_thickness)}\nnew_temperature {str(new_temperature)}\nwind_direction {str(wind_direction)}"
        )

        if wind_direction not in ["clockwise", "anticlockwise"]:
            raise ValueError(
                f"wind_direction should be 'clockwise' or 'anticlockwise', received {wind_direction}"
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
            self.span_model.parameter.shape
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
        if wind_direction == "clockwise":
            validated_wind = -validated_wind

        self.balance_model.cable_loads.wind_pressure = validated_wind

        # Ice thickness input in meters
        self.balance_model.cable_loads.ice_thickness = validate_input(
            ice_thickness, "ice_thickness"
        )

        new_t = validate_input(new_temperature, "new_temperature")
        self.balance_model.current_temperature = arr.decr(new_t)
        self.deformation_model.current_temperature = new_t

        self.balance_model.adjustment = False

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
                    <li>sag</li>
                    <li>sag_s2</li>
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

    def get_ruling_span_length(self) -> float:
        """Compute ruling span length:

        if we considered the whole section as a single span, the length would be ruling_span_length

        Used for tensions computation when unfolding the cable.

        $L_{R} = \\sqrt{\\frac{\\sum(L_n ^ 4 / C_n)}{\\sum{C_n}}}$

        where $L_n$ are the horizontal length of span n, and $C_n$ the chord length of span n

        Returns:
            float: span length of ruling span
        """
        # proto uses section_array.span_length instead of balance_model.a (called a_chain in proto)
        chord = np.sqrt(self.balance_model.a**2 + self.balance_model.b**2)
        return np.sqrt(np.sum(self.balance_model.a**4 / chord) / np.sum(chord))

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
            f"parameter: {self.span_model.parameter}\n"
            f"wind: {self.balance_model.cable_loads.wind_pressure}\n"
            f"ice: {self.balance_model.cable_loads.ice_thickness}\n"
            f"temperature: {self.balance_model.current_temperature}\n"
            f"load position (ratio): {self.span_loads.load_position}\n"
            f"load mass: {self.span_loads.load_mass}\n"
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
        return self.span_model.parameter
