# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations

import logging
from typing import Callable, Type

import numpy as np

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
from mechaphlowers.utils import arr, check_time

logger = logging.getLogger(__name__)


class DisplacementResult:
    def __init__(
        self,
        dxdydz: np.ndarray,
    ):
        self.dxdydz = dxdydz


class BalanceEngine:
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

        zeros_vector = np.zeros_like(
            section_array.data.conductor_attachment_altitude.to_numpy()
        )

        sagging_temperature = arr.decr(
            (section_array.data.sagging_temperature.to_numpy())
        )
        parameter = arr.decr(section_array.data.sagging_parameter.to_numpy())
        self.span_model = span_model_builder(
            section_array, cable_array, span_model_type
        )
        self.cable_loads = CableLoads(
            np.float64(cable_array.data.diameter.iloc[0]),
            np.float64(cable_array.data.linear_weight.iloc[0]),
            zeros_vector,
            zeros_vector,
        )
        self.deformation_model = deformation_model_builder(
            cable_array,
            self.span_model,
            sagging_temperature,
            deformation_model_type,
        )

        self.balance_model = balance_model_type(
            sagging_temperature,
            parameter,
            section_array,
            cable_array,
            self.span_model,
            self.deformation_model,
            self.cable_loads,
        )
        self.solver_change_state = BalanceSolver(
            **options.solver.balance_solver_change_state_params
        )
        self.solver_adjustment = BalanceSolver(
            **options.solver.balance_solver_adjustment_params
        )
        self.L_ref: np.ndarray

        self.get_displacement: Callable = self.balance_model.dxdydz
        logger.debug("Balance engine initialized.")

    @check_time
    def solve_adjustment(self) -> None:
        """Solve the chain positions in the adjustment case, updating L_ref in the balance model.
        In this case, there is no weather, no loads, and temperature is the sagging temperature.

        After running this method, many attributes are updated.
        Most interesting ones are `L_ref`, `sagging_parameter` in Span, and `dxdydz` in Nodes.
        """
        logger.debug("Starting adjustment.")

        self.balance_model.adjustment = True
        self.solver_adjustment.solve(self.balance_model)
        self.L_ref = self.balance_model.update_L_ref()

        logger.debug(f"Output : L_ref = {str(self.L_ref)}")

    @check_time
    def solve_change_state(
        self,
        wind_pressure: np.ndarray | float | None = None,
        ice_thickness: np.ndarray | float | None = None,
        new_temperature: np.ndarray | float | None = None,
    ) -> None:
        """Solve the chain positions, for a case of change of state.
        Updates weather conditions and/or sagging temperature if provided.
        Takes into account loads if any.

        Args:
            wind_pressure (np.ndarray | float | None): Wind pressure in Pa. Default to None
            ice_thickness (np.ndarray | float | None): Ice thickness in m. Default to None
            new_temperature (np.ndarray | float | None): New temperature in °C. Default to None

        After running this method, many attributes are updated.
        Most interesting ones are `L_ref`, `sagging_parameter` in Span, and `dxdydz` in Nodes.
        """
        logger.debug("Starting change state.")
        logger.debug(
            f"Parameters received: \nwind_pressure {str(wind_pressure)}\nice_thickness {str(ice_thickness)}\nnew_temperature {str(new_temperature)}"
        )

        span_shape = self.section_array.data.span_length.shape

        def validate_input(input_value, name: str):
            if input_value is not None and not isinstance(
                input_value, (int, float, np.ndarray)
            ):
                raise TypeError(f"{name} has incorrect type")
            if input_value is None:
                input_value = self.default_value[name]
            if isinstance(input_value, (int, float)):
                input_value = np.full(span_shape, input_value)
            if isinstance(input_value, np.ndarray):
                if input_value.shape != span_shape:
                    raise ValueError(
                        f"{name} has incorrect shape: {span_shape} is expected, recieved {input_value.shape}"
                    )

            return input_value

        self.balance_model.cable_loads.wind_pressure = validate_input(
            wind_pressure, "wind_pressure"
        )
        # TODO: convert ice thickness from cm to m? Right now, user has to input in m
        self.balance_model.cable_loads.ice_thickness = validate_input(
            ice_thickness, "ice_thickness"
        )

        new_t = validate_input(new_temperature, "new_temperature")
        self.balance_model.sagging_temperature = arr.decr(new_t)
        self.deformation_model.current_temperature = new_t

        # check if adjustment has been done before
        try:
            _ = self.L_ref
        except AttributeError as e:
            logger.error(
                "L_ref not defined. You must run solve_adjustment() before solve_change_state()."
            )
            raise e

        self.balance_model.adjustment = False
        self.span_model.load_coefficient = (
            self.balance_model.cable_loads.load_coefficient
        )
        self.solver_change_state.solve(self.balance_model)
        logger.debug(
            f"Output : get_displacement \n{str(self.get_displacement())}"
        )

    @property
    def support_number(self) -> int:
        return self.section_array.data.span_length.shape[0]
