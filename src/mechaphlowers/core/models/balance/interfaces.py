# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from mechaphlowers.core.models.balance.solvers.find_parameter_solver import (
    FindParamSolverForLoop,
    IFindParamSolver,
)
from mechaphlowers.core.models.cable.deformation import IDeformation
from mechaphlowers.core.models.cable.span import ISpan
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.entities.core import VhlStrength

logger = logging.getLogger(__name__)


class IModelForSolver(ABC):
    """Interface for models used by BalanceSolver.
    This interface defines the necessary methods and properties that a model in order to be compatible with BalanceSolver.
    """

    @property
    @abstractmethod
    def state_vector(self) -> np.ndarray:
        """Vector of state variables to be optimized."""
        pass

    @state_vector.setter
    @abstractmethod
    def state_vector(self, value) -> None:
        pass

    @abstractmethod
    def objective_function(self) -> np.ndarray:
        """Function to minimize that depends on state_vector."""
        pass

    @abstractmethod
    def update(self) -> None:
        """Update any variables of the model if necessary. Can stay empty."""
        pass

    def dict_to_store(self) -> dict:
        """Returns a dictionary of values to store after each loop iteration.
        Used only for debugging purposes. Can stay empty.
        """
        return {}


class IBalanceModel(IModelForSolver, ABC):
    """Interface for balance models. Used by BalanceEngine to find insulator chain positions."""

    def __init__(
        self,
        sagging_temperature: np.ndarray,
        parameter: np.ndarray,
        section_array: SectionArray,
        cable_array: CableArray,
        span_model: ISpan,
        deformation_model: IDeformation,
        cable_loads: CableLoads,
        find_param_solver_type: Type[
            IFindParamSolver
        ] = FindParamSolverForLoop,
    ):
        self.adjustment: bool = NotImplemented
        self.sagging_temperature = sagging_temperature
        self.cable_loads = cable_loads

    @abstractmethod
    def update_L_ref(self) -> np.ndarray:
        """Update the reference length L_ref after an adjustment solve."""
        pass

    @property
    @abstractmethod
    def adjustment(self) -> bool:
        """Boolean indicating if the model is in adjustment mode or change of state mode."""
        pass

    @adjustment.setter
    @abstractmethod
    def adjustment(self, value: bool) -> None:
        pass

    @abstractmethod
    def dxdydz(self) -> np.ndarray:
        """Get the displacement vector of the nodes."""
        pass

    @abstractmethod
    def vhl_under_chain(self) -> VhlStrength:
        """Get the VHL efforts under chain: without considering insulator_weight.
        Format: [[V0, H0, L0], [V1, H1, L1], ...]
        Default unit is daN"""
        pass

    @abstractmethod
    def vhl_under_console(self) -> VhlStrength:
        """Get the VHL efforts under console: considering insulator_weight.
        Format: [[V0, H0, L0], [V1, H1, L1], ...]
        Default unit is daN"""
        pass
