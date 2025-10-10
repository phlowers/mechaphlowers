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

logger = logging.getLogger(__name__)


class IModelForSolver(ABC):
    @property
    @abstractmethod
    def state_vector(self) -> np.ndarray:
        pass

    @state_vector.setter
    @abstractmethod
    def state_vector(self, value) -> None:
        pass

    @abstractmethod
    def objective_function(self) -> np.ndarray:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    def dict_to_store(self) -> dict:
        return {}


class IBalanceModel(IModelForSolver, ABC):
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
    def update_L_ref(self):
        pass

    @property
    @abstractmethod
    def adjustment(self):
        pass

