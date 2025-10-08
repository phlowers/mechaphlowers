# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from mechaphlowers.core.models.balance.utils_balance import reduce_to_span
from mechaphlowers.core.models.cable.deformation import IDeformation
from mechaphlowers.core.models.cable.span import Span

try:
    from scipy import optimize  # type: ignore
except ImportError:
    import mechaphlowers.core.numeric.numeric as optimize


logger = logging.getLogger(__name__)

PARAMETER_STEP = 1.0


class ModelToSolve(ABC):
    @property
    @abstractmethod
    def initial_value(self):
        pass

    @abstractmethod
    def _delta(self, parameter):
        pass

    @abstractmethod
    def _delta_prime(self, parameter):
        pass


class FindParamModel(ModelToSolve):
    def __init__(
        self,
        span_model: Span,
        deformation_model: IDeformation,
    ):
        self.span_model = span_model
        self.deformation_model = deformation_model

    def set_attributes(self, initial_parameter: np.ndarray, L_ref: np.ndarray):
        self.initial_parameter = initial_parameter
        self.L_ref = L_ref

    def update_models(self, parameter):
        self.span_model.set_parameter(parameter)
        # TODO: create getter for L
        self.deformation_model.cable_length = self.span_model.L
        self.deformation_model.tension_mean = self.span_model.T_mean()

    @property
    def initial_value(self):
        return self.initial_parameter

    def _delta(self, parameter):
        self.update_models(parameter)
        L = self.span_model.L
        eps_mecha = self.deformation_model.epsilon_mecha()
        eps_therm = self.deformation_model.epsilon_therm_0()
        return (L - self.L_ref) / self.L_ref - (eps_mecha + eps_therm)

    def _delta_prime(self, parameter):
        return (
            self._delta(parameter + PARAMETER_STEP) - self._delta(parameter)
        ) / PARAMETER_STEP


class FindParamSolver(ABC):
    def __init__(self, model: ModelToSolve) -> None:
        self.model = model

    @abstractmethod
    def find_parameter(self) -> np.ndarray:
        pass


class FindParamSolverScipy(FindParamSolver):
    def __init__(self, model: ModelToSolve):
        self.model = model

    def find_parameter(self) -> np.ndarray:
        p0 = self.model.initial_value

        solver_result = optimize.newton(
            self.model._delta,
            p0,
            fprime=self.model._delta_prime,
            tol=0.1,
            # tol=1.,
            full_output=True,
        )
        if not solver_result.converged.all():
            raise ValueError("Solver did not converge")
        return solver_result.root


class FindParamSolverForLoop(FindParamSolver):
    def __init__(self, model: ModelToSolve):
        self.model = model

    def find_parameter(self) -> np.ndarray:
        parameter = self.model.initial_value

        n_iter_max = 50
        for i in range(n_iter_max):
            mem = parameter
            delta = self.model._delta(parameter)
            delta_prime = self.model._delta_prime(parameter)
            parameter = parameter - delta / delta_prime

            if (
                np.linalg.norm(reduce_to_span(mem - parameter))
                < 0.1 * parameter.size
            ):
                break
            if i == n_iter_max:
                logger.info("max iter reached")

        return parameter
