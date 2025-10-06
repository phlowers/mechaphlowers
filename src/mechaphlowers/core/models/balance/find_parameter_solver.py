# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations

import logging
from typing import Type

import numpy as np
import pandas as pd

from mechaphlowers.core.models.balance.utils_balance import reduce_to_span
from mechaphlowers.core.models.cable.deformation import IDeformation
from mechaphlowers.core.models.cable.span import Span

try:
    from scipy import optimize  # type: ignore
except ImportError:
    import mechaphlowers.core.numeric.numeric as optimize


logger = logging.getLogger(__name__)

PARAMETER_STEP = 1.

class FindParamModel:
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
        self.span_model.sagging_parameter = parameter
        self.deformation_model.cable_length = self.span_model.L()
        self.deformation_model.tension_mean = self.span_model.T_mean()

    @property
    def initial_value(self):
        return self.initial_parameter
            
    def _delta(self, parameter):
        self.update_models(parameter)
        L = self.span_model.L()
        eps_mecha = self.deformation_model.epsilon_mecha()
        eps_therm = self.deformation_model.epsilon_therm_0()
        return (L - self.L_ref) / self.L_ref - (eps_mecha + eps_therm)
    
    def _delta_prime(self, parameter):
        return (self._delta(parameter + PARAMETER_STEP) - self._delta(parameter)) / PARAMETER_STEP


class FindParamSolverScipy:
    def __init__(self, model: FindParamModel):
          self.model = model


    def find_parameter(
        self,
        solver: str = "newton",
    ) -> np.ndarray:
        solver_dict = {"newton": optimize.newton}
        try:
            solver_method = solver_dict[solver]
        except KeyError:
            raise ValueError(f"Incorrect solver name: {solver}")

        p0 = self.model.initial_value

        solver_result = solver_method(
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


class FindParamSolverForLoop:
    def __init__(self, model: FindParamModel):
          self.model = model

    def find_parameter(
        self,
    ) -> np.ndarray:
        parameter = self.model.initial_parameter

        n_iter_max = 50
        for i in range(n_iter_max):
            mem = parameter.copy()
            delta = self.model._delta(parameter)
            delta_prime = self.model._delta_prime(parameter)
            parameter = parameter - delta / delta_prime

            if np.linalg.norm(reduce_to_span(mem - parameter)) < 0.1 * parameter.size:
                break
            if i == n_iter_max:
                logger.info("max iter reached")

        return parameter
