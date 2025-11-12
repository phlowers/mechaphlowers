# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from mechaphlowers.core.models.cable.deformation import IDeformation
from mechaphlowers.core.models.cable.span import ISpan
from mechaphlowers.utils import arr

try:
    from scipy import optimize  # type: ignore
except ImportError:
    import mechaphlowers.numeric.scipy as optimize


logger = logging.getLogger(__name__)


class IModelToSolve(ABC):
    """Interface for models to solve for a parameter using IFindParamSolver."""

    @property
    @abstractmethod
    def initial_value(self):
        """First value to start the solver from."""
        pass

    @abstractmethod
    def _delta(self, parameter: np.ndarray) -> np.ndarray:
        """Function to find the root of.
        The solver will solve $_delta(parameter) = 0$"""
        pass

    @abstractmethod
    def _delta_prime(self, parameter: np.ndarray) -> np.ndarray:
        """Derivative of the function to find the root of."""
        pass


class FindParamModel(IModelToSolve):
    # TODO: write docstring (use SagTensionSolver for inspiration)
    def __init__(
        self,
        span_model: ISpan,
        deformation_model: IDeformation,
        param_step=1.0,
    ):
        self.span_model = span_model
        self.deformation_model = deformation_model
        self.param_step = param_step

    def set_attributes(
        self, initial_parameter: np.ndarray, L_ref: np.ndarray
    ) -> None:
        self.initial_parameter = initial_parameter
        self.L_ref = L_ref

    def update_models(self, parameter: np.ndarray) -> None:
        """Update span_model and deformation_model with a new value of parameter.
        This causes cable_length and tension_mean of deformation_model to be updated.

        Args:
            parameter (np.ndarray): new value of the parameter: will change during each iteration.
        """
        self.span_model.set_parameter(parameter)
        self.deformation_model.cable_length = self.span_model.L
        self.deformation_model.tension_mean = self.span_model.T_mean()

    @property
    def initial_value(self) -> np.ndarray:
        if not hasattr(self, "initial_parameter"):
            raise AttributeError(
                "initial_parameter is not set. Please call set_attributes() before accessing initial_value."
            )
        return self.initial_parameter

    def _delta(self, parameter: np.ndarray) -> np.ndarray:
        """Equation to solve:
        $\\frac{L(p) - L_0}{L_0} - (\\varepsilon_{mecha}(p) + \\varepsilon_{therm}(p)) = 0$

        Args:
            parameter (np.ndarray): parameter

        Returns:
            np.ndarray: value of the function to find the root of
        """
        self.update_models(parameter)
        L = self.span_model.L
        eps_mecha = self.deformation_model.epsilon_mecha()
        eps_therm = self.deformation_model.epsilon_therm_0()
        return (L - self.L_ref) / self.L_ref - (eps_mecha + eps_therm)

    def _delta_prime(self, parameter) -> np.ndarray:
        return (
            self._delta(parameter + self.param_step) - self._delta(parameter)
        ) / self.param_step


class IFindParamSolver(ABC):
    """Interface for solvers that find a parameter for a model implementing IModelToSolve."""

    def __init__(
        self, model: IModelToSolve, stop_condition=0.1, max_iter=50
    ) -> None:
        self.model = model
        self.stop_condition = stop_condition
        self.max_iter = max_iter

    @abstractmethod
    def find_parameter(self) -> np.ndarray:
        pass


class FindParamSolverScipy(IFindParamSolver):
    """Implementation of IFindParamSolver using scipy.optimize.newton"""

    def find_parameter(self) -> np.ndarray:
        p0 = self.model.initial_value

        solver_result = optimize.newton(
            self.model._delta,
            p0,
            fprime=self.model._delta_prime,
            tol=self.stop_condition,
            full_output=True,
        )
        if not np.all(solver_result.converged):
            raise ValueError("Solver did not converge")
        return solver_result.root


class FindParamSolverForLoop(IFindParamSolver):
    """Implementation of IFindParamSolver using the Newton-Raphson method with a python for loop."""

    def find_parameter(self) -> np.ndarray:
        parameter = self.model.initial_value

        for i in range(self.max_iter):
            mem = parameter
            delta = self.model._delta(parameter)
            delta_prime = self.model._delta_prime(parameter)
            parameter = parameter - delta / delta_prime

            if (
                np.linalg.norm(arr.decr(mem - parameter))
                < self.stop_condition * parameter.size
            ):
                break
            if i == self.max_iter - 1:
                logger.error(
                    f"Maximum number of iterations reached in {str(__name__)}"
                )
                raise ValueError("Find parameter solver did not converge")

        return parameter
