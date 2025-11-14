# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations

import logging

import numpy as np

from mechaphlowers.core.models.balance.interfaces import IModelForSolver

logger = logging.getLogger(__name__)


class BalanceSolver:
    """Solver for balance models using a Newton-Raphson method.
    Takes a model implementing IModelForSolver as input, and solves it using.

    The main difference with a classic Newton-Raphson method is
    that the correction of the state vector is using a custom formula,
    using a relaxation value that decreases with the number of iterations.

    The jacobian matrix is computed using finite differences.

        >>> balance_model = ...  # some model implementing IModelForSolver
        >>> solver = BalanceSolver()
        >>> solver.solve(balance_model)
        >>> balance_model.state_vector  # updated state vector after solving
        np.array([...])
    """

    def __init__(
        self,
        perturb=0.0001,
        stop_condition=1e-2,
        relax_ratio=0.8,
        relax_power=3,
        max_iter=100,
    ) -> None:
        self.perturb = perturb
        self.stop_condition = stop_condition
        self.relax_ratio = relax_ratio
        self.relax_power = relax_power
        self.max_iter = max_iter

    def solve(
        self,
        model: IModelForSolver,
    ) -> None:
        # initialisation
        model.update()
        objective_vector = model.objective_function()

        # starting optimisation loop
        for counter in range(1, self.max_iter):
            # compute jacobian
            jacobian = self.jacobian(objective_vector, model, self.perturb)

            # memorize for norm
            mem = np.linalg.norm(objective_vector)

            # correction calculus
            correction = np.linalg.solve(jacobian.T, objective_vector)

            model.state_vector = model.state_vector - correction * (
                1 - self.relax_ratio ** (counter**self.relax_power)
            )

            model.update()

            # compute value to minimize
            objective_vector = model.objective_function()
            norm_d_param = np.abs(
                np.linalg.norm(objective_vector) ** 2 - mem**2
            )

            # store values for debug
            dict_to_store = {
                "num_loop": counter,
                "objective": objective_vector,
                "state_vector": model.state_vector,
            }
            dict_to_store.update(model.dict_to_store())

            # check value to minimze to break the loop
            if norm_d_param < self.stop_condition:
                break
            if counter == self.max_iter - 1:
                logger.info("max iteration reached")
                logger.info(f"{norm_d_param=}")
                raise ValueError("Balance solver did not converge")

    def jacobian(
        self,
        objective_vector: np.ndarray,
        model: IModelForSolver,
        perturb: float = 1e-4,
    ) -> np.ndarray:
        vector_perturb = np.zeros_like(objective_vector)
        df_list = []

        for i in range(len(vector_perturb)):
            vector_perturb[i] += perturb

            f_perturb = self._delta_d(model, vector_perturb)
            df_dperturb = (f_perturb - objective_vector) / perturb
            df_list.append(df_dperturb)

            vector_perturb[i] -= perturb

        jacobian = np.array(df_list)
        return jacobian

    def _delta_d(
        self, model: IModelForSolver, vector_perturb: np.ndarray
    ) -> np.ndarray:
        model.state_vector += vector_perturb
        model.update()
        perturbed_force_vector = model.objective_function()
        model.state_vector -= vector_perturb
        return perturbed_force_vector
