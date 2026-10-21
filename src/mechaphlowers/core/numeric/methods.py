# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
from typing import Callable, Protocol

import numpy as np

from mechaphlowers.core.numeric.result import EstimationResult

logger = logging.getLogger(__name__)


class OptimizationMethod(Protocol):
    """Protocol for root-finding / optimization algorithms.

    Implementations solve ``objective(x) = 0`` within given bounds.
    """

    def solve(
        self,
        objective: Callable[[float], float],
        bounds: tuple[float, float],
    ) -> EstimationResult: ...


class BisectionMethod:
    """Simple bisection root-finding.

    Requires that ``objective`` changes sign over ``bounds``.

    Args:
        tol: Absolute tolerance on the root. Defaults to 1e-3.
        maxiter: Maximum number of iterations. Defaults to 50.
    """

    FUNCTION_TOL = 1e-6

    def __init__(self, tol: float = 1e-3, maxiter: int = 50) -> None:
        self.tol = tol
        self.maxiter = maxiter

    def solve(
        self,
        objective: Callable[[float], float],
        bounds: tuple[float, float],
    ) -> EstimationResult:
        a, b = bounds
        fa = objective(a)
        fb = objective(b)
        iterations = 2

        if np.isnan(fa) or np.isnan(fb):
            raise ValueError("objective returned NaN at bounds")
        if fa == 0:
            return EstimationResult(
                value=a, residual=0.0, iterations=iterations, converged=True
            )
        if fb == 0:
            return EstimationResult(
                value=b, residual=0.0, iterations=iterations, converged=True
            )

        if np.sign(fa) == np.sign(fb):
            logger.warning(
                "Bisection: objective does not change sign over bounds "
                f"[{a}, {b}] (f(a)={fa:.3e}, f(b)={fb:.3e}). "
                "Returning best bound."
            )
            best = a if abs(fa) < abs(fb) else b
            residual = min(abs(fa), abs(fb))
            return EstimationResult(
                value=best,
                residual=residual,
                iterations=iterations,
                converged=residual <= self.FUNCTION_TOL,
            )

        for _ in range(self.maxiter):
            mid = (a + b) / 2.0
            fmid = objective(mid)
            iterations += 1

            if abs(fmid) <= self.FUNCTION_TOL or (b - a) / 2.0 <= self.tol:
                return EstimationResult(
                    value=mid,
                    residual=abs(fmid),
                    iterations=iterations,
                    converged=True,
                )

            if np.sign(fa) != np.sign(fmid):
                b = mid
            else:
                a = mid

        mid = (a + b) / 2.0
        return EstimationResult(
            value=mid,
            residual=abs(objective(mid)),
            iterations=iterations + 1,
            converged=False,
        )


class BrentMethod:
    """Brent's method for root-finding (``brentq``).

    Requires that ``objective`` changes sign over ``bounds``.
    Uses ``scipy.optimize.brentq`` when scipy is installed, otherwise the
    pure-Python port in ``mechaphlowers.numeric.scipy``. Falls back to
    bisection if the interval does not bracket a root.

    Args:
        tol: Absolute tolerance on the root. Defaults to 1e-3.
        maxiter: Maximum number of iterations. Defaults to 50.
    """

    def __init__(self, tol: float = 1e-3, maxiter: int = 50) -> None:
        self.tol = tol
        self.maxiter = maxiter

    def solve(
        self,
        objective: Callable[[float], float],
        bounds: tuple[float, float],
    ) -> EstimationResult:
        try:
            from scipy.optimize import brentq  # type: ignore
        except ImportError:
            from mechaphlowers.numeric.scipy import brentq  # type: ignore

        a, b = bounds

        try:
            root, result_info = brentq(
                objective,
                a,
                b,
                xtol=self.tol,
                maxiter=self.maxiter,
                full_output=True,
                disp=False,
            )
            return EstimationResult(
                value=root,
                residual=abs(objective(root)),
                iterations=result_info.iterations,
                converged=result_info.converged,
            )
        except ValueError as e:
            logger.warning(f"Brent's method failed: {e}. Trying bisection.")
            fallback = BisectionMethod(tol=self.tol, maxiter=self.maxiter)
            return fallback.solve(objective, bounds)
