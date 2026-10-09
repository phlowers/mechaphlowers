# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
from typing import Callable, Protocol

from mechaphlowers.core.models.estimation.result import EstimationResult

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

        if fa * fb > 0:
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
                converged=residual <= self.tol,
            )

        for _ in range(self.maxiter):
            mid = (a + b) / 2.0
            fmid = objective(mid)
            iterations += 1

            if abs(fmid) <= self.tol or (b - a) / 2.0 <= self.tol:
                return EstimationResult(
                    value=mid,
                    residual=abs(fmid),
                    iterations=iterations,
                    converged=True,
                )

            if fa * fmid < 0:
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
    """Brent's method for root-finding (via scipy.optimize.brentq).

    Requires that ``objective`` changes sign over ``bounds``.
    Falls back to bisection if scipy is not available.

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
            logger.warning(
                "scipy not available, falling back to BisectionMethod."
            )
            fallback = BisectionMethod(tol=self.tol, maxiter=self.maxiter)
            return fallback.solve(objective, bounds)

        a, b = bounds

        try:
            root, result_info = brentq(
                objective,
                a,
                b,
                xtol=self.tol,
                maxiter=self.maxiter,
                full_output=True,
            )
            return EstimationResult(
                value=root,
                residual=abs(
                    result_info.function_calls and abs(objective(root)) or 0.0
                ),
                iterations=result_info.iterations,
                converged=result_info.converged,
            )
        except ValueError as e:
            logger.warning(f"Brent's method failed: {e}. Trying bisection.")
            fallback = BisectionMethod(tol=self.tol, maxiter=self.maxiter)
            return fallback.solve(objective, bounds)


class NewtonMethod:
    """Newton-Raphson with finite-difference derivative.

    Does not require bounds for the algorithm itself, but bounds are used
    to clip iterates and provide the initial guess (midpoint).

    Args:
        tol: Absolute tolerance on the residual. Defaults to 1e-3.
        maxiter: Maximum number of iterations. Defaults to 20.
        dx: Step size for finite-difference derivative. Defaults to 1.0.
    """

    def __init__(
        self,
        tol: float = 1e-3,
        maxiter: int = 20,
        dx: float = 1.0,
    ) -> None:
        self.tol = tol
        self.maxiter = maxiter
        self.dx = dx

    def solve(
        self,
        objective: Callable[[float], float],
        bounds: tuple[float, float],
    ) -> EstimationResult:
        a, b = bounds
        x = (a + b) / 2.0
        iterations = 0

        for _ in range(self.maxiter):
            fx = objective(x)
            iterations += 1

            if abs(fx) <= self.tol:
                return EstimationResult(
                    value=x,
                    residual=abs(fx),
                    iterations=iterations,
                    converged=True,
                )

            fx_dx = objective(x + self.dx)
            iterations += 1
            derivative = (fx_dx - fx) / self.dx

            if abs(derivative) < 1e-12:
                logger.warning(
                    "Newton: near-zero derivative at x=%.6g, stopping.", x
                )
                return EstimationResult(
                    value=x,
                    residual=abs(fx),
                    iterations=iterations,
                    converged=False,
                )

            x_new = x - fx / derivative
            # Clip to bounds
            x_new = max(a, min(b, x_new))
            x = x_new

        fx = objective(x)
        iterations += 1
        return EstimationResult(
            value=x,
            residual=abs(fx),
            iterations=iterations,
            converged=abs(fx) <= self.tol,
        )
