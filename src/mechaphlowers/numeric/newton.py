from typing import Callable

import numpy as np

from mechaphlowers.entities.errors import ConvergenceError

try:
    from scipy import optimize  # type: ignore
except ImportError:
    import mechaphlowers.numeric.scipy as optimize


def newton_solver_wrapper(
    f: Callable,
    x0: np.ndarray,
    fprime: Callable,
    args: tuple,
    caller_name: str = "newton_solver_wrapper",
) -> np.ndarray:
    solver_result = optimize.newton(
        f,
        x0,
        fprime=fprime,
        args=args,
        maxiter=10,
        tol=1e-5,
        full_output=True,
    )

    # Solver result format depends on input length:
    # for 1-element array inputs, scipy.optimize.newton takes the scipy's scalar code path.
    if not hasattr(solver_result, "converged"):
        root, root_result = solver_result
        if not root_result.converged:
            raise ConvergenceError(
                "Solver did not converge", origin=caller_name
            )
    else:
        if not solver_result.converged.all():
            raise ConvergenceError(
                "Solver did not converge",
                origin=caller_name,
            )
        root = solver_result.root
    return root
