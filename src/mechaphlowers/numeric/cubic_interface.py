# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Interface for cubic polynomial solvers.

All cubic solvers must implement :class:`ICubicSolver` so that they can be
swapped transparently in the balance engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ICubicSolver(ABC):
    """Abstract interface for solving batches of cubic polynomials.

    A cubic polynomial is defined by four coefficients:

        a0 * x**3 + b0 * x**2 + c0 * x + d0 = 0

    Implementations receive an ``(M, 4)`` array of coefficients (one row per
    polynomial) and return an ``(M,)`` array containing the largest real root
    of each polynomial (when *only_max_real* is ``True``) or an ``(M, 3)``
    array of all roots.
    """

    @abstractmethod
    def solve(self, p: np.ndarray, only_max_real: bool = True) -> np.ndarray:
        """Solve a batch of cubic polynomials.

        Parameters
        ----------
        p : np.ndarray
            Coefficient array of shape ``(M, 4)`` where each row is
            ``[a0, b0, c0, d0]``.
        only_max_real : bool, optional
            If ``True`` (default), return only the largest real root per
            polynomial as an ``(M,)`` array.  If ``False``, return all
            three roots as an ``(M, 3)`` array (may contain complex values).

        Returns
        -------
        np.ndarray
            Roots array.
        """
        ...
