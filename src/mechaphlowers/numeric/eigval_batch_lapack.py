# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Cubic polynomial solver using batch companion-matrix eigenvalues (LAPACK).

This solver builds one 3×3 companion matrix per polynomial and calls
:func:`numpy.linalg.eigvals` on the whole batch in a single LAPACK call.
It is typically faster than the analytical Cardano solver for the small
batch sizes (3–6 polynomials) encountered in the balance engine because it
avoids Python-level branching and intermediate array allocation.
"""

from __future__ import annotations

import numpy as np

from mechaphlowers.numeric.cubic_interface import ICubicSolver


class EigvalBatchSolver(ICubicSolver):
    """Solve cubic polynomials via companion-matrix eigenvalues.

    For each polynomial  ``a0·x³ + b0·x² + c0·x + d0 = 0``  the monic
    companion matrix is:

    .. math::

        C = \\begin{pmatrix}
            0 & 0 & -d \\\\
            1 & 0 & -c \\\\
            0 & 1 & -b
        \\end{pmatrix}

    where ``b = b0/a0``, ``c = c0/a0``, ``d = d0/a0``.  The eigenvalues of
    *C* are the roots of the polynomial.

    Parameters
    ----------
    imag_threshold : float, optional
        Roots whose imaginary part (absolute value) is larger than this
        threshold are considered complex and excluded when
        *only_max_real* is ``True``.  Defaults to ``1e-10``.
    """

    def __init__(self, imag_threshold: float = 1e-10) -> None:
        self._imag_threshold = imag_threshold

    def solve(self, p: np.ndarray, only_max_real: bool = True) -> np.ndarray:
        """Solve a batch of cubic polynomials.

        Parameters
        ----------
        p : np.ndarray
            Coefficient array of shape ``(M, 4)``.
        only_max_real : bool, optional
            If ``True``, return the largest real root per polynomial.

        Returns
        -------
        np.ndarray
            ``(M,)`` if *only_max_real*, else ``(M, 3)`` complex array.
        """
        p = np.asarray(p)
        if p.ndim < 2:
            p = p[np.newaxis, :]
        if p.shape[1] != 4:
            raise ValueError(
                "Expected 3rd order polynomial with 4 "
                f"coefficients, got {p.shape[1]}."
            )

        n = p.shape[0]
        # Normalise to monic form: x³ + a·x² + b·x + c = 0
        a = p[:, 1] / p[:, 0]
        b = p[:, 2] / p[:, 0]
        c = p[:, 3] / p[:, 0]

        # Build companion matrices  (n, 3, 3)
        comp = np.zeros((n, 3, 3))
        comp[:, 0, 2] = -c
        comp[:, 1, 2] = -b
        comp[:, 2, 2] = -a
        comp[:, 1, 0] = 1.0
        comp[:, 2, 1] = 1.0

        # Single LAPACK call for all matrices
        eigenvalues = np.linalg.eigvals(comp)  # (n, 3)

        if only_max_real:
            real_mask = np.abs(eigenvalues.imag) < self._imag_threshold
            real_vals = np.where(real_mask, eigenvalues.real, -np.inf)
            return np.max(real_vals, axis=1)

        return eigenvalues
