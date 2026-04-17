# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Cubic polynomial solver using a direct trigonometric/Cardano formula.

This solver avoids LAPACK completely and works entirely in real arithmetic.
It targets the tiny batch sizes (n = 3–6 polynomials) in the balance engine,
where the fixed cost of a LAPACK ``dgeev`` call can exceed the actual
computation.

Algorithm
---------
For a monic cubic  ``x³ + a·x² + b·x + c = 0``  (normalised from  ``a0·x³…``),
substitute  ``x = t − a/3``  to obtain the **depressed cubic**
``t³ + p·t + q = 0``  with::

    p = b − a²/3
    q = a/3 · (2a²/9 − b) + c

Discriminant  ``D = q²/4 + p³/27``:

* **D ≥ 0** — one real root, two complex conjugates.
  Solved via the signed-real Cardano formula (no complex arithmetic)::

      A = −q/2 + √D,   B = −q/2 − √D
      t = sign(A)·|A|^(1/3) + sign(B)·|B|^(1/3)

* **D < 0** — three distinct real roots.
  Solved via the trigonometric method; the *maximum* root is always::

      t_max = 2·√(−p/3) · cos( arccos(−q / (2·(−p/3)^(3/2))) / 3 )

Both paths are merged branchlessly with ``numpy.where``; no masks, no array
indexing, no complex numbers.
"""

from __future__ import annotations

import numpy as np

from mechaphlowers.numeric.cubic_interface import ICubicSolver


class AnalyticalRealSolver(ICubicSolver):
    """Solve cubic polynomials using direct trigonometric/Cardano formulas.

    All arithmetic is performed in real numbers, avoiding the LAPACK dispatch
    and complex-array allocation cost of :class:`EigvalBatchSolver`.

    Parameters
    ----------
    imag_threshold : float, optional
        Not used for computation (this solver is always real), kept for
        API symmetry with :class:`EigvalBatchSolver`.  Defaults to ``1e-10``.
    """

    def __init__(self, imag_threshold: float = 1e-10) -> None:
        self._imag_threshold = imag_threshold

    def solve(self, p: np.ndarray, only_max_real: bool = True) -> np.ndarray:
        """Solve a batch of cubic polynomials.

        Parameters
        ----------
        p : np.ndarray
            Coefficient array of shape ``(M, 4)`` where each row is
            ``[a0, b0, c0, d0]`` for  ``a0·x³ + b0·x² + c0·x + d0 = 0``.
        only_max_real : bool, optional
            If ``True`` (default), return the largest real root per polynomial
            as an ``(M,)`` array.  If ``False``, return all three real roots
            as an ``(M, 3)`` array (ordered descending).

        Returns
        -------
        np.ndarray
            ``(M,)`` array of max real roots when *only_max_real* is ``True``,
            else ``(M, 3)`` array of all real roots.
        """
        p = np.asarray(p, dtype=float)
        if p.ndim < 2:
            p = p[np.newaxis, :]
        if p.shape[1] != 4:
            raise ValueError(
                "Expected 3rd order polynomial with 4 "
                f"coefficients, got {p.shape[1]}."
            )

        # --- Normalise to monic: x³ + a·x² + b·x + c = 0 ---
        inv_a0 = 1.0 / p[:, 0]
        a = p[:, 1] * inv_a0
        b = p[:, 2] * inv_a0
        c = p[:, 3] * inv_a0

        # --- Depress: t³ + p_dep·t + q_dep = 0  via  x = t − a/3 ---
        a3 = a / 3.0
        a3_sq = a3 * a3
        p_dep = b - 3.0 * a3_sq  # b − a²/3
        q_dep = a3 * (2.0 * a3_sq - b) + c  # a/3·(2a²/9 − b) + c

        # --- Discriminant ---
        D = 0.25 * q_dep * q_dep + p_dep * p_dep * p_dep / 27.0

        if only_max_real:
            return self._max_real_root(a3, p_dep, q_dep, D)

        return self._all_real_roots(a3, p_dep, q_dep, D)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_real_root(
        a3: np.ndarray,
        p_dep: np.ndarray,
        q_dep: np.ndarray,
        D: np.ndarray,
    ) -> np.ndarray:
        """Compute only the maximum real root for each polynomial."""
        half_q = 0.5 * q_dep

        # --- D ≥ 0 branch: signed real cube roots (Cardano) ---
        sqrt_D = np.sqrt(np.maximum(D, 0.0))
        A = -half_q + sqrt_D
        B = -half_q - sqrt_D
        cardano_root = np.sign(A) * np.cbrt(np.abs(A)) + np.sign(B) * np.cbrt(
            np.abs(B)
        )

        # --- D < 0 branch: trigonometric method ---
        # Guard against p_dep = 0 (triple root); p_dep < 0 always holds here.
        neg_p3 = np.maximum(-p_dep / 3.0, 0.0)
        sqrt_neg_p3 = np.sqrt(neg_p3)
        m = 2.0 * sqrt_neg_p3
        # cos argument: −q / (2·(−p/3)^(3/2));  clipped to [−1, 1] for safety
        denom = 2.0 * neg_p3 * sqrt_neg_p3 + 1e-300
        cos_arg = np.clip(-q_dep / denom, -1.0, 1.0)
        trig_root = m * np.cos(np.arccos(cos_arg) / 3.0)

        # Branchless merge: use Cardano when D ≥ 0, trig when D < 0
        return np.where(D >= 0.0, cardano_root, trig_root) - a3

    @staticmethod
    def _all_real_roots(
        a3: np.ndarray,
        p_dep: np.ndarray,
        q_dep: np.ndarray,
        D: np.ndarray,
    ) -> np.ndarray:
        """Compute all three roots for each polynomial.

        For D ≥ 0 (one real + two complex): the two complex roots are
        returned as their real parts (imaginary parts are discarded).
        For D < 0 (three real): all three trigonometric roots are returned.
        Results are sorted in descending order per polynomial.
        """

        half_q = 0.5 * q_dep
        sqrt_D = np.sqrt(np.maximum(D, 0.0))

        # D ≥ 0 path: one real root via Cardano; approximate the two complex
        # roots by their real part (−(S+U)/2 = −cardano_root/2).
        A = -half_q + sqrt_D
        B = -half_q - sqrt_D
        S = np.sign(A) * np.cbrt(np.abs(A))
        U = np.sign(B) * np.cbrt(np.abs(B))
        su = S + U
        # real parts of complex conjugate pair
        cardano_r1 = su
        cardano_r2 = -0.5 * su
        cardano_r3 = cardano_r2  # same real part

        # D < 0 path: three real trigonometric roots
        neg_p3 = np.maximum(-p_dep / 3.0, 0.0)
        sqrt_neg_p3 = np.sqrt(neg_p3)
        m = 2.0 * sqrt_neg_p3
        denom = 2.0 * neg_p3 * sqrt_neg_p3 + 1e-300
        cos_arg = np.clip(-q_dep / denom, -1.0, 1.0)
        theta = np.arccos(cos_arg) / 3.0
        two_pi_3 = 2.0 * np.pi / 3.0
        trig_r1 = m * np.cos(theta)
        trig_r2 = m * np.cos(theta - two_pi_3)
        trig_r3 = m * np.cos(theta + two_pi_3)

        r1 = np.where(D >= 0.0, cardano_r1, trig_r1) - a3
        r2 = np.where(D >= 0.0, cardano_r2, trig_r2) - a3
        r3 = np.where(D >= 0.0, cardano_r3, trig_r3) - a3

        roots = np.stack([r1, r2, r3], axis=1)  # (n, 3)
        roots.sort(axis=1)
        return roots[:, ::-1]  # descending
