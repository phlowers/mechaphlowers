# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Tests for EigvalBatchSolver, CardanoSolver, and AnalyticalRealSolver (cubic interface)."""

import numpy as np
import pytest

from mechaphlowers.numeric.analytical_cubic import AnalyticalRealSolver
from mechaphlowers.numeric.cubic import CardanoSolver, cubic_roots
from mechaphlowers.numeric.eigval_batch_lapack import EigvalBatchSolver


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture(params=["cardano", "eigval_batch", "analytical_real"])
def solver(request):
    """Parametrised fixture returning each ICubicSolver implementation."""
    if request.param == "cardano":
        return CardanoSolver()
    if request.param == "analytical_real":
        return AnalyticalRealSolver()
    return EigvalBatchSolver()


# --- Test data ---------------------------------------------------------------

# Coefficients from test_cubic.py
P_MULTI = np.array([
    [5, 2, -3, -4],
    [0.1, 2, -3, -4],
    [0.01, 0, -3, -4],
    [0.00001, 0, 0, -4],
    [1e-10, 0, 0, 0],
])

# Known max-real roots (from existing test_cubic.py)
EXPECTED_MAX_REAL = np.array([1.0, 2.17988479, 17.95219749, 73.68062997, 0.0])


# --- Tests -------------------------------------------------------------------

class TestEigvalBatchSolver:
    """Tests specific to the eigval-batch implementation."""

    def test_solve_only_max_real(self):
        solver = EigvalBatchSolver()
        result = solver.solve(P_MULTI, only_max_real=True)
        np.testing.assert_array_almost_equal(result, EXPECTED_MAX_REAL, decimal=4)

    def test_solve_all_roots_shape(self):
        solver = EigvalBatchSolver()
        result = solver.solve(P_MULTI, only_max_real=False)
        assert result.shape == (5, 3)

    def test_single_polynomial(self):
        solver = EigvalBatchSolver()
        p = np.array([1, 7, -806, -1050])
        result = solver.solve(p, only_max_real=True)
        np.testing.assert_almost_equal(result[0], 25.80760451, decimal=4)

    def test_invalid_shape_raises(self):
        solver = EigvalBatchSolver()
        with pytest.raises(ValueError, match="4 coefficients"):
            solver.solve(np.array([[1, 2, 3]]))


class TestCardanoSolver:
    """Tests specific to the Cardano wrapper."""

    def test_solve_only_max_real(self):
        solver = CardanoSolver()
        result = solver.solve(P_MULTI, only_max_real=True)
        np.testing.assert_array_almost_equal(result, EXPECTED_MAX_REAL, decimal=4)

    def test_solve_all_roots_shape(self):
        solver = CardanoSolver()
        result = solver.solve(P_MULTI, only_max_real=False)
        assert result.shape == (5, 3)


class TestAnalyticalRealSolver:
    """Tests specific to the trigonometric/Cardano real solver."""

    def test_solve_only_max_real(self):
        solver = AnalyticalRealSolver()
        result = solver.solve(P_MULTI, only_max_real=True)
        np.testing.assert_array_almost_equal(result, EXPECTED_MAX_REAL, decimal=4)

    def test_solve_all_roots_shape(self):
        solver = AnalyticalRealSolver()
        result = solver.solve(P_MULTI, only_max_real=False)
        assert result.shape == (5, 3)

    def test_single_polynomial(self):
        solver = AnalyticalRealSolver()
        p = np.array([1, 7, -806, -1050])
        result = solver.solve(p, only_max_real=True)
        np.testing.assert_almost_equal(result[0], 25.80760451, decimal=4)

    def test_invalid_shape_raises(self):
        solver = AnalyticalRealSolver()
        with pytest.raises(ValueError, match="4 coefficients"):
            solver.solve(np.array([[1, 2, 3]]))

    def test_three_real_roots_case(self):
        """Polynomial with three real roots: x³ − 6x² + 11x − 6 = (x−1)(x−2)(x−3)."""
        solver = AnalyticalRealSolver()
        p = np.array([[1, -6, 11, -6]])
        result = solver.solve(p, only_max_real=True)
        np.testing.assert_almost_equal(result[0], 3.0, decimal=5)

    def test_one_real_root_case(self):
        """Polynomial with one real root: x³ + x + 1 (D > 0)."""
        solver = AnalyticalRealSolver()
        p = np.array([[1, 0, 1, 1]])
        result = solver.solve(p, only_max_real=True)
        # Verify the root satisfies the polynomial
        r = result[0]
        residual = r**3 + r + 1
        np.testing.assert_almost_equal(residual, 0.0, decimal=6)


class TestSolversAgree:
    """Cross-check that all implementations agree on the same inputs."""

    def test_max_real_roots_agree(self, solver):
        result = solver.solve(P_MULTI, only_max_real=True)
        np.testing.assert_array_almost_equal(result, EXPECTED_MAX_REAL, decimal=4)

    def test_random_polynomials_agree(self):
        """All solvers must return the same max-real root for random cubics
        where at least one real root is positive (matching the balance engine
        use case where the physical parameter is always positive)."""
        rng = np.random.default_rng(42)
        p = rng.standard_normal((50, 4))
        p[:, 0] = np.abs(p[:, 0]) + 0.01  # ensure a0 > 0
        # Force d0 < 0 so there is always a positive real root
        # (sign change at x=0 since poly(0) = d0 < 0, poly(+inf) = +inf)
        p[:, 3] = -np.abs(p[:, 3]) - 0.01

        cardano = CardanoSolver().solve(p, only_max_real=True)
        eigval = EigvalBatchSolver().solve(p, only_max_real=True)
        analytical = AnalyticalRealSolver().solve(p, only_max_real=True)
        np.testing.assert_array_almost_equal(cardano, eigval, decimal=4)
        np.testing.assert_array_almost_equal(cardano, analytical, decimal=4)
