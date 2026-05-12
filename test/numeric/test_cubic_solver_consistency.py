# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np

from mechaphlowers.numeric.analytical_cubic import AnalyticalRealSolver
from mechaphlowers.numeric.cubic import CardanoSolver
from mechaphlowers.numeric.eigval_batch_lapack import EigvalBatchSolver

SOLVER_CLASSES = (
    EigvalBatchSolver,
    CardanoSolver,
    AnalyticalRealSolver,
)


COEFFICIENTS = np.array(
    [
        [5.0, 2.0, -3.0, -4.0],
        [1.0, -6.0, 11.0, -6.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
)

EXPECTED_MAX_REAL_ROOTS = np.array([1.0, 3.0, 0.0])


def test_cubic_solver_implementations_agree_on_same_batch() -> None:
    results = [
        solver_cls().solve(COEFFICIENTS) for solver_cls in SOLVER_CLASSES
    ]

    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], atol=1e-8)
