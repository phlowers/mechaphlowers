# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.numeric.methods import golden_section_search


class TestGoldenSectionSearch:
    def test_finds_minimum_of_convex_function(self):
        result = golden_section_search(
            lambda x: (x - 1.3) ** 2,
            np.array([0.0]),
            np.array([5.0]),
            np.array([1e-6]),
        )

        np.testing.assert_allclose(result, [1.3], atol=1e-6)

    def test_solves_several_intervals_at_once(self):
        minima = np.array([-2.0, 0.5, 4.0])

        result = golden_section_search(
            lambda x: (x - minima) ** 2,
            np.array([-5.0, 0.0, 3.0]),
            np.array([1.0, 2.0, 10.0]),
            np.full(3, 1e-6),
        )

        np.testing.assert_allclose(result, minima, atol=1e-6)

    def test_minimum_at_boundary_converges_to_bound(self):
        result = golden_section_search(
            lambda x: x,
            np.array([0.0]),
            np.array([1.0]),
            np.array([1e-6]),
        )

        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_degenerate_interval_returns_midpoint(self):
        result = golden_section_search(
            lambda x: (x - 2.0) ** 2,
            np.array([2.0]),
            np.array([2.0]),
            np.array([1e-6]),
        )

        np.testing.assert_allclose(result, [2.0])
