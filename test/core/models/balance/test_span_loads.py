# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.models.balance.span_loads import SpanLoads


def test_has_load_on_span__nan() -> None:
    span_loads = SpanLoads(
        load_position_distance=[np.nan, 250, np.nan],
        load_mass=[1000, np.nan, np.nan],
        span_length=[500, 500, 500, np.nan],
    )

    np.testing.assert_equal(
        span_loads.has_load_on_span,
        np.array([False, False, False]),
    )
