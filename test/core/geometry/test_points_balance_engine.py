# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from mechaphlowers.core.geometry.points import (
    SectionPointsChain,
)
from mechaphlowers.core.models.balance.engine import BalanceEngine


def test_graph(balance_engine_base_test: BalanceEngine):
    section_points = SectionPointsChain(
        balance_engine_base_test.section_array,
        balance_engine_base_test.span_model,
    )
