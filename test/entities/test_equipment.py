# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest

from mechaphlowers.entities.equipment import Spacer


@pytest.mark.parametrize(
    "bundle_number, expected",
    [
        (3, 0.2),
        (4, 0.2),
        (1, 0.0),
        (2, 0.0),
    ],
)
def test_spacer_height(bundle_number: int, expected: float) -> None:
    spacer = Spacer(length=0.2)
    assert spacer.height(bundle_number) == expected
