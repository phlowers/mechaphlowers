# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from mechaphlowers.core.models.cable.thermal import get_cable_temperature
from mechaphlowers.entities.arrays import CableArray


def test_thermohl_cable_temp(cable_array_AM600: CableArray):
    get_cable_temperature(cable_array_AM600)
