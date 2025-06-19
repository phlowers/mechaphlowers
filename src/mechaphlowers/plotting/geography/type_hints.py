# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import TypedDict


class GpsPoint(TypedDict):
    lon: float
    lat: float
    
class Lambert93Point(TypedDict):
    x: float
    y: float

class SupportGeoInfo(TypedDict):
    gps: GpsPoint
    elevation: float
    distance_to_next: float
    bearing_to_next: float
    direction_to_next: str
    lambert_93: Lambert93Point