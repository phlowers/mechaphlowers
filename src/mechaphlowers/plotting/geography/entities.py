# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import TypedDict

import numpy as np


class SupportGeoInfo(TypedDict):
    gps: tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]
    elevation: np.typing.NDArray[np.float64]
    distance_to_next: np.typing.NDArray[np.float64]
    bearing_to_next: np.typing.NDArray[np.float64]
    direction_to_next: np.typing.NDArray[np.str_]
    lambert_93: tuple[
        np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]
    ]
