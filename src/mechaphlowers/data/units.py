# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np


def grad_to_rad(angles_grad: np.ndarray):
    return angles_grad * np.pi / 200


def grad_to_deg(angles_grad: np.ndarray):
    return angles_grad * 180 / 200


def deg_to_rad(angles_deg: np.ndarray):
    return angles_deg * np.pi / 180
