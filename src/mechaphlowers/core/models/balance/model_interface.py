# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class ModelForSolver(ABC):
    @property
    @abstractmethod
    def state_vector(self) -> np.ndarray:
        pass

    @state_vector.setter
    @abstractmethod
    def state_vector(self, value) -> None:
        pass

    @abstractmethod
    def objective_function(self) -> np.ndarray:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    def dict_to_store(self) -> dict:
        return {}
