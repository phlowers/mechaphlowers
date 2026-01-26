# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Notifier:
    """Subject in Observer pattern."""
    def __init__(self) -> None:
        self._observers: List[Observer] = []

    def bind_to(self, callback: Observer) -> None:
        self._observers.append(callback)

    def notify(self, *args, **kwargs) -> None:
        for observer in self._observers:
            observer.update(self, *args, **kwargs)


class Observer(ABC):
    """Abstract observer in Observer pattern. To be implemented by concrete observers."""
    @abstractmethod
    def update(self, notifier: Notifier, *args, **kwargs) -> None:
        """
        Receive update from subject.
        """
        pass
