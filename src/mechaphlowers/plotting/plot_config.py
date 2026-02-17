# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Self
from typing import Self
from typing_extensions import Literal
from mechaphlowers.config import options as cfg

class TraceProfile:
    """TraceProfile is a configuration class to handle a trace parameter.
    It is designed to be used with some plotly specific figures and getters are specialized to return the right format for plotly.
    """

    def __init__(
        self,
        name: str = "Test",
        color: str = "blue",
        size: float = cfg.graphics.marker_size,
        width: float = 8.0,
        opacity: float = 1.0,
    ):
        self.color = color
        self.size = size
        self.width = width
        self.name = name
        self.opacity = opacity
        self._mode = "main"

    @property
    def dimension(self) -> str:
        return self._dimension

    @dimension.setter
    def dimension(self, value: Literal["2d", "3d"]):
        if not isinstance(value, str):
            raise TypeError()
        if value not in ["2d", "3d"]:
            raise ValueError("Dimension must be either '2d' or '3d'")
        self._dimension = value

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: Literal["background", "main"]):
        if value not in ["background", "main"]:
            raise ValueError("Mode must be either 'background' or 'main'")
        self._mode = value
        if value == "background":
            self.opacity = cfg.graphics.background_opacity
        elif value == "main":
            self.opacity = 1.0

    @property
    def dashed(self) -> dict:
        if self._mode == "background":
            return {'dash': 'dot'}
        return {}

    @property
    def line(self) -> dict:
        if self._dimension == "2d":
            width = self.size
        else:
            width = self.width
        return {'color': self.color, 'width': width} | self.dashed

    @property
    def marker(self) -> dict:
        if self._dimension == "2d":
            return {'size': self.size + 1, 'color': self.color}
        else:
            return {'size': self.size, 'color': self.color}

    @property
    def name(self) -> str:
        if self._mode == "background":
            return f"{self._name} baseline"
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        self._name = value

    def __call__(self, mode: Literal["background", "main"]) -> Self:
        self.mode = mode
        return self


cable_trace = TraceProfile(**cfg.graphics.cable_trace_profile)
insulator_trace = TraceProfile(**cfg.graphics.insulator_trace_profile)
support_trace = TraceProfile(**cfg.graphics.support_trace_profile)



