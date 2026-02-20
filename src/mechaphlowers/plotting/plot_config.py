# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from dataclasses import dataclass
from typing import Self

from typing_extensions import Literal

from mechaphlowers.config import options as cfg


@dataclass
class TraceConfig:
    """TraceConfig is a configuration class to handle a trace parameter.
    It is designed to be used with some plotly specific figures and getters are specialized to return the right format for plotly.

    Several parameters are available:
    - dimension: the dimension of the trace (2D or 3D)
    - scatter_mode: the scatter mode of the trace (markers, lines, or both)
    - color: the color of the trace
    - size: the size of the trace
    - width: the width of the trace (only for 3D)
    - name: the name of the trace (used in the legend)
    - opacity: the opacity of the trace
    - dashed: whether the trace is dashed or not
    - marker: the marker configuration of the trace
    - show_legend: whether to show the trace in the legend or not
    - legend_group: the legend group of the trace (used to group traces in the legend)
    - text: the text to display on hover for the trace
    - text_position: the position of the text on hover for the trace
    """

    dimension: Literal["2d", "3d"] = "3d"
    scatter_mode: Literal["markers", "lines", "markers+lines"] = "markers"
    color: str = "blue"
    size: float = cfg.graphics.marker_size
    width: float = 8.0
    name: str = "Test"
    opacity: float = 1.0
    dash: Literal["dot", "dash", "longdash", "solid"] | None = None
    show_legend: bool = True
    legend_group: str | None = None
    text: str | None = None
    text_position: Literal["top", "bottom", "left", "right"] | None = None
    hoverinfo: str = "skip"
    text_font: dict | None = None
    marker: dict | None = None
    line: dict | None = None
    marker_symbol: str | None = None

    def add_text_formatting(self, format_func: callable) -> Self:
        self.text_format_func = format_func

    def text_format(self, *args, **kwargs) -> str:
        if hasattr(self, "text_format_func"):
            return self.text_format_func(*args, **kwargs)
        return self.text or ""


distance_text = TraceConfig(
    dimension="3d",
    scatter_mode="text",
    color="black",
    size=cfg.graphics.marker_size,
    width=1.0,
    name="Distance",
    opacity=1.0,
    show_legend=False,
    text_position="top center",
    legend_group="distance",
    text_font={"size": 12, "color": "black"},
)
distance_text.add_text_formatting(lambda distance: f"{distance:.2f} m")


projected_distance_trace = TraceConfig(
    dimension="3d",
    scatter_mode="lines+markers",
    size=4,
    width=4.0,
    dash="dot",
    show_legend=True,
    legend_group="projections ",
    marker_symbol="diamond",
)

projected_distance_trace.add_text_formatting(
    lambda distance: f"{distance:.2f} m"
)

line3d_trace = TraceConfig(
    scatter_mode="lines",
    line=dict(color="darkred", width=4, dash="solid"),
    name="Plane Distance",
    legend_group="distance",
)

distance_points = TraceConfig(
    dimension="3d",
    scatter_mode="markers+text",
    size=10,
    text_position="top center",
    name="Distance Points",
    legend_group="distance",
)


class TraceProfile:
    """TraceProfile is a configuration class to handle a trace parameter.
    It is designed to be used with some plotly specific figures and getters are specialized to return the right format for plotly.

    Several parameters are available:
    - color: the color of the trace
    - size: the size of the trace (marker size for 2D, line width for 3D)
    - width: the width of the trace (only for 3D)
    - name: the name of the trace (used in the legend)
    - opacity: the opacity of the trace
    - dashed: whether the trace is dashed or not (only for 2D)

    A mechanism allow to set 3D parameter and the class will automatically adapt the 2D parameters:
    - width is only used for 3D, in 2D the size parameter is used as line width
    - size is used as marker size for 2D and line width for 3D
    - marker size is size + 1 for 2D to make it more visible

    Several modes are available:
    - "main": the trace is displayed as the main trace, with the color and size defined in the configuration
    - "background": the trace is displayed as a background trace, with a dashed line and



    """

    def __init__(
        self,
        name: str = "Trace",
        color: str = "blue",
        size: float = cfg.graphics.marker_size,
        width: float = 8.0,
        opacity: float = 1.0,
        scatter_mode: Literal[
            "markers", "lines", "markers+lines"
        ] = "markers+lines",
    ):
        self.color = color
        self.size = size
        self.width = width
        self.name = name
        self.opacity = opacity
        self._mode = "main"
        self._scatter_mode = scatter_mode

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

    @property
    def scatter_mode(self) -> str:
        if self._dimension == "2d":
            return "lines"
        else:
            return self._scatter_mode

    def __call__(self, mode: Literal["background", "main"]) -> Self:
        self.mode = mode
        return self


cable_trace = TraceProfile(**cfg.graphics.cable_trace_profile)
insulator_trace = TraceProfile(**cfg.graphics.insulator_trace_profile)
support_trace = TraceProfile(**cfg.graphics.support_trace_profile)
