# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Literal, Self

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]

from mechaphlowers.core.geometry.points import (
    SectionPoints,
)
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.entities.shapes import SupportShape  # type: ignore

if TYPE_CHECKING:
    from mechaphlowers.api.frames import SectionDataFrame

from mechaphlowers.config import options as cfg

logger = logging.getLogger(__name__)


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
    def mode(self, value):
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

    def __call__(self, mode) -> Self:
        self.mode = mode
        return self


cable_trace = TraceProfile(**cfg.graphics.cable_trace_profile)
insulator_trace = TraceProfile(**cfg.graphics.insulator_trace_profile)
support_trace = TraceProfile(**cfg.graphics.support_trace_profile)


def figure_factory(context=Literal["std", "blank"]) -> go.Figure:
    """create_figure creates a plotly figure

    Returns:
        go.Figure: plotly figure
    """
    fig = go.Figure()
    if context == "std":
        fig.update_layout(
            autosize=True,
            height=800,
            width=1400,
            scene=dict(
                xaxis=dict(
                    backgroundcolor="gainsboro",
                    gridcolor="dimgray",
                ),
                yaxis=dict(
                    backgroundcolor="gainsboro",
                    gridcolor="dimgray",
                ),
                zaxis=dict(
                    backgroundcolor="gainsboro",
                    gridcolor="dimgray",
                ),
            ),
            scene_camera=dict(eye=dict(x=0.9, y=0.1, z=-0.1)),
        )
    elif context == "blank":
        pass
    else:
        raise ValueError(
            f"Unknown context: {context} try 'blank' or 'jupyter'"
        )
    return fig


def plot_text_3d(
    fig: go.Figure,
    points: np.ndarray,
    text: np.ndarray,
    color=None,
    width=3,
    size=None,
    name="Points",
):
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers+text",
            name=name,
            text=text,
            textposition="top center",
        ),
    )


def plot_points_3d(
    fig: go.Figure,
    points: np.ndarray,
    trace_profile: TraceProfile | None = None,
) -> None:
    if trace_profile is None:
        trace_profile = TraceProfile()

    trace_profile.dimension = "3d"
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers+lines',
            marker=trace_profile.marker,
            line=trace_profile.line,
            opacity=trace_profile.opacity,
            name=trace_profile.name,
        ),
    )


def plot_points_2d(
    fig: go.Figure,
    points: np.ndarray,
    trace_profile: TraceProfile | None = None,
    view: Literal["profile", "line"] = "profile",
) -> None:
    if trace_profile is None:
        trace_profile = TraceProfile()

    trace_profile.dimension = "2d"
    v_coords = points[:, 2]
    if view == "line":
        h_coords = points[:, 1]
    elif view == "profile":
        h_coords = points[:, 0]
    else:
        raise ValueError(
            f"Incorrect value for 'view' argument: recieved {view}, expected 'profile' or 'line'"
        )

    fig.add_trace(
        go.Scatter(
            x=h_coords,
            y=v_coords,
            mode='markers+lines',
            marker=trace_profile.marker,
            line=trace_profile.line,
            opacity=trace_profile.opacity,
            name=trace_profile.name,
        )
    )


def plot_support_shape(fig: go.Figure, support_shape: SupportShape):
    """plot_support_shape enables to plot the support shape on a plotly figure

    Args:
        fig (go.Figure): plotly figure
        support_shape (SupportShape): SupportShape object to plot
    """
    plot_points_3d(fig, support_shape.support_points)
    plot_text_3d(
        fig, points=support_shape.labels_points, text=support_shape.set_number
    )


def set_layout(fig: go.Figure, auto: bool = True) -> None:
    """set_layout

    Args:
        fig (go.Figure): plotly figure where layout has to be updated
        auto (bool, optional): Automatic layout based on data (scale respect). False means manual with an aspectradio of x=1, y=.05, z=.5. Defaults to True.
    """

    # Check input
    auto = bool(auto)
    aspect_mode: str = "data" if auto else "manual"
    zoom: float = (
        1 if auto else 5
    )  # perhaps this approx of the zoom will not be adequate for all cases
    aspect_ratio = {'x': 1, 'y': 0.5, 'z': 0.5}

    fig.update_layout(
        scene={
            'aspectratio': aspect_ratio,
            'aspectmode': aspect_mode,
            'camera': {
                'up': {'x': 0, 'y': 0, 'z': 1},
                'eye': {'x': -0.5, 'y': -5 / zoom, 'z': 2 / zoom},
            },
        }
    )


class PlotEngine:
    def __init__(
        self,
        span_model,
        cable_loads,
        section_array,
        get_displacement: Callable,
    ) -> None:
        self.spans = span_model
        self.cable_loads = cable_loads
        self.section_array = section_array

        self.section_pts = SectionPoints(
            section_array=self.section_array,
            span_model=span_model,
            cable_loads=cable_loads,
            get_displacement=get_displacement,
        )

    @property
    def beta(self):
        return self.cable_loads.load_angle

    @staticmethod
    def builder_from_balance_engine(
        balance_engine: BalanceEngine,
    ) -> PlotEngine:
        logger.debug("Plot engine initialized from balance engine.")

        return PlotEngine(
            balance_engine.span_model,
            balance_engine.cable_loads,
            balance_engine.section_array,
            balance_engine.get_displacement,
        )

    def get_spans_points(
        self, frame: Literal["section", "localsection", "cable"]
    ) -> np.ndarray:
        return self.section_pts.get_spans(frame).points(True)

    def get_supports_points(self) -> np.ndarray:
        return self.section_pts.get_supports().points(True)

    def get_insulators_points(self) -> np.ndarray:
        return self.section_pts.get_insulators().points(True)

    def preview_line3d(
        self,
        fig: go.Figure,
        view: Literal["full", "analysis"] = "full",
        mode: Literal["main", "background"] = "main",
    ) -> None:
        """Plot 3D of power lines sections

        Args:
            fig (go.Figure): plotly figure where new traces has to be added
            view (Literal['full', 'analysis'], optional): full for scale respect view, analysis for compact view. Defaults to "full".

        Raises:
            ValueError: view is not an expected value
        """

        view_map = {"full": True, "analysis": False}

        try:
            _auto = view_map[view]
        except KeyError:
            raise ValueError(
                f"{view=} : this argument has to be set to 'full' or 'analysis'"
            )

        if mode not in ["main", "background"]:
            raise ValueError(
                f"Incorrect value for 'mode' argument: recieved {mode}, expected 'background' or 'main'"
            )

        span, supports, insulators = self.section_pts.get_points_for_plot(
            project=False
        )

        plot_points_3d(fig, span.points(True), cable_trace(mode=mode))
        plot_points_3d(fig, supports.points(True), support_trace(mode=mode))
        plot_points_3d(
            fig, insulators.points(True), insulator_trace(mode=mode)
        )

        set_layout(fig, auto=_auto)

    def preview_line2d(
        self,
        fig: go.Figure,
        view: Literal["profile", "line"] = "profile",
        frame_index: int = 0,
        mode: Literal["main", "background"] = "main",
    ) -> None:
        """Plot 2D of power lines sections

        Args:
            fig (go.Figure): plotly figure where new traces has to be added
            view (Literal['full', 'analysis'], optional): full for scale respect view, analysis for compact view. Defaults to "full".

        Raises:
            ValueError: view value is invalid
        """
        if view not in ["profile", "line"]:
            raise ValueError(
                f"Incorrect value for 'view' argument: recieved {view}, expected 'profile' or 'line'"
            )

        if mode not in ["main", "background"]:
            raise ValueError(
                f"Incorrect value for 'mode' argument: recieved {mode}, expected 'background' or 'main'"
            )

        if view == "profile":
            fig.update_layout(
                yaxis={"autorange": True},
            )

        else:
            fig.update_layout(
                yaxis={"scaleanchor": "x", "scaleratio": 1},
            )

        span, supports, insulators = self.section_pts.get_points_for_plot(
            project=True, frame_index=frame_index
        )

        plot_points_2d(
            fig,
            span.points(True),
            cable_trace(mode=mode),
            view=view,
        )
        plot_points_2d(
            fig,
            supports.points(True),
            support_trace(mode=mode),
            view=view,
        )
        plot_points_2d(
            fig,
            insulators.points(True),
            insulator_trace(mode=mode),
            view=view,
        )


class PlotAccessor:
    """First accessor class for plots."""

    def __init__(self, section: SectionDataFrame):
        self.section: SectionDataFrame = section

    def line3d(
        self, fig: go.Figure, view: Literal["full", "analysis"] = "full"
    ) -> None:
        """Plot 3D of power lines sections

        Args:
            fig (go.Figure): plotly figure where new traces has to be added
            view (Literal['full', 'analysis'], optional): full for scale respect view, analysis for compact view. Defaults to "full".

        Raises:
            ValueError: view is not an expected value
        """

        view_map = {"full": True, "analysis": False}

        try:
            _auto = view_map[view]
        except KeyError:
            raise ValueError(
                f"{view=} : this argument has to be set to 'full' or 'analysis'"
            )
        spans = self.section._span_model(
            **self.section.data_container.__dict__
        )
        section_pts = SectionPoints(
            span_model=spans, **self.section.data_container.__dict__
        )

        plot_points_3d(
            fig, section_pts.get_spans("section").points(True), cable_trace
        )
        plot_points_3d(
            fig, section_pts.get_supports().points(True), support_trace
        )
        plot_points_3d(
            fig, section_pts.get_insulators().points(True), insulator_trace
        )

        set_layout(fig, auto=_auto)
