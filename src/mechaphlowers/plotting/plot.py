# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]

from mechaphlowers.core.geometry.points import SectionPoints  # type: ignore

if TYPE_CHECKING:
    from mechaphlowers.api.frames import SectionDataFrame

from mechaphlowers.config import options as cfg


def plot_points_3d(fig, points, color=None, width=3, size=None, name="Points"):
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers+lines',
            marker={
                'size': cfg.graphics.marker_size if size is None else size,
                'color': color,
            },
            line={'color': color, 'width': width},
            name=name,
        ),
    )


def plot_line(fig: go.Figure, points: np.ndarray) -> None:
    """Plot the points of the cable onto the figure given

    Args:
        fig (go.Figure): plotly figure
        points (np.ndarray): points of all the cables of the section in point format (3 x n)
    """

    plot_points_3d(fig, points, color="red", width=8, name="Cable")


def plot_support(fig: go.Figure, points: np.ndarray) -> None:
    """Plot the points of the support onto the figure given

    Args:
        fig (go.Figure): plotly figure
        points (np.ndarray): points of all the supports of the section in point format (3 x n)
    """
    plot_points_3d(fig, points, color="green", width=8, name="Supports")


def plot_insulator(fig: go.Figure, points: np.ndarray) -> None:
    """Plot the points of the insulators onto the figure given

    Args:
        fig (go.Figure): plotly figure
        points (np.ndarray): points of all the insulators of the section in point format (3 x n)
    """
    plot_points_3d(fig, points, color="orange", width=8, name="Insulators")


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
        0.1 if auto else 1
    )  # perhaps this approx of the zoom will not be adequate for all cases
    aspect_ratio = {'x': 1, 'y': 0.05, 'z': 0.5}

    fig.update_layout(
        scene={
            'aspectratio': aspect_ratio,
            'aspectmode': aspect_mode,
            'camera': {
                'up': {'x': 0, 'y': 0, 'z': 1},
                'eye': {'x': 0, 'y': -1 / zoom, 'z': 0},
            },
        }
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
        beta = np.zeros_like(spans.span_length)
        if self.section.cable_loads is not None:
            beta = self.section.cable_loads.load_angle * 180 / np.pi
        section_pts.beta = beta
        plot_line(fig, section_pts.get_spans("section").points(True))

        plot_support(fig, section_pts.get_supports().points(True))

        plot_insulator(fig, section_pts.get_insulators().points(True))

        set_layout(fig, auto=_auto)
