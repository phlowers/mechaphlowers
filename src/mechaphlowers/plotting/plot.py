# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]

from mechaphlowers.core.geometry.distances import (
    DistanceEngine,
    DistanceResult,
)
from mechaphlowers.core.geometry.planes import (
    change_local_frame,
)
from mechaphlowers.core.geometry.points import (
    Points,
    SectionPoints,
)
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.entities.arrays import ObstacleArray
from mechaphlowers.entities.reactivity import Notifier, Observer
from mechaphlowers.entities.shapes import SupportShape  # type: ignore
from mechaphlowers.plotting.plot_config import (
    TraceProfile,
    cable_trace,
    insulator_trace,
    support_trace,
)

logger = logging.getLogger(__name__)


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
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
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
            mode=trace_profile.scatter_mode,
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
            f"Incorrect value for 'view' argument: received {view}, expected 'profile' or 'line'"
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
            'xaxis_title': "X (m)",
            'yaxis_title': "Y (m)",
            'zaxis_title': "Z (m)",
            'aspectratio': aspect_ratio,
            'aspectmode': aspect_mode,
            'camera': {
                'up': {'x': 0, 'y': 0, 'z': 1},
                'eye': {'x': -0.5, 'y': -5 / zoom, 'z': 2 / zoom},
            },
        }
    )


class PlotEngine(Observer):
    """PlotEngine object

    Engine to handle plotting of power line sections from a BalanceEngine object.

    Args:
        balance_engine (BalanceEngine): BalanceEngine object to link to the PlotEngine

    Example:
        >>> from mechaphlowers.core.models.balance.engine import BalanceEngine
        >>> from mechaphlowers.plotting.plot import PlotEngine
        >>> import plotly.graph_objects as go
        >>> # Initialize balance engine and plot engine
        >>> balance_engine = BalanceEngine(...)
        >>> plt_engine = PlotEngine(balance_engine)
        >>> # Create and display 3D plot
        >>> fig = go.Figure()
        >>> plt_engine.preview_line3d(fig, view="full")
        >>> fig.show()
        >>> # Create and display 2D profile plot
        >>> fig = go.Figure()
        >>> plt_engine.preview_line2d(fig, view="profile")
        >>> fig.show()
        >>> # When balance engine is modified, plot engine updates automatically
        >>> balance_engine.add_loads(wind_pressure=50, ice_thickness=10)
        >>> # PlotEngine receives update notification via observer pattern
    """

    def __init__(
        self,
        balance_engine: BalanceEngine,
    ) -> None:
        balance_engine.bind_to(self)

        self.distance_engine = DistanceEngine()
        self.initialize_engine(balance_engine)
        self.reset(balance_engine=balance_engine)

    def initialize_engine(self, balance_engine):
        self.spans = balance_engine.balance_model.nodes_span_model
        self.cable_loads = balance_engine.cable_loads
        self.section_array = balance_engine.section_array
        self.section_pts = SectionPoints(
            section_array=self.section_array,
            span_model=self.spans,
            cable_loads=self.cable_loads,
            get_displacement=balance_engine.get_displacement,
        )

    def reset(self, balance_engine: BalanceEngine) -> None:
        """Reset the plot engine with a new balance engine if needed (e.g. after re-initialization of the balance engine)."""

        if not isinstance(balance_engine, BalanceEngine):
            raise TypeError(
                "balance_engine must be an instance of BalanceEngine"
            )
        if balance_engine.initialized is False:
            self.initialize_engine(balance_engine)
        self.section_pts.reset()

    def update(self, notifier: Notifier) -> None:
        logger.debug("Plot engine notified from balance engine.")
        # BalanceEngine notifies observers; refresh plot state when possible.
        if isinstance(notifier, BalanceEngine):
            self.reset(balance_engine=notifier)

    def add_obstacles(self, obstacles_array: ObstacleArray):
        self.obstacles_array = obstacles_array
        self.section_pts.add_obstacles(obstacles_array)

    @property
    def beta(self) -> np.ndarray:
        return self.cable_loads.load_angle

    def get_spans_points(
        self, frame: Literal["section", "localsection", "cable"]
    ) -> np.ndarray:
        return self.section_pts.get_spans(frame).points(True)

    def get_supports_points(self) -> np.ndarray:
        return self.section_pts.get_supports().points(True)

    def get_insulators_points(self) -> np.ndarray:
        return self.section_pts.get_insulators().points(True)

    def get_obstacles_points(self) -> np.ndarray:
        return self.section_pts.compute_obstacle_coords().points(True)

    def obstacles_dict(self) -> dict:
        """Returns a dictionary storing object coordinates.

        Key is object name, value is coordinates of object.

        Format: {'obs_0': [[x0, y0, z0], [x1, y1, z1], ...]}
        """
        return self.section_pts.obstacles_dict()

    def get_loads_coords(self, project=False, frame_index=0) -> dict:
        """Get a dictionary of coordinates of the loads.

        If there are two loads in spans $0$ and $2$, the format is the following:

        `{0: [x0, y0, z0], 2: [x2, y2, z2]}`

        The arguments should be the same as `get_points_for_plot()`.

        Args:
            project (bool, optional): Set to True if 2d graph: this project all objects into a support frame. Defaults to False.
            frame_index (int, optional): Index of the frame the projection is made. Should be between 0 and nb_supports-1 included. Unused if project is set to False. Defaults to 0.

        Returns:
            dict: dictionary that stores the coordinates. Key is span index. Value is a np.array of coordinates.
        """
        spans_points, _, _ = self.get_points_for_plot(project, frame_index)
        loads_spans_idx, loads_points_idx = self.spans.loads_indices
        result_dict = {}
        for index_in_small_array, span_index in enumerate(loads_spans_idx):
            # point_index is the index of the load point in spans_points.coords
            point_index = loads_points_idx[index_in_small_array]
            result_dict[int(span_index)] = spans_points.coords[
                span_index, point_index
            ]
        return result_dict

    def get_points_for_plot(
        self, project=False, frame_index=0
    ) -> tuple[Points, Points, Points]:
        """Get Points objects for span, supports and insulators.
        Can be used for plotting 2D or 3D graphs.

        Args:
            project (bool, optional): Set to True if 2d graph: this project all objects into a support frame. Defaults to False.
            frame_index (int, optional): Index of the frame the projection is made. Should be between 0 and nb_supports-1 included. Unused if project is set to False. Defaults to 0.

        Returns:
            tuple[Points, Points, Points]: Points for spans, supports and insulators respectively.

        Raises:
            ValueError: frame_index is out of range
        """
        return self.section_pts.get_points_for_plot(project, frame_index)

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
                f"Incorrect value for 'mode' argument: received {mode}, expected 'background' or 'main'"
            )

        span, supports, insulators = self.get_points_for_plot(project=False)

        plot_points_3d(fig, span.points(True), cable_trace(mode=mode))
        plot_points_3d(fig, supports.points(True), support_trace(mode=mode))
        plot_points_3d(
            fig, insulators.points(True), insulator_trace(mode=mode)
        )

        if hasattr(self.section_pts, "obstacles_array"):
            obstacles = self.section_pts.compute_obstacle_coords()
            plot_points_3d(
                fig, obstacles.points(True), TraceProfile(name="Obstacles")
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
                f"Incorrect value for 'view' argument: received {view}, expected 'profile' or 'line'"
            )

        if mode not in ["main", "background"]:
            raise ValueError(
                f"Incorrect value for 'mode' argument: received {mode}, expected 'background' or 'main'"
            )

        if view == "profile":
            fig.update_layout(
                yaxis={"autorange": True},
            )

        else:
            fig.update_layout(
                yaxis={"scaleanchor": "x", "scaleratio": 1},
            )

        span, supports, insulators = self.get_points_for_plot(
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

    def point_relative_to_absolute(
        self, span_index: int, point_relative: np.ndarray
    ) -> np.ndarray:
        """Convert a point from span-local frame to absolute coordinates via frame change.

        Performs a coordinate frame transformation from the span-local reference frame
        to the absolute global coordinate system.

        Span-local frame definition:
        - X axis: along the span direction in the XY plane
        - Y axis: perpendicular to the span direction in the XY plane
        - Z axis: vertical (global Z)

        Args:
            span_index: Index of the span to analyze (0 to num_supports-2).
            point_relative: Relative coordinate [x, y, z] in the span-local frame.

        Returns:
            Absolute point coordinates in the global frame as array of shape (3,).

        Raises:
            IndexError: If span_index is out of range.
            ValueError: If point_relative has invalid shape or span has zero XY extent.
        """

        point_relative = np.asarray(point_relative)
        if point_relative.shape != (3,):
            raise ValueError("point_relative must be a 1D array of shape (3,)")

        ground_supports = self.section_pts.supports_ground_coords
        if span_index < 0 or span_index >= len(ground_supports) - 1:
            raise IndexError(
                f"span_index {span_index} out of range [0, {len(ground_supports) - 2}]"
            )

        # Perform frame change from span-local to absolute coordinates
        support_start = ground_supports[span_index]
        support_end = ground_supports[span_index + 1]

        absolute_point = change_local_frame(
            support_start, support_end, point_relative
        )

        return absolute_point

    def point_distance(
        self,
        span_index: int,
        point: np.ndarray,
        *,
        fig: go.Figure | None = None,
    ) -> DistanceResult:
        """Point distance analysis: compute the distance from a point to a span and plot the configuration on the provided figure.

        Args:
            span_index: Index of the span to analyze (0 to num_supports-2).
            point: Absolute coordinates of the point to analyze, as array of shape (3,).
            fig: Optional plotly figure where the configuration will be plotted. If None, no plot is generated.

        Returns:
            DistanceResult: Object containing the distance analysis results, including the distance value and coordinates of the closest point on the span.

        Example:
            >>> balance_engine = ...  # BalanceEngine object with computed balance (use data.catalog.sample_section_factory for sample data)
            >>> plt_engine = PlotEngine(balance_engine)
            >>> point = np.array(
            ...     [10.0, 5.0, 2.0]
            ... )  # Absolute coordinates of the point to analyze
            >>> fig = figure_factory()
            >>> distance_result = plt_engine.point_distance(span_index=0, point=point)
            # ...get a distance result object with the distance and closest point coordinates

            >>> fig.show()
        """
        # Validate inputs and convert relative coordinates to absolute
        point = np.asarray(point)
        if point.shape != (3,):
            raise ValueError("point must be a 1D array of shape (3,)")

        # Get support points
        ground_supports = self.section_pts.supports_ground_coords.copy()
        if span_index < 0 or span_index >= len(ground_supports) - 1:
            raise IndexError(
                f"span_index {span_index} out of range [0, {len(ground_supports) - 2}]"
            )

        self.distance_engine.add_span_frame(
            ground_supports[span_index], ground_supports[span_index + 1]
        )
        self.distance_engine.add_curves(
            self.section_pts.get_spans(frame="section").coords[span_index]
        )
        distance_result = self.distance_engine.plane_distance(
            point, frame="span"
        )

        if fig is not None:
            self.distance_engine.plot(
                distance_result=distance_result,
                fig=fig,
                show_plane=True,
                show_projections=True,
                title_addendum=f" - Span {span_index}",
                force_layout=True,
            )

            # Update layout
            fig.update_layout(
                title=f"Point Distance Analysis - Span {span_index}",
                scene=dict(
                    xaxis_title="X (m)",
                    yaxis_title="Y (m)",
                    zaxis_title="Z (m)",
                    aspectmode="data",
                ),
                showlegend=True,
                legend=dict(x=0.02, y=0.98),
            )

        return distance_result

    def __str__(self) -> str:
        return (
            f"number of supports: {self.section_array.data.span_length.shape[0]}\n"
            f"parameter: {self.spans.sagging_parameter}\n"
            f"wind: {self.cable_loads.wind_pressure}\n"
            f"ice: {self.cable_loads.ice_thickness}\n"
            f"beta: {self.beta}\n"
        )

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}\n{self.__str__()}"
