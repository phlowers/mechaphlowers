# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import copy
import logging
import warnings
from typing import Literal, Union

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]

from mechaphlowers.core.geometry.distances import DistanceResult
from mechaphlowers.core.geometry.points import Points
from mechaphlowers.core.geometry.position_engine import PositionEngine
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
from mechaphlowers.plotting.plot_distances import plot_distance_engine

logger = logging.getLogger(__name__)


SUPPORT_LABEL_PREFIX = "support: "
SPAN_LABEL_PREFIX = "span: "


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
    text: list[str] | np.ndarray,
    name: str = "Points",
) -> None:
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
    hovertext: list[str] | None = None,
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
            customdata=hovertext,
            hovertemplate=(
                f"x: %{{x}}<br>y: %{{y}}<br>z: %{{z}}<br>%{{customdata}}"
                f"<extra>{trace_profile.name}</extra>"
                if hovertext is not None
                else None
            ),
        ),
    )


def plot_points_2d(
    fig: go.Figure,
    points: np.ndarray,
    trace_profile: TraceProfile | None = None,
    view: Literal["profile", "line"] = "profile",
    hovertext: list[str] | None = None,
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
            customdata=hovertext,  # type: ignore[arg-type]
            hovertemplate=(
                f"x: %{{x}}<br>y: %{{y}}<br>%{{customdata}}"
                f"<extra>{trace_profile.name}</extra>"
                if hovertext is not None
                else None
            ),
        )
    )


def _group_labels_by_position(
    points: np.ndarray,
    set_numbers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Group set-number labels by position, joining coincident points with ", ".

    Args:
        points: (n, 3) array of point coordinates
        set_numbers: (n,) or (n, 1) array of set number labels

    Returns:
        tuple of (grouped_points (m, 3), grouped_labels (m,)) where m <= n
    """
    flat_numbers = np.ravel(set_numbers)
    seen: dict[tuple, int] = {}
    grouped_coords: list[np.ndarray] = []
    grouped_labels: list[str] = []
    for pt, sn in zip(points, flat_numbers):
        key = tuple(pt)
        if key in seen:
            idx = seen[key]
            grouped_labels[idx] = f"{grouped_labels[idx]}, {sn}"
        else:
            seen[key] = len(grouped_coords)
            grouped_coords.append(pt)
            grouped_labels.append(str(sn))
    return np.array(grouped_coords), np.array(grouped_labels)


def _build_hover_text(
    points: Points, labels: list[str], label_prefix: str = ""
) -> list[str]:
    """Build a flat list of hover labels matching the stacked-points layout.

    When ``points.points(stack=True)`` is called, each layer receives an extra
    NaN row as a separator.  This function produces a parallel list in which
    each real point in layer *i* maps to ``labels[i]``, and the NaN separator
    maps to an empty string.

    Args:
        points: Points object with ``coords`` shape ``(n_layers, n_pts, 3)``.
        labels: One label per layer; ``len(labels)`` must equal ``n_layers``.
        label_prefix: Optional prefix prepended to every label (e.g. ``"span: "``).

    Returns:
        Flat list of length ``n_layers * (n_pts + 1)``.
    """
    n_pts = points.coords.shape[1]
    result: list[str] = []
    for label in labels:
        result.extend([f"{label_prefix}{label}"] * n_pts)
        result.append("")
    return result


def _apply_name_config(
    trace: TraceProfile,
    mode: Literal["main", "background"],
    name: str | None = None,
    addendum: str = "",
) -> TraceProfile:
    """Build a configured copy of a TraceProfile without mutating the original.

    Args:
        trace: Source TraceProfile to copy.
        mode: Rendering mode ("main" or "background").
        name: Full name override. Takes precedence over addendum.
        addendum: String to append to the existing trace name.

    Returns:
        A new TraceProfile instance with mode and name applied.
    """
    t = copy.copy(trace)
    original_name = t._name
    t.mode = mode
    if name is not None:
        t.name = name
    elif addendum:
        t.name = f"{original_name} {addendum}"
    return t


def plot_support_shape(
    fig: go.Figure,
    support_shape: SupportShape,
    structure_name: str | None = None,
    points_name: str | None = None,
) -> None:
    """plot_support_shape enables to plot the support shape on a plotly figure

    Args:
        fig (go.Figure): plotly figure
        support_shape (SupportShape): SupportShape object to plot
        structure_name (str | None, optional): Legend label for the structure trace.
            Defaults to the support shape name.
        points_name (str | None, optional): Legend label for the attachment points trace.
            Defaults to "Attachment points".
    """
    _structure_name = (
        structure_name if structure_name is not None else support_shape.name
    )
    _points_name = (
        points_name if points_name is not None else "Attachment points"
    )
    grouped_points, grouped_labels = _group_labels_by_position(
        support_shape.labels_points, support_shape.set_number
    )
    plot_points_3d(
        fig, support_shape.support_points, TraceProfile(name=_structure_name)
    )
    plot_text_3d(
        fig, points=grouped_points, text=grouped_labels, name=_points_name
    )


def _validate_aspect_ratio(aspect_ratio: dict[str, float]) -> dict[str, float]:
    """Validate and normalise a custom aspect ratio dict.

    Args:
        aspect_ratio: Dictionary that must contain keys 'x', 'y', 'z' with positive float values.

    Returns:
        Validated dictionary with float values.

    Raises:
        ValueError: If the dict is missing a required key, a value is not float-convertible,
            or a value is not strictly positive.
    """
    if not isinstance(aspect_ratio, dict):
        raise ValueError(
            "aspect_ratio must be a dict with keys 'x', 'y', 'z' and positive float values."
        )

    required_keys = ("x", "y", "z")
    validated: dict[str, float] = {}
    for key in required_keys:
        if key not in aspect_ratio:
            raise ValueError(
                f"aspect_ratio is missing required key {key!r}. "
                "Expected keys are 'x', 'y', and 'z'."
            )
        value = aspect_ratio[key]
        try:
            value_float = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"aspect_ratio[{key!r}] must be a float-convertible number, got {value!r}."
            ) from exc
        if value_float <= 0:
            raise ValueError(
                f"aspect_ratio[{key!r}] must be a positive float, got {value_float!r}."
            )
        validated[key] = value_float
    return validated


def set_layout(
    fig: go.Figure,
    auto: bool = True,
    aspect_ratio: dict[str, float] | None = None,
) -> None:
    """set_layout

    Args:
        fig (go.Figure): plotly figure where layout has to be updated
        auto (bool, optional): Automatic layout based on data (scale respect). False means manual with an aspectratio of x=1, y=.5, z=.5. Only used when aspect_ratio is None. Defaults to True.
        aspect_ratio (dict[str, float] | None, optional): Custom aspect ratio dictionary with keys 'x', 'y', 'z'. When provided, forces aspectmode to 'manual' and uses these values. When None, behavior is controlled by the auto parameter. Defaults to None.

    Examples:
        >>> fig = go.Figure()
        >>> # Use default automatic layout
        >>> set_layout(fig, auto=True)
        >>>
        >>> # Use custom aspect ratio (e.g., from compute_aspect_ratio)
        >>> custom_aspect = {'x': 0.5, 'y': 0.3, 'z': 10.0}
        >>> set_layout(fig, aspect_ratio=custom_aspect)
    """

    auto = bool(auto)

    if aspect_ratio is not None:
        aspect_mode: str = "manual"
        final_aspect_ratio = _validate_aspect_ratio(aspect_ratio)
        zoom: float = 5
    else:
        aspect_mode = "data" if auto else "manual"
        final_aspect_ratio = {'x': 1, 'y': 0.5, 'z': 0.5}
        zoom = 1 if auto else 5

    fig.update_layout(
        scene={
            'xaxis_title': "X (m)",
            'yaxis_title': "Y (m)",
            'zaxis_title': "Z (m)",
            'aspectratio': final_aspect_ratio,
            'aspectmode': aspect_mode,
            'camera': {
                'up': {'x': 0, 'y': 0, 'z': 1},
                'eye': {'x': -0.5, 'y': -5 / zoom, 'z': 2 / zoom},
            },
        }
    )


class PlotEngine(Observer):
    """PlotEngine renders power-line sections on Plotly figures.

    It accepts either a :class:`~mechaphlowers.core.models.balance.engine.BalanceEngine`
    or an already-constructed :class:`~mechaphlowers.core.geometry.position_engine.PositionEngine`.
    When a ``BalanceEngine`` is passed, a ``PositionEngine`` is created
    automatically and exposed via :attr:`position_engine`.

    Reactivity is preserved through a two-hop observer chain:

    .. code-block:: text

        BalanceEngine  ──notifies──►  PositionEngine  ──notifies──►  PlotEngine

    Args:
        engine: A :class:`BalanceEngine` or :class:`PositionEngine` instance.

    Example:
        >>> import plotly.graph_objects as go
        >>> # Pass a BalanceEngine directly (PositionEngine is auto-created)
        >>> plt_engine = PlotEngine(balance_engine)
        >>> fig = go.Figure()
        >>> plt_engine.preview_line3d(fig)
        >>> fig.show()
        >>> # Access the position engine for headless computation:
        >>> pos_engine = plt_engine.position_engine
        >>> pos_engine.get_supports_points()
        array(...)
        >>> # Or build a PositionEngine first and pass it in:
        >>> from mechaphlowers.core.geometry.position_engine import PositionEngine
        >>> pos_engine = PositionEngine(balance_engine)
        >>> plt_engine = PlotEngine(pos_engine)
    """

    def __init__(
        self,
        engine: Union[BalanceEngine, PositionEngine],
    ) -> None:
        if isinstance(engine, BalanceEngine):
            self.position_engine = PositionEngine(engine)
        elif isinstance(engine, PositionEngine):
            self.position_engine = engine
        else:
            raise TypeError(
                "engine must be a BalanceEngine or PositionEngine instance"
            )
        self.position_engine.bind_to(self)

    # ── Observer callback ─────────────────────────────────────────────────────

    def update(self, notifier: Notifier) -> None:
        """Receive notification from :class:`PositionEngine`.

        The ``PositionEngine`` has already refreshed all coordinates before
        calling this method, so no additional state update is required here.
        """
        logger.debug("Plot engine notified from position engine.")

    # ── Backward-compatible delegating properties ─────────────────────────────
    # These forward attribute access to position_engine so that existing code
    # that accesses plt_engine.span_model, plt_engine.coords_calculator, etc. keeps
    # working without modification.

    @property
    def span_model(self):
        """Delegating property — see :attr:`PositionEngine.span_model`."""
        return self.position_engine.span_model

    @property
    def cable_loads(self):
        """Delegating property — see :attr:`PositionEngine.cable_loads`."""
        return self.position_engine.cable_loads

    @property
    def section_array(self):
        """Delegating property — see :attr:`PositionEngine.section_array`."""
        return self.position_engine.section_array

    @property
    def coords_calculator(self):
        """Delegating property — see :attr:`PositionEngine.coords_calculator`."""
        return self.position_engine.coords_calculator

    @property
    def section_pts(self):
        warnings.warn(
            "section_pts was renamed coords_calculator. Use self.coords_calculator instead",
            DeprecationWarning,
        )
        return self.coords_calculator

    @property
    def beta(self) -> np.ndarray:
        """Delegating property — see :attr:`PositionEngine.beta`."""
        return self.position_engine.beta

    @property
    def support_names(self) -> list[str]:
        """Names of the supports as defined in the section array."""
        return list(self.section_array.data_original["name"])

    @property
    def span_names(self) -> list[str]:
        """Names of the spans, each being the concatenation of the two adjacent support names.

        For supports ``["A", "B", "C"]`` this returns ``["A-B", "B-C"]``.
        """
        names = self.support_names
        return [f"{names[i]}-{names[i + 1]}" for i in range(len(names) - 1)]

    # ── Backward-compatible delegating methods ────────────────────────────────

    def initialize_engine(self, balance_engine: BalanceEngine) -> None:
        """Delegate to :meth:`PositionEngine.initialize_engine`."""
        self.position_engine.initialize_engine(balance_engine)

    def reset(self, balance_engine: BalanceEngine) -> None:
        """Delegate to :meth:`PositionEngine.reset`."""
        self.position_engine.reset(balance_engine)

    def add_obstacles(self, obstacles_array: ObstacleArray) -> None:
        """Delegate to :meth:`PositionEngine.add_obstacles`."""
        self.position_engine.add_obstacles(obstacles_array)

    def get_spans_points(
        self, frame: Literal["section", "localsection", "cable"]
    ) -> np.ndarray:
        """Delegate to :meth:`PositionEngine.get_spans_points`."""
        return self.position_engine.get_spans_points(frame)

    def get_supports_points(self) -> np.ndarray:
        """Delegate to :meth:`PositionEngine.get_supports_points`."""
        return self.position_engine.get_supports_points()

    def get_insulators_points(self) -> np.ndarray:
        """Delegate to :meth:`PositionEngine.get_insulators_points`."""
        return self.position_engine.get_insulators_points()

    def get_obstacles_points(self) -> np.ndarray:
        """Delegate to :meth:`PositionEngine.get_obstacles_points`."""
        return self.position_engine.get_obstacles_points()

    def obstacles_dict(self, project=False, frame_index=0) -> dict:
        """Returns a dictionary storing object coordinates.

        Key is object name, value is coordinates of object.

        Format: {'obs_0': [[x0, y0, z0], [x1, y1, z1], ...]}
        """
        return self.position_engine.obstacles_dict(project, frame_index)

    def get_loads_coords(
        self, project: bool = False, frame_index: int = 0
    ) -> dict:
        """Delegate to :meth:`PositionEngine.get_loads_coords`."""
        return self.position_engine.get_loads_coords(project, frame_index)

    def get_points_for_plot(
        self, project: bool = False, frame_index: int = 0
    ) -> tuple[Points, Points, Points]:
        """Delegate to :meth:`PositionEngine.get_points_for_plot`."""
        return self.position_engine.get_points_for_plot(project, frame_index)

    def preview_line3d(
        self,
        fig: go.Figure,
        view: Literal["full", "analysis"] = "full",
        mode: Literal["main", "background"] = "main",
        aspect_ratio: dict[str, float] | None = None,
        name_addendum: str = "",
        cable_name: str | None = None,
        cable_name_addendum: str = "",
        support_name: str | None = None,
        support_name_addendum: str = "",
        insulator_name: str | None = None,
        insulator_name_addendum: str = "",
    ) -> None:
        """Plot 3D of power lines sections

        Args:
            fig (go.Figure): plotly figure where new traces has to be added
            view (Literal['full', 'analysis'], optional): full for scale respect view, analysis for compact view. Defaults to "full".
            mode (Literal['main', 'background'], optional): Style mode for the traces. Defaults to "main".
            aspect_ratio (dict[str, float] | None, optional): Custom aspect ratio dictionary with keys 'x', 'y', 'z'.
                When provided, overrides the layout aspect ratio. Can be computed using compute_aspect_ratio(). Defaults to None.
            name_addendum (str, optional): String appended to all trace names. Defaults to "".
            cable_name (str | None, optional): Full override for the cable trace legend label.
            cable_name_addendum (str, optional): Addendum for the cable trace name (overrides name_addendum).
            support_name (str | None, optional): Full override for the support trace legend label.
            support_name_addendum (str, optional): Addendum for the support trace name (overrides name_addendum).
            insulator_name (str | None, optional): Full override for the insulator trace legend label.
            insulator_name_addendum (str, optional): Addendum for the insulator trace name (overrides name_addendum).

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

        _cable = _apply_name_config(
            cable_trace, mode, cable_name, cable_name_addendum or name_addendum
        )
        _support = _apply_name_config(
            support_trace,
            mode,
            support_name,
            support_name_addendum or name_addendum,
        )
        _insulator = _apply_name_config(
            insulator_trace,
            mode,
            insulator_name,
            insulator_name_addendum or name_addendum,
        )
        plot_points_3d(
            fig,
            span.points(True),
            _cable,
            hovertext=_build_hover_text(
                span, self.span_names, SUPPORT_LABEL_PREFIX
            ),
        )
        plot_points_3d(
            fig,
            supports.points(True),
            _support,
            hovertext=_build_hover_text(
                supports, self.support_names, SUPPORT_LABEL_PREFIX
            ),
        )
        plot_points_3d(
            fig,
            insulators.points(True),
            _insulator,
            hovertext=_build_hover_text(
                insulators, self.support_names, SUPPORT_LABEL_PREFIX
            ),
        )

        if hasattr(self.coords_calculator, "obstacles_array"):
            obstacles = self.coords_calculator.compute_obstacle_coords()
            plot_points_3d(
                fig, obstacles.points(True), TraceProfile(name="Obstacles")
            )

        set_layout(fig, auto=_auto, aspect_ratio=aspect_ratio)

    def preview_line2d(
        self,
        fig: go.Figure,
        view: Literal["profile", "line"] = "profile",
        frame_index: int = 0,
        mode: Literal["main", "background"] = "main",
        name_addendum: str = "",
        cable_name: str | None = None,
        cable_name_addendum: str = "",
        support_name: str | None = None,
        support_name_addendum: str = "",
        insulator_name: str | None = None,
        insulator_name_addendum: str = "",
    ) -> None:
        """Plot 2D of power lines sections

        Args:
            fig (go.Figure): plotly figure where new traces has to be added
            view (Literal['profile', 'line'], optional): profile or line view. Defaults to "profile".
            frame_index (int, optional): Index of the frame for projection. Defaults to 0.
            mode (Literal['main', 'background'], optional): Rendering mode. Defaults to "main".
            name_addendum (str, optional): String appended to all trace names. Defaults to "".
            cable_name (str | None, optional): Full override for the cable trace legend label.
            cable_name_addendum (str, optional): Addendum for the cable trace name (overrides name_addendum).
            support_name (str | None, optional): Full override for the support trace legend label.
            support_name_addendum (str, optional): Addendum for the support trace name (overrides name_addendum).
            insulator_name (str | None, optional): Full override for the insulator trace legend label.
            insulator_name_addendum (str, optional): Addendum for the insulator trace name (overrides name_addendum).

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

        _cable = _apply_name_config(
            cable_trace, mode, cable_name, cable_name_addendum or name_addendum
        )
        _support = _apply_name_config(
            support_trace,
            mode,
            support_name,
            support_name_addendum or name_addendum,
        )
        _insulator = _apply_name_config(
            insulator_trace,
            mode,
            insulator_name,
            insulator_name_addendum or name_addendum,
        )
        plot_points_2d(
            fig,
            span.points(True),
            _cable,
            view=view,
            hovertext=_build_hover_text(
                span, self.span_names, SUPPORT_LABEL_PREFIX
            ),
        )
        plot_points_2d(
            fig,
            supports.points(True),
            _support,
            view=view,
            hovertext=_build_hover_text(
                supports, self.support_names, SUPPORT_LABEL_PREFIX
            ),
        )
        plot_points_2d(
            fig,
            insulators.points(True),
            _insulator,
            view=view,
            hovertext=_build_hover_text(
                insulators, self.support_names, SUPPORT_LABEL_PREFIX
            ),
        )

        if hasattr(self.coords_calculator, "obstacles_array"):
            obstacles_dict = self.obstacles_dict(
                project=True, frame_index=frame_index
            )
            for obstacle_name, obstacle_coords in obstacles_dict.items():
                plot_points_2d(
                    fig,
                    np.array(obstacle_coords),
                    TraceProfile(name=obstacle_name),
                    view=view,
                )

    def point_relative_to_absolute(
        self, span_index: int, point_relative: np.ndarray
    ) -> np.ndarray:
        """Delegate to :meth:`PositionEngine.point_relative_to_absolute`."""
        return self.position_engine.point_relative_to_absolute(
            span_index, point_relative
        )

    def point_distance(
        self,
        span_index: int,
        point: np.ndarray,
        *,
        fig: go.Figure | None = None,
    ) -> DistanceResult:
        """Compute the distance from *point* to a span, with optional plotting.

        Delegates the geometric computation to
        :meth:`PositionEngine.point_distance` and, when *fig* is provided,
        plots the result on the figure.

        Args:
            span_index: Span index in ``[0, num_supports - 2]``.
            point: Absolute coordinates of shape ``(3,)``.
            fig: Optional Plotly figure.  When supplied, the geometry is
                rendered on it.

        Returns:
            :class:`~mechaphlowers.core.geometry.distances.DistanceResult`.

        Examples:

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
        distance_result = self.position_engine.point_distance(
            span_index, point
        )

        if fig is not None:
            plot_distance_engine(
                self.position_engine.distance_engine,
                distance_result=distance_result,
                fig=fig,
                show_plane=True,
                show_projections=True,
                title_addendum=f" - Span {span_index}",
                force_layout=True,
            )
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
        return str(self.position_engine)

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}\n{self.position_engine.__str__()}"
