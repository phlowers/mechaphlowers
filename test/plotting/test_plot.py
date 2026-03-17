# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytest
from pytest import fixture

from mechaphlowers.config import options as options
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.data.catalog.catalog import sample_cable_catalog
from mechaphlowers.data.catalog.sample_section import (
    section_factory_sample_data,
)
from mechaphlowers.data.units import convert_weight_to_mass
from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
)
from mechaphlowers.entities.shapes import SupportShape
from mechaphlowers.plotting.plot import (
    PlotEngine,
    TraceProfile,
    figure_factory,
    plot_points_2d,
    plot_support_shape,
)
from mechaphlowers.plotting.utils import compute_aspect_ratio
from test.conftest import show_figures
from test.tools.plot_tools import assert_cable_linked_to_attachment


@fixture
def balance_engine_local_initialized() -> BalanceEngine:
    data = {
        "name": ["1", "2", "three", "support 4"],
        "suspension": [False, True, True, False],
        "conductor_attachment_altitude": [50.0, 40.0, 20.0, 10.0],
        "crossarm_length": [
            5.0,
        ]
        * 4,
        "line_angle": [
            0,
        ]
        * 4,
        "insulator_length": [2, 4, 3.2, 2],
        "span_length": [100, 200, 300, np.nan],
        "insulator_mass": convert_weight_to_mass(
            np.array([1000.0, 500.0, 500.0, 1000.0])
        ),
        "load_mass": [0, 0, 0, 0],
        "load_position": [0, 0, 0, 0],
    }

    section_array = SectionArray(data=pd.DataFrame(data))
    section_array.sagging_parameter = 500
    section_array.sagging_temperature = 15
    cable_array_AM600: CableArray = sample_cable_catalog.get_as_object(
        ["ASTER600"]
    )  # type: ignore[assignment]

    balance_engine_local_test = BalanceEngine(
        cable_array=cable_array_AM600, section_array=section_array
    )
    balance_engine_local_test.solve_adjustment()
    balance_engine_local_test.solve_change_state()
    return balance_engine_local_test


def test_plot_line3d__all_line(
    balance_engine_local_initialized: BalanceEngine,
) -> None:
    fig = go.Figure()
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_local_initialized
    )
    plt_engine.preview_line3d(fig)

    if show_figures:
        fig.show()  # deactivate for auto unit testing
    assert (
        len(
            [
                f
                for f in fig.data
                if f.name == "Cable" and not np.isnan(f.x).all()  # type: ignore[attr-defined]
            ]
        )
        == 1
    )
    assert len(fig.data) == 3  # Just trying to see if the previous code raises
    span_points, _, insulators_points = plt_engine.get_points_for_plot()
    assert_cable_linked_to_attachment(span_points, insulators_points)


def test_plot_line3d__view_option(
    balance_engine_local_initialized: BalanceEngine,
) -> None:
    fig = go.Figure()
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_local_initialized
    )
    plt_engine.preview_line3d(fig, view="analysis")

    if show_figures:
        fig.show()
    assert (
        len(
            [
                f
                for f in fig.data
                if f.name == "Cable" and not np.isnan(f.x).all()  # type: ignore[attr-defined]
            ]
        )
        == 1
    )
    assert len(fig.data) == 3  # Just trying to see if the previous code raises


def test_plot_line3d__wrong_view_option(
    balance_engine_local_initialized: BalanceEngine,
) -> None:
    fig = go.Figure()
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_local_initialized
    )
    plt_engine.preview_line3d(fig)
    with pytest.raises(ValueError):
        plt_engine.preview_line3d(fig, view="wrong_parameter")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        plt_engine.preview_line3d(fig, view=22)  # type: ignore[arg-type]


def test_plot_line3d__with_beta(
    balance_engine_local_initialized: BalanceEngine,
):
    balance_engine_local_initialized.solve_change_state(
        ice_thickness=np.array([0.0, 0.6, 0.0, np.nan]),
        wind_pressure=np.array([240.12, 0.0, 600.0, np.nan]),
    )

    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_local_initialized
    )
    fig = go.Figure()
    plt_engine.preview_line3d(fig)

    if show_figures:
        fig.show()  # deactivate for auto unit testing
    assert (
        len(
            [
                f
                for f in fig.data
                if f.name == "Cable" and not np.isnan(f.x).all()  # type: ignore[attr-defined]
            ]
        )
        == 1
    )
    assert len(fig.data) == 3  # Just trying to see if the previous code raises

    span_points, _, insulators_points = plt_engine.get_points_for_plot()
    assert_cable_linked_to_attachment(span_points, insulators_points)


def test_plot_support_shape():
    fig = go.Figure()
    pyl_shape = SupportShape(
        name="pyl",
        xyz_arms=np.array(
            [
                [0, 0, 18.5],
                [0, 3, 16.5],
                [0, 6, 16.5],
                [0, 9, 16.5],
                [3, -3, 14.5],
                [-3, -3, 14.5],
                [0, -6, 14.5],
                [0, -9, 14.5],
            ]
        ),
        set_number=np.array([22, 28, 37, 44, 45, 46, 47, 55]),
    )
    plot_support_shape(fig, pyl_shape)

    if show_figures:
        fig.show()
    assert True


def test_reactive_plot(balance_engine_base_test: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_base_test
    )
    balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1, 1])
    )
    fig = go.Figure()
    plt_engine.preview_line3d(fig)
    balance_engine_base_test.solve_change_state(
        wind_pressure=1000 * np.array([1, 1, 1, np.nan]),
    )

    plt_engine.preview_line3d(fig)

    assert (
        len(
            [
                f
                for f in fig.data
                if f.name == "Cable" and not np.isnan(f.x).all()  # type: ignore[attr-defined]
            ]
        )
        == 2
    )

    if show_figures:
        fig.show()  # deactivate for auto unit testing

    span_points, _, insulators_points = plt_engine.get_points_for_plot()
    assert_cable_linked_to_attachment(span_points, insulators_points)


def test_plot_ice(balance_engine_base_test: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_base_test
    )
    balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(new_temperature=15)
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_base_test
    )
    fig = go.Figure()
    plt_engine.preview_line3d(fig)
    # balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(
        ice_thickness=np.array([1, 1, 1, np.nan]) * 1,
        # new_temperature=90 * np.array([1, 1, 1]),
    )

    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_base_test
    )

    plt_engine.preview_line3d(fig)

    if show_figures:
        fig.show()  # deactivate for auto unit testing
    assert (
        len(
            [
                f
                for f in fig.data
                if f.name == "Cable" and not np.isnan(f.x).all()  # type: ignore[attr-defined]
            ]
        )
        == 2
    )
    span_points, _, insulators_points = plt_engine.get_points_for_plot()
    assert_cable_linked_to_attachment(span_points, insulators_points)


def test_plot_2d(balance_engine_angles: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(balance_engine_angles)
    balance_engine_angles.solve_adjustment()
    balance_engine_angles.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1, 1])
    )

    # plt_engine.preview_line2d(fig)
    # balance_engine_base_test.solve_adjustment()
    balance_engine_angles.solve_change_state(
        wind_pressure=np.array([1, 1, 1, np.nan]) * 200,
        # new_temperature=90 * np.array([1, 1, 1]),
    )
    fig_line = go.Figure()
    plt_engine.preview_line2d(fig_line, "line", 1)

    fig_profile = go.Figure()
    plt_engine.preview_line2d(fig_profile, "profile", 1)

    assert fig_line.layout.yaxis.scaleanchor == "x"
    assert fig_profile.layout.yaxis.scaleanchor != "x"

    # fig_line.show()
    # fig_profile.show()  # deactivate for auto unit testing
    span_points, _, insulators_points = plt_engine.get_points_for_plot(
        project=True, frame_index=1
    )
    assert_cable_linked_to_attachment(span_points, insulators_points)


def test_plot_more_spans(cable_array_AM600: CableArray):
    section_array = SectionArray(pd.DataFrame(section_factory_sample_data(10)))
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600, section_array=section_array
    )
    plt_engine = PlotEngine.builder_from_balance_engine(balance_engine)
    balance_engine.solve_adjustment()
    balance_engine.solve_change_state(new_temperature=15)
    balance_engine.solve_change_state(
        wind_pressure=200,
    )
    fig_line = go.Figure()
    plt_engine.preview_line2d(fig_line, "line", 8)

    fig_profile = go.Figure()
    plt_engine.preview_line2d(fig_profile, "profile", 8)

    # fig_line.show()
    # fig_profile.show()  # deactivate for auto unit testing
    span_points, _, insulators_points = plt_engine.get_points_for_plot()
    assert_cable_linked_to_attachment(span_points, insulators_points)


def test_plot_3d_sandbox(balance_engine_angles: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(balance_engine_angles)
    balance_engine_angles.solve_adjustment()
    balance_engine_angles.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1, 1])
    )
    fig = go.Figure()
    # plt_engine.preview_line2d(fig)
    # balance_engine_base_test.solve_adjustment()
    balance_engine_angles.solve_change_state(
        wind_pressure=np.array([1, 1, 1, np.nan]) * 200,
        # new_temperature=90 * np.array([1, 1, 1]),
    )

    plt_engine.preview_line3d(fig)
    span_points, _, insulators_points = plt_engine.get_points_for_plot()
    assert_cable_linked_to_attachment(span_points, insulators_points)

    if show_figures:
        fig.show()  # deactivate for auto unit testing


def test_plot_3d__styles(balance_engine_angles: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(balance_engine_angles)
    balance_engine_angles.solve_adjustment()
    balance_engine_angles.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1, 1])
    )
    fig = figure_factory("std")
    plt_engine.preview_line3d(fig, mode="background")
    # plt_engine.preview_line2d(fig)
    # balance_engine_base_test.solve_adjustment()
    balance_engine_angles.solve_change_state(
        wind_pressure=500,
    )

    plt_engine.preview_line3d(fig)
    span_points, _, insulators_points = plt_engine.get_points_for_plot()
    assert_cable_linked_to_attachment(span_points, insulators_points)

    if show_figures:
        fig.show()  # deactivate for auto unit testing


def test_plot_point_2d_wrong_view():
    fig = go.Figure()
    points = np.array([[0, 0, 0], [1, 1, 1]])
    with pytest.raises(ValueError):
        plot_points_2d(fig, points, view="wrong_view")


def test_preview_2d_wrong_view(balance_engine_angles: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(balance_engine_angles)
    balance_engine_angles.solve_adjustment()
    balance_engine_angles.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1, 1])
    )
    fig_line = go.Figure()
    with pytest.raises(ValueError):
        plt_engine.preview_line2d(fig_line, view="wrong_view", frame_index=1)  # type: ignore[arg-type]


def test_trace_profile():
    tp = TraceProfile(name="test", color="red", width=2, opacity=0.5)

    with pytest.raises(ValueError):
        tp.mode = "other_string"

    with pytest.raises(ValueError):
        tp.mode = 0.0

    with pytest.raises(TypeError):
        tp.name = 0.0

    assert not tp.dashed
    assert tp("background").dashed == {'dash': 'dot'}

    # check tp keep the background state for mode
    assert tp.dashed == {'dash': 'dot'}

    # check the config reactivity : expected behavior no change
    options.graphics.cable_trace_profile["name"] = "test2"
    assert tp.name == "test baseline"


def test_plot_repr(balance_engine_base_test: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_base_test
    )
    plt_engine.__repr__()
    balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(
        new_temperature=30,
        wind_pressure=200,
        ice_thickness=2e-2,
    )
    repr_plot = plt_engine.__repr__()
    assert repr_plot.startswith("PlotEngine")


def test_compute_aspect_ratio__values_in_range(
    balance_engine_local_initialized: BalanceEngine,
) -> None:
    """Test that the maximum aspect ratio value equals the scale factor (default 1.0)."""
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_local_initialized
    )
    span, supports, insulators = plt_engine.get_points_for_plot()

    aspect = compute_aspect_ratio(span, supports, insulators)

    # Check that the returned dict has the expected keys
    assert set(aspect.keys()) == {"x", "y", "z"}

    # Check that all values are positive floats
    assert all(isinstance(v, float) and v > 0 for v in aspect.values())

    # Check that the maximum value equals 1.0 (before scaling)
    assert max(aspect.values()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "scale_axis,scale_value",
    [
        ("x_scale", 10.0),
        ("y_scale", 10.0),
        ("z_scale", 10.0),
    ],
)
def test_compute_aspect_ratio__scale_factors(
    balance_engine_local_initialized: BalanceEngine,
    scale_axis: str,
    scale_value: float,
) -> None:
    """Test that scale multipliers are applied correctly for all axes."""
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_local_initialized
    )
    span, supports, insulators = plt_engine.get_points_for_plot()

    # Get base aspect (no scaling)
    aspect_base = compute_aspect_ratio(span, supports, insulators)

    # Get scaled aspect
    kwargs = {scale_axis: scale_value}
    aspect_scaled = compute_aspect_ratio(span, supports, insulators, **kwargs)

    # Extract the axis name (x, y, or z)
    axis_letter = scale_axis[0]  # 'x', 'y', or 'z'

    # Check that the scaled axis is multiplied by the scale factor
    assert aspect_scaled[axis_letter] == pytest.approx(
        aspect_base[axis_letter] * scale_value, abs=1e-6
    )

    # For x and z axes with large scaling, check that they become maximum
    # (y axis is not considered because small y-range)
    if axis_letter in ("x", "z"):
        other_axes = [k for k in ("x", "y", "z") if k != axis_letter]
        assert aspect_scaled[axis_letter] >= max(
            aspect_scaled[ax] for ax in other_axes
        )


def test_compute_aspect_ratio__all_scales(
    balance_engine_local_initialized: BalanceEngine,
) -> None:
    """Test that all three scale factors are applied correctly."""
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_local_initialized
    )
    span, supports, insulators = plt_engine.get_points_for_plot()

    # Get base aspect ratio
    aspect_base = compute_aspect_ratio(span, supports, insulators)

    # Get scaled aspect ratio
    aspect_scaled = compute_aspect_ratio(
        span, supports, insulators, x_scale=2.0, y_scale=0.5, z_scale=10.0
    )

    # Check that scales are applied proportionally
    x_ratio = aspect_scaled["x"] / aspect_base["x"]
    y_ratio = aspect_scaled["y"] / aspect_base["y"]
    z_ratio = aspect_scaled["z"] / aspect_base["z"]

    assert x_ratio == pytest.approx(2.0, abs=1e-6)
    assert y_ratio == pytest.approx(0.5, abs=1e-6)
    assert z_ratio == pytest.approx(10.0, abs=1e-6)


def test_compute_aspect_ratio__error_on_no_points() -> None:
    """Test that ValueError is raised when no Points objects provided."""
    with pytest.raises(ValueError, match="At least one Points object"):
        compute_aspect_ratio()


def test_compute_aspect_ratio__error_on_invalid_scale(
    balance_engine_local_initialized: BalanceEngine,
) -> None:
    """Test that ValueError is raised for non-positive scale factors."""
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_local_initialized
    )
    span, supports, insulators = plt_engine.get_points_for_plot()

    with pytest.raises(ValueError, match="Scale factors must be positive"):
        compute_aspect_ratio(span, supports, insulators, z_scale=0.0)

    with pytest.raises(ValueError, match="Scale factors must be positive"):
        compute_aspect_ratio(span, supports, insulators, x_scale=-1.0)


def test_preview_line3d__custom_aspect_ratio(
    balance_engine_local_initialized: BalanceEngine,
) -> None:
    """Test that preview_line3d accepts and applies custom aspect ratio."""
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_local_initialized
    )

    # Compute custom aspect ratio
    span, supports, insulators = plt_engine.get_points_for_plot()
    custom_aspect = compute_aspect_ratio(
        span, supports, insulators, z_scale=10.0
    )

    # Create figure with custom aspect ratio
    fig = go.Figure()
    plt_engine.preview_line3d(fig, aspect_ratio=custom_aspect)

    # Check that the layout has the custom aspect ratio
    assert fig.layout.scene.aspectmode == "manual"
    assert fig.layout.scene.aspectratio["x"] == pytest.approx(
        custom_aspect["x"], abs=1e-6
    )
    assert fig.layout.scene.aspectratio["y"] == pytest.approx(
        custom_aspect["y"], abs=1e-6
    )
    assert fig.layout.scene.aspectratio["z"] == pytest.approx(
        custom_aspect["z"], abs=1e-6
    )
