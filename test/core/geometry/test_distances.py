# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.geometry.distances import (
    DistanceEngine,
    ObstacleDistances,
    PointDistanceResult,
    SamplingOptions,
    VectorizedDistanceEngine,
    _build_segments_samples,
    batch_plane_distances,
    get_projection_points,
    scalar_points_distance_inside_plane,
)
from mechaphlowers.core.geometry.planes import plane_basis, plane_from_line
from mechaphlowers.entities.errors import NoIntersectionPlaneWarning


def _vectorized_engine(
    curves: np.ndarray,
    normals: np.ndarray,
    options: SamplingOptions | None = None,
) -> VectorizedDistanceEngine:
    """Engine whose span frames have the given horizontal unit normals."""
    ground = np.vstack([np.zeros((1, 3)), np.cumsum(normals, axis=0)])
    return VectorizedDistanceEngine(
        ground_supports=ground, curves=curves, options=options
    )


def test_points_distance_inside_plane_returns_distance_and_projections():
    point_2 = np.array([1.0, 0.0, 0.0])
    point_1 = np.array([0.0, 0.0, 0.0])
    # plane_normal = np.array([0.0, 0.0, 1.0])
    # line_direction_normalized = np.array([1.0, 0.0, 0.0])
    u_plane = np.array([1.0, 0.0, 0.0])
    v_plane = np.array([0.0, 1.0, 0.0])

    distance_3d, projection_u, projection_v = (
        scalar_points_distance_inside_plane(
            point_1,
            point_2,
            u_plane,
            v_plane,
        )
    )

    np.testing.assert_allclose(distance_3d, 1.0)
    np.testing.assert_allclose(projection_u, 1.0)
    np.testing.assert_allclose(projection_v, 0.0)


def test_points_distance_inside_plane_invalid_projection_raises():
    point_2 = np.array([1.0, 0.0, 0.0])
    point_1 = np.array([0.0, 0.0, 0.0])
    # plane_normal = np.array([0.0, 0.0, 1.0])
    # line_direction_normalized = np.array([1.0, 0.0, 0.0])
    u_plane = np.array([0.0, 1.0, 0.0])
    v_plane = np.array([0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="Projected distance"):
        scalar_points_distance_inside_plane(
            point_1,
            point_2,
            u_plane,
            v_plane,
        )


def test_get_projection_points():
    obstacle = np.array([10.0, 20.0, 30.0])
    projection_u = 2.0
    projection_v = -3.0
    u_plane = np.array([1.0, 0.0, 0.0])
    v_plane = np.array([0.0, 0.0, 1.0])

    u_point, v_point = get_projection_points(
        obstacle, projection_u, projection_v, u_plane, v_plane
    )

    np.testing.assert_allclose(u_point, np.array([12.0, 20.0, 30.0]))
    np.testing.assert_allclose(v_point, np.array([10.0, 20.0, 27.0]))


def test_distance_result_reverse_y_axis():
    distance_result = PointDistanceResult(
        np.array([100.0000, 6.7000, 0.0000]),
        np.array([100.0000, 9.9940, 49.9035]),
        np.array([-0.0000, 1.0000, 0.0000]),
        np.array(
            [0.0000, 0.1000, 1.0000]
        ),  # incorrect value, just for testing reversed y
        50.894348,
        9.993974,
        49.903458,
    )
    reversed_distance_result = distance_result.generate_with_reversed_y_axis()
    # Check y axis is correctly reversed
    np.testing.assert_allclose(
        reversed_distance_result.point_base,
        np.array([100.0000, -6.7000, 0.0000]),
    )
    np.testing.assert_allclose(
        reversed_distance_result.point_target,
        np.array([100.0000, -9.9940, 49.9035]),
    )
    np.testing.assert_allclose(
        reversed_distance_result.u_plane, np.array([-0.0000, -1.0000, 0.0000])
    )
    np.testing.assert_allclose(
        reversed_distance_result.v_plane, np.array([0.0000, -0.1000, 1.0000])
    )

    # Check y axis is NOT reversed on original object
    np.testing.assert_allclose(
        distance_result.point_base, np.array([100.0000, 6.7000, 0.0000])
    )
    np.testing.assert_allclose(
        distance_result.point_target, np.array([100.0000, 9.9940, 49.9035])
    )
    np.testing.assert_allclose(
        distance_result.u_plane, np.array([-0.0000, 1.0000, 0.0000])
    )
    np.testing.assert_allclose(
        distance_result.v_plane, np.array([0.0000, 0.1000, 1.0000])
    )


def test_plane_distances_matches_scalar_plane_distance():
    engine = DistanceEngine()
    engine.add_span_frame(
        np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0])
    )
    x = np.linspace(0, 100, 30)
    curve = np.column_stack([x, np.zeros_like(x), 50 - 0.002 * (x - 50) ** 2])
    engine.add_curves(curve)
    vectorized = _vectorized_engine(
        curve[np.newaxis], np.array([[1.0, 0.0, 0.0]])
    )
    points = np.array(
        [
            [30.0, 2.0, 10.0],
            [60.0, -3.0, 20.0],
            [80.0, 1.0, 5.0],
            [10.0, 0.5, 30.0],
            [150.0, 0.0, 0.0],
        ]
    )

    distances, targets = vectorized.plane_distances(points)

    assert distances.shape == (5,)
    assert targets.shape == (5, 3)
    for i in range(4):
        expected = engine.scalar_plane_distance(points[i], frame="section")
        np.testing.assert_allclose(distances[i], expected.distance_3d)
        np.testing.assert_allclose(targets[i], expected.point_target)
    assert np.isinf(distances[4])
    assert np.all(np.isnan(targets[4]))


def test_plane_distances_accepts_single_point_and_keeps_state():
    engine = DistanceEngine()
    engine.add_span_frame(
        np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0])
    )
    engine.add_curves(
        np.column_stack(
            [np.linspace(0, 100, 10), np.zeros(10), np.linspace(20, 10, 10)]
        )
    )
    engine.scalar_plane_distance(np.array([40.0, 0.0, 5.0]), frame="section")
    point_base_before = engine.point_base.copy()
    vectorized = _vectorized_engine(
        np.column_stack(
            [np.linspace(0, 100, 10), np.zeros(10), np.linspace(20, 10, 10)]
        )[np.newaxis],
        np.array([[1.0, 0.0, 0.0]]),
    )

    distances, targets = vectorized.plane_distances(np.array([40.0, 1.0, 5.0]))

    assert distances.shape == (1,)
    assert targets.shape == (1, 3)
    np.testing.assert_array_equal(engine.point_base, point_base_before)


def _two_spans_curves_and_normals():
    x = np.linspace(0.0, 100.0, 11)
    curve_0 = np.column_stack([x, 0.2 * x, -0.01 * x * (100.0 - x)])
    curve_1 = np.column_stack(
        [x, -0.5 * x + 10.0, -0.02 * x * (100.0 - x) + 3]
    )
    curves = np.stack([curve_0, curve_1])
    normals = np.array([[0.0, 1.0, 0.0], [0.6, 0.8, 0.0]])
    return curves, normals


def test_batch_plane_distances_matches_plane_distance_per_point():
    curves, normals = _two_spans_curves_and_normals()
    points = np.array(
        [
            [30.0, 5.0, -10.0],
            [40.0, -8.0, -6.0],
            [20.0, -2.0, -3.0],
            [40.0, 12.0, -8.0],
            1000.0 * normals[1],  # no intersection
        ]
    )
    span_indices = np.array([0, 1, 1, 0, 1])

    distances, targets = batch_plane_distances(
        points, span_indices, curves, normals
    )

    for i, (point, span) in enumerate(zip(points, span_indices)):
        engine = DistanceEngine()
        engine.add_span_frame(np.array([0.0, 0.0, 0.0]), normals[span] * 100.0)
        engine.add_curves(curves[span])
        if i == 4:
            assert np.isinf(distances[i])
            assert np.all(np.isnan(targets[i]))
            continue
        expected = engine.scalar_plane_distance(point, frame="section")
        np.testing.assert_allclose(
            distances[i], expected.distance_3d, atol=1e-9
        )
        np.testing.assert_allclose(
            targets[i], expected.point_target, atol=1e-9
        )


def test_plane_basis_matches_plane_from_line():
    normals = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],  # vertical
        ]
    )
    u, v = plane_basis(normals)

    assert u.shape == (4, 3)
    assert v.shape == (4, 3)
    for i, normal in enumerate(normals):
        u_expected, v_expected, _ = plane_from_line(np.zeros(3), normal)
        np.testing.assert_allclose(u[i], u_expected, atol=1e-12)
        np.testing.assert_allclose(v[i], v_expected, atol=1e-12)


class TestBuildSegmentsSamples:
    def test_builds_polyline_segments_and_single_point_samples(self):
        coords_dict = {
            "point": np.array([[10, 0, 0]], dtype=float),
            "line": np.array(
                [[0, 0, 0], [2.5, 0, 0], [2.5, 4, 0]], dtype=float
            ),
        }

        result = _build_segments_samples(
            coords_dict=coords_dict,
            spans_dict={"line": 3, "point": 5},
            step=1.5,
        )

        assert result.spans == [5, 3]
        np.testing.assert_array_equal(
            result.seg_start,
            [[10, 0, 0], [0, 0, 0], [2.5, 0, 0]],
        )
        np.testing.assert_array_equal(
            result.seg_vec, [[0, 0, 0], [2.5, 0, 0], [0, 4, 0]]
        )
        np.testing.assert_array_equal(result.seg_loc, [0, 0, 1])
        np.testing.assert_array_equal(result.seg_sub, [1, 2, 3])
        np.testing.assert_array_equal(result.seg_span, [5, 3, 3])
        np.testing.assert_allclose(result.seg_len, [0, 2.5, 4])
        np.testing.assert_array_equal(result.sample_seg, [0, 1, 1, 2, 2, 2, 2])
        np.testing.assert_allclose(
            result.sample_t, [0, 0, 0.5, 0, 1 / 3, 2 / 3, 1]
        )
        np.testing.assert_array_equal(result.sample_obs, [0, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(
            result.sample_vertex, [0, 0, -1, 1, -1, -1, 2]
        )

    def test_points_geometry_yields_one_zero_length_segment_per_vertex(self):
        result = _build_segments_samples(
            coords_dict={
                "pts": np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0]], dtype=float)
            },
            spans_dict={"pts": 0},
            step=1.0,
            geometry_types={"pts": "points"},
        )

        np.testing.assert_array_equal(result.seg_vec, np.zeros((3, 3)))
        np.testing.assert_array_equal(result.seg_loc, [0, 1, 2])
        np.testing.assert_array_equal(result.sample_t, [0, 0, 0])
        np.testing.assert_array_equal(result.sample_vertex, [0, 1, 2])


class TestMinDistances:
    def test_empty_and_invalid_sampling(self):
        curves = np.zeros((1, 3, 3))
        normals = np.array([[1.0, 0.0, 0.0]])
        engine = _vectorized_engine(curves, normals)
        assert (
            engine.min_distances({}, {}, step=1.0, refine=True, xtol=1e-3)
            == {}
        )
        with pytest.raises(ValueError):
            engine.min_distances({}, {}, step=0.0, refine=True, xtol=1e-3)
        with pytest.raises(ValueError):
            engine.min_distances({}, {}, step=1.0, refine=True, xtol=-1.0)


class TestDistances:
    # straight cable along x at z=10: the plane of a point crosses it at the
    # point x, so the distance is the (y, z) offset to (0, 10)
    curves = np.column_stack(
        [np.linspace(0, 100, 101), np.zeros(101), np.full(101, 10.0)]
    )[np.newaxis]
    normals = np.array([[1.0, 0.0, 0.0]])

    def test_vertices_and_minimum_per_geometry_type(self):
        coords = np.array([[10, 0, 0], [20, 0, 5], [30, 0, 8]], dtype=float)
        line = np.array([[40, 5, 10], [60, -5, 10]], dtype=float)

        result = _vectorized_engine(self.curves, self.normals).distances(
            {"pts": coords, "line": line},
            {"pts": 0, "line": 0},
            geometry_types={"pts": "points", "line": "polyline"},
            step=1.0,
            refine=True,
            xtol=1e-6,
        )

        pts = result["pts"]
        assert pts.geometry_type == "points"
        assert {i: r.distance_3d for i, r in pts.points.items()} == (
            pytest.approx({0: 10.0, 1: 5.0, 2: 2.0})
        )
        np.testing.assert_allclose(pts.points[2].point_target, [30, 0, 10])
        assert pts.minimum is not None
        assert pts.minimum.segment_index == 2
        assert pts.minimum.segment_parameter == 0.0
        assert pts.minimum.distance_3d == pytest.approx(2.0)

        polyline = result["line"]
        assert polyline.geometry_type == "polyline"
        assert {i: r.distance_3d for i, r in polyline.points.items()} == (
            pytest.approx({0: 5.0, 1: 5.0})
        )
        assert polyline.minimum is not None
        assert polyline.minimum.segment_parameter == pytest.approx(0.5)
        assert polyline.minimum.distance_3d == pytest.approx(0.0, abs=1e-5)

    def test_vertex_without_intersection_is_absent_and_warns(self):
        coords = np.array([[50, 0, 0], [150, 0, 0]], dtype=float)

        with pytest.warns(NoIntersectionPlaneWarning, match="pts"):
            result = _vectorized_engine(self.curves, self.normals).distances(
                {"pts": coords},
                {"pts": 0},
                geometry_types={"pts": "points"},
                step=1.0,
                refine=True,
                xtol=1e-6,
            )

        assert set(result["pts"].points) == {0}
        assert result["pts"].minimum is not None
        assert result["pts"].minimum.segment_index == 0


class TestVectorizedDistanceEngine:
    @staticmethod
    def _obstacles():
        coords = {
            "a": np.array([[30, 5, -10], [40, 12, -8]], dtype=float),
            "b": np.array([[20, -2, -3], [25, -1, -2]], dtype=float),
        }
        return coords, {"a": 0, "b": 1}

    @staticmethod
    def _min_values(results):
        return {
            name: None if res is None else res.distance_3d
            for name, res in results.items()
        }

    def test_update_spans_equals_set_curves_with_full_array(self):
        curves, normals = _two_spans_curves_and_normals()
        new_curves = curves.copy()
        new_curves[1] += np.array([0.0, 0.0, 2.0])
        coords, spans = self._obstacles()

        full = _vectorized_engine(new_curves, normals)
        partial = _vectorized_engine(curves, normals)
        partial.update_spans([1], new_curves[[1]])

        np.testing.assert_array_equal(partial.curves, full.curves)
        np.testing.assert_array_equal(partial.curves_proj, full.curves_proj)
        points = np.array([[30.0, 5.0, -10.0], [20.0, -2.0, -3.0]])
        span_ids = np.array([0, 1])
        d_full, t_full = full.plane_distances(points, span_ids)
        d_part, t_part = partial.plane_distances(points, span_ids)
        np.testing.assert_array_equal(d_part, d_full)
        np.testing.assert_array_equal(t_part, t_full)
        assert self._min_values(
            partial.min_distances(coords, spans)
        ) == self._min_values(full.min_distances(coords, spans))

    def test_set_frames_after_curves_recomputes_projections(self):
        curves, normals = _two_spans_curves_and_normals()
        engine = VectorizedDistanceEngine()
        engine.set_curves(curves)
        assert engine.curves_proj is None
        engine.set_frames(
            np.vstack([np.zeros((1, 3)), np.cumsum(normals, axis=0)])
        )
        np.testing.assert_array_equal(
            engine.curves_proj, np.einsum("smk,sk->sm", curves, engine.normals)
        )

    def test_options_are_defaults_overridden_per_call(self):
        curves, normals = _two_spans_curves_and_normals()
        coords, spans = self._obstacles()
        coarse = _vectorized_engine(
            curves, normals, SamplingOptions(step=5.0, refine=False)
        )
        default = _vectorized_engine(curves, normals)

        assert self._min_values(coarse.min_distances(coords, spans)) == (
            self._min_values(
                default.min_distances(
                    coords, spans, step=5.0, refine=False, xtol=1e-3
                )
            )
        )
        assert self._min_values(
            coarse.min_distances(coords, spans, step=1.0, refine=True)
        ) == self._min_values(default.min_distances(coords, spans))

        with pytest.raises(ValueError):
            SamplingOptions(step=0.0)
        with pytest.raises(ValueError):
            default.min_distances(coords, spans, xtol=-1.0)

    def test_distances_before_frames_and_curves_are_set_raise(self):
        curves, normals = _two_spans_curves_and_normals()
        coords, spans = self._obstacles()
        engine = VectorizedDistanceEngine()

        with pytest.raises(RuntimeError):
            engine.distances(coords, spans)
        with pytest.raises(RuntimeError):
            engine.plane_distances(np.zeros(3))
        engine.set_curves(curves)
        with pytest.raises(RuntimeError):
            engine.min_distances(coords, spans)
        with pytest.raises(RuntimeError):
            engine.update_spans([0], curves[[0]])

    def test_update_spans_validates_ids_and_shape(self):
        curves, normals = _two_spans_curves_and_normals()
        engine = _vectorized_engine(curves, normals)

        with pytest.raises(IndexError):
            engine.update_spans([2], curves[[0]])
        with pytest.raises(ValueError):
            engine.update_spans([0], curves)

    def test_set_spans_allows_changing_the_number_of_spans(self):
        engine = VectorizedDistanceEngine(
            np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0]], dtype=float),
            np.zeros((2, 5, 3)),
        )
        engine.set_spans(
            np.array(
                [[0, 0, 0], [100, 0, 0], [200, 0, 0], [300, 0, 0]], float
            ),
            np.zeros((3, 5, 3)),
        )
        assert engine.normals.shape == (3, 3)
        assert engine.curves_proj.shape == (3, 5)

    @staticmethod
    def _obstacle(points: dict[int, PointDistanceResult]) -> ObstacleDistances:
        return ObstacleDistances(
            obstacle_name="pylone",
            span_index=0,
            geometry_type="points",
            points=points,
            minimum=None,
        )

    @staticmethod
    def _result(base, target, distance_3d) -> PointDistanceResult:
        return PointDistanceResult(
            point_base=np.array(base, dtype=float),
            point_target=np.array(target, dtype=float),
            u_plane=np.array([1.0, 0.0, 0.0]),
            v_plane=np.array([0.0, 1.0, 0.0]),
            distance_3d=distance_3d,
            distance_projection_u=0.0,
            distance_projection_v=distance_3d,
        )

    def test_get_table_has_one_row_per_point(self):
        obstacle = self._obstacle(
            {
                0: self._result([0, 1, 2], [3, 4, 5], 5.0),
                3: self._result([1, 1, 2], [6, 4, 5], 2.0),
            }
        )

        table = obstacle.get_table()

        assert table.shape == (2,)
        np.testing.assert_array_equal(table["point_index"], [0, 3])
        np.testing.assert_allclose(table["distance_3d"], [5.0, 2.0])
        np.testing.assert_allclose(table["point_base_x"], [0.0, 1.0])
        np.testing.assert_allclose(table["point_base_y"], [1.0, 1.0])
        np.testing.assert_allclose(table["point_base_z"], [2.0, 2.0])
        np.testing.assert_allclose(table["point_target_x"], [3.0, 6.0])
        np.testing.assert_allclose(table["point_target_y"], [4.0, 4.0])
        np.testing.assert_allclose(table["point_target_z"], [5.0, 5.0])

    def test_get_table_without_points_is_empty(self):
        table = self._obstacle({}).get_table()

        assert table.shape == (0,)
        assert "distance_3d" in table.dtype.names

    def test_get_table_converts_to_dataframe(self):
        obstacle = self._obstacle(
            {
                0: self._result([0, 1, 2], [3, 4, 5], 5.0),
                3: self._result([1, 1, 2], [6, 4, 5], 2.0),
            }
        )

        df = pd.DataFrame(obstacle.get_table())

        assert list(df.columns) == [
            "point_index",
            "distance_3d",
            "point_base_x",
            "point_base_y",
            "point_base_z",
            "point_target_x",
            "point_target_y",
            "point_target_z",
        ]
        assert df["point_index"].tolist() == [0, 3]
        assert df["distance_3d"].tolist() == pytest.approx([5.0, 2.0])
        assert df.loc[1, "point_target_x"] == pytest.approx(6.0)
