# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.api.section_study import SectionStudy
from mechaphlowers.core.geometry.references import (
    cable_to_localsection_frame,
)
from mechaphlowers.entities.arrays import CableArray, SectionArray

LINE_ANGLES_GRAD = [0, 40, 50, 60, 100]


class TestCableToLocalsectionFrame:
    """The function must be a proper (direct) rotation of angle azimuth_angle."""

    @pytest.mark.parametrize("azimuth", [0.0, 0.3, np.pi / 4, 1.2, np.pi / 2])
    def test_is_a_rotation(self, azimuth: float):
        x = np.array([[1.0, 2.0], [3.0, -1.0]])
        y = np.array([[0.5, -2.0], [4.0, 1.0]])
        z = np.ones_like(x)
        x_r, y_r, z_r = cable_to_localsection_frame(
            x, y, z, np.array([azimuth, azimuth])
        )

        # norm in the horizontal plane is preserved
        np.testing.assert_allclose(np.hypot(x_r, y_r), np.hypot(x, y))
        # same orientation: cross product of (x, y) with (x_r, y_r) matches sin(azimuth)
        np.testing.assert_allclose(
            x * y_r - y * x_r, np.sin(azimuth) * (x**2 + y**2), atol=1e-12
        )
        np.testing.assert_array_equal(z_r, z)

    def test_quarter_turn(self):
        x, y, z = cable_to_localsection_frame(
            np.array([[1.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([np.pi / 2]),
        )
        np.testing.assert_allclose([x[0, 0], y[0, 0]], [0.0, 1.0], atol=1e-12)

        x, y, z = cable_to_localsection_frame(
            np.array([[0.0]]),
            np.array([[1.0]]),
            np.array([[0.0]]),
            np.array([np.pi / 2]),
        )
        np.testing.assert_allclose([x[0, 0], y[0, 0]], [-1.0, 0.0], atol=1e-12)

    def test_wind_component_not_mirrored_at_45_degrees(self):
        # a pure lateral (wind) vector must stay lateral relative to the span axis
        azimuth = np.array([np.pi / 4])
        x, y, _ = cable_to_localsection_frame(
            np.array([[0.0]]),
            np.array([[1.0]]),
            np.array([[0.0]]),
            azimuth,
        )
        np.testing.assert_allclose(
            [x[0, 0], y[0, 0]], [-np.sqrt(2) / 2, np.sqrt(2) / 2], atol=1e-12
        )


def build_study(
    cable_array: CableArray, line_angle_grad: float
) -> SectionStudy:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 0, 0, 0],
                "line_angle": [line_angle_grad, 0, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [1000, 500, 500, 1000],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    return SectionStudy(cable_array=cable_array, section_array=section_array)


def lateral_offsets(study: SectionStudy) -> tuple[np.ndarray, np.ndarray]:
    """Return, for each span, the lateral offset (left positive) of the cable
    middle point from the chord between its attachments, and the lateral
    displacement of the insulators at both ends of the span."""
    coords_calculator = study.position_engine.coords_calculator
    attachments = coords_calculator.get_attachments_coords()
    displacement = (
        coords_calculator.plane.displacement_vector.dxdydz_global_frame
    )
    x_span, y_span, _ = coords_calculator.span_in_section_frame()

    mid = x_span.shape[0] // 2
    cable_offsets = []
    insulator_offsets = []
    for i in range(x_span.shape[1]):
        chord = attachments[i + 1, :2] - attachments[i, :2]
        left_normal = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
        mid_point = np.array([x_span[mid, i], y_span[mid, i]])
        cable_offsets.append((mid_point - attachments[i, :2]) @ left_normal)
        insulator_offsets.append(
            (displacement[i, :2] + displacement[i + 1, :2]) @ left_normal
        )
    return np.array(cable_offsets), np.array(insulator_offsets)


class TestSectionWindAngle:
    @pytest.fixture
    def solved_studies(
        self, cable_array_AM600: CableArray
    ) -> dict[int, SectionStudy]:
        studies = {}
        for line_angle in LINE_ANGLES_GRAD:
            study = build_study(cable_array_AM600, line_angle)
            study.solve_adjustment()
            study.solve_change_state(wind_pressure=500)
            studies[line_angle] = study
        return studies

    def test_cable_offset_independent_of_line_angle(self, solved_studies):
        """Rotating the whole line must not change how the cable is blown
        relatively to its span."""
        reference, _ = lateral_offsets(solved_studies[0])
        for line_angle, study in solved_studies.items():
            offsets, _ = lateral_offsets(study)
            np.testing.assert_allclose(
                offsets,
                reference,
                atol=1e-6,
                err_msg=f"line_angle={line_angle} grad",
            )

    def test_cable_is_blown_at_50_grad(self, solved_studies):
        """At 45 degrees of azimuth the wind effect used to vanish."""
        offsets, _ = lateral_offsets(solved_studies[50])
        assert np.all(np.abs(offsets) > 1.0)

    @pytest.mark.parametrize("line_angle", LINE_ANGLES_GRAD)
    def test_cable_and_insulators_move_same_side(
        self, solved_studies, line_angle
    ):
        cable_offsets, insulator_offsets = lateral_offsets(
            solved_studies[line_angle]
        )
        np.testing.assert_array_equal(
            np.sign(cable_offsets), np.sign(insulator_offsets)
        )

    def test_no_wind_geometry_unchanged_by_line_angle(
        self, cable_array_AM600: CableArray
    ):
        """Without wind the cable stays in the vertical plane of the span."""
        for line_angle in LINE_ANGLES_GRAD:
            study = build_study(cable_array_AM600, line_angle)
            study.solve_adjustment()
            study.solve_change_state(wind_pressure=0)
            offsets, _ = lateral_offsets(study)
            np.testing.assert_allclose(offsets, 0.0, atol=1e-6)
