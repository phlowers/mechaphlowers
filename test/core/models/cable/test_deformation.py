# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.core.models.cable.deformation import DeformationRte
from mechaphlowers.core.models.cable.span import (
    CatenarySpan,
)
from mechaphlowers.entities.data_container import DataContainer


@pytest.fixture
def cable_array_input_data() -> dict[str, list]:
    return {
        "section": [345.5, 345.5],
        "diameter": [22.4, 22.4],
        "linear_weight": [9.6, 9.6],
        "young_modulus": [59, 59],
        "dilatation_coefficient": [23, 23],
        "temperature_reference": [15, 15],
        "a0": [0] * 2,
        "a1": [59] * 2,
        "a2": [0] * 2,
        "a3": [0] * 2,
        "a4": [0] * 2,
    }


def test_deformation_impl(
    default_data_container_one_span: DataContainer,
) -> None:
    span_model = CatenarySpan(**default_data_container_one_span.__dict__)
    tension_mean = span_model.T_mean()
    cable_length = span_model.L()

    deformation_model = DeformationRte(
        **default_data_container_one_span.__dict__,
        tension_mean=tension_mean,
        cable_length=cable_length,
    )
    deformation_model.epsilon_mecha()
    deformation_model.epsilon_therm()
    deformation_model.epsilon()
    deformation_model.L_ref()


def test_deformation_values__default_data(
    default_data_container_one_span: DataContainer,
) -> None:
    default_data_container_one_span.sagging_temperature = np.array([30, 30])
    span_model = CatenarySpan(**default_data_container_one_span.__dict__)
    tension_mean = span_model.T_mean()
    cable_length = span_model.L()

    deformation_model = DeformationRte(
        **default_data_container_one_span.__dict__,
        tension_mean=tension_mean,
        cable_length=cable_length,
    )

    eps_mecha = deformation_model.epsilon_mecha()
    eps_therm = deformation_model.epsilon_therm()
    L_ref = deformation_model.L_ref()

    # Data given by the prototype
    np.testing.assert_allclose(
        eps_mecha,
        np.array([0.00093978, np.nan]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        eps_therm,
        np.array([0.000345, 0.000345]),
        atol=1e-6,
    )
    # our method L_ref returns L_15 but proto returns L_0 so that's why 480.6392123 is not the displayed value if you are using proto
    np.testing.assert_allclose(
        L_ref,
        np.array([480.6392123, np.nan]),
        atol=1e-6,
    )


def test_poly_deformation__degree_three(
    default_data_container_one_span: DataContainer,
) -> None:
    new_poly = Poly([0, 1e9 * 50, 1e9 * -3_000, 1e9 * 44_000, 0])
    default_data_container_one_span.polynomial_conductor = new_poly

    span_model = CatenarySpan(**default_data_container_one_span.__dict__)
    tension_mean = span_model.T_mean()
    cable_length = span_model.L()

    deformation_model = DeformationRte(
        **default_data_container_one_span.__dict__,
        tension_mean=tension_mean,
        cable_length=cable_length,
    )

    constraint = (
        tension_mean / default_data_container_one_span.cable_section_area
    )
    constraint = np.fmax(constraint, np.array([0, 0]))
    deformation_model.resolve_stress_strain_equation(
        constraint,
        default_data_container_one_span.polynomial_conductor,
    )
    deformation_model.epsilon_mecha()

    deformation_model.epsilon()


def test_poly_deformation__degree_four(
    default_data_container_one_span: DataContainer,
) -> None:
    new_poly = Poly(
        [0, 1e9 * 100, 1e9 * -24_000, 1e9 * 2_440_000, 1e9 * -90_000_000]
    )
    default_data_container_one_span.polynomial_conductor = new_poly

    span_model = CatenarySpan(**default_data_container_one_span.__dict__)
    tension_mean = span_model.T_mean()
    cable_length = span_model.L()

    deformation_model = DeformationRte(
        **default_data_container_one_span.__dict__,
        tension_mean=tension_mean,
        cable_length=cable_length,
    )

    constraint = (
        tension_mean / default_data_container_one_span.cable_section_area
    )
    constraint = np.fmax(constraint, np.array([0, 0]))
    deformation_model.resolve_stress_strain_equation(
        constraint,
        default_data_container_one_span.polynomial_conductor,
    )
    deformation_model.epsilon_mecha()

    deformation_model.epsilon()


def test_poly_deformation__degree_four__with_max_stress(
    default_data_container_one_span: DataContainer,
) -> None:
    new_poly = Poly(
        [0, 1e9 * 100, 1e9 * -24_000, 1e9 * 2_440_000, 1e9 * -90_000_000]
    )
    default_data_container_one_span.polynomial_conductor = new_poly

    span_model = CatenarySpan(**default_data_container_one_span.__dict__)
    tension_mean = span_model.T_mean()
    cable_length = span_model.L()

    deformation_model = DeformationRte(
        **default_data_container_one_span.__dict__,
        tension_mean=tension_mean,
        cable_length=cable_length,
    )

    constraint = (
        tension_mean / default_data_container_one_span.cable_section_area
    )
    constraint = np.fmax(constraint, np.array([0, 0]))
    deformation_model.max_stress = np.array([1000, 1e8])
    deformation_model.epsilon_mecha()
    deformation_model.epsilon()


def test_poly_deformation__no_solutions(
    default_data_container_one_span: DataContainer,
) -> None:
    new_poly = Poly(
        [0, 1e9 * 100, 1e9 * -24_000, 1e9 * 2_440_000, 1e9 * -90_000_000]
    )
    default_data_container_one_span.polynomial_conductor = new_poly

    span_model = CatenarySpan(**default_data_container_one_span.__dict__)
    tension_mean = span_model.T_mean()
    cable_length = span_model.L()

    deformation_model = DeformationRte(
        **default_data_container_one_span.__dict__,
        tension_mean=tension_mean,
        cable_length=cable_length,
    )

    deformation_model.max_stress = np.array([1000, 1e10])
    with pytest.raises(ValueError):
        deformation_model.epsilon_mecha()


def test_deformation__data_container(
    default_data_container_one_span: DataContainer,
) -> None:
    span_model = CatenarySpan(**default_data_container_one_span.__dict__)
    tension_mean = span_model.T_mean()
    cable_length = span_model.L()

    deformation_model = DeformationRte(
        **default_data_container_one_span.__dict__,
        tension_mean=tension_mean,
        cable_length=cable_length,
    )
    deformation_model.epsilon_mecha()
    deformation_model.epsilon_therm()
    deformation_model.epsilon()
    deformation_model.L_ref()
