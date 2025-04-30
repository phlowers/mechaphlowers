# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.models.cable.deformation import DeformationRte
from mechaphlowers.core.models.cable.span import (
    CatenarySpan,
)
from mechaphlowers.entities.data_container import DataContainer


def test_deformation_impl(
    default_data_container_one_span: DataContainer,
) -> None:
    span_model = CatenarySpan(**default_data_container_one_span.__dict__)
    tension_mean = span_model.T_mean()
    cable_length = span_model.L()

    deformation_model = DeformationRte(
        **default_data_container_one_span.__dict__,
        data_cable=default_data_container_one_span.data_cable,
        tension_mean=tension_mean,
        cable_length=cable_length,
    )
    deformation_model.epsilon()
    deformation_model.L_ref()


# Those tests are irrelevent now: polynomial cable with 1 material does not exists

# def test_poly_deformation__degree_three(
#     default_data_container_one_span: DataContainer,
# ) -> None:
#     new_poly = Poly([0, 1e9 * 50, 1e9 * -3_000, 1e9 * 44_000, 0])
#     default_data_container_one_span.polynomial_conductor = new_poly

#     span_model = CatenarySpan(**default_data_container_one_span.__dict__)
#     tension_mean = span_model.T_mean()
#     cable_length = span_model.L()

#     deformation_model = DeformationRte(
#         **default_data_container_one_span.__dict__,
#         data_cable=default_data_container_one_span.data_cable,
#         tension_mean=tension_mean,
#         cable_length=cable_length,
#     )

#     constraint = (
#         tension_mean / default_data_container_one_span.cable_section_area
#     )
#     constraint = np.fmax(constraint, np.array([0, 0]))
#     # deformation_model.resolve_stress_strain_equation(
#     #     constraint,
#     #     default_data_container_one_span.polynomial_conductor,
#     # )

#     deformation_model.epsilon()


# def test_poly_deformation__degree_four(
#     default_data_container_one_span: DataContainer,
# ) -> None:
#     new_poly = Poly(
#         [0, 1e9 * 100, 1e9 * -24_000, 1e9 * 2_440_000, 1e9 * -90_000_000]
#     )
#     default_data_container_one_span.polynomial_conductor = new_poly

#     span_model = CatenarySpan(**default_data_container_one_span.__dict__)
#     tension_mean = span_model.T_mean()
#     cable_length = span_model.L()

#     deformation_model = DeformationRte(
#         **default_data_container_one_span.__dict__,
#         data_cable=default_data_container_one_span.data_cable,
#         tension_mean=tension_mean,
#         cable_length=cable_length,
#     )

#     constraint = (
#         tension_mean / default_data_container_one_span.cable_section_area
#     )
#     constraint = np.fmax(constraint, np.array([0, 0]))
#     # deformation_model.resolve_stress_strain_equation(
#     #     constraint,
#     #     default_data_container_one_span.polynomial_conductor,
#     # )

#     deformation_model.epsilon()


# def test_poly_deformation__degree_four__with_max_stress(
#     default_data_container_one_span: DataContainer,
# ) -> None:
#     new_poly = Poly(
#         [0, 1e9 * 100, 1e9 * -24_000, 1e9 * 2_440_000, 1e9 * -90_000_000]
#     )
#     default_data_container_one_span.polynomial_conductor = new_poly

#     span_model = CatenarySpan(**default_data_container_one_span.__dict__)
#     tension_mean = span_model.T_mean()
#     cable_length = span_model.L()

#     deformation_model = DeformationRte(
#         **default_data_container_one_span.__dict__,
#         data_cable=default_data_container_one_span.data_cable,
#         tension_mean=tension_mean,
#         cable_length=cable_length,
#     )

#     deformation_model.max_stress = np.array([1000, 1e7])
#     deformation_model.epsilon()


# def test_poly_deformation__no_solutions(
#     default_data_container_one_span: DataContainer,
# ) -> None:
#     new_poly = Poly(
#         [0, 1e9 * 100, 1e9 * -24_000, 1e9 * 2_440_000, 1e9 * -90_000_000]
#     )
#     default_data_container_one_span.polynomial_conductor = new_poly

#     span_model = CatenarySpan(**default_data_container_one_span.__dict__)
#     tension_mean = span_model.T_mean()
#     cable_length = span_model.L()

#     deformation_model = DeformationRte(
#         **default_data_container_one_span.__dict__,
#         data_cable=default_data_container_one_span.data_cable,
#         tension_mean=tension_mean,
#         cable_length=cable_length,
#     )

#     with pytest.raises(ValueError):
#         deformation_model.max_stress = np.array([1e10, 1e10])


def test_deformation__linear_2_materials(
    data_container_one_span_crocus: DataContainer,
) -> None:
    span_model = CatenarySpan(**data_container_one_span_crocus.__dict__)
    tension_mean = span_model.T_mean()
    cable_length = span_model.L()

    deformation_model = DeformationRte(
        **data_container_one_span_crocus.__dict__,
        data_cable=data_container_one_span_crocus.data_cable,
        tension_mean=tension_mean,
        cable_length=cable_length,
    )
    deformation_model.epsilon()
    deformation_model.L_ref()


def test_deformation__polynomial_2_materials(
    data_container_one_span_narcisse: DataContainer,
) -> None:
    span_model = CatenarySpan(**data_container_one_span_narcisse.__dict__)
    tension_mean = span_model.T_mean()
    cable_length = span_model.L()

    deformation_model = DeformationRte(
        **data_container_one_span_narcisse.__dict__,
        data_cable=data_container_one_span_narcisse.data_cable,
        tension_mean=tension_mean,
        cable_length=cable_length,
    )
    deformation_model.epsilon()
    deformation_model.L_ref()


def test_deformation__polynomial_2_materials_with_max_stress(
    data_container_one_span_narcisse: DataContainer,
) -> None:
    span_model = CatenarySpan(**data_container_one_span_narcisse.__dict__)
    tension_mean = span_model.T_mean()
    cable_length = span_model.L()

    deformation_model = DeformationRte(
        **data_container_one_span_narcisse.__dict__,
        data_cable=data_container_one_span_narcisse.data_cable,
        tension_mean=tension_mean,
        cable_length=cable_length,
    )
    deformation_model.max_stress = np.array([1e7, 1e7])
    deformation_model.epsilon()
    deformation_model.L_ref()
