# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from pytest import fixture

import mechaphlowers.core.models.balance.functions as f
from mechaphlowers.core.models.balance.elements import Cable, Nodes, Span


@fixture
def cable_AM600():
    return Cable(600.4e-6, 17.658, 0.000023, 60e9, 31.86e-3, 320)


@fixture
def section_3d_simple(cable_AM600) -> Span:
    nodes = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 0, 0, 0]),
        line_angle=f.grad_to_rad(np.array([0, 0, 0, 0])),
        span_length=np.array([500, 300, 400]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([0, 0, 0]),
        load_position=np.array([0, 0, 0]),
    )

    return Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes,
        cable=cable_AM600,
    )


@fixture
def section_3d_no_altitude_change(cable_AM600) -> Span:
    nodes = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 0, 0, 0]),
        line_angle=f.grad_to_rad(np.array([0, 0, 0, 0])),
        span_length=np.array([500, 300, 400]),
        z=np.array([50, 50, 50, 50]),
        load=np.array([0, 0, 0, 0]),
        load_position=np.array([0, 0, 0]),
    )

    return Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes,
        cable=cable_AM600,
    )


@fixture
def section_3d_angles_arm(cable_AM600) -> Span:
    nodes = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, -10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 10, 0, 0])),
        span_length=np.array([500, 300, 400]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([0, 0, 0]),
        load_position=np.array([0, 0, 0]),
    )

    return Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes,
        cable=cable_AM600,
    )


def test_element_sandbox(cable_AM600: Cable):
    nodes_arm = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, -10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 10, 0, 0])),
        span_length=np.array([500, 300, 400]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([0, 0, 0, 0]),
        load_position=np.array([0, 0.5, 0, 0]),
    )

    section = Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes_arm,
        cable=cable_AM600,
    )
    section.solve_adjustment()
    section.L_ref

    # section.sagging_temperature = 30
    # section.cable_loads.ice_thickness = np.array([1,1,1,1]) * 1e-2
    section.cable_loads.wind_pressure = np.array([200] * 4)
    section.solve_change_state()
    assert True


def test_element_sandbox_load(cable_AM600: Cable):
    nodes_arm = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, -10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 10, 0, 0])),
        span_length=np.array([500, 300, 400]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([0, 1000, 0]),
        load_position=np.array([0.5, 0.5, 0.5]),
    )

    section_3d_angles_arm = Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes_arm,
        cable=cable_AM600,
    )

    section_3d_angles_arm.solve_adjustment()
    section_3d_angles_arm.L_ref

    section_3d_angles_arm.solve_change_state()

    assert True


def test_adjust_no_altitude_change(section_3d_no_altitude_change: Span):
    section_3d_no_altitude_change.solve_adjustment()

    expected_dx = np.array(
        [
            2.97187340066645,
            2.64189410427871e-11,
            1.79935767328916e-11,
            -2.981118867567,
        ]
    )
    expected_dy = np.array([0, 0, 0, 0])
    expected_dz = np.array([-0.4098395910734, -3.0, -3.0, -0.336051033975601])

    expected_L_ref = np.array(
        [497.647471727149, 299.883603547399, 397.144221942162]
    )

    np.testing.assert_allclose(
        section_3d_no_altitude_change.nodes.dx, expected_dx, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_no_altitude_change.nodes.dy, expected_dy, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_no_altitude_change.nodes.dz, expected_dz, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_no_altitude_change.L_ref, expected_L_ref, atol=1e-4
    )


def test_adjust_simple(section_3d_simple: Span):
    section_3d_simple.solve_adjustment()
    expected_dx = np.array(
        [
            2.98575572319031,
            8.47916734674841e-13,
            -2.53273435063311e-11,
            -2.97673233606413,
        ]
    )
    expected_dy = np.array([0, 0, 0, 0])
    expected_dz = np.array(
        [-0.291997879164052, -3.0, -3.0, -0.372913662166194]
    )
    expected_L_ref = np.array(
        [498.045379200674, 300.049683827778, 397.175338304954]
    )

    np.testing.assert_allclose(
        section_3d_simple.nodes.dx, expected_dx, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_simple.nodes.dy, expected_dy, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_simple.nodes.dz, expected_dz, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_simple.L_ref, expected_L_ref, atol=1e-4
    )


def test_adjust_with_arm(cable_AM600: Cable):
    nodes = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, 10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 0, 0, 0])),
        span_length=np.array([500, 300, 400]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([0, 0, 0, 0]),
        load_position=np.array([0, 0, 0]),
    )

    section_arm = Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes,
        cable=cable_AM600,
    )
    section_arm.solve_adjustment()
    expected_dx = np.array(
        [
            2.98518900184611,
            0.00263326146642976,
            -0.0042602288783464,
            -2.97587548324835,
        ]
    )
    expected_dy = np.array(
        [
            0.0580656840614192,
            -0.274319342968404,
            -0.349583090259238,
            0.0717953330506774,
        ]
    )
    expected_dz = np.array(
        [
            -0.292018834309542,
            -2.98743066262758,
            -2.97955928174868,
            -0.372841170411165,
        ]
    )

    expected_L_ref = np.array(
        [498.143645314619, 300.043054873601, 397.296174338544]
    )

    np.testing.assert_allclose(section_arm.nodes.dx, expected_dx, atol=1e-4)
    np.testing.assert_allclose(section_arm.nodes.dy, expected_dy, atol=1e-4)
    np.testing.assert_allclose(section_arm.nodes.dz, expected_dz, atol=1e-4)
    np.testing.assert_allclose(section_arm.L_ref, expected_L_ref, atol=1e-4)


def test_adjust_with_angles(section_3d_angles_arm: Span):
    section_3d_angles_arm.solve_adjustment()
    expected_dx = np.array(
        [
            2.98519820570843,
            0.0203329438008079,
            0.0259050035856634,
            -2.97622800585608,
        ]
    )
    expected_dy = np.array(
        [
            0.0651438662085866,
            0.92496568246559,
            1.18432536063032,
            -0.0655978926646298,
        ]
    )
    expected_dz = np.array(
        [
            -0.290427184214129,
            -2.85377382734816,
            -2.75621159763852,
            -0.371165426239995,
        ]
    )
    expected_L_ref = np.array(
        [497.329093771962, 299.849533788068, 397.243869443178]
    )

    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dx, expected_dx, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dy, expected_dy, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dz, expected_dz, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.L_ref, expected_L_ref, atol=1e-4
    )


def test_wind_no_altitude_change(section_3d_no_altitude_change: Span):
    section_3d_no_altitude_change.solve_adjustment()

    section_3d_no_altitude_change.cable_loads.wind_pressure = np.array(
        [300] * 4
    )
    section_3d_no_altitude_change.solve_change_state()
    expected_dx = np.array(
        [
            2.97140319423837,
            -0.0178208618340392,
            -0.00222111510271174,
            -2.98084374666492,
        ]
    )
    expected_dy = np.array(
        [
            0.187877333786573,
            1.35446839777292,
            1.33685663266011,
            0.153911806319682,
        ]
    )
    expected_dz = np.array(
        [
            -0.368055926075446,
            -2.67677002678937,
            -2.68566740501507,
            -0.301465941432917,
        ]
    )

    np.testing.assert_allclose(
        section_3d_no_altitude_change.nodes.dx, expected_dx, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_no_altitude_change.nodes.dy, expected_dy, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_no_altitude_change.nodes.dz, expected_dz, atol=1e-4
    )


def test_wind(cable_AM600: Cable):
    nodes_arm = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 50, 50, 0]),
        line_angle=f.grad_to_rad(np.array([0, 20, 30, 0])),
        span_length=np.array([500, 300, 400]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([0, 0, 0, 0]),
        load_position=np.array([0, 0, 0]),
    )

    section = Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes_arm,
        cable=cable_AM600,
    )
    section.solve_adjustment()

    section.cable_loads.wind_pressure = np.array([200] * 4)
    section.solve_change_state()
    expected_dx = np.array(
        [
            2.95577748400265,
            -0.162868268485096,
            0.184747934417769,
            -2.93962231535185,
        ]
    )
    expected_dy = np.array(
        [
            0.43919754975307,
            2.41449548641855,
            2.67750769405881,
            0.4896647687387,
        ]
    )
    expected_dz = np.array(
        [
            -0.265490070160588,
            -1.77304412612465,
            -1.34045542595949,
            -0.344744916338915,
        ]
    )

    np.testing.assert_allclose(section.nodes.dx, expected_dx, atol=1e-4)
    np.testing.assert_allclose(section.nodes.dy, expected_dy, atol=1e-4)
    np.testing.assert_allclose(section.nodes.dz, expected_dz, atol=1e-4)


def test_temperature(section_3d_angles_arm: Span):
    section_3d_angles_arm.solve_adjustment()

    section_3d_angles_arm.sagging_temperature = 90
    section_3d_angles_arm.solve_change_state()
    expected_dx = np.array(
        [
            2.96987911200016,
            -0.203601320556107,
            -0.00208703890122741,
            -2.9594874382405,
        ]
    )
    expected_dy = np.array(
        [
            0.0635024174143705,
            0.717771096088857,
            0.942411070979133,
            -0.0670142019873425,
        ]
    )
    expected_dz = np.array(
        [
            -0.419267817853534,
            -2.90574450974053,
            -2.8481321980492,
            -0.486768116898264,
        ]
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dx, expected_dx, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dy, expected_dy, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dz, expected_dz, atol=1e-4
    )


def test_ice(section_3d_angles_arm: Span):
    section_3d_angles_arm.solve_adjustment()

    section_3d_angles_arm.cable_loads.ice_thickness = (
        np.array([1, 1, 1, 1]) * 1e-2
    )
    section_3d_angles_arm.solve_change_state()
    expected_dx = np.array(
        [
            2.98328197567154,
            -0.0327202608024265,
            0.0165977073561802,
            -2.97432426093715,
        ]
    )
    expected_dy = np.array(
        [
            0.0647645798151866,
            0.87144066769098,
            1.12484584084215,
            -0.0659967043228111,
        ]
    )
    expected_dz = np.array(
        [
            -0.309570998051614,
            -2.87045650502273,
            -2.78108726048836,
            -0.386056505992619,
        ]
    )

    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dx, expected_dx, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dy, expected_dy, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dz, expected_dz, atol=1e-4
    )


def test_load_all_spans(cable_AM600: Cable):
    nodes_arm = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, -10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 10, 0, 0])),
        span_length=np.array([500, 300, 400]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([500, 1000, 500]),
        load_position=np.array([0.2, 0.4, 0.6]),
    )

    section_3d_angles_arm = Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes_arm,
        cable=cable_AM600,
    )

    section_3d_angles_arm.solve_adjustment()

    section_3d_angles_arm.solve_change_state()

    expected_dx = np.array(
        [
            2.98499074985482,
            0.103422693278665,
            0.0199239757392463,
            -2.97652120035798,
        ]
    )
    expected_dy = np.array(
        [
            0.0650921884495981,
            0.912350814929752,
            1.17581780256653,
            -0.0656666843197686,
        ]
    )
    expected_dz = np.array(
        [
            -0.292563207331304,
            -2.85603216666269,
            -2.75990136243279,
            -0.368794563937624,
        ]
    )

    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dx, expected_dx, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dy, expected_dy, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dz, expected_dz, atol=1e-4
    )


def test_load_all_spans_wind_ice_temp(cable_AM600: Cable):
    nodes_arm = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, -10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 10, 0, 0])),
        span_length=np.array([500, 300, 400]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([500, 1000, 500]),
        load_position=np.array([0.2, 0.4, 0.6]),
    )

    section_3d_angles_arm = Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes_arm,
        cable=cable_AM600,
    )

    section_3d_angles_arm.solve_adjustment()
    section_3d_angles_arm.sagging_temperature = 30
    section_3d_angles_arm.cable_loads.ice_thickness = np.array([1] * 4) * 1e-2
    section_3d_angles_arm.cable_loads.wind_pressure = np.array([500] * 4)
    section_3d_angles_arm.solve_change_state()

    expected_dx = np.array(
        [
            2.9639224331351,
            -0.0492527323586429,
            0.0230113114082075,
            -2.97582612580702,
        ]
    )
    expected_dy = np.array(
        [
            0.405625632045006,
            2.36632575817778,
            2.3905212332461,
            0.211183442571553,
        ]
    )
    expected_dz = np.array(
        [
            -0.225014792817208,
            -1.84338725571692,
            -1.8124234364372,
            -0.316006997625079,
        ]
    )

    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dx, expected_dx, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dy, expected_dy, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dz, expected_dz, atol=1e-4
    )


def test_load_one_span(cable_AM600: Cable):
    nodes_arm = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, -10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 10, 0, 0])),
        span_length=np.array([500, 300, 400]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([0, 1000, 0]),
        # currently does not work if a load_position is set to 0
        load_position=np.array([0.2, 0.4, 0.6]),
    )

    section_3d_angles_arm = Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes_arm,
        cable=cable_AM600,
    )

    section_3d_angles_arm.solve_adjustment()

    section_3d_angles_arm.solve_change_state()

    expected_dx = np.array(
        [
            2.98631241983472,
            0.106794467626418,
            -0.0276983016774005,
            -2.9774999521832,
        ]
    )
    expected_dy = np.array(
        [
            0.0649786506874284,
            0.888327055351789,
            1.16170727466419,
            -0.0657854850521439,
        ]
    )
    expected_dz = np.array(
        [
            -0.278775727235584,
            -2.86347166642424,
            -2.7658035020725,
            -0.360785676968292,
        ]
    )

    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dx, expected_dx, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dy, expected_dy, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_angles_arm.nodes.dz, expected_dz, atol=1e-4
    )


def test_many_spans(cable_AM600: Cable):
    nb_spans = 50
    nodes_arm = Nodes(
        L_chain=np.array([3] * nb_spans),
        weight_chain=np.array([500] * nb_spans),
        arm_length=np.array([0] * nb_spans),
        line_angle=f.grad_to_rad(np.array([0] * nb_spans)),
        span_length=np.array([500] * (nb_spans -1)),
        z=np.array([50] * nb_spans),
        load=np.array([0] * (nb_spans - 1)),
        load_position=np.array([0.5] * (nb_spans - 1)),
    )

    section = Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes_arm,
        cable=cable_AM600,
    )
    section.solve_adjustment()

    section.cable_loads.wind_pressure = np.array([-200] * nb_spans)
    section.solve_change_state()
