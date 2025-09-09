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
        x=np.array([0, 500, 800, 1200]),
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
        x=np.array([0, 500, 800, 1200]),
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


def test_element_sandbox(cable_AM600: Cable):
    nodes_arm = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, -10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 10, 0, 0])),
        x=np.array([0, 500, 800, 1200]),
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
    section.adjust()
    section.L_ref

    # section.sagging_temperature = 30
    # section.cable_loads.ice_thickness = np.array([1,1,1,1]) * 1e-2
    section.cable_loads.wind_pressure = np.array([200, 200, 200, 200])
    section.change_state()
    print("Th", section.Th)
    print("dx", section.sb.final_dx)
    print("dy", section.sb.final_dy)
    print("dz", section.sb.final_dz)
    assert True



def test_element_sandbox_load(cable_AM600: Cable):
    nodes_arm = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, -10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 10, 0, 0])),
        x=np.array([0, 500, 800, 1200]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([0, 1000, 0]),
        load_position=np.array([0.5, 0.5, 0.5]),
    )

    section = Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes_arm,
        cable=cable_AM600,
    )

    section.adjust()
    section.L_ref

    section.nodes.has_load = True

    section.change_state()

    assert True

def test_element_initialisation(section_3d_simple: Span):
    # load = section_2d_note.nodes.load

    print("\n")
    print(section_3d_simple)
    print(section_3d_simple.nodes)


def test_element_adjust(section_3d_simple: Span):
    section_3d_simple.adjust()
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

    section_3d_simple.L_ref

    np.testing.assert_allclose(
        section_3d_simple.nodes.dx, expected_dx, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_simple.nodes.dy, expected_dy, atol=1e-4
    )
    np.testing.assert_allclose(
        section_3d_simple.nodes.dz, expected_dz, atol=1e-4
    )

    assert True


def test_element_adjust_with_arm(cable_AM600: Cable):
    nodes = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, 10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 0, 0, 0])),
        x=np.array([0, 500, 800, 1200]),
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
    section_arm.adjust()
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

    np.testing.assert_allclose(section_arm.nodes.dx, expected_dx, atol=1e-4)
    np.testing.assert_allclose(section_arm.nodes.dy, expected_dy, atol=1e-4)
    np.testing.assert_allclose(section_arm.nodes.dz, expected_dz, atol=1e-4)


def test_element_adjust_with_angles(cable_AM600: Cable):
    nodes_arm = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, -10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 10, 0, 0])),
        x=np.array([0, 500, 800, 1200]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([0, 0, 0, 0]),
        load_position=np.array([0, 0, 0]),
    )

    section_arm = Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes_arm,
        cable=cable_AM600,
    )
    section_arm.adjust()
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

    np.testing.assert_allclose(section_arm.nodes.dx, expected_dx, atol=1e-4)
    np.testing.assert_allclose(section_arm.nodes.dy, expected_dy, atol=1e-4)
    np.testing.assert_allclose(section_arm.nodes.dz, expected_dz, atol=1e-4)


def test_element_change_state(section_3d_simple: Span):
    section_3d_simple.adjust()

    section_3d_simple
    section_3d_simple.change_state()
    assert True


def test_wind(cable_AM600: Cable):
    nodes_arm = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 50, 50, 0]),
        line_angle=f.grad_to_rad(np.array([0, 20, 30, 0])),
        x=np.array([0, 500, 800, 1200]),
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
    section.adjust()
    section.L_ref

    section.cable_loads.wind_pressure = np.array([200, 200, 200, 200]) * -1
    section.change_state()
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


def test_element_load():

    cable_AM600 = Cable(600.4e-6, 17.658, 0.000023, 60e9, 31.86e-3, 320)
    nodes_arm = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, -10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 10, 0, 0])),
        x=np.array([0, 500, 800, 1200]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([0, 1000, 0]),
        load_position=np.array([0.2, 0.4, 0.6]),
    )

    section = Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes_arm,
        cable=cable_AM600,
    )

    section.adjust()
    section.L_ref

    section.nodes.has_load = True
    section.cable_loads.wind_pressure = np.array([500, 500, 500, 500])

    section.change_state()

    assert True



def test_many_spans(cable_AM600: Cable):
    nb_spans = 50
    nodes_arm = Nodes(
        L_chain=np.array([3] * nb_spans),
        weight_chain=np.array([500] * nb_spans),
        arm_length=np.array([0] * nb_spans),
        line_angle=f.grad_to_rad(np.array([0] * nb_spans)),
        x=np.arange(0,500 * nb_spans, 500),
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
    section.adjust()
    section.L_ref

    section.cable_loads.wind_pressure = np.array([-200] * nb_spans)
    section.change_state()