import numpy as np
from pytest import fixture

from mechaphlowers.core.models.balance.elements import Cable, Nodes, Span
import mechaphlowers.core.models.balance.functions as f


@fixture
def cable_AM600():
    return Cable(600.4, 17.67, 0.000023, 60e3, 31.86, 320)


@fixture
def section_2d_note(cable_AM600):
    nodes = Nodes(
        # num=np.arange(0,),
        ntype=np.array([3, 2, 2, 3]),
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, -10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 10, 0, 0])),
        x=np.array([0, 500, 800, 1200]),
        z=np.array([30, 50, 60, 65]),  # = z0
        load=np.array([0, 0, 0, 0]),
    )

    return Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes,
        cable=cable_AM600,
    )


def test_element_initialisation(section_2d_note):
    # load = section_2d_note.nodes.load

    print("\n")
    print(section_2d_note)
    print(section_2d_note.nodes)


def test_element_adjust(section_2d_note):
    section_2d_note.adjust()
    pass


def test_element_adjust_arm(cable_AM600):
    nodes_arm = Nodes(
        # num=np.arange(0,),
        ntype=np.array([3, 2, 2, 3]),
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 3, 3, 0]),
        line_angle=np.array([0, 0, 0, 0]),
        x=np.array([0, 500, 800, 1200]),
        z=np.array([30, 50, 60, 65]),  # = z0
        load=np.array([0, 0, 0, 0]),
    )

    section_arm = Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes_arm,
        cable=cable_AM600,
    )
    section_arm.adjust()
    assert True
