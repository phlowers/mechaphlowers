from mechaphlowers.core.models.balance.elements import Cable, SolverBalance, Span, Nodes
import numpy as np
from pytest import fixture


@fixture
def cable_AM600():
    return Cable(600.4e-3, 17.67, 0.000023, 60e9)

@fixture
def section_2d_note(cable_AM600):
    nodes = Nodes(
        # num=np.arange(0,),
        ntype=np.array([3, 1, 2, 1, 2, 1, 3 ]),
        L_chain=np.array([3, 0, 3, 0, 3, 0, 3]),
        weight_chain=np.array([1000, 0, 500, 0, 500, 0, 1000]),
        x=np.array([0, 8, 500, 595, 800, 980, 1200]),
        z=np.array([30, 0, 50, 0, 60, 0, 65]),
        load=np.array([0, 0, 0, 0, 0, 0, 0]),
    )
    
    return Span(
        parameter=1500,
        cable_temperature=15,
        nodes=nodes,
        cable=cable_AM600,
    )
    

def test_element(section_2d_note):
    print("\n")
    print(section_2d_note)
    print(section_2d_note.nodes)
    
    ss = SolverBalance(section_2d_note).solve()
    
    
    # section_2d_note.compute_balance()
    
    assert True
    