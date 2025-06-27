from mechaphlowers.core.models.balance.elements import Cable, SolverBalance, Span, Nodes
import numpy as np
from pytest import fixture


@fixture
def cable_AM600():
    return Cable(600.4, 17.67, 0.000023, 60e3)

    
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
    
# @fixture
# def section_2d_note_with_load(cable_AM600):
#     nodes = Nodes(
#         # num=np.arange(0,),
#         ntype=np.array([3, 1, 2, 1, 2, 1, 3 ]),
#         L_chain=np.array([3, 0, 3, 0, 3, 0, 3]),
#         weight_chain=np.array([1000, 0, 500, 0, 500, 0, 1000]),
#         x=np.array([0, 8, 500, 595, 800, 980, 1200]),
#         z=np.array([30, 0, 50, 0, 60, 0, 65]),
#         load=np.array([0, 10_000, 0, 0, 0, 0, 0]),
#     )
    
#     return Span(
#         parameter=1500,
#         cable_temperature=15,
#         nodes=nodes,
#         cable=cable_AM600,
#     )
    

def test_element_no_load(section_2d_note):
    # load = section_2d_note.nodes.load
    

    print("\n")
    print(section_2d_note)
    print(section_2d_note.nodes)
    
    # Testing initialization
    x=np.array([3.00, 8.00, 500.00, 595.00, 800.00, 980.00, 1197.00])
    z=np.array([30.00, 28.95, 50.00, 46.67, 60.00, 49.01, 65.00])
    dx=np.array([-0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04])
    dz=np.array([-0.43, 0.00, 0.00, 0.00, 0.00, 0.00, -0.48])
    
    Th=np.array([26504.9, 26505.0, 26505.0, 26505.0, 26505.0, 26505.0])
    Tv_g=-np.array([3315.8, 3226.2, 1769.9, 90.0, 3215.0, 26.6]) # minus because of orientation of force to the ground in the reference
    Tv_d=-np.array([-3226.2, 5514.4, -90.0, 3542.8, -26.6, 3821.6]) # minus because of orientation of force to the ground in the reference
    
    np.testing.assert_allclose(section_2d_note.nodes.x , x, atol=0.01)
    np.testing.assert_allclose(section_2d_note.nodes.z , z, atol=0.01)
    np.testing.assert_allclose(section_2d_note.nodes.dx , dx, atol=0.01)
    np.testing.assert_allclose(section_2d_note.nodes.dz , dz, atol=0.01)
    
    np.testing.assert_allclose(section_2d_note.Th , Th, atol=0.1)
    np.testing.assert_allclose(section_2d_note.Tv_d , Tv_d, atol=0.1)
    np.testing.assert_allclose(section_2d_note.Tv_g , Tv_g, atol=0.1)
    
    # testing RealSpan formulas
    np.testing.assert_allclose( section_2d_note.L_ref , 
                               np.array([  5.06327961, 494.12046026,  94.97159033, 205.36991365,
       180.24810945, 217.54589161])
    )
    


# def test_element_no_load(section_2d_note_with_load):
#     # load = section_2d_note.nodes.load
    

#     print("\n")
#     print(section_2d_note_with_load)
#     print(section_2d_note_with_load.nodes)
    
#     # Testing initialization
#     x=np.array([3.00, 8.00, 500.00, 595.00, 800.00, 980.00, 1197.00])
#     z=np.array([30.00, 28.95, 50.00, 46.67, 60.00, 49.01, 65.00])
#     dx=np.array([-0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04])
#     dz=np.array([-0.43, 0.00, 0.00, 0.00, 0.00, 0.00, -0.48])
    
#     Th=np.array([26504.9, 26505.0, 26505.0, 26505.0, 26505.0, 26505.0])
#     Tv_g=-np.array([3315.8, 3226.2, 1769.9, 90.0, 3215.0, 26.6]) # minus because of orientation of force to the ground in the reference
#     Tv_d=-np.array([-3226.2, 5514.4, -90.0, 3542.8, -26.6, 3821.6]) # minus because of orientation of force to the ground in the reference
    
#     np.testing.assert_allclose(section_2d_note_with_load.nodes.x , x, atol=0.01)
#     np.testing.assert_allclose(section_2d_note_with_load.nodes.z , z, atol=0.01)
#     np.testing.assert_allclose(section_2d_note_with_load.nodes.dx , dx, atol=0.01)
#     np.testing.assert_allclose(section_2d_note_with_load.nodes.dz , dz, atol=0.01)
    
#     np.testing.assert_allclose(section_2d_note_with_load.Th , Th, atol=0.1)
#     np.testing.assert_allclose(section_2d_note_with_load.Tv_d , Tv_d, atol=0.1)
#     np.testing.assert_allclose(section_2d_note_with_load.Tv_g , Tv_g, atol=0.1)
    
#     # testing RealSpan formulas
#     np.testing.assert_allclose( section_2d_note_with_load.L_ref , 
#                                np.array([  5.06327961, 494.12046026,  94.97159033, 205.36991365,
#        180.24810945, 217.54589161])
#     )
    
    