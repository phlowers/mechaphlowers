from mechaphlowers.core.models.balance.elements import (
    Cable,
    Span,
    Nodes,
    SolverBalance,
    MapVectorToNodeSpace,
)
import numpy as np
from pytest import fixture


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
        arm_length=np.array([0, 3, 3, 0]),
        line_angle=np.array(
            [
                0,
                0,
                0,
                0,
            ]
        ),
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


def test_element_no_load(section_2d_note):
    # load = section_2d_note.nodes.load

    print("\n")
    print(section_2d_note)
    print(section_2d_note.nodes)

    # Testing initialization
    # x = np.array([3.00, 8.00, 500.00, 595.00, 800.00, 980.00, 1197.00])
    # z = np.array([30.00, 28.95, 50.00, 46.67, 60.00, 49.01, 65.00])
    # dx = np.array([-0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04])
    # dz = np.array([-0.43, 0.00, 0.00, 0.00, 0.00, 0.00, -0.48])

    # Th = np.array([26504.9, 26505.0, 26505.0, 26505.0, 26505.0, 26505.0])
    # Tv_g = -np.array(
    #     [3315.8, 3226.2, 1769.9, 90.0, 3215.0, 26.6]
    # )  # minus because of orientation of force to the ground in the reference
    # Tv_d = -np.array(
    #     [-3226.2, 5514.4, -90.0, 3542.8, -26.6, 3821.6]
    # )  # minus because of orientation of force to the ground in the reference

    # np.testing.assert_allclose(section_2d_note.nodes.x, x, atol=0.01)
    # np.testing.assert_allclose(section_2d_note.nodes.z, z, atol=0.01)
    # np.testing.assert_allclose(section_2d_note.nodes.dx, dx, atol=0.01)
    # np.testing.assert_allclose(section_2d_note.nodes.dz, dz, atol=0.01)

    # np.testing.assert_allclose(section_2d_note.Th, Th, atol=0.1)
    # np.testing.assert_allclose(section_2d_note.Tv_d, Tv_d, atol=0.1)
    # np.testing.assert_allclose(section_2d_note.Tv_g, Tv_g, atol=0.1)

    # # testing RealSpan formulas
    # np.testing.assert_allclose(
    #     section_2d_note.L_ref,
    #     np.array(
    #         [
    #             5.06327961,
    #             494.12046026,
    #             94.97159033,
    #             205.36991365,
    #             180.24810945,
    #             217.54589161,
    #         ]
    #     ),
    # )


@fixture
def section_2d_250m_load(cable_AM600):
    nodes = Nodes(
        # num=np.arange(0,),
        ntype=np.array([3, 1, 2, 1, 2, 1, 3]),
        L_chain=np.array([3, 0, 3, 0, 3, 0, 3]),
        weight_chain=np.array([1000, 0, 500, 0, 500, 0, 1000]),
        arm_length=np.array([20, 0, 20, 0, 20, 0, 20]),
        line_angle=np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
        x=np.array([0, 250, 500, 595, 800, 980, 1200]),
        z=np.array([30, 0, 50, 0, 60, 0, 65]),
        load=np.array([0, 5_000, 0, 0, 0, 0, 0]),
    )

    return Span(
        parameter=1500,
        sagging_temperature=15,
        nodes=nodes,
        cable=cable_AM600,
    )


def test_element_balance_load_temperature_90(section_2d_250m_load):
    init_force = np.array(
        [
            2740.09313533,
            62.69108688,
            -6497.46806333,
            25175.00529491,
            6609.18438809,
            -1311.10584335,
            3606.41217561,
            1726.96819799,
            -1464.59819502,
            -2792.25945076,
        ]
    )

    i_finition = 8

    final_dz = np.array(
        [
            -0.4965534,
            -12.17332206,
            0.37121645,
            0.95111065,
            0.10715173,
            0.57924502,
            -0.46387328,
        ]
    )

    final_dx = np.array(
        [
            -0.04137959,
            -0.20946584,
            -1.44550926,
            -1.25168205,
            -0.79462502,
            -0.44052127,
            0.03608003,
        ]
    )

    final_force = np.array(
        [
            8.19522757e-06,
            -1.13989096e-04,
            -2.99624171e-05,
            -2.58384816e-04,
            1.04279679e-05,
            -4.10316005e-06,
            2.52814152e-06,
            1.19415163e-05,
            -2.82962995e-06,
            -3.52824645e-06,
        ]
    )

    nb_iter = 6

    norm_list = [
        np.float64(486323547.15663373),
        np.float64(223772622.82367128),
        np.float64(39577786.257005334),
        np.float64(4879523.678757754),
        np.float64(280449.7042039623),
        np.float64(6728.3761899579495),
        np.float64(324598.2708183059),
        np.float64(324072.2973367483),
        np.float64(596.1914508942701),
        np.float64(0.6393241117213183),
        np.float64(0.00032964207508432396),
    ]

    correction_list = [
        np.array(
            [
                0.24388344,
                0.07215246,
                28.905246,
                2.41742109,
                1.82145037,
                -0.42366556,
                1.06320195,
                0.53897115,
                1.04538445,
                0.0312465,
            ]
        ),
        np.array(
            [
                1.63666570e-03,
                2.44630909e-01,
                3.92553865e00,
                8.64005695e-01,
                8.60637584e-01,
                -7.02536882e-01,
                5.81472101e-01,
                3.41110809e-01,
                -8.67886715e-01,
                -2.76489245e-02,
            ]
        ),
        np.array(
            [
                -0.00877914,
                0.06171136,
                1.00040508,
                0.23887669,
                0.23504085,
                -0.43246686,
                0.16202141,
                0.09596136,
                -0.47534476,
                -0.01509609,
            ]
        ),
        np.array(
            [
                -0.00331623,
                0.01407823,
                0.23686269,
                0.05855384,
                0.05772725,
                -0.1483194,
                0.03896906,
                0.02328961,
                -0.13051533,
                -0.00403281,
            ]
        ),
        np.array(
            [
                -0.00074509,
                0.00265211,
                0.04575543,
                0.01148231,
                0.01135161,
                -0.03454266,
                0.00759427,
                0.00463002,
                -0.02753359,
                -0.00081644,
            ]
        ),
        np.array(
            [
                -0.0001123,
                0.00037357,
                0.00653312,
                0.00165253,
                0.00163141,
                -0.00548932,
                0.00110256,
                0.00069319,
                -0.00436868,
                -0.00012304,
            ]
        ),
        np.array(
            [
                -1.24809669e-05,
                3.51525695e-05,
                6.04160441e-04,
                1.68479819e-04,
                1.65240861e-04,
                -6.13840126e-04,
                1.14846967e-04,
                7.50851619e-05,
                -5.22628386e-04,
                -1.39838058e-05,
            ]
        ),
        np.array(
            [
                0.00028174,
                0.00056378,
                0.01572567,
                -0.00042817,
                -0.00034437,
                0.00285128,
                -0.00036971,
                -0.0001692,
                0.00590151,
                0.00019445,
            ]
        ),
        np.array(
            [
                1.26696679e-05,
                2.41640174e-05,
                7.01466784e-04,
                -1.83437448e-05,
                -1.44490901e-05,
                1.77290612e-04,
                -1.65232794e-05,
                -9.11880234e-06,
                2.98376595e-04,
                9.25184332e-06,
            ]
        ),
        np.array(
            [
                4.25385196e-07,
                8.94403794e-07,
                2.38972940e-05,
                -6.12578953e-07,
                -4.88073194e-07,
                7.80584941e-06,
                -5.52239169e-07,
                -3.67685694e-07,
                1.07377944e-05,
                3.13440114e-07,
            ]
        ),
        np.array(
            [
                9.94746112e-09,
                2.36638816e-08,
                5.70456216e-07,
                -1.44165148e-08,
                -1.18605909e-08,
                2.34991274e-07,
                -1.26147336e-08,
                -9.94648897e-09,
                2.64734678e-07,
                7.28363438e-09,
            ]
        ),
    ]

    # load = section_2d_note.nodes.load

    sb = SolverBalance()
    ### Test relaxation and solver
    sb.solver_balance(section_2d_250m_load, 90)

    # np.testing.assert_allclose(sb.init_force, init_force, rtol=1e-3)
    np.testing.assert_allclose(sb.final_dx, final_dx, rtol=1e-4)
    np.testing.assert_allclose(sb.final_dz, final_dz, rtol=1e-4)
    assert np.linalg.norm(sb.final_force) < 1e-3
    # np.testing.assert_allclose(
    #     np.array([ii["norm_d_param"] for ii in sb.mem_loop]),
    #     np.array(norm_list),
    # )

    # assert sb.i_finition == i_finition
    assert nb_iter == len(sb.mem_loop)


def test_MapVectorToNodeSpace(section_2d_250m_load):
    mp = MapVectorToNodeSpace(section_2d_250m_load.nodes)

    vector_force = np.array(
        [
            0.00000000e00,
            0.00000000e00,
            -5.00000019e03,
            0.00000000e00,
            0.00000000e00,
            7.53175300e-13,
            0.00000000e00,
            0.00000000e00,
            -2.86149986e-04,
            0.00000000e00,
        ]
    )

    hidden_tensor = np.array(
        [
            [31, 11, 21, 11, 21, 11, 41],
            [32, 12, 22, 12, 22, 12, 42],
            [33, 13, 23, 13, 23, 13, 43],
        ]
    )

    np.testing.assert_allclose(mp.vector_value(), vector_force, atol=1e-10)

    np.testing.assert_allclose(mp.tensor, hidden_tensor, atol=1e-10)

    cat_num_vector = hidden_tensor.T.reshape(-1)[mp.mask_tensor_flat_to_vector]

    filter_nodes_equal_1 = np.array([0, 11, 12, 0, 11, 12, 0, 11, 12, 0])
    filter_nodes_equal_2 = np.array([0, 0, 0, 23, 0, 0, 23, 0, 0, 0])
    filter_nodes_equal_3 = np.array([33, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    filter_nodes_equal_4 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 43])

    np.testing.assert_allclose(
        mp.filter_to_nodes_values(cat_num_vector, 1), filter_nodes_equal_1
    )
    np.testing.assert_allclose(
        mp.filter_to_nodes_values(cat_num_vector, 2), filter_nodes_equal_2
    )
    np.testing.assert_allclose(
        mp.filter_to_nodes_values(cat_num_vector, 3), filter_nodes_equal_3
    )
    np.testing.assert_allclose(
        mp.filter_to_nodes_values(cat_num_vector, 4), filter_nodes_equal_4
    )

    assert True
