

import numpy as np
import pandas as pd
from pytest import fixture

from mechaphlowers.core.geometry.points import Points, SectionPoints, stack_nan, vectors_to_coords
from mechaphlowers.core.models.cable.span import CatenarySpan
from mechaphlowers.entities.arrays import SectionArray


@fixture
def section_array_line_angles():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["support 1", "2", "three", "support 4"]),
                "suspension": np.array([False, True, True, False]),
                "conductor_attachment_altitude": np.array([30, 40, 60, 70]),
                "crossarm_length": np.array([40, 20, -30, -50]),
                "line_angle": np.array([0, -45, 60, -30]),
                "insulator_length": np.array([0, 5, 82, 0]),
                "span_length": np.array([500, 460, 520, np.nan]),
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
    return section_array


def test_stack_nan():
    """Test the stack_nan function."""
    
    coords = np.array([[[-2.34427201e+02,  0.00000000e+00,  1.37547654e+01],
        [-1.49742252e+02,  0.00000000e+00,  5.60830465e+00],
        [-6.50573031e+01,  0.00000000e+00,  1.05820648e+00],
        [ 1.96276459e+01,  0.00000000e+00,  9.63118943e-02],
        [ 1.04312595e+02,  0.00000000e+00,  2.72089608e+00],
        [ 1.88997544e+02,  0.00000000e+00,  8.93666533e+00],
        [ 2.73682493e+02,  0.00000000e+00,  1.87547654e+01]],

       [[-4.76437543e+02,  0.00000000e+00,  5.70170544e+01],
        [-3.95654723e+02,  0.00000000e+00,  3.92634649e+01],
        [-3.14871902e+02,  0.00000000e+00,  2.48373168e+01],
        [-2.34089081e+02,  0.00000000e+00,  1.37150711e+01],
        [-1.53306260e+02,  0.00000000e+00,  5.87857990e+00],
        [-7.25234393e+01,  0.00000000e+00,  1.31505640e+00],
        [ 8.25938147e+00,  0.00000000e+00,  1.70543698e-02]],
       
        [[-4.76437543e+02,  0.00000000e+00,  5.70170544e+01],
        [-3.95654723e+02,  0.00000000e+00,  3.92634649e+01],
        [-3.14871902e+02,  0.00000000e+00,  2.48373168e+01],
        [-2.34089081e+02,  0.00000000e+00,  1.37150711e+01],
        [-1.53306260e+02,  0.00000000e+00,  5.87857990e+00],
        [-7.25234393e+01,  0.00000000e+00,  1.31505640e+00],
        [ 8.25938147e+00,  0.00000000e+00,  1.70543698e-02]],

       [[ 8.80823413e+01,  0.00000000e+00,  1.93993824e+00],
        [ 1.75171637e+02,  0.00000000e+00,  7.67618085e+00],
        [ 2.62260932e+02,  0.00000000e+00,  1.72198528e+01],
        [ 3.49350227e+02,  0.00000000e+00,  3.05890530e+01],
        [ 4.36439522e+02,  0.00000000e+00,  4.78091353e+01],
        [ 5.23528817e+02,  0.00000000e+00,  6.89127565e+01],
        [ 6.10618113e+02,  0.00000000e+00,  9.39399382e+01]]])
    
    expected_output = np.array([[-2.34427201e+02,  0.00000000e+00,  1.37547654e+01],
       [-1.49742252e+02,  0.00000000e+00,  5.60830465e+00],
       [-6.50573031e+01,  0.00000000e+00,  1.05820648e+00],
       [ 1.96276459e+01,  0.00000000e+00,  9.63118943e-02],
       [ 1.04312595e+02,  0.00000000e+00,  2.72089608e+00],
       [ 1.88997544e+02,  0.00000000e+00,  8.93666533e+00],
       [ 2.73682493e+02,  0.00000000e+00,  1.87547654e+01],
       [            np.nan,             np.nan,             np.nan],
       [-4.76437543e+02,  0.00000000e+00,  5.70170544e+01],
       [-3.95654723e+02,  0.00000000e+00,  3.92634649e+01],
       [-3.14871902e+02,  0.00000000e+00,  2.48373168e+01],
       [-2.34089081e+02,  0.00000000e+00,  1.37150711e+01],
       [-1.53306260e+02,  0.00000000e+00,  5.87857990e+00],
       [-7.25234393e+01,  0.00000000e+00,  1.31505640e+00],
       [ 8.25938147e+00,  0.00000000e+00,  1.70543698e-02],
       [            np.nan,             np.nan,             np.nan],
        [-4.76437543e+02,  0.00000000e+00,  5.70170544e+01],
        [-3.95654723e+02,  0.00000000e+00,  3.92634649e+01],
        [-3.14871902e+02,  0.00000000e+00,  2.48373168e+01],
        [-2.34089081e+02,  0.00000000e+00,  1.37150711e+01],
        [-1.53306260e+02,  0.00000000e+00,  5.87857990e+00],
        [-7.25234393e+01,  0.00000000e+00,  1.31505640e+00],
        [ 8.25938147e+00,  0.00000000e+00,  1.70543698e-02],
        [            np.nan,             np.nan,             np.nan],
       [ 8.80823413e+01,  0.00000000e+00,  1.93993824e+00],
       [ 1.75171637e+02,  0.00000000e+00,  7.67618085e+00],
       [ 2.62260932e+02,  0.00000000e+00,  1.72198528e+01],
       [ 3.49350227e+02,  0.00000000e+00,  3.05890530e+01],
       [ 4.36439522e+02,  0.00000000e+00,  4.78091353e+01],
       [ 5.23528817e+02,  0.00000000e+00,  6.89127565e+01],
       [ 6.10618113e+02,  0.00000000e+00,  9.39399382e+01],
       [            np.nan,             np.nan,             np.nan]])
    

    result = stack_nan(coords)
    assert result.shape[1] == coords.shape[2]
    assert result.shape[0] == (coords.shape[1]+1)*coords.shape[0]
    np.testing.assert_allclose(result, expected_output)



def test_point_class():
    """Test the Point class."""
    
    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]).T
    y = np.array([[0.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]]).T
    z = np.array([[10.0, 20.0, 30.0, 40.0], [40.0, 30.0, 20.0, 10.0]]).T

    coords = vectors_to_coords(x, y, z)
    p1 = Points.from_coords(coords)
    
    x_p1, y_p1, z_p1 = p1.vectors

    np.testing.assert_almost_equal(x_p1, x)
    np.testing.assert_almost_equal(y_p1, y)
    np.testing.assert_almost_equal(z_p1, z)
    
    assert p1.points().shape == (2*x.shape[0], 3)
    assert p1.points(stack=True).shape == (2*y.shape[0]+2, 3)
    
    p2 = Points.from_vectors(x, y, z)
    np.testing.assert_almost_equal(p1.coords, p2.coords)
    
    


def test_span_absolute_coords_new_obj(section_array_line_angles):


    span_model = CatenarySpan(**section_array_line_angles.to_numpy())
    
    s = SectionPoints(
        span_model=span_model, **section_array_line_angles.to_numpy()
    )
    
    s.get_spans("cable").points(True)
    s.get_spans("crossarm").points(True)
    s.get_spans("section").points(True)
    s.get_supports().points(True)
    s.get_insulators().points(True)

    s.init_span(span_model)

    # from plotly import graph_objects as go 
    # from mechaphlowers.plotting.plot import plot_points_3d, set_layout
    # fig = go.Figure()
    # plot_points_3d(fig, s.get_spans("cable").points(True))
    # plot_points_3d(fig, s.get_spans("crossarm").points(True))
    # plot_points_3d(fig, s.get_spans("section").points(True))

    # plot_points_3d(fig, s.get_supports().points(True))
    # plot_points_3d(fig, s.get_insulators().points(True))

    # set_layout(fig)
    # fig.show()
    assert True