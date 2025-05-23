# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import plotly.graph_objects as go
from mechaphlowers.core.geometry.points import Points, Frame
from pytest import fixture
from numpy.testing import assert_allclose

def plot_points(fig, points):

    fig.add_trace(go.Scatter3d(x=points[:,0],
                                y=points[:,1],
                                z=points[:,2],
                                mode='markers+lines',
                                marker=dict(size=4,),# color='red'),
                                name='Points'))
    
    
def plot_frame( fig, frame, one_object = False):
    
    pp = np.hstack((frame.origin, frame.origin+frame.x_axis, np.nan*np.ones_like(frame.origin),frame.origin, frame.origin+frame.y_axis, np.nan*np.ones_like(frame.origin), frame.origin,frame.origin+frame.z_axis, np.nan*np.ones_like(frame.origin)))
    if one_object:
        pp = pp.reshape(-1, 3)
        fig.add_trace(go.Scatter3d(x=pp[:,0], y=pp[:,1], z = pp[:,2], mode='lines', marker=dict(size=5, color='blue'), name="Frame"))
    else:
        for i in range(pp.shape[0]):
            fig.add_trace(go.Scatter3d(x=pp[i].reshape(-1,3)[:,0], y=pp[i].reshape(-1,3)[:,1], z = pp[i].reshape(-1,3)[:,2], mode='lines', marker=dict(size=5, color='blue'), name=f"R{str(i)}"))
    return fig
    
@fixture
def coords_fixture():
    return np.array(
        [[ 100.        ,  -40.        ,   30.        ],
        [ 183.33333333,  -40.        ,   22.13841697],
        [ 266.66666667,  -40.        ,   17.75890716],
        [ 350.        ,  -40.        ,   16.85386616],
        [ 433.33333333,  -40.        ,   19.42172249],
        [ 516.66666667,  -40.        ,   25.46693488],
        [ 600.        ,  -40.        ,   35.        ],
        [ 100.        ,  -10.        ,   30.        ],
        [ 183.33333333,  -10.        ,   22.13841697],
        [ 266.66666667,  -10.        ,   17.75890716],
        [ 350.        ,  -10.        ,   16.85386616],
        [ 433.33333333,  -10.        ,   19.42172249],
        [ 516.66666667,  -10.        ,   25.46693488],
        [ 600.        ,  -10.        ,   35.        ],
        [ 500.        ,  -40.        ,   35.        ],
        [ 583.33333333,  -40.        ,   28.80643923],
        [ 666.66666667,  -40.        ,   26.0905585 ],
        [ 750.        ,  -40.        ,   26.84764206],
        [ 833.33333333,  -40.        ,   31.07900449],
        [ 916.66666667,  -40.        ,   38.79199295],
        [1000.        ,  -40.        ,   50.        ],
        [1000.        ,  460.        ,   50.        ],
        [1083.33333333,  460.        ,   44.63915405],
        [1166.66666667,  460.        ,   42.75431097],
        [1250.        ,  460.        ,   44.342198  ],
        [1333.33333333,  460.        ,   49.40557229],
        [1416.66666667,  460.        ,   57.95322568],
        [1500.        ,  460.        ,   70.        ]]).reshape((4,7,3))
    
def test_plot(coords_fixture):
    fig = go.Figure()
    plot_points(fig, coords_fixture.reshape(-1,3))
    
    # fig.show()


def test_translate_all(coords_fixture):
    expected_translated_coords = coords_fixture.copy()
    points = Points(coords_fixture)
    translation_vector = np.array([0,1,0])
    points.translate_all(translation_vector)

    translation = np.full(expected_translated_coords.shape, [0,1,0])
    expected_translated_coords += translation

    assert_allclose(points.coords, expected_translated_coords)



def test_translate_layer(coords_fixture):
    expected_translated_coords = coords_fixture.copy()
    points = Points(coords_fixture)
    translation_vector = np.array([0,1,0])
    points.translate_layer(translation_vector, 3)

    translation = np.full(expected_translated_coords[3].shape, [0,1,0])
    expected_translated_coords[3] += translation

    assert_allclose(points.coords, expected_translated_coords)


def test_flatten(coords_fixture):
    points = Points(coords_fixture)
    assert(points.flatten().shape == (28,3)) 
    
def test_rotate_layer(coords_fixture):

    fig = go.Figure()
    plot_points(fig, coords_fixture.reshape(-1,3))
    points = Points(coords_fixture)
    line_angles = np.array([180, 90, 180, 0])
    rotation_axes = np.array([[0,0,1]] * 4)
    points.rotate_one_angle_per_layer(line_angles, rotation_axes)

    plot_points(fig, points.coords_for_plot())
    
    # fig.show()


def test_rotate_point(coords_fixture):

    fig = go.Figure()
    plot_points(fig, coords_fixture.reshape(-1,3))
    points = Points(coords_fixture)
    line_angles = np.array([20] * points.flatten().shape[0])
    rotation_axes = np.array([[0,0,1]] * 4)
    points.rotate_one_angle_per_point(line_angles, rotation_axes)

    plot_points(fig, points.coords_for_plot())
    
    # fig.show()


def test_rotate_point_same_axis(coords_fixture):

    fig = go.Figure()
    plot_points(fig, coords_fixture.reshape(-1,3))
    points = Points(coords_fixture)
    line_angles = np.array([180, 90, 180, 0])
    rotation_axis = np.array([0,0,1])
    points.rotate_same_axis(line_angles, rotation_axis)

    plot_points(fig, points.coords_for_plot())
    
    # fig.show()
    
    
def test_plot_frame(coords_fixture):
    frame = Frame(np.array([[0,0,0], [1,1,1], [1,1,1], [2,1,1]]))
    
    fig = go.Figure()
    
    plot_frame(fig, frame)

    
    # fig.show()
    
def test_rotate_frame_same_axis(coords_fixture):
    frame = Frame(np.array([[0,0,0], [1,1,1], [1,1,1], [2,1,1]]))
    fig = go.Figure()
    plot_frame(fig, frame)


    line_angles = np.array([180, 90, 180, 0])
    rotation_axis = np.array([0,0,1])
    frame.rotate_one_angle_per_layer(line_angles, rotation_axis)

    plot_frame(fig, frame)

    
    # fig.show()


def test_points_frame_rotate_layer(coords_fixture):
    frame = Frame(np.array([[0,0,0], [1,1,1], [1,1,1], [2,1,1]]))
    fig = go.Figure()
    plot_points(fig, coords_fixture.reshape(-1,3))
    points = Points(coords_fixture)
    line_angles = np.array([180, 90, 180, 0])
    rotation_axes = np.array([[0,0,1]] * 4)
    points.rotate_one_angle_per_layer(line_angles, rotation_axes)

    plot_points(fig, points.coords_for_plot())
    plot_frame(fig, frame)
    # fig.show()  