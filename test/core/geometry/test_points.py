# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import plotly.graph_objects as go
from mechaphlowers.core.geometry.points import Points
from pytest import fixture

def plot_points(fig, points):

    fig.add_trace(go.Scatter3d(x=points[:,0],
                                y=points[:,1],
                                z=points[:,2],
                                mode='markers+lines',
                                marker=dict(size=4,),# color='red'),
                                name='Points'))
    
@fixture
def points_fixture():
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
    
def test_plot(points_fixture):
    fig = go.Figure()
    plot_points(fig, points_fixture.reshape(-1,3))
    
    fig.show()
