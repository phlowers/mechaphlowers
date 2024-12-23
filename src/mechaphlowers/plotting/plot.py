# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mechaphlowers.entities import SectionFrame


def plot_line(fig: go.Figure, points: np.ndarray) -> None:
    """Plot the points of the cable onto the figure given

    Args:
        fig (go.Figure): plotly figure
        points (np.ndarray): points of all the cables of the section in point format (3 x n)
    """
    fig.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1],z=points[:,2], mode='lines+markers',
            marker=dict(size=5),
            line=dict(width=8, color='red'),))


def plot_support(fig: go.Figure, points: np.ndarray) -> None:
    """Plot the points of the support onto the figure given

    Args:
        fig (go.Figure): plotly figure
        points (np.ndarray): points of all the supports of the section in point format (3 x n)
    """
    fig.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1],z=points[:,2], mode='lines+markers',
            marker=dict(size=5),
            line=dict(width=8, color='green'),))


def get_support_points(data: pd.DataFrame) -> np.ndarray:
    """Temporary function to plot very simple 2-points-support with ground and attachment at the top. 

    Args:
        data (pd.DataFrame): SectionArray or SectionFrame data property

    Returns:
        np.ndarray: 3 x (2+1) number of support point for the data input
        Warning: every support is followed by a nan line to separate the traces on figure
    """
    
    x = np.pad(np.cumsum(data.span_length.to_numpy()[:-1]) , (1,0), "constant")
    init_xshape = len(x)
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    # get support points
    #ground points
    pp0 = np.vstack([x,y,z])
    
    # up points
    pp1 = pp0.copy()
    alt = data.conductor_attachment_altitude.to_numpy()
    pp1[2,:] = alt
    
    # add nan to separate
    ppf = np.concat([pp0.T, pp1.T, np.nan*pp0.T], axis = 1)

    return ppf.reshape(init_xshape*3, 3)


class PlotAccessor:
    """ First accessor class for plots. """

    def __init__(self, section: SectionFrame):
        self.section: SectionFrame = section

    def line3d(self, fig: go.Figure) -> None: 
        """Plot 3D of power lines sections

        Args:
            fig (go.Figure): plotly figure where new traces has to be added
        """
        
        plot_line(fig, self.section.get_coord())
        
        support_points = get_support_points(self.section.data)
        plot_support(fig, support_points)


    