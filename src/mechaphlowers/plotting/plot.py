# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore

if TYPE_CHECKING:
	from mechaphlowers.entities import SectionDataFrame


def plot_line(fig: go.Figure, points: np.ndarray) -> None:
	"""Plot the points of the cable onto the figure given

	Args:
	    fig (go.Figure): plotly figure
	    points (np.ndarray): points of all the cables of the section in point format (3 x n)
	"""
	fig.add_trace(
		go.Scatter3d(
			x=points[:, 0],
			y=points[:, 1],
			z=points[:, 2],
			mode="lines+markers",
			marker=dict(size=5),
			line=dict(width=8, color="red"),
		)
	)


def plot_support(fig: go.Figure, points: np.ndarray) -> None:
	"""Plot the points of the support onto the figure given

	Args:
	    fig (go.Figure): plotly figure
	    points (np.ndarray): points of all the supports of the section in point format (3 x n)
	"""
	fig.add_trace(
		go.Scatter3d(
			x=points[:, 0],
			y=points[:, 1],
			z=points[:, 2],
			mode="lines+markers",
			marker=dict(size=5),
			line=dict(width=8, color="green"),
		)
	)


def plot_insulator(fig: go.Figure, points: np.ndarray) -> None:
	"""Plot the points of the insulators onto the figure given

	Args:
	    fig (go.Figure): plotly figure
	    points (np.ndarray): points of all the insulators of the section in point format (3 x n)
	"""
	fig.add_trace(
		go.Scatter3d(
			x=points[:, 0],
			y=points[:, 1],
			z=points[:, 2],
			mode="lines+markers",
			marker=dict(size=5),
			line=dict(width=8, color="orange"),
		)
	)


def get_support_points(data: pd.DataFrame) -> np.ndarray:
	"""function to plot simple support with ground, attachment and crossarm.

	Args:
	    data (pd.DataFrame): SectionArray or SectionDataFrame data property

	Returns:
	    np.ndarray: 3 x (3 x 2  segment to plot x N) with N number of support point for the data input
	    Warning: every support is followed by a nan line to separate the traces on figure
	"""

	x = np.pad(np.cumsum(data.span_length.to_numpy()[:-1]), (1, 0), "constant")
	init_xshape = len(x)
	y = np.zeros_like(x)
	z = np.zeros_like(x)

	# get support points
	# ground points
	pp0 = np.vstack([x, y, z])

	# up points
	pp_up = pp0.copy()
	alt = data.conductor_attachment_altitude.to_numpy()
	pp_up[2, :] = alt

	# crossarm points
	pp_arm = pp_up.copy()
	lateral_shift = data.crossarm_length.to_numpy()
	pp_arm[1, :] = lateral_shift

	# insulators set points
	pp_insulator = pp_arm.copy()
	altitude_shift = data.insulator_length.to_numpy()
	pp_insulator[2, :] += -altitude_shift

	# add nan to separate
	pp_final = np.concat(
		[
			pp0.T,
			pp_up.T,
			np.nan * pp0.T,
			pp_up.T,
			pp_arm.T,
			np.nan * pp0.T,
		],
		axis=1,
	)

	# initxshape x 3 because each segment is composed by 3 points (pp0, pp_up, nan) x 2 because 2 segments, 3 for x,y,z
	return pp_final.reshape(init_xshape * 3 * 2, 3)


def get_insulator_points(data: pd.DataFrame) -> np.ndarray:
	"""function to plot very simple 2-points-insulator with crossarm and attachment down.

	Args:
	    data (pd.DataFrame): SectionArray or SectionDataFrame data property

	Returns:
	    np.ndarray: (3 x 1 segment to plot x N) and N number of support point for the data input
	    Warning: every support is followed by a nan line to separate the traces on figure
	"""

	x = np.pad(np.cumsum(data.span_length.to_numpy()[:-1]), (1, 0), "constant")
	init_xshape = len(x)
	y = np.zeros_like(x)
	z = np.zeros_like(x)

	# get support points
	# ground points
	pp0 = np.vstack([x, y, z])
	# move to altitude
	alt = data.conductor_attachment_altitude.to_numpy()
	pp0[2, :] = alt
	# move at the end of crossarm
	lateral_shift = data.crossarm_length.to_numpy()
	pp0[1, :] = lateral_shift

	# insulators set end points
	pp_insulator = pp0.copy()
	altitude_shift = data.insulator_length.to_numpy()
	pp_insulator[2, :] += -altitude_shift

	# add nan to separate
	pp_final = np.concat(
		[
			pp0.T,
			pp_insulator.T,
			np.nan * pp0.T,
		],
		axis=1,
	)
	# initxshape x 3 because each segment is composed by 3 points (pp0, pp_up, nan) x 1 because 1 segment, 3 for x,y,z
	return pp_final.reshape(init_xshape * 3 * 1, 3)


def set_layout(fig: go.Figure, auto: bool = True) -> None:
	"""set_layout

	Args:
	    fig (go.Figure): plotly figure where layout has to be updated
	    auto (bool, optional): Automatic layout based on data (scale respect). False means manual with an aspectradio of x=1, y=.05, z=.5. Defaults to True.
	"""

	# Check input
	auto = bool(auto)
	aspect_mode: str = "data" if auto else "manual"
	zoom: float = (
		0.1 if auto else 1
	)  # perhaps this approx of the zoom will not be adequate for all cases
	aspect_ratio: Dict = dict(x=1, y=0.05, z=0.5)

	fig.update_layout(
		scene=dict(
			aspectratio=aspect_ratio,
			aspectmode=aspect_mode,
			camera=dict(
				up=dict(x=0, y=0, z=1),
				eye=dict(
					x=0,
					y=1 / zoom,
					z=0,
				),
			),
		)
	)


class PlotAccessor:
	"""First accessor class for plots."""

	def __init__(self, section: SectionDataFrame):
		self.section: SectionDataFrame = section

	def line3d(
		self, fig: go.Figure, view: Literal["full", "analysis"] = "full"
	) -> None:
		"""Plot 3D of power lines sections

		Args:
		    fig (go.Figure): plotly figure where new traces has to be added
		    view (Literal['full', 'analysis'], optional): full for scale respect view, analysis for compact view. Defaults to "full".

		Raises:
		    ValueError: view is not an expected value
		"""

		view_map = {"full": True, "analysis": False}

		try:
			_auto = view_map[view]
		except KeyError:
			raise ValueError(
				f"{view=} : this argument has to be set to 'full' or 'analysis'"
			)

		plot_line(fig, self.section.get_coord())

		support_points = get_support_points(self.section.data)
		plot_support(fig, support_points)

		insulator_points = get_insulator_points(self.section.data)
		plot_insulator(fig, insulator_points)

		set_layout(fig, auto=_auto)
