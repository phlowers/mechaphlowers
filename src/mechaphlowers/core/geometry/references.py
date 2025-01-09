# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore


def spans2vector(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
	"""spans2vector is a function that allows to stack x, y and z arrays into a single array

	spans are a n x d array where n is the number of points per span and d is the number of spans
	vector are a n x 3 array where n is the number of points per span and 3 is the number of coordinates

	Args:
	    x (np.ndarray): n x d array spans x coordinates
	    y (np.ndarray): n x d array spans y coordinates
	    z (np.ndarray): n x d array spans z coordinates

	Returns:
	    np.ndarray: 3 x n array vector coordinates
	"""

	cc = np.vstack([x.reshape(-1), y.reshape(-1), z.reshape(-1)]).T
	return cc


def cable2span(
	x: np.ndarray, z: np.ndarray, beta: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""cable2span cable to span is a function that allows to rotate from cable 2D plan to span 3D frame with an angle beta


	Args:
	    x (np.ndarray): n x d array spans x coordinates
	    z (np.ndarray): n x d array spans x coordinates
	    beta (float): angle rotation

	Returns:
	    Tuple[np.ndarray, np.ndarray, np.ndarray]:
	        - x_span (np.ndarray): Rotated x coordinates in the span 3D frame.
	        - y_span (np.ndarray): Rotated y coordinates in the span 3D frame.
	        - z_span (np.ndarray): Rotated z coordinates in the span 3D frame.
	"""

	# TODO: the function here move the whole section with beta angle. This is not the expected behavior when loads will be implemented

	init_shape = z.shape
	# Warning here, x and z are shaped as (n point per span, d span)
	# elevation part has the same shape
	# However rotation is applied on [x,y,z] stacked matrix with x vector of shape (n x d, )
	elevation_part = np.linspace(
		tuple(np.nanmin(z, axis=1).tolist()),
		tuple(np.nanmax(z, axis=1).tolist()),
		x.shape[1],
	).T

	rotation_matrix = R.from_euler("x", beta, degrees=True)
	# span = np.dot(rotation_matrix, np.array([x, elevation_part, z]))

	vector = spans2vector(x, 0 * x, z - elevation_part)

	span = rotation_matrix.apply(vector)

	x_span, y_span, z_span = (
		span[:, 0].reshape(init_shape),
		span[:, 1].reshape(init_shape),
		span[:, 2].reshape(init_shape),
	)

	z_span += elevation_part

	return x_span, y_span, z_span


def translate_cable_to_support(
	x_span: np.ndarray,
	y_span: np.ndarray,
	z_span: np.ndarray,
	altitude: np.ndarray,
	span_length: np.ndarray,
	crossarm_length: np.ndarray,
	insulator_length: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Translate cable using altitude and span length

	Args:
	    x_span (np.ndarray): x coordinates rotated
	    y_span (np.ndarray): y coordinates rotated
	    z_span (np.ndarray): z coordinates rotated
	    altitude (np.ndarray): conductor heigth altitude
	    span_length (np.ndarray): span length
	    crossarm_length (np.ndarray): crossarm length
	    insulator_length (np.ndarray): insulator length

	Returns:
	    Tuple[np.ndarray]: translated x_span, y_span and z_span
	"""

	# Note : for every data, we dont need the last support information
	# Ex : altitude = array([50., 40., 20., 10.]) -> altitude[:-1] = array([50., 40., 20.])
	# "move" the cable to the conductor attachment altitude
	z_span += -z_span[0, :] + altitude[:-1]
	# "move" the cables at the end of the arm
	y_span += crossarm_length[:-1]
	# "move down" the cables at the end of the insulator chain
	z_span += -insulator_length[:-1]
	# "move" each cable to the x coordinate of the hanging point
	x_span += -x_span[0, :] + np.pad(
		np.cumsum(span_length[:-2]), (1, 0), "constant"
	)
	# why pad ? cumsum(...) = array([100., 300.]) and we need a zero to start
	# pad(...) = array([0., 100., 300.])

	return x_span, y_span, z_span
