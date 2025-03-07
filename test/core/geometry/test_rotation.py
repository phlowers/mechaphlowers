import numpy as np
from scipy.spatial.transform import Rotation as R

from mechaphlowers.core.geometry.rotation import (
	rotation,
	hamilton_array,
	rotation_multiple,
)


def test_roation_0():
	vector = np.array([0, 1, 0])
	beta = 90

	vector_rotated, vector_rotated_scipy = rotation(vector, beta)


def test_hamilton_array():
	q1 = np.array([[0.1, 0.2, 0.3, 0.4], [1] * 4])
	q2 = q1 * 2

	hamilton_array(q1, q2)


def test_roation_multiple_0():
	vector = np.array([[0, 1, 1], [0, 0.5, 0]])
	beta = np.array([90, 180])

	vector_rotated = rotation_multiple(vector, beta)


def test_roation_multiple_1():
	vector = np.array([[0, 1, 1], [0, 0.5, 0], [0, 3, 4]])
	beta = np.array([90, 180, 45])

	vector_rotated = rotation_multiple(vector, beta)
