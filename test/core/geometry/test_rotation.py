import numpy as np

from scipy.spatial.transform import Rotation as R  # type: ignore

from mechaphlowers.core.geometry.rotation import (
	hamilton_array,
	rotation_matrix_quaternion,
	rotation_quaternion_same_axis,
)


def test_hamilton_array():
	q1 = np.array([[0.1, 0.2, 0.3, 0.4], [1] * 4])
	q2 = q1 * 2

	hamilton_array(q1, q2)


def test_rotation_matrix_quaternion():
	beta = np.array([90, 180])
	rotation_matrix_0 = rotation_matrix_quaternion(
		beta, np.array([[1, 0, 0], [0, 2, 0]])
	)
	# rotation_matrix_2 = rotation_quaternion_same_axis(
	# 	beta, np.array([1, 1, 0])
	# )

	# quaternion rotation vector result using scipy.from_euler
	expected_result = np.array(
		[
			[2 ** (-1 / 2), 2 ** (-1 / 2), 0, 0],
			[0, 0, 1, 0],
		]
	)
	np.testing.assert_array_almost_equal(rotation_matrix_0, expected_result)


def test_rotation_multiple__same_axis_0():
	vector = np.array([[0, 1, 1], [0, 0.5, 0], [1, 0, 0], [0, 0, 7]])
	beta = np.array([90, 180, 90, 90])

	vector_rotated = rotation_quaternion_same_axis(vector, beta)
	expected_result = np.array(
		[
			[0, -1, 1],
			[0, -0.5, 0],
			[1, 0, 0],
			[0, -7, 0],
		]
	)
	np.testing.assert_array_almost_equal(vector_rotated, expected_result)
