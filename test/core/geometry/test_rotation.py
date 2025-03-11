import numpy as np

from mechaphlowers.core.geometry.rotation import (
	hamilton_product_array,
	rotation_matrix_quaternion,
	rotation_quaternion,
	rotation_quaternion_same_axis,
)


def test_hamilton_array():
	q0 = np.array(
		[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 2, 0, 3]]
	)
	q1 = np.array(
		[[0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [-1, 3, 4, 0]]
	)

	result = hamilton_product_array(q0, q1)
	expected_result = np.array(
		[
			[0, 0, -1, 0],
			[0, 0, 0, -1],
			[0, 0, 1, 0],
			[-1, 0, 0, 0],
			[-7, -11, 13, 5],
		]
	)
	assert (result == expected_result).all()


def test_rotation_matrix_quaternion__simple_axis():
	beta = np.array([90, 180])
	rotation_matrix_0 = rotation_matrix_quaternion(
		beta, np.array([[1, 0, 0], [0, 2, 0]])
	)
	expected_result = np.array(
		[
			[2 ** (-0.5), 2 ** (-0.5), 0, 0],
			[0, 0, 1, 0],
		]
	)
	np.testing.assert_array_almost_equal(rotation_matrix_0, expected_result)


def test_rotation_multiple__same_axis():
	vector_0 = np.array(
		[
			[0, 1, 1],
			[0, 0.5, 0],
			[7.1, 3.8, 6.5],
			[0, 0, 2**0.5],
			[0, -(3**0.5), 0],
		]
	)
	beta_0 = np.array([90, 180, 360, 45, 30])

	vector_rotated_0 = rotation_quaternion_same_axis(vector_0, beta_0)
	expected_result_0 = np.array(
		[
			[0, -1, 1],
			[0, -0.5, 0],
			[7.1, 3.8, 6.5],
			[0, -1, 1],
			[0, -1.5, (-(3**0.5)) / 2],
		]
	)
	np.testing.assert_array_almost_equal(vector_rotated_0, expected_result_0)

	vector_1 = np.array([[7.1, 3.8, 6.5], [1, 0, 0], [1, 0, 0]])
	beta_1 = np.array([360, -60, 270])
	axis_1 = np.array([0, 0, 1])
	vector_rotated_1 = rotation_quaternion_same_axis(vector_1, beta_1, axis_1)
	expected_result_1 = np.array(
		[[7.1, 3.8, 6.5], [0.5, -(3**0.5) / 2, 0], [0, -1, 0]]
	)
	np.testing.assert_array_almost_equal(vector_rotated_1, expected_result_1)


def test_rotation_multiple__different_axis():
	vector = np.array(
		[
			[1, 5, 1],
			[2, 0, 3],
			[0, 1, 0],
			[0, 1, 0],
			[0, 0, 1],
		]
	)
	beta = np.array([90, 180, 180, 180, 180])
	axes = np.array(
		[[1, 0, 0], [0, 1, 0], [1, 1, 0], [8, 8, 0], [0, 3**0.5, 1]]
	)
	vector_rotated = rotation_quaternion(vector, beta, axes)
	expected_result = np.array(
		[
			[1, -1, 5],
			[-2, 0, -3],
			[1, 0, 0],
			[1, 0, 0],
			[0, (3**0.5) / 2, -0.5],
		]
	)
	np.testing.assert_array_almost_equal(vector_rotated, expected_result)
