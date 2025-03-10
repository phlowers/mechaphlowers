import numpy as np


def hamilton_product_array(
	quaternion0: np.ndarray, quaternion1: np.ndarray
) -> np.ndarray:
	"""Hamilton product for array of quaternions. Product is not commutative, order of quaternions is important.

	Args:
		quaternion0 (np.ndarray): [[w0_0, x0_0, y0_0, z0_0], [w0_1, x0_1, y0_1, z0_1],...]
		quaternion1 (np.ndarray): [[w1_0, x1_0, y1_0, z1_0], [w1_1, x1_1, y1_1, z1_1],...]
	"""
	w0, x0, y0, z0 = np.split(quaternion0, 4, axis=-1)
	w1, x1, y1, z1 = np.split(quaternion1, 4, axis=-1)
	return np.concatenate(
		(
			w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
			w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
			w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1,
			w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1,
		),
		axis=-1,
	)


def rotation_matrix_quaternion(
	beta: np.ndarray, rotation_axis: np.ndarray
) -> np.ndarray:
	"""Create rotation matrix for quaternion rotation.
	One rotation vector equals to: [cos(beta/2), sin(beta/2)*x, sin(beta/2)*y, sin(beta/2)*z]
	where rotation_axis = [x, y, z]

	Args:
		beta (np.ndarray): array of angles in degrees
		rotation_axis (np.ndarray): array of axis of rotation (no need to normalize)

	Returns:
		np.ndarray: [[w0, x0, y0, z0], [w1, x1, y1, z1],...]
	"""
	beta_rad = np.radians(beta)
	# normalize the rotation axis
	unit_vector = (
		rotation_axis / np.linalg.norm(rotation_axis, axis=1)[:, np.newaxis]
	)

	# C =[[cos(beta_0/2)], [cos(beta_1/2)],...]
	C = np.cos(beta_rad / 2)[:, np.newaxis]
	S = np.sin(beta_rad / 2)[:, np.newaxis]
	# triple_sin = [[sin(beta_0/2), sin(beta_0/2), sin(beta_0/2)], [sin(beta_1/2), sin(beta_1/2), sin(beta_1/2)],...]
	triple_sin = np.repeat(S, 3, axis=1)

	quat = np.concatenate((C, triple_sin * unit_vector), axis=1)
	return quat


def rotation_quaternion_same_axis(
	vector: np.ndarray,
	beta: np.ndarray,
	rotation_axis_single: np.ndarray = np.array([1, 0, 0]),
) -> np.ndarray:
	"""Compute rotation of vector using quaternion rotation.
	All vectors are rotated around the same axis."""

	rotation_axis_multiple = np.full(vector.shape, rotation_axis_single)
	return rotation_quaternion(vector, beta, rotation_axis_multiple)


def rotation_quaternion(
	vector: np.ndarray,
	beta: np.ndarray,
	rotation_axis: np.ndarray,
) -> np.ndarray:
	"""Compute rotation of vector using quaternion rotation.

	Args:
		vector (np.ndarray): array of 3D points to rotate [[x0, y0, z0], [x1, y1, z1],...]
		beta (np.ndarray): array of angles in degrees
		rotation_axis (np.ndarray): array of axes of rotation (no need to normalize)

	Returns:
		np.ndarray: array of new points that have been rotated by angles beta around axes rotation_axis
	"""
	# compute the rotation matrix as quaternion
	rotation_matrix = rotation_matrix_quaternion(beta, rotation_axis)
	# compute the conjugate of the rotation matrix
	conj = np.full(rotation_matrix.shape, [1, -1, -1, -1])
	rotation_matrix_conj = rotation_matrix * conj

	# add a zero w coordinate to vector to make it a quaternion
	w_coord = np.zeros((vector.shape[0], 1))
	purequat = np.concat((w_coord, vector), axis=1)

	# compute the new rotated quaternion:
	# vector_rotated = R * vector * R_conj
	vector_rotated = hamilton_product_array(
		rotation_matrix, hamilton_product_array(purequat, rotation_matrix_conj)
	)

	# remove w coordinate to be back in 3D
	vector_rotated_3d = vector_rotated[:, 1:]
	return vector_rotated_3d
