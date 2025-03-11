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


def rotation_matrix_to_quaternion(
	beta: np.ndarray, rotation_axes: np.ndarray
) -> np.ndarray:
	"""Create rotation matrix for quaternion rotation.
	One rotation vector equals to: [cos(beta/2), sin(beta/2)*u_x, sin(beta/2)*u_y, sin(beta/2)*u_z]
	where unit_vector = [u_x, u_y, u_z].
	unit_vector is rotation_axes that has been normalized

	Args:
		beta (np.ndarray): array of angles in degrees [beta_0, beta_1]
		rotation_axes (np.ndarray): array of axes of rotation in 3D (will be normalized) [[r_x0, r_y0, r_z0], [r_x1, r_y1, r_z1], ...]

	Returns:
		np.ndarray: [[w0, x0, y0, z0], [w1, x1, y1, z1],...]

	>>> beta = [90, 180]
	>>> rotation_axes = np.array([[1, 0, 0], [0, 0, 1]])
	>>> rotation_matrix_to_quaternion(beta, rotation_axes)
	array([[0.707106781, 0.707106781, 0, 0], [0, 0, 0, 1]])
	"""
	beta_rad = np.radians(beta)
	# normalize the rotation axis
	unit_vector = (
		rotation_axes / np.linalg.norm(rotation_axes, axis=1)[:, np.newaxis]
	)

	# C = [[cos(beta_0/2)], [cos(beta_1/2)],...]
	C = np.cos(beta_rad / 2)[:, np.newaxis]
	S = np.sin(beta_rad / 2)[:, np.newaxis]
	x, y, z = np.split(unit_vector, 3, axis=-1)

	quat = np.concatenate((C, S * x, S * y, S * z), axis=-1)
	return quat


def rotation_quaternion_same_axis(
	vector: np.ndarray,
	beta: np.ndarray,
	rotation_axis: np.ndarray = np.array([1, 0, 0]),
) -> np.ndarray:
	"""Compute rotation of vector using quaternion rotation.
	All vectors are rotated around the same axis.

	Args:
		vector (np.ndarray): array of 3D points to rotate [[x0, y0, z0], [x1, y1, z1],...]
		beta (np.ndarray): array of angles in degrees [beta_0, beta_1, ...]
		rotation_axis (np.ndarray): single axis of rotation in 3D. Doesn't need to be normalized beforehand [r_x0, r_y0, r_z0]

	Returns:
		np.ndarray: array of new points that have been rotated by angles beta around rotation_axis

	>>> vector = np.array([[0, 1, 0], [0, 1, 0]])
	>>> beta = np.array([90, 45])
	>>> rotation_axis = np.array([1, 0, 0])
	>>> rotation_quaternion_same_axis(vector, beta, rotation_axis)
	array([[0,  0,  1], [0,  0.707106781,  0.707106781]])
	"""

	rotation_axes = np.full(vector.shape, rotation_axis)
	return rotation_quaternion(vector, beta, rotation_axes)


def rotation_quaternion(
	vector: np.ndarray,
	beta: np.ndarray,
	rotation_axes: np.ndarray,
) -> np.ndarray:
	"""Compute rotation of vector using quaternion rotation.

	Args:
		vector (np.ndarray): array of 3D points to rotate [[x0, y0, z0], [x1, y1, z1],...]
		beta (np.ndarray): array of angles in degrees [beta_0, beta_1, ...]
		rotation_axes (np.ndarray): array of axes of rotation in 3D. Doesn't need to be normalized beforehand [[r_x0, r_y0, r_z0], [r_x1, r_y1, r_z1], ...]

	Returns:
		np.ndarray: array of new points that have been rotated by angles beta around rotation_axes

	>>> vector = np.array([[0, 1, 0], [0, 1, 0]])
	>>> beta = np.array([90, -90])
	>>> rotation_axes = np.array([[1, 0, 0], [0, 0, 1]])
	>>> rotation_quaternion(vector, beta, rotation_axes)
	array([[ 0,  0,  1], [1,  0,  0]])
	"""
	# compute the rotation matrix as quaternion
	rotation_quaternion = rotation_matrix_to_quaternion(beta, rotation_axes)
	# compute the conjugate of the rotation matrix
	conj = np.full(rotation_quaternion.shape, [1, -1, -1, -1])
	rotation_quaternion_conj = rotation_quaternion * conj

	# add a zero w coordinate to vector to make it a quaternion
	w_coord = np.zeros((vector.shape[0], 1))
	purequat = np.concat((w_coord, vector), axis=1)

	# compute the new rotated quaternion:
	# vector_rotated = R * vector * R_conj
	vector_rotated = hamilton_product_array(
		rotation_quaternion,
		hamilton_product_array(purequat, rotation_quaternion_conj),
	)

	# remove w coordinate to be back in 3D
	vector_rotated_3d = vector_rotated[:, 1:]
	return vector_rotated_3d
