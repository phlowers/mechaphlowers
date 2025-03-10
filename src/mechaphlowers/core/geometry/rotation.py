import numpy as np


def hamilton_array(quaternion0, quaternion1):
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


def rotation_matrix_quaternion(beta: np.ndarray, rotation_axis: np.ndarray):
	beta_rad = np.radians(beta)
	C = np.cos(beta_rad / 2)[:, np.newaxis]
	S = np.sin(beta_rad / 2)[:, np.newaxis]
	triple_sin = np.repeat(S, 3, axis=1)
	unit_vector = (
		rotation_axis / np.linalg.norm(rotation_axis, axis=1)[:, np.newaxis]
	)
	quat = np.concatenate((C, triple_sin * unit_vector), axis=1)
	return quat


def rotation_quaternion_same_axis(
	vector: np.ndarray,
	beta: np.ndarray,
	rotation_axis_single: np.ndarray = np.array([1, 0, 0]),
):
	# here vector is [[x0, y0, z0], [x1, y1, z1],...]

	# rotation_object = R.from_euler("x", beta, degrees=True)
	# rotation_quat_scipy = rotation_object.as_quat()
	# rotation_quat_scipy = np.roll(rotation_quat_scipy, 1, axis=1)
	rotation_axis_multiple = np.full(vector.shape, rotation_axis_single)
	return rotation_quaternion(vector, beta, rotation_axis_multiple)


def rotation_quaternion(
	vector: np.ndarray,
	beta: np.ndarray,
	rotation_axis: np.ndarray,
):
	# here vector is [[x0, y0, z0], [x1, y1, z1],...]

	rotation_quat = rotation_matrix_quaternion(beta, rotation_axis)
	# quat will be needed to be converted as unitary quaternion
	# rotation_quat = np.roll(rotation_quat, 1, axis=1)

	conj = np.full(rotation_quat.shape, [1, -1, -1, -1])
	quatconj = rotation_quat * conj  # hand conjugate

	w_coord = np.zeros((vector.shape[0], 1))
	purequat = np.concat((w_coord, vector), axis=1)

	vector_rotated = hamilton_array(
		rotation_quat, hamilton_array(purequat, quatconj)
	)

	vector_rotated_3d = vector_rotated[:, 1:]
	return vector_rotated_3d
