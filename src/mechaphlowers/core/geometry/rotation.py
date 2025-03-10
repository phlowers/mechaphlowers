import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore


def hamilton(q1, q2):
	a1, b1, c1, d1 = q1
	a2, b2, c2, d2 = q2

	return np.array(
		[
			a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
			a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
			a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
			a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
		]
	)


def hamilton_array(quaternion0, quaternion1):
	x0, y0, z0, w0 = np.split(quaternion0, 4, axis=-1)
	x1, y1, z1, w1 = np.split(quaternion1, 4, axis=-1)
	return np.concatenate(
		(
			x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
			-x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
			x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
			-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
		),
		axis=-1,
	)


def rotation(vector: np.ndarray, beta: float):
	rotation_object = R.from_euler("x", beta, degrees=True)
	quat = rotation_object.as_quat()
	quat[0], quat[1], quat[2], quat[3] = quat[3], quat[0], quat[1], quat[2]
	conj = np.array([1, -1, -1, -1])
	quatconj = quat * conj  # hand conjugate
	purequat = np.array([0, vector[0], vector[1], vector[2]])
	vector_rotated = hamilton(quat, hamilton(purequat, quatconj))
	vector_rotated_scipy = rotation_object.apply(vector)
	return vector_rotated, vector_rotated_scipy


def rotation_matrix_quaternion(beta: np.ndarray, rotation_axis: np.ndarray):
	beta_rad = np.radians(beta)
	C = np.cos(beta_rad / 2)[:, np.newaxis]
	S = np.sin(beta_rad / 2)[:, np.newaxis]
	triple_sin = np.repeat(S, 3, axis=1)
	unit_vector = np.full(triple_sin.shape, rotation_axis)
	quat = np.concatenate((C, triple_sin * unit_vector), axis=1)
	return quat


def rotation_multiple(
	vector: np.ndarray,
	beta: np.ndarray,
	rotation_axis: np.ndarray = np.array([1, 0, 0]),
):
	# here vector is [[x0, y0, z0], [x1, y1, z1],...]

	# TODO: write a way to create rotation vector using quaternion by hand
	# rotation_object = R.from_euler("x", beta, degrees=True)
	# rotation_quat_scipy = rotation_object.as_quat()

	rotation_quat = rotation_matrix_quaternion(beta, rotation_axis)
	# quat will be needed to be converted as unitary quaternion

	rotation_quat = np.roll(rotation_quat, -1, axis=1)
	conj = np.full(rotation_quat.shape, [1, -1, -1, -1])
	quatconj = rotation_quat * conj  # hand conjugate

	w_coord = np.zeros((vector.shape[0], 1))
	purequat = np.concat((w_coord, vector), axis=1)

	vector_rotated = hamilton_array(
		rotation_quat, hamilton_array(purequat, quatconj)
	)

	vector_rotated_3d = vector_rotated[:, 1:]
	return vector_rotated_3d
