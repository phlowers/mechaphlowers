# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0



import numpy as np


def points_distance_inside_plane(point_1, point_2, plane_normal, line_direction_normalized, u_plane, v_plane):    

    # Calculate difference vector
    diff_vector = point_1 - point_2
    print(f"\nDifference vector: ({diff_vector[0]:.4f}, {diff_vector[1]:.4f}, {diff_vector[2]:.4f})")
    
    # Calculate 3D distance
    distance_3d = np.linalg.norm(diff_vector)
    
    # Project onto plane frame basis vectors
    # u_plane: first basis vector in the plane (y-direction in plane frame)
    # v_plane: second basis vector in the plane (z-direction in plane frame)
    distance_projection_u = np.dot(diff_vector, u_plane)
    distance_projection_v = np.dot(diff_vector, v_plane)
    
    # Verify: the projections should satisfy Pythagorean theorem
    projected_distance = np.sqrt(distance_projection_u**2 + distance_projection_v**2)
    if abs(projected_distance - distance_3d) > 1e-6:
        raise ValueError(f"Projected distance ({projected_distance:.6f} m) does not match 3D distance ({distance_3d:.6f} m) - check calculations!")

    return distance_3d, distance_projection_u, distance_projection_v

def get_projection_points(obstacle, projection_u, projection_v, u_plane, v_plane):
    
    u_projection_point = obstacle + projection_u * u_plane
    v_projection_point = obstacle + projection_v * v_plane
    return u_projection_point, v_projection_point