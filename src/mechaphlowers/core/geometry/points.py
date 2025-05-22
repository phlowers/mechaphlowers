# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from mechaphlowers.core.geometry.rotation import rotation_quaternion, rotation_quaternion_same_axis



class Frame:
    pass

class Points:
    def __init__(self, coords: np.ndarray, frame: Frame = Frame()):
        self.coords: np.ndarray = coords.copy() # give copy
        self.frame: Frame = frame
        
    def translate_all(self, translation_vector: np.ndarray):
        self.coords += translation_vector
    
    def translate_layer(self, translation_vector: np.ndarray, layer: int):
        self.coords[layer] += translation_vector
        
    def translate_point(self, translation_vector: np.ndarray, layer: int, index: tuple):
        self.coords[layer,index] += translation_vector
    
    def flatten(self):
        return self._flatten(self.coords)
    
    @staticmethod
    def _flatten(points):
        return points.reshape(-1,3)
        
    def unflatten(self, flattened_points: np.ndarray):
        self.coords = flattened_points.reshape(self.coords.shape)

    def rotate(self, line_angles: np.ndarray, rotation_axes: np.ndarray):
        flattened_points = self.flatten()
        angles_packed = np.ones((self.coords.shape[1], self.coords.shape[0])) * line_angles
        line_angle_array = angles_packed.T.reshape(-1)
        coords_rotated = rotation_quaternion(flattened_points, line_angle_array, rotation_axes)
        self.points = self.unflatten(coords_rotated)