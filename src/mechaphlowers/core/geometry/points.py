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
        return flattened_points.reshape(self.coords.shape)

    def rotate_one_angle_per_layer(self, line_angles_layer: np.ndarray, rotation_axes: np.ndarray):
        line_angles_points = np.repeat(line_angles_layer, self.coords.shape[1], axis=0)
        rotation_axes_array = np.repeat(rotation_axes, self.coords.shape[1], axis=0)
        flattened_points = self.flatten()
        coords_rotated = rotation_quaternion(flattened_points, line_angles_points, rotation_axes_array)
        self.coords = self.unflatten(coords_rotated)

    def rotate_one_angle_per_point(self, line_angles_points: np.ndarray, rotation_axes: np.ndarray):
        rotation_axes_array = np.repeat(rotation_axes, self.coords.shape[1], axis=0)
        flattened_points = self.flatten()
        coords_rotated = rotation_quaternion(flattened_points, line_angles_points, rotation_axes_array)
        self.coords = self.unflatten(coords_rotated)
    
    def rotate_same_axis(self, line_angles_layer: np.ndarray, rotation_axis: np.ndarray):
        line_angle_array = np.repeat(line_angles_layer, self.coords.shape[1], axis=0)
        flattened_points = self.flatten()
        coords_rotated = rotation_quaternion_same_axis(flattened_points, line_angle_array, rotation_axis)
        self.coords = self.unflatten(coords_rotated)

    def nan_points(self):
        a,b,c = self.coords.shape
        return np.hstack([self.coords, np.nan*np.ones((a,1,c))])
       
    def coords_for_plot(self):
        return self._flatten(self.nan_points())