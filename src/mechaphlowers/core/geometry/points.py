# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from mechaphlowers.core.geometry.rotation import rotation_quaternion, rotation_quaternion_same_axis






class Frame:
    def __init__(self, origin: np.ndarray = np.full((4,3),np.zeros(3)), axis: np.ndarray = np.full((4,3,3),np.eye(3))):
       self.origin: np.ndarray = origin
       self.axis: np.ndarray = axis
    
    @property   
    def x_axis(self):
        return self.axis[:,0]
        
    @property
    def y_axis(self):
        return self.axis[:,1]
       
    @property
    def z_axis(self):
        return self.axis[:,2]
    
    def unflatten(self, flattened_points: np.ndarray):
        return flattened_points.reshape(self.axis.shape)
    
    def translate_all(self, translation_vector: np.ndarray):
        self.coords += translation_vector
        
    def rotate_one_angle_per_layer(self, line_angles_layer: np.ndarray, rotation_axes: np.ndarray):
        line_angles_points = np.repeat(line_angles_layer, self.axis.shape[1], axis=0)
        rotation_axes_array = np.repeat(rotation_axes, self.axis.shape[1], axis=0)
        flattened_points = self.axis.reshape(-1,3) # to refector later with flatten method
        coords_rotated = rotation_quaternion(flattened_points, line_angles_points, rotation_axes_array)
        self.axis = self.unflatten(coords_rotated)

class Points:
    def __init__(self, coords: np.ndarray, frame: Frame = Frame()):
        self.coords: np.ndarray = coords.copy() # give copy
        self.frame: Frame = frame
        # check if coords and frame are compatible
        
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
    
    def transform(self, translation_vector: np.ndarray, line_angle: float, rotation_axis: np.ndarray= np.array([0,0,1])):
        self.translate_all(translation_vector)
        self.rotate_one_angle_per_layer(line_angle, rotation_axis)
        
    
    def transform_frame(self, translation_vector: np.ndarray, line_angle: float, rotation_axis: np.ndarray= np.array([0,0,1])):
        self.frame.translate(translation_vector)
        self.frame.rotate(line_angle, rotation_axis)
        self.transform(-translation_vector, -line_angle, rotation_axis)