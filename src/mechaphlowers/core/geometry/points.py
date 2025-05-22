# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from mechaphlowers.core.geometry.rotation import rotation_quaternion_same_axis


class Points:
    def __init__(self, coords: np.ndarray, frame: np.ndarray):
        self.coords: np.ndarray = coords
        self.frame: np.ndarray = frame
        
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