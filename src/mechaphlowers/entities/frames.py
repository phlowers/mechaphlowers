# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from mechaphlowers.core.geometry import references
from mechaphlowers.core.models.cable_models import CatenaryCableModel, GeometricCableModel
from mechaphlowers.entities.arrays import SectionArray



# This parameter has to be removed later.
# This is the default resolution for spans when exporting coordinates in get_coords
RESOLUTION: int = 10


class _SectionFrame:
    """ SectionFrame object is the top api object of the library. 
    
    Inspired from dataframe, it is designed to handle the datas and the models.
    TODO: for the moment the initialization with SectionArray and GeometricCableModel is explicit.
    It is not intended to be later.
    """

    def __init__(self, section: SectionArray, span_model : GeometricCableModel = CatenaryCableModel):
        self.section: SectionArray = section
        self.span_model: GeometricCableModel = span_model
        

    def get_coord(self) -> np.ndarray:
        """Get x,y,z cables coordinates

        Warning here : for the moment the code inside is calculating

        Returns:
            _type_: _description_

        Returns:
            np.ndarray: _description_
        """

        spans = self.span_model(
            self.section.data.span_length.to_numpy(), 
            self.section.data.elevation_difference.to_numpy(), 
            self.section.data.sagging_parameter.to_numpy()
            )

        # compute x_axis
        x_cable: np.ndarray = spans.x(RESOLUTION)

        # compute z_axis
        z_cable: np.ndarray = spans.z(x_cable)

        # change frame and drop last value
        x_span, y_span, z_span = references.cable2span(x_cable[:,:-1], z_cable[:,:-1], beta = 0)

        altitude: np.ndarray = self.section.data.conductor_attachment_altitude.to_numpy()
        span_length: np.ndarray = self.section.data.span_length.to_numpy()
        
        #TODO: the content of this function is not generic enough. An upcoming feature will change that.
        x_span, z_span = references.translate_cable_to_span(x_span, z_span, altitude, span_length)
        
        # dont forget to flatten the arrays and stack in a 3xNpoints array
        return np.vstack([x_span.T.reshape(-1), y_span.T.reshape(-1), z_span.T.reshape(-1)]).T
    
    @property
    def data(self):
        """data property to return SectionArray data property

        Returns:
            np.ndarray: SectionArray data from input 
        """
        return self.section.data
