# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.models.cable.deformation import DeformationRte
from mechaphlowers.core.models.cable.span import (
    CatenarySpan,
)
from mechaphlowers.entities.data_container import DataContainer


def test_deformation_values__default_data(
    default_data_container_one_span: DataContainer,
) -> None:
    span_model = CatenarySpan(**default_data_container_one_span.__dict__)
    tension_mean = span_model.T_mean()
    cable_length = span_model.L()

    deformation_model = DeformationRte(
        **default_data_container_one_span.__dict__,
        data_cable=default_data_container_one_span.data_cable,
        tension_mean=tension_mean,
        cable_length=cable_length,
    )
    deformation_model.current_temperature = np.array([30, 30])
    eps_total = deformation_model.epsilon()
    L_ref = deformation_model.L_ref()

    # Data given by the prototype
    np.testing.assert_allclose(
        eps_total,
        np.array([0.00093978 + 0.000345, np.nan]),
        atol=1e-6,
    )
    # our method L_ref returns L_15 but proto returns L_0 so that's why 480.6392123 is not the displayed value if you are using proto

    np.testing.assert_allclose(
        L_ref,
        np.array([480.6392123, np.nan]),
        atol=1e-6,
    )
    # Z - narcisse
    # first: L0 = 480.659
    # CRA 50% L0 = 480.649
    # récup epsilon plutôt?
