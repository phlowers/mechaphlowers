# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np

from mechaphlowers.core.papoto.papoto_model import papoto


def test_papoto_function_0():
    a = np.array([460.296269689673, np.nan])
    HG = np.array([0.0, np.nan])
    VG = np.array([95.3282575966228, np.nan])
    HD = np.array([165.995104445107, np.nan])
    VD = np.array([76.6693939496654, np.nan])
    H1 = np.array([4.0612735796946, np.nan])
    V1 = np.array([94.4241159093392, np.nan])
    H2 = np.array([15.2623371798393, np.nan])
    V2 = np.array([88.8639691159579, np.nan])
    parameter = papoto(
        a=a,
        HG=HG,
        VG=VG,
        HD=HD,
        VD=VD,
        H1=H1,
        V1=V1,
        H2=H2,
        V2=V2,
    )
    np.testing.assert_allclose(parameter, np.array([2000, np.nan]), atol=1.)
