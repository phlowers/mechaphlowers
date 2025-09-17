# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np

from mechaphlowers.core.papoto.papoto_model import papoto


def test_papoto_function_0():
    a = np.array([460.296269689673, np.nan], dtype = np.float32)
    HG = np.array([0.0, np.nan], dtype = np.float32)
    VG = np.array([95.3282575966228, np.nan], dtype = np.float32)
    HD = np.array([165.995104445107, np.nan], dtype = np.float32)
    VD = np.array([76.6693939496654, np.nan], dtype = np.float32)
    H1 = np.array([4.0612735796946, np.nan], dtype = np.float32)
    V1 = np.array([94.4241159093392, np.nan], dtype = np.float32)
    H2 = np.array([15.2623371798393, np.nan], dtype = np.float32)
    V2 = np.array([88.8639691159579, np.nan], dtype = np.float32)
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
    np.testing.assert_allclose(parameter, np.array([2000, np.nan]), atol=1.0)


def test_papoto_many_times():
    a = np.array([460.296269689673, np.nan], dtype=np.float32)
    HG = np.array([0.0, np.nan], dtype=np.float32)
    VG = np.array([95.3282575966228, np.nan], dtype=np.float32)
    HD = np.array([165.995104445107, np.nan], dtype=np.float32)
    VD = np.array([76.6693939496654, np.nan], dtype=np.float32)
    H1 = np.array([4.0612735796946, np.nan], dtype=np.float32)
    V1 = np.array([94.4241159093392, np.nan], dtype=np.float32)
    H2 = np.array([15.2623371798393, np.nan], dtype=np.float32)
    V2 = np.array([88.8639691159579, np.nan], dtype=np.float32)

    nb_loops = 10000
    for _ in range(nb_loops):
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


def test_papoto_many_spans():
    nb_spans = 1000000
    a = np.array([460.296269689673] * nb_spans, dtype=np.float32)
    HG = np.array([0.0] * nb_spans, dtype=np.float32)
    VG = np.array([95.3282575966228] * nb_spans, dtype=np.float32)
    HD = np.array([165.995104445107] * nb_spans, dtype=np.float32)
    VD = np.array([76.6693939496654] * nb_spans, dtype=np.float32)
    H1 = np.array([4.0612735796946] * nb_spans, dtype=np.float32)
    V1 = np.array([94.4241159093392] * nb_spans, dtype=np.float32)
    H2 = np.array([15.2623371798393] * nb_spans, dtype=np.float32)
    V2 = np.array([88.8639691159579] * nb_spans, dtype=np.float32)


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
    assert True


def print_all(name, arr):
    print_dim(name, arr)
    print_contiguous(name, arr)
    print_dtype(name, arr)
 
def print_contiguous(name, arr):
    print(
        f"name:{name}, shape:{arr.shape}, contiguous:{arr.data.contiguous}, C:{arr.data.c_contiguous}, F:{arr.data.f_contiguous}")
    if not arr.data.c_contiguous:
        raise Exception("ERROR NOT C CONTIGUOUS !!!!!!!!!")
 
def print_dim(name, arr):
    print(f"name:{name}, shape:{arr.shape}")
 
def print_dtype(name, arr):
    print(f"name:{name}, dtype:{arr.dtype}")
    if arr.dtype is not np.dtype(np.float32):
        raise Exception("ERROR NOT FLOAT 32 !!!!!!!!!")