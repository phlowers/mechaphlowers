# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
from pytest import fixture

import mechaphlowers.core.models.balance.functions as f
from mechaphlowers.core.models.balance.elements import (
    Cable,
    Nodes,
    Orchestrator,
)


@fixture
def cable_AM600():
    return Cable(600.4e-6, 17.658, 0.000023, 60e9, 31.86e-3, 320)


@fixture
def section_3d_simple(cable_AM600) -> Orchestrator:
    nodes = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000.0, 500.0, 500.0, 1000.0]),
        arm_length=np.array([0, 0, 0, 0]),
        line_angle=f.grad_to_rad(np.array([0, 0, 0, 0])),
        span_length=np.array([500, 300, 400]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([0, 0, 0]),
        load_position=np.array([0, 0, 0]),
    )

    return Orchestrator(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes,
        cable=cable_AM600,
    )


@fixture
def section_3d_angles_arm(cable_AM600) -> Orchestrator:
    nodes = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000.0, 500.0, 500.0, 1000.0]),
        arm_length=np.array([0, 10, -10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 10, 0, 0])),
        span_length=np.array([500, 300, 400]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([0, 0, 0]),
        load_position=np.array([0, 0, 0]),
    )

    return Orchestrator(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes,
        cable=cable_AM600,
    )


def test_element_initialisation(section_3d_simple: Orchestrator):
    # load = section_2d_note.nodes.load

    print("\n")
    print(section_3d_simple.balance_model)
    print(section_3d_simple.balance_model.nodes)


def test_element_change_state(section_3d_simple: Orchestrator):
    section_3d_simple.solve_adjustment()

    section_3d_simple.solve_change_state()
    assert True
