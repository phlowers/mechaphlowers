# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from mechaphlowers.core.models.balance.engine import BalanceEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BalanceEngineMemento:
    """Immutable snapshot of a [`BalanceEngine`][mechaphlowers.core.models.balance.engine.BalanceEngine]'s mutable state.

    Every numpy array stored here is an independent copy (via ``.copy()``)
    so mutating the originals after saving will not affect the memento.
    """

    # --- nodes (solver primary state) ---
    nodes_dxdydz: np.ndarray

    # --- balance_model scalars / arrays ---
    parameter: np.ndarray
    sagging_temperature: np.ndarray
    adjustment: bool
    Th: np.ndarray
    Tv_d: np.ndarray
    Tv_g: np.ndarray
    a: np.ndarray
    b: np.ndarray

    # --- span_model geometry ---
    span_sagging_parameter: np.ndarray
    span_span_length: np.ndarray
    span_elevation_difference: np.ndarray
    span_load_coefficient: np.ndarray

    # --- cable_loads ---
    wind_pressure: np.ndarray
    ice_thickness: np.ndarray

    # --- deformation_model ---
    deformation_current_temperature: np.ndarray
    deformation_tension_mean: np.ndarray
    deformation_cable_length: np.ndarray

    # --- engine ---
    L_ref: Optional[np.ndarray]


class BalanceEngineCaretaker:
    """Dedicated class responsible for saving and restoring [`BalanceEngine`][mechaphlowers.core.models.balance.engine.BalanceEngine] state.

    Implements the *Caretaker* role of the Memento pattern.  The engine itself
    (the *Originator*) is not modified — all snapshot logic lives here.

    Args:
        engine (BalanceEngine): The balance engine whose state is managed.
    """

    def __init__(self, engine: BalanceEngine) -> None:
        self._engine = engine

    def save(self) -> BalanceEngineMemento:
        """Create an immutable snapshot of the engine's current mutable state.

        Returns:
            BalanceEngineMemento: Independent copy of every mutable array.
        """
        engine = self._engine
        bm = engine.balance_model
        return BalanceEngineMemento(
            nodes_dxdydz=bm.nodes.dxdydz.copy(),
            parameter=bm.parameter.copy(),
            sagging_temperature=bm.sagging_temperature.copy(),
            adjustment=bm.adjustment,
            Th=bm.Th.copy(),
            Tv_d=bm.Tv_d.copy(),
            Tv_g=bm.Tv_g.copy(),
            a=bm.a.copy(),
            b=bm.b.copy(),
            span_sagging_parameter=engine.span_model.sagging_parameter.copy(),
            span_span_length=engine.span_model.span_length.copy(),
            span_elevation_difference=engine.span_model.elevation_difference.copy(),
            span_load_coefficient=engine.span_model.load_coefficient.copy(),
            wind_pressure=engine.cable_loads.wind_pressure.copy(),
            ice_thickness=engine.cable_loads.ice_thickness.copy(),
            deformation_current_temperature=engine.deformation_model.current_temperature.copy(),
            deformation_tension_mean=engine.deformation_model.tension_mean.copy(),
            deformation_cable_length=engine.deformation_model.cable_length.copy(),
            L_ref=np.array(engine.L_ref).copy()
            if hasattr(engine, "L_ref")
            else None,
        )

    def restore(self, memento: BalanceEngineMemento) -> None:
        """Restore the engine's mutable state from a previously saved memento.

        After restoration all internal caches (span geometry, deformation
        snapshots, node projections, moments) are refreshed so the engine is
        immediately ready for a new solve.

        Args:
            memento (BalanceEngineMemento): Snapshot previously returned by
                [`save`][mechaphlowers.core.models.balance.memento.BalanceEngineCaretaker.save].
        """
        engine = self._engine
        bm = engine.balance_model

        # (a) Restore nodes — in-place write preserves array identity
        bm.nodes.dxdydz[:] = memento.nodes_dxdydz

        # (b) Restore balance_model scalars / arrays
        bm.parameter = memento.parameter.copy()
        bm.sagging_temperature = memento.sagging_temperature.copy()
        bm.adjustment = memento.adjustment
        bm.Th = memento.Th.copy()
        bm.Tv_d = memento.Tv_d.copy()
        bm.Tv_g = memento.Tv_g.copy()
        bm.a = memento.a.copy()
        bm.b = memento.b.copy()

        # (c) Restore span_model arrays + refresh cached _x_m/_x_n/_L
        engine.span_model.sagging_parameter = memento.span_sagging_parameter.copy()
        engine.span_model.span_length = memento.span_span_length.copy()
        engine.span_model.elevation_difference = memento.span_elevation_difference.copy()
        engine.span_model.load_coefficient = memento.span_load_coefficient.copy()
        engine.span_model.compute_values()

        # (d) Restore cable_loads
        engine.cable_loads.wind_pressure = memento.wind_pressure.copy()
        engine.cable_loads.ice_thickness = memento.ice_thickness.copy()

        # (e) Restore deformation_model (snapshots, not refs)
        engine.deformation_model.current_temperature = (
            memento.deformation_current_temperature.copy()
        )
        engine.deformation_model.tension_mean = (
            memento.deformation_tension_mean.copy()
        )
        engine.deformation_model.cable_length = (
            memento.deformation_cable_length.copy()
        )

        # (f) L_ref
        if memento.L_ref is not None:
            engine.L_ref = memento.L_ref.copy()
        elif hasattr(engine, "L_ref"):
            del engine.L_ref

        # (g) Re-sync derived objects
        bm.nodes_span_model.mirror(engine.span_model)
        bm.nodes.compute_dx_dy_dz()
        bm.nodes.vector_projection.set_tensions(bm.Th, bm.Tv_d, bm.Tv_g)
        bm.nodes.compute_moment()

        logger.debug("Balance engine state restored from memento.")
