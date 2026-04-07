# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from mechaphlowers.config import options
from mechaphlowers.entities.errors import RtsDataNotAvailable


class ITensileStrength(ABC):
    """Abstract interface for cable tensile strength models.

    Implementations are responsible for computing the Residual Rated Tensile
    Strength (RRTS) of a cable and the associated utilization rate, taking
    into account optional strand damage (``cut_strands``) and safety margins
    (``high_safety``).
    """

    @property
    @abstractmethod
    def cut_strands(self) -> np.ndarray:
        """Number of cut strands per layer, as an 8-element integer array."""

    @cut_strands.setter
    @abstractmethod
    def cut_strands(self, value: list[int] | np.ndarray) -> None: ...

    @property
    @abstractmethod
    def high_safety(self) -> bool:
        """Whether the additional safety coefficient security mechanism is enabled."""

    @high_safety.setter
    @abstractmethod
    def high_safety(self, value: bool) -> None: ...

    @property
    @abstractmethod
    def safety_coefficient(self) -> float:
        """Effective safety coefficient used in the utilization rate."""

    @property
    @abstractmethod
    def nb_strand_per_layer(self) -> np.ndarray:
        """Number of strands per layer, as an 8-element integer array."""

    @abstractmethod
    def rts_coverage(self) -> float:
        """Ratio of cable RTS to the sum of strand-level RTS contributions."""

    @property
    @abstractmethod
    def rrts(self) -> float:
        """Residual Rated Tensile Strength (RRTS) in N."""

    @abstractmethod
    def utilization_rate(self, tension_sup_N: np.ndarray) -> np.ndarray:
        """Utilization rate as a percentage of RTS, one value per span."""


class AdditiveLayerRts(ITensileStrength):
    """Tensile strength model based on additive per-layer RTS contributions.

    Computes RRTS as:

    $$RRTS = RTS_{cable} - \\sum_{i} cut\\_strands_i \\times rts\\_layer_i$$

    Args:
        cable_data: Unit-converted cable data snapshot (e.g. from
            ``CableArray.data``, validated against
            [`CableArrayInput`][mechaphlowers.entities.schemas.CableArrayInput]).
            The class reads the following columns — all values must already be
            expressed in SI units (N for forces) as delivered by
            [`CableArray`][mechaphlowers.entities.arrays.CableArray]:

            - ``rts_cable`` *(int, optional/nullable)*: Rated Tensile Strength
              of the whole cable in N. Required by [`rrts`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts.rrts] and
              [`rts_coverage`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts.rts_coverage]; raises
              [`RtsDataNotAvailable`][mechaphlowers.entities.errors.RtsDataNotAvailable] when
              absent or NaN.
            - ``rts_layer_1`` … ``rts_layer_8`` *(int, optional/nullable)*:
              RTS contribution of a single strand in each layer, in N.
              Used by [`rrts`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts.rrts] (when the corresponding
              [`cut_strands`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts.cut_strands] entry is non-zero) and by
              [`rts_coverage`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts.rts_coverage]. Missing or NaN values are treated as 0
              in [`rts_coverage`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts.rts_coverage] but raise
              [`RtsDataNotAvailable`][mechaphlowers.entities.errors.RtsDataNotAvailable] in
              [`rrts`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts.rrts] if the layer has cut strands.
            - ``nb_strand_layer_1`` … ``nb_strand_layer_8`` *(int,
              optional/nullable)*: Number of strands in each layer.
              Used by [`nb_strand_per_layer`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts.nb_strand_per_layer], [`rts_coverage`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts.rts_coverage], and
              to enforce the ``cut_strands`` upper bound. Missing or NaN
              values are treated as 0.
            - ``safety_coefficient`` *(float, optional/nullable)*: Cable-
              specific safety coefficient. Falls back to
              ``options.data.safety_coefficient_default`` when absent or NaN.
    """

    _RTS_LAYERS: list[str] = [
        "rts_layer_1",
        "rts_layer_2",
        "rts_layer_3",
        "rts_layer_4",
        "rts_layer_5",
        "rts_layer_6",
        "rts_layer_7",
        "rts_layer_8",
    ]
    _NB_STRAND_LAYERS: list[str] = [
        "nb_strand_layer_1",
        "nb_strand_layer_2",
        "nb_strand_layer_3",
        "nb_strand_layer_4",
        "nb_strand_layer_5",
        "nb_strand_layer_6",
        "nb_strand_layer_7",
        "nb_strand_layer_8",
    ]
    MAX_ALLOWED_CUT_STRANDS = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=int)

    def __init__(self, cable_data: pd.DataFrame) -> None:
        self._cable_data: pd.DataFrame = cable_data
        self._cut_strands: np.ndarray = np.zeros(8, dtype=int)
        self._high_safety: bool = False

    # ------------------------------------------------------------------
    # high_safety
    # ------------------------------------------------------------------

    @property
    def high_safety(self) -> bool:
        """Enable or disable the additional safety coefficient security mechanism.

        When enabled, the effective
        [`safety_coefficient`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts.safety_coefficient] is multiplied by ``1.5``, which tightens
        the utilization rate limit computed by [`utilization_rate`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts.utilization_rate].
        """
        return self._high_safety

    @high_safety.setter
    def high_safety(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("high_safety must be a boolean.")
        self._high_safety = value

    # ------------------------------------------------------------------
    # safety_coefficient
    # ------------------------------------------------------------------

    @property
    def safety_coefficient(self) -> float:
        """Effective safety coefficient used in the utilization rate.

        Returns the catalog ``safety_coefficient`` value, or
        ``options.data.safety_coefficient_default`` when the column is absent
        or NaN. If [`high_safety`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts.high_safety] is ``True``, the value is multiplied by
        ``options.data.safety_security_factor``.
        """
        col = self._cable_data.get("safety_coefficient")
        if col is None or pd.isna(col.iloc[0]):
            base = options.data.safety_coefficient_default
        else:
            base = float(col.iloc[0])
        if self._high_safety:
            return base * options.data.safety_security_factor
        return base

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rts_layers_array(self) -> np.ndarray:
        """Per-layer RTS values as an 8-element float array.

        Missing or NaN columns produce ``nan`` entries.
        """
        return (
            self._cable_data.reindex(columns=self._RTS_LAYERS)
            .iloc[0]
            .to_numpy(dtype=float)
        )

    # ------------------------------------------------------------------
    # nb_strand_per_layer
    # ------------------------------------------------------------------

    @property
    def nb_strand_per_layer(self) -> np.ndarray:
        """Number of strands per layer, as an 8-element integer array.

        Index 0 = layer 1, …, index 7 = layer 8.
        Returns zeros for layers whose column is absent or NaN in the catalog.

        This value is informational. Cut strand enforcement is controlled by
        ``MAX_ALLOWED_CUT_STRANDS``: when ``MAX_ALLOWED_CUT_STRANDS[i] > 0``,
        ``cut_strands[i]`` cannot exceed that value.
        """
        arr = (
            self._cable_data.reindex(columns=self._NB_STRAND_LAYERS)
            .iloc[0]
            .to_numpy(dtype=float)
        )
        return np.nan_to_num(arr, nan=0.0).astype(int)

    # ------------------------------------------------------------------
    # rts_coverage
    # ------------------------------------------------------------------

    def rts_coverage(self) -> float:
        """Ratio of cable RTS to the sum of strand-level RTS contributions.

        $$\\text{rts_coverage} = \\frac{\\sum_{i} rts_{layer,i} \\times nb_{strand,i}}{rts_{cable}}$$

        This method is a checker for catalog data consistency: the output should ideally be close to 1.0 for a well-documented cable, but may be lower if the catalog is missing some strand-level RTS data (NaN values are treated as 0). A very low value may indicate that the cable-level RTS is not properly documented by the catalog.

        Returns:
            float: ``sum(rts_layer_i * nb_strand_layer_i) / rts_cable``.

        Raises:
            RtsDataNotAvailable: if ``rts_cable`` is missing or NaN.
        """
        rts_layers = np.nan_to_num(self._rts_layers_array(), nan=0.0)
        nb_strands = self.nb_strand_per_layer.astype(float)
        rts_layer_sum = float(rts_layers @ nb_strands)
        rts_cable_col = self._cable_data.get("rts_cable")
        if rts_cable_col is None or pd.isna(rts_cable_col.iloc[0]):
            raise RtsDataNotAvailable(
                "Cannot compute rts_coverage: 'rts_cable' is missing or NaN."
            )
        rts_cable = float(rts_cable_col.iloc[0])
        return rts_layer_sum / rts_cable

    # ------------------------------------------------------------------
    # cut_strands
    # ------------------------------------------------------------------

    @property
    def cut_strands(self) -> np.ndarray:
        """Number of cut strands per layer, as an 8-element integer array.

        Index 0 = layer 1, …, index 7 = layer 8.
        Defaults to all zeros until explicitly set.
        """
        return self._cut_strands

    @cut_strands.setter
    def cut_strands(self, cut_strands: list[int] | np.ndarray) -> None:
        """Set the number of cut strands per layer.

        Args:
            cut_strands: Sequence of up to 8 integers where index 0 = layer 1,
                index 1 = layer 2, …, index 7 = layer 8.

        Raises:
            ValueError: if any value is negative.
            ValueError: if more than 8 elements are provided.
            ValueError: if ``cut_strands[i] > self.max_allowed`` for
                any layer where strand count data is available in the catalog.
        """
        cut_strands_arr = np.asarray(cut_strands)
        if (cut_strands_arr < 0).any():
            raise ValueError(
                f"cut_strands values must be non-negative, got: {cut_strands_arr.tolist()}."
            )
        cut_strands_arr = cut_strands_arr.astype(int)
        if len(cut_strands_arr) > 8:
            raise ValueError(
                f"cut_strands must have at most 8 elements, got {len(cut_strands_arr)}."
            )
        padded = np.zeros(8, dtype=int)
        padded[: len(cut_strands_arr)] = cut_strands_arr

        max_allowed = self.MAX_ALLOWED_CUT_STRANDS
        # 0 means no restriction for that layer; only enforce when > 0
        violation_mask = (max_allowed > 0) & (padded > max_allowed)
        if violation_mask.any():
            details = "; ".join(
                f"layer {i + 1}: cut_strands={padded[i]} > max_allowed={max_allowed[i]}"
                for i in np.nonzero(violation_mask)[0]
            )
            raise ValueError("cut_strands exceeds allowed maximum: " + details)

        self._cut_strands = padded

    # ------------------------------------------------------------------
    # rrts
    # ------------------------------------------------------------------

    @property
    def rrts(self) -> float:
        """Residual Rated Tensile Strength (RRTS) in N.

        RRTS = rts_cable - sum(cut_strands[i] * rts_layer{i+1} for i in 0..7)

        Defaults to rts_cable when no cut strands have been set (all zeros).

        Warning: rrts is designed to act globally for the section. If one
        strand is cut on a span, the whole section gets the same reduced RRTS,
        even if the cut strand is not on this span.

        Raises:
            RtsDataNotAvailable: if ``rts_cable`` is missing or NaN, or if
                ``cut_strands[i] > 0`` but the corresponding rts layer column
                is missing or NaN in the catalog data.
        """
        rts_cable_col = self._cable_data.get("rts_cable")
        if rts_cable_col is None or pd.isna(rts_cable_col.iloc[0]):
            raise RtsDataNotAvailable(
                "Cannot compute RRTS: 'rts_cable' is missing or NaN in the cable catalog."
            )
        rts_cable = float(rts_cable_col.iloc[0])

        rts_layers = self._rts_layers_array()

        invalid_mask = (self._cut_strands > 0) & np.isnan(rts_layers)
        if invalid_mask.any():
            details = "; ".join(
                f"cut_strands[{i}] = {self._cut_strands[i]} but '{self._RTS_LAYERS[i]}' is missing or NaN"
                for i in np.nonzero(invalid_mask)[0]
            )
            raise RtsDataNotAvailable(details)

        rts_layers = np.nan_to_num(rts_layers, nan=0.0)
        return rts_cable - float(self._cut_strands @ rts_layers)

    # ------------------------------------------------------------------
    # utilization_rate
    # ------------------------------------------------------------------

    def utilization_rate(self, tension_sup_N: np.ndarray) -> np.ndarray:
        """Utilization rate as a percentage of RTS, one value per span.

        rate (%) = tension_sup_N / (RRTS * safety_coefficient) * 100

        Args:
            tension_sup_N: Cable tensions in N, one value per span (e.g.
                tension_sup from BalanceEngine.get_data_spans()).

        Returns:
            Array of utilization rates in % of RTS, same length as tension_sup_N.
        """
        return (
            np.asarray(tension_sup_N)
            / self.rrts
            * self.safety_coefficient
            * 100
        )
