# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mechaphlowers.config import options
from mechaphlowers.data.units import Q_
from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.utils import arr

logger = logging.getLogger(__name__)


class Manipulation:
    """Stores and applies geometric manipulations to a SectionArray.

    A Manipulation collects support offsets, rope replacements, and virtual
    support insertions.  Calling :meth:`apply` produces a **copy** of the
    original :class:`SectionArray` whose ``_data`` incorporates every active
    overlay.  The original array is never modified.

    Args:
        section_array: The original (clean) section array to manipulate.
    """

    def __init__(self, section_array: SectionArray) -> None:
        self._section_array = section_array
        self._support_overlay: dict[int, dict[str, float]] | None = None
        self._rope_overlay: dict[int, float] | None = None
        self._rope_lineic_mass: float | None = None
        self._virtual_support_overlay: dict[int, dict] | None = None

    # ── Query helpers ─────────────────────────────────────────────────────

    @property
    def has_manipulations(self) -> bool:
        return (
            self._support_overlay is not None
            or self._rope_overlay is not None
            or self._virtual_support_overlay is not None
        )

    @property
    def has_virtual_support(self) -> bool:
        return self._virtual_support_overlay is not None

    # ── Support manipulation ──────────────────────────────────────────────

    def support_manipulation(
        self, manipulation: dict[int, dict[str, float]]
    ) -> None:
        """Apply additive offsets to support geometry.

        Stores the offsets as an overlay applied by :meth:`apply`.
        Use :meth:`reset_manipulation` to remove the overlay.

        For each affected support, ``counterweight_mass`` is set to 0 in
        the applied copy.

        Args:
            manipulation: Dictionary mapping support index (0-based) to
                offsets.  Each value is a dict with optional keys:

                - ``"y"``: added to ``crossarm_length`` (meters)
                - ``"z"``: added to ``conductor_attachment_altitude`` (meters)

        Raises:
            ValueError: If a support index is out of range.
            ValueError: If an inner dict contains keys other than ``"y"`` or ``"z"``.

        Examples:
            >>> manip.support_manipulation({1: {"z": 2.0, "y": -1.0}})
            >>> manip.support_manipulation({0: {"z": 0.5}, 2: {"y": 3.0}})
        """
        n_supports = len(self._section_array._data)
        allowed_keys = {"y", "z"}

        for idx, offsets in manipulation.items():
            if idx < 0 or idx >= n_supports:
                raise ValueError(
                    f"Support index {idx} is out of range [0, {n_supports - 1}]"
                )
            invalid_keys = set(offsets.keys()) - allowed_keys
            if invalid_keys:
                raise ValueError(
                    f"Invalid keys {invalid_keys} for support {idx}. Allowed keys: {allowed_keys}"
                )

        if self._support_overlay is None:
            self._support_overlay = {}
        for idx, offsets in manipulation.items():
            if idx not in self._support_overlay:
                self._support_overlay[idx] = {}
            for key, value in offsets.items():
                self._support_overlay[idx][key] = (
                    self._support_overlay[idx].get(key, 0.0) + value
                )

        logger.debug(f"Support manipulation applied: {manipulation}")

    def reset_manipulation(self) -> None:
        """Remove the support manipulation overlay.

        Does nothing if no manipulation has been applied.

        Examples:
            >>> manip.support_manipulation({1: {"z": 5.0}})
            >>> manip.reset_manipulation()
        """
        if self._support_overlay is None:
            logger.debug(
                "reset_manipulation called but no manipulation was applied."
            )
            return

        self._support_overlay = None
        logger.debug("Support manipulation reset to original values.")

    # ── Rope manipulation ─────────────────────────────────────────────────

    def rope_manipulation(
        self,
        rope: dict[int, float],
        rope_lineic_mass: float | None = None,
    ) -> None:
        """Override insulator length and mass for specified supports with rope values.

        The override is applied by :meth:`apply`; the original ``_data``
        is never modified.
        Use :meth:`reset_rope_manipulation` to remove the overlay.

        For each affected support, ``counterweight_mass`` is set to 0 in
        the applied copy.

        Args:
            rope: Dictionary mapping support index (0-based) to rope length (meters).
            rope_lineic_mass: Linear mass of the rope in kg/m. Defaults to
                ``options.data.rope_lineic_mass_default`` (``0.01`` kg/m).

        Raises:
            ValueError: If a support index is out of range.

        Examples:
            >>> manip.rope_manipulation({1: 4.5, 2: 3.0})
            >>> manip.rope_manipulation({0: 2.0}, rope_lineic_mass=0.05)
        """
        n_supports = len(self._section_array._data)
        for idx in rope:
            if idx < 0 or idx >= n_supports:
                raise ValueError(
                    f"Support index {idx} is out of range [0, {n_supports - 1}]"
                )

        self._rope_overlay = rope
        self._rope_lineic_mass = (
            rope_lineic_mass
            if rope_lineic_mass is not None
            else options.data.rope_lineic_mass_default
        )
        logger.debug(f"Rope manipulation applied: {rope}")

    def reset_rope_manipulation(self) -> None:
        """Remove the rope overlay.

        Does nothing if no rope manipulation has been applied.

        Examples:
            >>> manip.rope_manipulation({1: 4.5})
            >>> manip.reset_rope_manipulation()
        """
        if self._rope_overlay is None:
            logger.debug(
                "reset_rope_manipulation called but no rope manipulation was applied."
            )
            return
        self._rope_overlay = None
        self._rope_lineic_mass = None
        logger.debug("Rope manipulation cleared.")

    # ── Virtual support ───────────────────────────────────────────────────

    def add_virtual_support(
        self, virtual_support: dict[int, dict[str, float]]
    ) -> None:
        """Insert virtual supports.

        Each virtual support splits the span containing it.
        The override is applied by :meth:`apply`; the original ``_data``
        is never modified.
        Use :meth:`reset_virtual_support` to remove all virtual supports.

        Args:
            virtual_support: Dictionary mapping left-support index (0-based,
                must not be the last support) to a dict with keys:

                - ``"x"``: distance from the left support in meters.
                - ``"y"``: lateral offset in meters.
                - ``"z"``: ``conductor_attachment_altitude`` of the new
                  virtual support in meters.
                - ``"insulator_length"``: insulator length in meters.
                - ``"insulator_mass"``: insulator mass in kg.
                - ``"hanging_cable_point_from_left_support"``: distance from
                  the left support to the cable hanging point in meters.

        Raises:
            ValueError: If a span index is out of range.
            ValueError: If ``x`` or ``hanging_cable_point_from_left_support`` is
                out of the allowed range.
            ValueError: If required keys are missing.

        Examples:
            >>> manip.add_virtual_support(
            ...     {
            ...         1: {
            ...             "x": 200.0,
            ...             "y": 0.0,
            ...             "z": 55.0,
            ...             "insulator_length": 3.0,
            ...             "insulator_mass": 500.0,
            ...             "hanging_cable_point_from_left_support": 200.0,
            ...         }
            ...     }
            ... )
        """
        n_supports = len(self._section_array._data)
        required_keys = {
            "x",
            "y",
            "z",
            "insulator_length",
            "insulator_mass",
            "hanging_cable_point_from_left_support",
        }

        for span_idx, vs in virtual_support.items():
            if span_idx < 0 or span_idx >= n_supports - 1:
                raise ValueError(
                    f"Span index {span_idx} is out of range [0, {n_supports - 2}]"
                )
            missing_keys = required_keys - set(vs.keys())
            if missing_keys:
                raise ValueError(
                    f"Missing keys {missing_keys} for span {span_idx}. Required: {required_keys}"
                )
            span_length = float(
                self._section_array._data["span_length"].iloc[span_idx]
            )
            crossarm_left = float(
                self._section_array._data["crossarm_length"].iloc[span_idx]
            )
            crossarm_right = float(
                self._section_array._data["crossarm_length"].iloc[span_idx + 1]
            )
            x = vs["x"]
            x_lower = -abs(crossarm_left)
            x_upper = abs(span_length) + abs(crossarm_right)
            if x <= x_lower or x >= x_upper:
                raise ValueError(
                    f"x={x} is out of range ({x_lower}, {x_upper}) for span {span_idx}"
                )
            hcp = vs["hanging_cable_point_from_left_support"]
            if hcp <= x_lower or hcp >= x_upper:
                raise ValueError(
                    f"hanging_cable_point_from_left_support={hcp} is out of range ({x_lower}, {x_upper}) for span {span_idx}"
                )

        if self._virtual_support_overlay is None:
            self._virtual_support_overlay = {}
        self._virtual_support_overlay.update(virtual_support)
        logger.debug(f"Virtual support overlay updated: {virtual_support}")

    def reset_virtual_support(self) -> None:
        """Remove all virtual supports.

        Does nothing if no virtual supports have been added.

        Examples:
            >>> manip.add_virtual_support({...})
            >>> manip.reset_virtual_support()
        """
        if self._virtual_support_overlay is None:
            logger.debug(
                "reset_virtual_support called but no virtual support was added."
            )
            return
        self._virtual_support_overlay = None
        logger.debug("Virtual support overlay cleared.")

    # ── Apply ─────────────────────────────────────────────────────────────

    def apply(self) -> SectionArray:
        """Create a copy of the section array with all manipulations baked into ``_data``.

        The original section array is never modified.
        The returned copy has:

        * Support offsets applied to ``conductor_attachment_altitude`` / ``crossarm_length``
        * Rope values replacing ``insulator_length`` / ``insulator_mass``
        * ``counterweight_mass`` set to 0 for affected supports
        * Virtual support rows inserted

        Returns:
            A new :class:`SectionArray` whose ``_data`` reflects every
            active overlay.
        """
        original = self._section_array
        raw_data = original._data.copy()

        # Apply support overlay
        if self._support_overlay is not None:
            for idx, offsets in self._support_overlay.items():
                if "z" in offsets:
                    raw_data.loc[idx, "conductor_attachment_altitude"] += (
                        self._to_input(
                            offsets["z"], "conductor_attachment_altitude"
                        )
                    )
                if "y" in offsets:
                    raw_data.loc[idx, "crossarm_length"] += self._to_input(
                        offsets["y"], "crossarm_length"
                    )

        # Apply rope overlay
        if self._rope_overlay is not None:
            for idx, rope_length in self._rope_overlay.items():
                raw_data.loc[idx, "insulator_length"] = self._to_input(
                    rope_length, "insulator_length"
                )
                raw_data.loc[idx, "insulator_mass"] = self._to_input(
                    rope_length * self._rope_lineic_mass, "insulator_mass"
                )

        # Counterweight masking for affected supports
        affected: set[int] = set()
        if self._support_overlay is not None:
            affected |= set(self._support_overlay.keys())
        if self._rope_overlay is not None:
            affected |= set(self._rope_overlay.keys())
        if "counterweight_mass" in raw_data.columns and affected:
            for idx in affected:
                raw_data.loc[idx, "counterweight_mass"] = 0.0

        # Virtual support insertion
        if self._virtual_support_overlay is not None:
            raw_data = self._apply_virtual_support_overlay(raw_data)

        # Create new SectionArray from manipulated data
        sa = SectionArray(
            raw_data,
            sagging_parameter=original.sagging_parameter,
            sagging_temperature=original.sagging_temperature,
            bundle_number=original.bundle_number,
        )
        sa.input_units = original.input_units.copy()
        sa._angles_sense = original._angles_sense
        return sa

    # ── Virtual support L_ref splitting ───────────────────────────────────

    def compute_split_L_ref(
        self,
        initial_L_ref: np.ndarray,
        span_model: object,
    ) -> np.ndarray:
        """Split ``initial_L_ref`` to account for virtual support insertion.

        Each impacted span is split into two semi-spans using the span
        model's ``compute_partial_L`` method.

        Args:
            initial_L_ref: Reference cable lengths from the clean solve
                (span-based, ``n_supports - 1`` elements).
            span_model: The clean engine's span model (must have
                ``compute_partial_L`` and ``span_length``).

        Returns:
            Expanded L_ref array with additional entries for virtual spans.
        """
        if self._virtual_support_overlay is None:
            return initial_L_ref

        n_spans = int(initial_L_ref.shape[0])
        hanging_points = np.zeros(n_spans, dtype=np.float64)
        impacted_spans = np.zeros(n_spans, dtype=bool)

        for span_idx in sorted(self._virtual_support_overlay.keys()):
            if 0 <= span_idx < n_spans:
                hanging_points[span_idx] = float(
                    self._virtual_support_overlay[span_idx][
                        "hanging_cable_point_from_left_support"
                    ]
                )
                impacted_spans[span_idx] = True

        new_L_ref = arr.incr(initial_L_ref.copy())

        new_L_ref_0 = span_model.compute_partial_L(x=hanging_points)[
            impacted_spans
        ]
        new_L_ref_1 = new_L_ref[impacted_spans] - new_L_ref_0

        new_L_ref[impacted_spans] = new_L_ref_1
        new_L_ref = np.insert(new_L_ref, impacted_spans, new_L_ref_0)

        return new_L_ref

    # ── Private helpers ───────────────────────────────────────────────────

    def _to_input(self, value: float, column: str) -> float:
        """Convert *value* from SI (target units) to input units for *column*."""
        target = SectionArray.target_units[column]
        inp = self._section_array.input_units.get(column, target)
        if inp == target:
            return value
        return float(Q_(value, target).to(inp).magnitude)

    def _apply_virtual_support_overlay(
        self, raw_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Insert virtual support rows into *raw_data* (in input units)."""
        sorted_keys = sorted(self._virtual_support_overlay.keys())
        for offset, span_idx in enumerate(sorted_keys):
            vs = self._virtual_support_overlay[span_idx]
            effective_idx = span_idx + offset

            x = vs["x"]
            y = vs["y"]
            angle = np.arctan2(y, x)  # radians

            original_span_input = float(
                raw_data.loc[effective_idx, "span_length"]
            )
            x_input = self._to_input(x, "span_length")

            # Modify left support
            raw_data.loc[effective_idx, "span_length"] = x_input
            raw_data.loc[effective_idx, "line_angle"] = self._to_input(
                angle, "line_angle"
            )

            # Build virtual row
            remaining_span = abs(original_span_input - x_input)
            virtual_row: dict[str, object] = {
                col: np.nan for col in raw_data.columns
            }
            virtual_row.update(
                {
                    "name": f"virtual_{span_idx}",
                    "suspension": True,
                    "conductor_attachment_altitude": self._to_input(
                        float(vs["z"]), "conductor_attachment_altitude"
                    ),
                    "crossarm_length": self._to_input(0.0, "crossarm_length"),
                    "line_angle": self._to_input(-angle, "line_angle"),
                    "insulator_length": self._to_input(
                        max(float(vs["insulator_length"]), 0.01),
                        "insulator_length",
                    ),
                    "span_length": remaining_span,
                    "insulator_mass": self._to_input(
                        float(vs["insulator_mass"]), "insulator_mass"
                    ),
                }
            )

            for optional_col, fill in (
                ("load_mass", 0.0),
                ("load_position", np.nan),
                ("counterweight_mass", 0.0),
            ):
                if optional_col in raw_data.columns:
                    virtual_row[optional_col] = fill

            if "ground_altitude" in raw_data.columns:
                virtual_row["ground_altitude"] = (
                    float(vs["z"]) - options.ground.default_support_length
                )

            virtual_df = pd.DataFrame([virtual_row])
            top = raw_data.iloc[: effective_idx + 1]
            bottom = raw_data.iloc[effective_idx + 1 :]
            raw_data = pd.concat(
                [top, virtual_df, bottom], ignore_index=True
            )

        return raw_data
