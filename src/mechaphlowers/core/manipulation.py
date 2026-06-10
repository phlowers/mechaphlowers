# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
import warnings
from copy import copy
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from mechaphlowers.config import options
from mechaphlowers.core.models.cable.span import ISpan
from mechaphlowers.data.units import Q_
from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.entities.errors import (
    BalanceEngineWarning,
    InvalidManipulationIndex,
    InvalidManipulationKeys,
    InvalidManipulationRange,
)
from mechaphlowers.utils import arr

if TYPE_CHECKING:
    from mechaphlowers.core.models.balance.engine import BalanceEngine

logger = logging.getLogger(__name__)


class Manipulation:
    """Stores and applies geometric manipulations to a SectionArray.

    A Manipulation collects support offsets, rope replacements, and virtual
    support insertions.  Calling [`from_section_array`][mechaphlowers.core.manipulation.Manipulation.from_section_array] produces a **copy** of the
    original [`SectionArray`][mechaphlowers.entities.arrays.SectionArray] whose ``_data`` incorporates every active
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
        self._shifting_distance_support: np.ndarray | None = None
        self._shortening_distance_span: np.ndarray | None = None

    # ── Query helpers ─────────────────────────────────────────────────────

    @property
    def has_manipulations(self) -> bool:
        return (
            self._support_overlay is not None
            or self._rope_overlay is not None
            or self._virtual_support_overlay is not None
            or self._shifting_distance_support is not None
        )

    @property
    def has_virtual_support(self) -> bool:
        return self._virtual_support_overlay is not None

    @property
    def has_shifting(self) -> bool:
        return self._shifting_distance_support is not None

    # ── Cable shifting ────────────────────────────────────────────────────

    @property
    def shift_support(self) -> np.ndarray | None:
        """Shifting distance, in meters. ``None`` when no shifting is set."""
        return self._shifting_distance_support

    @property
    def shortening_span(self) -> np.ndarray | None:
        """Shortening distance, in meters. ``None`` when no shifting is set."""
        return self._shortening_distance_span

    def modify_cable(
        self,
        shift_support: dict[int, float] | None = None,
        shorten_span: dict[int, float] | None = None,
    ) -> None:
        """Validate and store cable shifting values.

        Inputs are sparse dicts mapping indices to values in meters.
        Unspecified supports/spans default to 0.

        Args:
            shift_support (dict[int, float] | None): Horizontal shifting of each support, in meters.
                Dictionary mapping support index (0-based) to shift value. The first and last
                supports (index 0 and ``support_number - 1``) are enforced to 0. If ``None``,
                no shifting is applied.
            shorten_span (dict[int, float] | None): Span length modification, in meters.
                Dictionary mapping span index (0-based) to shortening value. Positive values
                shorten the spans, negative values lengthen them. If ``None``, no shortening
                is applied.

        Raises:
            InvalidManipulationIndex: If a support or span index is out of range.
        """
        n_supports = len(self._section_array._data)
        n_spans = n_supports - 1

        shift_support = shift_support or {}
        shorten_span = shorten_span or {}

        # Validate indices
        for idx in shift_support:
            if idx < 0 or idx >= n_supports:
                raise InvalidManipulationIndex(
                    f"shift_support index {idx} is out of range (0 to {n_supports - 1})"
                )
        for idx in shorten_span:
            if idx < 0 or idx >= n_spans:
                raise InvalidManipulationIndex(
                    f"shorten_span index {idx} is out of range (0 to {n_spans - 1})"
                )

        # Convert to numpy arrays
        shift_support_arr = np.zeros(n_supports, dtype=np.float64)
        for idx, val in shift_support.items():
            shift_support_arr[idx] = val

        shorten_span_arr = np.zeros(n_spans, dtype=np.float64)
        for idx, val in shorten_span.items():
            shorten_span_arr[idx] = val

        # Enforce constraints: shifting_distance first and last are 0
        if abs(shift_support_arr[0]) > 0.0 or abs(shift_support_arr[-1]) > 0.0:
            logger.warning(
                "shift_support first and last values must be 0 (support based). "
                "Enforcing this constraint."
            )
            warnings.warn(
                "First and last values of shift_support have been reset to 0",
                BalanceEngineWarning,
            )
        shift_support_arr[0] = 0.0
        shift_support_arr[-1] = 0.0

        # Store in private attributes
        self._shifting_distance_support = shift_support_arr
        self._shortening_distance_span = shorten_span_arr

        logger.debug(
            f"Cable shifting stored: shift_support={shift_support_arr}, shorten_span={shorten_span_arr}"
        )

    def reset_cable(self) -> None:
        """Remove cable shifting.

        Does nothing if no shifting has been applied.
        """
        if self._shifting_distance_support is None:
            logger.debug("reset_cable called but no shifting was applied.")
            return
        self._shifting_distance_support = None
        self._shortening_distance_span = None
        logger.debug("Cable shifting cleared.")

    def _compute_shift_delta(self) -> np.ndarray:
        """Per-span cable length delta from shifting and shortening.

        Must only be called when ``_shifting_distance_support`` is not None.

        Returns:
            Array of length ``n_spans`` (original indices, before any
            virtual-support expansion).
        """
        assert self._shifting_distance_support is not None
        assert self._shortening_distance_span is not None
        shift_span = (
            self._shifting_distance_support[:-1]
            - self._shifting_distance_support[1:]
        )
        return -shift_span - self._shortening_distance_span

    def compute_shifted_L_ref(
        self, initial_L_ref: np.ndarray, *, expand: bool | None = None
    ) -> np.ndarray:
        """Compute L_ref with cable shifting and shortening applied.

        When virtual supports are present the shift vector is expanded to
        match the split span count (zeros inserted for new right sub-spans).

        Args:
            initial_L_ref: Reference cable lengths (span-based).
            expand: Controls virtual-support expansion of the shift vector.

                - ``None`` (default): auto-detect — expand if virtual supports
                  are active.
                - ``True``: force expansion (caller guarantees *initial_L_ref*
                  is already split).
                - ``False``: never expand (original span indices only).

        Returns:
            Shifted L_ref array with same length as *initial_L_ref*.
        """
        if self._shifting_distance_support is None:
            return initial_L_ref

        shifted_length = self._compute_shift_delta()

        should_expand = (
            expand if expand is not None else self.has_virtual_support
        )
        if should_expand and self._virtual_support_overlay is not None:
            for offset, span_idx in enumerate(
                sorted(self._virtual_support_overlay.keys())
            ):
                # The virtual node splits span_idx into left (same index) and
                # right (new index at span_idx + offset + 1).  The shift stays
                # on the left sub-span; insert 0 for the right sub-span.
                insert_pos = span_idx + offset + 1
                shifted_length = np.insert(shifted_length, insert_pos, 0.0)

        return initial_L_ref + shifted_length

    # ── Support manipulation ──────────────────────────────────────────────

    def modify_support(
        self, manipulation: dict[int, dict[str, float]]
    ) -> None:
        """Apply additive offsets to support geometry.

        Stores the offsets as an overlay applied by [`from_section_array`][mechaphlowers.core.manipulation.Manipulation.from_section_array].
        Use [`reset_support`][mechaphlowers.core.manipulation.Manipulation.reset_support] to remove the overlay.

        For each affected support, ``counterweight_mass`` is set to 0 in
        the applied copy.

        Args:
            manipulation: Dictionary mapping support index (0-based) to
                offsets.  Each value is a dict with optional keys:

                - ``"y"``: added to ``crossarm_length`` (meters)
                - ``"z"``: added to ``conductor_attachment_altitude`` (meters)

        Raises:
            InvalidManipulationIndex: If a support index is out of range.
            InvalidManipulationKeys: If an inner dict contains keys other than ``"y"`` or ``"z"``.

        Examples:
            >>> manip.modify_support({1: {"z": 2.0, "y": -1.0}})
            >>> manip.modify_support({0: {"z": 0.5}, 2: {"y": 3.0}})
        """
        n_supports = len(self._section_array._data)
        allowed_keys = {"y", "z"}

        for idx, offsets in manipulation.items():
            if idx < 0 or idx >= n_supports:
                raise InvalidManipulationIndex(
                    f"Support index {idx} is out of range [0, {n_supports - 1}]"
                )
            invalid_keys = set(offsets.keys()) - allowed_keys
            if invalid_keys:
                raise InvalidManipulationKeys(
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

    def reset_support(self) -> None:
        """Remove the support manipulation overlay.

        Does nothing if no manipulation has been applied.

        Examples:
            >>> manip.modify_support({1: {"z": 5.0}})
            >>> manip.reset_support()
        """
        if self._support_overlay is None:
            logger.debug(
                "reset_support called but no manipulation was applied."
            )
            return

        self._support_overlay = None
        logger.debug("Support manipulation reset to original values.")

    # ── Rope manipulation ─────────────────────────────────────────────────

    def add_rope(
        self,
        rope: dict[int, float],
        rope_lineic_mass: float | None = None,
    ) -> None:
        """Override insulator length and mass for specified supports with rope values.

        The override is applied by [`from_section_array`][mechaphlowers.core.manipulation.Manipulation.from_section_array]; the original ``_data``
        is never modified.
        Use [`reset_rope`][mechaphlowers.core.manipulation.Manipulation.reset_rope] to remove the overlay.

        For each affected support, ``counterweight_mass`` is set to 0 in
        the applied copy.

        Args:
            rope: Dictionary mapping support index (0-based) to rope length (meters).
            rope_lineic_mass: Linear mass of the rope in kg/m. Defaults to
                ``options.data.rope_lineic_mass_default`` (``0.01`` kg/m).

        Raises:
            InvalidManipulationIndex: If a support index is out of range.

        Examples:
            >>> manip.add_rope({1: 4.5, 2: 3.0})
            >>> manip.add_rope({0: 2.0}, rope_lineic_mass=0.05)
        """
        n_supports = len(self._section_array._data)
        for idx in rope:
            if idx < 0 or idx >= n_supports:
                raise InvalidManipulationIndex(
                    f"Support index {idx} is out of range [0, {n_supports - 1}]"
                )

        self._rope_overlay = rope
        self._rope_lineic_mass = (
            rope_lineic_mass
            if rope_lineic_mass is not None
            else options.data.rope_lineic_mass_default
        )
        logger.debug(f"Rope manipulation applied: {rope}")

    def reset_rope(self) -> None:
        """Remove the rope overlay.

        Does nothing if no rope manipulation has been applied.

        Examples:
            >>> manip.add_rope({1: 4.5})
            >>> manip.reset_rope()
        """
        if self._rope_overlay is None:
            logger.debug(
                "reset_rope called but no rope manipulation was applied."
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
        The override is applied by [`from_section_array`][mechaphlowers.core.manipulation.Manipulation.from_section_array]; the original ``_data``
        is never modified.
        Use [`reset_virtual_support`][mechaphlowers.core.manipulation.Manipulation.reset_virtual_support] to remove all virtual supports.

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
            InvalidManipulationIndex: If a span index is out of range.
            InvalidManipulationRange: If ``x`` or ``hanging_cable_point_from_left_support`` is
                out of the allowed range.
            InvalidManipulationKeys: If required keys are missing.

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
                raise InvalidManipulationIndex(
                    f"Span index {span_idx} is out of range [0, {n_supports - 2}]"
                )
            missing_keys = required_keys - set(vs.keys())
            if missing_keys:
                raise InvalidManipulationKeys(
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
                raise InvalidManipulationRange(
                    f"x={x} is out of range ({x_lower}, {x_upper}) for span {span_idx}"
                )
            hcp = vs["hanging_cable_point_from_left_support"]
            if hcp <= x_lower or hcp >= x_upper:
                raise InvalidManipulationRange(
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

    def reset_all(self) -> None:
        """Remove all active manipulations.

        Calls [`reset_support`][mechaphlowers.core.manipulation.Manipulation.reset_support],
        [`reset_rope`][mechaphlowers.core.manipulation.Manipulation.reset_rope],
        [`reset_virtual_support`][mechaphlowers.core.manipulation.Manipulation.reset_virtual_support],
        and [`reset_cable`][mechaphlowers.core.manipulation.Manipulation.reset_cable] in sequence.
        Does nothing for each manipulation that was not active.

        Examples:
            >>> manip.modify_support({1: {"z": 2.0}})
            >>> manip.add_rope({2: 4.0})
            >>> manip.reset_all()  # both overlays cleared
        """
        self.reset_support()
        self.reset_rope()
        self.reset_virtual_support()
        self.reset_cable()
        logger.debug("All manipulations reset.")

    # ── Apply ─────────────────────────────────────────────────────────────

    def from_section_array(self, section_array: SectionArray) -> SectionArray:
        """Create a copy of the section array with all manipulations baked into ``_data``.

        The original section array is never modified.
        The returned copy has:

        * Support offsets applied to ``conductor_attachment_altitude`` / ``crossarm_length``
        * Rope values replacing ``insulator_length`` / ``insulator_mass``
        * ``counterweight_mass`` set to 0 for affected supports
        * Virtual support rows inserted

        Args:
            section_array: The original (clean) section array.

        Returns:
            A new [`SectionArray`][mechaphlowers.entities.arrays.SectionArray] whose ``_data`` reflects every
            active overlay.
        """
        original = section_array
        raw_data = original._data.copy()
        input_units = original.input_units

        # Apply support overlay
        if self._support_overlay is not None:
            for idx, offsets in self._support_overlay.items():
                if "z" in offsets:
                    raw_data.loc[idx, "conductor_attachment_altitude"] = cast(
                        float,
                        raw_data.loc[idx, "conductor_attachment_altitude"],
                    ) + self._to_input(
                        offsets["z"],
                        "conductor_attachment_altitude",
                        input_units,
                    )
                if "y" in offsets:
                    raw_data.loc[idx, "crossarm_length"] = cast(
                        float, raw_data.loc[idx, "crossarm_length"]
                    ) + self._to_input(
                        offsets["y"], "crossarm_length", input_units
                    )

        # Apply rope overlay
        if self._rope_overlay is not None:
            if self._rope_lineic_mass is None:
                logger.warning(
                    "Rope overlay is set but rope_lineic_mass is None; skipping rope overlay application."
                )
            else:
                for idx, rope_length in self._rope_overlay.items():
                    raw_data.loc[idx, "insulator_length"] = self._to_input(
                        rope_length, "insulator_length", input_units
                    )
                    raw_data.loc[idx, "insulator_mass"] = self._to_input(
                        rope_length * self._rope_lineic_mass,
                        "insulator_mass",
                        input_units,
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
            raw_data = self._apply_virtual_support_overlay(
                raw_data, input_units
            )

        # Create new SectionArray from manipulated data
        sa = SectionArray(
            raw_data,
            bundle_number=original.bundle_number,
        )
        sa.input_units = original.input_units.copy()
        sa._angle_direction = original._angle_direction
        sa.geolocator = copy(original.geolocator)

        applied = []
        if self._support_overlay is not None:
            applied.append(f"support_overlay={self._support_overlay}")
        if self._rope_overlay is not None:
            applied.append(f"rope_overlay={self._rope_overlay}")
        if self._virtual_support_overlay is not None:
            applied.append(
                f"virtual_support_overlay={list(self._virtual_support_overlay.keys())}"
            )
        if self._shifting_distance_support is not None:
            applied.append(
                f"cable_shifting={self._shifting_distance_support}, cable_shortening={self._shortening_distance_span}"
            )
        logger.debug(
            "from_section_array applied: %s",
            ", ".join(applied) if applied else "no overlays",
        )
        return sa

    def initialize_engine(
        self,
        clean_engine: BalanceEngine,
        section_array: SectionArray,
        initial_L_ref: np.ndarray,
    ) -> BalanceEngine:
        """Build a target [`BalanceEngine`][mechaphlowers.core.models.balance.engine.BalanceEngine] with manipulated geometry.

        The returned engine has ``L_ref`` / ``initial_L_ref`` injected from
        outside and its adjustment is **blocked** — only ``solve_change_state``
        may be called on it.

        Steps performed:

        1. Split ``initial_L_ref`` if virtual supports are present.
        2. Apply cable shifting to the (possibly split) ``L_ref``.
        3. Create a new [`BalanceEngine`][mechaphlowers.core.models.balance.engine.BalanceEngine] from *section_array*
           (the manipulated copy produced by [`from_section_array`][mechaphlowers.core.manipulation.Manipulation.from_section_array]).
        4. Inject ``L_ref`` and block adjustment.

        Args:
            clean_engine: The engine that ran the adjustment on clean geometry.
                Its ``span_model`` is used for virtual-support L_ref splitting.
            section_array: The manipulated section array (output of
                [`from_section_array`][mechaphlowers.core.manipulation.Manipulation.from_section_array]).
            initial_L_ref: ``initial_L_ref`` from the clean adjustment solve.

        Returns:
            A configured [`BalanceEngine`][mechaphlowers.core.models.balance.engine.BalanceEngine] ready for
            ``solve_change_state`` calls.
        """
        from mechaphlowers.core.models.balance.engine import (
            BalanceEngine as _BE,
        )

        L_ref = initial_L_ref.copy()

        # Virtual-support L_ref splitting (must happen before shifting so
        # that the split uses the clean, unshifted L_ref)
        if self.has_virtual_support:
            L_ref = self.compute_split_L_ref(L_ref, clean_engine.span_model)

        # Cable shifting (applied after split so the shift maps to correct
        # expanded span indices — the shift only affects the left sub-span
        # adjacent to the shifted support)
        if self.has_shifting:
            L_ref = self.compute_shifted_L_ref(L_ref, expand=True)

        # Build target engine
        target_engine = _BE(
            cable_array=clean_engine.cable_array,
            section_array=section_array,
            span_model_type=clean_engine.span_model_type,
            deformation_model_type=clean_engine.deformation_model_type,
        )

        # Propagate clean engine's solved node state to the target engine so
        # that span_model effective lengths are consistent with the injected
        # L_ref. Without this, nodes retain default positions (dx=±insulator_length,
        # dz=0 for anchors) which inflate effective spans and can lead to
        # an impossible catenary configuration.
        clean_state = clean_engine.balance_model.state_vector
        expanded_state = self._expand_state_vector_for_virtual_supports(
            clean_state
        )
        target_engine.balance_model.state_vector = expanded_state
        target_engine.balance_model.update()

        # Inject L_ref and block adjustment. The target engine's balance model is already initialized with the manipulated section array
        target_engine.initial_L_ref = initial_L_ref.copy()
        target_engine.L_ref = L_ref
        target_engine.balance_model.L_ref = L_ref
        target_engine._adjustment_blocked = True

        return target_engine

    # ── Virtual support L_ref splitting ───────────────────────────────────

    def compute_split_L_ref(
        self,
        initial_L_ref: np.ndarray,
        span_model: ISpan,
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

        new_L_ref = initial_L_ref.copy()

        new_L_ref_0 = arr.decr(
            span_model.compute_partial_L(new_a=arr.incr(hanging_points))
        )[impacted_spans]
        new_L_ref_1 = new_L_ref[impacted_spans] - new_L_ref_0

        new_L_ref[impacted_spans] = new_L_ref_1
        new_L_ref = np.insert(
            new_L_ref, np.where(impacted_spans)[0], new_L_ref_0
        )

        return new_L_ref

    def _expand_state_vector_for_virtual_supports(
        self, state_vector: np.ndarray
    ) -> np.ndarray:
        """Expand a state vector to account for virtual support insertions.

        Each virtual support inserts a new suspension node into the section
        array.  This method inserts a ``[0.0, 0.0]`` pair (neutral
        displacement) at the corresponding position in the state vector so
        that it matches the target engine's node count.

        Args:
            state_vector: State vector from the clean engine
                (size ``2 * n_original_nodes``).

        Returns:
            Expanded state vector (size ``2 * (n_original_nodes + n_virtual)``).
        """
        if self._virtual_support_overlay is None:
            return state_vector

        result = state_vector.copy()
        for offset, span_idx in enumerate(
            sorted(self._virtual_support_overlay.keys())
        ):
            # Virtual node is inserted at position span_idx + offset + 1 in
            # the expanded node list (after the left support of that span).
            insert_node_idx = span_idx + offset + 1
            insert_pos = insert_node_idx * 2
            result = np.insert(result, insert_pos, [0.0, 0.0])

        return result

    # ── Private helpers ───────────────────────────────────────────────────

    def _to_input(
        self,
        value: float,
        column: str,
        input_units: dict | None = None,
    ) -> float:
        """Convert *value* from SI (target units) to input units for *column*.

        Args:
            value: Value in SI / target units.
            column: Column name used to look up the target unit.
            input_units: Explicit unit mapping to use.  When ``None`` the
                mapping from ``self._section_array.input_units`` is used as a
                fallback (only correct when operating on the instance's own
                section array).
        """
        target = SectionArray.target_units[column]
        units = (
            input_units
            if input_units is not None
            else self._section_array.input_units
        )
        inp = units.get(column, target)
        if inp == target:
            return value
        return float(Q_(value, target).to(inp).magnitude)

    def _apply_virtual_support_overlay(
        self,
        raw_data: pd.DataFrame,
        input_units: dict | None = None,
    ) -> pd.DataFrame:
        """Insert virtual support rows into *raw_data* (in input units)."""

        if self._virtual_support_overlay is None:
            logger.warning(
                "_apply_virtual_support_overlay called but no virtual support overlay is set; returning original data."
            )
            return raw_data
        sorted_keys = sorted(self._virtual_support_overlay.keys())
        for offset, span_idx in enumerate(sorted_keys):
            vs = self._virtual_support_overlay[span_idx]
            effective_idx = span_idx + offset

            x = vs["x"]
            y = vs["y"]
            angle = np.arctan2(y, x)  # radians

            original_span_input = cast(
                float, raw_data.loc[effective_idx, "span_length"]
            )
            x_input = self._to_input(x, "span_length", input_units)

            # Modify left support
            raw_data.loc[effective_idx, "span_length"] = x_input
            raw_data.loc[effective_idx, "line_angle"] = self._to_input(
                angle, "line_angle", input_units
            )
            # no load if adding a virtual support in the middle of the span
            if "load_mass" in raw_data.columns:
                raw_data.loc[effective_idx, "load_mass"] = 0.0
                raw_data.loc[effective_idx, "load_position"] = 0.0

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
                        float(vs["z"]),
                        "conductor_attachment_altitude",
                        input_units,
                    ),
                    "crossarm_length": self._to_input(
                        0.0, "crossarm_length", input_units
                    ),
                    "line_angle": self._to_input(
                        -angle, "line_angle", input_units
                    ),
                    "insulator_length": self._to_input(
                        max(float(vs["insulator_length"]), 0.01),
                        "insulator_length",
                        input_units,
                    ),
                    "span_length": remaining_span,
                    "insulator_mass": self._to_input(
                        float(vs["insulator_mass"]),
                        "insulator_mass",
                        input_units,
                    ),
                }
            )

            for optional_col, fill in (
                ("load_mass", 0.0),
                ("load_position", 0.0),
                ("counterweight_mass", 0.0),
                (
                    "sagging_parameter",
                    raw_data.loc[effective_idx, "sagging_parameter"],
                ),
                (
                    "sagging_temperature",
                    raw_data.loc[effective_idx, "sagging_temperature"],
                ),
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
            raw_data = pd.concat([top, virtual_df, bottom], ignore_index=True)

        return raw_data
