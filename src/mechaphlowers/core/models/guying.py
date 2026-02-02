# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from numbers import Real
from typing import Literal

import numpy as np
from pint import Quantity

from mechaphlowers.config import options
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.data.units import Q_


class GuyingResults:
    """Class to store guying loads results."""

    atol_map: dict[str, tuple[float, str]] = {
        "guying_tension": (5.0, "daN"),
        "vertical_force": (5.0, "daN"),
        "longitudinal_force": (5.0, "daN"),
        "guying_angle_degrees": (0.1, "degree"),
    }

    def __init__(
        self,
        guying_tension: Quantity,
        vertical_force: Quantity,
        longitudinal_force: Quantity,
        guying_angle_degrees: Quantity,
    ):
        """Initialize GuyingResults with calculated values.

        Args:
            guying_tension (Quantity): Tension in guying cable
            vertical_force (Quantity): Total vertical force under console
            longitudinal_force (Quantity): Longitudinal load component
            guying_angle_degrees (Quantity): Angle of guying cable with horizontal plane

        Raises:
            TypeError: If any of the inputs are not of type Quantity

        Examples:
            >>> from pint import UnitRegistry
            >>> ureg = UnitRegistry()
            >>> guying_tension = 1000 * ureg.newton
            >>> vertical_force = 500 * ureg.newton
            >>> longitudinal_force = 200 * ureg.newton
            >>> guying_angle_degrees = 30 * ureg.degree
            >>> results = GuyingResults(
            ...     guying_tension, vertical_force, longitudinal_force, guying_angle_degrees
            ... )
            >>> print(results)
            Guying Load: 1000.0 newton
            Vertical Load: 500.0 newton
            Longitudinal Load: 200.0 newton
            Guying Angle (degrees): 30.0 degree

        """
        for name, qty in {
            "guying_tension": guying_tension,
            "vertical_force": vertical_force,
            "longitudinal_force": longitudinal_force,
            "guying_angle_degrees": guying_angle_degrees,
        }.items():
            if not isinstance(qty, Quantity):
                raise TypeError(f"{name} must be a Quantity")
        self.guying_tension = guying_tension
        self.vertical_force = vertical_force
        self.longitudinal_force = longitudinal_force
        self.guying_angle_degrees = guying_angle_degrees

        self.output_unit_force = options.output_units.force
        self.output_unit_length = options.output_units.length

    @property
    def value_dict(self) -> dict[str, float]:
        """Return the values as a dictionary with appropriate units."""
        return {
            "guying_tension": self.guying_tension.to(self.output_unit_force).m,
            "vertical_force": self.vertical_force.to(self.output_unit_force).m,
            "longitudinal_force": self.longitudinal_force.to(
                self.output_unit_force
            ).m,
            "guying_angle_degrees": self.guying_angle_degrees.to("deg").m,
        }

    @property
    def quantity_dict(self) -> dict[str, Quantity]:
        """Return the values as a dictionary with appropriate units."""
        return {
            "guying_tension": self.guying_tension.to(self.output_unit_force),
            "vertical_force": self.vertical_force.to(self.output_unit_force),
            "longitudinal_force": self.longitudinal_force.to(
                self.output_unit_force
            ),
            "guying_angle_degrees": self.guying_angle_degrees.to("deg"),
        }

    @property
    def str_repr(self) -> str:
        """Return a string representation of the results."""
        return (
            f"Guying Tension: {self.guying_tension.to(self.output_unit_force)}\n"
            f"Vertical Load: {self.vertical_force.to(self.output_unit_force)}\n"
            f"Longitudinal Load: {self.longitudinal_force.to(self.output_unit_force)}\n"
            f"Guying Angle (degrees): {self.guying_angle_degrees.to('deg')}\n"
        )

    def __repr__(self) -> str:
        """Return a detailed representation of the results."""
        return f"<GuyingResults>\n{self.str_repr}"

    def __str__(self) -> str:
        """Return a user-friendly string representation of the results."""
        return self.str_repr

    def __eq__(self, value: object) -> bool:
        if isinstance(value, GuyingResults):
            for attr_name in self.value_dict.keys():
                if abs(
                    getattr(self, attr_name) - getattr(value, attr_name)
                ) > Q_(*self.atol_map[attr_name]):
                    return False
        else:
            raise TypeError(
                "Comparison is only supported between GuyingResults instances"
            )

        return True


class Guying:
    """Guying is a class that allows to calculate the loads on the guying system of a support."""

    def __init__(self, balance_engine: BalanceEngine):
        """Initialize Guying with a BalanceEngine instance.

        Args:
            balance_engine (BalanceEngine): balance engine instance
        """
        self.balance_engine = balance_engine
        self.counterweight = 0.0  # TODO: link later with balance_engine.section_array.data.counterweight
        self.bundle_number = 1.0  # TODO: link later with balance_engine.section_array.data.bundle_number

    def get_guying_results(
        self,
        support_index: int,
        with_pulley: bool,
        guying_altitude: float,
        guying_horizontal_distance: float,
        guying_side: Literal['left', 'right'] = 'left',
    ) -> GuyingResults:
        """Calculate guying system loads and forces.

        Args:
            support_index (int): Index of the support (0 to number of supports - 1)
            with_pulley (bool): Whether the guying system uses a pulley. If True, support_index must be a suspension support (between 1 and number of spans - 2)
            guying_altitude (float): Guying cable attachment height (m)
            guying_horizontal_distance (float): Horizontal distance to guying attachment point (m)
            guying_side (Literal['left', 'right']): Side of the guying system ('left' or 'right')

        Returns:
            GuyingResults: Results containing guying_tension, vertical_force, angle

        Raises:
            ValueError: If with_pulley is True and support_index is not a suspension support.
            TypeError: If input types are incorrect.
            AttributeError: If guying_side is not 'left' or 'right'.

        Examples:
            >>> from mechaphlowers.core.models.balance.engine import BalanceEngine
            >>> from mechaphlowers.core.models.guying import Guying
            >>> balance_engine = BalanceEngine(...)  # Initialize with appropriate parameters
            >>> guying_calculator = Guying(balance_engine)
            >>> results = guying_calculator.get_guying_results(
            ...     support_index=2,
            ...     with_pulley=False,
            ...     guying_altitude=30.0,
            ...     guying_horizontal_distance=50.0,
            ...     guying_side='left',
            ... )
            >>> print(results)
            Guying Load: 1234.0 N
            Vertical Load: 567.0 N
            Longitudinal Load: 89.0 N
            Guying Angle (degrees): 25.0 deg
        """
        if not isinstance(support_index, int):
            raise TypeError("support_index must be int")
        if not isinstance(with_pulley, bool):
            raise TypeError("with_pulley must be bool")
        for name, value in {
            "guying_altitude": guying_altitude,
            "guying_horizontal_distance": guying_horizontal_distance,
        }.items():
            if not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")

        span_shape = self.balance_engine.support_number

        # counterweight = self.balance_engine.section_array.data.counterweight.iloc[
        #     support_index
        # ]

        if with_pulley and (
            support_index == 0 or support_index >= span_shape - 1
        ):
            raise ValueError(
                "With pulley, guying number must be between 1 and number of spans - 2"
            )

        if support_index < 0 or support_index >= span_shape:
            raise ValueError(
                f"support_index must be between 0 and {span_shape - 1}"
            )

        if guying_side == 'left':
            vhl_right = (
                self.balance_engine.balance_model.vhl_under_chain_right()
            )
            vhl_v = vhl_right.V.value('N')[support_index]
            vhl_h = vhl_right.H.value('N')[support_index]
            vhl_l = vhl_right.L.value('N')[support_index]

        elif guying_side == 'right':
            vhl_left = self.balance_engine.balance_model.vhl_under_chain_left()
            vhl_v = vhl_left.V.value('N')[support_index]
            vhl_h = vhl_left.H.value('N')[support_index]
            vhl_l = vhl_left.L.value('N')[support_index]

        else:
            raise AttributeError("side must be 'left' or 'right'")

        slope = self.balance_engine.span_model.slope(guying_side)[
            support_index
        ]
        span_tension = self.balance_engine.span_model.T_h()[support_index]

        if with_pulley:
            return self.static_calculate_guying_loads_with_pulley(
                vhl_v=vhl_v,
                attachment_altitude=self.balance_engine.balance_model.attachment_altitude_after_solve[
                    support_index
                ],
                guying_altitude=guying_altitude,
                guying_horizontal_distance=guying_horizontal_distance,
                insulator_weight=self.balance_engine.section_array.data.insulator_weight.iloc[
                    support_index
                ],
                counterweight=self.counterweight,
                cable_linear_weight=self.balance_engine.cable_array.data.linear_weight.iloc[
                    0
                ],
                bundle_number=self.bundle_number,
                span_tension=span_tension,
                span_slope=slope,
            )
        else:
            return self.static_calculate_guying_loads(
                vhl_h=vhl_h,
                vhl_l=vhl_l,
                vhl_v=vhl_v,
                attachment_altitude=self.balance_engine.balance_model.attachment_altitude_after_solve[
                    support_index
                ],
                guying_altitude=guying_altitude,
                guying_horizontal_distance=guying_horizontal_distance,
                insulator_weight=self.balance_engine.section_array.data.insulator_weight.iloc[
                    support_index
                ],
                counterweight=self.counterweight,
                cable_linear_weight=self.balance_engine.cable_array.data.linear_weight.iloc[
                    0
                ],
                bundle_number=self.bundle_number,
            )

    @staticmethod
    def static_calculate_guying_loads(
        vhl_h: float,
        vhl_l: float,
        vhl_v: float,
        attachment_altitude: float,
        guying_altitude: float,
        guying_horizontal_distance: float,
        insulator_weight: float,  # chain weight
        counterweight: float,  # counter weight
        cable_linear_weight: float,  # linear weight of conductor
        bundle_number: float,  # bundle coefficient
    ) -> GuyingResults:
        """
        Calculate guying system loads and forces (direct guying without pulley).

        Args:
            vhl_h (float): Horizontal load component (N)
            vhl_l (float): Lateral/longitudinal load component (N)
            vhl_v (float): Vertical load component (N)
            attachment_altitude (float): Attachment altitude of chain (m)
            guying_altitude (float): Guying cable attachment height (m)
            guying_horizontal_distance (float): Horizontal distance to guying attachment point (m)
            insulator_weight (float): Chain/insulator weight (N)
            counterweight (float): Counter weight (N)
            cable_linear_weight (float): Linear weight of conductor (N/m)
            bundle_number (float): Bundle coefficient (dimensionless)

        Returns:
            GuyingResults: Results containing guying_tension, vertical_force, longitudinal_force, and guying_angle_degrees
        """

        # Calculate resultant of H and L loads for guying system compensation
        result_h_l = np.sqrt(vhl_h**2 + vhl_l**2)

        # Calculate altitude difference between chain attachment and guying attachment
        delta_alt = attachment_altitude - guying_altitude

        # Calculate the load in the guying cable
        # (horizontal_distance = horizontal distance to guying point)
        guying_tension = (
            result_h_l
            / guying_horizontal_distance
            * np.sqrt(guying_horizontal_distance**2 + delta_alt**2)
        )

        # Calculate total vertical load under console
        # Total vertical load = base + chain weight + counter weight
        #                      + vertical component of guying load
        #                      + guying cable self-weight

        #  warning here, /10 means unit conversion from N to daN
        charge_v = np.round(
            vhl_v
            + insulator_weight
            + counterweight
            + (result_h_l / guying_horizontal_distance) * delta_alt
            + np.sqrt(guying_horizontal_distance**2 + delta_alt**2)
            * cable_linear_weight
            * bundle_number,
            0,
        )

        # Calculate angle of guying cable with horizontal (in degrees)
        if guying_tension != 0:
            guying_angle_rad = np.arccos(result_h_l / guying_tension)
            guying_angle_deg = np.round(np.degrees(guying_angle_rad), 1)
        else:
            guying_angle_deg = 0.0

        # Longitudinal load (L) is zero
        charge_l = 0.0

        return GuyingResults(
            **{
                "guying_tension": Q_(np.round(guying_tension, 0), "N"),
                "vertical_force": Q_(np.round(charge_v, 0), "N"),
                "longitudinal_force": Q_(charge_l, "N"),
                "guying_angle_degrees": Q_(guying_angle_deg, "deg"),
            }
        )

    @staticmethod
    def static_calculate_guying_loads_with_pulley(
        vhl_v: float,
        attachment_altitude: float,
        guying_altitude: float,
        guying_horizontal_distance: float,
        insulator_weight: float,  # chain weight (pds_chaine)
        counterweight: float,  # counter weight (contre_pds)
        cable_linear_weight: float,  # linear weight of conductor (pds_lin)
        bundle_number: float,  # bundle coefficient (faisceau)
        span_tension: float,  # span horizontal tension (th)
        span_slope: float,  # span slope in radians (pente_g)
    ) -> GuyingResults:
        """
        Calculate guying system loads and forces with pulley.

        This method calculates the tension in the bundle at the pulley point,
        which becomes the tension in the guying cable. It includes longitudinal load
        calculation which is specific to pulley configuration.

        Args:
            vhl_v (float): Vertical load component (N)
            attachment_altitude (float): Attachment altitude of chain (m)
            guying_altitude (float): Guying cable attachment height (m)
            guying_horizontal_distance (float): Horizontal distance to guying attachment point (m)
            insulator_weight (float): Chain/insulator weight (N)
            counterweight (float): Counter weight (N)
            cable_linear_weight (float): Linear weight of conductor (N/m)
            bundle_number (float): Bundle coefficient (dimensionless)
            span_tension (float): Horizontal tension in span (N)
            span_slope (float): Span slope angle (radians)

        Returns:
            GuyingResults: Results containing:
                <ul>
                    <li>guying_tension: tension in guying cable (daN)</li>
                    <li>vertical_force: total vertical load under console (daN)</li>
                    <li>longitudinal_force: longitudinal load component (daN)</li>
                    <li>guying_angle_degrees: angle of guying cable with horizontal plane (degrees)</li>
                </ul>
        """

        # ===== CALCULATION WITH PULLEY =====

        # Calculate tension in bundle at pulley point
        slope_rad = span_slope  # left slope (pente_g)

        # Tension in guying cable = span tension / cos(slope) * bundle factor
        guying_tension = span_tension / np.cos(slope_rad) * bundle_number

        # Calculate altitude difference between chain attachment and guying attachment
        delta_alt = attachment_altitude - guying_altitude

        # Calculate angle of guying cable with horizontal plane (in degrees)
        # Using arctan for angle calculation with pulley configuration
        guying_angle_rad = np.arctan(delta_alt / guying_horizontal_distance)
        guying_angle_deg = np.round(np.degrees(guying_angle_rad), 1)

        # Calculate total vertical load under console
        charge_v = np.round(vhl_v, 0)

        # Add all vertical components:
        # - chain weight
        # - counter weight
        # - vertical component of guying load (sin of angle * tension)
        # - self-weight of guying cable
        cable_length = np.sqrt(guying_horizontal_distance**2 + delta_alt**2)
        charge_v = np.round(
            charge_v
            + insulator_weight
            + counterweight
            + guying_tension * np.sin(guying_angle_rad)
            + cable_length * cable_linear_weight * bundle_number,
            0,
        )

        # Calculate lateral load (L)
        # = bundle tension - horizontal component of guying load
        charge_l = span_tension * bundle_number - guying_tension * np.cos(
            guying_angle_rad
        )

        return GuyingResults(
            **{
                "guying_tension": Q_(np.round(guying_tension, 0), "N"),
                "vertical_force": Q_(np.round(charge_v, 0), "N"),
                "longitudinal_force": Q_(np.round(charge_l, 0), "N"),
                "guying_angle_degrees": Q_(guying_angle_deg, "deg"),
            }
        )
