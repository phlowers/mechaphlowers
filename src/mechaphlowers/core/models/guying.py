# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Literal

import numpy as np

from mechaphlowers.core.models.balance.engine import BalanceEngine


class GuyingLoads:
    """GuyingLoads is a class that allows to calculate the loads on the guying system of a support."""

    def __init__(self, balance_engine: BalanceEngine):
        """Initialize GuyingLoads with a BalanceEngine instance.

        Args:
            balance_engine (BalanceEngine): balance engine instance
        """
        self.balance_engine = balance_engine
        self.counterweight = 0.0
        self.bundle_number = 1.0

    def get_guying_loads(
        self,
        num_guying: int,
        with_pulley: bool,
        guying_height: float,
        guying_horizontal_distance: float,
        side: Literal['left', 'right'] = 'left',
    ) -> dict:
        """Calculate guying system loads and forces.

        Args:
            num_guying (int): guying number
            with_pulley (bool): whether the guying system uses a pulley. Warning: if with_pulley is True, num_guying must be a suspension support (between 1 and number of spans - 2)            guying_height (float): guying attachment height
            guying_horizontal_distance (float): horizontal distance to guying attachment

        Returns:
            dict: Results containing guying_load, vertical_load, angle

        Raises:
            ValueError: If with_pulley is True and num_guying is not a suspension support.
        """
        span_shape = self.balance_engine.support_number

        if with_pulley and (num_guying == 0 or num_guying >= span_shape - 1):
            raise ValueError(
                "With pulley, guying number must be between 1 and number of spans - 2"
            )
        if side == 'left':
            vhl_left = (
                self.balance_engine.balance_model.vhl_under_chain_right()
            )
            vhl_v = vhl_left.V.value('N')[num_guying]
            vhl_h = vhl_left.H.value('N')[num_guying]
            vhl_l = vhl_left.L.value('N')[num_guying]

        elif side == 'right':
            vhl_right = (
                self.balance_engine.balance_model.vhl_under_chain_left()
            )
            vhl_v = vhl_right.V.value('N')[num_guying]
            vhl_h = vhl_right.H.value('N')[num_guying]
            vhl_l = vhl_right.L.value('N')[num_guying]

        else:
            raise AttributeError("side must be 'left' or 'right'")

        slope = self.balance_engine.span_model.slope(side)[num_guying]
        span_tension = self.balance_engine.span_model.T_h()[num_guying]

        if with_pulley:
            return self.static_calculate_guying_loads_with_pulley(
                vhl_v_g=vhl_v,
                alt_acc=self.balance_engine.section_array.data.conductor_attachment_altitude.iloc[
                    num_guying
                ],
                guying_height=guying_height,
                guying_horizontal_distance=guying_horizontal_distance,
                insulator_mass=self.balance_engine.section_array.data.insulator_weight.iloc[
                    num_guying
                ],
                counterweight=self.counterweight,
                cable_linear_weight=self.balance_engine.cable_array.data.linear_weight.iloc[
                    0
                ],
                bundle_number=self.bundle_number,
                span_tension=span_tension,
                span_slope_left=slope,
            )
        else:
            return self.static_calculate_guying_loads(
                # num_guying=num_guying,
                # num_profil=num_guying,
                # is_suspension=self.balance_engine.section_array.data.suspension.iloc[
                #     num_guying
                # ],
                # vhl_h_d=vhl_h_d,
                # vhl_l_d=vhl_l_d,
                # vhl_v_d=vhl_v_d,
                vhl_h_g=vhl_h,
                vhl_l_g=vhl_l,
                vhl_v_g=vhl_v,
                alt_acc=self.balance_engine.section_array.data.conductor_attachment_altitude.iloc[
                    num_guying
                ],
                # with_pulley=with_pulley,
                guying_height=guying_height,
                guying_horizontal_distance=guying_horizontal_distance,
                insulator_mass=self.balance_engine.section_array.data.insulator_weight.iloc[
                    num_guying
                ],
                counterweight=self.counterweight,
                cable_linear_weight=self.balance_engine.cable_array.data.linear_weight.iloc[
                    0
                ],
                bundle_number=self.bundle_number,
            )

    @staticmethod
    def static_calculate_guying_loads(
        vhl_h_g: float,
        vhl_l_g: float,
        vhl_v_g: float,
        alt_acc: float,
        guying_height: float,  # TextBox21
        guying_horizontal_distance: float,  # TextBox22
        insulator_mass: float,  # chain weight
        counterweight: float,  # counter weight
        cable_linear_weight: float,  # linear weight of conductor
        bundle_number: float,  # bundle coefficient
    ) -> dict:
        """
        Calculate guying system loads and forces.

        Args:
            num_haubanage: guying number
            num_profil: profile number
            vhl_h_d: horizontal load right
            vhl_l_d: lateral load right
            vhl_h_g: horizontal load left
            vhl_l_g: lateral load left
            vhl_v_d: vertical load right
            vhl_v_g: vertical load left
            alt_acc: attachment altitude
            guying_height: guying attachment height
            horizontal_distance: horizontal distance to guying attachment
            pds_chaine: chain weight (kg/m)
            contre_pds: counter weight (kg/m)
            pds_lin_conducteur: linear weight of conductor
            faisceau: bundle coefficient

        Returns:
            dict: Results containing guying_load, vertical_load, angle
        """

        # Calculate resultant of H and L loads for guying system compensation
        result_h_l = np.round(
            np.sqrt(vhl_h_g**2 + vhl_l_g**2), 0
        )  # same here (/10)

        # Calculate altitude difference between chain attachment and guying attachment
        delta_alt = alt_acc - guying_height

        # Calculate the load in the guying cable
        # (horizontal_distance = horizontal distance to guying point)
        guying_load = (
            result_h_l
            / guying_horizontal_distance
            * np.sqrt(guying_horizontal_distance**2 + delta_alt**2)
        )
        guying_load_rounded = np.round(guying_load, 0)

        # Calculate total vertical load under console
        # (Assumption: linear weight of guying = linear weight of cable * cable/bundle)
        charge_v = np.round(vhl_v_g, 0)

        # Total vertical load = base + chain weight + counter weight
        #                      + vertical component of guying load
        #                      + guying cable self-weight

        #  warning here, /10 means unit conversion from N to daN
        charge_v = np.round(
            charge_v
            + insulator_mass
            + counterweight
            + (result_h_l / guying_horizontal_distance) * delta_alt
            + np.sqrt(guying_horizontal_distance**2 + delta_alt**2)
            * cable_linear_weight
            * bundle_number,
            0,
        )

        # Calculate angle of guying cable with horizontal (in degrees)
        if guying_load_rounded != 0:
            guying_angle_rad = np.arccos(result_h_l / guying_load_rounded)
            guying_angle_deg = np.round(np.degrees(guying_angle_rad), 1)
        else:
            guying_angle_deg = 0.0

        # Longitudinal load (L) is zero
        charge_l = 0.0

        return {
            "guying_load": guying_load_rounded,
            "vertical_load": charge_v,
            "longitudinal_load": charge_l,
            "guying_angle_degrees": guying_angle_deg,
            "delta_altitude": delta_alt,
            "resultant_h_l": result_h_l,
        }

    @staticmethod
    def static_calculate_guying_loads_with_pulley(
        vhl_v_g: float,
        alt_acc: float,
        guying_height: float,  # TextBox21
        guying_horizontal_distance: float,  # TextBox22
        insulator_mass: float,  # chain weight (pds_chaine)
        counterweight: float,  # counter weight (contre_pds)
        cable_linear_weight: float,  # linear weight of conductor (pds_lin)
        bundle_number: float,  # bundle coefficient (faisceau)
        span_tension: float,  # span horizontal tension (th)
        span_slope_left: float,  # left span slope in radians (pente_g)
    ) -> dict:
        """
        Calculate guying system loads and forces WITH PULLEY.

        This method calculates the tension in the bundle at the pulley point,
        which becomes the tension in the guying cable. It includes lateral load
        calculation which is specific to pulley configuration.

        Args:
            num_guying: guying profile number
            num_profil: reference profile number (used to select slope direction)
            vhl_v_d: vertical load right (daN)
            vhl_v_g: vertical load left (daN)
            alt_acc: attachment altitude of chain (m)
            guying_height: attachment altitude of guying cable (m)
            guying_horizontal_distance: horizontal distance to guying attachment point (m)
            insulator_mass: chain weight (daN)
            counterweight: counter weight (daN)
            cable_linear_mass: linear weight of conductor (kg/m)
            bundle_number: bundle coefficient (dimensionless)
            span_tension: horizontal tension in span (daN)
            span_slope_left: left span slope angle (radians)
            span_slope_right: right span slope angle (radians)

        Returns:
            dict: Calculated loads and angles containing:
                - "guying_load": tension in guying cable (daN)
                - "vertical_load": total vertical load under console (daN)
                - "lateral_load": lateral load component (daN)
                - "angle_degrees": angle of guying cable with horizontal plane (degrees)
                - "delta_altitude": altitude difference between chain and guying attachments (m)
        """

        # ===== CALCULATION WITH PULLEY =====

        # Calculate tension in bundle at pulley point
        slope_rad = span_slope_left  # left slope (pente_g)

        # Tension in guying cable = span tension / cos(slope) * bundle factor
        guying_load = span_tension / np.cos(slope_rad) * bundle_number
        guying_load_rounded = np.round(guying_load, 0)

        # Calculate altitude difference between chain attachment and guying attachment
        delta_alt = alt_acc - guying_height

        # Calculate angle of guying cable with horizontal plane (in degrees)
        # Using arctan for angle calculation with pulley configuration
        guying_angle_rad = np.arctan(delta_alt / guying_horizontal_distance)
        guying_angle_deg = np.round(np.degrees(guying_angle_rad), 1)

        # Calculate total vertical load under console
        charge_v = np.round(vhl_v_g, 0)

        # Add all vertical components:
        # - chain weight (insulator_mass)
        # - counter weight
        # - vertical component of guying load (sin of angle * tension)
        # - self-weight of guying cable
        cable_length = np.sqrt(guying_horizontal_distance**2 + delta_alt**2)
        charge_v = np.round(
            charge_v
            + insulator_mass
            + counterweight
            + guying_load * np.sin(guying_angle_rad)
            + cable_length * cable_linear_weight * bundle_number,
            0,
        )

        # Calculate lateral load (L)
        # = bundle tension - horizontal component of guying load
        charge_l = span_tension * bundle_number - guying_load * np.cos(
            guying_angle_rad
        )
        charge_l_rounded = np.round(charge_l, 0)

        return {
            "guying_load": guying_load_rounded,
            "vertical_load": charge_v,
            "longitudinal_load": charge_l_rounded,
            "guying_angle_degrees": guying_angle_deg,
            "delta_altitude": delta_alt,
        }
