# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations

import logging
from typing import Tuple, Type

import numpy as np
import pandas as pd

from mechaphlowers.config import options
from mechaphlowers.core.models.balance.interfaces import (
    IBalanceModel,
    IModelForSolver,
)
from mechaphlowers.core.models.balance.models.utils_model_ducloux import (
    Masks,
    VectorProjection,
)
from mechaphlowers.core.models.balance.solvers.balance_solver import (
    BalanceSolver,
)
from mechaphlowers.core.models.balance.solvers.find_parameter_solver import (
    FindParamModel,
    FindParamSolverForLoop,
    IFindParamSolver,
)
from mechaphlowers.core.models.cable.deformation import (
    IDeformation,
    deformation_model_builder,
)
from mechaphlowers.core.models.cable.span import CatenarySpan, ISpan
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.entities.core import VhlStrength
from mechaphlowers.numeric import cubic
from mechaphlowers.utils import arr

logger = logging.getLogger(__name__)


class BalanceModel(IBalanceModel):
    """Model for solving the balance of the chains.
    Used by the BalanceSolver class in order to solve adjustement and change state cases.

    """

    def __init__(
        self,
        sagging_temperature: np.ndarray,
        parameter: np.ndarray,
        section_array: SectionArray,
        cable_array: CableArray,
        span_model: ISpan,
        deformation_model: IDeformation,
        cable_loads: CableLoads,
        find_param_solver_type: Type[
            IFindParamSolver
        ] = FindParamSolverForLoop,
    ) -> None:
        # tempertaure and parameter size n-1 here
        self.sagging_temperature = sagging_temperature
        self.parameter = parameter
        self.nodes = nodes_builder(section_array)
        self.cable_array = cable_array
        # TODO: temporary solution to manage cable data, must find a better way to do this
        self.cable_section = np.float64(self.cable_array.data.section.iloc[0])
        self.diameter = np.float64(self.cable_array.data.diameter.iloc[0])
        self.linear_weight = np.float64(
            self.cable_array.data.linear_weight.iloc[0]
        )
        self.young_modulus = np.float64(
            self.cable_array.data.young_modulus.iloc[0]
        )
        self.dilatation_coefficient = np.float64(
            self.cable_array.data.dilatation_coefficient.iloc[0]
        )
        self.temperature_reference = np.float64(
            self.cable_array.data.temperature_reference.iloc[0]
        )
        self.span_model = span_model
        self.deformation_model = deformation_model
        self.cable_loads = cable_loads

        self.a: np.ndarray
        self.b: np.ndarray
        self.Th: np.ndarray
        self.Tv_d: np.ndarray
        self.Tv_g: np.ndarray

        self._adjustment: bool = True
        # TODO: during adjustment computation, perhaps set cable_temperature = 0
        # temperature here is tuning temperature / only in the real span part
        # there is another temperature : change state

        self.load_model = LoadModel(
            self.cable_array,
            self.nodes.load_weight[self.nodes.has_load_on_span],
            self.nodes.load_position[self.nodes.has_load_on_span],
            arr.decr(self.span_model.span_length)[self.nodes.has_load_on_span],
            arr.decr(self.span_model.elevation_difference)[
                self.nodes.has_load_on_span
            ],
            self.k_load[self.nodes.has_load_on_span],
            self.sagging_temperature[self.nodes.has_load_on_span],
            self.parameter[self.nodes.has_load_on_span],
        )
        self.find_param_model = FindParamModel(
            self.span_model, self.deformation_model
        )
        self.find_param_solver = find_param_solver_type(self.find_param_model)
        self.load_solver = BalanceSolver(
            **options.solver.balance_solver_load_params
        )
        self.update()
        self.nodes.compute_dx_dy_dz()
        self.nodes.vector_projection.set_tensions(
            self.Th, self.Tv_d, self.Tv_g
        )
        self.nodes.compute_moment()

    @property
    def adjustment(self) -> bool:
        """Boolean property indicating if the model is in adjustment mode or change state mode."""
        return self._adjustment

    @adjustment.setter
    def adjustment(self, value: bool) -> None:
        self._adjustment = value

    def dxdydz(self) -> np.ndarray:
        """Get the displacement vector of the nodes."""
        return self.nodes.dxdydz.T

    @property
    def k_load(self) -> np.ndarray:
        # TODO: fix length array issues with mechaphlowers
        """Load coefficient, taking into account wind and ice.
        Last value removed because it is a NaN.
        """
        return arr.decr(self.cable_loads.load_coefficient)

    @property
    def alpha(self) -> np.ndarray:
        beta = self.beta
        return (
            -np.acos(
                self.a / (self.a**2 + self.b**2 * np.sin(beta) ** 2) ** 0.5
            )
            * np.sign(beta)
            * np.sign(self.b)
        )

    @property
    def beta(self) -> np.ndarray:
        """Angle between the cable plane and the vertical plane, created because of wind, in radians.
        Remove last value for consistency between this and mechaphlowers

        mechaphlowers: `beta = [beta0, beta1, beta2, nan]`

        Here: `beta = [beta0, beta1, beta2]` because the last value refers to the last support (no span related to this support)
        """
        # TODO: check why sign different from what already exists in CableLoads
        return -arr.decr(self.cable_loads.load_angle)

    def update_L_ref(self) -> np.ndarray:
        """Update the reference length L_ref, after an adjustment computation.

        Returns:
            np.ndarray: unstressed length, with reference temperature at 0°C.
        """
        self.span_model.compute_values()

        self.deformation_model.tension_mean = self.span_model.T_mean()
        self.deformation_model.cable_length = self.span_model.L
        self.deformation_model.current_temperature = arr.incr(
            self.sagging_temperature
        )

        L_0 = arr.decr(self.deformation_model.L_0())
        self.L_ref = L_0
        return L_0

    @property
    def a_prime(self) -> np.ndarray:
        """Span length after wind angle into taking account"""
        return (self.a**2 + self.b**2 * np.sin(self.beta) ** 2) ** 0.5

    @property
    def b_prime(self) -> np.ndarray:
        """Elevation difference after wind angle into taking account"""
        return self.b * np.cos(self.beta)

    def compute_Th_and_extremum(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run SagTensionSolver to compute parameter, and then compute Th, x_m and x_n.
        Used only in change_state case.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Th, x_m, x_n, parameter
        """
        # First run approx_parameter to have a closer first value of the parameter. This uses the parabola model.
        parameter_parabola = approx_parameter(
            self.a_prime,
            self.b_prime,
            self.L_ref,
            self.k_load,
            self.cable_section,
            self.linear_weight,
            self.young_modulus,
            self.dilatation_coefficient,
            self.sagging_temperature,
        )
        parameter_parabola = arr.incr(parameter_parabola)
        self.find_param_model.set_attributes(
            initial_parameter=parameter_parabola,
            L_ref=arr.incr(self.L_ref),
        )

        parameter = self.find_param_solver.find_parameter()

        self.span_model.set_parameter(parameter)
        Th = arr.decr(self.span_model.T_h())
        x_m = arr.decr(self.span_model.x_m)
        x_n = arr.decr(self.span_model.x_n)
        return Th, x_m, x_n, arr.decr(parameter)

    def update_tensions(self) -> np.ndarray:
        """Compute values of `Th`, `Tv_d` and `Tv_g`.

        There are three cases:

        - adjustment: simple computation

        - change state without loads: run SagTensionSolver once

        - change state with loads: solve LoadModel
        """
        # Case: adjustment
        if self._adjustment:
            Th = arr.decr(self.span_model.T_h())
            x_m = arr.decr(self.span_model.compute_x_m())
            x_n = arr.decr(self.span_model.compute_x_n())

        # Case: change state + no load:
        else:
            Th, x_m, x_n, parameter = self.compute_Th_and_extremum()

            # Case: change state + with load
            if not self._adjustment and self.nodes.has_load_on_span.any():
                self.x_i = self.nodes.load_position * self.a_prime
                # TODO: Don't know why adding x_m is needed
                z_i_m = self.span_model.z_one_point(arr.incr(self.x_i + x_m))
                z_m = self.span_model.z_one_point(arr.incr(x_m))
                self.z_i = arr.decr(z_i_m - z_m)

                self.solve_xi_zi_loads()

                # Left Th and right Th are equal because balance just got solved (same for parameter)
                parameter[self.nodes.has_load_on_span] = (
                    self.load_model.span_model_left.sagging_parameter
                )
                Th[self.nodes.has_load_on_span] = (
                    self.load_model.span_model_left.T_h()
                )
                x_m[self.nodes.has_load_on_span] = (
                    self.load_model.span_model_left.x_m
                )
                x_n[self.nodes.has_load_on_span] = (
                    self.load_model.span_model_right.x_n
                )

            self.parameter = parameter
            self.span_model.set_parameter(arr.incr(parameter))

        self.Tv_g = arr.decr(self.span_model.T_v(arr.incr(x_m)))
        self.Tv_d = -arr.decr(self.span_model.T_v(arr.incr(x_n)))
        self.Th = Th
        self.update_projections()

        return Th

    def solve_xi_zi_loads(self) -> None:
        """Run solver to compute the position of the loads on the span.
        Also update the semi-span models in LoadModel: attributes span_model_left and span_model_right.
        They could be used for graphs.
        Only used in change_state case with loads.
        """
        self.load_model.set_all(
            self.L_ref[self.nodes.has_load_on_span],
            self.x_i[self.nodes.has_load_on_span],
            self.z_i[self.nodes.has_load_on_span],
            self.a_prime[self.nodes.has_load_on_span],
            self.b_prime[self.nodes.has_load_on_span],
            self.k_load[self.nodes.has_load_on_span],
            self.sagging_temperature[self.nodes.has_load_on_span],
        )

        self.load_solver.solve(self.load_model)

    def update_projections(self) -> None:
        """update attributes of VectorProjection in order to compute with correct values"""
        alpha = self.alpha
        beta = self.beta

        self.compute_inter()
        self.nodes.vector_projection.set_all(
            self.Th,
            self.Tv_d,
            self.Tv_g,
            alpha,
            beta,
            self.nodes.line_angle,
            self.proj_angle,
            self.nodes.insulator_weight,
        )

    def compute_inter(self) -> None:
        # TODO: understand this and write docstring
        # warning: counting from right to left
        proj_d_i = self.nodes.proj_d
        proj_g_ip1 = np.roll(self.nodes.proj_g, -1, axis=1)
        proj_diff = (proj_g_ip1 - proj_d_i)[:, :-1]
        # x: initial input to calculate span_length
        self.inter1: np.ndarray = self.nodes.span_length + proj_diff[0]
        self.inter2: np.ndarray = proj_diff[1]
        self.proj_angle: np.ndarray = np.atan2(self.inter2, self.inter1)

    def update_span(self) -> None:
        """Update span lengths and elevation differences in the span model.
        Those values may change because of chain position changing between iterations."""
        # warning for dev : we dont use the first element of span vectors for the moment
        z = self.nodes.z

        self.compute_inter()

        self.a = (self.inter1**2 + self.inter2**2) ** 0.5
        b = np.roll(z, -1) - z
        self.b = b[:-1]
        # TODO: fix array lengths?
        self.span_model.set_lengths(
            arr.incr(self.a_prime), arr.incr(self.b_prime)
        )

    def objective_function(self) -> np.ndarray:
        """Objective foncton to minimize: moments at the nodes.
        The balance position is found when every moment is equal to zero.

        Returns:
            np.ndarray: moments at the nodes. The format of the output is: `[Mx0, My0, Mx1, My1,...]`"""
        # Need to run update() before this method if necessary
        self.nodes.compute_moment()

        out = np.array([self.nodes.Mx, self.nodes.My]).flatten('F')
        return out

    @property
    def state_vector(self) -> np.ndarray:
        """State vector: displacements of chains that matters.
        For suspension, `dx` and `dy` are the variables. For anchors, they are `dz` and `dy`.

        Returns:
            np.ndarray: state vector. Format is `[dz_0, dy_0, dx_1, dy_1, ... , dz_n, dy_n]`
        """
        return self.nodes.state_vector

    @state_vector.setter
    def state_vector(self, value: np.ndarray) -> None:
        self.nodes.state_vector = value

    def update(self) -> None:
        # for the init only, update_span() needs to be call before update_tensions()
        # for other cases, order does not seem to matter
        self.update_span()
        self.update_tensions()

    def vhl_under_chain(self, output_unit: str = "daN") -> VhlStrength:
        V = -(self.nodes.Fz - self.nodes.insulator_weight / 2)
        vhl_result = np.array([V, self.nodes.Fy, self.nodes.Fx])
        return VhlStrength(vhl_result, "N")

    def vhl_under_console(self, output_unit: str = "daN") -> VhlStrength:
        vhl_result = np.array([self.nodes.Fz, self.nodes.Fy, self.nodes.Fx])
        return VhlStrength(vhl_result, "N")

    def dict_to_store(self) -> dict:
        return {
            "dx": self.nodes.dx,
            "dy": self.nodes.dy,
            "dz": self.nodes.dz,
        }

    def __repr__(self) -> str:
        data = {
            'parameter': self.parameter,
            'cable_temperature': self.sagging_temperature,
            'Th': self.Th,
            'Tv_d': self.Tv_d,
            'Tv_g': self.Tv_g,
        }
        out = pd.DataFrame(data)

        return str(out)

    def __str__(self) -> str:
        return self.__repr__()


def approx_parameter(
    a: np.ndarray,
    b: np.ndarray,
    L0: np.ndarray,
    k_load: np.ndarray,
    cable_section: np.float64,
    linear_weight: np.float64,
    young_modulus: np.float64,
    dilatation_coefficient: np.float64,
    cable_temperature: np.ndarray,
) -> np.ndarray:
    """Computes an approximation of the parameter using the parabola model.
    The solution is found by resolving a cubic equation, using the Cardano method.

    Args:
        a (np.ndarray): span length
        b (np.ndarray): elevation difference
        L0 (np.ndarray): unstressed length of cable
        k_load (np.ndarray): load coefficient
        cable_section (np.float64): area of the section of the cable
        linear_weight (np.float64): linear weight of the cable
        young_modulus (np.float64): young modulus of the cable
        dilatation_coefficient (np.float64): dilatation coefficient of the cable
        cable_temperature (np.ndarray): cable current temperature

    Returns:
        np.ndarray: approximated values of the parameter using the parabola model
    """
    # TODO: mathematical documentation
    circle_chord = (a**2 + b**2) ** 0.5

    factor = linear_weight * k_load / young_modulus / cable_section

    p3 = factor * L0
    p2 = L0 - circle_chord + dilatation_coefficient * cable_temperature * L0
    p1 = 0 * L0
    p0 = -(a**4) / 24 / circle_chord

    # we have to do p3 * x**3 + p2 * x**2 + p1 * x + p0 = 0
    # p = p3 | p2 | p1 | p0
    p = np.vstack((p3, p2, p1, p0)).T
    roots = cubic.cubic_roots(p)
    return roots.real


class Nodes:
    def __init__(
        self,
        insulator_length: np.ndarray,
        insulator_weight: np.ndarray,
        crossarm_length: np.ndarray,
        line_angle: np.ndarray,
        z: np.ndarray,
        span_length: np.ndarray,
        # load and load_position must of length nb_supports - 1
        load_weight: np.ndarray,
        load_position: np.ndarray,
    ):
        # TODO: docstring of this whole class

        self.insulator_length = insulator_length
        # insulator_weight: positive weight means downward force
        self.insulator_weight = -insulator_weight
        # arm length: positive length means further from observer
        self.crossarm_length = crossarm_length
        # line_angle: anti clockwise
        self.line_angle = line_angle
        self.init_coordinates(span_length, z)
        # dx, dy, dz are the distances between the attachment point and the arm, including the chain
        self.dxdydz = np.zeros((3, len(z)), dtype=np.float64)
        self.load_weight = load_weight
        self.load_position = load_position
        self.has_load_on_span = np.logical_and(
            load_weight != 0, load_position != 0
        )

        nodes_type = [
            "anchor_first"
            if i == 0
            else "anchor_last"
            if i == len(insulator_weight) - 1
            else "suspension"
            for i in range(len(insulator_weight))
        ]
        self.masks = Masks(nodes_type, self.insulator_length)

        self.vector_projection = VectorProjection()

    @property
    def dx(self) -> np.ndarray:
        return self.dxdydz[0]

    @dx.setter
    def dx(self, value: np.ndarray) -> None:
        self.dxdydz[0] = value

    @property
    def dy(self) -> np.ndarray:
        return self.dxdydz[1]

    @dy.setter
    def dy(self, value: np.ndarray) -> None:
        self.dxdydz[1] = value

    @property
    def dz(self) -> np.ndarray:
        return self.dxdydz[2]

    @dz.setter
    def dz(self, value: np.ndarray) -> None:
        self.dxdydz[2] = value

    def __len__(self):
        return len(self.insulator_length)

    @property
    def z(self) -> np.ndarray:
        """This property returns the altitude of the end the chain.  $z = z_arm + dz$"""
        return self.z_arm + self.dz

    @property
    def z_arm(self) -> np.ndarray:
        """This property returns the altitude of the end the arm. Should not be modified during computation."""
        return self._z0 - self.z_suspension_chain

    @property
    def proj_g(self) -> np.ndarray:
        arm_length = self.crossarm_length
        gamma = self.line_angle
        proj_s_axis = -(arm_length + self.dy) * np.sin(gamma / 2) + (
            self.dx
        ) * np.cos(gamma / 2)
        proj_t_axis = (arm_length + self.dy) * np.cos(gamma / 2) + (
            self.dx
        ) * np.sin(gamma / 2)

        return np.array([proj_s_axis, proj_t_axis])

    @property
    def proj_d(self) -> np.ndarray:
        arm_length = self.crossarm_length
        gamma = self.line_angle
        proj_s_axis = (arm_length + self.dy) * np.sin(gamma / 2) + (
            self.dx
        ) * np.cos(gamma / 2)
        proj_t_axis = (arm_length + self.dy) * np.cos(gamma / 2) - (
            self.dx
        ) * np.sin(gamma / 2)

        return np.array([proj_s_axis, proj_t_axis])

    @property
    def state_vector(self) -> np.ndarray:
        # [dz_0, dy_0, dx_1, dy_1, ... , dz_n, dy_n]
        dxdy = self.dxdydz[[0, 1], 1:-1]
        dzdy = self.dxdydz[[2, 1]][:, [0, -1]]
        return np.vstack([dzdy[:, 0], dxdy.T, dzdy[:, 1]]).flatten()

    @state_vector.setter
    def state_vector(self, state_vector: np.ndarray):
        # TODO: refactor with mask
        dzdxdz = state_vector[::2]
        dy = state_vector[1::2]

        self.dy = dy
        self.dx[1:-1] = dzdxdz[1:-1]
        self.dz[[0, -1]] = dzdxdz[[0, -1]]
        self.compute_dx_dy_dz()

    def init_coordinates(self, span_length: np.ndarray, z: np.ndarray) -> None:
        # unused code?
        # self.x_anchor_chain = np.zeros_like(z)
        # self.x_anchor_chain[0] = self.insulator_length[0]
        # self.x_anchor_chain[-1] = -self.insulator_length[-1]
        self.z_suspension_chain = np.zeros_like(z)
        self.z_suspension_chain[1:-1] = -self.insulator_length[1:-1]

        # warning: x0 and z0 does not mean the same thing
        # x0 is the absissa of the arm
        # z0 is the altitude of the attachement point
        self.span_length = span_length
        self._z0 = z
        self._y = np.zeros_like(z, dtype=np.float64)

    def compute_dx_dy_dz(self) -> None:
        """Update dx and dz according to insulator_length and the othre displacement, using Pytagoras.

        For anchor chains: update dx

        For suspension chains: update dz
        """
        self.dx, self.dz = self.masks.compute_dx_dy_dz(
            self.dx, self.dy, self.dz
        )

    def compute_moment(self) -> None:
        Fx, Fy, Fz = self.vector_projection.forces()

        lever_arm = np.array([self.dx, self.dy, self.dz]).T
        # size : (nb nodes , 3 for 3D)

        force_3d = np.vstack((Fx, Fy, Fz)).T

        M = np.cross(lever_arm, force_3d)
        Mx = M[:, 0]
        My = M[:, 1]
        # Mz is supposed to be equal to 0 on each span
        Mz = M[:, 2]
        self.Fx = Fx
        self.Fy = Fy
        self.Fz = Fz
        self.Mx = Mx
        self.My = My
        self.Mz = Mz

    def __repr__(self):
        data = {
            'dx': self.dx,
            'dy': self.dy,
            'dz': self.dz,
            'Fx': self.Fx,
            'Fy': self.Fy,
            'Fz': self.Fz,
            'Mx': self.Mx,
            'My': self.My,
        }
        out = pd.DataFrame(data)

        return str(out)

    def __str__(self):
        return self.__repr__()


def nodes_builder(section_array: SectionArray) -> Nodes:
    """Builds a Nodes object from a SectionArray by extracting and transforming data.

    Args:
        section_array (SectionArray): Section Array to extract the data from.

    Returns:
        Nodes: An instance of Nodes initialized with data from section array.

    line_angle is in radians.
    """

    insulator_length = section_array.data.insulator_length.to_numpy()
    insulator_weight = section_array.data.insulator_weight.to_numpy()
    crossarm_length = section_array.data.crossarm_length.to_numpy()
    line_angle = section_array.data.line_angle.to_numpy()
    z = section_array.data.conductor_attachment_altitude.to_numpy()
    span_length = arr.decr(section_array.data.span_length.to_numpy())
    load_weight = arr.decr(section_array.data.load_weight.to_numpy())
    load_position = arr.decr(section_array.data.load_position.to_numpy())
    return Nodes(
        insulator_length,
        insulator_weight,
        crossarm_length,
        line_angle,
        z,
        span_length,
        load_weight,
        load_position,
    )


class LoadModel(IModelForSolver):
    """Model for solving the position of the loads on the span.
    Used by the BalanceSolver class.
    Also creates two semi-span models (left and right of the load) for solving purposing, and that could be used later for graphs.
    """

    def __init__(
        self,
        cable_array: CableArray,
        load_weight: np.ndarray,
        load_position: np.ndarray,
        # placeholder values for initialization
        a: np.ndarray,
        b: np.ndarray,
        k_load: np.ndarray,
        temperature: np.ndarray,
        parameter: np.ndarray,
        find_param_solver_type: Type[
            IFindParamSolver
        ] = FindParamSolverForLoop,
    ):
        self.cable_array = cable_array
        self.cable_section = np.float64(self.cable_array.data.section.iloc[0])
        self.diameter = np.float64(self.cable_array.data.diameter.iloc[0])
        self.linear_weight = np.float64(
            self.cable_array.data.linear_weight.iloc[0]
        )
        self.young_modulus = np.float64(
            self.cable_array.data.young_modulus.iloc[0]
        )
        self.dilatation_coefficient = np.float64(
            self.cable_array.data.dilatation_coefficient.iloc[0]
        )
        self.load_weight = load_weight
        self.load_position = load_position
        self.find_param_solver_type = find_param_solver_type
        # TODO: Need a way to choose Span model

        # init objects with placeholder values, but they will be updated when needed
        self.span_model_left = CatenarySpan(
            a, b, parameter, k_load, self.linear_weight
        )
        self.span_model_right = CatenarySpan(
            a, b, parameter, k_load, self.linear_weight
        )
        self.deformation_model_left = deformation_model_builder(
            self.cable_array,
            self.span_model_left,
            temperature,
        )
        self.deformation_model_right = deformation_model_builder(
            self.cable_array,
            self.span_model_right,
            temperature,
        )
        self.find_param_model_left = FindParamModel(
            self.span_model_left, self.deformation_model_left
        )
        self.find_param_model_right = FindParamModel(
            self.span_model_right, self.deformation_model_right
        )
        self.find_param_solver_left = self.find_param_solver_type(
            self.find_param_model_left
        )
        self.find_param_solver_right = self.find_param_solver_type(
            self.find_param_model_right
        )

    def set_all(
        self,
        L_ref: np.ndarray,
        x_i: np.ndarray,
        z_i: np.ndarray,
        a_prime: np.ndarray,
        b_prime: np.ndarray,
        k_load: np.ndarray,
        temperature: np.ndarray,
    ) -> None:
        """Update attributes. This method is used frequently, at every iteration of the main loop

        Args:
            L_ref (np.ndarray): unstressed length of the cable
            x_i (np.ndarray): abscissa of the load on the span
            z_i (np.ndarray): height of the load on the span
            a_prime (np.ndarray): span length
            b_prime (np.ndarray): elevation difference
            k_load (np.ndarray): load coefficient
            temperature (np.ndarray): current temperature
        """
        # put L_ref in constructor?
        self.L_ref = L_ref
        self.x_i = x_i
        self.z_i = z_i
        self.a_prime = a_prime
        self.b_prime = b_prime
        self.k_load = k_load
        self.temperature = temperature
        # Warning: arrays legnth are n-1
        # Easier to use, but less consistent with other mechaphlowers objects
        self.update_objects()

    def update_objects(self) -> None:
        self.update_lengths_span_models()
        self.span_model_left.load_coefficient = self.k_load
        self.span_model_right.load_coefficient = self.k_load

        self.deformation_model_left.current_temperature = self.temperature
        self.deformation_model_right.current_temperature = self.temperature

    def update_lengths_span_models(self) -> None:
        self.span_model_left.span_length = self.x_i
        self.span_model_left.elevation_difference = self.z_i
        self.span_model_right.span_length = self.a_prime - self.x_i
        self.span_model_right.elevation_difference = self.b_prime - self.z_i

    @property
    def state_vector(self) -> np.ndarray:
        """State vector: position of the loads on the span.
        Returns:
            np.ndarray: state vector. Format is `[x_i0, z_i0, x_i1, z_i1,...]`
        """
        return np.array([self.x_i, self.z_i]).flatten('F')

    @state_vector.setter
    def state_vector(self, value: np.ndarray) -> None:
        self.x_i = value[::2]
        self.z_i = value[1::2]
        self.update_lengths_span_models()

    def objective_function(self) -> np.ndarray:
        """Objective function to minimize: differences of tensions at the load position.
        The balance position is found when the tensions are equal on both side of the load.

        Returns:
            np.ndarray: differences of tensions at the load position. Format is `[Th_diff0, Tv_diff0, Th_diff1, Tv_diff1,...]`
        """
        # left side of the load
        Th_left = self.span_model_left.T_h()
        # need to update?
        x_n_left = self.span_model_left.x_n

        Tv_d_loc = -self.span_model_left.T_v(x_n_left)

        # right
        Th_right = self.span_model_right.T_h()
        x_m_right = self.span_model_right.x_m

        Tv_g_loc = self.span_model_right.T_v(x_m_right)

        Th_diff = Th_left - Th_right
        Tv_diff = Tv_d_loc + Tv_g_loc - self.load_weight * self.k_load

        return np.array([Th_diff, Tv_diff]).flatten('F')

    def update(self) -> None:
        """Update attributes on both span models: `span length`, `elevation difference`, `L_ref` and `parameter`"""
        self.update_lengths_span_models()
        # update parameter left
        L_ref_left = self.load_position * self.L_ref
        a_left = self.x_i
        b_left = self.z_i
        parameter_left = approx_parameter(
            a_left,
            b_left,
            L_ref_left,
            self.k_load,
            self.cable_section,
            self.linear_weight,
            self.young_modulus,
            self.dilatation_coefficient,
            self.temperature,
        )
        self.find_param_model_left.set_attributes(
            initial_parameter=parameter_left,
            L_ref=L_ref_left,
        )

        parameter_left = self.find_param_solver_left.find_parameter()
        self.span_model_left.set_parameter(parameter_left)

        # update parameter right
        L_ref_right = self.L_ref - L_ref_left
        a_right = self.a_prime - self.x_i
        b_right = self.b_prime - self.z_i
        parameter_right = approx_parameter(
            a_right,
            b_right,
            L_ref_right,
            self.k_load,
            self.cable_section,
            self.linear_weight,
            self.young_modulus,
            self.dilatation_coefficient,
            self.temperature,
        )
        self.find_param_model_right.set_attributes(
            initial_parameter=parameter_right,
            L_ref=L_ref_right,
        )

        parameter_right = self.find_param_solver_right.find_parameter()
        self.span_model_right.set_parameter(parameter_right)
