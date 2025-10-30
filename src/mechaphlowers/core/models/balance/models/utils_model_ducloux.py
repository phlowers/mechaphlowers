# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from typing import List, Tuple

import numpy as np


class Masks:
    """
    Current types: "suspension", "anchor_first", "anchor_last"
    """

    # TODO: wriste docstring
    def __init__(
        self, nodes_type: List[str], insulator_length: np.ndarray
    ) -> None:
        self.nodes_type = nodes_type
        self.insulator_length = insulator_length
        self.is_suspension = [x == "suspension" for x in self.nodes_type]
        self.is_anchor_first = [x == "anchor_first" for x in self.nodes_type]
        self.is_anchor_last = [x == "anchor_last" for x in self.nodes_type]

    def compute_dx_dy_dz(
        self, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        L_chain = self.insulator_length
        is_suspension = self.is_suspension
        is_anchor_first = self.is_anchor_first
        is_anchor_last = self.is_anchor_last
        new_dx = dx.copy()
        new_dz = dz.copy()
        # case: suspension chains
        suspension_shift = -(
            (
                L_chain[is_suspension] ** 2
                - dx[is_suspension] ** 2
                - dy[is_suspension] ** 2
            )
            ** 0.5
        )
        new_dz[is_suspension] = suspension_shift

        # case: first anchor chain
        anchor_shift_first = (
            L_chain[is_anchor_first] ** 2
            - dz[is_anchor_first] ** 2
            - dy[is_anchor_first] ** 2
        ) ** 0.5
        new_dx[is_anchor_first] = anchor_shift_first

        # case: first anchor last
        anchor_shift_last = (
            L_chain[is_anchor_last] ** 2
            - dz[is_anchor_last] ** 2
            - dy[is_anchor_last] ** 2
        ) ** 0.5
        new_dx[is_anchor_last] = -anchor_shift_last

        return new_dx, new_dz


class VectorProjection:
    # TODO: understand this and write docstring + eventually refactor this
    def set_tensions(
        self, Th: np.ndarray, Tv_d: np.ndarray, Tv_g: np.ndarray
    ) -> None:
        self.Th = Th
        self.Tv_d = Tv_d
        self.Tv_g = Tv_g

    def set_angles(
        self, alpha: np.ndarray, beta: np.ndarray, line_angle: np.ndarray
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.line_angle = line_angle

    def set_proj_angle(self, proj_angle: np.ndarray) -> None:
        self.proj_angle = proj_angle

    def set_all(
        self,
        Th: np.ndarray,
        Tv_d: np.ndarray,
        Tv_g: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        line_angle: np.ndarray,
        proj_angle: np.ndarray,
        insulator_weight: np.ndarray,
    ) -> None:
        self.set_tensions(Th, Tv_d, Tv_g)
        self.set_angles(alpha, beta, line_angle)
        self.set_proj_angle(proj_angle)
        self.insulator_weight = insulator_weight

    # properties?
    def T_attachments_plane_left(self) -> np.ndarray:
        beta = self.beta
        Th = self.Th
        Tv_g = self.Tv_g
        alpha = self.alpha
        vg = Tv_g * np.cos(beta) - Th * np.sin(beta) * np.sin(alpha)
        hg = Tv_g * np.sin(beta) + Th * np.cos(beta) * np.sin(alpha)
        lg = Th * np.cos(alpha)
        return np.array([lg, hg, vg])

    def T_attachments_plane_right(self) -> np.ndarray:
        beta = self.beta
        Th = self.Th
        Tv_d = self.Tv_d
        alpha = self.alpha
        vd = Tv_d * np.cos(beta) + Th * np.sin(beta) * np.sin(alpha)
        hd = Tv_d * np.sin(beta) - Th * np.cos(beta) * np.sin(alpha)
        ld = -Th * np.cos(alpha)
        return np.array([ld, hd, vd])

    def T_line_plane_left(self) -> np.ndarray:
        lg, hg, vg = self.T_attachments_plane_left()
        proj_angle = self.proj_angle
        r_s_g = lg * np.cos(proj_angle) - hg * np.sin(proj_angle)
        r_t_g = lg * np.sin(proj_angle) + hg * np.cos(proj_angle)
        r_z_g = vg
        return np.array([r_s_g, r_t_g, r_z_g])

    def T_line_plane_right(self) -> np.ndarray:
        ld, hd, vd = self.T_attachments_plane_right()
        proj_angle = self.proj_angle
        r_s_d = ld * np.cos(proj_angle) - hd * np.sin(proj_angle)
        r_t_d = ld * np.sin(proj_angle) + hd * np.cos(proj_angle)
        r_z_d = vd
        return np.array([r_s_d, r_t_d, r_z_d])

    def forces(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s_right, t_right, z_right = self.T_line_plane_right()
        T_line_plane_left = self.T_line_plane_left()
        s_left, t_left, z_left = T_line_plane_left
        s_left_rolled, t_left_rolled, z_left_rolled = np.roll(
            T_line_plane_left, -1, axis=1
        )

        gamma = (self.line_angle / 2)[1:]

        # Not entierly sure about indices and left/right

        # index 1 ou 0?
        Fx_first = s_left[0] * np.cos((self.line_angle / 2)[0]) - t_left[
            0
        ] * np.sin((self.line_angle / 2)[0])
        Fy_first = t_left[0] * np.cos((self.line_angle / 2)[0]) + s_left[
            0
        ] * np.sin((self.line_angle / 2)[0])
        Fz_first = z_left[0] + self.insulator_weight[0] / 2  # also add load?

        Fx_suspension = (s_right + s_left_rolled) * np.cos(gamma) - (
            -t_right + t_left_rolled
        ) * np.sin(gamma)
        Fy_suspension = (t_right + t_left_rolled) * np.cos(gamma) - (
            s_right - s_left_rolled
        ) * np.sin(gamma)
        Fz_suspension = z_right + z_left_rolled + self.insulator_weight[1:] / 2

        Fx_last = (s_right[-1]) * np.cos(gamma[-1]) - (-t_right[-1]) * np.sin(
            gamma[-1]
        )
        Fy_last = (t_right[-1]) * np.cos(gamma[-1]) - (s_right[-1]) * np.sin(
            gamma[-1]
        )
        Fz_last = z_right[-1] + self.insulator_weight[-1] / 2

        Fx = np.concat(([Fx_first], Fx_suspension[:-1], [Fx_last]))
        Fy = np.concat(([Fy_first], Fy_suspension[:-1], [Fy_last]))
        Fz = np.concat(([Fz_first], Fz_suspension[:-1], [Fz_last]))
        return Fx, Fy, Fz
