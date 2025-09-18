# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from functools import lru_cache, wraps

import numpy as np


# not used for the moment
def np_cache(*args, **kwargs):
    """
    LRU cache implementation for functions whose parameter at ``array_argument_index`` is a numpy array of dimensions <= 2

    https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75

    Example:
    >>> from sem_env.utils.cache import np_cache
    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> @np_cache(maxsize=256)
    ... def multiply(array, factor):
    ...     return factor * array
    >>> multiply(array, 2)
    >>> multiply(array, 2)
    >>> multiply.cache_info()
    CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
    """

    def decorator(function):
        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            hashable_array = tuple(np_array)
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(array, *args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper

    return decorator


class VectorProjection:
    def __init__(self):
        pass

    def set_tensions(self, Th: np.ndarray, Tv_d: np.ndarray, Tv_g: np.ndarray):
        self.Th = Th
        self.Tv_d = Tv_d
        self.Tv_g = Tv_g

    def set_angles(
        self, alpha: np.ndarray, beta: np.ndarray, line_angle: np.ndarray
    ):
        self.alpha = alpha
        self.beta = beta
        self.line_angle = line_angle

    def set_proj_angle(self, proj_angle: np.ndarray):
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
        weight_chain: np.ndarray,
    ):
        self.set_tensions(Th, Tv_d, Tv_g)
        self.set_angles(alpha, beta, line_angle)
        self.set_proj_angle(proj_angle)
        self.weight_chain = weight_chain

    # properties?
    def T_attachments_plane_left(self):
        beta = self.beta
        Th = self.Th
        Tv_g = self.Tv_g
        alpha = self.alpha
        vg = Tv_g * np.cos(beta) - Th * np.sin(beta) * np.sin(alpha)
        hg = Tv_g * np.sin(beta) + Th * np.cos(beta) * np.sin(alpha)
        lg = Th * np.cos(alpha)
        # order x, y, z ?
        return np.array([lg, hg, vg])

    def T_attachments_plane_right(self):
        beta = self.beta
        Th = self.Th
        Tv_d = self.Tv_d
        alpha = self.alpha
        vd = Tv_d * np.cos(beta) + Th * np.sin(beta) * np.sin(alpha)
        hd = Tv_d * np.sin(beta) - Th * np.cos(beta) * np.sin(alpha)
        ld = -Th * np.cos(alpha)
        # order x, y, z ?
        return np.array([ld, hd, vd])

    def T_line_plane_left(self):
        lg, hg, vg = self.T_attachments_plane_left()
        proj_angle = self.proj_angle
        r_s_g = lg * np.cos(proj_angle) - hg * np.sin(proj_angle)
        r_t_g = lg * np.sin(proj_angle) + hg * np.cos(proj_angle)
        r_z_g = vg
        # order between s and t?
        return np.array([r_s_g, r_t_g, r_z_g])

    def T_line_plane_right(self):
        ld, hd, vd = self.T_attachments_plane_right()
        proj_angle = self.proj_angle
        r_s_d = ld * np.cos(proj_angle) - hd * np.sin(proj_angle)
        r_t_d = ld * np.sin(proj_angle) + hd * np.cos(proj_angle)
        r_z_d = vd
        # order between s and t?
        return np.array([r_s_d, r_t_d, r_z_d])

    def forces(self):
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
        Fz_first = z_left[0] + self.weight_chain[0] / 2  # also add load?

        Fx_suspension = (s_right + s_left_rolled) * np.cos(gamma) - (
            -t_right + t_left_rolled
        ) * np.sin(gamma)
        Fy_suspension = (t_right + t_left_rolled) * np.cos(gamma) - (
            s_right - s_left_rolled
        ) * np.sin(gamma)
        Fz_suspension = z_right + z_left_rolled + self.weight_chain[1:] / 2

        Fx_last = (s_right[-1]) * np.cos(gamma[-1]) - (-t_right[-1]) * np.sin(
            gamma[-1]
        )
        Fy_last = (t_right[-1]) * np.cos(gamma[-1]) - (s_right[-1]) * np.sin(
            gamma[-1]
        )
        Fz_last = z_right[-1] + self.weight_chain[-1] / 2

        Fx = np.concat(([Fx_first], Fx_suspension[:-1], [Fx_last]))
        Fy = np.concat(([Fy_first], Fy_suspension[:-1], [Fy_last]))
        Fz = np.concat(([Fz_first], Fz_suspension[:-1], [Fz_last]))
        return Fx, Fy, Fz
