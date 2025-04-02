# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from typing import Type

import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy import optimize  # type: ignore

from mechaphlowers.config import options as cfg
from mechaphlowers.core.models.cable.deformation import (
	DeformationRte,
	IDeformation,
)
from mechaphlowers.core.models.cable.span import (
	CatenarySpan,
	Span,
)
from mechaphlowers.core.models.external_loads import CableLoads


class SagTensionSolver:
	"""This class reprensents the sag tension calculation.
	It computes the horizontal tension of a cable after a change of parameters
	"""

	_ZETA = cfg.solver.sagtension_zeta  # type: ignore

	def __init__(
		self,
		span_length: np.ndarray,
		elevation_difference: np.ndarray,
		sagging_parameter: np.ndarray,
		cable_section_area: np.float64,
		diameter: np.float64,
		linear_weight: np.float64,
		young_modulus: np.float64,
		dilatation_coefficient: np.float64,
		temperature_reference: np.float64,
		polynomial_conductor: Poly,
		unstressed_length: np.ndarray,
		span_model: Type[Span] = CatenarySpan,
		deformation_model: Type[IDeformation] = DeformationRte,
		**kwargs,
	) -> None:
		self.span_length = span_length
		self.elevation_difference = elevation_difference
		self.sagging_parameter = sagging_parameter
		self.cable_section_area = cable_section_area
		self.diameter = diameter
		self.linear_weight = linear_weight
		self.young_modulus = young_modulus
		self.dilatation_coefficient = dilatation_coefficient
		self.temperature_reference = temperature_reference
		self.polynomial_conductor = polynomial_conductor
		self.L_ref = unstressed_length
		self.span_model = span_model
		self.deformation_model = deformation_model
		self.T_h_after_change: np.ndarray | None = None
		self.cable_loads = CableLoads(
			self.diameter,
			self.linear_weight,
			np.zeros(span_length.shape),
			np.zeros(span_length.shape),
		)

	def change_state(
		self,
		ice_thickness: np.ndarray,
		wind_pressure: np.ndarray,
		temp: np.ndarray,
		solver: str = "newton",
		**kwargs,
	) -> None:
		"""Method that solves the finds the new horizontal tension after a change of weather. Parameters are given in SI units.
		The equation to solve is : $\\delta(T_h) = O$
		Args:
			ice_thickness (np.ndarray): new ice_thickness (in m)
			wind_pressure (np.ndarray): new wind_pressure (in Pa)
			temp (np.ndarray): new temperature
			solver (str, optional): resolution method of the equation. Defaults to "newton", which is the only method implemented for now.
		"""

		solver_dict = {"newton": optimize.newton}
		try:
			solver_method = solver_dict[solver]
		except KeyError:
			raise ValueError(f"Incorrect solver name: {solver}")
		self.cable_loads.ice_thickness = ice_thickness
		self.cable_loads.wind_pressure = wind_pressure

		m = self.cable_loads.load_coefficient
		T_h0 = self.span_model.compute_T_h(
			self.sagging_parameter, m, self.linear_weight
		)

		# TODO adapt code to use other solving methods

		solver_result = solver_method(
			self._delta,
			T_h0,
			fprime=self._delta_prime,
			args=(m, temp, self.L_ref),
			tol=1e-5,
			full_output=True,
		)
		if not solver_result.converged.all():
			raise ValueError("Solver did not converge")
		self.T_h_after_change = solver_result.root

	def _delta(self, T_h, m, temp, L_ref):
		"""Function to solve.
		This function is the difference between two ways to compute epsilon.
		Therefore, its value should be zero.
		$\\delta = \\varepsilon_{L} - \\varepsilon_{T}$
		"""
		p = self.span_model.compute_p(T_h, m, self.linear_weight)
		L = self.span_model.compute_L(
			self.span_length, self.elevation_difference, p
		)

		T_mean = self.span_model.compute_T_mean(
			self.span_length, self.elevation_difference, p, T_h
		)
		epsilon_total = self.deformation_model.compute_epsilon_mecha(
			T_mean,
			self.young_modulus,
			self.cable_section_area,
			self.polynomial_conductor,
		) + self.deformation_model.compute_epsilon_therm(
			temp, self.temperature_reference, self.dilatation_coefficient
		)

		return (L / L_ref - 1) - epsilon_total

	def _delta_prime(
		self,
		Th,
		m,
		temp,
		L_ref,
	):
		"""Approximation of the derivative of the function to solve
		$$\\delta'(T_h) = \\frac{\\delta(T_h + \\zeta) - \\delta(T_h)}{\\zeta}$$
		"""
		kwargs = {
			"m": m,
			"temp": temp,
			"L_ref": L_ref,
		}
		return (
			self._delta(Th + self._ZETA, **kwargs) - self._delta(Th, **kwargs)
		) / self._ZETA

	def p_after_change(self):
		"""Compute the new value of the sagging parameter after sag tension calculation"""
		m = self.cable_loads.load_coefficient
		if self.T_h_after_change is None:
			raise ValueError(
				"method change_state has to be run before calling this method"
			)
		return self.span_model.compute_p(
			self.T_h_after_change, m, self.linear_weight
		)

	def L_after_change(self):
		"""Compute the new value of the length of the cable after sag tension calculation"""
		p = self.p_after_change()
		if self.T_h_after_change is None:
			raise ValueError(
				"method change_state has to be run before calling this method"
			)
		return self.span_model.compute_L(
			self.span_length, self.elevation_difference, p
		)
