# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.core.models.cable.deformation_laws import (
    DeformationState,
    IPreparedLaw,
    IStressStrainLaw,
    LinearLaw,
    deformation_law_builder,
)
from mechaphlowers.core.models.cable.span import ISpan
from mechaphlowers.entities.arrays import CableArray


class IDeformation(ABC):
    """This abstract class is a base class for models to compute relative cable deformations.

    The constitutive behaviour is delegated to a stress-strain law
    ([`IStressStrainLaw`][mechaphlowers.core.models.cable.deformation_laws.IStressStrainLaw]),
    and the loading history to an immutable
    [`DeformationState`][mechaphlowers.core.models.cable.deformation_laws.DeformationState].
    The law is prepared lazily for the current temperature and state, and re-prepared only when
    `current_temperature` or `state` are assigned new values.

    `tension_mean`, `cable_length` and `current_temperature` are updated by callers (solvers).
    `current_temperature` is copied when the law is prepared, so in-place modifications
    are ignored until the next assignment: assign the array instead.
    """

    def __init__(
        self,
        tension_mean: np.ndarray,
        cable_length: np.ndarray,
        cable_section_area: np.float64,
        linear_weight: np.float64,
        young_modulus: np.float64,
        dilatation_coefficient: np.float64,
        temperature_reference: np.float64,
        polynomial_conductor: Poly,
        sagging_temperature: np.ndarray,
        max_stress: np.ndarray | None = None,
        law: IStressStrainLaw | None = None,
        state: DeformationState | None = None,
        **_,
    ):
        self.tension_mean = tension_mean
        self.cable_length = cable_length
        self.cable_section_area = cable_section_area
        self.linear_weight = linear_weight
        self.young_modulus = young_modulus
        self.dilatation_coefficient = dilatation_coefficient
        self.temp_ref = temperature_reference
        self.polynomial_conductor = polynomial_conductor
        self.is_polynomial = polynomial_conductor.trim().degree() >= 2
        # kept for compatibility, history is now stored in `state`
        self.max_stress = max_stress

        if law is None and not self.is_polynomial:
            law = LinearLaw(
                cable_section_area,
                young_modulus,
                dilatation_coefficient,
                temperature_reference,
            )
        self._law = law
        self._prepared: IPreparedLaw | None = None
        self._prepared_temperature: np.ndarray | None = None
        self._current_temperature = sagging_temperature
        self._state = (
            state
            if state is not None
            else (
                law.initial_state(np.shape(cable_length)[0])
                if law is not None
                else DeformationState()
            )
        )

    def __copy__(self) -> IDeformation:
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.tension_mean = self.tension_mean.copy()
        new.cable_length = self.cable_length.copy()
        new._current_temperature = self._current_temperature.copy()
        new._prepared = None
        return new

    @property
    def law(self) -> IStressStrainLaw:
        """Stress-strain law of the cable."""
        if self._law is None:
            raise NotImplementedError(
                "No stress-strain law given for a polynomial cable: "
                "use deformation_model_builder() to build the deformation model."
            )
        return self._law

    @property
    def current_temperature(self) -> np.ndarray:
        return self._current_temperature

    @current_temperature.setter
    def current_temperature(self, value: np.ndarray) -> None:
        # solvers re-assign equal temperatures at each iteration: keep the prepared law
        if self._prepared is not None and not np.array_equal(
            value,
            self._prepared_temperature,  # type: ignore[arg-type]
            equal_nan=True,
        ):
            self._prepared = None
        self._current_temperature = value

    @property
    def state(self) -> DeformationState:
        """Loading history (immutable). Assign a new state to change it."""
        return self._state

    @state.setter
    def state(self, value: DeformationState) -> None:
        if value is not self._state:
            self._prepared = None
        self._state = value

    @property
    def prepared(self) -> IPreparedLaw:
        """Law prepared for the current temperature and state (cached)."""
        if self._prepared is None:
            self._prepared_temperature = np.array(
                self._current_temperature, dtype=np.float64
            )
            self._prepared = self.law.prepare(
                self._prepared_temperature, self._state
            )
        return self._prepared

    def updated_state(self) -> DeformationState:
        """State after loading at the current `tension_mean` and `current_temperature`
        (ratchet). The model is not modified."""
        return self.law.update_state(
            self.tension_mean, self._current_temperature, self._state
        )

    @abstractmethod
    def L_ref(self) -> np.ndarray:
        """Unstressed cable length, at a chosen reference temperature, compared to the temperature reference"""

    @abstractmethod
    def L_0(self) -> np.ndarray:
        """Unstressed cable length, at a chosen reference temperature, whrer temperature_reference = 0°C"""

    @abstractmethod
    def epsilon(self) -> np.ndarray:
        """Total relative strain of the cable."""

    @abstractmethod
    def epsilon_0(self) -> np.ndarray:
        """Total relative strain of the cable, relative to the unstressed length at 0°C."""

    @abstractmethod
    def epsilon_mecha(self) -> np.ndarray:
        """Mechanical part of the relative strain  of the cable."""

    @abstractmethod
    def epsilon_therm(self) -> np.ndarray:
        """Thermal part of the relative deformation of the cable, compared to a temperature_reference."""

    @abstractmethod
    def epsilon_therm_0(self) -> np.ndarray:
        """Thermal part of the relative deformation of the cable, where temperature_reference = 0."""


class DeformationRte(IDeformation):
    """This class implements the deformation model used by RTE.

    Linear cables use [`LinearLaw`][mechaphlowers.core.models.cable.deformation_laws.LinearLaw],
    polynomial cables use [`PolynomialLaw`][mechaphlowers.core.models.cable.deformation_laws.PolynomialLaw]
    (selected by [`deformation_model_builder`][mechaphlowers.core.models.cable.deformation.deformation_model_builder]).

    For multi-material cables, mechanical and thermal strains are not separable:
    `epsilon_therm` uses the equivalent dilatation coefficient and `epsilon_mecha` is
    `epsilon - epsilon_therm`.
    """

    def L_ref(self) -> np.ndarray:
        return self.cable_length / (1 + self.epsilon())

    def L_0(self) -> np.ndarray:
        return self.cable_length / (1 + self.epsilon_0())

    def epsilon(self) -> np.ndarray:
        return self.prepared.epsilon(self.tension_mean)

    def epsilon_0(self) -> np.ndarray:
        return self.prepared.epsilon_0(self.tension_mean)

    def epsilon_mecha(self) -> np.ndarray:
        return self.prepared.epsilon_mecha(self.tension_mean)

    def epsilon_therm(self) -> np.ndarray:
        return self.prepared.epsilon_therm()

    def epsilon_therm_0(self) -> np.ndarray:
        return self.prepared.epsilon_therm_0()


def deformation_model_builder(
    cable_array: CableArray,
    span_model: ISpan,
    sagging_temperature: np.ndarray,
    deformation_model_type: Type[IDeformation] = DeformationRte,
    state: DeformationState | None = None,
    law: IStressStrainLaw | None = None,
) -> IDeformation:
    """Build a deformation model. The stress-strain law is selected from cable data by
    [`deformation_law_builder`][mechaphlowers.core.models.cable.deformation_laws.deformation_law_builder]
    unless `law` is given. `state` defaults to a virgin cable."""
    tension_mean = span_model.T_mean()
    cable_length = span_model.compute_L()
    cable_section = np.float64(cable_array.data.section.iloc[0])
    linear_weight = np.float64(cable_array.data.linear_weight.iloc[0])
    young_modulus = np.float64(cable_array.data.young_modulus.iloc[0])
    dilatation_coefficient = np.float64(
        cable_array.data.dilatation_coefficient.iloc[0]
    )
    temperature_reference = np.float64(
        cable_array.data.temperature_reference.iloc[0]
    )
    polynomial_conductor = cable_array.polynomial_conductor
    if law is None:
        law = deformation_law_builder(cable_array)
    return deformation_model_type(
        tension_mean,
        cable_length,
        cable_section,
        linear_weight,
        young_modulus,
        dilatation_coefficient,
        temperature_reference,
        polynomial_conductor,
        sagging_temperature,
        law=law,
        state=state,
    )
