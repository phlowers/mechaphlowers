# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Stress-strain laws used by deformation models.

Three concepts are kept separate:

- a **law** ([`IStressStrainLaw`][mechaphlowers.core.models.cable.deformation_laws.IStressStrainLaw])
  describes the constitutive behaviour of the cable (what the cable *is*), built once from a
  [`CableArray`][mechaphlowers.entities.arrays.CableArray];
- a **state** ([`DeformationState`][mechaphlowers.core.models.cable.deformation_laws.DeformationState])
  is an immutable value storing the loading history of each span (where the cable *has been*);
- a **prepared law** ([`IPreparedLaw`][mechaphlowers.core.models.cable.deformation_laws.IPreparedLaw])
  freezes temperature and state so that strains can be evaluated at minimal cost inside solver loops.

All strains returned by prepared laws are **total** strains (mechanical + thermal), because the
mechanical and thermal parts are not separable for multi-material cables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import comb
from typing import TYPE_CHECKING, Sequence

import numpy as np

from mechaphlowers.entities.errors import ConvergenceError

if TYPE_CHECKING:
    from mechaphlowers.entities.arrays import CableArray

POLY_DEGREE = 4
_N_COEFFS = POLY_DEGREE + 1
_MIN_SLOPE = 1.0  # Pa, guards Newton steps where a segment is flat
_STRAIN_TOL = (
    1e-15  # absolute strain tolerance (relative ~1e-12 for usual strains)
)
# Newton converges quadratically: after a step smaller than this, the error is below _STRAIN_TOL
_NEWTON_STEP_TOL = 1e-12


@dataclass(frozen=True, eq=False)
class DeformationState:
    """Loading history of a cable, one value per span. Immutable.

    Attributes:
        max_strain: Maximum mechanical strain reached by each material, measured at the reference
            temperature. Shape `(n_materials, n_spans)`. `None` for history-independent laws.
    """

    max_strain: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.max_strain is not None:
            frozen = np.array(self.max_strain, dtype=np.float64, ndmin=2)
            frozen.flags.writeable = False
            object.__setattr__(self, "max_strain", frozen)

    def select(self, index: np.ndarray) -> DeformationState:
        """State restricted to some spans (boolean mask or integer indices)."""
        if self.max_strain is None:
            return self
        return DeformationState(self.max_strain[:, index])

    @staticmethod
    def concatenate(states: Sequence[DeformationState]) -> DeformationState:
        """Concatenate span-wise several states of the same law."""
        if states[0].max_strain is None:
            return states[0]
        return DeformationState(
            np.concatenate([s.max_strain for s in states], axis=1)  # type: ignore[misc]
        )

    def maximum(self, other: DeformationState) -> DeformationState:
        """Element-wise maximum of two states (NaN are ignored)."""
        if self.max_strain is None or other.max_strain is None:
            return self
        return DeformationState(np.fmax(self.max_strain, other.max_strain))


class IPreparedLaw(ABC):
    """Stress-strain law with temperature and state frozen. Used in solver hot loops.

    Inputs are mean tensions in N, one per span.
    """

    @abstractmethod
    def epsilon(self, tension_mean: np.ndarray) -> np.ndarray:
        """Total strain relative to the unstressed length at the reference temperature."""

    @abstractmethod
    def epsilon_0(self, tension_mean: np.ndarray) -> np.ndarray:
        """Total strain relative to the unstressed length at 0°C."""

    @abstractmethod
    def epsilon_therm(self) -> np.ndarray:
        """Thermal strain relative to the reference temperature (equivalent value for multi-material laws)."""

    @abstractmethod
    def epsilon_therm_0(self) -> np.ndarray:
        """Thermal strain relative to 0°C (equivalent value for multi-material laws)."""

    def epsilon_mecha(self, tension_mean: np.ndarray) -> np.ndarray:
        """Mechanical strain: `epsilon - epsilon_therm`."""
        return self.epsilon(tension_mean) - self.epsilon_therm()


class IStressStrainLaw(ABC):
    """Constitutive law of a cable: relation between mean tension, temperature, history and strain.

    Implementations must be immutable: all evaluation-dependent data goes into
    [`DeformationState`][mechaphlowers.core.models.cable.deformation_laws.DeformationState]
    or into the [`IPreparedLaw`][mechaphlowers.core.models.cable.deformation_laws.IPreparedLaw].
    """

    section: float
    temperature_reference: float

    @classmethod
    @abstractmethod
    def from_cable_array(cls, cable_array: CableArray) -> IStressStrainLaw:
        """Build the law from cable data (SI units)."""

    @property
    @abstractmethod
    def young_modulus_equivalent(self) -> np.float64:
        """Equivalent young modulus, used for approximations (e.g. solver initial values)."""

    @property
    @abstractmethod
    def dilatation_coefficient_equivalent(self) -> np.float64:
        """Equivalent dilatation coefficient, used for approximations and length conversions."""

    @property
    def is_history_dependent(self) -> bool:
        """Whether the strain depends on the loading history stored in the state."""
        return False

    def initial_state(self, n_spans: int) -> DeformationState:
        """State of a virgin cable."""
        return DeformationState()

    @abstractmethod
    def prepare(
        self, temperature: np.ndarray, state: DeformationState
    ) -> IPreparedLaw:
        """Freeze temperature (°C) and state to evaluate strains efficiently."""

    def update_state(
        self,
        tension_mean: np.ndarray,
        temperature: np.ndarray,
        state: DeformationState,
    ) -> DeformationState:
        """New state after the cable has been loaded at `tension_mean` and `temperature`.
        Pure function: `state` is not modified."""
        return state


class LinearLaw(IStressStrainLaw):
    """Linear elastic law (historical RTE model): $\\varepsilon = \\frac{T_{mean}}{E S} + \\alpha (\\theta - \\theta_{ref})$."""

    def __init__(
        self,
        section: float,
        young_modulus: float,
        dilatation_coefficient: float,
        temperature_reference: float,
    ) -> None:
        self.section = section
        self.young_modulus = young_modulus
        self.dilatation_coefficient = dilatation_coefficient
        self.temperature_reference = temperature_reference

    @classmethod
    def from_cable_array(cls, cable_array: CableArray) -> LinearLaw:
        data = cable_array.data
        return cls(
            np.float64(data.section.iloc[0]),
            np.float64(data.young_modulus.iloc[0]),
            np.float64(data.dilatation_coefficient.iloc[0]),
            np.float64(data.temperature_reference.iloc[0]),
        )

    @property
    def young_modulus_equivalent(self) -> np.float64:
        return np.float64(self.young_modulus)

    @property
    def dilatation_coefficient_equivalent(self) -> np.float64:
        return np.float64(self.dilatation_coefficient)

    def prepare(
        self, temperature: np.ndarray, state: DeformationState
    ) -> PreparedLinearLaw:
        return PreparedLinearLaw(self, temperature)


class PreparedLinearLaw(IPreparedLaw):
    def __init__(self, law: LinearLaw, temperature: np.ndarray) -> None:
        # Operation order kept identical to the historical DeformationRte implementation
        self._ES = law.young_modulus * law.section
        self._eps_therm = (
            temperature - law.temperature_reference
        ) * law.dilatation_coefficient
        self._eps_therm_0 = temperature * law.dilatation_coefficient

    def epsilon(self, tension_mean: np.ndarray) -> np.ndarray:
        return tension_mean / self._ES + self._eps_therm

    def epsilon_0(self, tension_mean: np.ndarray) -> np.ndarray:
        return tension_mean / self._ES + self._eps_therm_0

    def epsilon_mecha(self, tension_mean: np.ndarray) -> np.ndarray:
        return tension_mean / self._ES

    def epsilon_therm(self) -> np.ndarray:
        return self._eps_therm

    def epsilon_therm_0(self) -> np.ndarray:
        return self._eps_therm_0


@dataclass(frozen=True, eq=False)
class PolynomialMaterial:
    """One material of a polynomial cable (e.g. conductor or heart).

    Attributes:
        coefficients: Coefficients $(c_0..c_4)$ in Pa of the virgin (first loading) curve
            $\\tilde\\sigma(x) = \\sum c_k x^k$, in **reduced** stress (force divided by the total cable section).
        young_modulus: Unloading modulus in Pa (reduced).
        dilatation_coefficient: Dilatation coefficient in 1/K.
    """

    coefficients: np.ndarray
    young_modulus: float
    dilatation_coefficient: float

    def __post_init__(self) -> None:
        coefficients = np.zeros(_N_COEFFS)
        given = np.asarray(self.coefficients, dtype=np.float64)
        if given.size > _N_COEFFS:
            raise ValueError(f"Polynomial degree must be <= {POLY_DEGREE}")
        coefficients[: given.size] = given
        coefficients.flags.writeable = False
        object.__setattr__(self, "coefficients", coefficients)
        if not self.young_modulus > 0:
            raise ValueError("young_modulus of a material must be positive")
        if not coefficients[1] > 0:
            raise ValueError(
                "Stress-strain polynomial must be increasing at zero strain (c1 > 0)"
            )
        x_lim = _first_positive_root(
            np.polynomial.Polynomial(coefficients).deriv()
        )
        object.__setattr__(self, "max_virgin_strain", x_lim)
        object.__setattr__(
            self,
            "max_virgin_stress",
            float(_horner(coefficients, np.array(x_lim)))
            if np.isfinite(x_lim)
            else np.inf,
        )

    max_virgin_strain: float = field(init=False, repr=False)
    """Strain where the virgin polynomial stops increasing (`inf` if it never does).
    Beyond, the material stress is kept constant (plateau)."""
    max_virgin_stress: float = field(init=False, repr=False)
    """Reduced stress of the plateau (`inf` if no plateau)."""

    def virgin_stress(self, x: np.ndarray) -> np.ndarray:
        """Reduced stress on the virgin curve, plateau included."""
        return _horner(
            self.coefficients, np.minimum(x, self.max_virgin_strain)
        )


class PolynomialLaw(IStressStrainLaw):
    """Multi-material polynomial law with permanent strain (one conductor + optional heart).

    For each material $i$, with mechanical strain $x_i = \\varepsilon - \\alpha_i(\\theta - \\theta_{ref})$
    and history $x_{max,i}$:

    - $x_i \\geq x_{max,i}$: virgin curve $P_i(x_i)$ (constant beyond its maximum);
    - $x_{p,i} \\leq x_i < x_{max,i}$: elastic line $E_i (x_i - x_{p,i})$ with
      $x_{p,i} = x_{max,i} - P_i(x_{max,i}) / E_i$ (permanent strain);
    - $x_i < x_{p,i}$: slack material, no stress.

    The cable stress is the sum of reduced stresses: $\\sigma = T_{mean}/S = \\sum_i \\tilde\\sigma_i(x_i)$.
    Strains returned are relative to the unstressed length at the reference temperature, and
    $L_{\\theta_{ref}} = L_0 (1 + \\alpha_{eq} \\theta_{ref})$ with
    $\\alpha_{eq} = \\sum \\alpha_i E_i / \\sum E_i$.
    """

    def __init__(
        self,
        materials: Sequence[PolynomialMaterial],
        section: float,
        temperature_reference: float,
    ) -> None:
        if not 1 <= len(materials) <= 2:
            raise ValueError("PolynomialLaw supports one or two materials")
        self.materials = tuple(materials)
        self.section = section
        self.temperature_reference = temperature_reference

    @classmethod
    def from_cable_array(cls, cable_array: CableArray) -> PolynomialLaw:
        """Conductor uses `a0..a4`, heart uses `b0..b4` (dropped if all zero).

        Unloading modulus: polynomial linear coefficient (`a1`/`b1`), or
        `young_modulus_conductor`/`young_modulus_heart` if this coefficient is zero.
        Dilatation coefficient: `dilatation_coefficient_conductor`/`_heart`, or
        `dilatation_coefficient` if missing.
        """
        data = cable_array.data
        row = data.iloc[0]

        def value(column: str, default: float = np.nan) -> float:
            if column not in data.columns or not np.isfinite(row[column]):
                return default
            return float(row[column])

        materials = []
        for name, prefix in (("conductor", "a"), ("heart", "b")):
            coefficients = np.array(
                [value(f"{prefix}{k}", 0.0) for k in range(_N_COEFFS)]
            )
            if not coefficients.any():
                continue
            young_modulus = (
                coefficients[1]
                if coefficients[1] != 0
                else value(f"young_modulus_{name}")
            )
            dilatation = value(
                f"dilatation_coefficient_{name}",
                float(row["dilatation_coefficient"]),
            )
            materials.append(
                PolynomialMaterial(coefficients, young_modulus, dilatation)
            )
        return cls(
            materials,
            float(row["section"]),
            float(row["temperature_reference"]),
        )

    @property
    def young_modulus_equivalent(self) -> np.float64:
        return np.float64(sum(m.young_modulus for m in self.materials))

    @property
    def dilatation_coefficient_equivalent(self) -> np.float64:
        return np.float64(
            sum(
                m.dilatation_coefficient * m.young_modulus
                for m in self.materials
            )
            / self.young_modulus_equivalent
        )

    @property
    def is_history_dependent(self) -> bool:
        return True

    def initial_state(self, n_spans: int) -> DeformationState:
        return DeformationState(np.zeros((len(self.materials), n_spans)))

    def prepare(
        self, temperature: np.ndarray, state: DeformationState
    ) -> PreparedPolynomialLaw:
        temperature = np.asarray(temperature, dtype=np.float64)
        if state.max_strain is None:
            state = self.initial_state(temperature.shape[0])
        return PreparedPolynomialLaw(self, temperature, state)

    def thermal_strains(self, temperature: np.ndarray) -> np.ndarray:
        """Thermal strain of each material relative to the reference temperature, shape `(n_materials, n)`."""
        alphas = np.array([m.dilatation_coefficient for m in self.materials])
        return alphas[:, None] * (temperature - self.temperature_reference)

    def update_state(
        self,
        tension_mean: np.ndarray,
        temperature: np.ndarray,
        state: DeformationState,
    ) -> DeformationState:
        prepared = self.prepare(temperature, state)
        mechanical_strain = prepared.epsilon(
            tension_mean
        ) - self.thermal_strains(prepared.temperature)
        return prepared.state.maximum(DeformationState(mechanical_strain))


class PreparedPolynomialLaw(IPreparedLaw):
    """Temperature and state frozen for [`PolynomialLaw`][mechaphlowers.core.models.cable.deformation_laws.PolynomialLaw].

    $\\sigma(\\varepsilon)$ is continuous, increasing and piecewise polynomial; its breakpoints are known
    analytically. Evaluation strategy:

    1. elastic closed form, exact when every material stays below its history (usual case once a
       loading history has been set);
    2. for the other spans, Newton iterations restricted to the polynomial segment containing the
       solution, safeguarded by bisection. The last solution of each span is used as initial
       guess (warm start), as solvers call this law repeatedly with close tensions.
    """

    max_iter = 20

    def __init__(
        self,
        law: PolynomialLaw,
        temperature: np.ndarray,
        state: DeformationState,
    ) -> None:
        self.law = law
        self.temperature = temperature
        self.state = state
        self._inv_section = 1.0 / law.section
        self._strain_0_factor = (
            1.0
            + law.dilatation_coefficient_equivalent * law.temperature_reference
        )
        materials = law.materials
        n = temperature.shape[0]
        x_max = state.max_strain
        if x_max is None or x_max.shape != (len(materials), n):
            raise ValueError(
                f"Deformation state shape {None if x_max is None else x_max.shape} "
                f"does not match ({len(materials)}, {n})"
            )
        eps_th = law.thermal_strains(temperature)
        self._eps_therm = eps_th

        e_perm, e_max, e_plateau = [], [], []
        for i, material in enumerate(materials):
            x_p = (
                x_max[i]
                - material.virgin_stress(x_max[i]) / material.young_modulus
            )
            e_perm.append(eps_th[i] + x_p)
            e_max.append(eps_th[i] + x_max[i])
            e_plateau.append(
                eps_th[i] + np.maximum(material.max_virgin_strain, x_max[i])
            )
        self._valid = np.isfinite(temperature) & np.all(
            np.isfinite(x_max), axis=0
        )

        self._prepare_elastic(e_perm, e_max)
        self._prepare_segments(eps_th, e_perm, e_max, e_plateau)
        self._last_solution = np.full(n, np.nan)
        self._segment_cache: tuple | None = None

    def _prepare_elastic(self, e_perm: list, e_max: list) -> None:
        moduli = [np.float64(m.young_modulus) for m in self.law.materials]
        self._E_lo: np.ndarray | np.float64
        self._elastic_limit = np.minimum.reduce(e_max)
        if len(moduli) == 1:
            self._e_lo, self._E_lo = e_perm[0], moduli[0]
            self._e_hi = np.full_like(e_perm[0], np.inf)
            self._num = e_perm[0] * moduli[0]
            self._E_sum = moduli[0]
            return
        first = e_perm[0] <= e_perm[1]
        self._e_lo = np.where(first, e_perm[0], e_perm[1])
        self._e_hi = np.where(first, e_perm[1], e_perm[0])
        self._E_lo = np.where(first, moduli[0], moduli[1])
        self._num = moduli[0] * e_perm[0] + moduli[1] * e_perm[1]
        self._E_sum = moduli[0] + moduli[1]

    def _prepare_segments(
        self, eps_th: np.ndarray, e_perm: list, e_max: list, e_plateau: list
    ) -> None:
        n = eps_th.shape[1]
        columns = np.arange(n)
        breakpoints = np.sort(np.stack([*e_perm, *e_max, *e_plateau]), axis=0)
        n_bp = breakpoints.shape[0]
        lower = np.concatenate([np.full((1, n), -np.inf), breakpoints])
        coefficients = np.zeros((n_bp + 1, n, _N_COEFFS))
        for i, material in enumerate(self.law.materials):
            # pieces: 0 slack, 1 elastic, 2 virgin, 3 plateau
            pieces = np.zeros((4, n, _N_COEFFS))
            pieces[1, :, 0] = -material.young_modulus * e_perm[i]
            pieces[1, :, 1] = material.young_modulus
            pieces[2] = _shift_coefficients(material.coefficients, eps_th[i])
            pieces[3, :, 0] = material.max_virgin_stress
            piece = (
                (lower >= e_perm[i]).astype(np.intp)
                + (lower >= e_max[i])
                + (lower >= e_plateau[i])
            )
            coefficients += pieces[piece, columns]
        finite = np.isfinite(breakpoints)
        # stress at the start of each segment j >= 1 (continuity)
        stress_bp = _horner_rows(
            coefficients[1:], np.where(finite, breakpoints, 0.0)
        )
        stress_bp = np.where(finite, stress_bp, np.inf)
        self._breakpoints = breakpoints
        self._stress_bp = np.maximum.accumulate(stress_bp, axis=0)
        self._coefficients = coefficients
        self._d_coefficients = coefficients[..., 1:] * np.arange(1, _N_COEFFS)

    def epsilon(self, tension_mean: np.ndarray) -> np.ndarray:
        sigma = np.asarray(tension_mean, dtype=np.float64) * self._inv_section
        eps_one = self._e_lo + sigma / self._E_lo
        eps = np.where(
            eps_one <= self._e_hi, eps_one, (sigma + self._num) / self._E_sum
        )
        to_solve = (
            ~(eps <= self._elastic_limit) & np.isfinite(sigma) & self._valid
        )
        if to_solve.any():
            index = np.flatnonzero(to_solve)
            eps[index] = self._solve_segments(sigma[index], index)
        return eps

    def epsilon_0(self, tension_mean: np.ndarray) -> np.ndarray:
        return self._strain_0_factor * (1.0 + self.epsilon(tension_mean)) - 1.0

    def epsilon_therm(self) -> np.ndarray:
        return self.law.dilatation_coefficient_equivalent * (
            self.temperature - self.law.temperature_reference
        )

    def epsilon_therm_0(self) -> np.ndarray:
        return self.law.dilatation_coefficient_equivalent * self.temperature

    def _solve_segments(
        self, sigma: np.ndarray, index: np.ndarray
    ) -> np.ndarray:
        cache = self._segment_cache
        if (
            cache is not None
            and cache[0].size == index.size
            and np.array_equal(cache[0], index)
            and np.all((sigma >= cache[1]) & (sigma < cache[2]))
        ):
            # same segments as the previous call: skip the lookup
            _, _, _, lo, hi, c, d = cache
        else:
            lo, hi, c, d = self._find_segments(sigma, index)
        c0, c1, c2, c3, c4 = c
        d0, d1, d2, d3 = d
        last = self._last_solution[index]
        eps = np.minimum(np.where(last >= lo, last, lo), hi)
        no_guess = ~(eps == last)
        if no_guess.any():
            eps = np.where(no_guess, _initial_guess(sigma, lo, hi, c, d), eps)

        for _ in range(self.max_iter):
            f = (((c4 * eps + c3) * eps + c2) * eps + c1) * eps + c0 - sigma
            df = ((d3 * eps + d2) * eps + d1) * eps + d0
            step = f / np.maximum(df, _MIN_SLOPE)
            new = np.minimum(np.maximum(eps - step, lo), hi)
            if np.abs(step).max() <= _NEWTON_STEP_TOL:
                self._last_solution[index] = new
                return new
            eps = new
        new = self._safeguarded_newton(sigma, eps, lo, hi, c, d)
        self._last_solution[index] = new
        return new

    def _find_segments(self, sigma: np.ndarray, index: np.ndarray) -> tuple:
        """Polynomial segment containing the solution, for each span of `index`."""
        breakpoints = self._breakpoints[:, index]
        stress_bp = self._stress_bp[:, index]
        n_bp = breakpoints.shape[0]
        columns = np.arange(index.size)

        segment = np.maximum((stress_bp <= sigma).sum(axis=0), 1)
        c = tuple(self._coefficients[segment, index].T)
        d = tuple(self._d_coefficients[segment, index].T)
        lo = breakpoints[segment - 1, columns]
        s_lo = stress_bp[segment - 1, columns]
        upper = np.minimum(segment, n_bp - 1)
        bounded = (segment < n_bp) & np.isfinite(breakpoints[upper, columns])
        hi = np.where(bounded, breakpoints[upper, columns], np.inf)
        s_hi = np.where(bounded, stress_bp[upper, columns], np.inf)

        flat_end = ~bounded & ~(_horner_d(d, lo) > 0)
        if np.any(flat_end & (sigma > s_lo)):
            raise ConvergenceError(
                "Stress exceeds the maximum stress the cable can carry",
                origin="deformation_model",
            )
        # stress range of each segment, to reuse them in next calls (segment 1 also
        # contains lower stresses, a flat last segment only its start stress)
        s_min = np.where(segment == 1, -np.inf, s_lo)
        s_max = np.where(flat_end, s_lo, s_hi)
        self._segment_cache = (index.copy(), s_min, s_max, lo, hi, c, d)
        return lo, hi, c, d

    def _safeguarded_newton(
        self,
        sigma: np.ndarray,
        eps: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        c: tuple,
        d: tuple,
    ) -> np.ndarray:
        """Newton with bisection fallback, the bracket shrinking at each iteration.
        Only used if plain Newton did not converge (inflection points)."""
        c0, c1, c2, c3, c4 = c
        d0, d1, d2, d3 = d
        for _ in range(4 * self.max_iter):
            f = (((c4 * eps + c3) * eps + c2) * eps + c1) * eps + c0 - sigma
            df = ((d3 * eps + d2) * eps + d1) * eps + d0
            lo = np.where(f < 0, eps, lo)
            hi = np.where(f > 0, eps, hi)
            new = eps - f / np.maximum(df, _MIN_SLOPE)
            inside = (df > 0) & (new >= lo) & (new <= hi)
            fallback = np.where(
                np.isfinite(hi), 0.5 * (lo + hi), 2 * eps - lo + 1e-12
            )
            new = np.where(inside, new, fallback)
            if np.abs(new - eps).max() <= _STRAIN_TOL:
                return new
            eps = new
        raise ConvergenceError(
            "Polynomial deformation law did not converge",
            origin="deformation_model",
        )


def _first_positive_root(polynomial: np.polynomial.Polynomial) -> float:
    if polynomial.trim().degree() < 1:
        return np.inf
    roots = polynomial.roots()
    positive = roots[
        (np.abs(roots.imag) <= 1e-12 * np.abs(roots.real)) & (roots.real > 0)
    ].real
    return float(positive.min()) if positive.size else np.inf


def _initial_guess(
    sigma: np.ndarray, lo: np.ndarray, hi: np.ndarray, c: tuple, d: tuple
) -> np.ndarray:
    """Secant on bounded segments (exact on linear ones), tangent otherwise."""
    bounded = np.isfinite(hi)
    s_lo = _horner_c(c, lo)
    s_hi = _horner_c(c, np.where(bounded, hi, lo))
    slope = np.where(
        bounded & (s_hi > s_lo),
        (s_hi - s_lo) / np.where(bounded, hi - lo, 1.0),
        np.maximum(_horner_d(d, lo), _MIN_SLOPE),
    )
    return np.minimum(np.maximum(lo + (sigma - s_lo) / slope, lo), hi)


def _horner_c(c: tuple, x: np.ndarray) -> np.ndarray:
    c0, c1, c2, c3, c4 = c
    return (((c4 * x + c3) * x + c2) * x + c1) * x + c0


def _horner_d(d: tuple, x: np.ndarray) -> np.ndarray:
    d0, d1, d2, d3 = d
    return ((d3 * x + d2) * x + d1) * x + d0


def _horner(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x, dtype=np.float64) + coefficients[-1]
    for c in coefficients[-2::-1]:
        result = result * x + c
    return result


def _horner_rows(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Batched evaluation: `coefficients` shape `(s, n, d+1)`, `x` shape `(s, n)`."""
    result = coefficients[..., -1].copy()
    for k in range(coefficients.shape[-1] - 2, -1, -1):
        result = result * x + coefficients[..., k]
    return result


def _shift_coefficients(
    coefficients: np.ndarray, shift: np.ndarray
) -> np.ndarray:
    """Coefficients in $\\varepsilon$ of $P(\\varepsilon - h)$, one row per value of `shift` $h$."""
    shifted = np.zeros((shift.shape[0], _N_COEFFS))
    minus_h = -shift
    for j, c_j in enumerate(coefficients):
        if c_j == 0:
            continue
        for k in range(j + 1):
            shifted[:, k] += c_j * comb(j, k) * minus_h ** (j - k)
    return shifted


def deformation_law_builder(cable_array: CableArray) -> IStressStrainLaw:
    """Select the law from cable data: polynomial if a stress-strain polynomial has degree >= 2."""
    data = cable_array.data
    nonlinear = any(
        float(data[f"{prefix}{k}"].iloc[0]) != 0
        for prefix in ("a", "b")
        for k in range(2, _N_COEFFS)
    )
    if nonlinear:
        return PolynomialLaw.from_cable_array(cable_array)
    return LinearLaw.from_cable_array(cable_array)
