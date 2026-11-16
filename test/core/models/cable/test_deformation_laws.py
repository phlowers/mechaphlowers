# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import warnings

import numpy as np
import pytest

from mechaphlowers.core.models.cable.deformation import DeformationRte
from mechaphlowers.core.models.cable.deformation_laws import (
    DeformationState,
    LinearLaw,
    PolynomialLaw,
    deformation_law_builder,
)
from mechaphlowers.entities.arrays import CableArray
from mechaphlowers.entities.errors import ConvergenceError


def direct_stress(
    law: PolynomialLaw,
    eps: np.ndarray,
    theta: np.ndarray,
    state: DeformationState,
) -> np.ndarray:
    """Reference implementation of the total stress, from the piecewise definition."""
    total = np.zeros_like(eps, dtype=np.float64)
    for i, material in enumerate(law.materials):
        strain = eps - material.dilatation_coefficient * (
            theta - law.temperature_reference
        )
        x_max = state.max_strain[i]
        poly = np.polynomial.Polynomial(material.coefficients)

        def virgin(x):
            return poly(np.minimum(x, material.max_virgin_strain))

        x_perm = x_max - virgin(x_max) / material.young_modulus
        stress = np.where(
            strain >= x_max,
            virgin(strain),
            np.where(
                strain >= x_perm,
                material.young_modulus * (strain - x_perm),
                0.0,
            ),
        )
        total += stress
    return total


def loaded_state(
    law: PolynomialLaw, n: int, sigma: float, theta: float
) -> DeformationState:
    """History after a loading at reduced stress `sigma` (Pa) and temperature `theta`."""
    tension = np.full(n, sigma * law.section)
    return law.update_state(tension, np.full(n, theta), law.initial_state(n))


@pytest.fixture
def narcisse_law(cable_array_NARCISSE600G: CableArray) -> PolynomialLaw:
    law = deformation_law_builder(cable_array_NARCISSE600G)
    assert isinstance(law, PolynomialLaw)
    return law


@pytest.fixture
def conductor_law(narcisse_law: PolynomialLaw) -> PolynomialLaw:
    """One-material law: the conductor alone, which has a finite plateau."""
    return PolynomialLaw(
        [narcisse_law.materials[0]],
        narcisse_law.section,
        narcisse_law.temperature_reference,
    )


LAW_CASES = [("narcisse_law", 150e6), ("conductor_law", 40e6)]


@pytest.mark.parametrize("law_name, sigma_max", LAW_CASES)
@pytest.mark.parametrize("with_history", [False, True])
def test_residual_matches_piecewise_definition(
    request: pytest.FixtureRequest,
    law_name: str,
    sigma_max: float,
    with_history: bool,
) -> None:
    law = request.getfixturevalue(law_name)
    rng = np.random.default_rng(0)
    n = 50
    sigma = rng.uniform(0.0, sigma_max, n)
    theta = rng.uniform(-20.0, 80.0, n)
    if with_history:
        state = loaded_state(law, n, sigma=0.8 * sigma_max, theta=-5.0)
    else:
        state = law.initial_state(n)

    eps = law.prepare(theta, state).epsilon(sigma * law.section)

    np.testing.assert_allclose(
        direct_stress(law, eps, theta, state), sigma, rtol=1e-9, atol=1e-6
    )


def test_elastic_range_is_linear_in_tension(
    narcisse_law: PolynomialLaw,
) -> None:
    n = 1
    theta = np.full(n, 15.0)
    state = loaded_state(narcisse_law, n, sigma=120e6, theta=-5.0)
    prepared = narcisse_law.prepare(theta, state)
    sigma = np.linspace(0.6, 0.95, 5) * 120e6
    eps = np.array(
        [
            prepared.epsilon(np.full(n, s * narcisse_law.section))[0]
            for s in sigma
        ]
    )

    second_difference = eps[2:] - 2 * eps[1:-1] + eps[:-2]
    np.testing.assert_allclose(second_difference, 0.0, atol=1e-13)


@pytest.mark.parametrize("with_history", [False, True])
def test_epsilon_is_monotonic_across_breakpoints(
    narcisse_law: PolynomialLaw, with_history: bool
) -> None:
    n = 3001
    theta = np.full(n, 15.0)
    state = (
        loaded_state(narcisse_law, n, sigma=120e6, theta=-5.0)
        if with_history
        else narcisse_law.initial_state(n)
    )
    # one span per grid point: the state is per span, so each point is independent
    sigma = np.linspace(0.0, 150e6, n)
    eps = narcisse_law.prepare(theta, state).epsilon(
        sigma * narcisse_law.section
    )
    steps = np.diff(eps)
    assert np.all(steps > 0)
    assert steps.max() / steps.min() < 10


def test_update_state_ratchet(narcisse_law: PolynomialLaw) -> None:
    n = 4
    theta = np.full(n, 10.0)
    virgin = narcisse_law.initial_state(n)
    high = narcisse_law.update_state(
        np.full(n, 100e6 * narcisse_law.section), theta, virgin
    )
    assert np.all(high.max_strain >= virgin.max_strain)
    assert np.all(high.max_strain > 0)

    again = narcisse_law.update_state(
        np.full(n, 100e6 * narcisse_law.section), theta, high
    )
    np.testing.assert_array_equal(again.max_strain, high.max_strain)

    lower = narcisse_law.update_state(
        np.full(n, 20e6 * narcisse_law.section), theta, high
    )
    np.testing.assert_array_equal(lower.max_strain, high.max_strain)

    higher = narcisse_law.update_state(
        np.full(n, 140e6 * narcisse_law.section), theta, high
    )
    assert np.all(higher.max_strain > high.max_strain)


def test_nan_inputs_give_nan_for_this_span_only(
    narcisse_law: PolynomialLaw,
) -> None:
    n = 3
    state = loaded_state(narcisse_law, n, sigma=100e6, theta=-5.0)
    theta = np.array([np.nan, 15.0, 15.0])
    tension = np.array([50e6, np.nan, 50e6]) * narcisse_law.section

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        eps = narcisse_law.prepare(theta, state).epsilon(tension)
        eps_nan_free = narcisse_law.prepare(
            theta[2:], state.select(np.array([2]))
        ).epsilon(tension[2:])

    assert np.isnan(eps[0])
    assert np.isnan(eps[1])
    np.testing.assert_allclose(eps[2:], eps_nan_free, rtol=1e-13)


@pytest.mark.parametrize("ratio, raises", [(0.99, False), (1.01, True)])
def test_stress_above_plateau_raises(
    conductor_law: PolynomialLaw, ratio: float, raises: bool
) -> None:
    sigma = ratio * conductor_law.materials[0].max_virgin_stress
    prepared = conductor_law.prepare(
        np.array([15.0]), conductor_law.initial_state(1)
    )
    tension = np.array([sigma * conductor_law.section])
    if raises:
        with pytest.raises(ConvergenceError):
            prepared.epsilon(tension)
    else:
        assert np.isfinite(prepared.epsilon(tension)).all()


def test_warm_start_cache_does_not_change_results(
    narcisse_law: PolynomialLaw,
) -> None:
    n = 2
    theta = np.array([15.0, -10.0])
    state = loaded_state(narcisse_law, n, sigma=110e6, theta=-5.0)
    t1 = np.array([60e6, 30e6]) * narcisse_law.section
    t2 = np.array([100e6, 90e6]) * narcisse_law.section

    prepared = narcisse_law.prepare(theta, state)
    first = prepared.epsilon(t1)
    prepared.epsilon(t2)
    again = prepared.epsilon(t1)

    fresh = narcisse_law.prepare(theta, state).epsilon(t1)
    np.testing.assert_allclose(again, fresh, rtol=1e-13)
    np.testing.assert_allclose(again, first, rtol=1e-13)


def test_linear_law_is_exact(cable_array_AM600: CableArray) -> None:
    law = deformation_law_builder(cable_array_AM600)
    assert isinstance(law, LinearLaw)
    n = 3
    theta = np.array([-5.0, 15.0, 60.0])
    tension = np.array([1e4, 5e4, 2e4])
    E, S, alpha = law.young_modulus, law.section, law.dilatation_coefficient
    tref = law.temperature_reference

    prepared = law.prepare(theta, law.initial_state(n))

    np.testing.assert_array_equal(
        prepared.epsilon(tension), tension / (E * S) + (theta - tref) * alpha
    )
    np.testing.assert_array_equal(
        prepared.epsilon_0(tension), tension / (E * S) + theta * alpha
    )
    state = law.initial_state(n)
    assert state.max_strain is None
    assert law.update_state(tension, theta, state) is state


def test_deformation_law_builder_selects_law(
    cable_array_AM600: CableArray, cable_array_NARCISSE600G: CableArray
) -> None:
    assert isinstance(deformation_law_builder(cable_array_AM600), LinearLaw)

    law = deformation_law_builder(cable_array_NARCISSE600G)
    assert isinstance(law, PolynomialLaw)
    assert len(law.materials) == 2
    materials = law.materials
    np.testing.assert_allclose(
        law.young_modulus_equivalent,
        materials[0].young_modulus + materials[1].young_modulus,
        rtol=1e-15,
    )
    expected_alpha = sum(
        m.dilatation_coefficient * m.young_modulus for m in materials
    ) / sum(m.young_modulus for m in materials)
    np.testing.assert_allclose(
        law.dilatation_coefficient_equivalent, expected_alpha, rtol=1e-15
    )


def test_deformation_state_operations() -> None:
    state = DeformationState(np.arange(6.0).reshape(2, 3))
    selected = state.select(np.array([True, False, True]))
    assert selected.max_strain.shape == (2, 2)
    assert state.select(np.array([2])).max_strain.shape == (2, 1)

    joined = DeformationState.concatenate([state, state])
    assert joined.max_strain.shape == (2, 6)

    with_nan = DeformationState(np.array([[np.nan, 1.0], [0.5, np.nan]]))
    other = DeformationState(np.array([[2.0, np.nan], [0.1, 0.2]]))
    np.testing.assert_array_equal(
        with_nan.maximum(other).max_strain, [[2.0, 1.0], [0.5, 0.2]]
    )

    with pytest.raises(ValueError):
        state.max_strain[0, 0] = 1.0


def test_deformation_state_without_history() -> None:
    empty = DeformationState()
    other = DeformationState()
    assert empty.select(np.array([0])) is empty
    assert DeformationState.concatenate([empty, other]) is empty
    assert empty.maximum(other) is empty


def make_rte(cable_array: CableArray, n: int = 2) -> DeformationRte:
    law = deformation_law_builder(cable_array)
    return DeformationRte(
        tension_mean=np.full(n, 5e4),
        cable_length=np.full(n, 500.0),
        cable_section_area=np.float64(law.section),
        linear_weight=np.float64(cable_array.data.linear_weight.iloc[0]),
        young_modulus=np.float64(law.young_modulus_equivalent),
        dilatation_coefficient=np.float64(
            law.dilatation_coefficient_equivalent
        ),
        temperature_reference=np.float64(law.temperature_reference),
        polynomial_conductor=cable_array.polynomial_conductor,
        sagging_temperature=np.full(n, 15.0),
        law=law,
    )


def test_facade_caches_prepared_law(
    cable_array_NARCISSE600G: CableArray,
) -> None:
    model = make_rte(cable_array_NARCISSE600G)
    model.epsilon()
    first = model.prepared
    model.epsilon()
    assert model.prepared is first

    model.current_temperature = np.full(2, 15.0)
    assert model.prepared is first

    model.current_temperature = np.full(2, 30.0)
    assert model.prepared is not first

    model.state = model.state
    second = model.prepared
    model.state = DeformationState(model.state.max_strain.copy())
    assert model.prepared is not second
