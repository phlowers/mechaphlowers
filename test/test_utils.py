# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import logging
from time import sleep, time

import numpy as np
import pytest
from xxhash import xxh3_64

from mechaphlowers.config import options
from mechaphlowers.utils import (
    CachedAccessor,
    acotan,
    check_inputs_are_numbers,
    check_time,
    cotan,
    hash_numpy_xxhash,
    numpy_cache,
    ppnp,
    span_to_support_view_guying,
)

# FILE: src/mechaphlowers/test_utils.py


class MockAccessor:
    def __init__(self, obj):
        self.obj = obj


class MockClass:
    accessor = CachedAccessor("accessor", MockAccessor)


def test_accessor_from_class() -> None:
    assert MockClass.accessor == MockAccessor


def test_accessor_from_instance() -> None:
    instance = MockClass()
    accessor_instance = instance.accessor
    assert isinstance(accessor_instance, MockAccessor)
    assert accessor_instance.obj == instance


def test_accessor_is_cached() -> None:
    instance = MockClass()
    accessor_instance1 = instance.accessor
    accessor_instance2 = instance.accessor
    assert accessor_instance1 is accessor_instance2


def test_ppnp(capsys) -> None:
    arr = np.array([1.123456, 2.123456, 3.123456])
    ppnp(arr, prec=2)
    captured = capsys.readouterr()
    assert captured.out == "[1.12 2.12 3.12]\n"


def test_ppnp_default_precision(capsys) -> None:
    arr = np.array([1.123456, 2.123456, 3.123456])
    ppnp(arr)
    captured = capsys.readouterr()
    assert captured.out == "[1.12 2.12 3.12]\n"


def test_ppnp_high_precision(capsys) -> None:
    arr = np.array([1.123456, 2.123456, 3.123456])
    ppnp(arr, prec=4)
    captured = capsys.readouterr()
    assert captured.out == "[1.1235 2.1235 3.1235]\n"


def test_log(caplog) -> None:
    save_option = options.log.perfs

    class TestClass:
        @check_time
        def function_to_test(self) -> int:
            logging.info("function executed")
            return 1

    caplog.set_level(logging.DEBUG)
    test_class = TestClass()

    options.log.perfs = False
    test_class.function_to_test()
    assert 'function_to_test' not in caplog.text
    assert 'seconds' not in caplog.text
    assert 'function executed' in caplog.text

    options.log.perfs = True
    test_class.function_to_test()

    assert 'function_to_test' in caplog.text
    assert 'seconds' in caplog.text
    assert 'function executed' in caplog.text

    options.log.perfs = save_option


def test_hash_numpy_xxhash() -> None:
    arr = np.random.rand(10, 100, 3)
    expected_hash = xxh3_64(arr.tobytes()).digest()
    computed_hash = hash_numpy_xxhash(arr)
    assert computed_hash == expected_hash

    A = np.random.randn(10, 10, 3)
    A.ravel()[np.random.choice(A.size, 10, replace=False)] = np.nan
    assert np.isnan(A).sum() > 1

    computed_hash_1 = hash_numpy_xxhash(A)
    computed_hash_2 = hash_numpy_xxhash(A)
    assert computed_hash_1 == computed_hash_2

    computed_hash_1 = hash_numpy_xxhash(A + 1)
    computed_hash_2 = hash_numpy_xxhash(A)
    assert computed_hash_1 != computed_hash_2


def test_numpy_cache_decorator() -> None:
    call_count = {"count": 0}

    @numpy_cache
    def compute_sum(arr: np.ndarray, arg2=1) -> float:
        sleep(0.000001)  # Simulate a time-consuming computation
        call_count["count"] += 1
        return np.nansum(arr)

    A = np.random.randn(10, 10, 3)
    A.ravel()[np.random.choice(A.size, 10, replace=False)] = np.nan

    result1 = compute_sum(A)
    result2 = compute_sum(A)

    assert result1 == result2
    assert call_count["count"] == 1

    B = A + 1
    result3 = compute_sum(B)

    assert result3 != result1
    assert call_count["count"] == 2

    t0 = time()
    compute_sum(A + 2)
    t1 = time()
    compute_sum(A + 2)
    t2 = time()
    print(
        f"First call took {t1 - t0:.6f} seconds, second call took {t2 - t1:.6f} seconds"
    )
    assert t1 - t0 > t2 - t1  # Second call should be faster due to caching

    call_count["count"] = 0
    compute_sum(B, 10)
    compute_sum(B, 10)
    assert call_count["count"] == 1
    compute_sum(B, 1)
    assert call_count["count"] == 2
    compute_sum(B, "arg")
    assert call_count["count"] == 3
    compute_sum(B, "arg")
    assert call_count["count"] == 3
    compute_sum(B, np.nan)
    assert call_count["count"] == 4
    compute_sum(B, np.nan)
    assert call_count["count"] == 4

    assert len(compute_sum._cache) == 7

    compute_sum.cache_clear()
    assert len(compute_sum._cache) == 0


def test_perf_numpy_cache_decorator() -> None:
    @numpy_cache
    def compute_sum(arr: np.ndarray) -> float:
        sleep(0.01)  # Simulate a time-consuming computation
        return np.nansum(arr)

    A = np.random.randn(1000, 1000, 3)
    A.ravel()[np.random.choice(A.size, 1000, replace=False)] = np.nan

    start_time = time()
    compute_sum(A)
    first_call_duration = time() - start_time

    start_time = time()
    compute_sum(A)
    second_call_duration = time() - start_time

    print(f"First call duration: {first_call_duration:.6f} seconds")
    print(f"Second call duration: {second_call_duration:.6f} seconds")

    assert second_call_duration < first_call_duration
    compute_sum.cache_clear()
    compute_sum._cache
    assert len(compute_sum._cache) == 0


def test_view_error() -> None:
    with pytest.raises(ValueError):
        span_to_support_view_guying(1, "xxx")  # type: ignore[arg-type]


def test_change_view_guying() -> None:
    # span 1 and left support means guying system is on support 1, at its left
    assert span_to_support_view_guying(
        span_index=1, selected_support="left"
    ) == (1, "left")
    # span 1 and right support means guying system is on support 2, at its right
    assert span_to_support_view_guying(
        span_index=1, selected_support="right"
    ) == (2, "right")


@pytest.mark.parametrize(
    "valid_input",
    [
        0,
        42,
        3.14,
        -2.5,
        1e-5,
        float("-inf"),
        float("nan"),
        np.int32(10),
        np.float64(2.5),
    ],
)
def test_check_inputs_are_numbers_valid(valid_input) -> None:
    # Single and multiple valid numeric arguments should not raise
    check_inputs_are_numbers(a=valid_input)
    check_inputs_are_numbers(x=valid_input, y=10, z=2.5)


def test_check_inputs_are_numbers_empty() -> None:
    # Calling with no kwargs should succeed without error
    check_inputs_are_numbers()


@pytest.mark.parametrize(
    ("invalid_input", "type_name"),
    [
        (True, "bool"),
        (False, "bool"),
        ("123", "str"),
        (None, "NoneType"),
        ([1, 2], "list"),
        ({"a": 1}, "dict"),
        ((1, 2), "tuple"),
        (complex(1, 2), "complex"),
        (np.array([1, 2]), "ndarray"),
    ],
)
def test_check_inputs_are_numbers_invalid(invalid_input, type_name) -> None:
    with pytest.raises(
        TypeError,
        match=f"Argument val should be a number, but got {type_name}",
    ):
        check_inputs_are_numbers(val=invalid_input)


def test_check_inputs_are_numbers_mixed_valid_and_invalid() -> None:
    with pytest.raises(
        TypeError,
        match="Argument bad should be a number, but got str",
    ):
        check_inputs_are_numbers(good1=1, good2=2.5, bad="not a number")


def test_cotan_passing_cases() -> None:
    assert cotan(np.pi / 2) == pytest.approx(0.0, abs=1e-12)

    assert cotan(3 * np.pi / 4) == pytest.approx(-1.0, rel=1e-7)

    # Test with numpy array input
    angles = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    expected = np.array([1.0, 0.0, -1.0])
    np.testing.assert_allclose(cotan(angles), expected, atol=1e-12)


@pytest.mark.parametrize(
    "invalid_input",
    [
        0,
        np.pi,
        -np.pi,
        np.array([np.pi / 4, np.pi]),
        np.array([0.0, np.pi / 2]),
    ],
)
def test_cotan_error_cases(invalid_input) -> None:
    with pytest.raises(
        ValueError, match="x must be different from 0, pi etc."
    ):
        cotan(invalid_input)


def test_acotan() -> None:
    # Scalar inputs
    assert acotan(0) == pytest.approx(np.pi / 2, abs=1e-12)
    assert acotan(1.0) == pytest.approx(np.pi / 4, abs=1e-12)
    assert acotan(np.sqrt(3)) == pytest.approx(np.pi / 6, abs=1e-12)
    assert acotan(float("-inf")) == pytest.approx(np.pi, abs=1e-12)

    # Array input
    inputs = np.array([0.0, 1 / np.sqrt(3)])
    expected = np.array(
        [
            np.pi / 2,
            np.pi / 3,
        ]
    )
    np.testing.assert_allclose(acotan(inputs), expected, atol=1e-12)
