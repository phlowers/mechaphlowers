# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import logging
from time import sleep, time

import numpy as np
from xxhash import xxh3_64

from mechaphlowers.config import options
from mechaphlowers.utils import (
    CachedAccessor,
    check_time,
    hash_numpy_xxhash,
    numpy_cache,
    ppnp,
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
    np.isnan(A).sum() > 1

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
    len(compute_sum._cache) == 0
