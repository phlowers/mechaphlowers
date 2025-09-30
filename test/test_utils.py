# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import hashlib
from time import sleep
import numpy as np

from mechaphlowers.utils import CachedAccessor, memoizer, ppnp
from functools import wraps, lru_cache
from mechaphlowers.utils import np_cache

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
    # def test_np_cache():

    @np_cache(maxsize=256)
    def multiply(array, factor):
        sleep(5)
        return factor * array

    array = np.array([1, 2, 3,4, 5, 6])
    
    @np_cache(maxsize=256)
    def cached_x_m(
        a: np.ndarray,
        b: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:

        return -a / 2 + p * np.arcsinh(b / (2 * p * np.sinh(a / (2 * p))))
    
    def x_m(
        a: np.ndarray,
        b: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:

        return -a / 2 + p * np.arcsinh(b / (2 * p * np.sinh(a / (2 * p))))
    
    a = np.array([501.3, 499.0]*100)  # test here int and float
    b = np.array([0.0, -5.0]*100)
    p = np.array([2_112.2, 2_112.0]*100)
    
    import timeit
    
    
    timeit.timeit(memoizer(x_m(a, b, p)))
    timeit.timeit(cached_x_m(a, b, p))
    timeit.timeit(x_m(a, b, p))
    
    
    hashlib.sha256( a.data )
        
    # Test basic functionality
    result1 = multiply(array, 2)
    np.testing.assert_array_equal(result1, array * 2)
    
    # Test caching - second call should hit cache
    result2 = multiply(array, 2) 
    assert multiply.cache_info().hits == 1
    assert multiply.cache_info().misses == 1
    
    # Test cache miss with different args
    result3 = multiply(array, 3)
    assert multiply.cache_info().hits == 1 
    assert multiply.cache_info().misses == 2
    
    # Test cache works with same array data but different object
    array2 = np.array([[1, 2, 3], [4, 5, 6]])
    result4 = multiply(array2, 2)
    assert multiply.cache_info().hits == 2
    assert multiply.cache_info().misses == 2
    
    # Test cache clear
    multiply.cache_clear()
    result5 = multiply(array, 2)
    assert multiply.cache_info().hits == 0
    assert multiply.cache_info().misses == 1

