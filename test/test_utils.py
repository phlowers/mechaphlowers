# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np

from mechaphlowers.utils import CachedAccessor, ppnp

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
