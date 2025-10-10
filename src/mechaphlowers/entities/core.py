# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


"""This module is dedicated to the base objects core relies on."""

from __future__ import annotations

from typing import Annotated

import numpy as np


def reduce_to_span(array_to_reduce: np.ndarray):
    return array_to_reduce[0:-1]


def fill_to_support(array_to_fill: np.ndarray):
    return np.concatenate((array_to_fill, [np.nan]))


# class SupportArray:
#     """"""

#     def __init__(self, support_array: np.ndarray) -> None:
#         self._data = support_array

#     def __str__(self) -> str:
#         return str(self._data)

#     def __repr__(self) -> str:
#         return repr(self._data)

#     def to_numpy(self) -> np.ndarray:
#         return self._data

#     def __copy__(self) -> SupportArray:
#         return type(self)(self._data.copy())

#     def to_spans(self) -> SpanArray:
#         return SpanArray.from_supports(self._data)

#     @staticmethod
#     def from_spans(span_array) -> SupportArray:
#         return SupportArray(fill_to_support(span_array))


# class SpanArray:
#     """"""

#     def __init__(self, span_array: np.ndarray) -> None:
#         self._data = span_array

#     def __str__(self) -> str:
#         return str(self._data)

#     def __repr__(self) -> str:
#         return repr(self._data)

#     def to_numpy(self) -> np.ndarray:
#         return self._data

#     def __copy__(self) -> SupportArray:
#         return type(self)(self._data.copy())

#     def to_supports(self) -> np.ndarray:
#         return SupportArray.from_spans(self._data)

#     @staticmethod
#     def from_supports(support_array) -> np.ndarray:
#         return SpanArray(reduce_to_span(support_array))


class Resize:
    @staticmethod
    def to_span(array: np.ndarray) -> np.ndarray:
        return reduce_to_span(array)

    @staticmethod
    def to_support(array: np.ndarray) -> np.ndarray:
        return fill_to_support(array)


size = Resize()


SpanSizeArray = Annotated[np.ndarray, "SpanSize"]
SupportSizeArray = Annotated[np.ndarray, "SupportSize"]
