# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from functools import wraps
from time import time
from typing import Any, Callable, Dict, Protocol, TypeVar, cast

import numpy as np
import pandas as pd
from xxhash import xxh3_64

from mechaphlowers.config import options

logger = logging.getLogger(__name__)


def ppnp(arr: np.ndarray, prec: int = 2):
    """ppnp helper function to force display without scientific notation

    Args:
        arr (np.ndarray): array to print
        prec (float, optional): floating precision. Defaults to 2.
    """
    print(np.array_str(arr, precision=prec, suppress_small=True))


class CachedAccessor:
    """
    Custom property-like object.

    A descriptor for caching accessors.

    Parameters
    ----------
    name : str
        Namespace that will be accessed under, e.g. ``df.foo``.
    accessor : cls
        Class with the extension methods.

    Notes
    -----
    For accessor, the class's __init__ method assume to get the object in parameter
    """

    def __init__(self, name: str, accessor: Callable) -> None:
        self._name: str = name
        self._accessor: Callable = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.geo
            return self._accessor
        accessor_obj = self._accessor(obj)
        # Replace the property with the accessor object. Inspired by:
        # https://www.pydanny.com/cached-property.html and pandas CachedAccessor
        # https://github.com/pandas-dev/pandas/blob/v2.2.3/pandas/core/accessor.py
        # We need to use object.__setattr__ because we overwrite __setattr__ on
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


def float_to_array(data: Dict[str, np.ndarray | float | int]) -> Dict:
    """Convert inputs to the required format."""
    for key, value in data.items():
        if isinstance(value, (int, float)):
            data[key] = np.array([value, np.nan])
        if isinstance(value, np.ndarray) and value.size == 1:
            data[key] = np.array([value[0], np.nan])
    return data


def add_stderr_logger(
    level: int = logging.DEBUG,
):
    """Helper for quickly adding a StreamHandler to the logger. Useful for
    debugging. Inspired by the urllib3 library.

    Args:
            level (int): The logging level to set for the handler. Default is DEBUG.

    Returns:
            (logging.StreamHandler): the handler after adding it.

    Examples:
                >>> from mechaphlowers import add_stderr_logger
                >>> add_stderr_logger(logging.DEBUG)
                >>> # In the example.log file:
                >>> # 2025-03-28 21:33:42,437 - mechaphlowers - INFO - Added a stderr logging handler to logger: mechaphlowers
    """
    # This method needs to be in this __init__.py to get the __name__ correct
    # even if mechaphlowers is vendored within another package.
    logger = logging.getLogger("mechaphlowers")
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added a stderr logging handler to logger: %s", __name__)
    return handler


def df_to_dict(df: pd.DataFrame) -> dict:
    """Convert a pandas.DataFrame to a dictionary.

    Would be an equivalent to df.to_dict(orient='numpy.ndarray') if it existed.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
            DESCRIPTION.
    """
    q = df.to_dict(orient="list")
    for k in q.keys():
        if len(q[k]) > 1:
            q[k] = np.array(q[k])
        else:
            q[k] = q[k][0]
    return q


def reduce_to_span(array_to_reduce: np.ndarray):
    return array_to_reduce[0:-1]


def fill_to_support(array_to_fill: np.ndarray):
    return np.concatenate((array_to_fill, [np.nan]))


class ArrayTools:
    @staticmethod
    def decr(array: np.ndarray) -> np.ndarray:
        """Reduces size of array by removing last element.

        There are two sizes of arrays in mechaphlowers:
        - number of supports (N), usually with NaN as last element
        - number of spans (N-1)
        This function is used to convert an array of support size (N) to span size (N-1).

        >>> array = np.array([1, 2, 3, np.nan])
        >>> ArrayTools.decr(array)
        array([1, 2, 3])

        Args:
            array (np.ndarray): full array to reduce

        Returns:
            np.ndarray: array with size reduced by one
        """
        return reduce_to_span(array)

    @staticmethod
    def incr(array: np.ndarray) -> np.ndarray:
        """Increases size of array by adding a NaN as last element.

        There are two sizes of arrays in mechaphlowers:
        - number of supports (N), usually with NaN as last element
        - number of spans (N-1)
        This function is used to convert an array of span size (N-1) to support size (N).

        >>> array = np.array([1, 2, 3])
        >>> ArrayTools.incr(array)
        array([1, 2, 3, np.nan])

        Args:
            array (np.ndarray): full array to increase

        Returns:
            np.ndarray: array with size increased by one
        """
        return fill_to_support(array)


arr = ArrayTools()


def check_time(f):
    @wraps(f)
    def wrap(self, *args, **kw):
        if options.log.perfs:
            ts = time()
            result = f(self, *args, **kw)
            te = time()
            logger.debug(
                f"function: {f.__name__} with args:[{args}, {kw} solved in {te-ts:.4f} seconds."
            )
        else:
            result = f(self, *args, **kw)
        return result

    return wrap


def hash_numpy_xxhash(array: np.ndarray) -> bytes:
    """Hash a numpy array using xxhash.

    Args:
        array (np.ndarray): The numpy array to hash.

    Returns:
        bytes: The hash digest of the array.
    """
    return xxh3_64(array.tobytes()).digest()


T = TypeVar("T", bound=Callable[..., Any])


class CachedCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    _cache: Dict[bytes, Any]

    def cache_clear(self) -> None: ...


def numpy_cache(f: Callable[..., Any]):
    """Decorator to cache numpy array results of a function based on its arguments.

    Warning: it is not designed for complex uses. For example, view of a same array are not distinguished, and there is no checks on contiguousness of arrays.

    Args:
        f (Callable[..., Any]): The function to be cached.

    Returns:
        Callable[..., Any]: The wrapped function with caching.
    """
    cache: dict[bytes, Any] = {}

    @wraps(f)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        key_args = str(
            [
                hash_numpy_xxhash(arg)
                if isinstance(arg, np.ndarray)
                else str(arg)
                for arg in args
            ]
        )
        key_args += str(
            [
                hash_numpy_xxhash(kwargs[k])
                if isinstance(kwargs[k], np.ndarray)
                else str(kwargs[k])
                for k in kwargs
            ]
        )
        # key_args += key_kwargs
        key = xxh3_64(key_args).digest()
        if key not in cache:
            cache[key] = f(*args, **kwargs)
        return cache[key]

    def cache_clear() -> None:
        """Clear the cache."""
        cache.clear()

    _wrapped_any = cast(Any, wrapped)
    _wrapped_any.cache_clear = cache_clear
    _wrapped_any._cache = cache
    return cast(CachedCallable, _wrapped_any)
