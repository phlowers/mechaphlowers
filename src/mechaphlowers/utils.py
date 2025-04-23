# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging

import numpy as np
import pandas as pd


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

    def __init__(self, name: str, accessor: object) -> None:
        self._name: str = name
        self._accessor: object = accessor

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
