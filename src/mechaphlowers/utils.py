# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np


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
