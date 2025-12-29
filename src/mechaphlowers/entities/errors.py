# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


class SolverError(Exception):
    """Base class for solver errors."""
    def __init__(self, message: str, level: str = "", details: str = "") -> None:
       self.level = level
       self.details = details
       prefix = f"[{level}]"
       
       super().__init__(f"{prefix} {message} | {details}")

class ConvergenceError(SolverError):
    """Raised when solver fails to converge."""
    pass

class ShapeError(ValueError):
    """Raised when there is a shape mismatch in arrays."""
    pass