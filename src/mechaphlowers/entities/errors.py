# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


class SolverError(Exception):
    """Base class for solver errors."""

    def __init__(
        self, message: str, level: str = "ERROR", details: str = ""
    ) -> None:
        """SolverError specific exception.

        origin attribute is available to add origin of the error (e.g., class name, calling function, etc.)

        Args:
            message (str): error message
            level (str, optional): error level. Defaults to "".
            details (str, optional): error details. Defaults to "".

        Example:

            >>> error = SolverError(
            ...     "An error occurred", level="CRITICAL", details="Matrix is singular"
            ... )
            >>> error.origin = "MatrixSolver"
            >>> raise error
            [CRITICAL][MatrixSolver] An error occurred | Matrix is singular

        """
        self.level = level
        self.details = details
        self.origin = "unknown"
        prefix = f"[{level}][{self.origin}]"

        super().__init__(f"{prefix} {message} | {details}")


class ConvergenceError(SolverError):
    """Raised when solver fails to converge."""


class ShapeError(ValueError):
    """Raised when there is a shape mismatch in arrays."""
