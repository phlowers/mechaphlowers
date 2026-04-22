# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


class SolverError(Exception):
    """Base class for solver errors."""

    def __init__(
        self,
        message: str,
        origin: str = "unknown",
        level: str = "ERROR",
        details: str = "",
    ) -> None:
        """SolverError specific exception.

        origin attribute is available to add origin of the error (e.g., class name, calling function, etc.)

        Args:
            message (str): error message
            origin (str, optional): origin of the error. Defaults to "unknown"
            level (str, optional): error level. Defaults to "ERROR".
            details (str, optional): error details. Defaults to "".

        Examples:

            >>> error = SolverError(
            ...     "An error occurred", level="CRITICAL", details="Matrix is singular"
            ... )
            >>> error.origin = "MatrixSolver"
            >>> raise error
            [CRITICAL][MatrixSolver] An error occurred | Matrix is singular

        """
        self.level = level
        self.details = details
        self.origin = origin
        prefix = f"[{level}][{self.origin}]"

        super().__init__(f"{prefix} {message} | {details}")


class SuspectedChainReversal(SolverError):
    """Raised in solver when chain is suspected to be reversed
    (above horizontal position)
    """


class ConvergenceError(SolverError):
    """Raised when solver fails to converge."""


class ShapeError(ValueError):
    """Raised when there is a shape mismatch in arrays."""


class DataWarning(UserWarning):
    """Base class for data-related warnings."""


class BalanceEngineWarning(UserWarning):
    """Base class for balance-related warnings."""


class ViewChoiceWarning(UserWarning):
    """Base class for choice of view (ex: choice of support or span view)."""


class RtsDataNotAvailable(ValueError):
    """Raised when RTS catalog data (rts_cable, rts_layer_*) is missing or NaN."""


class MeasurementDataNotAvailable(ValueError):
    """Raised when measurement data are not available for computation."""
