# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EstimationResult:
    """Result of an inverse estimation solve.

    Attributes:
        value: The estimated variable value that satisfies the target.
        residual: Final residual ``|f(value) - target|``.
        iterations: Number of algorithm iterations.
        converged: Whether the algorithm converged within tolerance.
        extra: Additional named values computed by the estimation, as a
            ``{name: value}`` mapping (e.g. ``{"distance": 8.99}``). Empty
            when the estimation provides none.
    """

    value: float
    residual: float
    iterations: int
    converged: bool
    extra: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "converged" if self.converged else "NOT converged"
        extra = "".join(f", {k}={v:.6g}" for k, v in self.extra.items())
        return (
            f"EstimationResult(value={self.value:.6g}, "
            f"residual={self.residual:.3e}, "
            f"iterations={self.iterations}{extra}, {status})"
        )
