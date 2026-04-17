# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


class Spacer:
    """Spacer equipment for bundle conductors.

    A spacer maintains separation between conductors in a bundle.
    Its height contribution depends on the bundle configuration.

    Args:
        length: Length of the spacer in meters. Defaults to 0.2.
    """

    def __init__(self, length: float = 0.2) -> None:
        self.length = length

    def height(self, bundle_number: int) -> float:
        """Return spacer height contribution based on bundle number.

        Returns ``length`` for bundle_number 3 or 4, else 0.

        Args:
            bundle_number: Number of conductors in the bundle.

        Returns:
            float: Spacer height contribution in meters.
        """
        if bundle_number in (3, 4):
            return self.length
        return 0.0
