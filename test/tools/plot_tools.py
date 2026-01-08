# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np

from mechaphlowers.core.geometry.points import Points


def assert_cable_linked_to_attachment(
    span_points: Points,
    insulators_points: Points,
    rtol=1e-7,
    atol=0,
):
    """Assert that span coordinates start and end at the same points than the attachments.
    Points can be fetched by calling SectionPoints.get_points_for_plot(), or SectionsPoints.get_spans("section")/SectionsPoints.get_insulators()

    Args:
        span_points (Points): Points object for spans.
        insulators_points (Points): Points for insulators.
    """
    # Get first point of each span
    start_cable = span_points.coords[:, 0]
    # Get last point of each span
    end_cable = span_points.coords[:, -1]
    # Get attachments coords
    attachments = insulators_points.coords[:, -1]
    # Test that first point of the span has same coords than the corresponding attachment
    np.testing.assert_allclose(
        start_cable, attachments[:-1], rtol=rtol, atol=atol
    )
    # Test that last point of the span has same coords than the corresponding attachment
    np.testing.assert_allclose(
        end_cable, attachments[1:], rtol=rtol, atol=atol
    )
