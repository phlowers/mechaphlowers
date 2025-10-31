# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from mechaphlowers.data.catalog import section_factory_sample_data


def test_section_factory_sample_data():
    size_section = 10
    data = section_factory_sample_data(size_section=size_section)

    assert len(data["name"]) == size_section
    assert len(data["suspension"]) == size_section
    assert len(data["conductor_attachment_altitude"]) == size_section
    assert len(data["crossarm_length"]) == size_section
    assert len(data["insulator_length"]) == size_section
    assert len(data["span_length"]) == size_section
    assert len(data["line_angle"]) == size_section
    assert len(data["insulator_mass"]) == size_section

    # Check first and last sections are not suspensions
    assert data["suspension"][0] is False
    assert data["suspension"][-1] is False

    # Check insulator lengths for first and last sections
    assert data["insulator_length"][0] == 0.0
    assert data["insulator_length"][-1] == 0.0
