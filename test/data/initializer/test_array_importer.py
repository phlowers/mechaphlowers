# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from mechaphlowers.data.initializer.array_importer import ImporterRte



def test_import_data_from_proto():
	importer = ImporterRte("section_import_from_proto_utf8.csv")
	importer.section_array
	importer.cable_array
	importer.weather_array
