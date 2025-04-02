# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from importlib import reload

import mechaphlowers


def test_log(caplog):
	caplog.set_level(logging.INFO)
	logging.info("Test log is working")
	assert "Test log is working" in caplog.text

	with caplog.at_level(logging.DEBUG):
		reload(mechaphlowers)  # noqa

		assert "Mechaphlowers package initialized." in caplog.text

		mechaphlowers.utils.add_stderr_logger(logging.DEBUG)
		assert (
			"Added a stderr logging handler to logger: mechaphlowers"
			in caplog.text
		)
