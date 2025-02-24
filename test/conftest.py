# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import sys
from pathlib import Path

projet_dir: Path = Path(__file__).resolve().parents[1]
source_dir: Path = projet_dir / "src"
sys.path.insert(0, str(source_dir))
