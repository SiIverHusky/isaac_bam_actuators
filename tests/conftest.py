# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Make the test helpers in this directory importable as top-level modules.

``tests/`` is not a package, so ``offline_rollout`` would not resolve without this.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
