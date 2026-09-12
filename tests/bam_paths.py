# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Locating the BAM checkout that the parity tests compare against.

``bam`` cannot be pip-installed (its ``requires-python`` is ``>=3.12,<3.13``, and
the extension targets 3.11 for Isaac Lab), so the tests import it from a source
checkout instead. Point them at it with the ``BAM_ROOT`` environment variable::

    BAM_ROOT=/path/to/BAM python -m pytest tests/

The parity tests skip cleanly when the checkout is missing, so the rest of the
suite still runs.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variables checked, in order.
ENV_VARS = ("BAM_ROOT", "BAM_REPO")

#: Where the checkout lived on the development machine; only a convenience.
DEFAULT = "/home/hharis/Mangdang/BAM"


def bam_root() -> Path:
    """Path to the BAM source checkout, or the default if none is configured."""
    for name in ENV_VARS:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()
    return Path(DEFAULT)
