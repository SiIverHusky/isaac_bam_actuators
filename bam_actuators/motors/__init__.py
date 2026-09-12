# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Motor control laws for the BAM Actuators extension.

One module per servo family (``feetech_sts3215.py``, ``md01.py``, ...), each
contributing a :class:`~bam_actuators.motors.base.MotorBase` subclass. Motors are
discovered from this package automatically, so adding a servo is a matter of
dropping in ``<name>.py``.

Everything in here is **pure torch** - no Isaac Lab, no hardware - which keeps the
control laws testable against BAM's reference implementation without launching a
simulator.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

from .base import MotorBase

if TYPE_CHECKING:  # pragma: no cover
    pass


def _discover() -> dict[str, type[MotorBase]]:
    """Import every module in this package and collect the motors it defines."""
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_") or module_info.name == "base":
            continue
        importlib.import_module(f"{__name__}.{module_info.name}")

    found: dict[str, type[MotorBase]] = {}

    def walk(cls: type[MotorBase]) -> None:
        for subclass in cls.__subclasses__():
            if getattr(subclass, "name", ""):
                found[subclass.name] = subclass
            walk(subclass)

    walk(MotorBase)
    return dict(sorted(found.items()))


#: Registry of available motors, keyed by the name written in BAM params files
#: (the ``"actuator"`` key) - e.g. ``"sts3215"``.
MOTORS: dict[str, type[MotorBase]] = _discover()


def available_motors() -> list[str]:
    """Names of every registered motor, sorted."""
    return sorted(MOTORS)


def get_motor(name: str) -> type[MotorBase]:
    """Look up a motor class by name.

    :param name: Motor name, e.g. ``"sts3215"``.
    :returns: The :class:`MotorBase` subclass.
    :raises KeyError: If the name is not registered.
    """
    try:
        return MOTORS[name]
    except KeyError:
        raise KeyError(f"Unknown motor {name!r}. Available motors: {available_motors()}") from None


__all__ = ["MOTORS", "MotorBase", "available_motors", "get_motor"]
