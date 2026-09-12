# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""BAM Actuators: BAM's extended friction models as an Isaac Lab extension.

The public API is re-exported lazily so that the pure-torch pieces (e.g.
:class:`~bam_actuators.friction.BamFrictionModel`) can be imported and tested
without Isaac Sim / Isaac Lab being installed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    # Isaac Lab actuator (requires isaaclab)
    "BamActuator",
    "BamActuatorCfg",
    "BamFrictionCfg",
    # torch-only pieces (no isaaclab needed)
    "BamFrictionModel",
    "MotorBase",
    "available_motors",
    "get_motor",
    "resolve_params_file",
    "BUNDLED_PARAMS_DIR",
]

_LAZY_IMPORTS = {
    "BamActuator": "bam_actuators.actuators",
    "BamActuatorCfg": "bam_actuators.actuators",
    "BamFrictionCfg": "bam_actuators.actuators",
    "BamFrictionModel": "bam_actuators.friction",
    "MotorBase": "bam_actuators.motors.base",
    "available_motors": "bam_actuators.motors",
    "get_motor": "bam_actuators.motors",
    "resolve_params_file": "bam_actuators.params",
    "BUNDLED_PARAMS_DIR": "bam_actuators.params",
}


def __getattr__(name: str):
    """Resolve the public API on first access (PEP 562)."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
