# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Locating the identified parameter files bundled with the extension.

Identified models are copied into ``bam_actuators/params/<motor>/<model>.json`` so
they travel with the extension. A config can then point at one by name::

    BamActuatorCfg(params_file="sts3215/m5")

which resolves to the bundled file for that motor and BAM model variant.
"""

from __future__ import annotations

from pathlib import Path

#: Directory holding the bundled identified models.
BUNDLED_PARAMS_DIR = Path(__file__).parent / "params"


def bundled_params_file(motor: str, model: str) -> Path:
    """Path to a bundled params file.

    :param motor: Motor name, e.g. ``"sts3215"``.
    :param model: BAM model variant, e.g. ``"m5"``.
    :returns: The path, whether or not it exists (callers get a clear error later).
    """
    return BUNDLED_PARAMS_DIR / motor / f"{model}.json"


def available_bundled() -> dict[str, list[str]]:
    """Bundled models, as ``{motor: [model, ...]}``."""
    if not BUNDLED_PARAMS_DIR.is_dir():
        return {}
    return {
        motor_dir.name: sorted(path.stem for path in motor_dir.glob("*.json"))
        for motor_dir in sorted(BUNDLED_PARAMS_DIR.iterdir())
        if motor_dir.is_dir()
    }


def resolve_params_file(params_file: str | Path) -> str:
    """Resolve a params file reference to a real path.

    Accepts either an explicit path (used as-is) or a ``"<motor>/<model>"``
    shorthand resolved against :data:`BUNDLED_PARAMS_DIR`.

    :param params_file: Path, or ``"<motor>/<model>"`` shorthand.
    :returns: A filesystem path.
    :raises FileNotFoundError: If the reference cannot be resolved.
    """
    candidate = Path(params_file).expanduser()

    # An explicit file always wins.
    if candidate.suffix == ".json" or candidate.exists():
        if not candidate.is_file():
            raise FileNotFoundError(f"Params file not found: {candidate}")
        return str(candidate)

    # Otherwise treat it as "<motor>/<model>".
    parts = candidate.parts
    if len(parts) == 2:
        bundled = bundled_params_file(*parts)
        if bundled.is_file():
            return str(bundled)
        raise FileNotFoundError(
            f"No bundled params for {params_file!r} (looked in {bundled}). "
            f"Bundled models: {available_bundled()}"
        )

    raise FileNotFoundError(
        f"Could not resolve params_file={str(params_file)!r}. Pass a path to a .json "
        f"file, or '<motor>/<model>' - bundled models: {available_bundled()}"
    )
