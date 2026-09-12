# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""BAM's pendulum testbench, and the URDF that reproduces it in a simulator.

``bam.testbench.Pendulum`` is a point mass at the tip of a uniform rod::

    I_pivot        = mass * L^2 + (arm_mass / 3) * L^2
    tau_gravity(q) = (mass + arm_mass / 2) * g * L * sin(q)      [g = -9.80665]

Rather than tuning link geometry until a simulator happens to agree, :func:`build_urdf`
derives the URDF's ``<inertial>`` block so the body is *analytically* the same: a link of
mass ``mass + arm_mass`` whose COM sits at the distance that reproduces ``tau_gravity``
exactly, with a diagonal inertia chosen so the swing inertia about the pivot equals
``I_pivot`` exactly. The visual geometry is cosmetic and does not affect the dynamics.

Nothing here needs Isaac Sim or BAM, so the derivation can be checked directly - see
``tests/test_testbench.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: BAM's gravity, from ``bam.testbench.Pendulum.compute_bias``. Deliberately not 9.81:
#: the identified parameters were fitted against this value.
GRAVITY = -9.80665

#: Visual radii [m]. Cosmetic only. The tip stays small so it behaves like the point
#: mass BAM models it as.
TIP_RADIUS = 0.01
ROD_RADIUS = 0.005

#: One colour per model variant, so m1..m6 stay identifiable when they share a
#: viewport. Spread around the wheel and distinct in the mid-tones, so they survive a
#: range of lighting.
MODEL_COLOURS: dict[str, tuple[float, float, float]] = {
    "m1": (0.90, 0.20, 0.20),  # red
    "m2": (0.95, 0.55, 0.10),  # orange
    "m3": (0.85, 0.80, 0.15),  # yellow
    "m4": (0.25, 0.75, 0.30),  # green
    "m5": (0.20, 0.55, 0.95),  # blue
    "m6": (0.65, 0.35, 0.85),  # violet
}

#: Fallback for an arm that cannot be mapped to a variant, e.g. a custom params path.
UNKNOWN_COLOUR = (0.75, 0.75, 0.78)


@dataclass(frozen=True)
class Arm:
    """One pendulum in a scene: which model drives it, and how it should look.

    :param label: Short name, used for the link and joint names, the actuator group,
        and report/CSV columns. For a bundled model this is the variant, e.g. ``"m3"``.
    :param params_file: Identified parameters for this arm, or ``None`` for a
        params-free run. Only the caller driving the actuators uses this;
        :func:`build_urdf` ignores it.
    :param colour: Linear RGB, normally from :data:`MODEL_COLOURS`.
    """

    label: str
    params_file: str | None = None
    colour: tuple[float, float, float] = UNKNOWN_COLOUR

    @property
    def slug(self) -> str:
        """Name-safe form of :attr:`label`, for URDF ids and actuator group names."""
        return re.sub(r"[^0-9A-Za-z_]", "_", self.label)

    @property
    def joint_name(self) -> str:
        return f"joint_{self.slug}"

    @property
    def link_name(self) -> str:
        return f"arm_{self.slug}"


def arm_for(label: str, params_file: str | None = None) -> Arm:
    """An :class:`Arm` for a model variant, coloured from :data:`MODEL_COLOURS`."""
    return Arm(label=label, params_file=params_file, colour=MODEL_COLOURS.get(label, UNKNOWN_COLOUR))


def pendulum_spacing(length: float) -> float:
    """Distance between neighbouring pivots [m].

    Each arm swings a full circle about its pivot, so two of them can come within
    ``2 * length`` of each other. A little more than that leaves a visible gap at every
    angle.
    """
    return 2.4 * length


def build_urdf(mass: float, arm_mass: float, length: float, arms: list[Arm]) -> str:
    """A URDF whose rigid-body dynamics equal BAM's testbench.

    Derived quantities, with ``m = mass + arm_mass``:

    * ``d = (mass + arm_mass / 2) * L / m`` -- COM distance that reproduces the gravity
      torque ``(mass + arm_mass/2) * g * L * sin(q)``;
    * ``I_com = I_pivot - m * d^2`` -- parallel-axis shift so the swing inertia about the
      pivot is ``I_pivot = mass*L^2 + (arm_mass/3)*L^2``.

    The joints are continuous about +Y, so ``q = 0`` is an arm hanging straight down and
    positive ``q`` is counter-clockwise - BAM's convention, which makes
    ``tau_y = -(mass + arm_mass/2) * 9.80665 * L * sin(q)`` on both sides.

    Everything lives on one link per arm on purpose: a separate tip link joined by a
    fixed joint would be merged by the importer, perturbing the inertia derived above.
    There is no ``<collision>`` geometry either - nothing in this scene should contact
    anything, and a shape here would be a chance for the scene to stop matching BAM's
    contact-free single-axis model.

    With more than one arm they share the ``base`` link and are spaced along X, which is
    the direction they swing through, so their arcs never overlap. Each arm carries its
    own ``<material>``, so the colour travels with the asset.

    :param mass: Tip mass [kg].
    :param arm_mass: Arm mass [kg].
    :param length: Arm length [m].
    :param arms: The pendulums to emit, left to right along X.
    """
    total_mass = mass + arm_mass
    com_distance = (mass + arm_mass / 2.0) * length / total_mass
    inertia_pivot = mass * length**2 + (arm_mass / 3.0) * length**2
    inertia_com = inertia_pivot - total_mass * com_distance**2
    # Isotropic: only Iyy matters for this 1-DOF swing, and it is exact.
    i = inertia_com

    offset = (len(arms) - 1) / 2.0
    spacing = pendulum_spacing(length)

    parts = ['<?xml version="1.0"?>', '<robot name="bam_pendulum">']

    for arm in arms:
        r, g, b = arm.colour
        parts.append(
            f'  <material name="{arm.slug}_colour">'
            f'<color rgba="{r:.4g} {g:.4g} {b:.4g} 1.0"/></material>'
        )

    parts.append('  <link name="base"/>')

    for index, arm in enumerate(arms):
        x = (index - offset) * spacing
        parts.append(
            f'  <joint name="{arm.joint_name}" type="continuous">\n'
            f'    <parent link="base"/>\n'
            f'    <child link="{arm.link_name}"/>\n'
            f'    <origin xyz="{x:.12g} 0 0" rpy="0 0 0"/>\n'
            f'    <axis xyz="0 1 0"/>\n'
            f'    <dynamics damping="0.0" friction="0.0"/>\n'
            f'  </joint>'
        )

    for arm in arms:
        parts.append(
            f'  <link name="{arm.link_name}">\n'
            f'    <inertial>\n'
            f'      <origin xyz="0 0 {-com_distance:.12g}" rpy="0 0 0"/>\n'
            f'      <mass value="{total_mass:.12g}"/>\n'
            f'      <inertia ixx="{i:.12g}" ixy="0" ixz="0" iyy="{i:.12g}"'
            f' iyz="0" izz="{i:.12g}"/>\n'
            f'    </inertial>\n'
            f'    <visual>\n'
            f'      <origin xyz="0 0 {-length / 2.0:.12g}" rpy="0 0 0"/>\n'
            f'      <geometry>'
            f'<cylinder radius="{ROD_RADIUS}" length="{length:.12g}"/>'
            f'</geometry>\n'
            f'      <material name="{arm.slug}_colour"/>\n'
            f'    </visual>\n'
            f'    <visual>\n'
            f'      <origin xyz="0 0 {-length:.12g}" rpy="0 0 0"/>\n'
            f'      <geometry><sphere radius="{TIP_RADIUS}"/></geometry>\n'
            f'      <material name="{arm.slug}_colour"/>\n'
            f'    </visual>\n'
            f'  </link>'
        )

    parts.append('</robot>')
    return "\n".join(parts) + "\n"
