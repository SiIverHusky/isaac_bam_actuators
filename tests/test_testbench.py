# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Tests for the URDF that reproduces BAM's testbench.

The point of ``bam_actuators.testbench`` is that the simulated body's rigid-body
dynamics are *analytically* equal to ``bam.testbench.Pendulum``, so these tests
recover the physics back out of the generated XML rather than trusting the code that
wrote it. The comparison against BAM itself needs a checkout (see ``bam_paths``) and
skips without one.
"""

import math
import xml.etree.ElementTree as ET

import pytest

from bam_actuators.testbench import (
    GRAVITY,
    MODEL_COLOURS,
    UNKNOWN_COLOUR,
    Arm,
    arm_for,
    build_urdf,
    pendulum_spacing,
)

from bam_paths import bam_root

MASS, ARM_MASS, LENGTH = 0.5, 0.02, 0.15


# ----------------------------------------------------------------------
# Reading the physics back out of the URDF
# ----------------------------------------------------------------------


def link_physics(link: ET.Element) -> tuple[float, float, float]:
    """``(mass, com_distance, inertia_about_com)`` of a ``<link>``.

    The arm hangs down, so its COM rides at negative z and the distance from the
    pivot is the magnitude.
    """
    inertial = link.find("inertial")
    assert inertial is not None, f"link {link.get('name')} has no <inertial>"
    mass = float(inertial.find("mass").get("value"))
    com_distance = abs(float(inertial.find("origin").get("xyz").split()[2]))
    inertia = float(inertial.find("inertia").get("iyy"))
    return mass, com_distance, inertia


def test_urdf_is_well_formed_and_has_one_arm_per_pendulum():
    arms = [arm_for("m1"), arm_for("m3"), arm_for("m6")]
    robot = ET.fromstring(build_urdf(MASS, ARM_MASS, LENGTH, arms))

    assert robot.tag == "robot"
    assert [link.get("name") for link in robot.findall("link")] == [
        "base",
        "arm_m1",
        "arm_m3",
        "arm_m6",
    ]
    assert [joint.get("name") for joint in robot.findall("joint")] == [
        "joint_m1",
        "joint_m3",
        "joint_m6",
    ]


def test_every_arm_reproduces_bams_gravity_torque():
    """``mass * g * d * sin(q)`` must equal BAM's ``(mass + arm_mass/2) * g * L * sin(q)``.

    The COM distance is the whole reason the URDF is derived rather than tuned, so
    this is the property that matters most.
    """
    robot = ET.fromstring(build_urdf(MASS, ARM_MASS, LENGTH, [arm_for("m1")]))
    mass, com_distance, _ = link_physics(robot.findall("link")[1])

    assert mass == pytest.approx(MASS + ARM_MASS)
    assert mass * com_distance == pytest.approx((MASS + ARM_MASS / 2) * LENGTH)


def test_every_arm_reproduces_bams_swing_inertia():
    """``I_com`` must shift, by the parallel-axis theorem, onto BAM's ``I_pivot``."""
    robot = ET.fromstring(build_urdf(MASS, ARM_MASS, LENGTH, [arm_for("m1")]))
    mass, com_distance, inertia_com = link_physics(robot.findall("link")[1])

    expected = MASS * LENGTH**2 + (ARM_MASS / 3.0) * LENGTH**2
    assert inertia_com + mass * com_distance**2 == pytest.approx(expected)


def test_the_link_inertia_is_diagonal_and_isotropic():
    """Only Iyy drives this 1-DOF swing, and making all three equal keeps the rest of
    the tensor from being arbitrary."""
    robot = ET.fromstring(build_urdf(MASS, ARM_MASS, LENGTH, [arm_for("m1")]))
    inertia = robot.findall("link")[1].find("inertial").find("inertia")

    for off_diagonal in ("ixy", "ixz", "iyz"):
        assert float(inertia.get(off_diagonal)) == 0.0
    assert inertia.get("ixx") == inertia.get("iyy") == inertia.get("izz")


def test_arms_are_spaced_so_their_arcs_cannot_overlap():
    """Each arm sweeps a full circle about its own pivot; the pitch has to clear it."""
    arms = [arm_for("m1"), arm_for("m2")]
    robot = ET.fromstring(build_urdf(MASS, ARM_MASS, LENGTH, arms))

    xs = [float(joint.find("origin").get("xyz").split()[0]) for joint in robot.findall("joint")]
    pitch = abs(xs[1] - xs[0])

    assert pitch == pytest.approx(pendulum_spacing(LENGTH))
    assert pitch > 2 * LENGTH, "neighbouring arms would collide at some angle"
    # Centred on the origin, so the camera framing is symmetric.
    assert sum(xs) == pytest.approx(0.0)


def test_a_lone_arm_sits_on_the_origin():
    """The single-pendulum case must not be shifted off to one side."""
    robot = ET.fromstring(build_urdf(MASS, ARM_MASS, LENGTH, [arm_for("m5")]))

    assert float(robot.find("joint").find("origin").get("xyz").split()[0]) == 0.0


# ----------------------------------------------------------------------
# Colours
# ----------------------------------------------------------------------


def test_every_arm_carries_its_own_material():
    arms = [arm_for("m1"), arm_for("m5")]
    robot = ET.fromstring(build_urdf(MASS, ARM_MASS, LENGTH, arms))

    materials = {
        material.get("name"): material.find("color").get("rgba") for material in robot.findall("material")
    }
    assert set(materials) == {"m1_colour", "m5_colour"}

    # Both visuals of an arm reference its own material, or half the rod would be
    # the wrong colour.
    for arm in arms:
        link = robot.find(f"./link[@name='{arm.link_name}']")
        assert [visual.find("material").get("name") for visual in link.findall("visual")] == [
            f"{arm.slug}_colour",
            f"{arm.slug}_colour",
        ]


def test_model_variants_have_distinct_colours():
    assert set(MODEL_COLOURS) == {f"m{i}" for i in range(1, 7)}
    assert len(set(MODEL_COLOURS.values())) == 6, "colours must be distinguishable"


def test_an_unrecognised_model_falls_back_to_grey():
    """A custom params path cannot be mapped to a variant, so it gets the fallback."""
    assert arm_for("my_custom_model").colour == UNKNOWN_COLOUR
    assert arm_for("m3").colour == MODEL_COLOURS["m3"]


def test_labels_are_sanitised_into_valid_urdf_ids():
    """A label from a filename can contain anything; URDF ids and actuator group
    names cannot."""
    arm = Arm(label="my model/v2.json")

    assert arm.slug == "my_model_v2_json"
    assert arm.joint_name == "joint_my_model_v2_json"
    assert arm.link_name == "arm_my_model_v2_json"


# ----------------------------------------------------------------------
# Against BAM itself
# ----------------------------------------------------------------------


def test_matches_bams_testbench_dynamics_exactly():
    """The derived body must reproduce ``bam.testbench.Pendulum`` at every angle.

    This is the end-to-end check: the URDF is the only thing the simulator sees, so if
    these agree, the rig is BAM's rig.
    """
    root = bam_root()
    if not (root / "bam" / "testbench.py").is_file():
        pytest.skip("no BAM checkout; set BAM_ROOT")

    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from bam.testbench import Pendulum as BamPendulum

    reference = BamPendulum({"mass": MASS, "arm_mass": ARM_MASS, "length": LENGTH})
    robot = ET.fromstring(build_urdf(MASS, ARM_MASS, LENGTH, [arm_for("m1")]))
    mass, com_distance, inertia_com = link_physics(robot.findall("link")[1])

    # The simulated body, as PhysX will see it: a point mass at the COM plus its
    # inertia about that COM.
    simulated_bias = lambda q: mass * GRAVITY * com_distance * math.sin(q)  # noqa: E731
    simulated_mass = inertia_com + mass * com_distance**2

    for q in (0.0, 0.25, -0.9, 1.4, math.pi / 2, -math.pi):
        assert simulated_bias(q) == pytest.approx(reference.compute_bias(q, 0.0)), f"bias at q={q}"
    assert simulated_mass == pytest.approx(reference.compute_mass(0.0, 0.0))

    # And the gravity constant is BAM's, not 9.81: the parameters were fitted to it.
    assert GRAVITY == -9.80665
