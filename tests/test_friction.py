# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the torch port of the BAM friction budget.

These tests only need ``torch`` (not Isaac Sim), so they can run in a plain
Python environment::

    pytest tests/test_friction.py
"""

import json
import math

import pytest
import torch

from bam_actuators.friction import BamFrictionModel, variant_flags


def test_m1_is_constant_coulomb_plus_viscous():
    """BAM m1: the budget does not depend on the state."""
    model = BamFrictionModel(friction_base=0.2, friction_viscous=0.3)

    motor = torch.zeros(2, 3)
    external = torch.full((2, 3), 5.0)
    dq = torch.tensor([[0.0, 1.0, -1.0], [10.0, -10.0, 0.5]])

    frictionloss, damping = model.compute(motor, external, dq)

    torch.testing.assert_close(frictionloss, torch.full((2, 3), 0.2))
    torch.testing.assert_close(damping, torch.full((2, 3), 0.3))


def test_m2_stribeck_decays_with_velocity():
    """BAM m2: extra friction at rest that vanishes as the joint moves."""
    model = BamFrictionModel(stribeck=True, friction_base=0.05, friction_stribeck=0.1, dtheta_stribeck=0.2)

    dq = torch.tensor([[0.0, 0.2, 100.0]])
    frictionloss, _ = model.compute(torch.zeros(1, 3), torch.zeros(1, 3), dq)

    # At rest the Stribeck term is at full strength.
    assert torch.isclose(frictionloss[0, 0], torch.tensor(0.15))
    # One Stribeck velocity away: exp(-1) * 0.1.
    expected = 0.05 + 0.1 * math.exp(-1.0)
    assert torch.isclose(frictionloss[0, 1], torch.tensor(expected), atol=1e-6)
    # Fast motion: the Stribeck term has decayed away.
    assert frictionloss[0, 2] < 0.051


def test_m5_directional_load_friction_uses_motor_and_external_torque():
    """BAM m5: the budget grows with the (directional) gearbox torque."""
    model = BamFrictionModel(
        load_dependent=True,
        directional=True,
        stribeck=True,
        friction_base=0.0,
        friction_stribeck=0.0,
        load_friction_motor=0.1,
        load_friction_external=0.2,
        load_friction_motor_stribeck=0.0,
        load_friction_external_stribeck=0.0,
        friction_viscous=0.0,
    )

    motor = torch.tensor([[1.0, 0.0]])
    external = torch.tensor([[0.0, 1.0]])
    dq = torch.full((1, 2), 1.0)  # fast -> Stribeck coefficient ~ 0

    frictionloss, damping = model.compute(motor, external, dq)

    # |external * 0.2 - motor * 0.1|
    torch.testing.assert_close(frictionloss, torch.tensor([[0.1, 0.2]]), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(damping, torch.zeros(1, 2))


def test_stiction_holds_a_joint_at_rest():
    """A net torque below the budget must not move the joint."""
    model = BamFrictionModel(friction_base=0.05, friction_viscous=0.0)

    net = model.apply(motor_torque=torch.tensor([0.01]), external_torque=torch.tensor([0.0]), dq=torch.tensor([0.0]))

    torch.testing.assert_close(net, torch.zeros(1))


def test_friction_never_reverses_the_motion():
    """Friction opposes motion without flipping the sign of the net torque."""
    model = BamFrictionModel(friction_base=0.05, friction_viscous=0.0)

    net = model.apply(motor_torque=torch.tensor([0.5]), external_torque=torch.tensor([0.0]), dq=torch.tensor([0.0]))

    # 0.5 - 0.05, not 0.5 + 0.05 and not -0.05.
    torch.testing.assert_close(net, torch.tensor([0.45]))


def test_stopping_torque_brakes_a_moving_joint():
    """The stopping-torque term is what lets friction hold a moving joint."""
    model = BamFrictionModel(friction_base=10.0, friction_viscous=0.0)

    net = model.apply(
        motor_torque=torch.tensor([0.0]),
        external_torque=torch.tensor([0.0]),
        dq=torch.tensor([1.0]),
        inertia=torch.tensor([1.0]),
        dt=0.1,
    )

    # Stopping torque needed is inertia/dt * dq = 10 Nm; the budget (10 Nm)
    # covers it exactly, so the joint is brought to rest in one step.
    torch.testing.assert_close(net, torch.tensor([-10.0]))


def test_stopping_torque_is_capped_by_the_budget():
    """A small budget can only remove part of the stopping torque."""
    model = BamFrictionModel(friction_base=2.0, friction_viscous=0.0)

    net = model.apply(
        motor_torque=torch.tensor([0.0]),
        external_torque=torch.tensor([0.0]),
        dq=torch.tensor([1.0]),
        inertia=torch.tensor([1.0]),
        dt=0.1,
    )

    torch.testing.assert_close(net, torch.tensor([-2.0]))


def test_load_params_ignores_unknown_keys():
    """Identified-model JSON files contain actuator keys we do not own."""
    model = BamFrictionModel()
    applied = model.load_params({"friction_base": 0.42, "kt": 0.09, "kp": 30.0, "R": 1.2})

    assert applied == ["friction_base"]
    assert model.friction_base == 0.42


def test_from_bam_model_reads_parameter_values():
    """The BAM ``Model`` objects can be adapted into the torch model."""

    class _Param:
        def __init__(self, value):
            self.value = value

    class _FakeBamModel:
        load_dependent = True
        directional = False
        stribeck = True
        quadratic = False

        friction_base = _Param(0.11)
        friction_stribeck = _Param(0.22)
        friction_viscous = _Param(0.33)
        dtheta_stribeck = _Param(0.44)
        alpha = _Param(2.0)
        load_friction_base = _Param(0.55)

    model = BamFrictionModel.from_bam_model(_FakeBamModel())

    assert model.stribeck and model.load_dependent and not model.directional
    assert model.friction_base == 0.11
    assert model.friction_viscous == 0.33
    assert model.load_friction_base == 0.55


# ----------------------------------------------------------------------
# Loading a BAM params file as-is: the file's "model" key picks the maths
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant, expected",
    [
        ("m1", {"load_dependent": False, "directional": False, "stribeck": False, "quadratic": False}),
        ("m2", {"load_dependent": False, "directional": False, "stribeck": True, "quadratic": False}),
        ("m3", {"load_dependent": True, "directional": False, "stribeck": False, "quadratic": False}),
        ("m4", {"load_dependent": True, "directional": False, "stribeck": True, "quadratic": False}),
        ("m5", {"load_dependent": True, "directional": True, "stribeck": True, "quadratic": False}),
        ("m6", {"load_dependent": True, "directional": True, "stribeck": True, "quadratic": True}),
    ],
)
def test_params_file_model_key_selects_the_variant(variant, expected):
    """An m5.json must bring the m5 maths into play, with no flags set by hand."""
    model = BamFrictionModel.from_params({"model": variant, "friction_base": 0.3})

    assert model.model_name == variant
    assert model.active_flags == [name for name in expected if expected[name]]
    for flag, value in expected.items():
        assert getattr(model, flag) is value
    # ...and the file's parameters still land.
    assert model.friction_base == 0.3


def test_variant_table_matches_bam():
    """m1..m6 must map onto the same terms BAM's ``models`` dict enables."""
    assert variant_flags("m1") == {}
    assert variant_flags("m2") == {"stribeck": True}
    assert variant_flags("m6") == {
        "load_dependent": True,
        "directional": True,
        "stribeck": True,
        "quadratic": True,
    }


def test_variant_name_is_case_insensitive():
    assert BamFrictionModel.from_params({"model": "M5"}).model_name == "m5"


def test_unknown_variant_raises_a_helpful_error():
    with pytest.raises(ValueError, match="Unknown BAM model variant"):
        BamFrictionModel.from_params({"model": "m7"})


def test_params_file_wins_over_an_explicit_model_argument():
    """The file describes itself; a caller's guess must not override it."""
    model = BamFrictionModel.from_params({"model": "m5"}, model="m1")

    assert model.model_name == "m5"
    assert model.directional


def test_explicit_flags_override_the_variant():
    """Flags stay available for ablating a single term of a variant."""
    model = BamFrictionModel.from_params({"model": "m6"}, quadratic=False)

    assert model.model_name == "m6"
    assert model.load_dependent and model.directional and model.stribeck
    assert not model.quadratic


def test_invalid_flag_combination_is_rejected():
    """Terms whose definitions depend on a disabled one cannot be combined."""
    # Directional load-friction is meaningless without load-dependence.
    with pytest.raises(ValueError, match="requires"):
        BamFrictionModel.from_params({"model": "m5"}, load_dependent=False)
    # The quadratic term is defined through the Stribeck coefficient.
    with pytest.raises(ValueError, match="requires"):
        BamFrictionModel.from_params({"model": "m6"}, stribeck=False)


def test_loading_a_params_file_changes_the_applied_torque(tmp_path):
    """The loaded variant must actually change the physics, not just the flags."""
    params = {
        "model": "m5",
        "kt": 0.5,
        "R": 2.0,
        "armature": 0.01,
        "friction_base": 0.0,
        "friction_stribeck": 0.0,
        "friction_viscous": 0.0,
        "load_friction_motor": 0.5,
        "load_friction_external": 0.5,
    }
    path = tmp_path / "m5.json"
    path.write_text(json.dumps(params))

    m5 = BamFrictionModel.from_json(str(path))
    m1 = BamFrictionModel.from_params({"model": "m1", **{k: v for k, v in params.items() if k != "model"}})

    motor = torch.tensor([1.0])
    external = torch.tensor([2.0])
    dq = torch.full((1,), 10.0)  # fast: Stribeck term has decayed

    fl_m5, _ = m5.compute(motor, external, dq)
    fl_m1, _ = m1.compute(motor, external, dq)

    # m5 sees the gearbox torque |2.0 * 0.5 - 1.0 * 0.5| = 0.5; m1 sees nothing.
    torch.testing.assert_close(fl_m5, torch.tensor([0.5]), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(fl_m1, torch.tensor([0.0]), rtol=0.0, atol=1e-6)


def test_from_json_ignores_the_non_parameter_keys(tmp_path):
    """A real BAM file also carries kt/R/armature/actuator; none are friction params."""
    params = {
        "kt": 1.66,
        "R": 3.15,
        "armature": 0.0109,
        "friction_base": 0.0615,
        "model": "m4",
        "actuator": "mx64",
    }
    path = tmp_path / "m4.json"
    path.write_text(json.dumps(params))

    model = BamFrictionModel.from_json(str(path))

    assert model.model_name == "m4"
    assert model.load_dependent and model.stribeck and not model.directional
    assert model.friction_base == 0.0615
    assert not hasattr(model, "kt")
