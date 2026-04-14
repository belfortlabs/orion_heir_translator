"""Tests for HEIR dialect definitions."""

from orion_heir.dialects.ckks import (
    CKKS,
    AddOp,
    SubOp,
    MulOp,
    NegateOp,
    AddPlainOp,
    SubPlainOp,
    MulPlainOp,
    RelinearizeOp,
    RescaleOp,
    RotateOp,
    BootstrapOp,
    SchemeParamAttr,
)
from orion_heir.dialects.lwe import (
    LWE,
    NewLWECiphertextType,
    NewLWEPlaintextType,
    RLWEEncodeOp,
    InverseCanonicalEncodingAttr,
    PlaintextSpaceAttr,
    CiphertextSpaceAttr,
    ApplicationDataAttr,
    KeyAttr,
    ModulusChainAttr,
)
from orion_heir.dialects.polynomial import Polynomial, RingAttr, PolynomialAttr
from orion_heir.dialects.mod_arith import ModArith, ModArithType
from orion_heir.dialects.rns import RNS, RNSType
from orion_heir.dialects.mgmt import MGMT
from orion_heir.dialects.orion import ORION as Orion, LinearTransformOp, ChebyshevOp
from orion_heir.dialects.lwe_traits import (
    SameOperandsAndResultRings,
    AllCiphertextTypesMatch,
    IsCiphertextPlaintextOp,
)
from xdsl.context import Context


def test_all_dialects_load():
    """All custom dialects can be registered in an xDSL context."""
    ctx = Context()
    for dialect in [CKKS, LWE, Polynomial, ModArith, RNS, MGMT, Orion]:
        ctx.load_dialect(dialect)


def test_ckks_dialect_has_expected_ops():
    """CKKS dialect exposes the expected operation set."""
    op_names = {op.name for op in CKKS.operations}
    expected = {
        "ckks.add",
        "ckks.sub",
        "ckks.mul",
        "ckks.negate",
        "ckks.add_plain",
        "ckks.sub_plain",
        "ckks.mul_plain",
        "ckks.relinearize",
        "ckks.rescale",
        "ckks.rotate",
        "ckks.bootstrap",
    }
    assert expected.issubset(op_names), f"Missing ops: {expected - op_names}"


def test_lwe_dialect_has_expected_types():
    """LWE dialect registers the expected type attributes."""
    attr_names = {attr.name for attr in LWE.attributes}
    assert "lwe.new_lwe_ciphertext" in attr_names
    assert "lwe.new_lwe_plaintext" in attr_names
    assert "lwe.rlwe_encode" in {op.name for op in LWE.operations}


def test_orion_dialect_has_expected_ops():
    """Orion dialect exposes linear_transform and chebyshev ops."""
    op_names = {op.name for op in Orion.operations}
    assert "ckks.linear_transform" in op_names or "orion.linear_transform" in op_names
    assert "orion.chebyshev" in op_names
