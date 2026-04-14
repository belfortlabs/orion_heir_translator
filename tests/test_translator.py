"""Tests for the end-to-end translation pipeline."""

from io import StringIO

from xdsl.printer import Printer
from xdsl.parser import Parser
from xdsl.context import Context
from xdsl.dialects import builtin as builtin_dialect
from xdsl.dialects import func as func_dialect
from xdsl.dialects.builtin import ModuleOp

from orion_heir.core.translator import GenericTranslator
from orion_heir.core.types import FHEOperation
from orion_heir.core.operation_registry import OperationRegistry
from orion_heir.dialects.ckks import CKKS
from orion_heir.dialects.lwe import LWE
from orion_heir.dialects.polynomial import Polynomial
from orion_heir.dialects.mod_arith import ModArith
from orion_heir.dialects.rns import RNS
from orion_heir.dialects.mgmt import MGMT
from orion_heir.dialects.orion import Orion


def _make_op(op_type, result_var=None, metadata=None):
    return FHEOperation(
        op_type=op_type,
        method_name=op_type,
        args=[],
        kwargs={},
        result_var=result_var,
        metadata=metadata or {},
    )


def _print_module(module: ModuleOp) -> str:
    stream = StringIO()
    Printer(stream=stream).print(module)
    return stream.getvalue()


def _parse_mlir(mlir_text: str) -> ModuleOp:
    ctx = Context()
    ctx.load_dialect(builtin_dialect.Builtin)
    ctx.load_dialect(func_dialect.Func)
    for d in [CKKS, LWE, Polynomial, ModArith, RNS, MGMT, Orion]:
        ctx.load_dialect(d)
    return Parser(ctx, mlir_text).parse_module()


# --- translator instantiation ---


def test_translator_creates():
    t = GenericTranslator()
    assert t is not None


def test_operation_registry_default_handlers():
    reg = OperationRegistry()
    expected = {"add", "sub", "mul", "rotate", "encode", "linear_transform"}
    assert expected.issubset(reg.handlers.keys())


# --- single-op translations ---


def test_translate_mul(scheme_params):
    """mul produces ckks.mul + ckks.relinearize (works on single input)."""
    module = GenericTranslator().translate(
        [_make_op("mul", "r")], scheme_params, "f"
    )
    mlir = _print_module(module)
    assert "ckks.mul" in mlir
    assert "ckks.relinearize" in mlir


def test_translate_rotate(scheme_params):
    module = GenericTranslator().translate(
        [_make_op("rotate", "r", {"rotation_offset": 5})],
        scheme_params,
        "f",
    )
    mlir = _print_module(module)
    assert "ckks.rotate" in mlir


# --- multi-op pipeline ---


def test_translate_mul_rotate(scheme_params):
    ops = [
        _make_op("mul", "m"),
        _make_op("rotate", "r", {"rotation_offset": 1}),
    ]
    module = GenericTranslator().translate(ops, scheme_params, "pipeline")
    mlir = _print_module(module)
    assert "func.func @pipeline" in mlir
    for keyword in ["ckks.mul", "ckks.rotate"]:
        assert keyword in mlir, f"{keyword} missing from output"


# --- MLIR round-trip ---


def test_mlir_roundtrip(scheme_params):
    """Generated MLIR can be parsed back by the xDSL parser."""
    ops = [
        _make_op("mul", "m"),
        _make_op("rotate", "r", {"rotation_offset": 2}),
    ]
    module = GenericTranslator().translate(ops, scheme_params, "rt")
    mlir = _print_module(module)
    parsed = _parse_mlir(mlir)
    assert isinstance(parsed, ModuleOp)
