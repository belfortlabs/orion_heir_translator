"""Tests for the TypeBuilder component."""

from orion_heir.core.type_builder import TypeBuilder
from orion_heir.dialects.lwe import (
    NewLWECiphertextType,
    NewLWEPlaintextType,
)
from xdsl.dialects.builtin import TensorType, f64


def test_type_builder_creates_default_ciphertext(scheme_params):
    tb = TypeBuilder(scheme_params)
    ct = tb.get_default_ciphertext_type()
    assert isinstance(ct, NewLWECiphertextType)


def test_type_builder_creates_default_plaintext(scheme_params):
    tb = TypeBuilder(scheme_params)
    pt = tb.get_default_plaintext_type()
    assert isinstance(pt, NewLWEPlaintextType)


def test_type_builder_creates_plaintext_for_tensor(scheme_params):
    tb = TypeBuilder(scheme_params)
    tensor_type = TensorType(f64, [4096])
    pt = tb.create_plaintext_type_for_tensor(tensor_type)
    assert isinstance(pt, NewLWEPlaintextType)


def test_type_builder_scaling_factor_default(scheme_params):
    tb = TypeBuilder(scheme_params)
    ct = tb.get_default_ciphertext_type()
    scale = tb.get_scaling_factor(ct)
    assert scale == scheme_params.log_scale


def test_type_builder_rescaled_type(scheme_params):
    tb = TypeBuilder(scheme_params)
    ct = tb.get_default_ciphertext_type()
    original_scale = tb.get_scaling_factor(ct)
    rescaled = tb.create_rescaled_type(ct, original_scale // 2)
    new_scale = tb.get_scaling_factor(rescaled)
    assert new_scale == original_scale // 2


def test_type_builder_module_attributes(scheme_params):
    tb = TypeBuilder(scheme_params)
    attrs = tb.create_module_attributes()
    assert "ckks.schemeParam" in attrs
