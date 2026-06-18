"""
Generic translator infrastructure for converting FHE operations to HEIR MLIR.

This module provides the core translation framework that can be used by
different frontends (Orion, OpenFHE, SEAL, etc.).
"""

from typing import Dict, List, Any

from xdsl.ir import Region
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, ModuleOp, FunctionType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.context import Context

from orion_heir.core.types import FHEOperation, SchemeParameters, FrontendInterface
from orion_heir.core.operation_registry import OperationRegistry
from orion_heir.core.type_builder import TypeBuilder
from orion_heir.frontends.orion.orion_frontend import fix_encode_operations


class GenericTranslator:
    """
    Generic translator for converting FHE operations to HEIR MLIR.

    This class provides the core translation logic that can be used by
    any frontend. It uses an operation registry to handle different
    operation types in a modular way.
    """

    def __init__(self):
        self.context = Context()
        self.operation_registry = OperationRegistry()
        self._register_dialects()

    def _register_dialects(self):
        """Register all HEIR dialects with the context."""
        # Import dialects here to avoid circular imports
        from orion_heir.dialects.ckks import CKKS
        from orion_heir.dialects.orion import Orion
        from orion_heir.dialects.lwe import LWE
        from orion_heir.dialects.polynomial import Polynomial
        from orion_heir.dialects.mod_arith import ModArith
        from orion_heir.dialects.rns import RNS
        from orion_heir.dialects.mgmt import MGMT

        dialects = [CKKS, LWE, Polynomial, ModArith, RNS, MGMT, Orion]
        for dialect in dialects:
            self.context.load_dialect(dialect)

    def translate(
        self,
        operations: List[FHEOperation],
        scheme_params: SchemeParameters,
        function_name: str = "fhe_computation",
        num_inputs: int | None = None,
    ) -> ModuleOp:
        """
        Translate a list of FHE operations to HEIR MLIR.

        Args:
            operations: List of FHE operations to translate
            scheme_params: FHE scheme parameters
            function_name: Name for the generated function
            num_inputs: Number of ciphertext function arguments (= number of
                forward() placeholders in the source model). If omitted,
                infer it from `__set_current__` input sentinels.

        Returns:
            Complete MLIR module containing the translated operations
        """
        print(f"🔄 Translating {len(operations)} operations to HEIR...")

        # Build type system based on scheme parameters
        type_builder = TypeBuilder(scheme_params)

        # Create module with scheme parameters
        module = self._create_module(scheme_params, type_builder)

        # Create function containing the operations
        resolved_num_inputs = (
            num_inputs if num_inputs is not None else self._infer_num_inputs(operations)
        )
        func = self._create_function(
            operations, type_builder, function_name, resolved_num_inputs
        )
        func.update_function_type()
        module.body.block.add_op(func)
        fixes_applied = fix_encode_operations(module, type_builder)

        if fixes_applied:
            print("✅ Encode operations have been fixed for proper scaling factors")

        print("✅ Translation completed")
        return module

    def _infer_num_inputs(self, operations: List[FHEOperation]) -> int:
        """Infer function arity from `__set_current__` references."""
        max_input_index = 0
        for operation in operations:
            if operation.op_type != "__set_current__" or not operation.args:
                continue

            target = operation.args[0]
            if not isinstance(target, str):
                continue

            if not (target.startswith("@__input_") and target.endswith("__")):
                continue

            suffix = target[len("@__input_") : -2]
            if suffix.isdigit():
                max_input_index = max(max_input_index, int(suffix))

        return max_input_index + 1

    def _create_module(
        self, scheme_params: SchemeParameters, type_builder: TypeBuilder
    ) -> ModuleOp:
        """Create the top-level MLIR module with scheme attributes."""
        # Create module attributes based on scheme parameters
        attributes = type_builder.create_module_attributes()

        return ModuleOp([], attributes)

    def _create_function(
        self,
        operations: List[FHEOperation],
        type_builder: TypeBuilder,
        function_name: str,
        num_inputs: int = 1,
    ) -> FuncOp:
        """Create a function with simple sequential operation processing.

        `num_inputs` controls the function arity: each forward() placeholder
        becomes one ciphertext function argument. Single-input models
        register the legacy `__input__` sentinel; multi-input models also
        register `__input_0__`, `__input_1__`, … pointing at args[0], args[1].
        """

        # Setup function. The model input is encrypted at Orion's
        # `input_level`, not at MaxLevel — Orion's compiler picks this per
        # model so the first LT runs at its assigned orion_level. Use the
        # input-level type for the function arg; subsequent ops will type-
        # propagate from there.
        input_type = type_builder.get_input_ciphertext_type()
        func_type = FunctionType.from_lists(
            [input_type] * num_inputs, [input_type]
        )
        func = FuncOp(name=function_name, function_type=func_type, region=Region.DEFAULT)
        entry_block = func.body.blocks.first
        func.arg_attrs = ArrayAttr([DictionaryAttr({}) for _ in range(len(entry_block.args))])

        # Simple constants dictionary - just stores operation results by name
        constants = {}
        current_value = entry_block.args[0]  # First function input is the
        # initial current_value. The orion frontend emits `__set_current__`
        # at every layer that starts a parallel branch (single-input case)
        # OR consumes an input other than the very-first one (multi-input
        # case), so rewinding to a non-default input is always explicit.
        if num_inputs == 1:
            # Legacy single-input sentinel — keep working for ToyHELRM,
            # orionMLP, ResNet, …
            constants["__input__"] = current_value
        else:
            for i in range(num_inputs):
                constants[f"__input_{i}__"] = entry_block.args[i]

        # Process operations one by one
        for i, operation in enumerate(operations):
            print(f"  Processing operation {i+1}/{len(operations)}: {operation.op_type}")

            # The orion frontend emits __set_current__ markers in front of
            # layers that start a parallel branch (their fx-graph input is
            # not the previously emitted layer's output). The op carries a
            # single arg "@<name>" that we look up and use as the new
            # current_value. No MLIR op is produced.
            if operation.op_type == "__set_current__":
                target = None
                if operation.args:
                    a0 = operation.args[0]
                    if isinstance(a0, str) and a0.startswith("@"):
                        target = a0[1:]
                if target and target in constants:
                    current_value = constants[target]
                else:
                    print(
                        f"⚠️  __set_current__ target {target!r} not in constants; keeping current_value"
                    )
                continue

            # Get handler
            handler = self.operation_registry.handlers.get(operation.op_type)
            if not handler:
                print(f"⚠️ No handler for {operation.op_type}")
                continue

            # Process operation
            try:
                result = handler.handle(
                    operation, current_value, entry_block, constants, type_builder
                )

                # Store result by operation name
                if operation.result_var:
                    constants[operation.result_var] = result

                # Save the layer's "last output so far" under a stable name
                # so subsequent branches / merges can reference it via
                # `@<layer>_layer_output`. Multiple ops per layer overwrite;
                # the final write reflects the layer's exit value.
                # Also propagate the alias up the layer-name prefix chain
                # (e.g. "bot_l.1.bootstrapper" -> "bot_l.1") so auxiliary
                # management layers like ".bootstrapper" / ".mult1" hand
                # their post-processed output to the logical parent that the
                # fx graph identifies as the data-flow ancestor.
                layer_for_alias = (operation.metadata or {}).get("layer")
                if layer_for_alias and result is not None:
                    constants[f"{layer_for_alias}_layer_output"] = result
                    parts = layer_for_alias.split(".")
                    for stop in range(len(parts) - 1, 0, -1):
                        ancestor = ".".join(parts[:stop])
                        constants[f"{ancestor}_layer_output"] = result

                # Update current value only for non-encode operations
                if operation.op_type != "encode":
                    current_value = result

            except Exception as e:
                print(f"❌ Error processing {operation.op_type}: {e}")
                import traceback

                traceback.print_exc()

        # Finish function
        func.function_type = FunctionType.from_lists([input_type], [current_value.type])
        entry_block.add_op(ReturnOp(current_value))

        return func


class TranslatorBuilder:
    """Builder class for creating configured translators."""

    def __init__(self):
        self.translator = GenericTranslator()

    def with_custom_operations(self, operations: Dict[str, Any]) -> "TranslatorBuilder":
        """Add custom operation handlers."""
        for op_name, handler in operations.items():
            self.translator.operation_registry.register_operation(op_name, handler)
        return self

    def with_frontend(self, frontend: FrontendInterface) -> "TranslatorBuilder":
        """Configure with a specific frontend."""
        self.frontend = frontend
        return self

    def build(self) -> GenericTranslator:
        """Build the configured translator."""
        return self.translator


def create_translator() -> GenericTranslator:
    """Create a standard translator instance."""
    return GenericTranslator()
