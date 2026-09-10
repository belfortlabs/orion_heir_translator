from typing import Any


def effective_bias(layer: Any):
    """Return the bias used by Orion's compiled linear operation.

    Fusion updates on_bias even when the original PyTorch layer has bias=False.
    """
    bias = getattr(layer, "on_bias", None)
    return bias if bias is not None else getattr(layer, "bias", None)
