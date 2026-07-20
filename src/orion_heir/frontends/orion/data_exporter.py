"""
Data exporter for Orion compiled models.

Exports model weights (diagonals), biases, and inputs as raw little-endian
float64 binary files for consumption by HEIR-generated Go code.
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from orion_heir.frontends.orion.scheme_params import OrionSchemeParameters


@dataclass
class ExportedFile:
    """Metadata about a single exported binary file."""

    name: str  # e.g., "fc1_weights"
    file: str  # relative path from output_dir, e.g., "data/fc1_weights.bin"
    shape: List[int]  # shape of the exported array
    role: str  # "weights", "bias", or "input"


@dataclass
class ExportManifest:
    """Complete manifest of exported data."""

    func_name: str
    slots: int
    args: List[ExportedFile]
    input: dict
    crypto_params: dict
    # Multi-input models (e.g. CriteoHELRM with `forward(dense, expanded_sparse)`)
    # carry one entry per forward placeholder; single-input models leave this
    # empty and rely on `input` (legacy back-compat).
    inputs: Optional[List[dict]] = None

    def write(self, path: Path):
        data = {
            "func_name": self.func_name,
            "slots": self.slots,
            "args": [asdict(f) for f in self.args],
            "input": self.input,
            "crypto_params": self.crypto_params,
        }
        if self.inputs:
            data["inputs"] = self.inputs
        path.write_text(json.dumps(data, indent=2) + "\n")


def _write_f64_bin(path: Path, data: np.ndarray):
    """Write a flat float64 array as raw little-endian binary."""
    data = np.ascontiguousarray(data, dtype=np.float64)
    path.write_bytes(data.tobytes())


def _pad_to(arr: np.ndarray, length: int) -> np.ndarray:
    """Pad a 1-D float64 array with zeros to the given length."""
    if len(arr) >= length:
        return arr[:length]
    padded = np.zeros(length, dtype=np.float64)
    padded[: len(arr)] = arr
    return padded


class OrionDataExporter:
    """Exports compiled Orion model data as binary files for HEIR Go harnesses."""

    def __init__(self, scheme_params: OrionSchemeParameters):
        self.scheme_params = scheme_params
        self.slots = scheme_params.slots

    def export_model_data(
        self,
        model: Any,
        input_tensor: Any,
        output_dir: Path,
        func_name: str = "mlp",
    ) -> ExportManifest:
        """
        Export all model data (weights, biases, input) as raw float64 binary files.

        Walks model layers in registration order (same as OrionFrontend.extract_operations),
        filtering to layers with precomputed diagonals (Linear, Conv2d).

        `input_tensor` is either a single torch.Tensor (1-input models) or a
        tuple/list of tensors (multi-input models like CriteoHELRM). For the
        multi-input case we write `data/input.bin` for the first input
        (back-compat with the single-example harness) AND a per-input
        `data/input_arg<k>.bin` file for k=0..N-1 that the multi-arg Go
        harness loads.

        Returns an ExportManifest describing all exported files.
        """
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Normalise to a list of tensors so the rest of the export is uniform.
        if isinstance(input_tensor, (tuple, list)):
            inputs: list = list(input_tensor)
        else:
            inputs = [input_tensor]

        # Export each input as a separate file. The first input is also
        # mirrored at the legacy `data/input.bin` path so existing
        # single-input harnesses keep working.
        input_files: list[dict] = []
        for k, inp in enumerate(inputs):
            flat = inp.detach().numpy().flatten().astype(np.float64)
            padded = _pad_to(flat, self.slots)
            if len(inputs) == 1:
                fname = "input.bin"
            else:
                fname = f"input_arg{k}.bin"
            _write_f64_bin(data_dir / fname, padded)
            input_files.append({"file": f"data/{fname}", "len": self.slots})
        if len(inputs) > 1:
            # Mirror the first input to the legacy path so callers that
            # haven't been updated for multi-input still find input.bin.
            _write_f64_bin(
                data_dir / "input.bin",
                _pad_to(
                    inputs[0].detach().numpy().flatten().astype(np.float64),
                    self.slots,
                ),
            )

        # Walk layers in the same order as extract_operations
        args: List[ExportedFile] = []
        for name, layer in model.named_modules():
            if not name:
                continue
            if not (hasattr(layer, "diagonals") and layer.diagonals):
                continue

            args.extend(self._export_layer(layer, name, data_dir))

        manifest = ExportManifest(
            func_name=func_name,
            slots=self.slots,
            args=args,
            input={"file": "data/input.bin", "len": self.slots},
            crypto_params=self._crypto_params_dict(),
            inputs=input_files if len(inputs) > 1 else None,
        )
        manifest.write(output_dir / "manifest.json")

        print(f"  Exported model data to {data_dir}/")
        return manifest

    def _export_layer(
        self, layer: Any, name: str, data_dir: Path
    ) -> List[ExportedFile]:
        """Export diagonals and bias for a single layer."""
        files = []

        # -- Diagonals (weights) — one file per block, in key order --
        for block_key in layer.diagonals.keys():
            block_diags = layer.diagonals[block_key]
            sorted_indices = sorted(block_diags.keys())
            flat_diags = np.zeros(len(sorted_indices) * self.slots, dtype=np.float64)
            for i, diag_idx in enumerate(sorted_indices):
                d = block_diags[diag_idx]
                if hasattr(d, "numpy"):
                    d = d.detach().cpu().numpy()
                d = np.asarray(d, dtype=np.float64)
                flat_diags[i * self.slots : (i + 1) * self.slots] = _pad_to(
                    d, self.slots
                )

            block_suffix = (
                f"_block_{block_key[0]}_{block_key[1]}"
                if len(layer.diagonals) > 1
                else ""
            )
            fname = f"{name}_weights{block_suffix}.bin"
            _write_f64_bin(data_dir / fname, flat_diags)
            files.append(
                ExportedFile(
                    name=f"{name}_weights{block_suffix}",
                    file=f"data/{fname}",
                    shape=[len(sorted_indices), self.slots],
                    role="weights",
                )
            )

        # -- Bias — use Orion's packing formulas to match the FHE slot layout --
        # Only export bias when the layer actually has one; the translator's
        # _get_linear_operations likewise only emits encode+add_plain for bias
        # when layer.bias is not None, so the counts must agree.
        if not (hasattr(layer, "bias") and layer.bias is not None):
            return files

        # The bias vector may span multiple ciphertext IDs (chunks of `slots` elements).
        # Export each chunk as a separate file so the translator can apply them correctly
        # to each row-block ciphertext in the pipelined BSGS structure.
        bias_full = self._pack_bias_full(layer)
        num_chunks = max(1, math.ceil(len(bias_full) / self.slots))

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.slots
            chunk = bias_full[start : start + self.slots]
            chunk = _pad_to(chunk, self.slots)

            chunk_suffix = f"_chunk{chunk_idx}" if num_chunks > 1 else ""
            fname = f"{name}_bias{chunk_suffix}.bin"
            _write_f64_bin(data_dir / fname, chunk)
            files.append(
                ExportedFile(
                    name=f"{name}_bias{chunk_suffix}",
                    file=f"data/{fname}",
                    shape=[self.slots],
                    role="bias",
                )
            )

        return files

    def _pack_bias_full(self, layer: Any) -> np.ndarray:
        """Return the full FHE-packed bias vector (may exceed slots for multi-ID layers)."""
        from orion.core import packing as orion_packing
        from orion.nn.linear import (
            Linear as OrionLinear,
            Conv1d as OrionConv1d,
            Conv2d as OrionConv2d,
        )

        if isinstance(layer, OrionLinear):
            bias_torch = orion_packing.construct_linear_bias(layer)
        elif isinstance(layer, OrionConv1d):
            bias_torch = orion_packing.construct_conv1d_bias(layer)
        elif isinstance(layer, OrionConv2d):
            bias_torch = orion_packing.construct_conv2d_bias(layer)
        else:
            if layer.bias is not None:
                return layer.bias.detach().cpu().numpy().astype(np.float64)
            return np.zeros(self.slots, dtype=np.float64)

        return bias_torch.detach().cpu().numpy().astype(np.float64)

    def _crypto_params_dict(self) -> dict:
        """Build the crypto_params section of the manifest."""
        return {
            "logN": self.scheme_params.logN,
            "Q": list(self.scheme_params.ciphertext_modulus_chain),
            "P": list(self.scheme_params.auxiliary_modulus_chain),
            "logScale": self.scheme_params.logScale,
            "input_level": int(self.scheme_params.input_level),
        }


def generate_go_wrapper(
    manifest: ExportManifest,
    output_dir: Path,
    has_bootstrapping: bool = False,
    num_bootstrap_evals: int = 0,
) -> None:
    """Generate model-specific Go wrapper ({func_name}_run.go) and main.go.

    All model-specific details (file paths, function name, argument arity,
    bootstrapping support) live only in the generated files so that main.go
    stays static across all models.

    `num_bootstrap_evals` is the count of distinct `*bootstrapping.Evaluator`
    parameters the heir-emitted function takes (one per sparse logSlots that
    HEIR's LWEToLattigo pass discovered). When 0, defaults to 1 if
    `has_bootstrapping` is true.
    """
    func_name = manifest.func_name
    go_func_name = func_name.lstrip("_")
    go_func_name = go_func_name[:1].upper() + go_func_name[1:]
    arg_files = [(arg.file, arg.name) for arg in manifest.args]
    # Multi-input models (e.g. CriteoHELRM) have manifest.inputs populated
    # with one entry per forward placeholder. Single-input models stick with
    # the legacy `manifest.input` (one file).
    input_files = (
        [entry["file"] for entry in manifest.inputs]
        if manifest.inputs
        else [manifest.input["file"]]
    )
    num_inputs = len(input_files)
    if has_bootstrapping and num_bootstrap_evals == 0:
        num_bootstrap_evals = 1
    if num_bootstrap_evals > 0:
        has_bootstrapping = True

    # ── Build {func_name}_run.go ─────────────────────────────────────
    imports = [
        '"encoding/binary"',
        '"fmt"',
        '"math"',
        '"os"',
        '"time"',
        "",
        '"github.com/tuneinsight/lattigo/v6/core/rlwe"',
        '"github.com/tuneinsight/lattigo/v6/schemes/ckks"',
    ]
    if has_bootstrapping:
        # Insert before the rlwe import
        imports.insert(
            6, '"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"'
        )

    imports_str = "\n".join(f"\t{imp}" for imp in imports)
    arg_files_str = "\n".join(f'\t"{file}",  // {name}' for file, name in arg_files)
    arg_call = ", ".join(f"args[{i}]" for i in range(len(arg_files)))

    if has_bootstrapping:
        bt_eval_names = [
            f"bootstrappingEval{i}" if num_bootstrap_evals > 1 else "bootstrappingEval"
            for i in range(num_bootstrap_evals)
        ]
        bt_eval_types = ", ".join("*bootstrapping.Evaluator" for _ in bt_eval_names)
        bt_eval_vars = ", ".join(bt_eval_names)
        bt_eval_sig = ", ".join(
            f"{name} *bootstrapping.Evaluator" for name in bt_eval_names
        )
        configure_ret = (
            f"({bt_eval_types}, *ckks.Evaluator, ckks.Parameters, "
            "*ckks.Encoder, *rlwe.Encryptor, *rlwe.Decryptor)"
        )
        configure_vars = (
            f"{bt_eval_vars}, evaluator, params, encoder, encryptor, decryptor"
        )
        run_sig = (
            f"{bt_eval_sig}, "
            "evaluator *ckks.Evaluator, params ckks.Parameters, "
            "encoder *ckks.Encoder,\n"
            "\tencryptor *rlwe.Encryptor, decryptor *rlwe.Decryptor"
        )
        call_prefix = f"{bt_eval_vars}, evaluator, params, encoder"
    else:
        configure_ret = (
            "(*ckks.Evaluator, ckks.Parameters, "
            "*ckks.Encoder, *rlwe.Encryptor, *rlwe.Decryptor)"
        )
        configure_vars = "evaluator, params, encoder, encryptor, decryptor"
        run_sig = (
            "evaluator *ckks.Evaluator, params ckks.Parameters, "
            "encoder *ckks.Encoder,\n"
            "\tencryptor *rlwe.Encryptor, decryptor *rlwe.Decryptor"
        )
        call_prefix = "evaluator, params, encoder"

    # Per-input encryption block. For each input k we load → encode →
    # encrypt → name it `ct{k}` (k = 0 .. num_inputs-1).
    encrypt_block_parts: list[str] = []
    for k in range(num_inputs):
        suffix = str(k) if num_inputs > 1 else ""
        encrypt_block_parts.append(
            f"\tinputVec{suffix} := loadF64(inputPath{suffix})\n"
            f"\tpt{suffix} := ckks.NewPlaintext(params, {manifest.crypto_params['input_level']})\n"
            f"\tpt{suffix}.Scale = params.DefaultScale()\n"
            f"\tif err := encoder.Encode(inputVec{suffix}, pt{suffix}); err != nil {{\n"
            f"\t\tpanic(err)\n"
            f"\t}}\n"
            f"\tct{suffix}, err{suffix} := encryptor.EncryptNew(pt{suffix})\n"
            f"\tif err{suffix} != nil {{\n"
            f"\t\tpanic(err{suffix})\n"
            f"\t}}\n"
        )
    encrypt_block = "\n".join(encrypt_block_parts)

    # Per-input ciphertext args (`ct, ct1, ct2, ...`).
    if num_inputs == 1:
        ct_args = "ct"
    else:
        ct_args = ", ".join(f"ct{k}" for k in range(num_inputs))

    # runOn signature: each input gets its own inputPath param.
    if num_inputs == 1:
        run_extra_sig = ", inputPath string"
        run_extra_call = "inputFile"
    else:
        run_extra_sig = ", " + ", ".join(
            f"inputPath{k} string" for k in range(num_inputs)
        )
        run_extra_call = ", ".join(f"inputFile{k}" for k in range(num_inputs))

    # Default-input-file constants. For multi-input we emit
    # `inputFile0`, `inputFile1`, … pointing at the per-arg bins.
    if num_inputs == 1:
        input_const_block = f'const inputFile = "{input_files[0]}"\n'
    else:
        lines = "\n".join(
            f'\tinputFile{k} = "{path}"' for k, path in enumerate(input_files)
        )
        input_const_block = f"const (\n{lines}\n)\n"

    run_go = (
        f"// {func_name}_run.go — AUTO-GENERATED by orion_heir. Do not edit by hand.\n"
        f"// Provides loadF64, configure(), and run() for the model-specific harness.\n"
        f"package main\n"
        f"\n"
        f"import (\n"
        f"{imports_str}\n"
        f")\n"
        f"\n"
        f"// Default input file(s) when `run` is called without explicit paths\n"
        f"// (single-example back-compat). The N-example harness in main.go\n"
        f"// passes data/input_<i>.bin (single-input) or data/input_<i>_arg<k>.bin\n"
        f"// (multi-input) for each example.\n"
        f"{input_const_block}"
        f"\n"
        f"var argFiles = []string{{\n"
        f"{arg_files_str}\n"
        f"}}\n"
        f"\n"
        f"func loadF64(path string) []float64 {{\n"
        f"\tdata, err := os.ReadFile(path)\n"
        f"\tif err != nil {{\n"
        f'\t\tpanic(fmt.Sprintf("failed to read %s: %v", path, err))\n'
        f"\t}}\n"
        f"\tn := len(data) / 8\n"
        f"\tresult := make([]float64, n)\n"
        f"\tfor i := range n {{\n"
        f"\t\tresult[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8 : (i+1)*8]))\n"
        f"\t}}\n"
        f"\treturn result\n"
        f"}}\n"
        f"\n"
        f"func configure() {configure_ret} {{\n"
        f"\treturn {go_func_name}__configure()\n"
        f"}}\n"
        f"\n"
        f"// runOn evaluates the model on specific input file(s). Used by the\n"
        f"// N-example harness to amortise keygen across multiple inputs. It\n"
        f"// returns the per-phase wall-clock split (encrypt / eval / decrypt in\n"
        f"// ms) alongside the result so the benchmark table can fill those\n"
        f"// columns; keygen is timed by the caller around configure().\n"
        f"func runOn({run_sig}{run_extra_sig}) (result []float64, encryptMs float64, inferenceMs float64, decryptMs float64) {{\n"
        f"\targs := make([][]float64, len(argFiles))\n"
        f"\tfor i, f := range argFiles {{\n"
        f"\t\targs[i] = loadF64(f)\n"
        f"\t}}\n"
        f"\n"
        f"\ttEnc := time.Now()\n"
        f"{encrypt_block}\n"
        f"\tencryptMs = float64(time.Since(tEnc)) / float64(time.Millisecond)\n"
        f"\n"
        f"\ttEval := time.Now()\n"
        f"\tresultCt := {go_func_name}({call_prefix}, {ct_args}, {arg_call})\n"
        f"\tinferenceMs = float64(time.Since(tEval)) / float64(time.Millisecond)\n"
        f"\n"
        f"\ttDec := time.Now()\n"
        f"\tresultPt := decryptor.DecryptNew(resultCt)\n"
        f"\tresult = make([]float64, params.MaxSlots())\n"
        f"\tif err := encoder.Decode(resultPt, result); err != nil {{\n"
        f"\t\tpanic(err)\n"
        f"\t}}\n"
        f"\tdecryptMs = float64(time.Since(tDec)) / float64(time.Millisecond)\n"
        f"\treturn\n"
        f"}}\n"
        f"\n"
        f"// Single-example back-compat shim.\n"
        f"func run({run_sig}) []float64 {{\n"
        f"\tresult, _, _, _ := runOn({configure_vars}, {run_extra_call})\n"
        f"\treturn result\n"
        f"}}\n"
    )

    out_path = output_dir / f"{func_name}_run.go"
    out_path.write_text(run_go)
    print(f"  Wrote {out_path}")

    # ── Build main.go ────────────────────────────────────────────────
    main_go = (
        "package main\n"
        "\n"
        "import (\n"
        '\t"encoding/json"\n'
        '\t"fmt"\n'
        '\t"time"\n'
        ")\n"
        "\n"
        "func main() {\n"
        "\ttKey := time.Now()\n"
        f"\t{configure_vars} := configure()\n"
        "\tkeygenMs := float64(time.Since(tKey)) / float64(time.Millisecond)\n"
        f"\tresult, encryptMs, inferenceMs, decryptMs := runOn({configure_vars}, {run_extra_call})\n"
        "\ttype Result struct {\n"
        '\t\tResult      []float64 `json:"result"`\n'
        '\t\tKeygenMs    float64   `json:"keygen_ms"`\n'
        '\t\tEncryptMs   float64   `json:"encrypt_ms"`\n'
        '\t\tInferenceMs float64   `json:"inference_ms"`\n'
        '\t\tDecryptMs   float64   `json:"decrypt_ms"`\n'
        "\t}\n"
        "\tenc, _ := json.Marshal(Result{Result: result, KeygenMs: keygenMs, EncryptMs: encryptMs, InferenceMs: inferenceMs, DecryptMs: decryptMs})\n"
        "\tfmt.Println(string(enc))\n"
        "}\n"
    )

    main_path = output_dir / "main.go"
    main_path.write_text(main_go)
    print(f"  Wrote {main_path}")
