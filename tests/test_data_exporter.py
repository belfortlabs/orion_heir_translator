from orion_heir.frontends.orion.data_exporter import ExportManifest, generate_go_wrapper


def test_go_wrapper_uses_exported_function_name(tmp_path):
    manifest = ExportManifest(
        func_name="_network",
        slots=8,
        args=[],
        input={"file": "data/input.bin"},
        crypto_params={"input_level": 0},
    )

    generate_go_wrapper(manifest, tmp_path)

    wrapper = (tmp_path / "_network_run.go").read_text()
    assert "return Network__configure()" in wrapper
    assert "resultCt := Network(" in wrapper


def test_fused_bias_export_matches_extracted_operations(tmp_path):
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import torch

    from orion_heir.frontends.orion.data_exporter import OrionDataExporter
    from orion_heir.frontends.orion.orion_frontend import OrionFrontend

    # BatchNorm fusion can introduce a bias into an originally bias-free layer.
    compiled_bias = torch.tensor([1.25, -2.5])
    layer = SimpleNamespace(
        bias=None,
        on_bias=compiled_bias,
        diagonals={(0, 0): {0: torch.ones(4)}},
    )
    frontend = OrionFrontend()
    for extract in (frontend._get_linear_operations, frontend._get_conv_operations):
        operations = extract(layer, "fused")
        bias_encode = next(op for op in operations if op.result_var == "fused_bias_encoded")
        assert bias_encode.args[0] is compiled_bias
        assert operations[-1].op_type == "add_plain"

    exporter = OrionDataExporter(SimpleNamespace(slots=4))
    files = exporter._export_layer(layer, "fused", tmp_path)
    bias_file = next(file for file in files if file.role == "bias")
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / Path(bias_file.file).name, dtype="<f8"),
        [1.25, -2.5, 0.0, 0.0],
    )


def test_wide_input_uses_distinct_ciphertext_blocks(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import importlib
    import numpy as np
    import torch
    from torch import fx, nn

    from orion_heir.core.translator import GenericTranslator
    from orion_heir.frontends.orion.data_exporter import OrionDataExporter
    from orion_heir.frontends.orion.orion_frontend import OrionFrontend

    model = nn.Module()
    model.wide = nn.Linear(6, 1, bias=False)
    model.wide.diagonals = {(0, 0): {0: torch.ones(4)}, (0, 1): {0: 2 * torch.ones(4)}}
    model.wide.level = 1
    graph = fx.Graph()
    dense = graph.placeholder("dense")
    sparse = graph.placeholder("sparse")
    dense.output_shape = (1, 2)
    sparse.output_shape = (1, 6)
    graph.output(graph.call_module("wide", (sparse,)))
    scheme = SimpleNamespace(
        params=SimpleNamespace(get_slots=lambda: 4), traced=SimpleNamespace(graph=graph)
    )
    monkeypatch.setattr(importlib.import_module("orion.core.orion"), "scheme", scheme)

    frontend = OrionFrontend()
    operations = frontend.extract_operations(model)
    assert frontend.num_inputs == 3
    params = SimpleNamespace(
        slots=4,
        ring_degree=8,
        logN=3,
        log_n=3,
        logScale=5,
        log_scale=5,
        ciphertext_modulus_chain=[97, 113],
        auxiliary_modulus_chain=[193],
        input_level=1,
    )
    module = GenericTranslator().translate(operations, params)
    func = module.body.block.first_op
    transforms = [op for op in func.body.block.ops if op.name == "orion.linear_transform"]
    assert len(transforms) == 2
    assert transforms[0].operands[0] is func.body.block.args[1]
    assert transforms[1].operands[0] is func.body.block.args[2]

    manifest = OrionDataExporter(params).export_model_data(
        model,
        (torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])),
        tmp_path,
    )
    assert len(manifest.inputs) == 3
    exported = [np.fromfile(tmp_path / item["file"], dtype="<f8") for item in manifest.inputs]
    np.testing.assert_array_equal(exported, [[1, 2, 0, 0], [3, 4, 5, 6], [7, 8, 0, 0]])
