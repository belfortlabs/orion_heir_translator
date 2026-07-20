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
