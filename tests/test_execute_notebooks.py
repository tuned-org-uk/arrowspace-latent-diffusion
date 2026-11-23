"""Checks for notebook discovery and execution from notebook-relative paths."""

import hashlib
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "execute_notebooks", ROOT / "scripts/execute_notebooks.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_discovers_all_top_level_notebooks():
    expected = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert list(runner.NOTEBOOKS) == expected
    assert any(path.name == "09_generation_filmstrip.ipynb" for path in expected)


@pytest.mark.parametrize("name", ["04_dit_conditioning", "05_train_diffusion"])
def test_dit_notebook_runs_all_cells(name, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()
    (tmp_path / "results").mkdir()
    monkeypatch.chdir(notebook_dir)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(plt, "show", lambda: None)
    notebook = json.loads((ROOT / f"notebooks/{name}.ipynb").read_text())
    namespace = {}
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]), namespace)
        if name == "04_dit_conditioning":
            assert namespace["diffs"][0] > 1e-4
            assert namespace["diffs"][-1] == 0
        else:
            assert namespace["t"].shape == (namespace["z0"].shape[0],)
    finally:
        plt.close("all")


def test_executes_notebook_relative_paths_and_records_outputs(tmp_path, monkeypatch):
    (tmp_path / ".venv").mkdir()
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    path = notebooks / "01_example.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "from pathlib import Path\n"
                    "assert Path.cwd().name == 'notebooks'\n"
                    "Path('../results/example.txt').write_text('executed')\n"
                    "print('finished')"
                )
            ]
        ),
        path,
    )
    out = tmp_path / "docs/paper/v2-results"

    def export_study(*args, **kwargs):
        out.mkdir(parents=True)
        (out / "results.json").write_text(
            json.dumps({"provenance": {"source_sha256": {}}})
        )

    class InProcessNotebookClient:
        def __init__(self, notebook, **kwargs):
            # IPC endpoints must be absolute when kernel and runner cwd differ.
            assert Path(kwargs["km"].ip).is_absolute()
            self.notebook = notebook

        def execute(self, *, cwd, **kwargs):
            namespace = {}
            with monkeypatch.context() as context:
                context.chdir(cwd)
                for cell in self.notebook.cells:
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exec(cell.source, namespace)
                    cell.execution_count = 1
                    cell.outputs = [
                        nbformat.v4.new_output(
                            "stream", name="stdout", text=output.getvalue()
                        )
                    ]

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "NOTEBOOKS", [path])
    monkeypatch.setattr(runner.sys, "prefix", str(tmp_path / ".venv"))
    monkeypatch.setattr(runner.subprocess, "run", export_study)
    monkeypatch.setattr(runner, "NotebookClient", InProcessNotebookClient)

    runner.main()

    assert (tmp_path / "results/example.txt").read_text() == "executed"
    notebook = nbformat.read(path, as_version=4)
    assert notebook.cells[0].outputs[0].text == "finished\n"
    receipt = json.loads((out / "execution-receipt.json").read_text())
    assert receipt["notebooks"] == {
        "notebooks/01_example.ipynb": {
            "code_cells": 1,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    }
    assert receipt["status"].startswith("all notebooks passed")
