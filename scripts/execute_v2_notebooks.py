"""Execute both full v2 notebooks with this project's Python kernel."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "notebooks/01-v2-encoding-quality.ipynb",
    ROOT / "notebooks/02-v2-prime-decoding.ipynb",
)


def main() -> None:
    """Run every cell, fail on errors and save successful outputs in place."""
    python = ROOT / ".venv/bin/python"
    if Path(sys.prefix).absolute() != ROOT / ".venv":
        raise RuntimeError("Run with uv run --locked --group notebooks python.")
    # Construct a minimal child environment; never copy the parent environment.
    kernel_env = {
        "PATH": f"{python.parent}:/usr/bin:/bin",
        "HOME": str(Path.home()),
        "PYTHONNOUSERSITE": "1",
        "MPLBACKEND": "Agg",
    }
    subprocess.run(
        [str(python), str(ROOT / "scripts/run_prime_study.py")],
        cwd=ROOT,
        env=kernel_env,
        check=True,
    )
    out = ROOT / "docs/paper/v2-results"
    report = json.loads((out / "results.json").read_text(encoding="utf-8"))
    executed = {}
    with tempfile.TemporaryDirectory(
        prefix="v2-kernel-", dir=ROOT / ".venv"
    ) as directory:
        spec_dir = Path(directory) / "ald-sc-project"
        spec_dir.mkdir()
        (spec_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        str(python),
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Python 3 (project .venv)",
                    "language": "python",
                    "env": {},
                }
            ),
            encoding="utf-8",
        )
        for path in NOTEBOOKS:
            notebook = nbformat.read(path, as_version=4)
            manager = AsyncKernelManager(
                kernel_name="ald-sc-project",
                transport="ipc",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]),
            )
            client = NotebookClient(
                notebook,
                km=manager,
                timeout=600,
                allow_errors=False,
                record_timing=False,
                ipython_hist_file=":memory:",
            )
            client.execute(cwd=str(ROOT), env=kernel_env, cleanup_kc=True)
            nbformat.validate(notebook)
            # Replace only after successful execution of the entire notebook.
            temporary = path.with_suffix(".ipynb.tmp")
            nbformat.write(notebook, temporary)
            os.replace(temporary, path)
            count = sum(cell.cell_type == "code" for cell in notebook.cells)
            executed[str(path.relative_to(ROOT))] = {
                "code_cells": count,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            print(f"Executed {path.relative_to(ROOT)}: {count} code cells")
    for path, digest in report["provenance"]["source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during notebook execution: {path}")
    receipt = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "kernel_python": ".venv/bin/python",
        "packages": {
            name: version(name)
            for name in ("nbclient", "nbformat", "ipykernel", "torch")
        },
        "notebooks": executed,
        "artefact_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(out.iterdir())
            if path.is_file() and path.name != "execution-receipt.json"
        },
        "status": "both full notebooks passed, including exporter parity and source hash checks",
    }
    (out / "execution-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
