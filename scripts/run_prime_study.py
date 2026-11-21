"""Run the fixed v2 study and generate auditable manuscript artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ald_sc.prime_gating import decoding_experiment, encoding_experiment  # noqa: E402

COLOURS = {
    "graph": "#0072B2",
    "pca": "#D55E00",
    "random": "#009E73",
    "heat": "#222222",
    "envelope": "#0072B2",
    "prime": "#D55E00",
}


def encoding_figure(result: dict) -> Figure:
    """Plot measured held-out energy retention by basis and coefficient count."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for ax, regime in zip(axes, ("smooth", "broadband"), strict=True):
        for basis, marker in zip(
            ("graph", "pca", "random"), ("o", "s", "^"), strict=True
        ):
            rows = [
                r
                for r in result["rows"]
                if r["regime"] == regime and r["basis"] == basis
            ]
            ax.plot(
                [r["rank"] for r in rows],
                [r["retained_energy"] for r in rows],
                marker=marker,
                color=COLOURS[basis],
                label=basis,
            )
        ax.set(
            title=regime.capitalize(),
            xlabel="Retained coefficients q",
            ylabel="Mean retained energy fraction",
            ylim=(0, 1.03),
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    return fig


def decoding_figure(result: dict) -> Figure:
    """Plot all measured paired MSE changes, with descriptive bootstrap intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, regime in zip(axes, ("smooth", "broadband"), strict=True):
        rows = [
            r
            for r in result["rows"]
            if r["regime"] == regime and r["arm"] not in ("projection", "noisy")
        ]
        for i, row in enumerate(rows):
            colour = COLOURS.get(row["arm"], "#777777")
            position = 3 * i if i < 3 else i + 7
            ax.plot(
                [position, position],
                [row["ci_low"], row["ci_high"]],
                color=colour,
                linewidth=1,
            )
            ax.plot(position, row["delta_mse"], "o", color=colour, markersize=4)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set(
            title=regime.capitalize(),
            ylabel="Paired MSE change from heat",
            xlabel="Fixed arm (all 32 phase controls shown)",
            xticks=[0, 3, 6, 25.5],
            xticklabels=["Heat", "Env.", "Prime", "Phase controls"],
        )
        for label in ax.get_xticklabels()[:3]:
            label.set_rotation(45)
            label.set_horizontalalignment("right")
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def notebook_source_hash(path: Path) -> str:
    """Hash notebook inputs independently of execution outputs and metadata."""
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    inputs = [{"cell_type": c["cell_type"], "source": c["source"]} for c in cells]
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def verify_exported_result(result: dict, kind: str, out: Path | None = None) -> None:
    """Check notebook outcomes, inputs and current sources against the exporter."""
    out = out or ROOT / "docs/paper/v2-results"
    report = json.loads((out / "results.json").read_text(encoding="utf-8"))
    for path, digest in report["provenance"]["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    for path, digest in report["provenance"]["notebook_input_sha256"].items():
        assert notebook_source_hash(ROOT / path) == digest, path
    raw_path = out / "raw-outcomes.npz"
    assert (
        hashlib.sha256(raw_path.read_bytes()).hexdigest()
        == report["provenance"]["raw_sha256"]
    )
    # These runs use the same interpreter, library versions and fixed streams.
    for key, expected in report[kind].items():
        assert result[key] == expected, f"{kind}.{key}"
    with np.load(raw_path, allow_pickle=False) as raw:
        for key, values in result["per_signal"].items():
            np.testing.assert_array_equal(values, raw[f"{kind}_{key}"])
        if kind == "decoding":
            for key, values in result["graph_error_per_signal"].items():
                np.testing.assert_array_equal(values, raw[f"graph_error_{key}"])
            np.testing.assert_array_equal(result["gains"], raw["gains"])
            for key, values in result["waveforms"].items():
                if key != "names":
                    np.testing.assert_array_equal(values, raw[key])


def save_results(encoding: dict, decoding: dict, out: Path) -> None:
    """Write raw outcomes, provenance, figures and LaTeX tables from measured data."""
    out.mkdir(parents=True, exist_ok=True)
    arrays = {f"encoding_{k}": v for k, v in encoding["per_signal"].items()}
    arrays.update({f"decoding_{k}": v for k, v in decoding["per_signal"].items()})
    arrays.update(
        {f"graph_error_{k}": v for k, v in decoding["graph_error_per_signal"].items()}
    )
    arrays.update({k: v for k, v in decoding["waveforms"].items() if k != "names"})
    arrays["gains"] = decoding["gains"]
    np.savez_compressed(out / "raw-outcomes.npz", **arrays)

    def summary(result: dict) -> dict:
        return {
            k: v
            for k, v in result.items()
            if k not in ("per_signal", "graph_error_per_signal", "waveforms", "gains")
        }

    sources = [
        "src/ald_sc/prime_gating.py",
        "src/ald_sc/modal_gate.py",
        "src/ald_sc/graph_decoder.py",
        "src/ald_sc/vae.py",
        "src/ald_sc/arrow_prior.py",
        "src/ald_sc/spectral_schedule.py",
        "src/ald_sc/__init__.py",
        "scripts/execute_v2_notebooks.py",
        "scripts/run_prime_study.py",
        "docs/experiments/prime-gating-protocol.md",
        "pyproject.toml",
        "uv.lock",
    ]
    report = {
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.system(),
            "machine": platform.machine(),
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_sha256": {
                p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources
            },
            "notebook_input_sha256": {
                str(path.relative_to(ROOT)): notebook_source_hash(path)
                for path in sorted((ROOT / "notebooks").glob("*-v2-*.ipynb"))
            },
            "raw_sha256": hashlib.sha256(
                (out / "raw-outcomes.npz").read_bytes()
            ).hexdigest(),
            "raw_array_schema": {
                key: {
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "sha256": hashlib.sha256(
                        np.ascontiguousarray(values, dtype="<f8").tobytes()
                    ).hexdigest(),
                }
                for key, values in arrays.items()
            },
            "scope": "Synthetic path-graph signals; no trained image-quality claim",
        },
        "encoding": summary(encoding),
        "decoding": summary(decoding),
        "arm_order": decoding["waveforms"]["names"] + ["projection", "noisy"],
        "encoding_basis_order": ["graph", "pca", "random"],
        "numerical_checks": {
            "multiplier_min": float(decoding["waveforms"]["multipliers"].min()),
            "multiplier_max": float(decoding["waveforms"]["multipliers"].max()),
            "maximum_multiplier_mean_error": float(
                np.max(np.abs(decoding["waveforms"]["multipliers"].mean(axis=1) - 1))
            ),
            "maximum_squared_gain_norm_error": float(
                np.max(
                    np.abs(
                        np.sum(decoding["gains"] ** 2, axis=1)
                        - np.sum(decoding["gains"][0] ** 2)
                    )
                )
            ),
            "maximum_exact_clock_error": max(
                row["exact_error"] for row in decoding["clock"]
            ),
        },
    }
    (out / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for name, figure in (
        ("encoding-quality", encoding_figure(encoding)),
        ("decoding-quality", decoding_figure(decoding)),
    ):
        figure.savefig(
            out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None}
        )
        figure.savefig(out / f"{name}.png", dpi=160)
        plt.close(figure)

    lines = [
        r"\begin{tabular}{llrrr}\toprule",
        r"Regime & Basis & $q$ & MSE & Retained energy \\\midrule",
    ]
    for row in encoding["rows"]:
        if row["rank"] in (16, 64):
            lines.append(
                f"{row['regime']} & {row['basis']} & {row['rank']} & "
                f"{row['mse']:.6f} & {row['retained_energy']:.4f}" + r" \\"
            )
    lines += [r"\bottomrule\end{tabular}"]
    (out / "encoding-table.tex").write_text("\n".join(lines) + "\n")
    lines = [
        r"\begin{tabular}{llrrl}\toprule",
        r"Regime & Arm & MSE & $\Delta$MSE & Paired 95\% interval \\\midrule",
    ]
    for row in decoding["rows"]:
        if not row["arm"].startswith("phase"):
            lines.append(
                f"{row['regime']} & {row['arm']} & {row['mse']:.6f} & "
                f"{row['delta_mse']:+.6f} & "
                f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]" + r" \\"
            )
    lines += [r"\bottomrule\end{tabular}"]
    (out / "decoding-table.tex").write_text("\n".join(lines) + "\n")
    exact = max(row["exact_error"] for row in decoding["clock"])
    lines = [
        r"\begin{tabular}{lrr}\toprule",
        r"Steps & Max exact error & Max Euler error \\\midrule",
    ]
    for steps in sorted({row["steps"] for row in decoding["clock"]}):
        rows = [row for row in decoding["clock"] if row["steps"] == steps]
        lines.append(
            f"{steps} & {max(r['exact_error'] for r in rows):.3e} & "
            f"{max(r['euler_error'] for r in rows):.3e}" + r" \\"
        )
    lines += [r"\bottomrule\end{tabular}"]
    (out / "clock-table.tex").write_text("\n".join(lines) + "\n")
    print(f"Saved {out}; maximum exact-clock error: {exact:.3e}")
    for regime, rank in decoding["phase_ranks"].items():
        print(
            f"{regime}: prime rank {rank['prime_rank_among_phase_controls']}/{rank['out_of']}"
        )


def main() -> None:
    """Execute the unchanged, fixed protocol with no outcome-based selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/paper/v2-results")
    args = parser.parse_args()
    save_results(encoding_experiment(), decoding_experiment(), args.out.resolve())


if __name__ == "__main__":
    main()
