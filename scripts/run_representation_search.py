"""Execute the bounded representation autoresearch protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ald_sc.representation_search import run_search


def main() -> None:
    """Search development data, export the selected representation and audit it."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=root / "autoresearch/representation-v2-001"
    )
    parser.add_argument("--log", type=Path, default=root / "autoresearch.jsonl")
    args = parser.parse_args()
    report = run_search(args.out, args.log)
    chosen = report["selection"]
    print(f"Completed {report['trial_count']} trials")
    print(f"Selected basis: {chosen['basis']}; recipe: {chosen['recipe']}")
    print(f"Development balanced ratio: {chosen['score']:.6f}")
    print(f"Fresh audit balanced ratio: {report['audit']['balanced_ratio']:.6f}")
    lines = [
        "# Representation search results",
        "",
        f"Trials: {report['trial_count']}. Selected basis: `{chosen['basis']}`.",
        f"Selected recipe: `{chosen['recipe']}`.",
        "",
        f"Development balanced MSE ratio: {chosen['score']:.6f}.",
        f"Fresh audit balanced MSE ratio: {report['audit']['balanced_ratio']:.6f}.",
        f"Audit regime guard passed: {report['audit']['regime_guard_passed']}.",
        "",
        "| Regime | Heat MSE | Selected MSE | Paired change | 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for regime, row in report["audit"]["rows"].items():
        print(f"{regime}: {row['heat_mse']:.6f} -> {row['selected_mse']:.6f}")
        lines.append(
            f"| {regime} | {row['heat_mse']:.6f} | {row['selected_mse']:.6f} | "
            f"{row['delta_mse']:+.6f} | [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] |"
        )
    lines += [
        "",
        "The audit was generated only after selection was written and sealed.",
        "It did not change the selected recipe. Intervals are descriptive paired",
        "bootstrap intervals conditional on the fixed synthetic generator.",
        "",
        f"Projector distance from reference: {report['representation']['projector_distance_from_reference']:.6g}.",
        "A zero distance means the improvement changes decoder gains, not the encoded subspace.",
        "",
        "This is the best eligible development configuration among the tested",
        "candidates under a fixed rank and filter-noise budget. It does not establish",
        "a globally optimal representation, real-image quality, or prime-specific benefit.",
    ]
    (args.out / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
