"""Regenerate the eight Supplement S2 tables from a completed state-transfer run."""

import argparse
import json
import os
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

# Match the historical reconstruction's CPU thread count before importing NumPy.
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import scipy

import budget_tables
import fidelity
from evidence import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", required=True, type=Path, help="Protocol JSON with adjacent assets/"
    )
    parser.add_argument(
        "--analysis", required=True, type=Path, help="Sealed primary analysis directory"
    )
    parser.add_argument(
        "--run", required=True, type=Path, help="Sealed eight-stage run directory"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New directory for CSVs and receipt"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        dest = Path(scratch) / "tables"
        dest.mkdir()
        checks = budget_tables.compute(args.plan, args.analysis, dest)
        spans = fidelity.compute(args.plan, args.run, dest)
        receipt = {
            "status": "passed",
            "plan_sha256": sha(args.plan),
            "analysis_manifest_sha256": sha(args.analysis / "artifact_manifest.json"),
            "run_manifest_sha256": sha(args.run / "artifact_manifest.json"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "checks": checks,
            "reconstruction": spans,
            "tables": {p.name: sha(p) for p in sorted(dest.glob("*.csv"))},
        }
        (dest / "verification.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        dest.rename(args.output)
    print(
        f"Reproduced {len(receipt['tables'])} tables from {spans['spans']} held-out spans."
    )


if __name__ == "__main__":
    main()
