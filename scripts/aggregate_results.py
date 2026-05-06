"""Walk a results directory of per-episode JSONL files and produce a single summary CSV
with one row per (model, task, channel, budget, network) cell.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(results_root: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main(Path(sys.argv[1]))
