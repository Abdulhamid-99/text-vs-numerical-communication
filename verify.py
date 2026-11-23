"""Run the paper's CPU checks from the files in this checkout."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CHECKS = (
    ("Journal supplements", "evidence/verify.py"),
    ("State-transfer source and frozen inputs", "state_transfer/state_study.py", "verify"),
    ("State-transfer responses and delivery policies", "state_replay/replay.py"),
    ("ArguAna ranking and byte-count records", "retrieval/replay/check.py"),
    ("ArguAna receiver-side ranking from saved inputs", "replay/verify.py"),
    ("Measured budgets, fitted codec and routing decisions", "retrieval/secondary/verify.py"),
)


def main():
    for label, script, *arguments in CHECKS:
        print(f"\n{label}", flush=True)
        subprocess.run(
            [sys.executable, "-E", "-B", str(ROOT / script), *arguments],
            cwd=ROOT,
            check=True,
        )
    print("\nAll offline checks completed.")


if __name__ == "__main__":
    main()
