"""Run the three journal supplement checks against their released ZIP files."""

import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


SUPPLEMENTS = (
    ("Supplement_S1.zip", "beca98eb7741711a5ba6f99d4c4fba67ae2bc3123d3f11a892e91621976339fb", "verify_supplement.py"),
    ("Supplement_S2.zip", "f8acda37f3a17af45bbb1f2cf20e9ab31b57744a963791124002bca1c0424312", "verify_supplement.py"),
    ("Supplement_S3.zip", "0d68def998960697a83cf68fad2f7cd89c7d04618b47a229b3db9bee5bdb1005", "verify_summary.py"),
)


def main():
    root = Path(__file__).resolve().parent
    for filename, expected_digest, checker in SUPPLEMENTS:
        print(filename, flush=True)
        archive_path = root / filename
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != expected_digest:
            raise SystemExit(f"SHA256 mismatch for {filename}")

        with tempfile.TemporaryDirectory() as workdir:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(workdir)
            # Two published checkers use Python assertions. Ignore any inherited
            # PYTHONOPTIMIZE setting so the integrity checks always run.
            subprocess.run([sys.executable, "-E", "-B", checker], cwd=workdir, check=True)


if __name__ == "__main__":
    main()
