"""Download one upstream input into a caller-managed staging path."""

from pathlib import Path
import shutil
import subprocess
import urllib.request


def download_file(url, path):
    temp_path = Path(path)
    curl = shutil.which("curl")
    if curl:
        subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--retry-all-errors",
                "--connect-timeout",
                "15",
                "--max-time",
                "120",
                "--output",
                str(temp_path),
                url,
            ],
            check=True,
        )
    else:
        with (
            urllib.request.urlopen(url, timeout=120) as source,
            temp_path.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
