"""Build the Lambda code folder for Graviton (arm64) without Docker.

pip downloads the Linux aarch64 wheels for the Lambda Python runtime, whatever machine this runs on, and the
gaugeguard package is copied next to them. ``aws cloudformation package`` zips and uploads the folder.

    python infra/build_lambda.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "infra", "build")
PYTHON_VERSION = "3.13"            # must match Runtime in template.yaml
LIMIT = 250 * 1024 * 1024           # Lambda limit for unzipped code


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                    "--target", OUT, "--implementation", "cp", "--python-version", PYTHON_VERSION,
                    "--platform", "manylinux2014_aarch64", "--platform", "manylinux_2_28_aarch64",
                    "--only-binary=:all:", "-r", os.path.join(ROOT, "requirements.txt")], check=True)
    shutil.copytree(os.path.join(ROOT, "gaugeguard"), os.path.join(OUT, "gaugeguard"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for dirpath, dirnames, _ in os.walk(OUT):
        for d in [d for d in dirnames if d == "__pycache__"]:
            shutil.rmtree(os.path.join(dirpath, d))

    size = sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(OUT) for f in fs)
    print(f"built {OUT}: {size / 2**20:.0f} MiB unzipped (limit 250 MiB)")
    if not any(n.startswith("opencv_python_headless-5.") for n in os.listdir(OUT)):
        sys.exit("the build does not contain OpenCV 5")
    if size > LIMIT:
        sys.exit("too large for a Lambda zip")


if __name__ == "__main__":
    main()
