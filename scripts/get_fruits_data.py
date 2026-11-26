"""Download the 8-class subset of Fruits-360 (sparse git checkout, ~25 MB).

usage:  python scripts/get_fruits_data.py
needs:  git >= 2.27
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dl.data_pipeline import CLASSES  # noqa: E402

REPO = "https://github.com/Horea94/Fruit-Images-Dataset"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, "data", "_fruits_git")
DST = os.path.join(ROOT, "data", "fruits")


def run(*cmd, cwd=None):
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


if __name__ == "__main__":
    if not os.path.isdir(TMP):
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, TMP)
    patterns = "\n".join(f"/{s}/{c}/" for c in CLASSES for s in ("Training", "Test"))
    subprocess.run(["git", "sparse-checkout", "set", "--no-cone", "--stdin"], cwd=TMP,
                   input=patterns.encode(), check=True)
    for split in ("Training", "Test"):
        for c in CLASSES:
            shutil.copytree(os.path.join(TMP, split, c), os.path.join(DST, split, c), dirs_exist_ok=True)
    print("done ->", DST)
