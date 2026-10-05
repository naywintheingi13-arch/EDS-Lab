"""Prepare all local assets required by the Case 6 teaching notebook.

This is a release helper, not a new experiment. It executes the fixed pipeline already
implemented in download_data.py, audit.py, check_preprocessing.py, and plots.py.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import audit
import check_preprocessing
import download_data
import plots

ROOT = Path(__file__).resolve().parent
WEIGHT_NAME = "resnet18-f37072fd.pth"


def prepare(data_dir: Path, fresh: bool = False) -> None:
    data_dir = data_dir.resolve()

    if fresh:
        for name in ("cache", "outputs", "samples"):
            path = ROOT / name
            if path.exists():
                print(f"Removing generated {path}")
                shutil.rmtree(path)

    print("1/5 Downloading/verifying public data and pretrained weights")
    download_data.download(data_dir)

    print("2/5 Copying pretrained weights into the notebook quick-run location")
    weights_dir = ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    source_weight = data_dir / WEIGHT_NAME
    if not source_weight.exists():
        raise FileNotFoundError(source_weight)
    shutil.copy2(source_weight, weights_dir / WEIGHT_NAME)

    print("3/5 Running the fixed audit and generating cached features/results")
    audit.run(data_dir)

    print("4/5 Running the development-only preprocessing diagnostic")
    check_preprocessing.run(data_dir)

    print("5/5 Rendering standalone diagnostic figures")
    plots.save()

    print("\nCase 6 teaching assets are ready.")
    print("Open 06_pet_classifier_audit.ipynb and run from the top.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "pet_data")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="remove generated cache/outputs/samples before rebuilding",
    )
    args = parser.parse_args()
    prepare(args.data_dir, fresh=args.fresh)


if __name__ == "__main__":
    main()
