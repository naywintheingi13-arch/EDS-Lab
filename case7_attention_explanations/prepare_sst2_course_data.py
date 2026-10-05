from pathlib import Path
import urllib.request
import zipfile
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = OUT / "sst2_train.csv"
VALID_CSV = OUT / "sst2_validation.csv"
URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

if TRAIN_CSV.exists() and VALID_CSV.exists():
    print("SST-2 course data already exists.")
else:
    archive = OUT / "SST-2.zip"
    extract_root = OUT / "_glue_extract"

    if not archive.exists():
        print("Downloading official GLUE SST-2 data...")
        urllib.request.urlretrieve(URL, archive)

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_root)

    source = extract_root / "SST-2"
    train = pd.read_csv(source / "train.tsv", sep="\t", keep_default_na=False)
    valid = pd.read_csv(source / "dev.tsv", sep="\t", keep_default_na=False)

    expected = {"sentence", "label"}
    if not expected.issubset(train.columns) or not expected.issubset(valid.columns):
        raise RuntimeError(
            f"Unexpected SST-2 format. Train columns={train.columns.tolist()}, "
            f"dev columns={valid.columns.tolist()}"
        )

    train[["sentence", "label"]].to_csv(TRAIN_CSV, index=False)
    valid[["sentence", "label"]].to_csv(VALID_CSV, index=False)

    print("Saved:", TRAIN_CSV, train.shape)
    print("Saved:", VALID_CSV, valid.shape)
    print("Train labels:", train["label"].value_counts().sort_index().to_dict())
    print("Validation labels:", valid["label"].value_counts().sort_index().to_dict())

print("Ready.")
