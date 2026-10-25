"""Convert the MedMNIST+ DermaMNIST (224x224) .npz archive into per-split .npy files.

The .npy files are memory-mapped by the data loader, so the full 1.5 GB image
tensor never has to sit in RAM at once.

Usage: python src/prepare_data.py [--npz data/dermamnist_224.npz]
"""
import argparse
import hashlib

import numpy as np

from common import DATA_DIR, SPLITS

URL = "https://zenodo.org/records/10519652/files/dermamnist_224.npz?download=1"
# Official checksum from medmnist/info.py ("MD5_224" for dermamnist).
EXPECTED_MD5 = "8974907d8e169bef5f5b96bc506ae45d"


def md5sum(path, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(DATA_DIR / "dermamnist_224.npz"))
    args = ap.parse_args()

    digest = md5sum(args.npz)
    print(f"MD5 of {args.npz}: {digest}")
    if digest != EXPECTED_MD5:
        raise SystemExit(f"checksum mismatch (expected {EXPECTED_MD5}); re-download from {URL}")
    archive = np.load(args.npz)
    for split in SPLITS:
        imgs = archive[f"{split}_images"]
        labels = archive[f"{split}_labels"].reshape(-1).astype(np.int64)
        np.save(DATA_DIR / f"derma224_{split}_images.npy", imgs)
        np.save(DATA_DIR / f"derma224_{split}_labels.npy", labels)
        print(f"{split:5s}: images {imgs.shape} {imgs.dtype}, labels {labels.shape}")
        del imgs


if __name__ == "__main__":
    main()
