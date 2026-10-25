"""Shared paths, constants, dataset and data-loader helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"
FIG_DIR = RESULTS_DIR / "figures"

# DermaMNIST label order (MedMNIST v2 / HAM10000 diagnostic categories).
CLASS_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = [
    "Actinic keratoses / intraepithelial carcinoma",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]
NUM_CLASSES = len(CLASS_CODES)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = ("train", "val", "test")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (images[N,224,224,3] uint8 memory-mapped, labels[N] int64)."""
    images = np.load(DATA_DIR / f"derma224_{split}_images.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / f"derma224_{split}_labels.npy").astype(np.int64)
    return images, labels


class RandomRot90:
    """Rotate by a random multiple of 90 degrees (dermoscopy has no canonical orientation)."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return torch.rot90(img, k=random.randint(0, 3), dims=(-2, -1))


def build_transforms(train: bool, image_size: int = 224) -> v2.Compose:
    ops: list = []
    if train:
        ops += [
            v2.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            RandomRot90(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        ]
    elif image_size != 224:
        ops.append(v2.Resize(image_size, antialias=True))
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(ops)


class DermaDataset(Dataset):
    def __init__(self, split: str, train: bool, image_size: int = 224):
        self.images, self.labels = load_split(split)
        self.transform = build_transforms(train, image_size)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(np.array(self.images[idx])).permute(2, 0, 1)
        return self.transform(img), self.labels[idx]


def make_loader(split: str, train: bool, batch_size: int, workers: int = 6) -> DataLoader:
    ds = DermaDataset(split, train=train)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def class_weights(labels: np.ndarray, power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency weights raised to `power` (0.5 = sqrt), normalised to mean 1."""
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    w = (counts.sum() / (NUM_CLASSES * counts)) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
