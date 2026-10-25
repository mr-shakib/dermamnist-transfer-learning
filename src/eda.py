"""Exploratory data analysis of DermaMNIST (224x224).

Writes results/eda.json and figures: class distribution, sample grid, per-class mean images,
colour statistics, and a near-duplicate (train <-> val/test) audit based on 64-bit difference hashes.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F

import plot_style
from common import CLASS_CODES, CLASS_NAMES, FIG_DIR, NUM_CLASSES, RESULTS_DIR, SPLITS, load_split, save_json

plot_style.apply()
import matplotlib.pyplot as plt  # noqa: E402


def channel_stats(images: np.ndarray, chunk: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-image mean RGB (N,3) plus dataset-level channel mean/std in [0,1] (float64 accumulation)."""
    per_img, s, s2, n = [], np.zeros(3), np.zeros(3), 0
    for i in range(0, len(images), chunk):
        x = images[i:i + chunk].astype(np.float64) / 255.0
        per_img.append(x.mean(axis=(1, 2)))
        s += x.sum(axis=(0, 1, 2))
        s2 += (x ** 2).sum(axis=(0, 1, 2))
        n += x.shape[0] * x.shape[1] * x.shape[2]
    mean = s / n
    return np.concatenate(per_img), mean, np.sqrt(s2 / n - mean ** 2)


def dhash(images: np.ndarray, chunk: int = 512) -> np.ndarray:
    """64-bit difference hash: grayscale -> 9x8 area resize -> horizontal gradient sign."""
    out = []
    for i in range(0, len(images), chunk):
        x = torch.from_numpy(images[i:i + chunk].astype(np.float32)).permute(0, 3, 1, 2)
        g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        small = F.adaptive_avg_pool2d(g, (8, 9)).squeeze(1)
        bits = (small[:, :, 1:] > small[:, :, :-1]).reshape(len(x), 64).numpy().astype(np.uint8)
        out.append(np.packbits(bits, axis=1).view(">u8").reshape(-1))
    return np.concatenate(out)


def min_hamming(query: np.ndarray, ref: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Hamming distance from each query hash to its nearest reference hash, and that reference's index."""
    ref_bytes = ref.view(np.uint8).reshape(len(ref), 8)
    dist = np.empty(len(query), dtype=np.int64)
    idx = np.empty(len(query), dtype=np.int64)
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk].view(np.uint8).reshape(-1, 1, 8)
        d = np.unpackbits(np.bitwise_xor(q, ref_bytes[None]), axis=2).sum(axis=2)
        idx[i:i + chunk] = d.argmin(axis=1)
        dist[i:i + chunk] = d.min(axis=1)
    return dist, idx


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {s: load_split(s) for s in SPLITS}
    counts = {s: np.bincount(data[s][1], minlength=NUM_CLASSES).tolist() for s in SPLITS}
    total = np.sum([counts[s] for s in SPLITS], axis=0)
    train_counts = np.array(counts["train"])

    # ---- class distribution
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    width = 0.27
    xs = np.arange(NUM_CLASSES)
    for j, s in enumerate(SPLITS):
        bars = ax.bar(xs + (j - 1) * width, counts[s], width - 0.03, label=f"{s} (n={sum(counts[s]):,})",
                      color=plot_style.SERIES[j])
        if s == "train":
            for b, c in zip(bars, counts[s]):
                ax.text(b.get_x() + b.get_width() / 2, c + 60, f"{c:,}", ha="center", va="bottom",
                        fontsize=7, color=plot_style.TEXT_2)
    ax.set_xticks(xs, CLASS_CODES)
    ax.set_ylabel("Images")
    ax.set_title("Class distribution per split (DermaMNIST)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    fig.savefig(FIG_DIR / "eda_class_distribution.png")
    plt.close(fig)

    # ---- sample grid (train): 6 random images per class
    rng = np.random.default_rng(7)
    imgs, labels = data["train"]
    fig, axes = plt.subplots(NUM_CLASSES, 6, figsize=(6.4, 7.8))
    for k in range(NUM_CLASSES):
        idx = rng.choice(np.where(labels == k)[0], 6, replace=False)
        for j, i in enumerate(idx):
            a = axes[k, j]
            a.imshow(imgs[i])
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            for sp in a.spines.values():
                sp.set_visible(False)
        axes[k, 0].set_ylabel(CLASS_CODES[k], rotation=0, ha="right", va="center", fontsize=9,
                              color=plot_style.TEXT)
    fig.suptitle("Random training images per class", fontsize=10, fontweight="bold", color=plot_style.TEXT)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, top=0.95)
    fig.savefig(FIG_DIR / "eda_samples.png")
    plt.close(fig)

    # ---- colour statistics
    per_img_rgb, mean, std = channel_stats(imgs)
    brightness = per_img_rgb @ np.array([0.299, 0.587, 0.114])
    fig, axes = plt.subplots(1, 8, figsize=(9.6, 1.7))
    grand = np.zeros((224, 224, 3), dtype=np.float64)
    for k in range(NUM_CLASSES):
        sel = np.where(labels == k)[0]
        acc = np.zeros((224, 224, 3), dtype=np.float64)
        for i in range(0, len(sel), 512):
            acc += imgs[sel[i:i + 512]].astype(np.float64).sum(0)
        grand += acc
        axes[k].imshow((acc / len(sel)).astype(np.uint8))
        axes[k].set_title(CLASS_CODES[k], fontsize=8)
    axes[7].imshow((grand / len(labels)).astype(np.uint8))
    axes[7].set_title("all classes", fontsize=8)
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.suptitle("Per-class mean image (training split)", fontsize=10, fontweight="bold", y=1.08,
                 color=plot_style.TEXT)
    fig.savefig(FIG_DIR / "eda_mean_images.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    groups = [brightness[labels == k] for k in range(NUM_CLASSES)]
    bp = axes[0].boxplot(groups, tick_labels=CLASS_CODES, widths=0.55, patch_artist=True, showfliers=False,
                         medianprops={"color": plot_style.TEXT, "linewidth": 1.2})
    for patch in bp["boxes"]:
        patch.set_facecolor(plot_style.SEQ_BLUE[1])
        patch.set_edgecolor(plot_style.SEQ_BLUE[4])
    for el in ("whiskers", "caps"):
        for line in bp[el]:
            line.set_color(plot_style.SEQ_BLUE[4])
    axes[0].set_ylabel("Mean luminance (0-1)")
    axes[0].set_title("Image brightness by class")
    axes[0].grid(axis="x", visible=False)
    rgb_by_class = np.array([per_img_rgb[labels == k].mean(0) for k in range(NUM_CLASSES)])
    for c, (name, col) in enumerate(zip(("R", "G", "B"), ("#e34948", "#008300", "#2a78d6"))):
        axes[1].plot(np.arange(NUM_CLASSES) + (c - 1) * 0.12, rgb_by_class[:, c], marker="o", markersize=6,
                     linestyle="none", color=col, label=name)
    axes[1].set_xticks(range(NUM_CLASSES), CLASS_CODES)
    axes[1].grid(axis="x", visible=False)
    axes[1].set_ylabel("Mean channel intensity (0-1)")
    axes[1].set_title("Mean RGB by class")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_colour_stats.png")
    plt.close(fig)

    # ---- exact and near duplicates across splits
    md5 = {s: [hashlib.md5(np.ascontiguousarray(data[s][0][i]).tobytes()).hexdigest() for i in range(len(data[s][1]))]
           for s in SPLITS}
    train_md5 = set(md5["train"])
    exact = {s: int(sum(h in train_md5 for h in md5[s])) for s in ("val", "test")}
    hashes = {s: dhash(data[s][0]) for s in SPLITS}
    near, train_labels = {}, data["train"][1]
    for s in ("val", "test"):
        d, j = min_hamming(hashes[s], hashes["train"])
        agree = data[s][1] == train_labels[j]
        near[s] = {f"<= {t} bits": {"pairs": int((d <= t).sum()),
                                    "label_agreement": float(agree[d <= t].mean()) if (d <= t).any() else None}
                   for t in (0, 2, 4)}
        near[s]["all (nearest train image)"] = {"pairs": int(len(d)), "label_agreement": float(agree.mean())}
        near[s]["median_min_distance"] = float(np.median(d))
        if s == "test":
            test_d, test_j = d, j

    # visual check: the 6 closest test/train pairs
    order = np.argsort(test_d, kind="stable")[:6]
    fig, axes = plt.subplots(2, 6, figsize=(8.4, 3.3))
    for c, i in enumerate(order):
        for r, (img, lab, tag) in enumerate(((data["test"][0][i], data["test"][1][i], "test"),
                                             (data["train"][0][test_j[i]], train_labels[test_j[i]], "train"))):
            a = axes[r, c]
            a.imshow(img)
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            for sp in a.spines.values():
                sp.set_visible(False)
            a.set_title(f"{tag}: {CLASS_CODES[lab]}" + (f"  (d={test_d[i]})" if r == 0 else ""), fontsize=7,
                        fontweight="normal")
    fig.suptitle("Closest test/train pairs by 64-bit dHash (d = Hamming distance)", fontsize=10,
                 fontweight="bold", color=plot_style.TEXT)
    fig.subplots_adjust(wspace=0.05, hspace=0.3)
    fig.savefig(FIG_DIR / "eda_near_duplicates.png")
    plt.close(fig)

    stats = {
        "image_shape": list(imgs.shape[1:]),
        "dtype": str(imgs.dtype),
        "counts": counts,
        "total_per_class": total.tolist(),
        "total_images": int(total.sum()),
        "class_share_train": (train_counts / train_counts.sum()).round(4).tolist(),
        "imbalance_ratio_max_over_min": float(train_counts.max() / train_counts.min()),
        "majority_class_baseline_acc_test": float(np.mean(data["test"][1] == np.argmax(train_counts))),
        "train_channel_mean": mean.round(4).tolist(),
        "train_channel_std": std.round(4).tolist(),
        "brightness_median_by_class": {CLASS_CODES[k]: float(np.median(groups[k])) for k in range(NUM_CLASSES)},
        "rgb_mean_by_class": {CLASS_CODES[k]: rgb_by_class[k].round(4).tolist() for k in range(NUM_CLASSES)},
        "exact_duplicates_of_train": exact,
        "dhash_near_duplicates_of_train": near,
        "class_names": dict(zip(CLASS_CODES, CLASS_NAMES)),
    }
    save_json(stats, RESULTS_DIR / "eda.json")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
