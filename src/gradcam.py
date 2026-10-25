"""Grad-CAM explanations and embedding visualisation for a trained checkpoint.

Writes
  results/figures/gradcam_<run>.png   one correctly classified test image per class + its Grad-CAM
  results/figures/tsne_<run>.png      t-SNE of penultimate-layer test embeddings, ImageNet vs fine-tuned

Usage: python src/gradcam.py --run resnet50_finetune
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

import plot_style
from common import CKPT_DIR, CLASS_CODES, FIG_DIR, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, RESULTS_DIR, \
    load_split, make_loader, save_json
from metrics import summary_metrics
from models import build_model, embed, gradcam_target_layer

plot_style.apply()
import matplotlib.pyplot as plt  # noqa: E402


MARKERS = ["o", "s", "^", "D", "v", "P", "X"]  # shape per class: identity is not carried by colour alone


def normalise(img_uint8: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(np.array(img_uint8)).permute(2, 0, 1).float().div(255)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0).to(device)


def grad_cam(model, layer, x: torch.Tensor, target: int) -> np.ndarray:
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits[0, target].backward()
    h1.remove()
    h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy()


@torch.no_grad()
def embeddings(model, arch, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        out.append(embed(model, arch, x.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")

    ckpt = torch.load(CKPT_DIR / f"{args.run}.pt", map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    model = build_model(arch, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    images, labels = load_split("test")
    probs = np.load(RESULTS_DIR / "runs" / args.run / "probs_test.npz")["probs"]
    preds = probs.argmax(1)
    rng = np.random.default_rng(3)

    # ---- Grad-CAM grid: per class one confidently correct and (if any) one misclassified example
    layer = gradcam_target_layer(model, arch)
    fig, axes = plt.subplots(4, NUM_CLASSES, figsize=(10.5, 6.6))
    for k in range(NUM_CLASSES):
        correct = np.where((labels == k) & (preds == k))[0]
        wrong = np.where((labels == k) & (preds != k))[0]
        picks = [("correct", correct[np.argsort(-probs[correct, k])[:5]]), ("missed", wrong)]
        for row, (kind, pool) in enumerate(picks):
            ax_img, ax_cam = axes[2 * row, k], axes[2 * row + 1, k]
            for a in (ax_img, ax_cam):
                a.set_xticks([])
                a.set_yticks([])
                a.grid(False)
                for sp in a.spines.values():
                    sp.set_visible(False)
            if len(pool) == 0:
                ax_img.text(0.5, 0.5, "none", ha="center", va="center", color=plot_style.TEXT_2,
                            transform=ax_img.transAxes)
                continue
            i = int(rng.choice(pool))
            x = normalise(images[i], device).requires_grad_(False)
            cam = grad_cam(model, layer, x, int(preds[i]))
            ax_img.imshow(images[i])
            ax_cam.imshow(images[i])
            ax_cam.imshow(cam, cmap="jet", alpha=0.45)
            ax_img.set_title(f"true {CLASS_CODES[k]}\npred {CLASS_CODES[preds[i]]} ({probs[i, preds[i]]:.2f})",
                             fontsize=7, fontweight="normal")
    axes[0, 0].set_ylabel("correct", fontsize=8)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=8)
    axes[2, 0].set_ylabel("misclassified", fontsize=8)
    axes[3, 0].set_ylabel("Grad-CAM", fontsize=8)
    fig.suptitle(f"Grad-CAM on test images ({args.run}); heat = evidence for predicted class",
                 fontsize=10, fontweight="bold", color=plot_style.TEXT)
    fig.subplots_adjust(wspace=0.05, hspace=0.35)
    fig.savefig(FIG_DIR / f"gradcam_{args.run}.png")
    plt.close(fig)

    # ---- t-SNE: ImageNet features vs fine-tuned features on the test split
    loader = make_loader("test", False, 64, 4)
    feats = {"fine-tuned": embeddings(model, arch, loader, device)}
    if arch != "simplecnn":
        base = build_model(arch, pretrained=True).to(device)
        feats = {"ImageNet (before fine-tuning)": embeddings(base, arch, loader, device)} | feats
    fig, axes = plt.subplots(1, len(feats), figsize=(4.4 * len(feats), 4.2))
    axes = np.atleast_1d(axes)
    sil = {}
    from sklearn.metrics import silhouette_score
    for a, (title, f) in zip(axes, feats.items()):
        z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(f)
        sil[title] = float(silhouette_score(f, labels, metric="cosine"))
        for k in range(NUM_CLASSES):
            m = labels == k
            a.scatter(z[m, 0], z[m, 1], s=9, color=plot_style.SERIES[k], label=CLASS_CODES[k], alpha=0.8,
                      marker=MARKERS[k], linewidths=0)
        a.set_title(f"{title}\nsilhouette (cosine) = {sil[title]:.3f}", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    axes[-1].legend(markerscale=2.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("t-SNE of penultimate-layer test embeddings", fontsize=10, fontweight="bold",
                 color=plot_style.TEXT)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"tsne_{args.run}.png")
    plt.close(fig)
    save_json({"silhouette_cosine": sil, "sanity_test_metrics": summary_metrics(probs, labels)},
              RESULTS_DIR / "runs" / args.run / "embedding_analysis.json")
    print(sil)


if __name__ == "__main__":
    main()
