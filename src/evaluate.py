"""Evaluate saved checkpoints on the validation and test splits.

For every run: plain predictions and 8-view dihedral test-time augmentation (4 rotations x flip),
summary metrics with bootstrap 95% CIs, per-class report, confusion matrix and saved probabilities.

Usage: python src/evaluate.py --runs scratch_cnn resnet50_frozen ...
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from common import CKPT_DIR, RESULTS_DIR, make_loader, save_json
from metrics import bootstrap_ci, confusion, per_class_report, summary_metrics
from models import build_model


def dihedral_views(x: torch.Tensor):
    for flip in (False, True):
        xf = torch.flip(x, dims=(-1,)) if flip else x
        for k in range(4):
            yield torch.rot90(xf, k, dims=(-2, -1))


@torch.no_grad()
def predict(model, loader, device, tta: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        views = dihedral_views(x) if tta else [x]
        p = None
        for v in views:
            out = torch.softmax(model(v).float(), 1)
            p = out if p is None else p + out
        p = p / (8 if tta else 1)
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    device = torch.device("cuda")

    loaders = {s: make_loader(s, False, 64, args.workers) for s in ("val", "test")}
    for name in args.runs:
        ckpt = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
        model = build_model(ckpt["arch"], pretrained=False)
        model.load_state_dict(ckpt["model"])
        model = model.to(device)

        out_dir = RESULTS_DIR / "runs" / name
        report = {"checkpoint_epoch": ckpt["epoch"]}
        for split, loader in loaders.items():
            for tta in (False, True):
                tag = f"{split}{'_tta' if tta else ''}"
                probs, labels = predict(model, loader, device, tta)
                np.savez_compressed(out_dir / f"probs_{tag}.npz", probs=probs, labels=labels)
                entry = {"metrics": summary_metrics(probs, labels)}
                if split == "test":
                    entry["ci95"] = bootstrap_ci(probs, labels)
                    entry["per_class"] = per_class_report(probs, labels)
                    entry["confusion"] = confusion(probs, labels)
                report[tag] = entry
        save_json(report, out_dir / "eval.json")
        t, tt = report["test"]["metrics"], report["test_tta"]["metrics"]
        print(f"{name:28s} test acc={t['accuracy']:.4f} bal={t['balanced_accuracy']:.4f} "
              f"F1={t['macro_f1']:.4f} AUC={t['macro_auc']:.4f} | TTA acc={tt['accuracy']:.4f} "
              f"F1={tt['macro_f1']:.4f} AUC={tt['macro_auc']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
