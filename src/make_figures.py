"""Aggregate all runs into comparison figures, LaTeX tables and a numbers file for the report.

Usage: python src/make_figures.py
"""
from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.metrics import roc_curve

import plot_style
from common import CLASS_CODES, FIG_DIR, NUM_CLASSES, RESULTS_DIR, ROOT, save_json

plot_style.apply()
import matplotlib.pyplot as plt  # noqa: E402

RUNS = {
    "scratch_cnn": "SimpleCNN (scratch)",
    "resnet50_frozen": "ResNet-50 (frozen)",
    "effnetb0_frozen": "EfficientNet-B0 (frozen)",
    "resnet50_finetune": "ResNet-50 (fine-tune)",
    "resnet50_gradual": "ResNet-50 (gradual unfreeze)",
    "effnetb0_finetune": "EfficientNet-B0 (fine-tune)",
    "resnet50_finetune_noweights": "ResNet-50 (fine-tune, unweighted CE)",
}
GEN_DIR = ROOT / "report" / "generated"


def load_run(name: str) -> dict:
    d = RESULTS_DIR / "runs" / name
    with open(d / "history.csv") as f:
        hist = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return {
        "history": hist,
        "config": json.load(open(d / "config.json")),
        "train": json.load(open(d / "train_summary.json")),
        "eval": json.load(open(d / "eval.json")),
    }


def ink_on(rgba) -> str:
    """Text colour with the better contrast on a given cell colour (WCAG relative luminance)."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#ffffff" if (1.05 / (lum + 0.05)) > ((lum + 0.05) / 0.05) else plot_style.TEXT


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main() -> None:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    runs = {k: load_run(k) for k in RUNS if (RESULTS_DIR / "runs" / k / "eval.json").exists()}
    names = list(runs)
    # model selection strictly on validation macro-F1 (TTA variant chosen on validation as well)
    best = max(names, key=lambda n: runs[n]["eval"]["val"]["metrics"]["macro_f1"])
    use_tta = runs[best]["eval"]["val_tta"]["metrics"]["macro_f1"] > runs[best]["eval"]["val"]["metrics"]["macro_f1"]

    # ---- validation macro-F1 curves
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    for j, n in enumerate(names):
        ep = [h["epoch"] for h in runs[n]["history"]]
        axes[0].plot(ep, [h["val_macro_f1"] for h in runs[n]["history"]], color=plot_style.SERIES[j],
                     label=RUNS[n], linewidth=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation macro-F1")
    axes[0].set_title("Validation macro-F1 per epoch")
    h = runs[best]["history"]
    ep = [r["epoch"] for r in h]
    axes[1].plot(ep, [r["train_loss"] for r in h], color=plot_style.SERIES[0], label="train loss")
    axes[1].plot(ep, [r["val_loss"] for r in h], color=plot_style.SERIES[1], label="validation loss")
    axes[1].axvline(runs[best]["train"]["best_epoch"], color=plot_style.TEXT_2, linewidth=1, linestyle="--")
    lo_, hi_ = axes[1].get_ylim()
    axes[1].text(runs[best]["train"]["best_epoch"] - 0.15, lo_ + 0.55 * (hi_ - lo_), "selected\nepoch",
                 fontsize=7, color=plot_style.TEXT_2, va="center", ha="right")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy")
    axes[1].set_title(f"Loss curves: {RUNS[best]}")
    axes[1].legend(loc="upper center")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png")
    plt.close(fig)

    # ---- test macro-F1 and accuracy with bootstrap CIs (dot plot)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    order = sorted(names, key=lambda n: runs[n]["eval"]["test"]["metrics"]["macro_f1"])
    ys = np.arange(len(order))
    for a, key, title in zip(axes, ("macro_f1", "balanced_accuracy", "accuracy"),
                             ("Macro-F1", "Balanced accuracy", "Accuracy")):
        vals = [runs[n]["eval"]["test"]["metrics"][key] for n in order]
        lo = [runs[n]["eval"]["test"]["ci95"][key][0] for n in order]
        hi = [runs[n]["eval"]["test"]["ci95"][key][1] for n in order]
        a.hlines(ys, lo, hi, color=plot_style.SEQ_BLUE[2], linewidth=2.5)
        a.plot(vals, ys, "o", color=plot_style.SERIES[0], markersize=7)
        for y, v in zip(ys, vals):
            a.text(v, y + 0.25, f"{v:.3f}", ha="center", fontsize=7, color=plot_style.TEXT_2)
        a.set_title(title)
        a.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [RUNS[n] for n in order])
    fig.suptitle("Test-set metrics (dot) with 95% bootstrap confidence interval (bar)", fontsize=10,
                 fontweight="bold", color=plot_style.TEXT)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "test_metric_comparison.png")
    plt.close(fig)

    # ---- per-class F1 heatmap (runs x classes)
    mat = np.array([[runs[n]["eval"]["test"]["per_class"][c]["f1"] for c in CLASS_CODES] for n in names])
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", plot_style.SEQ_BLUE)
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color=ink_on(cmap(mat[i, j])))
    ax.set_xticks(range(NUM_CLASSES), CLASS_CODES)
    ax.set_yticks(range(len(names)), [RUNS[n] for n in names])
    ax.grid(False)
    ax.set_title("Per-class test F1")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_class_f1_heatmap.png")
    plt.close(fig)

    # ---- best model: confusion matrix, ROC, reliability
    tag = "test_tta" if use_tta else "test"
    z = np.load(RESULTS_DIR / "runs" / best / f"probs_{tag}.npz")
    probs, labels = z["probs"], z["labels"]
    cm = np.array(runs[best]["eval"][tag]["confusion"])
    cmn = cm / cm.sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    im = axes[0].imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[0].text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=6.5,
                         color=ink_on(cmap(cmn[i, j])))
    axes[0].set_xticks(range(NUM_CLASSES), CLASS_CODES)
    axes[0].set_yticks(range(NUM_CLASSES), CLASS_CODES)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].grid(False)
    axes[0].set_title("Row-normalised confusion matrix (test)")
    for k in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(labels == k, probs[:, k])
        auc = runs[best]["eval"][tag]["per_class"][CLASS_CODES[k]]["auc"]
        axes[1].plot(fpr, tpr, color=plot_style.SERIES[k], linewidth=1.6, label=f"{CLASS_CODES[k]} (AUC {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color=plot_style.GRID, linewidth=1, linestyle="--")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("One-vs-rest ROC curves (test)")
    axes[1].legend(loc="lower right")
    fig.suptitle(f"{RUNS[best]}{' + TTA' if use_tta else ''}", fontsize=10, fontweight="bold",
                 color=plot_style.TEXT)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_confusion_roc.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    for j, (t, lab) in enumerate((("test", "single view"), ("test_tta", "8-view TTA"))):
        zz = np.load(RESULTS_DIR / "runs" / best / f"probs_{t}.npz")
        conf, corr = zz["probs"].max(1), zz["probs"].argmax(1) == zz["labels"]
        edges = np.linspace(0, 1, 11)
        xs, ys_ = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 10:
                xs.append(conf[m].mean())
                ys_.append(corr[m].mean())
        ece = runs[best]["eval"][t]["metrics"]["ece"]
        ax.plot(xs, ys_, marker="o", markersize=5, color=plot_style.SERIES[j], label=f"{lab} (ECE {ece:.3f})")
    ax.plot([0, 1], [0, 1], color=plot_style.GRID, linestyle="--", linewidth=1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability diagram (test)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_reliability.png")
    plt.close(fig)

    # ---- LaTeX: main results table
    rows = []
    for n in names:
        r = runs[n]
        t, v = r["eval"]["test"]["metrics"], r["eval"]["val"]["metrics"]
        ci = r["eval"]["test"]["ci95"]
        params = r["config"]["params_total"] / 1e6
        trainable = r["history"][-1]["trainable_params"] / 1e6
        cells = [RUNS[n], f"{params:.2f}", f"{trainable:.2f}", f"{r['train']['best_epoch']}/{r['train']['epochs_run']}",
                 pct(v["macro_f1"]), pct(t["accuracy"]), pct(t["balanced_accuracy"]),
                 f"{pct(t['macro_f1'])} \\scriptsize[{pct(ci['macro_f1'][0])}, {pct(ci['macro_f1'][1])}]",
                 f"{t['macro_auc']:.3f}", f"{t['ece']:.3f}"]
        if n == best:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        rows.append(" & ".join(cells) + r" \\")
    (GEN_DIR / "results_table.tex").write_text("\n".join(rows) + "\n")

    # ---- LaTeX: TTA table for all runs
    tta_rows = []
    for n in names:
        a, b = runs[n]["eval"]["test"]["metrics"], runs[n]["eval"]["test_tta"]["metrics"]
        tta_rows.append(f"{RUNS[n]} & {pct(a['accuracy'])} & {pct(b['accuracy'])} & {pct(a['macro_f1'])} & "
                        f"{pct(b['macro_f1'])} & {a['macro_auc']:.3f} & {b['macro_auc']:.3f} \\\\")
    (GEN_DIR / "tta_table.tex").write_text("\n".join(tta_rows) + "\n")

    # ---- LaTeX: per-class table for the selected model
    pc = runs[best]["eval"][tag]["per_class"]
    pc_rows = [f"\\texttt{{{c}}} & {pc[c]['support']} & {pct(pc[c]['precision'])} & {pct(pc[c]['recall'])} & "
               f"{pct(pc[c]['f1'])} & {pc[c]['auc']:.3f} \\\\" for c in CLASS_CODES]
    (GEN_DIR / "per_class_table.tex").write_text("\n".join(pc_rows) + "\n")

    # ---- LaTeX: compute table
    comp_rows = [f"{RUNS[n]} & {runs[n]['train']['mean_epoch_time_s']:.0f} & "
                 f"{runs[n]['train']['total_train_time_s'] / 60:.1f} & {runs[n]['train']['peak_gpu_mem_gb']:.2f} \\\\"
                 for n in names]
    (GEN_DIR / "compute_table.tex").write_text("\n".join(comp_rows) + "\n")

    summary = {
        "best_run": best, "best_label": RUNS[best], "best_uses_tta": use_tta,
        "runs": {n: {"label": RUNS[n], "val": runs[n]["eval"]["val"]["metrics"],
                     "val_tta": runs[n]["eval"]["val_tta"]["metrics"],
                     "test": runs[n]["eval"]["test"]["metrics"], "test_tta": runs[n]["eval"]["test_tta"]["metrics"],
                     "ci95": runs[n]["eval"]["test"]["ci95"], "train": runs[n]["train"],
                     "per_class_f1": {c: runs[n]["eval"]["test"]["per_class"][c]["f1"] for c in CLASS_CODES}}
                 for n in names},
        "best_per_class": pc,
        "best_confusion": runs[best]["eval"][tag]["confusion"],
    }
    save_json(summary, RESULTS_DIR / "summary.json")
    print(json.dumps({n: {k: round(v, 4) for k, v in summary["runs"][n]["test"].items()} for n in names}, indent=1))
    print("best:", best, "tta:", use_tta)


if __name__ == "__main__":
    main()
