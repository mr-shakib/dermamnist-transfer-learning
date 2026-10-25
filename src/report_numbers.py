"""Write report/generated/numbers.tex: every number quoted in the report text, as LaTeX macros.

Keeps the prose in sync with results/eda.json and results/summary.json (no hand-copied numbers).
Usage: python src/report_numbers.py
"""
import json
import os

import torch

from common import RESULTS_DIR, ROOT

ALIAS = {  # run name -> macro prefix
    "scratch_cnn": "Scratch",
    "resnet50_frozen": "RFrozen",
    "effnetb0_frozen": "EFrozen",
    "resnet50_finetune": "RFine",
    "effnetb0_finetune": "EFine",
    "resnet50_gradual": "RGrad",
    "resnet50_finetune_noweights": "RNoW",
}
METRICS = {"accuracy": "Acc", "balanced_accuracy": "BalAcc", "macro_f1": "FOne", "macro_auc": "AUC", "ece": "ECE",
           "weighted_f1": "WFOne", "cohen_kappa": "Kappa"}


def main() -> None:
    eda = json.load(open(RESULTS_DIR / "eda.json"))
    summ = json.load(open(RESULTS_DIR / "summary.json"))
    m = {
        "MajorityAcc": f"{100 * eda['majority_class_baseline_acc_test']:.1f}",
        "NumRuns": str(len(summ["runs"])),
        "DatasetSizeMB": f"{os.path.getsize(ROOT / 'data' / 'dermamnist_224.npz') / 1e6:,.0f}".replace(",", "{,}"),
        "TorchVersion": torch.__version__.split("+")[0],
        "BestLabel": summ["best_label"],
        "BestTTA": "with 8-view TTA" if summ["best_uses_tta"] else "without TTA",
    }
    for run, pre in ALIAS.items():
        if run not in summ["runs"]:
            continue
        r = summ["runs"][run]
        for split in ("test", "test_tta", "val"):
            suffix = {"test": "", "test_tta": "TTA", "val": "Val"}[split]
            for key, short in METRICS.items():
                v = r[split][key]
                m[f"{pre}{short}{suffix}"] = f"{v:.3f}" if key in ("macro_auc", "ece", "cohen_kappa") else f"{100 * v:.1f}"
        for key, short in (("macro_f1", "FOne"), ("accuracy", "Acc"), ("macro_auc", "AUC")):
            lo, hi = r["ci95"][key]
            fmt = (lambda x: f"{x:.3f}") if key == "macro_auc" else (lambda x: f"{100 * x:.1f}")
            m[f"{pre}{short}CI"] = f"[{fmt(lo)}, {fmt(hi)}]"
        ev = json.load(open(RESULTS_DIR / "runs" / run / "eval.json"))
        for key, short in (("macro_f1", "FOne"), ("accuracy", "Acc"), ("balanced_accuracy", "BalAcc"),
                           ("macro_auc", "AUC")):
            lo, hi = ev["test_tta"]["ci95"][key]
            fmt = (lambda x: f"{x:.3f}") if key == "macro_auc" else (lambda x: f"{100 * x:.1f}")
            m[f"{pre}{short}TTACI"] = f"[{fmt(lo)}, {fmt(hi)}]"
        m[f"{pre}BestEpoch"] = str(r["train"]["best_epoch"])
        m[f"{pre}EpochTime"] = f"{r['train']['mean_epoch_time_s']:.0f}"
        m[f"{pre}TrainMin"] = f"{r['train']['total_train_time_s'] / 60:.1f}"
        for c, f1 in r["per_class_f1"].items():
            m[f"{pre}FOne{c.capitalize()}"] = f"{100 * f1:.1f}"
    # selected model, per class (on the evaluation variant chosen on validation)
    for c, d in summ["best_per_class"].items():
        cc = c.capitalize()
        m[f"BestPrec{cc}"] = f"{100 * d['precision']:.1f}"
        m[f"BestRec{cc}"] = f"{100 * d['recall']:.1f}"
        m[f"BestFOne{cc}"] = f"{100 * d['f1']:.1f}"
        m[f"BestAUC{cc}"] = f"{d['auc']:.3f}"
    emb = json.load(open(RESULTS_DIR / "runs" / summ["best_run"] / "embedding_analysis.json"))["silhouette_cosine"]
    m["SilImageNet"] = f"{emb['ImageNet (before fine-tuning)']:.3f}"
    m["SilFineTuned"] = f"{emb['fine-tuned']:.3f}"
    m["TotalTrainMin"] = f"{sum(r['train']['total_train_time_s'] for r in summ['runs'].values()) / 60:.0f}"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    out = ROOT / "report" / "generated" / "numbers.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} macros to {out}")


if __name__ == "__main__":
    main()
