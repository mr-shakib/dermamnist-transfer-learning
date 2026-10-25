"""Train one experiment configuration on DermaMNIST (224x224).

Strategies
  scratch   random initialisation, every layer trained at --lr-head
  frozen    ImageNet weights, backbone frozen (BN in eval mode), only the head is trained
  finetune  ImageNet weights, all layers trained with discriminative LRs (backbone --lr-backbone, head --lr-head)
  gradual   ImageNet weights, head first, then one backbone stage unfrozen every --unfreeze-every epochs
            (top stage first); deeper-from-output stages get geometrically smaller LRs (ULMFiT-style)

Example
  python src/train.py --name resnet50_finetune --arch resnet50 --strategy finetune --epochs 12
"""
from __future__ import annotations

import argparse
import csv
import math
import time

import numpy as np
import torch
from torch import nn

from common import CKPT_DIR, NUM_CLASSES, RESULTS_DIR, class_weights, load_split, make_loader, save_json, \
    seed_everything
from metrics import summary_metrics
from models import backbone_stages, build_model, count_params, head_module


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", choices=["simplecnn", "resnet50", "efficientnet_b0"], required=True)
    ap.add_argument("--strategy", choices=["scratch", "frozen", "finetune", "gradual"], required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="gradient-accumulation steps")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--stage-decay", type=float, default=2.0, help="LR divisor per stage (gradual)")
    ap.add_argument("--unfreeze-every", type=int, default=2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--class-weight-power", type=float, default=0.5, help="0 disables class weighting")
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true",
                    help="FP16 autocast (off by default: ~3x slower than FP32 on GTX 16-series without tensor cores)")
    return ap.parse_args()


def build_param_groups(model, args):
    head = head_module(model, args.arch)
    head_ids = {id(p) for p in head.parameters()}
    if args.strategy == "scratch":
        return [{"params": list(model.parameters()), "lr": args.lr_head, "name": "all"}]
    groups = [{"params": list(head.parameters()), "lr": args.lr_head, "name": "head"}]
    if args.strategy == "frozen":
        return groups
    stages = backbone_stages(model, args.arch)
    if args.strategy == "finetune":
        groups.append({"params": [p for p in model.parameters() if id(p) not in head_ids],
                       "lr": args.lr_backbone, "name": "backbone"})
        return groups
    # gradual: stage i (0 = closest to input) gets lr_backbone / decay^(n-1-i)
    n = len(stages)
    for i, st in enumerate(stages):
        groups.append({"params": list(st.parameters()), "lr": args.lr_backbone / args.stage_decay ** (n - 1 - i),
                       "name": f"stage{i}"})
    return groups


def frozen_modules(model, args, epoch: int) -> list[nn.Module]:
    """Modules whose parameters receive no gradient (and whose BN statistics stay fixed) at this epoch."""
    if args.strategy in ("scratch", "finetune"):
        return []
    stages = backbone_stages(model, args.arch)
    if args.strategy == "frozen":
        return stages
    n_unfrozen = min(len(stages), epoch // args.unfreeze_every)
    return stages[: len(stages) - n_unfrozen]


def set_trainable(model, frozen: list[nn.Module]) -> None:
    for p in model.parameters():
        p.requires_grad_(True)
    for m in frozen:
        for p in m.parameters():
            p.requires_grad_(False)


@torch.no_grad()
def predict(model, loader, device, amp: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs, labels, loss_sum = [], [], 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
        loss_sum += ce(logits.float(), y).item()
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        labels.append(y.cpu().numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    return probs, labels, loss_sum / len(labels)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    out_dir = RESULTS_DIR / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)

    train_loader = make_loader("train", True, args.batch_size, args.workers)
    val_loader = make_loader("val", False, 64, args.workers)
    _, y_train = load_split("train")

    model = build_model(args.arch, pretrained=args.strategy != "scratch", dropout=args.dropout)
    model = model.to(device)

    weights = class_weights(y_train, args.class_weight_power) if args.class_weight_power > 0 else None
    criterion = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device),
                                    label_smoothing=args.label_smoothing)

    groups = build_param_groups(model, args)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader) // args.accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    config = vars(args) | {
        "class_weights": None if weights is None else [round(float(w), 4) for w in weights],
        "params_total": count_params(model)[0],
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
    }
    save_json(config, out_dir / "config.json")
    print({k: config[k] for k in ("name", "arch", "strategy", "params_total", "class_weights")}, flush=True)

    history, best, best_epoch, bad_epochs = [], -1.0, -1, 0
    torch.cuda.reset_peak_memory_stats()
    run_start = time.time()
    for epoch in range(args.epochs):
        frozen = frozen_modules(model, args, epoch)
        set_trainable(model, frozen)
        model.train()
        for m in frozen:
            m.eval()
        trainable = count_params(model)[1]

        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for it, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                logits = model(x)
                loss = criterion(logits.float(), y)
            scaler.scale(loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"] if p.requires_grad], 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        train_time = time.time() - t0

        probs, labels, val_loss = predict(model, val_loader, device, args.amp)
        m = summary_metrics(probs, labels)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / seen,
            "train_acc": correct / seen,
            "val_loss": val_loss,
            "val_acc": m["accuracy"],
            "val_bal_acc": m["balanced_accuracy"],
            "val_macro_f1": m["macro_f1"],
            "val_macro_auc": m["macro_auc"],
            "trainable_params": trainable,
            "lr_head": optimizer.param_groups[0]["lr"],
            "epoch_time_s": train_time,
        }
        history.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

        if m["macro_f1"] > best:
            best, best_epoch, bad_epochs = m["macro_f1"], epoch + 1, 0
            torch.save({"model": model.state_dict(), "arch": args.arch, "epoch": epoch + 1, "val": m},
                       CKPT_DIR / f"{args.name}.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping at epoch {epoch + 1}", flush=True)
                break

    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    save_json({"best_epoch": best_epoch, "best_val_macro_f1": best,
               "epochs_run": len(history), "total_train_time_s": time.time() - run_start,
               "mean_epoch_time_s": float(np.mean([h["epoch_time_s"] for h in history])),
               "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1024 ** 3},
              out_dir / "train_summary.json")
    print(f"best val macro-F1 {best:.4f} at epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
