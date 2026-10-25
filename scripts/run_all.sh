#!/usr/bin/env bash
# Reproduce every experiment in the report, in order. Run from the repository root:
#   bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

if [ ! -f data/derma224_train_images.npy ]; then
  [ -f data/dermamnist_224.npz ] || curl -L -o data/dermamnist_224.npz \
    "https://zenodo.org/records/10519652/files/dermamnist_224.npz?download=1"
  $PY src/prepare_data.py
fi

$PY src/eda.py

# E1  baseline trained from scratch
$PY src/train.py --name scratch_cnn --arch simplecnn --strategy scratch --epochs 30 --patience 8 --lr-head 1e-3
# E2/E3  feature extraction (frozen ImageNet backbone, only the head is trained)
$PY src/train.py --name resnet50_frozen --arch resnet50 --strategy frozen --epochs 10 --lr-head 1e-3
$PY src/train.py --name effnetb0_frozen --arch efficientnet_b0 --strategy frozen --epochs 10 --lr-head 1e-3
# E4/E5  full fine-tuning with discriminative learning rates
$PY src/train.py --name resnet50_finetune --arch resnet50 --strategy finetune --epochs 12 --lr-backbone 1e-4 --lr-head 1e-3
$PY src/train.py --name effnetb0_finetune --arch efficientnet_b0 --strategy finetune --epochs 12 --lr-backbone 1e-4 --lr-head 1e-3
# E6  gradual unfreezing (head -> layer4 -> layer3 -> layer2 -> stem, one stage every 2 epochs)
$PY src/train.py --name resnet50_gradual --arch resnet50 --strategy gradual --epochs 12 --unfreeze-every 2 --lr-backbone 1e-4 --lr-head 1e-3
# E7  ablation: same as E4 but without class-weighted loss
$PY src/train.py --name resnet50_finetune_noweights --arch resnet50 --strategy finetune --epochs 12 --lr-backbone 1e-4 --lr-head 1e-3 --class-weight-power 0

RUNS="scratch_cnn resnet50_frozen effnetb0_frozen resnet50_finetune effnetb0_finetune resnet50_gradual resnet50_finetune_noweights"
$PY src/evaluate.py --runs $RUNS
$PY src/make_figures.py
BEST=$($PY -c "import json;print(json.load(open('results/summary.json'))['best_run'])")
$PY src/gradcam.py --run "$BEST"
$PY src/report_numbers.py
