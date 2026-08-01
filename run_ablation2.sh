#!/bin/bash
set -e
for S in 1 2 3; do
  for FILM in none context; do
    echo "=== seed=$S film=$FILM ==="
    PYTHONPATH=STFlow python STFlow/stflow/app/flow/train.py \
      --datasets CCRCC PAAD PRAD SKCM --feature_encoder resnet50_trunc \
      --source_dataroot dataset --embed_dataroot embed_dataroot \
      --save_dir results_ablation --seed $S --film $FILM \
      --exp_code seed${S}_${FILM}_v2
  done
done
echo "ABLATION2 DONE"
