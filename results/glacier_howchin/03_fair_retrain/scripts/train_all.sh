#!/usr/bin/env bash
# Train the three variants sequentially with the original trainer (gsplat_eval @ github HEAD,
# code identical to 1ce1dd8 used for the old table): 30k iters, --downsample 4 (1000x750),
# test every 8th, seed 42. Skips a variant whose metrics.json already exists.
set -uo pipefail
R="/mnt/windows/Dataset/2016-11-28_Howchin-AlphLake_Imagery-Files.beh/retrain_fair"
PY=/home/otter77/miniconda3/envs/gsplat/bin/python
export TORCH_CUDA_ARCH_LIST=12.0
cd /home/otter77/git_project/gsplat_eval
for v in ${VARIANTS:-agent sam3_noagent nomask}; do
  out="$R/runs/$v"
  if [ -f "$out/metrics.json" ]; then echo "[train_all] $v done already, skip"; continue; fi
  echo "[train_all] === $v start $(date '+%F %T') ==="
  "$PY" -m gsplat_eval.cli train --data "$R/data/$v" --output "$out" \
      --iters 30000 --downsample 4 --test-every 8 > "$R/runs/$v.stdout.log" 2>&1
  rc=$?
  echo "[train_all] === $v exit=$rc $(date '+%F %T') ==="
done
echo "[train_all] ALL DONE $(date '+%F %T')"
