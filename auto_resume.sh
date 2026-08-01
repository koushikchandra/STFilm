#!/bin/bash
# Poll until the GPU is actually usable, then launch the seed-3 resume once.
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology
echo "[auto-resume] started $(date +%H:%M:%S), waiting for GPU..."
until PYTHONPATH=STFlow python gpu_ready.py 2>/dev/null; do
  sleep 60
done
echo "[auto-resume] GPU is ready at $(date +%H:%M:%S) -> launching seed-3 resume"
PYTHONPATH=STFlow bash run_ablation2_resume.sh
echo "[auto-resume] resume script exited at $(date +%H:%M:%S)"
