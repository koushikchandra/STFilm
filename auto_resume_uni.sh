#!/bin/bash
# Wait for a GPU that is not just momentarily up but STABLE, then launch the
# UNI resume sweep once. Stability check guards against flapping nodes like the
# two that died mid-run (NVLink reset / fabric-manager Error 802).
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology

STABLE_NEEDED=3   # consecutive passes required
GAP=30            # seconds between checks
echo "[auto-uni] started $(date '+%F %T') on $(hostname), waiting for a STABLE GPU..."

streak=0
while true; do
  if PYTHONPATH=STFlow python gpu_ready.py >/dev/null 2>&1; then
    streak=$((streak+1))
    echo "[auto-uni] $(date '+%T') gpu_ready OK ($streak/$STABLE_NEEDED)"
    [ "$streak" -ge "$STABLE_NEEDED" ] && break
  else
    [ "$streak" -ne 0 ] && echo "[auto-uni] $(date '+%T') gpu_ready FAILED, resetting streak"
    streak=0
  fi
  sleep "$GAP"
done

echo "[auto-uni] GPU stable at $(date '+%F %T') on $(hostname) -> launching resume"
bash run_uni_resume.sh
echo "[auto-uni] resume exited code $? at $(date '+%F %T')"
