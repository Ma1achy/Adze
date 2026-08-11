#!/bin/zsh
# dag10 M7, sequentially — one MPS job at a time.
#
# --band-width 6 is NOT dag10's distance_max (8); it is dec10's. The interior
# band keeps blocks 2..n-1-width, so at chain 10 a width of 8 admits nothing.
# Width 6 reproduces dec10's committed selection geometry (blocks 2 and 3)
# exactly, which is what makes the RESULT gap comparable to its +0.43%.
# The unbanded run is the wider read on the same checkpoint.
set -u
CKPT=checkpoints/denoiser_dag10_d16_mixedP50.pt
COMMON=(--config configs/dag10.yaml
        --vae checkpoints/vae_dag10_d16.pt
        --denoiser "$CKPT"
        --traces 3000 --nfe 32 --eta 1.0 --seed 0)

echo "=== M7 early, interior band width 6 (dec10-matched) ==="
PYTHONPATH=src python -u scripts/m7_central.py "${COMMON[@]}" \
  --interior-band --band-width 6 \
  --corrupt early --out runs/dag10_early_band6.json 2>&1

echo "=== M7 early, unbanded ==="
PYTHONPATH=src python -u scripts/m7_central.py "${COMMON[@]}" \
  --corrupt early --out runs/dag10_early.json 2>&1

echo "=== M7 final, unbanded ==="
PYTHONPATH=src python -u scripts/m7_central.py "${COMMON[@]}" \
  --corrupt final --out runs/dag10_final.json 2>&1
