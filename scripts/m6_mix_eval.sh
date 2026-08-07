#!/bin/bash
# Evaluate one arm of the regime A/B mix sweep. One GPU job at a time, in order.
#
#   (a) THE HANDICAP    root null on the pad-free subset, plus the 2x2 crossing
#                       table (conditioning x mask). Prediction: shrinks as the
#                       regime-B share rises. If it does not, specialisation is
#                       not the cause and the M7 diagnosis is wrong.
#   (b) THE EFFECT      `early` and `final` at n=2000, dumped for m7_strata.py,
#                       which reports the aggregate gap, the DISTANCE PROFILE,
#                       global's absolute rate against chance, and the
#                       recalibration against THIS arm's own handicap.
#
# Draft quality — the cost side — is a separate script, run once over all arms.
#
# Usage:  bash scripts/m6_mix_eval.sh checkpoints/denoiser_cap100_d16_L4_mixedP50.pt p50
set -e
CKPT="$1"
TAG="$2"
[ -n "$CKPT" ] && [ -n "$TAG" ] || { echo "usage: $0 <checkpoint> <tag>"; exit 2; }

echo "##### (a) HANDICAP — root null, pad-free, plus the crossing table: $TAG"
PYTHONPATH=src python3 scripts/m7_shield.py --denoiser "$CKPT"

echo
echo "##### (b) EFFECT — early and final at n=2000: $TAG"
PYTHONPATH=src python3 scripts/m7_central.py --corrupt final --traces 2000 \
    --denoiser "$CKPT" --out "runs/${TAG}_final.json" | sed -n '9,18p'
PYTHONPATH=src python3 scripts/m7_central.py --corrupt early --traces 2000 \
    --denoiser "$CKPT" --out "runs/${TAG}_early.json" | sed -n '9,18p'

echo
echo "##### STRATA — distance profile, chance rates, recalibration: $TAG"
PYTHONPATH=src python3 scripts/m7_strata.py "runs/${TAG}_final.json" "runs/${TAG}_early.json"
