#!/bin/bash
# Avalia um checkpoint no servidor (perplexidade, HellaSwag, ARC-Easy, aritmetica, GSM8K) e compara com M.
# Uso: sbatch --export=ALL,NAME=mid_c,CKPT=/scratch/ap211/slm-shared/checkpoints/mid_c/final.pt[,EXTRA=--skip-gsm8k] slurm/eval.sbatch
# Mesmos lotes de avaliacao da RTX 4060 (run_local.py). Sem GPU livre: sbatch --gres=gpu:0 -c 32 ... (CPU, fp32 em vez de bf16).
#SBATCH -J slm-eval
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --no-requeue
#SBATCH -o logs/%x.%j.out

set -euo pipefail
umask 002
source .venv/bin/activate
export SLM_SCRATCH="${SLM_SCRATCH:-/scratch/$USER/slm}"
export SLM_SHARED="${SLM_SHARED:-/scratch/ap211/slm-shared}"
export HF_HOME="$SLM_SCRATCH/hf-cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
: "${NAME:?defina NAME}" "${CKPT:?defina CKPT}"

python scripts/evaluate.py --checkpoint "$CKPT" --output "results/final/$NAME.json" \
    --val-dir "$SLM_SHARED/data/pretrain/val" --benchmarks hellaswag arc_easy \
    --ppl-batch-size 4 --gsm-batch-size 8 ${EXTRA:-}
if [ "$NAME" = mid_c ] && [ -f results/final/mid_m.json ]; then
    python scripts/compare.py results/final/mid_c.json results/final/mid_m.json | tee results/final/compare_mid_c_x_mid_m.txt
    python scripts/compare.py results/final/pretrain.json results/final/mid_c.json | tee results/final/compare_pretrain_x_mid_c.txt
fi
