#!/bin/bash
# Fase 1: pipeline completo em miniatura, pre-treino -> mid-training -> SFT -> avaliacao.
# Requer slurm/smoke_data.sbatch.
#SBATCH -J slm-smoke
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 60
#SBATCH --no-requeue
#SBATCH -o logs/%x.%j.out

set -euo pipefail
source .venv/bin/activate
export SLM_SCRATCH="${SLM_SCRATCH:-/scratch/$USER/slm}"
export HF_HOME="$SLM_SCRATCH/hf-cache"
runs="$SLM_SCRATCH/runs/smoke"

python scripts/pretrain.py --config configs/smoke/pretrain.yaml
python scripts/midtrain.py --config configs/smoke/mid.yaml
python scripts/sft.py --config configs/smoke/sft.yaml

for stage in pretrain mid sft; do
    echo "=== avaliacao: $stage"
    python scripts/evaluate.py --checkpoint "$runs/$stage/final.pt" \
        --output "results/smoke/$stage.json" \
        --val-dir "$SLM_SCRATCH/data/smoke/pretrain/val" \
        --benchmarks hellaswag arc_easy --mc-limit 200 --gsm-limit 50 --gsm-new-tokens 32
done

echo "=== comparacao"
python scripts/compare.py results/smoke/pretrain.json results/smoke/mid.json
