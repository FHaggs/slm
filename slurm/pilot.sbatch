#!/bin/bash
# Fase 3: piloto (~32M params). Pre-treino em 500M tokens -> mid-training M e C -> avaliacao e McNemar.
# Le os dados do grupo em $SLM_SHARED (somente leitura) e grava os runs em $SLM_SCRATCH (por usuario).
# Cada etapa pula se ja tiver final.pt (ou o JSON da avaliacao): resubmeter retoma de onde parou.
#SBATCH -J slm-pilot
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --no-requeue
#SBATCH -o logs/%x.%j.out

set -euo pipefail
source .venv/bin/activate
export SLM_SCRATCH="${SLM_SCRATCH:-/scratch/$USER/slm}"        # runs do piloto
export SLM_SHARED="${SLM_SHARED:-/scratch/ap211/slm-shared}"   # dados do grupo (so leitura)
case "$SLM_SCRATCH" in
    "$SLM_SHARED"*) echo "SLM_SCRATCH nao pode ficar dentro de SLM_SHARED"; exit 1 ;;
esac
export HF_HOME="$SLM_SCRATCH/hf-cache"                         # benchmarks fora do /home (cota de 10 GB)
export TRITON_CACHE_DIR="$SLM_SCRATCH/triton-cache"
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_SHAPE_PADDING=0                           # sem isso o torch.compile estoura a memoria
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
runs="$SLM_SCRATCH/runs/pilot"
results=results/pilot

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true

echo "=== pre-treino"
python scripts/pretrain.py --config configs/pilot/pretrain.yaml

for run in mid_m mid_c; do
    echo "=== mid-training: $run"
    python scripts/midtrain.py --config "configs/pilot/$run.yaml"
done

for stage in pretrain mid_m mid_c; do
    if [ -f "$results/$stage.json" ]; then
        echo "=== avaliacao: $stage ja existe, pulando"
        continue
    fi
    echo "=== avaliacao: $stage"
    python scripts/evaluate.py --checkpoint "$runs/$stage/final.pt" --output "$results/$stage.json" \
        --val-dir "$SLM_SHARED/data/pretrain/val" --benchmarks hellaswag arc_easy
done

echo "=== M contra o pre-treino"
python scripts/compare.py "$results/pretrain.json" "$results/mid_m.json"
echo "=== M contra C"
python scripts/compare.py "$results/mid_c.json" "$results/mid_m.json"
