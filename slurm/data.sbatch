#!/bin/bash
# Fase 2: gera todos os dados finais em /scratch/ap211/slm-shared/data. Nao usa GPU.
# Uma etapa so e refeita se o diretorio dela nao tiver meta.json; para refazer, apague o diretorio.
#SBATCH -J slm-data
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --no-requeue
#SBATCH -o logs/%x.%j.out

set -euo pipefail
umask 002  # o grupo precisa escrever nos arquivos novos
source .venv/bin/activate
export SLM_SCRATCH="${SLM_SCRATCH:-/scratch/ap211/slm}"           # caches (regeneraveis)
export SLM_SHARED="${SLM_SHARED:-/scratch/ap211/slm-shared}"      # dados persistentes do grupo
export HF_HOME="$SLM_SCRATCH/hf-cache"
export CUDA_VISIBLE_DEVICES=""
data="$SLM_SHARED/data"

once() {  # once <diretorio> <comando...>
    local dir="$1"; shift
    if [ -f "$dir/train/meta.json" ] && [ -f "$dir/val/meta.json" ]; then
        echo "== $dir ja existe, pulando"
    else
        echo "== gerando $dir"
        "$@"
    fi
}

once "$data/pretrain" python scripts/tokenize_pretrain.py --out "$data/pretrain" --seq-len 1024 \
    --max-tokens 2500000000 --val-tokens 50000000 --num-train-files 12 --workers 32
# M: 75% chat + 25% matematica. C: so chat. Mesma seed, entao o chat de M e prefixo do de C.
once "$data/mid_m" python scripts/prepare_chat.py --out "$data/mid_m" --seq-len 1024 \
    --max-tokens 300000000 --math-frac 0.25
once "$data/mid_c" python scripts/prepare_chat.py --out "$data/mid_c" --seq-len 1024 \
    --max-tokens 300000000 --math-frac 0
once "$data/sft" python scripts/prepare_chat.py --out "$data/sft" --seq-len 1024 \
    --max-tokens 100000000 --math-frac 0.25 --seed 1
