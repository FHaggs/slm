#!/bin/bash
# Benchmark em 1 GPU, com um processo por medida para nao herdar memoria de uma tentativa anterior.
# A medida em 2 GPUs deve ser enviada quando as duas estiverem livres.
#SBATCH -J slm-bench
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 40
#SBATCH --no-requeue
#SBATCH -o logs/%x.%j.out

set -uo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() { python scripts/bench.py "$@" 2>&1 | grep -E "^(torch |batch=)|Error" | cut -c1-200; }

echo "=== sem compile"
for batch in 16 32 48 64; do run --preset final --batch-sizes "$batch" --no-compile; done

echo "=== com compile, sem shape padding"
export TORCHINDUCTOR_SHAPE_PADDING=0
for batch in 16 24 32 48 64; do run --preset final --batch-sizes "$batch"; done
run --preset plan_b --batch-sizes 32
run --preset pilot --batch-sizes 32 64
