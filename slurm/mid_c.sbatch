#!/bin/bash
# Fase 5 (parte do servidor): mid-training C (controle: so chat), a partir do pre-treino final.
# So treina. As avaliacoes dos 4 modelos (pre, M, C e SFT) rodam no fim numa mesma maquina
# (decisao de 15/11). M e SFT rodam localmente (branch midtraining-m).
# Le e grava em $SLM_SHARED: qualquer membro do grupo retoma do ultimo checkpoint ao resubmeter.
# Pre-requisito: checkpoints/pretrain/final.pt (Fase 4). Sem ele, o job aborta.
#SBATCH -J slm-mid-c
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --no-requeue
#SBATCH -o logs/%x.%j.out

set -euo pipefail
umask 002  # o grupo precisa escrever e retomar
source .venv/bin/activate
export SLM_SCRATCH="${SLM_SCRATCH:-/scratch/$USER/slm}"        # caches por usuario
export SLM_SHARED="${SLM_SHARED:-/scratch/ap211/slm-shared}"   # dados e checkpoints do grupo
export HF_HOME="$SLM_SCRATCH/hf-cache"                         # fora do /home (cota de 10 GB)
export TRITON_CACHE_DIR="$SLM_SCRATCH/triton-cache"
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_SHAPE_PADDING=0                           # sem isso o torch.compile estoura a memoria
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ck="$SLM_SHARED/checkpoints"
run_dir="$ck/mid_c"                                            # o mesmo run_dir de configs/final/mid_c.yaml

if [ ! -f "$ck/pretrain/final.pt" ]; then
    echo "falta $ck/pretrain/final.pt: o pre-treino (Fase 4) ainda nao terminou. Abortando"
    exit 1
fi

# Um job por vez neste run_dir: dois jobs gravando os mesmos checkpoints se corromperiam.
mkdir -p "$run_dir"
exec 9>"$run_dir/.lock"
if ! flock -n 9; then
    echo "outro job ja esta usando $run_dir: abortando"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true
echo "=== mid-training C"
# Corte por tempo meia hora antes do limite do Slurm (6h): salva checkpoint e sai; resubmeter retoma.
python scripts/midtrain.py --config configs/final/mid_c.yaml --max-hours 5.5
ls -la "$run_dir"
