# Onde estão os dados

Dados e checkpoints **nunca vão para o Git**. Ficam no `/scratch` do cluster CDIA (disco local, sem backup conhecido).

## Diretório compartilhado

```text
/scratch/ap211/slm-shared/        # dono ap211, grupo AP2, modo 2770 (setgid)
├── data/
│   ├── pretrain/{train,val}/     # FineWeb-Edu, GPT-2: 2,5B tokens de treino e ~50M de validação
│   ├── mid_m/{train,val}/        # mid-training M: 75% chat + 25% matemática (~300M tokens)
│   ├── mid_c/{train,val}/        # mid-training C (controle): só chat (~300M tokens)
│   └── sft/{train,val}/          # SFT: chat + matemática (~100M tokens)
├── checkpoints/                  # pesos e estado do otimizador de cada etapa
└── runs/                         # logs de treino e amostras de geração
```

Quem tem acesso: `ap211`, `ap212`, `ap205` e `ap206`, como membros do grupo `AP2`.

> **Atenção:** o `AP2` reúne os 24 usuários da turma, então na prática **todos eles** leem e escrevem aqui.
> Não foi possível restringir aos 4: o `/scratch` (NFSv4) recusa ACLs e não há um grupo só nosso.
> Se o administrador criar um grupo, basta `chgrp -R <grupo> /scratch/ap211/slm-shared`.

**Para escrever aqui**, use `umask 002` (por exemplo no `~/.bashrc`). Com o padrão 022, os arquivos novos saem sem escrita para o grupo.
Para consertar: `chmod -R g+rwX /scratch/ap211/slm-shared`. O `slurm/data.sbatch` já define o `umask`.

O diretório fica dentro de `/scratch/ap211/` de propósito: a raiz `/scratch` tem permissão 777 sem sticky bit, e qualquer usuário poderia renomear ou apagar uma pasta criada nela.

## Fora do diretório compartilhado

`/scratch/ap211/slm/` guarda só o que se regenera: `hf-cache` (parquets baixados do FineWeb-Edu, ~25 GB), `uv-cache`, e os dados e `runs` da miniatura (`data/smoke`).
Os caminhos vêm das variáveis `SLM_SCRATCH` (caches) e `SLM_SHARED` (dados persistentes), definidas nos `.sbatch`.

## Formato

Cada conjunto (`train/` e `val/`) é um diretório com:

| Arquivo | Conteúdo |
|---|---|
| `tokens.bin` | `uint16`, forma `(num_records, seq_len + 1)`. Entradas são `[:-1]` e alvos são `[1:]` |
| `mask.bin` | `uint8`, mesma forma, só nos dados de chat: 1 onde o SFT aprende (respostas e o `<|end|>` que as encerra) |
| `meta.json` | número de registros, `seq_len`, arquivos de origem e contagens (ver abaixo) |

Leitura: `slm.data.RecordDataset(diretorio)`, com memmap. O contexto é 1.024, então cada registro tem 1.025 tokens.

Tokenizer: GPT-2 via `tiktoken`, mais `<|user|>`, `<|assistant|>`, `<|end|>` e `<|pad|>` (`slm/tokenizer.py`).
Ele é definido no código, então não há arquivo de tokenizer para guardar.

## Como foram gerados

```bash
sbatch slurm/data.sbatch      # a partir da raiz do repositório; só CPU, sem GPU
```

O job roda os 4 preparos em sequência e pula os que já têm `meta.json`.
Para refazer um conjunto, apague o diretório dele e submeta de novo.
A seed fixa (`--seed`) torna a geração reproduzível.

| Conjunto | Script | Fonte |
|---|---|---|
| `pretrain` | `scripts/tokenize_pretrain.py` | FineWeb-Edu: 12 dumps espaçados no tempo para treino; validação de um dump que o treino não usa, sem URLs repetidas |
| `mid_m`, `mid_c`, `sft` | `scripts/prepare_chat.py` | SmolTalk sem `metamathqa-50k` e `numina-cot-100k`; M e SFT somam 25% de GSM8K treino (repetido 4×) e MetaMathQA |

Vazamento: nenhuma pergunta do GSM8K teste entra no treino. O `meta.json` de cada conjunto de chat registra `removed_gsm_test_leak` e `removed_math_residue`.

## Contagens

*(preenchido após o job de preparo)*
