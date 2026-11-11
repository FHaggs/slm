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

**Piloto (Fase 3):** só **lê** do diretório compartilhado. Grava os runs em `/scratch/$USER/slm/runs/pilot/{pretrain,mid_m,mid_c}` (checkpoints e `metrics.jsonl`, por usuário), o cache dos benchmarks em `/scratch/$USER/slm/hf-cache` e os resultados em `results/pilot/*.json` no clone. O `slurm/pilot.sbatch` aborta se `SLM_SCRATCH` estiver dentro de `SLM_SHARED`.
Execução de 11/11 (job 1140): os `final.pt` de pré-treino, M e C ficaram em `/scratch/ap205/slm/runs/pilot/`, e os resultados em `results/pilot/{pretrain,mid_m,mid_c}.json` no clone do `ap205`, ainda não versionados.

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

Gerado em 2026-11-09 pelo job 1134 (43 min, pico de 9,8 GB de RAM). Cada registro tem 1.025 tokens.
"Tokens úteis" são os que não são `<|pad|>`: `fill_rate × registros × 1.025`.

| Conjunto | Registros | Tokens úteis | Origem do orçamento |
|---|---|---|---|
| `pretrain/train` | 2.439.024 | 2.499.999.600 (sem padding) | 12 arquivos, 2.457.600 documentos |
| `pretrain/val` | 48.780 | 49.999.500 | dump `CC-MAIN-2025-26`, que o treino não usa |
| `mid_m/train` | 365.386 | ~293,7M (78,4%) | 225,0M de chat + 75,0M de matemática |
| `mid_c/train` | 352.035 | ~291,7M (80,8%) | 300,0M de chat |
| `sft/train` | 121.663 | ~97,9M (78,5%) | 75,0M de chat + 25,0M de matemática |
| `mid_m/val`, `mid_c/val` | 5.856 | ~4,9M | chat do split `test` do SmolTalk |
| `sft/val` | 5.863 | ~4,9M | idem, com outra seed |

- **Padding:** com contexto de 1.024, 19% a 22% de cada registro de chat é `<|pad|>` (a miniatura, com contexto de 256, tinha ~20%).
  Os alvos em `<|pad|>` viram `-100`, então o treino por `epochs` cobre cada conversa uma vez (M: 365.386 registros por época; C: 352.035).
  Os orçamentos de 300M e 100M contam os tokens antes do empacotamento, por isso os tokens úteis ficam ~2% abaixo.
- **M contra C:** têm quase o mesmo número de tokens úteis (293,7M e 291,7M). A diferença entre eles é a composição, não o tamanho.
- **Validação do chat:** `mid_m/val` e `mid_c/val` são idênticos (só chat), então a perplexidade dos dois runs é comparável.
- **Sobreposição:** `mid_*` e `sft` amostram o mesmo SmolTalk sem exclusão mútua (seeds 0 e 1), então parte das conversas do SFT já apareceu no mid-training.
- **Matemática:** GSM8K treino repetido 4× mais MetaMathQA. A `numina-cot` e as 5.558 conversas com `\boxed` ou `####` ficaram de fora.

## Conferências feitas

| Conferência | Resultado |
|---|---|
| Vazamento do GSM8K teste | **0** nos três conjuntos de chat. O `meta.json` diz `removed_gsm_test_leak: 0`, e uma busca independente nos tokens gravados confirmou: 0 das 1.319 perguntas em `mid_m`, `mid_c` e `sft` |
| Controle da busca | O GSM8K treino aparece 91.211 vezes em `mid_m` (7.472 das 7.473 perguntas) e 30.575 em `sft`. Se a busca estivesse errada, o controle seria 0 |
| Limite da busca | Só pega correspondência exata. Paráfrases, fora o `original_question` do MetaMathQA, não são detectadas |
| Decodificação | Amostras de cada conjunto decodificadas e lidas. Nenhum token acima de 50.256 no pré-treino |
| Máscara do chat | Em ~20 mil registros de cada conjunto: nenhum token `<|pad|>`, `<|user|>` ou `<|assistant|>` com máscara 1. Só entram as respostas e o `<|end|>` que as encerra |
| Validação sem treino | Val do pré-treino vem de `CC-MAIN-2025-26`, e os dumps de treino vão de 2013 a 2025-21 |
