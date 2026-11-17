---
language: en
tags:
  - small-language-model
  - pytorch
  - from-scratch
datasets:
  - HuggingFaceFW/fineweb-edu
  - HuggingFaceTB/smoltalk
  - openai/gsm8k
  - meta-math/MetaMathQA
---

# SLM 114M: pré-treino, mid-training e SFT do zero

Modelo de linguagem pequeno (114.117.120 parâmetros) treinado do zero num trabalho de disciplina (PUCRS, Aprendizado Profundo II, 2026). O objetivo do experimento: verificar se um mid-training com matemática melhora a matemática do modelo, usando um grupo de controle sem matemática.

Código, dados e decisões: https://github.com/Xererr/slm-scherer (branch `midtraining-m`; `PLANO.md`, `DADOS.md` e `ENTREGA.md`).

> Os arquivos `.pt` são checkpoints de PyTorch para o código do projeto. **Não** são carregáveis com `transformers`.

## Arquivos

| Caminho | Conteúdo |
|---|---|
| `pretrain/final.pt` | Pré-treino (FineWeb-Edu) |
| `mid_m/final.pt` | Mid-training **M**: chat + 25% de matemática |
| `mid_c/final.pt` | Mid-training **C** (controle): só chat. *A publicar* |
| `sft/final.pt` | SFT a partir de M. **Use este para conversar** |
| `results/*.json` | Avaliações item a item (usadas no teste de McNemar) |
| `*/metrics.jsonl` | Curvas de treino (loss, LR, validação, aritmética). *A publicar* |

## Arquitetura

Transformer decoder-only: 12 camadas, `d_model` 768, 12 cabeças de atenção com 4 cabeças de chave/valor (GQA), FFN SwiGLU de 2.048, RMSNorm, RoPE, contexto de 1.024 tokens. Tokenizer BPE do GPT-2 (tiktoken) com 5 tokens especiais de chat (vocabulário de 50.261).

Template: `<|endoftext|> <|user|> ... <|end|> <|assistant|> ... <|end|>`.

## Treino

| Etapa | Dados | Tokens | Lote | LR | Hardware |
|---|---|---|---|---|---|
| Pré-treino | FineWeb-Edu | 2.497.560.576 (1 época, ~22 × N, Chinchilla) | 48 × 1.024 | 3e-4, WSD (warmup de 100M tokens, decaimento nos 10% finais) | 2× RTX A5000, DDP, 5h44 |
| Mid M | SmolTalk + 25% de matemática (GSM8K treino e MetaMathQA) | ~293,7M úteis (1 época) | 24 × 1.024 (3 × 8 com acúmulo de gradiente) | 3e-4 | 1× RTX 4060 |
| Mid C | SmolTalk | ~291,7M úteis (1 época) | 24 × 1.024 | 3e-4 | 1× RTX A5000, 1h30 |
| SFT | SmolTalk + 25% de matemática, loss só nas respostas | ~97,9M úteis (1 época) | 24 × 1.024 (3 × 8) | 2e-5 | 1× RTX 4060 |

Mid-training e SFT partem só dos pesos da etapa anterior (otimizador zerado). Nenhuma pergunta do GSM8K teste entra no treino. O acúmulo de gradiente pondera a loss pelos tokens válidos do passo inteiro; testes de equivalência no repositório mostram que ele reproduz um lote único de 24.

## Resultados

Avaliados na mesma máquina (RTX 4060). Perplexidade em 5,12M tokens de validação do FineWeb-Edu (um dump que o treino não usa); HellaSwag e ARC-Easy por log-likelihood; aritmética em 1.107 contas (soma, subtração e multiplicação, 1 e 2 dígitos); GSM8K com 3 exemplos no prompt.

| Modelo | Perplexidade | HellaSwag | ARC-Easy | Aritmética | GSM8K |
|---|---|---|---|---|---|
| Pré-treino | 18,77 | 28,1% | 50,8% | 0,7% | 1,9% |
| M | 61,07 | 27,1% | 37,2% | 3,2% | 4,1% |
| SFT | 62,65 | 27,4% | 37,3% | 3,1% | 2,8% |
| C | *a avaliar* | | | | |

- M contra o pré-treino (teste de McNemar): aritmética p = 7,4e-6; GSM8K p = 0,0009. HellaSwag e ARC-Easy caem de forma significativa.
- SFT contra M: diferenças pequenas (GSM8K p = 0,0498; HellaSwag p = 0,047).

## Como usar

```bash
git clone https://github.com/Xererr/slm-scherer.git
cd slm-scherer
git checkout midtraining-m
uv sync
uv run hf download Xererr/slm-114m sft/final.pt --local-dir modelos
SLM_CHECKPOINT=modelos/sft/final.pt uv run streamlit run app/chat.py
```

No Python, a partir da raiz do repositório:

```python
import torch
from slm.checkpoint import load_model
from slm.chat import render_prompt
from slm.generate import stream
from slm.tokenizer import END, ENDOFTEXT

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model("modelos/sft/final.pt", device)
prompt = render_prompt([{"role": "user", "content": "What is 7 + 5?"}], tokenizer)
tokens = list(stream(model, prompt, max_new_tokens=200, temperature=0.7, top_k=40,
                     stop_tokens=frozenset({END, ENDOFTEXT})))
print(tokenizer.decode(tokens))
```

## Limitações

- Modelo pequeno, só em inglês: erra fatos e contas com frequência.
- Uma execução por configuração (sem várias sementes).
- M e SFT foram treinados numa GPU diferente do pré-treino e do C (mesmo lote efetivo e mesmos hiperparâmetros).
- O mid-training em chat piora o conhecimento geral medido (perplexidade no FineWeb, HellaSwag e ARC-Easy).
- Licença ainda não definida pelo grupo.

Código escrito com auxílio de IA, revisado pelo grupo. Créditos a código e trabalhos de terceiros no repositório.
