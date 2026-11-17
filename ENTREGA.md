# Entrega do projeto SLM: o que falta e como usar o modelo

Atualizado em 16/11/2026, à noite. **Prazo: 19/11/2026.** Este arquivo complementa o `PLANO.md` (decisões e histórico) e o `DADOS.md` (onde está cada arquivo).

---

## 1. Onde estamos

| Fase | Situação | Onde está o resultado |
|---|---|---|
| 0–3. Ambiente, código, dados e piloto | ✅ Concluídas | `PLANO.md` |
| 4. Pré-treino (114M parâmetros, 2,5B tokens, 2 GPUs) | ✅ 15–16/11, 5h44, perplexidade final 18,8 | Servidor (`checkpoints/pretrain/`), PC local e Hugging Face |
| 5. Mid-training C (só chat, controle) | ✅ 16/11, servidor, 1h30 | Servidor (`checkpoints/mid_c/`) |
| 5. Mid-training M (chat + 25% de matemática) | ✅ 16/11, RTX 4060 local, com acúmulo de gradiente | PC local e Hugging Face |
| 6. SFT (a partir de M) | ✅ 16/11, RTX 4060 local | PC local e Hugging Face |
| 7. Avaliação | 🟡 Pré-treino, M e SFT avaliados; **falta o C** | PC local (`results/final/*.json`) e Hugging Face (`results/`) |
| 8. Demo, README, slides | ⏳ A fazer | — |

### Resultados até agora (mesma máquina, RTX 4060)

| Modelo | Perplexidade (FineWeb) | HellaSwag | ARC-Easy | Aritmética | GSM8K | Respondeu com número |
|---|---|---|---|---|---|---|
| Pré-treino | 18,77 | 28,1% | 50,8% | 0,7% | 1,9% | 86,4% |
| M | 61,07 | 27,1% | 37,2% | **3,2%** | **4,1%** | 99,4% |
| SFT | 62,65 | 27,4% | 37,3% | 3,1% | 2,8% | 99,5% |
| C | *a avaliar* | | | | | |

- **M contra o pré-treino (McNemar):** aritmética p = 7,4e-6 (de 8 para 35 acertos); GSM8K p = 0,0009. Entre as perguntas respondidas com número, o acerto do GSM8K sobe de 2,2% para 4,1%. **Não é só formato**, diferente do piloto.
- **Regressões:** HellaSwag −1,0 ponto (p = 0,0007) e ARC-Easy −13,6 pontos (p = 2e-34). O C vai dizer se a causa é o chat ou a matemática.
- **SFT contra M:** praticamente igual. GSM8K −1,3 ponto (p = 0,0498, no limite).

---

## 2. Disponibilizar o modelo para o grupo

**Publicado em 16/11:** https://huggingface.co/Xererr/slm-114m (público), com `pretrain/final.pt`, `mid_m/final.pt`, `sft/final.pt` e `results/{pretrain,mid_m,sft}.json`. **Falta enviar:** as curvas (`metrics.jsonl`), tudo do C (depois da tarefa 1) e a página do modelo (`README.md`, a partir de `hf/README.md`). Três destinos, em ordem de importância:

| Destino | Para quê | Quem acessa |
|---|---|---|
| **A. Hugging Face** | Exigido pelo enunciado ("checkpoints no HF ou Drive"); qualquer pessoa baixa com um comando | Grupo e professor |
| **B. Diretório compartilhado do servidor** | Cópia de segurança e avaliação no servidor | Quem tem conta no cluster |
| **C. GitHub** | Os `results/final/*.json` (pequenos), para o README e os slides | Grupo |

### A. Hugging Face (PowerShell, na pasta `repo`)

Pré-requisitos: uma conta no [Hugging Face](https://huggingface.co/join) e um token com permissão de **escrita** ([Settings → Access Tokens](https://huggingface.co/settings/tokens)). O comando `hf` já vem no ambiente do projeto (`huggingface-hub` 2.4.0).

```powershell
uv run hf auth login
```
Cole o token quando ele pedir. Esperado: `Login successful`.

Envio dos checkpoints (~456 MB cada). O primeiro `upload` criou o repositório. **Feitos em 16/11:** pré-treino, M, SFT e `results`.

```powershell
uv run hf upload Xererr/slm-114m ..\shared\checkpoints\pretrain\final.pt pretrain/final.pt
uv run hf upload Xererr/slm-114m ..\shared\checkpoints\mid_m\final.pt mid_m/final.pt
uv run hf upload Xererr/slm-114m ..\shared\checkpoints\sft\final.pt sft/final.pt
uv run hf upload Xererr/slm-114m ..\shared\checkpoints\mid_c\final.pt mid_c/final.pt
uv run hf upload Xererr/slm-114m results\final results
```
- O `mid_c` só existe localmente depois da cópia do servidor (seção 3, item 1).
- **Falta enviar** (mesma pasta `repo`, depois de um `git pull`):
  ```powershell
  uv run hf upload Xererr/slm-114m ..\shared\checkpoints\pretrain\metrics.jsonl pretrain/metrics.jsonl
  uv run hf upload Xererr/slm-114m ..\shared\checkpoints\mid_m\metrics.jsonl mid_m/metrics.jsonl
  uv run hf upload Xererr/slm-114m ..\shared\checkpoints\sft\metrics.jsonl sft/metrics.jsonl
  uv run hf upload Xererr/slm-114m hf\README.md README.md
  ```
  O último cria a página do modelo. Os do C estão na tarefa 1 (seção 3).
- **Público** (decisão de 16/11): o professor e o grupo acessam sem convite.
- Se o envio cair, rode o mesmo comando de novo.
- Conferir: abrir `https://huggingface.co/Xererr/slm-114m` e ver as 4 pastas.

### B. Cópia no servidor (PowerShell, na pasta `SLM`)

Use nomes com `_local` para não se misturar com o que um job do servidor escreveria em `mid_m/` e `sft/`.

No servidor:
```bash
mkdir -p <diretorio-compartilhado>/checkpoints/mid_m_local <diretorio-compartilhado>/checkpoints/sft_local
```
No PC local:
```powershell
scp -J <usuario-entrada>@<servidor-entrada> shared\checkpoints\mid_m\final.pt <usuario-cluster>@<servidor-cluster>:<diretorio-compartilhado>/checkpoints/mid_m_local/
scp -J <usuario-entrada>@<servidor-entrada> shared\checkpoints\sft\final.pt <usuario-cluster>@<servidor-cluster>:<diretorio-compartilhado>/checkpoints/sft_local/
```
Conferir com `Get-FileHash ... -Algorithm SHA256` no PC e `sha256sum ...` no servidor: os códigos têm de ser iguais.

### C. Resultados no GitHub (PowerShell, na pasta `repo`)

```powershell
git pull
git add results/final/*.json
git commit -m "Resultados finais: pre-treino, M e SFT (RTX 4060)"
git push
```
Na primeira vez, o Windows abre o navegador para o login no GitHub.

---

## 3. O que falta

Estimativas de esforço (*estimativa*; entre parênteses, o dobro, por segurança).

| # | Tarefa | Quem | Tempo |
|---|---|---|---|
| 1 | **Avaliar o C** no PC local | Guilherme | 30 min (1h) |
| 2 | **Disponibilizar o modelo** (seção 2): 🟡 checkpoints publicados; faltam curvas, C e página do modelo | Guilherme | 15 min (30 min) |
| 3 | **Exemplos de geração** | Qualquer membro | 30 min (1h) |
| 4 | **Gráficos** das curvas (pré-treino, M, C, SFT) | Orange escreve o script; um membro roda | 1,5h (3h) |
| 5 | **README** reproduzível por etapa + `CREDITS.md` | Orange escreve; grupo revisa | 2h (4h) |
| 6 | **Slides** (10–15 min) | Grupo, com o roteiro da seção 5 | 4h (8h) |
| 7 | **Vídeo de reserva** da demo | Qualquer membro | 30 min (1h) |
| 8 | **PR** para o repositório do grupo e **ensaio** | Grupo | 1,5h (3h) |
| 9 | *Opcional:* exportar em fp16 e medir a velocidade em CPU | Orange escreve; um membro roda | 1,5h (3h) |

### 1. Avaliar o C
1. Copiar `checkpoints/mid_c/final.pt` do servidor para `shared\checkpoints\mid_c\` e conferir o SHA-256 nos dois lados.
2. Copiar também `checkpoints/mid_c/metrics.jsonl` (curva do C, para os gráficos).
3. Rodar de novo `uv run python scripts/run_local.py`, na pasta `repo`. M e SFT são pulados (já existem), só o C é avaliado (~25 min), e o script roda a comparação **`mid_c x mid_m`**.
4. Publicar no Hugging Face:
   ```powershell
   uv run hf upload Xererr/slm-114m ..\shared\checkpoints\mid_c\final.pt mid_c/final.pt
   uv run hf upload Xererr/slm-114m ..\shared\checkpoints\mid_c\metrics.jsonl mid_c/metrics.jsonl
   uv run hf upload Xererr/slm-114m results\final results
   ```
5. Isso fecha o critério principal da Seção 6 do `PLANO.md`: M acima do pré-treino **e** acima de C em aritmética.

### 3. Exemplos de geração
A entrega pede exemplos. Sugestão:
- Usar a demo (seção 4.2) com o **SFT** e com o **M**, as mesmas perguntas nos dois, temperatura 0,7 e top-k 40.
- **Perguntas em inglês** (os dados são em inglês). Misture tipos:
  - `What is 7 + 5?`, `What is 12 times 3?` (aritmética, onde M deveria ir melhor)
  - `Tom has 3 apples and buys 4 more. How many apples does he have?` (problema de texto)
  - `What is photosynthesis?` (conhecimento geral)
  - `Write a short poem about the sea.` (geração livre)
  - `Hi! Who are you?` (formato de conversa e parada)
- Salvar em `results/final/exemplos.md` (pergunta, resposta do M, resposta do SFT, parâmetros).
- **Honestidade:** incluir também respostas ruins. Com 114M parâmetros, o modelo erra fatos e contas com frequência.

### 4. Gráficos
Os dados já existem em `metrics.jsonl` (uma linha JSON por registro, com `step`, `loss`, `val_loss`, `val_ppl` e, no mid-training, `arith_acc`):

| Etapa | Arquivo |
|---|---|
| Pré-treino | `shared/checkpoints/pretrain/metrics.jsonl` (PC local e servidor) |
| M | `shared/checkpoints/mid_m/metrics.jsonl` (PC local) |
| SFT | `shared/checkpoints/sft/metrics.jsonl` (PC local) |
| C | `checkpoints/mid_c/metrics.jsonl` (**só no servidor**: baixar junto com o `final.pt` do C) |

Gráficos sugeridos: (a) loss de treino e de validação do pré-treino, com a fase de decaimento do WSD marcada; (b) `arith_acc` de M e de C ao longo do mid-training, no mesmo gráfico; (c) loss de validação do SFT (para mostrar que não houve overfitting).

### 5. README e créditos
- **README:** o que é o projeto; como instalar (`uv sync`); como baixar o modelo; um comando por etapa (preparar dados, pré-treino, mid, SFT, avaliação, demo); tabela de resultados; links para o Hugging Face e os slides.
- **Página do modelo no Hugging Face:** escrita em `hf/README.md` (16/11); falta enviar (seção 2). Atualizar a linha do C na tabela depois da tarefa 1.
- **CREDITS.md:** o enunciado exige crédito a código de terceiros adaptado. Listar as referências da Seção "Referências" do `PLANO.md` que inspiraram código (por exemplo, nanochat e SmolLM3) e o uso de IA na escrita do código.

### 6. Slides: ver a seção 5.

### 8. PR e ensaio
- PRs #1 (piloto e pré-treino) e #2 (C) já aceitos. PR final da `midtraining-m`: a abrir em https://github.com/FHaggs/slm/compare/main...Xererr:slm-scherer:midtraining-m. Commits novos nesta branch (por exemplo, o resultado do C) entram no PR automaticamente enquanto ele estiver aberto.
- A `mid-sft` é uma versão antiga, só para o servidor: **não abrir PR dela**.
- **Ensaio cronometrado:** 10–15 min, com a demo ao vivo. Todo o grupo precisa saber explicar o código (haverá perguntas; ver seção 6).

### Pontos para conferir no enunciado antes de entregar
- **Tokens do mid-training (exigência: 100M–300M):** M processou 374M tokens e C 360M, contando o preenchimento (*padding*). Os tokens **úteis** são ~294M (M) e ~292M (C), dentro da faixa. Explicar isso nos slides ou no README.
- **M e SFT em GPU diferente do pré-treino e do C:** documentado no `PLANO.md`. O acúmulo de gradiente reproduz o lote de 24 do servidor e foi validado por testes de equivalência (`tests/test_grad_accum.py`).

### Arrumação
- Cancelar o job 1175 no servidor (`scancel 1175`), se o grupo usar o M e o SFT locais como oficiais: ele repetiria um trabalho já feito.
- Depois da entrega: gerar tokens novos (GitHub, Notion, Hugging Face) e apagar as branches antigas (`pilot`, `mid-sft`) que já tiverem sido incorporadas.

---

## 4. Tutorial: como usar o modelo

### 4.1. Instalar (uma vez)
Requisitos: `git` e o [`uv`](https://docs.astral.sh/uv/). Testado no Linux do servidor e no Windows 11; **não testado no macOS**.

```bash
git clone https://github.com/Xererr/slm-scherer.git
cd slm-scherer
git checkout midtraining-m
uv sync
```
Depois que o PR final for aceito, o mesmo código fica na branch `main` de https://github.com/FHaggs/slm.
O `uv sync` instala o Python 3.12 e o PyTorch 2.11 com CUDA 12.8 (~3 GB). Sem GPU NVIDIA, o mesmo pacote roda em CPU, só que mais devagar.

Baixar o modelo do Hugging Face:
```bash
uv run hf download Xererr/slm-114m sft/final.pt mid_m/final.pt --local-dir modelos
```
Os arquivos ficam em `modelos/sft/final.pt` e `modelos/mid_m/final.pt`.

### 4.2. Conversar com o modelo (demo)
PowerShell:
```powershell
$env:SLM_CHECKPOINT = "modelos\sft\final.pt"
uv run streamlit run app/chat.py
```
Linux ou macOS:
```bash
SLM_CHECKPOINT=modelos/sft/final.pt uv run streamlit run app/chat.py
```
O navegador abre em `http://localhost:8501`. Na barra lateral:

| Controle | O que faz | Sugestão |
|---|---|---|
| Checkpoint | Caminho do modelo. Troque por `modelos/mid_m/final.pt` para comparar | SFT |
| Temperatura | 0 = sempre a palavra mais provável (repetitivo); perto de 1,5 = muito aleatório | 0,7 |
| Top-k | Só sorteia entre as k palavras mais prováveis | 40 |
| Máximo de tokens | Limite do tamanho da resposta (o modelo também para sozinho no `<|end|>`) | 200 |
| Limpar conversa | Começa uma conversa nova | — |

**O que esperar:** responde no formato de conversa e para sozinho, mas é um modelo pequeno: erra fatos e contas com frequência. Escreva em **inglês**. Conversas longas passam do contexto de 1.024 tokens; use "Limpar conversa".

### 4.3. Usar no Python
Rodar da raiz do repositório com `uv run python seu_script.py`. Este trecho foi testado em 16/11:

```python
import torch
from slm.checkpoint import load_model
from slm.chat import render_prompt
from slm.generate import stream
from slm.tokenizer import END, ENDOFTEXT

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model("modelos/sft/final.pt", device)
print(f"{model.num_params() / 1e6:.1f}M parametros em {device}")

messages = [{"role": "user", "content": "What is 7 + 5?"}]
prompt = render_prompt(messages, tokenizer)  # termina em <|assistant|>
tokens = list(stream(model, prompt, max_new_tokens=200, temperature=0.7, top_k=40,
                     stop_tokens=frozenset({END, ENDOFTEXT})))
print(tokenizer.decode(tokens))
```
- Para continuar a conversa, acrescente `{"role": "assistant", "content": <resposta>}` e a próxima pergunta em `messages`.
- O template é `<|endoftext|> <|user|> ... <|end|> <|assistant|> ... <|end|>`. Mensagens `system` entram no começo da primeira fala do usuário.

### 4.4. Avaliar um checkpoint e comparar dois
```bash
uv run python scripts/evaluate.py --checkpoint modelos/sft/final.pt --output results/sft.json --val-dir <dados>/pretrain/val --benchmarks hellaswag arc_easy
uv run python scripts/compare.py results/mid_m.json results/sft.json
```
- Sem `--val-dir`, a perplexidade é pulada (os dados de validação não estão no Hugging Face).
- Em GPU de 8 GB, acrescente `--ppl-batch-size 4 --gsm-batch-size 8`.
- Cada avaliação leva ~25 min numa RTX 4060. Para um teste rápido: `--mc-limit 200 --gsm-limit 100`.
- O `compare.py` faz o teste de McNemar item a item: `so A` e `so B` contam os itens que só um dos dois acertou; p < 0,05 indica diferença real.

---

## 5. Roteiro da apresentação (10–15 min)

| # | Slide | Conteúdo | Tempo |
|---|---|---|---|
| 1 | Objetivo e pipeline | Pré-treino → mid-training → SFT. Hipótese: treinar com matemática melhora a matemática | 1 min |
| 2 | Arquitetura | 114M parâmetros; GQA, RoPE, SwiGLU, RMSNorm; contexto de 1.024 | 1,5 min |
| 3 | Dados e Chinchilla | 2,5B tokens (~22 × N); FineWeb-Edu, SmolTalk, matemática | 1,5 min |
| 4 | Infraestrutura | DDP em 2 GPUs (122 mil tok/s, 89% do ideal); fila compartilhada; acúmulo de gradiente para rodar M e SFT numa RTX 4060, validado por testes | 2 min |
| 5 | Curvas | Pré-treino (perplexidade 18,8, efeito do decaimento do WSD); aritmética de M e de C ao longo do mid | 1,5 min |
| 6 | Desenho do experimento | M (chat + matemática) contra C (só chat): por que um grupo de controle | 1 min |
| 7 | Resultados | Tabela dos 4 modelos e McNemar | 2 min |
| 8 | Discussão | Ganho real em aritmética e GSM8K; queda no HellaSwag e no ARC-Easy; o que o C mostra | 1,5 min |
| 9 | Demo ao vivo | SFT e M nas mesmas perguntas (vídeo de reserva) | 2 min |
| 10 | Conclusões e limitações | Modelo pequeno, uma execução por configuração, GPUs diferentes no mid | 0,5 min |

Um resultado negativo, bem explicado, também conta (`PLANO.md`, Seção 6).

---

## 6. Perguntas prováveis da banca

| Pergunta | Resposta curta |
|---|---|
| Por que 114M parâmetros e 2,5B tokens? | Chinchilla: ~20 tokens por parâmetro. 2,5B / 114M ≈ 22 |
| O que é o WSD? | LR sobe no aquecimento (*warmup*), fica estável e cai até zero nos 10% finais. O efeito da queda aparece na curva de loss do pré-treino (medir no gráfico da tarefa 4) |
| Para que serve o C? | Separar o efeito da matemática do efeito do formato de chat. Sem ele, não dá para atribuir o ganho à matemática |
| O que é o teste de McNemar? | Compara dois modelos nos **mesmos** itens, contando só os que um acertou e o outro errou. Adequado para dados pareados |
| Como M e SFT rodaram numa GPU de 8 GB? | Acúmulo de gradiente: 8 micro-lotes de 3 somados num passo de 24, com a loss ponderada pelos tokens válidos. Testes mostram que é equivalente a um lote único de 24 |
| Por que o ARC-Easy caiu? | O mid-training em chat muda o formato que o modelo espera. No piloto, a queda apareceu em M e em C (a confirmar no modelo final com o C) |
| O SFT melhorou alguma coisa? | Nos benchmarks, quase nada. O papel dele é o formato: responder como assistente e parar no lugar certo, o que se vê na demo |
| Como o pré-treino retoma sem repetir dados? | O checkpoint guarda a posição nos dados; a ordem é uma permutação fixa por semente; a LR depende dos tokens vistos |
