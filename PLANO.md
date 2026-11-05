# Plano do Projeto 2: Small Language Model

Pipeline: **Pré-treino (FineWeb-Edu) → Mid-training (chat + matemática, com benchmark antes e depois) → SFT → Avaliação & Demo**

Atualizado em 05/11/2026. **Prazo: 19/11/2026** (14 dias). Os itens marcados como *estimativa* são validados na Fase 0 ou na Fase 3.

**Decisões fechadas**
- Modelo de ~114M parâmetros, com 2,5B tokens no pré-treino.
- Benchmark simples de matemática, rodado antes e depois do mid-training.
- Uso de IA no código é permitido; código de terceiros adaptado continua precisando de crédito.
- **Princípio:** nada complicado entra sem evidência de que funciona. Começamos pelo simples, e cada adição precisa de um teste.

---

## 1. Calendário

| Datas | Etapa | Marco |
|---|---|---|
| Qui 05 – Sex 06 | Fases 0 e 1: ambiente, benchmark, núcleo de treino, smoke tests | Pipeline minúsculo de ponta a ponta |
| Sex 06 – Sáb 07 | Fase 2: dados e tokenizer | Memmaps prontos |
| Sáb 07 – Dom 08 | Fase 3: piloto com antes/depois em miniatura | **Decisão: manter a matemática ou trocar de tarefa** |
| Seg 09 – Qua 11 | Fase 4: pré-treino final (~6h + fila) | Checkpoint publicado |
| Qua 11 – Qui 12 | Fase 5: mid-training | Veredito da matemática |
| Sex 13 | Fase 6: SFT | Checkpoint final |
| Sáb 14 – Dom 15 | Fases 7 e 8: avaliação final e demo | Tabelas e demo funcionando |
| Seg 16 – Ter 17 | Fase 8: README, checkpoints no HF, slides, ensaio | Entrega pronta |
| Qua 18 | **Folga** | |
| Qui 19 | Entrega | |

**Limites para acionar o plano B**
- **Sem as 2 GPUs livres até Qui 12:** rodar o pré-treino em 1 GPU (~11h), sem esperar a outra.
- **Pré-treino sem resultado até Sáb 14:** usar o último checkpoint salvo, documentando o total de tokens treinados.

---

## 2. O que o enunciado exige

| Item | Exigência |
|---|---|
| Modelo | 50M–200M parâmetros treináveis; fora disso, justificar |
| Pré-treino | Tokens por Chinchilla (≈ 20×N); DDP com 2 GPUs; curva de loss; checkpoints retomáveis |
| Mid-training | Só os pesos do pré-treino; LR 1e-4 a 3e-4; 100M–300M tokens; template de chat; sequence packing |
| SFT | Parte do checkpoint do mid; máscara `-100` fora das respostas do assistente; LR 1e-5 a 3e-5; 1–2 épocas |
| Avaliação | Perplexidade em split de validação do pré-treino, **ao longo do treino**; ≥ 1 benchmark de múltipla escolha por log-likelihood |
| Demo | Streamlit com checkpoint SFT; temperatura, top-k e máx. de tokens; ao vivo na apresentação |
| Entrega | GitHub com README reproduzível; checkpoints no HF ou Drive; curvas das 3 etapas; resultados; exemplos de geração; slides |
| Regras | Grupo de 3–5; código de terceiros adaptado creditado; −10%/dia de atraso |

---

## 3. Infraestrutura (medida no cluster)

| Recurso | Valor | Consequência |
|---|---|---|
| Login `CDIA` | 2 CPUs, 3,8 GB, já usando swap | Só editar e submeter; **nada pesado aqui** |
| Nó `CDIA01` | 2× **RTX A5000 24 GB**, 96 threads (Xeon 5318Y), 251 GB RAM | Único nó com GPU |
| Fila | Slurm, partição `LAD_geral`, `priority/basic` (FIFO) + backfill, sem preempção, sem limite de tempo; **máximo de 1 job por usuário, somando os que rodam e os da fila** | Etapas rodam uma de cada vez; não dá para encadear jobs nem preparar dados durante um treino; `--time` realista ajuda o backfill |
| `/home` | NFS, 1,4 TB livres e compartilhados; escrita de 94 MB/s | Código, logs e resultados; não dataset |
| `/scratch` | Disco local de 37 TB, 35 TB livres; escrita de 547 MB/s, leitura de 907 MB/s | Dados e checkpoints em `/scratch/ap211/slm`; backup desconhecido |
| Rede no nó | HF e PyPI acessíveis | Download direto em job |
| Python | Ambiente com `uv`: Python 3.12, PyTorch 2.11 (cu128), bf16 e Flash Attention confirmados | `sbatch slurm/setup.sbatch` recria o ambiente |

**Benchmark medido em 1 GPU** (modelo de 114M, contexto de 1.024, tokens aleatórios, sem leitura de dados, avaliação nem checkpoints):

| Batch | Compilado | Tokens/s | Pico de memória |
|---|---|---|---|
| 16 | Não | 43.105 | 20,0 GB |
| 16 | Sim | 67.095 | 12,1 GB |
| **24 (escolhido)** | **Sim** | **69.082** | **17,4 GB** |
| 32 | Sim | 69.767 | 22,6 GB |
| 48 e 64 | Sim | Sem memória | — |

- **`torch.compile` é indispensável:** ganho de ~55% em velocidade, com menos memória.
- **Exige `TORCHINDUCTOR_SHAPE_PADDING=0`:** a compilação falhou por falta de memória no run com a opção padrão, em um teste interno de padding.
- **Quem limita a memória são os logits:** batch × contexto × vocabulário ocupa 6,6 GB em fp32 só para o batch de 32, por causa do vocabulário de 50k.
- **Batch de 24 por GPU:** praticamente a mesma velocidade do batch de 32, com ~7 GB de folga para fragmentação.
- **Não medido:** velocidade em 2 GPUs, que será vista no início do pré-treino. As estimativas abaixo supõem ganho quase linear.

**Template de job.** Use `--mem` explícito: sem ele, o padrão observado reserva ~128 GB por GPU. Evite `--exclusive`.

```bash
#!/bin/bash
#SBATCH -J slm-pretrain
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH --no-requeue
#SBATCH -o logs/%x.%j.out
#SBATCH -D /home/ap211/slm

source .venv/bin/activate
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_SHAPE_PADDING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node=2 scripts/pretrain.py \
    --config configs/final/pretrain.yaml --max-hours 23.5
```

- **Corte por tempo:** o rank 0 decide, com `--max-hours`, salvar e sair antes do limite. É mais simples que sinais, porque o `torchrun` não repassa `SIGUSR1`.
- **Retomada automática:** repetir o mesmo comando continua do último checkpoint, mesmo com outro número de GPUs.
- **Runs de 1 GPU** (pilotos, mid, SFT) usam `--gres=gpu:1`, sem esperar as duas.

---

## 4. Arquitetura

| Hiperparâmetro | Valor |
|---|---|
| Vocabulário | ~50.260 (BPE do GPT-2 + tokens de chat) |
| `d_model` / camadas | 768 / 12 (mesma escala do `d12` do nanochat) |
| Cabeças | 12 de query, 4 de key/value (GQA) |
| FFN | SwiGLU, oculto ≈ 8/3 × d (2.048) |
| Contexto | 1.024 |
| **Total** | **≈ 114M** (75,5M em blocos, 38,6M em embeddings) |

**Chinchilla:** o mínimo de 20×N dá ~2,3B tokens. Usamos 2,5B (~22×N). O nanochat relata ótimo perto de 10×N no setup dele, então não há motivo para ir muito além do mínimo pedido no enunciado. Tempo estimado: ~6h em 2 GPUs, ou ~11h em 1 GPU, partindo de 69 mil tokens/s por GPU e somando uma folga para dados, avaliação e checkpoints.

| Componente | Escolha | Justificativa |
|---|---|---|
| Base | Decoder-only, pre-norm, sem bias | Padrão de LLaMA, SmolLM e nanochat |
| Posição | **RoPE** | Sem parâmetros e padrão atual |
| Atenção | **GQA 12/4**, via `scaled_dot_product_attention` (Flash) | O SmolLM3 usa 4 grupos e relata desempenho igual ao de MHA; aqui economiza ~9,4M parâmetros |
| Ativação | **SwiGLU** | Padrão da linha LLaMA |
| Normalização | **RMSNorm** | Mais barata que LayerNorm; citada como diferencial |
| Embeddings | **Tied**, sem weight decay | Evita gastar mais 38,6M na saída; SmolLM3 faz o mesmo |
| Tokenizer | **BPE do GPT-2** via `tiktoken` | Ver abaixo |
| Otimizador | **AdamW** (WD=0,1, grad clip 1) | Simples e usado no SmolLM3. O Muon do nanochat depende de vários ajustes; não dá para validá-lo no prazo |
| Schedule | **WSD**: warmup de ~2.000 steps, decaimento linear nos 10% finais | Usado no SmolLM3 |
| Precisão | bf16 + `torch.compile` + DDP | Confirmar suporte a bf16 na Fase 0 |

**Tokenizer**
- **Padrão:** GPT-2 sem treinar nada. É a primeira opção listada no enunciado e não traz risco de defeito no tokenizer.
- **Tokens especiais:** `<|endoftext|>` separa documentos; `<|user|>`, `<|assistant|>`, `<|end|>` e `<|pad|>` entram como IDs novos.
- **Dígitos:** separar os dígitos é uma hipótese, não uma medição. O nanochat usa blocos de até 2 dígitos, escolhidos para economizar vocabulário, não por testes de aritmética. A variante só entra se ganhar no piloto (Fase 3).
- **BPE próprio de 32k:** é o que o nanochat faz, e economizaria ~14M de parâmetros. Fica de fora porque o ganho não compensa o trabalho e o risco neste prazo.
- **Trocar o tokenizer depois obrigaria a refazer o pré-treino**, então a decisão fecha no fim do piloto.

**Não vamos adotar:** Muon, value embeddings, janela deslizante, QK-norm, escalares de resíduo, NoPE, YaRN, MoE, Mamba, intra-document masking, RL e ferramenta de calculadora. No nanochat, parte desses recursos ajudou, mas com um sweep de 320 runs que não temos como repetir, e vários outros testes não deram ganho.

---

## 5. Dados

### Pré-treino: FineWeb-Edu

O dataset tem 1,53B documentos e **4,5 TB de parquet: não baixar tudo**.
- **Amostragem:** ~12 arquivos de dumps de anos diferentes (~25 GB). Cada arquivo rende ~600M tokens (*estimativa*). Tokenizar, embaralhar por documento e cortar em 2,5B.
- **Validação:** ~50M tokens de um **dump não usado no treino**, excluindo URLs repetidas.
- **Formato:** `uint16` memmap em `/scratch` (~5 GB).
- **Sampler determinístico:** permutação fixa por seed, sem reposição, com cursor salvo no checkpoint. A retomada fica exata, e a contagem de tokens também.

### Mid-training

`SmolTalk/all` é a união exata de 13 configs, e **duas são de matemática: `metamathqa-50k` e `numina-cot-100k` (14,7% das linhas)**. Por isso:

| Run | Composição (300M tokens) |
|---|---|
| **M** (principal) | 75% SmolTalk sem as 2 configs + 25% matemática (GSM8K treino repetido + MetaMathQA) |
| **C** (controle) | 100% SmolTalk sem as 2 configs, mais um filtro simples de resíduos (`\boxed`, `####`) |

A `numina-cot` fica de fora: tem problemas de olimpíada, muito acima do nível deste modelo. Não há aritmética sintética no treino, para o benchmark não testar exatamente o que foi treinado.

O preparo conta tokens de conversa, e os registros ainda têm padding (~20% na miniatura, com contexto de 256). Na Fase 2, medir essa fração com contexto de 1.024 e definir o treino por `epochs`, para cobrir cada conversa uma vez.

---

## 6. Benchmark de matemática: antes e depois

**Pergunta:** o mid-training com matemática melhorou o modelo em matemática?

**Por que não só antes/depois:** o mid-training de chat também pode mudar o resultado, então a melhora sozinha não prova que a causa foi a matemática. O run **C** resolve isso reutilizando o mesmo código, e custa só ~1,5h em uma GPU (*estimativa*). Se faltar tempo, ele é o primeiro corte.

**Sobre o mid-training:** o nanochat removeu essa etapa depois de mudar o carregamento de dados, mas o enunciado a exige, então ela continua. Isso reforça a importância do controle: parte de uma melhora pode vir só da adaptação ao novo formato.

### Os testes

Os dois usam **o mesmo prompt em texto simples em todos os checkpoints**. O modelo do pré-treino não conhece o template de chat, e usar formatos diferentes distorceria a comparação.

| Teste | Exemplo | Nota |
|---|---|---|
| **Aritmética** (principal), 6 células: +, −, × com 1 e 2 dígitos, seed fixa. 1 dígito: todas as combinações (81 ou 45 itens); 2 dígitos: 300 itens | `17 + 25 =` | Número gerado exatamente igual ao correto |
| **GSM8K teste** (n=1.319), com 3 exemplos fixos do GSM8K treino no prompt | Problema de texto | Último número da resposta, comparado numericamente; "sem resposta" é erro |

Expectativa: acerto baixo no GSM8K. Reproduções do nanochat, com modelos muito maiores, relatam [~3% no mid e ~6% no SFT](https://objectgraph.com/blog/nanochat-dgx-spark/). Por isso a aritmética é o teste principal e o GSM8K mede se houve transferência para raciocínio.

### Quando rodar

- **Durante o mid-training:** só a aritmética, a cada N steps, para gerar uma curva.
- **Nos checkpoints finais:** pré-treino, C, M e SFT. Isso produz a tabela da apresentação.

### Critério de "funcionou"

1. **Principal:** M acerta mais aritmética que o pré-treino **e** que C, com McNemar pareado, p < 0,05.
2. **Transferência:** reportar o GSM8K com IC de 95%. Se M só subir na aritmética, o modelo aprendeu a *calcular*, não a *raciocinar*.
3. **Sem regressão:** HellaSwag de M no máximo 1 ponto percentual abaixo de C.

Um resultado negativo também é um resultado válido para a apresentação.

### Vazamento

Feito dentro de `scripts/prepare_chat.py`: normalizar o texto e buscar as perguntas do GSM8K teste (correspondência exata) em toda a matemática de treino. No MetaMathQA, comparar também a coluna `original_question`, que guarda a pergunta antes da paráfrase. Remover e reportar os casos. Fora do MetaMathQA, a checagem não pega paráfrases.

### Decisão no piloto (Dom 08)

Rodar o antes/depois em miniatura com o modelo piloto. **Trocar de tarefa só se** a aritmética não se mexer em M comparado ao pré-treino, ou se o preparo dos dados travar.

**Tarefa reserva: ciências.** Treinar com SciQ e testar com ARC-Easy, por log-likelihood. O benchmark ARC é um dos aceitos pelo enunciado e funciona igual antes e depois. O resto do pipeline não muda. Limitação: o SciQ tem só 11.679 linhas de treino, então a dose será bem menor que os 25% da matemática, mesmo repetindo os dados.

---

## 7. Plano de passos

### Fase 0: Ambiente e calibração
- [x] Criar o repositório GitHub; instalar `uv`, Python e PyTorch; criar `/scratch/ap211/slm`
- [x] Benchmark em 1 GPU: tokens/s, maior batch, velocidade do `/scratch`

**Saída:** bf16 confirmado e estimativa recalculada. Cumprido. A velocidade em 2 GPUs será observada nos primeiros minutos do pré-treino, que também será o primeiro uso do DDP em 2 GPUs.

### Fase 1: Núcleo e smoke tests
- [x] `model.py` com contagem de parâmetros; núcleo de treino (DDP, bf16, WSD, grad clip, checkpoint, retomada, log)
- [x] Loss inicial ≈ 10,98 (medido: ln(50261) ≈ 10,82, mais ~0,15 pela variância dos logits na inicialização); overfit de um mini-batch até loss perto de 0
- [x] Interromper e retomar: pesos finais idênticos aos de um treino sem interrupção
- [x] Testes de máscara do SFT e do alinhamento dos alvos
- [x] Pipeline completo com dados reais e modelo de 11M: pré → mid → SFT → eval, em 4 minutos; demo testada com o checkpoint

**Saída:** cumprida; 44 testes automatizados passando (`sbatch slurm/tests.sbatch`).

**Limites do que foi validado**
- O DDP foi testado com 2 processos em CPU, não em 2 GPUs. O primeiro uso em GPU será o pré-treino.
- A geração e os benchmarks da miniatura não dizem nada sobre o modelo final: 11M e 20M de tokens são pouco para qualquer conclusão.
- A demo foi testada sem navegador e usando CPU. Falta rodar em notebook.

### Fase 2: Dados (jobs só de CPU)
- [ ] Baixar os ~12 arquivos do FineWeb-Edu; gerar memmaps com o GPT-2
- [ ] Preparar M, C e o dataset do SFT; conferir a contagem de vazamento no `meta.json`

**Saída:** contagem de tokens de cada fonte registrada; decodificação conferida manualmente.

### Fase 3: Piloto
Modelo de ~32M (d=384, L=8) em 500M tokens: ~1h por run em 1 GPU (medido: 184 mil tokens/s, mais folga). O objetivo é validar o pipeline, não caçar otimizações.
- [ ] Run base com GPT-2 + antes/depois de matemática em miniatura
- [ ] Um único teste extra: GPT-2 com dígitos separados (~10 linhas). Adotar só se a aritmética melhorar com McNemar p < 0,05

**Saída:** decisão sobre a matemática e o tokenizer. Ressalva: como alerta o próprio autor do nanochat, resultado em escala pequena pode não se transferir para o modelo final.

### Fase 4: Pré-treino final (~6h em 2 GPUs, batch de 24 por GPU)
- [ ] Checkpoint a cada ~30 min, mantendo os 2 últimos + marcos
- [ ] Logar loss e perplexidade de validação; guardar amostras de geração

**Saída:** curvas sem anomalias; texto coerente; HellaSwag acima do acaso. **Publicar o checkpoint** no HF.

### Fase 5: Mid-training
- [ ] Pesos do pré-treino, otimizador zerado, LR 1e-4 a 3e-4, warmup curto, 300M tokens
- [ ] Rodar M e C em sequência, no mesmo job (limite de 1 job por usuário)
- [ ] Rodar os testes e aplicar o critério da Seção 6
- [ ] Acompanhar a perplexidade no FineWeb: na miniatura, ela piorou após o mid (de ~154 para ~242)

**Saída:** veredito e checkpoint publicado.

### Fase 6: SFT
- [ ] A partir de M. Máscara `-100` em tudo que não for resposta do assistente (incluir `<|end|>`, para o modelo aprender a parar)
- [ ] LR 1e-5 a 3e-5, 1–2 épocas, subset curado; monitorar validação para detectar overfitting

**Saída:** para as mesmas perguntas, respostas melhores que no mid, com parada correta.

### Fase 7: Avaliação final
- [ ] Em pré, C, M e SFT: perplexidade, HellaSwag, ARC-Easy (múltiplos benchmarks são diferencial), aritmética e GSM8K
- [ ] Comparar antes e depois do SFT nas mesmas perguntas
- [ ] Gerar tabelas e gráficos a partir de `results/*.json`

### Fase 8: Demo e entrega
- [ ] **Demo em Streamlit rodando localmente em notebook.** O login não aguenta e a GPU pode estar ocupada no dia. Exportar `safetensors` fp16 (~200 MB) e medir tokens/s em CPU
- [ ] Ter um vídeo de reserva, caso a demo ao vivo falhe
- [ ] Publicar os 3 checkpoints; README com reprodução por etapa e créditos
- [ ] Slides: arquitetura, Chinchilla, curvas, resultados, benchmark de matemática, discussão, demo (10–15 min)
- [ ] Todo o grupo deve entender o código gerado com IA, pois haverá perguntas na apresentação

### Orçamento de GPU (*estimativa*)

| Etapa | GPU·h |
|---|---|
| Benchmark e smoke tests | ~3 |
| Piloto (2 runs + miniatura) | ~4 |
| Pré-treino final | ~12 |
| Mid M e C | ~3 |
| SFT e avaliações | ~3 |
| **Total** | **~25** |

---

## 8. Estrutura do repositório

```text
slm/
├── README.md, PLANO.md, CREDITS.md, pyproject.toml
├── configs/{pilot,final}/{pretrain,mid_m,mid_c,sft}.yaml
├── slm/            # model, tokenizer, schedule, trainer, checkpoint, data, chat, generate, evaluation
├── scripts/        # tokenize_pretrain, prepare_chat, pretrain, midtrain, sft, evaluate, compare, plot_curves, export
├── slurm/          # um .sbatch por etapa
├── app/chat.py     # Streamlit
├── tests/          # máscara, alinhamento, retomada, overfit
└── results/        # JSONs e gráficos versionados
```

Dados e checkpoints ficam em `/scratch/ap211/slm/`, nunca no Git.

---

## 9. Riscos

| Risco | Mitigação |
|---|---|
| Fila cheia perto do prazo | Pré-treino até Qua 11; a partir de Qui 12, rodar em 1 GPU; folga na Qua 18 |
| Falta de memória na GPU durante o treino | Batch de 24 (~7 GB de folga); avaliação sem gradiente e com batch menor |
| Run interrompido | Checkpoints frequentes e retomada testada na Fase 1 |
| `/scratch` sem backup | Copiar os marcos para `/home` e publicar no HF |
| Matemática sem efeito | Decisão no piloto; tarefa reserva; resultado negativo reportado com IC |
| Geração em CPU lenta na demo | Medir com antecedência; vídeo de reserva |

---

## 10. Fontes

- [SmolLM3](https://huggingface.co/blog/smollm3): GQA, WSD, otimizador
- [nanochat](https://github.com/karpathy/nanochat): `gpt.py`, `tokenizer.py`, `tok_train.py`, `chat_sft.py`, `tasks/gsm8k.py`
- [Beating GPT-2 for <$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Benchmarking Optimizers for LLM Pretraining](https://arxiv.org/html/2509.01440v1)
- [nanochat no DGX Spark](https://objectgraph.com/blog/nanochat-dgx-spark/)
- Datasets via API do Hugging Face: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA), [SciQ](https://huggingface.co/datasets/allenai/sciq), [ARC](https://huggingface.co/datasets/allenai/ai2_arc)
