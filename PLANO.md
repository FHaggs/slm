# Plano do Projeto 2: Small Language Model

Pipeline: **Pré-treino (FineWeb-Edu) → Mid-training (chat + matemática, com benchmark antes e depois) → SFT → Avaliação & Demo**

Atualizado em 05/11/2026. **Prazo: 19/11/2026** (14 dias). Os itens marcados como *estimativa* são validados na Fase 0 ou na Fase 3.

**Decisões fechadas**
- Modelo de ~100M parâmetros, com 3B tokens no pré-treino.
- Benchmark simples de matemática, rodado antes e depois do mid-training.
- Uso de IA no código é permitido; código de terceiros adaptado continua precisando de crédito.

---

## 1. Calendário

| Datas | Etapa | Marco |
|---|---|---|
| Qui 05 – Sex 06 | Fases 0 e 1: ambiente, benchmark, núcleo de treino, smoke tests | Pipeline minúsculo de ponta a ponta |
| Sex 06 – Sáb 07 | Fase 2: dados e tokenizer | Memmaps prontos |
| Sáb 07 – Dom 08 | Fase 3: piloto com antes/depois em miniatura | **Decisão: manter a matemática ou trocar de tarefa** |
| Seg 09 – Qua 11 | Fase 4: pré-treino final (~18h + fila) | Checkpoint publicado |
| Qua 11 – Qui 12 | Fase 5: mid-training | Veredito da matemática |
| Sex 13 | Fase 6: SFT | Checkpoint final |
| Sáb 14 – Dom 15 | Fases 7 e 8: avaliação final e demo | Tabelas e demo funcionando |
| Seg 16 – Ter 17 | Fase 8: README, checkpoints no HF, slides, ensaio | Entrega pronta |
| Qua 18 | **Folga** | |
| Qui 19 | Entrega | |

**Limites para acionar o plano B**
- **Pré-treino não iniciado até Qui 12:** trocar para a config de ~72M com 2,5B tokens (~10h).
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
| Fila | Slurm, partição `LAD_geral`, `priority/basic` (FIFO) + backfill, sem preempção, sem limite de tempo | `--time` realista ajuda o backfill; job de 2 GPUs espera as duas livres |
| `/home` | NFS, 1,4 TB livres e compartilhados; escrita de 94 MB/s | Código, logs e resultados; não dataset |
| `/scratch` | Disco local de 37 TB, 35 TB livres, gravável | Dados e checkpoints em `/scratch/ap211/slm`; velocidade ainda não medida; backup desconhecido |
| Rede no nó | HF e PyPI acessíveis | Download direto em job |
| Python | 3.10 sem `ensurepip` nem torch; driver CUDA 12.8 | Usar `uv` em vez de `venv`; confirmar a wheel do torch na Fase 0 |

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
torchrun --standalone --nproc_per_node=2 scripts/pretrain.py \
    --config configs/final/pretrain.yaml --max-hours 23.5 --resume auto
```

- **Corte por tempo:** o rank 0 decide, com `--max-hours`, salvar e sair antes do limite. É mais simples que sinais, porque o `torchrun` não repassa `SIGUSR1`.
- **Runs de 1 GPU** (pilotos, mid, SFT) usam `--gres=gpu:1`, sem esperar as duas.

---

## 4. Arquitetura

| Hiperparâmetro | Valor |
|---|---|
| Vocabulário | 32.000 (BPE próprio) |
| `d_model` / camadas | 768 / 12 |
| Cabeças | 12 de query, 4 de key/value (GQA) |
| FFN | SwiGLU, oculto ≈ 8/3 × d (2.048) |
| Contexto | 1.024 |
| **Total** | **≈ 100,1M** (75,5M em blocos, 24,6M em embeddings) |

**Chinchilla:** o mínimo de 20×N dá 2B tokens. Usamos 3B (30×N), a mesma folga do exemplo do enunciado (63M → 2B = 32×N). Tempo estimado: ~18h em 2 GPUs, escalando a referência do enunciado pelo produto parâmetros de bloco × tokens; o benchmark da Fase 0 substitui esse número.

| Componente | Escolha | Justificativa |
|---|---|---|
| Base | Decoder-only, pre-norm, sem bias | Padrão de LLaMA, SmolLM e nanochat |
| Posição | **RoPE** | Sem parâmetros e padrão atual |
| Atenção | **GQA 12/4**, via `scaled_dot_product_attention` (Flash) | O SmolLM3 usa 4 grupos e relata desempenho igual ao de MHA; aqui economiza ~9,4M parâmetros |
| Ativação | **SwiGLU** | Padrão da linha LLaMA |
| Normalização | **RMSNorm** | Mais barata que LayerNorm; citada como diferencial |
| Embeddings | **Tied**, sem weight decay | Evita gastar mais 24,6M na saída; SmolLM3 faz o mesmo |
| Tokenizer | **BPE de 32k próprio, dígito a dígito** | Ver abaixo |
| Otimizador | **AdamW**, comparado com **D-Muon** no piloto | As fontes divergem ([EPFL](https://arxiv.org/html/2509.01440v1), SmolLM3), e o enunciado valoriza a justificativa por experimento |
| Schedule | **WSD**: warmup de ~2.000 steps, decaimento linear nos 10% finais | Usado no SmolLM3 |
| Precisão | bf16 + `torch.compile` + DDP | Confirmar suporte a bf16 na Fase 0 |

**Tokenizer**
- **Dígito a dígito:** o BPE do GPT-2 funde dígitos de forma arbitrária (`1234` pode virar `12`+`34`), o que deve atrapalhar a aritmética. O ganho é uma hipótese, não uma medição.
- **Vocabulário de 32k:** com 50k e 65k, os embeddings tomariam 34% e 40% dos parâmetros, contra 25% aqui.
- **Treino:** ~90% FineWeb-Edu e ~10% chat + GSM8K treino. **Nunca** GSM8K teste.
- **Tokens especiais:** `<|bos|>`, `<|doc_end|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<|pad|>`. Trocar o tokenizer depois obrigaria a refazer o pré-treino.

**Não vamos adotar:** NoPE e YaRN (servem para contexto longo), truques de resíduo do nanochat, MoE, Mamba, intra-document masking, RL e ferramenta de calculadora. Nenhum cabe no prazo com ganho claro.

---

## 5. Dados

### Pré-treino: FineWeb-Edu

O dataset tem 1,53B documentos e **4,5 TB de parquet: não baixar tudo**.
- **Amostragem:** ~12 arquivos de dumps de anos diferentes (~25 GB). Cada arquivo rende ~600M tokens (*estimativa*). Tokenizar, embaralhar por documento e cortar em 3B.
- **Validação:** ~50M tokens de um **dump não usado no treino**, excluindo URLs repetidas.
- **Formato:** `uint16` memmap em `/scratch` (~6 GB).
- **Sampler determinístico:** permutação fixa por seed, sem reposição, com cursor salvo no checkpoint. A retomada fica exata, e a contagem de tokens também.

### Mid-training

`SmolTalk/all` é a união exata de 13 configs, e **duas são de matemática: `metamathqa-50k` e `numina-cot-100k` (14,7% das linhas)**. Por isso:

| Run | Composição (300M tokens) |
|---|---|
| **M** (principal) | 75% SmolTalk sem as 2 configs + 25% matemática (GSM8K treino repetido + MetaMathQA) |
| **C** (controle) | 100% SmolTalk sem as 2 configs, mais um filtro simples de resíduos (`\boxed`, `####`) |

A `numina-cot` fica de fora: tem problemas de olimpíada, muito acima do nível deste modelo. Não há aritmética sintética no treino, para o benchmark não testar exatamente o que foi treinado.

---

## 6. Benchmark de matemática: antes e depois

**Pergunta:** o mid-training com matemática melhorou o modelo em matemática?

**Por que não só antes/depois:** o mid-training de chat também pode mudar o resultado, então a melhora sozinha não prova que a causa foi a matemática. O run **C** resolve isso reutilizando o mesmo código, e custa só ~3,6h em uma GPU (*estimativa*). Se faltar tempo, ele é o primeiro corte.

### Os testes

Os dois usam **o mesmo prompt em texto simples em todos os checkpoints**. O modelo do pré-treino não conhece o template de chat, e usar formatos diferentes distorceria a comparação.

| Teste | Exemplo | Nota |
|---|---|---|
| **Aritmética** (principal), ~300 itens em cada uma de 6 células: +, −, × com 1 e 2 dígitos, seed fixa | `17 + 25 =` | Número gerado exatamente igual ao correto |
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

Script simples: normalizar o texto e buscar as perguntas do GSM8K teste (correspondência exata) em toda a matemática de treino. No MetaMathQA, comparar também a coluna `original_question`, que guarda a pergunta antes da paráfrase. Remover e reportar os casos. Fora do MetaMathQA, a checagem não pega paráfrases.

### Decisão no piloto (Dom 08)

Rodar o antes/depois em miniatura com o modelo piloto. **Trocar de tarefa só se** a aritmética não se mexer em M comparado ao pré-treino, ou se o preparo dos dados travar.

**Tarefa reserva: ciências.** Treinar com SciQ e testar com ARC-Easy, por log-likelihood. O benchmark ARC é um dos aceitos pelo enunciado e funciona igual antes e depois. O resto do pipeline não muda. Limitação: o SciQ tem só 11.679 linhas de treino, então a dose será bem menor que os 25% da matemática, mesmo repetindo os dados.

---

## 7. Plano de passos

### Fase 0: Ambiente e calibração
- [ ] Criar o repositório GitHub; instalar `uv`, Python e PyTorch; criar `/scratch/ap211/slm`
- [ ] Benchmark curto em 1 e 2 GPUs: tokens/s, maior batch, velocidade do `/scratch`

**Saída:** bf16 confirmado e estimativa de ~18h recalculada com tokens/s medidos.

### Fase 1: Núcleo e smoke tests
- [ ] `model.py` com contagem de parâmetros; núcleo de treino (DDP, bf16, WSD, grad clip, checkpoint, retomada, log)
- [ ] Loss inicial ≈ ln(32000) ≈ 10,4; overfit de um mini-batch até loss perto de 0
- [ ] Interromper e retomar: mesma curva e mesmo cursor
- [ ] Testes de máscara do SFT e do alinhamento dos alvos
- [ ] Pipeline completo com modelo minúsculo: pré → mid → SFT → eval → demo

**Saída:** pipeline de ponta a ponta funcionando em menos de 1h.

### Fase 2: Dados e tokenizer (jobs só de CPU)
- [ ] Baixar os ~12 arquivos do FineWeb-Edu; treinar o tokenizer; gerar memmaps
- [ ] Preparar M, C e o dataset do SFT; rodar o script de vazamento

**Saída:** contagem de tokens de cada fonte registrada; decodificação conferida manualmente.

### Fase 3: Piloto
Modelo de ~25M (d=384, L=8) em 500M tokens: ~1h por run em 1 GPU (*estimativa*).
- [ ] 3 runs: AdamW, D-Muon e GPT clássico (LayerNorm, GELU, MHA, posição aprendida)
- [ ] Antes/depois de matemática em miniatura

**Saída:** configuração escolhida e decisão sobre a matemática. Ressalva: o melhor no piloto pode não ser o melhor no modelo final.

### Fase 4: Pré-treino final (~18h em 2 GPUs)
- [ ] Checkpoint a cada ~30 min, mantendo os 2 últimos + marcos
- [ ] Logar loss e perplexidade de validação; guardar amostras de geração

**Saída:** curvas sem anomalias; texto coerente; HellaSwag acima do acaso. **Publicar o checkpoint** no HF.

### Fase 5: Mid-training
- [ ] Pesos do pré-treino, otimizador zerado, LR 1e-4 a 3e-4, warmup curto, 300M tokens
- [ ] Rodar M e C (podem ser simultâneos, um em cada GPU)
- [ ] Rodar os testes e aplicar o critério da Seção 6

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
| Benchmark e smoke tests | ~4 |
| Pilotos (3 runs + miniatura) | ~6 |
| Pré-treino final | ~36 |
| Mid M e C | ~8 |
| SFT e avaliações | ~5 |
| **Total** | **~59** |

---

## 8. Estrutura do repositório

```text
slm/
├── README.md, PLANO.md, CREDITS.md, pyproject.toml
├── configs/{pilot,final}/{pretrain,mid_m,mid_c,sft}.yaml
├── slm/            # model.py, tokenizer.py, optim.py, schedule.py, trainer.py, data.py, chat.py, generate.py
├── scripts/        # fetch_fineweb, train_tokenizer, tokenize_pretrain, prepare_chat, check_leakage,
│                   # pretrain, midtrain, sft, eval_ppl, eval_mc, eval_math, plot_curves, export
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
| Fila cheia perto do prazo | Pré-treino até Qua 11; limite de Qui 12 para o plano B de 72M; folga na Qua 18 |
| Run interrompido | Checkpoints frequentes e retomada testada na Fase 1 |
| `/scratch` sem backup | Copiar os marcos para `/home` e publicar no HF |
| Matemática sem efeito | Decisão no piloto; tarefa reserva; resultado negativo reportado com IC |
| Geração em CPU lenta na demo | Medir com antecedência; vídeo de reserva |

---

## 10. Fontes

- [SmolLM3](https://huggingface.co/blog/smollm3): GQA, WSD, otimizador
- [nanochat](https://github.com/karpathy/nanochat): `gpt.py`, `chat_sft.py`, `tasks/gsm8k.py`
- [Benchmarking Optimizers for LLM Pretraining](https://arxiv.org/html/2509.01440v1)
- [nanochat no DGX Spark](https://objectgraph.com/blog/nanochat-dgx-spark/)
- Datasets via API do Hugging Face: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA), [SciQ](https://huggingface.co/datasets/allenai/sciq), [ARC](https://huggingface.co/datasets/allenai/ai2_arc)
