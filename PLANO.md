# Plano do Projeto 2: Small Language Model

Pipeline: **Pré-treino (FineWeb-Edu) → Mid-training (chat + matemática, com benchmark antes e depois) → SFT → Avaliação & Demo**

Atualizado em 16/11/2026. **Prazo: 19/11/2026** (3 dias). Os itens marcados como *estimativa* são validados na Fase 0 ou na Fase 3.

**Decisões fechadas**
- Modelo de ~114M parâmetros, com 2,5B tokens no pré-treino.
- Benchmark simples de matemática, rodado antes e depois do mid-training.
- Uso de IA no código é permitido; código de terceiros adaptado continua precisando de crédito.
- **Princípio:** nada complicado entra sem evidência de que funciona. Começamos pelo simples, e cada adição precisa de um teste.
- **(15/11) Fases 5 e 6 divididas:** mid-training C (controle) no servidor, por `slurm/mid_c.sbatch` (submetido pelo `ap205` em Seg 16, job 1173, assim que o pré-treino liberou a vaga); mid-training M e SFT localmente, numa RTX 4060 (8 GB) de um membro, a partir de Seg 16 à noite. Motivo: no servidor, 3 jobs de 1 GPU com limite de 2 dias estavam na frente (início só ~Qua 18).
- **(15/11) C mantido como grupo de controle:** separa o efeito da matemática do efeito do formato de chat. No piloto, foi o C que mostrou que parte do ganho no GSM8K era formato e que a queda no ARC-Easy vinha do chat. Continua sendo o primeiro corte se faltar tempo.
- **(15/11) Acúmulo de gradiente no M local:** micro-lotes somados até 24 registros por passo, com a loss ponderada pelos tokens válidos do passo inteiro (não pela média de cada micro-lote, que distorce com padding e máscara). Assim o M local tem o mesmo batch efetivo, LR, schedule e ordem de dados do C no servidor. Só entra depois dos testes de equivalência em CPU (Fase 5).
- **(15/11) Avaliação dos 4 modelos (pré, M, C e SFT) numa única máquina**, no fim, para a comparação (McNemar) não misturar GPUs.
- **(15/11) Branches:** `midtraining-c` sai da `pre-treino` (um PR só para pré-treino e C); `midtraining-m` sai da `midtraining-c`. A `mid-sft` fica parada, para ser reaproveitada no SFT.
- **(15/11) GPUs alternativas avaliadas e descartadas:** RX 7800 XT de 16 GB (ROCm só tem suporte no Linux; no Windows, a AMD declara que não há suporte a treino), RTX 3050 e RTX 2060 Super (8 GB ou menos; a 2060 Super não tem bf16).
- **(16/11) M e SFT também no servidor:** com o C pronto e uma GPU aparentemente livre, foi submetido o `slurm/mid_sft.sbatch` (branch `mid-sft`, job 1175): avaliação do pré-treino → M → C (pulado, já existe) → SFT → avaliações de M, C e SFT → McNemar, tudo em 1 GPU, lote 24 sem acúmulo. O M local (RTX 4060, com acúmulo) continua como reserva e como teste do acúmulo de gradiente. *Proposta (a decidir):* o M e o SFT do servidor como oficiais, por terem a mesma GPU, o mesmo lote e o mesmo ambiente do C.
- *(16/11, noite) A decidir:* M, SFT e as avaliações locais terminaram às 11h18 de Seg 16. Proposta do Orange: usar o M e o SFT **locais** como oficiais, avaliar o C na mesma máquina local e cancelar o job 1175 (que só começaria Qua 18, 11h58), liberando a GPU para as outras turmas.

---

## 1. Calendário

Revisado em 10/11 e 11/11: a Fase 3 começou com ~2 dias de atraso em relação ao calendário original, e a folga de Qua 18 foi absorvida. As datas a partir da Fase 4 dependem da fila de GPU, que é compartilhada com outras turmas. Em 13/11, a previsão do Slurm para o início do pré-treino passou para Sáb 14, 22h57: as datas abaixo são o pior caso e deixam a entrega sem folga. O pré-treino começou Dom 15 às 18h32, ~20h depois dessa previsão. As Fases 5 a 8 serão replanejadas quando o grupo definir onde rodar as Fases 5 e 6 (no servidor, só a partir de ~Qua 18; localmente, em avaliação).

| Datas | Etapa | Marco |
|---|---|---|
| Qui 05 – Seg 09 | Fases 0, 1 e 2: ambiente, núcleo de treino, smoke tests, dados | ✅ Concluídas (dados prontos em 09/11) |
| Ter 10 – Qua 11 | Fase 3: piloto ✅ (job 1140, Qua 11, 13h54–15h24) | **Decisão (pendente do grupo): manter a matemática; tokenizer GPT-2** |
| Qua 11 – Seg 16 | Fase 4: pré-treino final em **2 GPUs** (job 1152) ✅: Dom 15, 18h32 → Seg 16, 00h16 (5h44); val_ppl final 18,8 | Checkpoint publicado |
| Seg 16 – Qua 18 | Fase 5: mid-training. **C no servidor** (job 1173 ✅: Seg 16, 10h28–11h58, 1h30) e **M local** (RTX 4060 com acúmulo de gradiente, a partir de Seg 16 à noite) | Veredito da matemática |
| Ter 17 | Fase 6: SFT local, a partir de M, na sequência do M | Checkpoint final |
| Qua 18 | Fase 7: avaliação dos 4 modelos numa só máquina; Fase 8: demo | Tabelas e demo funcionando |
| Ter 17 – Qua 18 | Fase 8: README, checkpoints no HF, slides, ensaio | Entrega pronta |
| Qui 19 | Entrega | |

**Limites para acionar o plano B**
- **2 GPUs obrigatórias** (decisão do grupo em 11/11; exigência do enunciado): não há plano B em 1 GPU. O `slurm/pretrain.sbatch` aborta se não tiver as duas. Para absorver imprevistos, o job reserva 36h, bem acima da estimativa.
- **Pré-treino sem resultado até Sáb 14:** usar o último checkpoint salvo, documentando o total de tokens treinados. **A rever pelo grupo (13/11):** pela previsão do Slurm, o job só começa Sáb 14 às 22h57, então não haverá nenhum checkpoint nesse prazo. Superada em 15/11: o pré-treino começou Dom 15 às 18h32.
- **O pré-treino final não depende do piloto:** o tokenizer padrão já é o GPT-2, e a matemática só afeta o mid-training. Ele pode entrar na fila em paralelo ao piloto, pela conta de outro membro do grupo (o limite de 1 job é por usuário).

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
| Fila | Slurm, partição `LAD_geral`, `priority/basic` (FIFO) + backfill, sem preempção; limite de **96h por job** (`max walltime`, exibido no login); **máximo de 1 job por usuário, somando os que rodam e os da fila**; fila compartilhada com outras turmas | Etapas rodam uma de cada vez; não dá para encadear jobs nem preparar dados durante um treino; `--time` realista ajuda o backfill |
| `/home` | NFS, 1,4 TB livres e compartilhados, mas com **cota de 10 GB por usuário**; escrita de 94 MB/s | Código, logs e resultados; não dataset. O `.venv` ocupa ~6,9 GB da cota |
| `/scratch` | Disco local de 37 TB, 35 TB livres; escrita de 547 MB/s, leitura de 907 MB/s | Dados e checkpoints em `/scratch/ap211/slm-shared` (ver `DADOS.md`); caches por usuário em `/scratch/$USER/slm`; sem backup, e o sistema apaga dados com mais de 6 meses |
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
- **Medido em 15/11 (job 1152):** ~122,4 mil tokens/s em 2 GPUs, ~89% de 2 × 69 mil (ganho quase linear, como suposto).

**Job do pré-treino final:** `slurm/pretrain.sbatch` (2 GPUs, 32 CPUs, 96 GB, limite de 36h). Use `--mem` explícito: sem ele, o padrão observado reserva ~128 GB por GPU. Evite `--exclusive`.

```bash
sbatch slurm/pretrain.sbatch   # da raiz do clone; qualquer membro do grupo
```

- **Corte por tempo:** o rank 0 decide, com `--max-hours` (35,5h no pré-treino final), salvar e sair antes do limite. É mais simples que sinais, porque o `torchrun` não repassa `SIGUSR1`.
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

**Chinchilla:** o mínimo de 20×N dá ~2,3B tokens. Usamos 2,5B (~22×N). O nanochat relata ótimo perto de 10×N no setup dele, então não há motivo para ir muito além do mínimo pedido no enunciado. Na prática, treinamos 1 época completa dos dados: 2.497.560.576 tokens (~21,9×N). Tempo estimado: ~5–6h em 2 GPUs, partindo de 69 mil tokens/s por GPU e somando uma folga para dados, avaliação e checkpoints.

| Componente | Escolha | Justificativa |
|---|---|---|
| Base | Decoder-only, pre-norm, sem bias | Padrão de LLaMA, SmolLM e nanochat |
| Posição | **RoPE** | Sem parâmetros e padrão atual |
| Atenção | **GQA 12/4**, via `scaled_dot_product_attention` (Flash) | O SmolLM3 usa 4 grupos e relata desempenho igual ao de MHA; aqui economiza ~9,4M parâmetros |
| Ativação | **SwiGLU** | Padrão da linha LLaMA |
| Normalização | **RMSNorm** | Mais barata que LayerNorm; citada como diferencial |
| Embeddings | **Tied**, sem weight decay | Evita gastar mais 38,6M na saída; SmolLM3 faz o mesmo |
| Tokenizer | **BPE do GPT-2** via `tiktoken` | Ver abaixo |
| Otimizador | **AdamW** (WD=0,1, grad clip 1), **LR de pico 3e-4** | Simples e usado no SmolLM3. O Muon do nanochat depende de vários ajustes; não dá para validá-lo no prazo. LR: ver abaixo |
| Schedule | **WSD**: warmup de 100M tokens (~2.000 steps em 2 GPUs), decaimento linear nos 10% finais | Usado no SmolLM3 |
| Precisão | bf16 + `torch.compile` + DDP | Confirmar suporte a bf16 na Fase 0 |

**Tokenizer**
- **Padrão:** GPT-2 sem treinar nada. É a primeira opção listada no enunciado e não traz risco de defeito no tokenizer.
- **Tokens especiais:** `<|endoftext|>` separa documentos; `<|user|>`, `<|assistant|>`, `<|end|>` e `<|pad|>` entram como IDs novos.
- **Dígitos:** separar os dígitos é uma hipótese, não uma medição. O nanochat usa blocos de até 2 dígitos, escolhidos para economizar vocabulário, não por testes de aritmética. A variante só entra se ganhar no piloto (Fase 3).
- **BPE próprio de 32k:** é o que o nanochat faz, e economizaria ~14M de parâmetros. Fica de fora porque o ganho não compensa o trabalho e o risco neste prazo.
- **Trocar o tokenizer depois obrigaria a refazer o pré-treino**, então a decisão fecha no fim do piloto.

**Taxa de aprendizado (pré-treino)**
- **Escolha: 3e-4**, decidida pelo grupo em 11/11.
- **Ponto de partida:** o GPT-3 Small (125M, tamanho próximo ao nosso) usou 6e-4 com batch de ~0,5M tokens (Brown et al., 2020, Tabela 2.1).
- **Ajuste pelo batch:** o nosso batch é de ~49 mil tokens (2 GPUs × 24 × 1.024), ~10× menor. Escalar a LR pela raiz do batch, regra usada no nanochat, daria ~1,8e-4. O 3e-4 fica entre as duas referências.
- **Por que não a LR do nanochat:** ele usa Muon (LR 0,02 nas matrizes) e AdamW com uma LR por grupo de parâmetros (`scripts/base_train.py`). Esses valores não se transferem para um AdamW único.
- **Verificação:** acompanhar a curva de loss na primeira hora. O trainer aborta se a loss deixar de ser finita.

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

**Resultado (11/11):** a aritmética se mexeu (M 2,9% contra 0,9% do pré-treino, p = 0,0003). Pela regra, a matemática fica; a decisão final é do grupo. Detalhes na Fase 3.

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

**Saída:** cumprida; 44 testes automatizados passando (`sbatch slurm/tests.sbatch`). Reproduzido no usuário `ap205` em 10/11: setup em 5 min e 44 testes passando.

**Limites do que foi validado**
- O DDP foi testado com 2 processos em CPU, não em 2 GPUs. O primeiro uso em GPU será o pré-treino.
- A geração e os benchmarks da miniatura não dizem nada sobre o modelo final: 11M e 20M de tokens são pouco para qualquer conclusão.
- A demo foi testada sem navegador e usando CPU. Falta rodar em notebook.

### Fase 2: Dados (jobs só de CPU)
- [x] Baixar os 12 arquivos do FineWeb-Edu; gerar memmaps com o GPT-2
- [x] Preparar M, C e o dataset do SFT; conferir a contagem de vazamento no `meta.json`

**Saída:** cumprida em 43 min de CPU (`sbatch slurm/data.sbatch`). Contagens, local e conferências em `DADOS.md`.
- **Pré-treino:** 2,5B tokens de treino e 50M de validação, de um dump que o treino não usa.
- **Vazamento:** 0 perguntas do GSM8K teste nos dados de chat, confirmado por busca independente nos tokens.
- **Padding:** 19% a 22% dos registros de chat com contexto de 1.024; M e C ficam com quase os mesmos tokens úteis (~294M e ~292M).

### Fase 3: Piloto
Modelo de ~32M (d=384, L=8) em 500M tokens: ~1h por run em 1 GPU (medido: 184 mil tokens/s, mais folga). O objetivo é validar o pipeline, não caçar otimizações.
- [x] Run base com GPT-2 + antes/depois de matemática em miniatura
  - Configs em `configs/pilot/` e job em `slurm/pilot.sbatch`: pré-treino de 500M tokens → mid M e C de 100M tokens cada → avaliação dos 3 checkpoints → McNemar (M × pré e M × C). Lê os dados de `slm-shared` e grava em `/scratch/$USER/slm`.
  - **Rodou em 11/11, das 13h54 às 15h24 (1h30), job 1140 (`ap205`), em 1 RTX A5000: `COMPLETED`.** A GPU liberou antes do previsto porque o job da frente terminou antes do limite.
  - **Pré-treino:** 500.006.912 tokens em 15.259 steps, a ~182 mil tokens/s (igual ao benchmark: a leitura do `/scratch` não atrasou). Perplexidade de validação de 54.692 para **32,5**; o decaimento do WSD nos 10% finais baixou de 39,3 para 32,5. LR 1e-3 estável nessa escala.
  - **Mid-training:** M e C com 100.007.936 tokens cada (3.052 steps). Perplexidade na validação do chat: de 41,2 para 8,4 (M) e 7,8 (C).

    | Métrica | Pré-treino | M (chat + mat.) | C (só chat) |
    |---|---|---|---|
    | Perplexidade no FineWeb (val) | **32,4** | 61,4 | 56,4 |
    | HellaSwag (acaso = 25%) | 26,3% | 26,2% | 26,4% |
    | ARC-Easy | **43,4%** | 34,9% | 33,7% |
    | Aritmética (1.107 itens) | 0,9% (10) | **2,9% (32)** | 1,3% (14) |
    | GSM8K teste (1.319) | 1,5% (20) | **2,1% (28)** | 1,1% (14) |
    | GSM8K: respondeu com um número | 71,6% | 93,7% | 49,7% |
    | GSM8K: acerto entre as respondidas | 2,1% | 2,3% | 2,1% |

    | McNemar (p) | Aritmética | GSM8K | ARC-Easy | HellaSwag |
    |---|---|---|---|---|
    | M × pré-treino | **0,0003** (M melhor) | 0,30 | **3e-18** (M pior) | 0,97 |
    | M × C | **0,006** (M melhor) | **0,024** (M melhor) | 0,11 | 0,59 |

  - **Critério 1 (principal): atendido.** M acerta mais aritmética que o pré-treino e que C. O ganho está concentrado em 1 dígito: soma (12,3% contra 1,2% em C) e multiplicação (17,3% contra 3,7%). Com 2 dígitos, os três ficam perto de 0. Na subtração de 1 dígito, C acerta mais que M (7 contra 3 de 45 itens).
  - **Curva durante o mid:** em M, a aritmética sobe de forma quase contínua (0,9% → 2,9%); em C, oscila entre 0,5% e 1,8%. Isso indica tendência, não ruído no ponto final.
  - **Critério 2 (transferência):** M supera C no GSM8K (p = 0,024), mas o acerto entre as perguntas respondidas é praticamente igual nos três (~2,1–2,3%). O ganho vem de M responder com um número com mais frequência: é efeito de formato, não de raciocínio.
  - **Critério 3 (HellaSwag):** M fica 0,1 pp abaixo de C, dentro da regra. Pouco informativo: os três estão no nível do acaso.
  - **Achados extras:** o mid-training derruba o ARC-Easy em ~9 pp e quase dobra a perplexidade no FineWeb, em M e em C, então o efeito é do chat, não da matemática. Hipótese não verificada para o ARC: mudança de formato (prompt em texto simples depois de treinar em chat).
- [ ] Um único teste extra: GPT-2 com dígitos separados. Adotar só se a aritmética melhorar com McNemar p < 0,05
  - Custo maior que as ~10 linhas previstas: o trainer recusa checkpoint e dados com tokenizers diferentes, então o teste exige retokenizar o pré-treino (ou parte dele) e o chat. Decidir depois do run base.
  - **Recomendação (11/11, pendente do grupo): não fazer.** Exigiria retokenizar os dados e rodar outro piloto, atrasando o pré-treino final, sem indício de que o GPT-2 padrão seja um gargalo.

**Saída:** decisão sobre a matemática e o tokenizer (pendente do grupo; recomendação: manter a matemática e o GPT-2 padrão). Ressalva: como alerta o próprio autor do nanochat, resultado em escala pequena pode não se transferir para o modelo final.

### Fase 4: Pré-treino final (~5–6h em 2 GPUs obrigatórias, batch de 24 por GPU)
- [x] Config e job: `configs/final/pretrain.yaml` e `slurm/pretrain.sbatch`. 1 época (2.497.560.576 tokens), LR 3e-4, warmup de 100M tokens, avaliação a cada 2.500 steps (~15 min), checkpoint a cada 5.000 steps (~30 min). Reserva de 36h
- [x] Submetido: job 1152 (`ap205`), Qua 11 à tarde, commit `ab19f1a`. Pasta de checkpoints vazia antes da submissão (treino parte do zero)
- [x] Início: Dom 15/11 às 18h32 (job 1152), em 2× RTX A5000, ~20h depois da previsão de 13/11 e ~95h depois da submissão. O 1156, submetido depois, começou antes; causa não identificada
- [x] **Concluído** (job 1152, `COMPLETED`, exit 0): Seg 16 às 00h16, em 5h44 (estimativa: ~5–6h). 50.813 steps, 2.497.560.576 tokens, ~122,6 mil tok/s até o fim. Perplexidade de validação: 23,6 no step 25.000 → **18,82** no fim; loss de treino ~3,13 nos últimos steps. *Correção (16/11):* uma versão anterior deste item dizia que o decaimento baixou a loss de treino "de ~3,25 para ~3,13"; o ~3,25 não tinha fonte (o trecho do log conferido começa no step 50.300). O efeito do decaimento será medido no gráfico do `metrics.jsonl`. `final.pt` com 456.503.035 bytes; marcos guardados em `checkpoints/pretrain_marcos/` (`final.pt` e `ckpt_000045000.pt`)
- [x] Marcos: `keep_last: 3`, e no fim do job o `slurm/pretrain.sbatch` copia o `final.pt` e o `ckpt_000045000.pt` para `checkpoints/pretrain_marcos/`. O decaimento do WSD começa no passo 45.733; o ckpt 45.000 é o último da fase estável e permite refazer o decaimento
- [x] Checkpoint a cada ~34 min (1,37 GB cada), mantendo os 3 últimos: confirmado em 15/11 (ckpts 15.000, 20.000 e 25.000)
- [ ] Logar loss e perplexidade de validação (`metrics.jsonl`, junto dos checkpoints); guardar amostras de geração
- [x] Velocidade e curva conferidas em 15/11, após 3h (step 26.600 de 50.813): ~122,4 mil tok/s; val_ppl 59.030 → 53,4 (step 2.500) → 28,5 (10.000) → 23,6 (25.000), sempre caindo e já abaixo do piloto no fim (32,5); norma do gradiente estável (~0,55–0,60). LR 3e-4 sem instabilidade
- [x] Trava (`flock`) testada no `/scratch` do CDIA em 11/11: um segundo processo é barrado e a trava se solta quando o primeiro termina
- [ ] Regra do grupo: o pré-treino só roda por `sbatch slurm/pretrain.sbatch`. Rodar `scripts/pretrain.py` direto ignora a trava e pode corromper os checkpoints compartilhados
- [ ] Amostras de geração, HellaSwag e ARC do checkpoint final: *proposta* de rodar o `evaluate.py` no início do job da Fase 5, sem gastar uma entrada a mais na fila

**Saída:** curvas sem anomalias; texto coerente; HellaSwag acima do acaso. **Publicar o checkpoint** no HF.

### Fase 5: Mid-training
- [x] **Onde rodar (decidido em 15/11):** C no servidor; M e SFT localmente, com acúmulo de gradiente. Ver Decisões fechadas
- [x] C: config `configs/final/mid_c.yaml` (pesos do pré-treino final, otimizador zerado, LR 3e-4, warmup de 10M tokens, 1 época = 14.669 steps, ~292M tokens úteis; aritmética a cada 1.500 steps) e job `slurm/mid_c.sbatch` (1 GPU, limite de 6h, só treino; aborta sem o `pretrain/final.pt`; trava própria)
- [x] C: submetido pelo `ap205` em Seg 16 ~00h40 (job 1173, branch `midtraining-c`), assim que o pré-treino liberou a vaga. 2º na fila de GPU, atrás do 1172 (1 GPU, 2 dias); as 2 GPUs estão com jobs de 2 dias iniciados Seg 16 ~00h16. Pior caso de início: Qua 18 ~00h16; fim ~Qua 18 ~03h (*estimativa*)
- [x] **C concluído** (job 1173, `COMPLETED`, exit 0): Seg 16, das 10h28 às 11h58 (1h30; estimativa: 1,5–3h), a ~68,8 mil tok/s. 14.669 steps. Perplexidade na validação do chat: 22,8 → **4,86** (o decaimento final baixou de 5,47 para 4,86). Aritmética: 0,81% → **1,17%**, oscilando entre 0,99% e 2,44% ao longo do treino, sem tendência (no piloto, C terminou em 1,3%). Células finais: soma de 1 dígito 4,9%, multiplicação de 1 dígito 9,9%, 2 dígitos ~0
- [x] Mesma avaliação em GPUs diferentes: o `step 0` do C (servidor) e o do M local medem o mesmo modelo (o pré-treino). `val_loss` idêntica (3,1256), mas a aritmética deu 0,81% no servidor e 0,72% localmente (1 item de 1.107). Reforça a decisão de avaliar os 4 modelos na mesma máquina
- [ ] M e SFT no servidor: job 1175 (`slurm/mid_sft.sbatch`), submetido Seg 16 à tarde; na fila (Resources). O 1174 (`ap220`) ocupa as **2 GPUs** (`gres:gpu:2`) desde Seg 16 11h58; previsão do Slurm para o 1175: Qua 18, 11h58. Redundante com o run local (ver decisão a tomar)
- [x] M: acúmulo de gradiente implementado em `slm/trainer.py` (`grad_accum`, padrão 1 = caminho anterior, sem mudança). Cada passo soma `grad_accum` micro-lotes, com a loss somada por token e dividida pelos tokens válidos do passo inteiro
- [x] Testes de equivalência em CPU (`tests/test_grad_accum.py`, **6 de 6 passando** em 16/11): um passo com 24 registros = 6 micro-lotes de 4 (loss e gradientes), também com padding e máscara desiguais entre micro-lotes; controle mostrando que a média simples por micro-lote **não** bate; treino curto inteiro (pesos finais, curva e validação iguais); retomada com acúmulo; valor inválido recusado
- [x] Testes antigos no notebook (Windows, CPU): 48 de 50 passando. Os 2 que falham **falham igual no código original** (falta o `streamlit` no ambiente de teste; o teste de 2 processos não abre a conexão local no Windows). Não têm relação com o acúmulo; no servidor (Linux) os 44 passavam
- [x] Configs locais (`configs/local/mid_m.yaml` e `sft.yaml`): mesmos hiperparâmetros do servidor, lote efetivo de 24, sem `torch.compile` (sem suporte oficial no Windows) e a mesma validação de 800 registros do C. O micro-lote é escolhido pelo `scripts/run_local.py` conforme a memória livre
- [x] `scripts/evaluate.py`: opções `--ppl-batch-size` e `--gsm-batch-size` (padrões iguais aos do servidor), para a avaliação caber em 8 GB
- [x] Dados transferidos para o PC local em 16/11 (2,44 GB, SHA-256 idêntico nos dois lados)
- [x] Checagem local (16/11, 02h06): dados ok, 394 GB livres, RTX 4060 com 6,9 de 8,0 GiB livres, bf16 ok. Medição dos micro-lotes: 8 e 6 passaram da memória física (10,86 e 9,22 GiB) e o Windows usou a RAM sem erro, ficando ~6–8x mais lentos (2,2 e 2,8 mil tok/s); 4 ficou no limite (6,85 GiB, 12,2 mil); **3 escolhido** (5,09 GiB, 18,0 mil tok/s; 3 × 8 = 24); 2 e 1 cabem, mais lentos
- [x] **M e SFT locais concluídos** (RTX 4060, micro-lote 3 × 8 = 24): início Seg 16 às 02h29; M e SFT prontos antes das 10h01; avaliações das 10h01 às 11h18 (~25–27 min por modelo). `step 0` do M: val_loss 3,1256 (val_ppl 22,8) no chat de M; `step 100`: 16,5 mil tok/s. O "step ~1000" visto às 08h30 era, muito provavelmente, do SFT (5.070 steps), já em andamento
- [ ] Rodar os testes e aplicar o critério da Seção 6
- [ ] Acompanhar a perplexidade no FineWeb: na miniatura, ela piorou após o mid (de ~154 para ~242); no piloto, de 32,4 para 61,4 (M) e 56,4 (C)

#### Como rodar o mid-training C no servidor (qualquer membro do grupo)

Pré-requisitos: acesso ao CDIA com o **seu próprio usuário** e nenhum outro job seu na fila (o limite é 1 job por usuário). O pré-treino final pode ainda estar na fila ou rodando: enquanto ele ocupa as 2 GPUs, o C não começa; e, se o pré-treino falhar, o C aborta sozinho na partida, porque confere o `checkpoints/pretrain/final.pt`.

1. No nó de login, clonar o repositório do grupo numa pasta nova e entrar na branch do C:
   ```bash
   git clone <URL do repositório> ~/slm-c
   cd ~/slm-c && git checkout midtraining-c && mkdir -p logs
   ```
   A pasta `logs/` não vem no clone, e sem ela o Slurm não grava o log do job.
2. Instalar o `uv` só para o seu usuário, se ainda não tiver:
   ```bash
   python3 -m pip install --user uv
   ```
3. Criar o ambiente Python (job só de CPU, ~5 min) e esperar ele terminar:
   ```bash
   sbatch slurm/setup.sbatch
   squeue -u $USER                    # repetir até a lista ficar vazia
   tail -n 3 logs/slm-setup.*.out     # a última linha deve ser "torch ... | cuda build 12.8"
   ```
4. Submeter o C (só depois do setup, por causa do limite de 1 job):
   ```bash
   sbatch slurm/mid_c.sbatch
   squeue -u $USER                    # PD = na fila; R = rodando
   ```
5. Quando estiver rodando, conferir o início do log:
   ```bash
   tail -n 20 logs/slm-mid-c.*.out
   ```
   Esperado: `pesos iniciais de .../pretrain/final.pt; otimizador zerado`, `params=114.1M | gpus=1 | batch=24`, e linhas `step ... | loss ... | tok/s` a ~65–69 mil tok/s (*estimativa*). A cada 1.500 steps, uma linha com `val_loss` e `arith_acc`. Duração estimada: ~1,5–3h.

Cuidados:
- Enquanto o job estiver na fila ou rodando, **não trocar de branch nem fazer `git pull` nesse clone**: o job lê o código e a config da pasta no momento em que começa.
- Rodar só pelo `sbatch`: `scripts/midtrain.py` direto ignora a trava (`.lock`) e pode corromper os checkpoints compartilhados.
- Se o job cair ou estourar o limite, basta resubmeter o mesmo `sbatch`: ele retoma do último checkpoint. Se o `final.pt` do C já existir, o job não faz nada.
- Avisar o grupo ao submeter e ao terminar. O resultado fica em `checkpoints/mid_c/final.pt`, no diretório compartilhado (ver `DADOS.md`).

#### Como rodar o M e o SFT localmente (Windows 11, RTX 4060 de 8 GB)

Estrutura esperada (os nomes das pastas `repo` e `shared` importam; o resto do caminho pode ter espaços):

```text
<pasta do projeto>\
├── shared\   # dados e checkpoints, com a mesma estrutura do servidor (data\..., checkpoints\pretrain\final.pt)
└── repo\     # clone deste repositório, na branch midtraining-m
```

1. **Dados:** no servidor, juntar num `.tar` os conjuntos `mid_m`, `mid_c` (sem os `mask.bin`), `sft`, `pretrain/val` e o `checkpoints/pretrain/final.pt`; baixar com `scp -J <usuário>@<servidor de entrada> <usuário>@<servidor do cluster>:<arquivo> .`; conferir o SHA-256 nos dois lados (`sha256sum` e `Get-FileHash`); extrair com `tar xf <arquivo> -C shared`.
2. **Ambiente:** instalar o `uv`, clonar o repositório em `repo`, `git checkout midtraining-m` e `uv sync` (PyTorch 2.11 + CUDA 12.8). Conferir: `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())"` deve imprimir `True True`.
3. **Checagens (~5 min):** `uv run python scripts/run_local.py --check`. Confere os dados (tamanhos contra os `meta.json`), o espaço em disco (mínimo 12 GB), a GPU e o bf16, e mede a memória para escolher o micro-lote (o maior divisor de 24 que cabe com folga). Nada é treinado.
4. **Rodar:** `uv run python scripts/run_local.py`. Ordem: M → SFT → avaliações (pré, M, SFT e, se o `mid_c\final.pt` estiver em `shared`, também o C) → comparações. O Windows não suspende enquanto ele roda (a tela pode apagar).
5. **Acompanhar:** `logs\local_status.txt` (uma linha por etapa) e `logs\local_run.log` (tudo). Resultados em `results\final\*.json`; checkpoints em `shared\checkpoints\{mid_m,sft}`.
6. **Se parar:** o script para no primeiro erro e registra o motivo no log. Depois de corrigir, rodar o mesmo comando: os treinos retomam do último checkpoint e as etapas concluídas são puladas.

Cuidados: não usar a GPU para outra coisa durante o treino; desligar a suspensão e pausar as atualizações do Windows; não apagar nada em `shared\checkpoints` enquanto roda.

**Saída:** veredito e checkpoint publicado.

### Fase 6: SFT
- [x] SFT local concluído (16/11): a partir do M local, LR 2e-5, 1 época (5.070 steps, 124.600.320 tokens), micro-lote 3 × 8 = 24. Perplexidade na validação do SFT no fim: **4,66**
- [ ] A partir de M. Máscara `-100` em tudo que não for resposta do assistente (incluir `<|end|>`, para o modelo aprender a parar)
- [ ] LR 1e-5 a 3e-5, 1–2 épocas, subset curado; monitorar validação para detectar overfitting

**Saída:** para as mesmas perguntas, respostas melhores que no mid, com parada correta.

### Fase 7: Avaliação final
- [x] **Avaliações locais (16/11, RTX 4060), pré-treino, M e SFT:**

  | Modelo | Perplexidade (FineWeb) | HellaSwag | ARC-Easy | Aritmética | GSM8K | GSM8K: respondeu |
  |---|---|---|---|---|---|---|
  | Pré-treino | 18,77 | 28,1% | 50,8% | 0,7% | 1,9% | 86,4% |
  | M | 61,07 | 27,1% | 37,2% | **3,2%** | **4,1%** | 99,4% |
  | SFT | 62,65 | 27,4% | 37,3% | 3,1% | 2,8% | 99,5% |
  | C | *a avaliar na mesma máquina* | | | | | |

  McNemar, M contra o pré-treino: aritmética p = 7,4e-6 (8 → 35 acertos); GSM8K p = 0,0009; HellaSwag p = 0,0007 (queda de 1,0 ponto); ARC-Easy p = 2e-34 (queda de 13,6 pontos).
  McNemar, SFT contra M: GSM8K p = 0,0498 (queda de 1,3 ponto, limítrofe); HellaSwag p = 0,047 (alta de 0,3 ponto, limítrofe); aritmética e ARC-Easy sem diferença.
  Aritmética do M por célula: soma de 1 dígito 21,0%, multiplicação de 1 dígito 14,8%, subtração de 1 dígito 4,4%; 2 dígitos entre 0,3% e 0,7%.
- [ ] **Critérios da Seção 6 (parcial, falta o C):** M acima do pré-treino em aritmética ✅ (p = 7,4e-6); M acima de C ⏳ (C ainda não avaliado na mesma máquina).
  Transferência para o GSM8K: M acima do pré-treino (p = 0,0009) e, diferente do piloto, **não é só formato**: entre as perguntas respondidas com número, o acerto vai de 2,2% para 4,1%.
  Sem regressão no HellaSwag: ❌ queda pequena, mas significativa (28,1% → 27,1%). A queda do ARC-Easy se repete (no piloto, ocorreu em M e em C); o C dirá se a causa é o chat ou a matemática
- [ ] Avaliar o C localmente: copiar `checkpoints/mid_c/final.pt` (456.503.035 bytes, SHA-256 conferido) para `shared/checkpoints/mid_c/` e rodar de novo `scripts/run_local.py` (pula o que já existe, avalia o C e roda o McNemar M × C; ~25 min)
- [ ] Em pré, C, M e SFT: perplexidade, HellaSwag, ARC-Easy (múltiplos benchmarks são diferencial), aritmética e GSM8K. Os 4 avaliados na mesma máquina local (decisão de 15/11), com o `mid_c/final.pt` copiado do servidor
- [ ] Comparar antes e depois do SFT nas mesmas perguntas
- [ ] Gerar tabelas e gráficos a partir de `results/*.json`

### Fase 8: Demo e entrega
- [ ] **Checklist, comandos e tutorial de uso do modelo em `ENTREGA.md`** (16/11): disponibilizar os checkpoints (Hugging Face, cópia no servidor, resultados no Git), o que falta com estimativas, roteiro dos slides e perguntas prováveis
- [ ] **Demo em Streamlit rodando localmente em notebook.** O login não aguenta e a GPU pode estar ocupada no dia. Exportar `safetensors` fp16 (~200 MB) e medir tokens/s em CPU
- [ ] Ter um vídeo de reserva, caso a demo ao vivo falhe
- [ ] Publicar os 3 checkpoints; README com reprodução por etapa e créditos
- [ ] Slides: arquitetura, Chinchilla, curvas, resultados, benchmark de matemática, discussão, demo (10–15 min)
- [ ] Todo o grupo deve entender o código gerado com IA, pois haverá perguntas na apresentação

### Orçamento de GPU (*estimativa*)

| Etapa | GPU·h |
|---|---|
| Benchmark e smoke tests | ~3 |
| Piloto (1 run; realizado em 11/11) | 1,5 |
| Pré-treino final | ~12 |
| Mid M e C | ~3 |
| SFT e avaliações | ~3 |
| **Total** | **~22,5** |

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

Dados e checkpoints ficam em `/scratch/ap211/slm-shared/`, nunca no Git. Estrutura, permissões e formato em `DADOS.md`.

---

## 9. Riscos

| Risco | Mitigação |
|---|---|
| Fila cheia perto do prazo | Fila compartilhada com outras turmas (em 10 e 11/11, as 2 GPUs ficaram ocupadas por ~2 dias). Pré-treino até Sex 13, com **2 GPUs obrigatórias**: o job só começa com as duas livres ao mesmo tempo. Submetido em 11/11 (job 1152, reserva de 36h). Em 12/11, um job submetido depois do 1152 começou antes dele (causa não identificada), adiando a previsão em ~24h. A folga de Qua 18 já foi consumida. O limite de 1 job é por usuário: membros diferentes do grupo podem ter jobs simultâneos. Um job na fila pode ter o limite de tempo aumentado (`scontrol update`) sem perder a posição. Em 15/11, com o pré-treino rodando, 3 jobs de 1 GPU (limite de 2 dias) entraram na fila: no servidor, as Fases 5 e 6 só começariam ~Qua 18. Em avaliação: rodar localmente |
| Falta de memória na GPU durante o treino | Batch de 24 (~7 GB de folga); avaliação sem gradiente e com batch menor |
| Run interrompido | Checkpoints frequentes e retomada testada na Fase 1 |
| LR alta para o nosso batch (pré-treino instável) | 3e-4 é metade do valor do GPT-3 Small; curva de loss conferida na primeira hora; o trainer aborta em loss não finita. Se divergir: reduzir a LR e recomeçar |
| `/scratch` sem backup | Marcos copiados para `checkpoints/pretrain_marcos/` no fim do job (protege do `keep_last`, não de falha do disco). Publicar o `final.pt` no HF logo depois do treino |
| Matemática sem efeito | Piloto (11/11): efeito significativo na aritmética (M × pré p = 0,0003; M × C p = 0,006), concentrado em 1 dígito. Tarefa reserva mantida; resultado negativo reportado com IC |
| Mid-training piora o conhecimento geral | No piloto, o ARC-Easy caiu ~9 pp e a perplexidade no FineWeb quase dobrou após o mid (M e C). Medir no modelo final e discutir nos slides |
| Acúmulo de gradiente com erro sutil (M diferente do C sem ninguém perceber) | Loss ponderada pelos tokens válidos do passo; 6 testes de equivalência em CPU passando (16/11); primeira `val_loss` do M conferida |
| GPU de 8 GB usar a RAM sem avisar (Windows), deixando o treino muito lento | O `run_local.py` escolhe o micro-lote pela memória livre, com folga, e mede a velocidade antes de começar |
| Máquina local desligar ou reiniciar durante o M ou o SFT | Retomada pelo último checkpoint; desligar suspensão e atualizações automáticas antes de começar |
| Geração em CPU lenta na demo | Medir com antecedência; vídeo de reserva |

---

## 10. Fontes

- [SmolLM3](https://huggingface.co/blog/smollm3): GQA, WSD, otimizador
- [nanochat](https://github.com/karpathy/nanochat): `gpt.py`, `tokenizer.py`, `tok_train.py`, `chat_sft.py`, `tasks/gsm8k.py`, `scripts/base_train.py` (LRs e escala pelo batch)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3), Tabela 2.1: LR e batch do GPT-3 Small
- [Beating GPT-2 for <$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Benchmarking Optimizers for LLM Pretraining](https://arxiv.org/html/2509.01440v1)
- [nanochat no DGX Spark](https://objectgraph.com/blog/nanochat-dgx-spark/)
- Datasets via API do Hugging Face: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA), [SciQ](https://huggingface.co/datasets/allenai/sciq), [ARC](https://huggingface.co/datasets/allenai/ai2_arc)
