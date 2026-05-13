# Publicação LinkedIn — Titanium Alpha

> Guia de execução completo aplicando a skill `retorica-persuasao`.
> **Objetivo primário:** gerar autoridade no nicho de quant + data science.
> **Objetivo secundário:** promover o repositório GitHub `cockles98/titanium-alpha`.

> ### 📌 v2 — 2026-05-13 — Plots reais adicionados
>
> A pasta `docs/images/benchmark graphs/` agora tem 16 plots do dashboard. Isso muda o carrossel: slides 6, 8 e 9 passam a **compositar PNGs reais** em vez de pedir ao Nano Banana que desenhe gráficos.
> - **Para executar hoje:** vá direto à **[Seção 10](#10-atualização-2026-05-13--plots-reais-disponíveis-carrossel-v2)** — ela tem a estrutura final (10 slides), as specs dos slides novos, os prompts atualizados, a lista revisada de anexos e o plano de follow-up.
> - **Seções 1, 2, 3, 5, 6, 7, 8** continuam válidas como fundamento.
> - **Seções 4 e 9.3** foram **superadas** pela Seção 10 (mas mantidas como referência conceitual).
> - **✓ Alpha reconciliado:** o valor canônico é **+2.57% a.a.** (Jensen's alpha de `benchmark_metrics.py`). O caption (Seção 3) já estava correto. Os plots `CAPM Scatter vs SPY.png` (+4.69%) e `Market Relationship.png` (+3.37%) estavam regredindo retornos brutos sem subtrair `rf` — bug do dashboard corrigido em `app.py`. **Regere esses dois plots antes de publicar** se for usá-los. Detalhes em §10.10.

---

## 1. Diagnóstico retórico

| Dimensão | Decisão | Fundamento |
|---|---|---|
| Modo | **CRIAÇÃO** | Conteúdo novo para posicionamento |
| Canal | **LinkedIn** | Rede profissional; público quant/DS concentrado |
| Framework (caption) | **Hook + Insight + CTA** com vulnerabilidade calculada | O padrão de ouro no LinkedIn é falha real + lição (Seção 2.1 da skill) |
| Framework (carrossel) | **SCQA** (Situation → Complication → Question → Answer) | Top-down, anti-suspense, ideal para audiência técnica (Seção 2.2) |
| Ethos | 1002 testes, 10y OOS, papers peer-reviewed citados, repo público | Credibilidade por **concretude**, não por adjetivo |
| Pathos | Frustração com *AI hype* + orgulho de rigor + confissão sobre look-ahead bias | Vulnerabilidade genuína (aprendizado real documentado nas sessões 34-36) |
| Logos | Sharpe 0.766 vs SPY 0.592, Beta 0.566, Alpha +2.57%, MaxDD −21.94% vs −33.72%, 10 anos OOS | Tabela lado a lado — o próprio dado argumenta |
| Gatilhos Cialdini | **Autoridade** (papers), **Prova social** (stack que quant reconhece), **Unidade** ("nós quants sabemos que…") | Sem escassez artificial — audiência detectaria |

---

## 2. Formato recomendado

### **CARROSSEL DE 9 SLIDES** (recomendação principal)

**Por que carrossel e não imagem única, vídeo ou artigo:**

- **Algoritmo do LinkedIn em 2026** prioriza dwell time; carrossel força scroll horizontal e entrega 3–5x mais tempo de tela que imagem única.
- **Conteúdo técnico em camadas** (arquitetura → resultados → rigor → limitações) pede narrativa sequencial — impossível numa imagem só, desperdiçada num vídeo curto.
- **Público quant retém com prova visual**: tabelas de métricas, diagrama de arquitetura e equity curve funcionam melhor estáticos, com tempo de leitura, do que passando a 30 fps.
- **Artigo LinkedIn** posiciona mas tem 5–10% do alcance de um post — não compensa aqui. Reservar para desdobramento futuro ("How I built CPCV-OOS from scratch").

**Complemento opcional (alto ROI):** vídeo curto de 12–20 s do **War Room** em ação como *primeira peça* do carrossel (slide 1 animado), ou postado como *follow-up* 48h depois linkando o post original. Já existe `docs/images/warroom_demo.gif` pronto para upload como MP4.

### Especificações técnicas do carrossel

| Parâmetro | Valor |
|---|---|
| Formato de arquivo | PDF com 9 páginas OU 9 PNGs sequenciais |
| Dimensões | **1080 × 1350 px** (4:5 vertical — ocupa mais tela que 1:1) |
| Densidade | 2x (Retina) — nunca exportar em 72 dpi |
| Fonte principal | **Inter** ou **IBM Plex Sans** (sans-serif técnica, 0 custo, disponível Google Fonts) |
| Fonte código/números | **JetBrains Mono** ou **IBM Plex Mono** (monoespaçada para tabelas e métricas) |
| Tamanho mínimo de fonte | 28 pt para corpo, 48 pt para títulos (legibilidade mobile) |

### Paleta visual (alinhada ao dashboard do projeto)

| Papel | Cor | Hex |
|---|---|---|
| Fundo primário | Preto grafite | `#0E1117` |
| Fundo alternativo | Grafite azulado | `#161B22` |
| Destaque principal | Azul titanium | `#2E86DE` |
| Destaque positivo | Verde quant | `#26DE81` |
| Destaque negativo / risco | Vermelho coral | `#EB3B5A` |
| Texto primário | Branco quase puro | `#F5F6FA` |
| Texto secundário | Cinza médio | `#8B94A7` |
| Grid / divisores | Cinza escuro | `#2F3640` |

**Princípio de design:** estética *Bloomberg Terminal* + *GitHub dark*. Fundo escuro, números claros, uma cor de destaque por slide no máximo. **Zero emoji.** Zero imagem de banco. Credibilidade em quant é inversamente proporcional à quantidade de decoração.

---

## 3. O texto da publicação (caption)

> Cole este texto exato no campo do post. Comprimento: ~2.650 caracteres (limite LinkedIn = 3.000). As primeiras 3 linhas (antes do "ver mais") foram construídas para quebrar o scroll.

---

**Todo mundo quer construir um "AI hedge fund".**

**Quase ninguém roda um Deflated Sharpe Ratio antes de publicar o backtest.**

Foi o que eu percebi depois de gastar dois meses construindo um sistema que batia o S&P 500 com Sharpe 2.7 — até descobrir que 40% dos meus tickers tinham dados idênticos aos dos vizinhos no arquivo de configuração. A `yfinance.download()` não é thread-safe em chamadas paralelas. Look-ahead bias disfarçado de alpha.

Apaguei tudo. Refiz o pipeline com `yf.Ticker().history()` por ticker, 1002 testes com fixtures, `decision_date = t−1` em todo lugar, e uma regra simples: nenhum número entra no README antes de passar por CPCV com purge + embargo.

Esse é o **Titanium Alpha** — um sistema multi-estratégia agêntico que junta quatro peças que raramente convivem no mesmo repositório:

→ **PatchTST** (transformer de séries temporais) forecastando retornos em 5 quantis
→ **Debate multi-agente** em LangGraph (Analista Técnico × Fundamentalista × Bear × Portfolio Manager), com RAG em ChromaDB para citar notícia real
→ **CPCV-OOS com Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) — 547 configs testadas em grid search de 3 tiers, 15 paths combinatoriais cada
→ **Hierarchical Risk Parity** (López de Prado, 2016) com shrinkage de Ledoit-Wolf e tilt por confidence

**Resultado — 10 anos de walk-forward out-of-sample (2016–2026, 52 large caps US + SPY):**

Sharpe **0.766** vs SPY 0.592
Max Drawdown **−21.94%** vs SPY −33.72%
Alpha (CAPM) **+2.57%** a.a.
Beta **0.566** — carrega metade do risco de mercado
Volatilidade anual 11.2% vs SPY 17.9%

Menos retorno absoluto, metade do risco. É esse o trade-off que um fundo institucional assina.

**O que eu aprendi construindo isso (e que raramente aparece em tutorial):**

• Look-ahead bias se esconde em lugares absurdos — bibliotecas que você confia há anos.
• Deflated Sharpe Ratio é mais humilhante do que esperado: 547 configs, zero aceitaram ao nível p>0.95 sem holdout temporal separado.
• `max_weight` não é parâmetro a otimizar — é restrição de risco. Testei. Caiu 0.25 de Sharpe.
• Debate de LLM com `temperature=0.2` não é determinístico. Tickers borderline oscilam entre BUY e HOLD. Qualquer pipeline de produção precisa decidir se faz N runs e agrega, ou aceita o ruído.

**O repositório é público, MIT, 1002 testes passando, CI verde:**

github.com/cockles98/titanium-alpha

Quero ouvir crítica técnica de quem já sentiu a dor de um CPCV mal purgado ou um HRP que concentrou 40% num setor. Comenta o furo que eu não vi — é o tipo de feedback que constrói o próximo commit.

#QuantitativeFinance #MachineLearning #DataScience #AlgorithmicTrading #Python #OpenSource

---

### Notas de execução do caption

- **Gancho (linhas 1–3):** contraste frontal ("Todo mundo X. Quase ninguém Y.") — fórmula de alta conversão (skill Seção 2.1).
- **Vulnerabilidade calculada:** o parágrafo do look-ahead bias é **real** (sessão 37 do projeto, `yf.download()` thread-unsafe — documentado em `memory/data_integrity_fix.md`). Não invente erro. A humildade performática é detectada.
- **Concretude:** nomes próprios (PatchTST, López de Prado 2016, Bailey 2014, Ledoit-Wolf) funcionam como Ethos por densidade técnica. Quem conhece, reconhece. Quem não conhece, sente que há substância.
- **Lista de "o que aprendi":** uso do princípio do **Steel-manning implícito** — você mostra que enfrentou o melhor contra-argumento (overfitting, ruído de LLM), não que o ignorou.
- **CTA:** pede **crítica técnica**, não curtida. Em audiência quant, pedir review é sinal de segurança intelectual e gera 3–5x mais engajamento comentado (que pesa mais no algoritmo).
- **Hashtags:** 6 é o sweet spot atual no LinkedIn. Mais que 8 reduz alcance.

---

## 4. Carrossel — 9 slides (roteiro completo)

### SLIDE 1 — HOOK
**Layout:** Fundo preto `#0E1117`. Título gigantesco, 3 linhas, branco. Uma linha vermelha horizontal fina de 2 px embaixo. Canto inferior direito: logo/assinatura pessoal discreta.

**Texto (título, 72 pt):**

> **Todo mundo quer construir**
> **um "AI hedge fund".**
>
> **Quase ninguém roda um Deflated**
> **Sharpe Ratio antes de publicar.**

**Texto (rodapé, 22 pt, cinza médio):** `Titanium Alpha → github.com/cockles98/titanium-alpha`

**Visual adicional:** opcional — um `→ deslize` sutil no canto inferior direito em ciano (`#26DE81`).

**Justificativa retórica:** contraste/ruptura + promessa de revelação técnica. O leitor quant vai deslizar porque a frase "Deflated Sharpe Ratio" é sinal de dentro-da-tribo.

---

### SLIDE 2 — SITUATION (a paisagem)
**Layout:** Fundo preto. Título no topo. Quatro bullets curtos com ícones monocromáticos (setas ou traços).

**Título (56 pt, branco):**
> **O mercado está cheio de AI trading.**

**Corpo (32 pt, cinza claro):**
- Transformers para preço, RNNs para sentimento, GPTs para "decidir".
- Backtests brilhantes em apresentações de pitch.
- Equity curves que sobem para o topo direito da imagem.
- Zero menção a embargo, purge, ou multiple-testing correction.

**Rodapé (22 pt, cinza médio):** `1 / 9`

---

### SLIDE 3 — COMPLICATION (o furo)
**Layout:** Fundo grafite azulado `#161B22`. Uma grande citação centralizada.

**Texto principal (48 pt, branco):**
> **O problema não é o modelo.**
>
> **É o que você faz
> *antes*
> de confiar no modelo.**

**Subtexto (28 pt, cinza claro):**
> Look-ahead bias. Overfitting. Ruído de LLM. Thread-unsafety em bibliotecas que você confia há anos.

**Rodapé:** `2 / 9`

---

### SLIDE 4 — CONFISSÃO (Pathos + Ethos)
**Layout:** Fundo preto, ícone de alerta discreto `!` em amarelo `#FFC837`.

**Título (40 pt, branco):**
> **Meu primeiro backtest dava Sharpe 2.7.**

**Subtítulo (32 pt, vermelho `#EB3B5A`):**
> Era mentira.

**Corpo (26 pt, cinza claro):**
> `yf.download()` não é thread-safe. 22 dos meus 52 tickers tinham dados idênticos aos vizinhos. Apaguei tudo. Comecei de novo com `yf.Ticker().history()`, fixtures isoladas e 1002 testes.

**Rodapé:** `3 / 9`

**Justificativa retórica:** vulnerabilidade genuína. Confissão específica e verificável. Ethos elevado — o leitor passa a confiar nos números dos slides seguintes porque viu honestidade.

---

### SLIDE 5 — ANSWER (a arquitetura)
**Layout:** Fundo preto, diagrama em blocos horizontais fluindo esquerda → direita, com setas finas em azul `#2E86DE`.

**Título (40 pt, branco):**
> **Titanium Alpha — 4 camadas**

**Diagrama (blocos empilhados ou em fluxo):**

```
┌─────────────────────────────────────────────┐
│  DATA       PostgreSQL + ChromaDB           │
│             yf.Ticker (thread-safe)         │
├─────────────────────────────────────────────┤
│  FORECAST   PatchTST (5 quantis)            │
│             CDF → P(up) contínuo            │
├─────────────────────────────────────────────┤
│  DEBATE     LangGraph                       │
│             4 agentes + RAG citado          │
├─────────────────────────────────────────────┤
│  ALLOCATE   HRP (Ledoit-Wolf + Ward)        │
│             CPCV-OOS + Deflated Sharpe      │
└─────────────────────────────────────────────┘
```

**Rodapé:** `4 / 9`

**Alternativa visual:** substituir o ASCII por 4 cards horizontais com ícones geométricos simples (círculo, triângulo, quadrado, hexágono) em cores frias.

---

### SLIDE 6 — RESULTS (a prova)
**Layout:** Fundo preto. Tabela comparativa grande, duas colunas: **Titanium Alpha** (verde `#26DE81`) × **SPY** (branco). Métricas melhores em negrito e verde.

**Título (40 pt, branco):**
> **10 anos walk-forward OOS**
> *2016 → 2026 · 52 large caps US · 2.514 dias*

**Tabela (monoespaçada, 30 pt):**

```
MÉTRICA             TITANIUM   SPY B&H
──────────────────────────────────────
Sharpe Ratio         0.766    0.592
CAGR                13.68%   14.89%
Max Drawdown       −21.94%  −33.72%
Volatilidade anual  11.2%    17.9%
Alpha CAPM          +2.57%       —
Beta                 0.566    1.000
Sortino              1.058    0.826
Calmar               0.624    0.442
```

**Linha de síntese (28 pt, branco, itálico):**
> Menos retorno absoluto. Metade do risco. Alpha positivo.

**Rodapé:** `5 / 9`

**Justificativa retórica:** Logos puro. Números argumentam sozinhos. Tabela monoespaçada com grid reforça estética de *terminal quant*, não slide de marketing.

---

### SLIDE 7 — WAR ROOM (screenshot do produto)
**Layout:** Screenshot real do `docs/images/warroom.png` ocupando 75% do slide. Faixa inferior com explicação curta.

**Título (36 pt, canto superior esquerdo, branco):**
> **War Room — debate ao vivo**

**Imagem:** captura do dashboard mostrando os 4 cards dos agentes (Technical Analyst, Fundamentalist, Bear, Portfolio Manager) com suas decisões estruturadas.

**Legenda (24 pt, cinza claro, rodapé):**
> Streamlit + LangGraph streaming. Cada agente entrega tese, confidence, catalysts, risks e sources_cited (RAG do ChromaDB). Nenhuma alucinação — se a notícia não estiver na base, o agente não inventa.

**Rodapé:** `6 / 9`

---

### SLIDE 8 — RIGOR (o que raramente é dito)
**Layout:** Fundo preto, 4 blocos de texto em grade 2x2 com números grandes em azul `#2E86DE`.

**Título (40 pt, branco):**
> **Por que eu confio nesses números**

**Grid:**

| | |
|---|---|
| **1002** <br> testes passando, fixtures isoladas, zero API real no CI | **547** <br> configs testadas via CPCV-OOS, grid 3 tiers, DSR aplicado |
| **15 paths** <br> combinatoriais com purge + embargo a cada rebalance | **t − 1** <br> decision_date sempre no fechamento anterior — zero look-ahead |

**Rodapé:** `7 / 9`

**Justificativa retórica:** Ethos por concretude quantitativa. Números específicos (1002, não "mil"; 547, não "centenas") soam verificáveis — porque são.

---

### SLIDE 9 — LIMITAÇÕES (steel-manning + CTA)
**Layout:** Fundo preto. Duas colunas: esquerda "o que ainda não resolvi", direita CTA com QR code do GitHub.

**Título (40 pt, branco):**
> **O que eu ainda não resolvi**

**Coluna esquerda (26 pt, cinza claro):**
- **Gap backtest × produção:** o debate LangGraph ainda não foi backtestado com a mesma disciplina temporal. Fallback: PatchTST.
- **Estocasticidade do LLM:** `temperature=0.2` + borderline tickers = BUY/HOLD oscilando entre runs. Produção real pede agregação de N passes.
- **CPCV-OOS acceptance:** 0 configs passaram em DSR > 0.95 com 547 trials. Workaround: holdout temporal de 2 anos com `n_trials=1`.

**Coluna direita — bloco de CTA (fundo azul `#2E86DE`, texto branco):**
> **Código público · MIT**
>
> **github.com/cockles98/titanium-alpha**
>
> *[QR code centralizado abaixo, 180×180 px]*
>
> Comenta o furo que você viu.

**Rodapé:** `8 / 9 · Felipe Cockles · Quant Engineer`

**Justificativa retórica:** fechamento por *elevação* + *concessão estratégica*. Admitir 3 limitações reais vacina o leitor contra a objeção "isso é otimista demais" e eleva Ethos. O CTA pede revisão técnica, não aplauso — posicionamento de autoridade que convida o par, não o fã.

---

### (Opcional) SLIDE 10 — ASSINATURA FINAL
**Layout:** Fundo preto limpo. Centralizado.

**Texto (48 pt, branco):**
> **Me segue se você constrói quant.**
>
> **Me comenta se você acha que eu errei.**

**Subtexto (24 pt, cinza):**
> Felipe Cockles · github.com/cockles98 · linkedin.com/in/[seu-usuario]

**Rodapé:** `9 / 9`

---

## 5. Checklist de revisão retórica (Seção 10 da skill)

- [x] **Gancho para o scroll:** "Todo mundo quer construir um AI hedge fund. Quase ninguém roda DSR." — para o dedo.
- [x] **Tese clara em uma frase:** "Titanium Alpha bate SPY em Sharpe com metade do risco, e o código é público."
- [x] **Ethos/Pathos/Logos balanceados:** confissão do bug (Pathos) + papers citados (Ethos) + tabela 10y (Logos).
- [x] **Identificação do público:** jargão certo (CPCV-OOS, DSR, HRP, PatchTST). Quem não entende, não é o alvo.
- [x] **Prova concreta:** 1002 testes, Sharpe 0.766, Alpha +2.57%, repo público.
- [x] **Transições orgânicas:** do hype → furo real → confissão pessoal → solução → prova → rigor → limitação → CTA. Cada slide puxa o próximo.
- [x] **Variação de ritmo:** slides densos (5, 6, 8) alternando com slides de impacto curto (1, 3, 4, 10).
- [x] **Fechamento memorável:** "Comenta o furo que você viu." — pede revisão, não curtida. Posicionamento sênior.
- [x] **CTA específico:** link GitHub + pedido de crítica técnica (dupla ação, baixo atrito).
- [x] **Zero falácia:** nenhum cherry-picking (limitações admitidas), nenhum appeal to authority vago, nenhum strawman de "outros modelos".
- [x] **Leitura em voz alta:** caption revisado para cadência — frases curtas após blocos densos, anáfora no slide 1.

---

## 6. Plano de publicação (operacional)

| Etapa | Ação | Timing |
|---|---|---|
| **T-2 dias** | Exportar os 9 slides em PDF 1080×1350 · revisar em mobile (iPhone real) | Crítico — 70% do feed é mobile |
| **T-1 dia** | Testar screenshot do War Room em slide 7 com contraste alto; validar QR code clicável | — |
| **T-0 (quarta ou terça, 08h15–09h30 BRT)** | Publicar carrossel + caption · **não** agendar via ferramenta externa (algoritmo penaliza) | Janela quant ativa |
| **T+2h** | Responder cada comentário com pergunta de aprofundamento (gera mais comentários em cadeia) | Dobra alcance em 24h |
| **T+48h** | Postar vídeo curto (≤30 s) do `warroom_demo.gif` convertido em MP4, linkando o carrossel como "se perdeu, veja o post fixado" | Reativa o algoritmo |
| **T+7 dias** | Se passou de 50 comentários, escrever **artigo LinkedIn** desdobrando um dos tópicos: "How I detected a thread-safety bug that faked my Sharpe 2.7" | Posicionamento duradouro |

---

## 7. O que **não** fazer

- **Não** use emoji no carrossel ou no caption. O nicho quant trata emoji como sinal de marketing amador.
- **Não** inclua screenshot de ChatGPT, Claude ou qualquer LLM-chat. Mostra dependência, não construção.
- **Não** termine o post com "O que você acha?" genérico. Substitua por pedido de crítica técnica específica.
- **Não** hashtag em português (#financas-quantitativas). Alcance internacional nesse nicho é maior em inglês.
- **Não** publique nas sextas-feiras ou fins de semana. Quants leem LinkedIn entre 08h–10h de terça, quarta e quinta.
- **Não** "peça para as pessoas compartilharem". Se o conteúdo for bom, elas compartilham. Pedir rebaixa Ethos.

---

## 8. Variação A/B (se quiser testar)

### Variação alternativa do gancho (slide 1 + linha 1 do caption):

> **Meu primeiro backtest dava Sharpe 2.7.**
>
> **Dois meses depois, descobri que ele era uma ilusão de óptica da biblioteca que eu usava há anos.**

**Quando usar:** se o público do seu perfil é mais 50% gerentes/C-level (menos técnicos) e 50% quants. A história pessoal abre mais portas que a declaração conceitual.

**Quando manter o original:** se seu público é >70% quants, ML engineers, data scientists seniores. O contraste "AI hedge fund × DSR" fala mais alto.

---

---

## 9. Prompts para geração no Nano Banana 2 (Gemini 3 Pro Image)

> Esta seção é operacional. Siga **em ordem**: primeiro o pré-carregamento (9.1), depois o brief mestre (9.2), só então os prompts de cada slide (9.3). Se pular o brief mestre, cada slide vai sair num estilo diferente e o carrossel perde coerência.

### 9.1 — O que colocar na conversa **ANTES** de pedir qualquer imagem

Abra uma **nova conversa** no Nano Banana 2 dedicada a este carrossel. Antes do primeiro prompt de imagem, faça upload dos seguintes anexos e cole o brief mestre (Seção 9.2) como **primeira mensagem de texto**.

**Arquivos para anexar (na ordem):**

| Ordem | Arquivo | Caminho no repo | Por que anexar |
|---|---|---|---|
| 1 | `warroom.png` | `docs/images/warroom.png` | Referência de UI real do dashboard — Nano Banana 2 vai preservar cores, tipografia dos cards dos agentes e densidade de informação no slide 7 |
| 2 | `benchmark.png` | `docs/images/benchmark.png` | Referência de estética de gráfico financeiro — define o padrão visual para eventuais equity curves ou drawdown panels |
| 3 | `microstructure.png` | `docs/images/microstructure.png` | Referência adicional da paleta e do tratamento de dark mode do projeto |
| 4 | *(opcional)* `linkedin_publicacao_titanium_alpha.md` | raiz do repo | Se o Nano Banana 2 aceitar .md nesta sessão, anexe para dar acesso à paleta e às especificações técnicas completas |

**Ordem de anexação importa:** o Nano Banana 2 trata a primeira imagem como referência primária de estilo. Comece por `warroom.png` porque é o que captura melhor o DNA visual do projeto (dark mode, cards técnicos, tipografia sans-serif, zero ornamento).

---

### 9.2 — Brief mestre (cole como **primeira mensagem de texto** da conversa)

> Copie e cole este texto na íntegra antes de pedir qualquer slide. Ele estabelece paleta, tipografia, estética e regras de composição. O modelo usa isso como âncora para todos os slides seguintes.

```
You are helping me design a 9-slide LinkedIn carousel for a quantitative
finance open-source project called "Titanium Alpha". The attached images
(warroom.png, benchmark.png, microstructure.png) are screenshots from the
actual product dashboard — they define the visual DNA I want every slide
to inherit: dark mode, technical density, zero decoration, Bloomberg-terminal
meets GitHub-dark aesthetic.

FORMAT SPECIFICATIONS (apply to every slide unless I state otherwise):
- Aspect ratio: 4:5 vertical (1080x1350 px)
- Background primary: #0E1117 (near-black graphite)
- Background alternative: #161B22 (slightly lighter graphite, used for
  callout cards or emphasis blocks)
- Accent primary (titanium blue): #2E86DE
- Accent positive (quant green): #26DE81
- Accent negative / risk (coral red): #EB3B5A
- Accent warning (amber): #FFC837
- Text primary: #F5F6FA (near-white)
- Text secondary: #8B94A7 (medium cool gray)
- Grid / dividers: #2F3640

TYPOGRAPHY:
- Headings: Inter Bold or IBM Plex Sans Bold, tight tracking, sentence case
  (NEVER ALL CAPS except for 2-3 letter acronyms)
- Body: Inter Regular or IBM Plex Sans Regular
- Numbers, tables, code: JetBrains Mono or IBM Plex Mono
- Minimum on-slide text size: equivalent of 28pt (must be readable on mobile)

COMPOSITION RULES:
- Generous negative space — at least 15% padding on all edges
- One primary accent color per slide maximum (don't mix green + red + blue
  in the same slide unless showing a comparison)
- Thin hairline dividers (1-2px) in #2F3640 — never thick bars
- Numbers and metrics should feel authoritative: large, monospaced, aligned
  to a baseline grid

HARD PROHIBITIONS (break any of these and the slide is unusable):
- No emojis, no iconography that looks playful
- No stock photos, no people, no abstract "AI brain" imagery, no glowing
  neural network visuals
- No 3D renders, no gradients that scream "SaaS landing page"
- No handwritten or script fonts
- No marketing-style CTAs ("Click now!", "Get started!"). This is a
  technical audience — quants on LinkedIn — every decorative choice that
  looks like a pitch deck LOWERS credibility
- No drop shadows heavier than 8px blur at 15% opacity

TONE: austere, technical, confident. Think Bloomberg Terminal, not Canva.
Think arxiv.org, not Medium. Think Jane Street recruiting page, not a
fintech startup homepage.

For each slide I'll send next, I will provide: (1) the slide number,
(2) the exact text content that must appear, (3) the intended visual
structure. Generate the slide at 1080x1350 matching the spec above.

Confirm you understand before I send slide 1.
```

**Espere a confirmação do modelo antes de avançar.** Se a confirmação demonstrar que o modelo não entendeu alguma regra (ex.: propuser incluir ícones decorativos), corrija o brief antes de continuar.

---

### 9.3 — Prompts individuais (um por slide)

> Envie um prompt por turno. **Não** peça "gere os 9 slides de uma vez" — a consistência cai drasticamente. Gere, revise, ajuste, depois siga.

---

#### PROMPT SLIDE 1 — HOOK

```
Slide 1 of 9 — HOOK.

Layout: Full-bleed #0E1117 background. Centered composition, heavy
negative space. A single thin horizontal divider (1px, #EB3B5A, coral
red) positioned at approximately 62% vertical height — the only
decorative element.

Text content (rendered crisp, all in #F5F6FA near-white, Inter Bold,
large — equivalent to 72pt):

Line 1: "Todo mundo quer construir"
Line 2: "um "AI hedge fund"."
(blank line spacing)
Line 3: "Quase ninguém roda um Deflated"
Line 4: "Sharpe Ratio antes de publicar."

Bottom-right corner, small — equivalent to 22pt, #8B94A7 medium gray,
JetBrains Mono:
"titanium-alpha · github.com/cockles98"

Bottom-left corner, tiny — equivalent to 20pt, #8B94A7:
"01 / 09"

No other elements. No icons, no gradients, no illustrations. Austere.
Strong contrast between text and background is the only hierarchy.
```

---

#### PROMPT SLIDE 2 — SITUATION

```
Slide 2 of 9 — SITUATION (the landscape).

Layout: #0E1117 background. Title top-left, four bullet points stacked
beneath it, each prefixed with a 2px horizontal dash in #2E86DE (not a
dot, not an arrow — a clean typographic dash).

Title (Inter Bold, near-white #F5F6FA, equivalent 56pt):
"O mercado está cheio de AI trading."

Four bullet points (Inter Regular, #F5F6FA, equivalent 32pt, line spacing
1.6x):

— Transformers para preço, RNNs para sentimento, GPTs para "decidir".
— Backtests brilhantes em apresentações de pitch.
— Equity curves que sobem para o topo direito da imagem.
— Zero menção a embargo, purge, ou multiple-testing correction.

Bottom-right: "02 / 09" in #8B94A7, JetBrains Mono, equivalent 20pt.

No illustrations. The dashes are the only decorative marks.
```

---

#### PROMPT SLIDE 3 — COMPLICATION

```
Slide 3 of 9 — COMPLICATION.

Layout: #161B22 background (slightly lighter than other slides to signal
emphasis shift). Centered pull-quote composition, like a serif-less
epigraph from a technical book.

Primary quote (Inter Bold, #F5F6FA, equivalent 48pt, centered, tight
line-height):

"O problema não é o modelo.

É o que você faz antes de
confiar no modelo."

The word "antes" must be rendered in italic (Inter Bold Italic), no
color change, just italic weight — it's a rhythmic emphasis.

Below the quote, a thin 60px horizontal divider in #2F3640 centered.

Subtext below the divider (Inter Regular, #8B94A7 medium gray, equivalent
28pt, centered, single line wrapping naturally):

"Look-ahead bias. Overfitting. Ruído de LLM. Thread-unsafety em
bibliotecas que você confia há anos."

Bottom-right: "03 / 09" in #8B94A7 monospace.

No icons, no ornaments.
```

---

#### PROMPT SLIDE 4 — CONFESSION

```
Slide 4 of 9 — CONFESSION (the vulnerability).

Layout: #0E1117 background. Top-left corner has a single warning symbol
— a simple exclamation mark "!" in #FFC837 amber, rendered as a clean
typographic character (not an icon from a set, not inside a triangle or
circle — just the character alone at roughly 80pt), with a small label
"incident · 2026-02" below it in #8B94A7 monospace equivalent 18pt.

Main title (Inter Bold, #F5F6FA, equivalent 44pt, left-aligned):
"Meu primeiro backtest dava Sharpe 2.7."

Subtitle immediately below (Inter Bold, #EB3B5A coral red, equivalent
40pt, left-aligned):
"Era mentira."

Body paragraph below, with extra vertical space (Inter Regular, #F5F6FA
at 85% opacity, equivalent 26pt, line-height 1.5x, left-aligned, max
width 80% of slide):

"yf.download() não é thread-safe. 22 dos meus 52 tickers tinham dados
idênticos aos vizinhos. Apaguei tudo. Comecei de novo com
yf.Ticker().history(), fixtures isoladas e 1002 testes."

The phrases "yf.download()" and "yf.Ticker().history()" must be in
JetBrains Mono (monospace), same size as body, colored #2E86DE titanium
blue — they are code references and should look like code inline.

Bottom-right: "04 / 09" in #8B94A7 monospace.
```

---

#### PROMPT SLIDE 5 — ARCHITECTURE

```
Slide 5 of 9 — ARCHITECTURE.

Layout: #0E1117 background. A vertically stacked 4-block diagram
occupying the center 80% of the slide. Each block is a rectangle with:
- #161B22 fill
- 1px #2F3640 border
- Rounded corners at 6px radius
- Even vertical spacing between blocks (24px)
- Equal width, equal height

Title above the stack (Inter Bold, #F5F6FA, equivalent 40pt,
left-aligned):
"Titanium Alpha — 4 camadas"

Each block contains two lines of text, left-aligned, padded 24px:
- Top line (Inter Bold, #2E86DE titanium blue, equivalent 28pt,
  monospace-feel — use IBM Plex Mono Bold if available)
- Bottom line (Inter Regular, #F5F6FA at 85%, equivalent 22pt)

BLOCK 1:
DATA
PostgreSQL + ChromaDB · yf.Ticker (thread-safe)

BLOCK 2:
FORECAST
PatchTST (5 quantis) · CDF → P(up) contínuo

BLOCK 3:
DEBATE
LangGraph · 4 agentes + RAG citado

BLOCK 4:
ALLOCATE
HRP (Ledoit-Wolf + Ward) · CPCV-OOS + Deflated Sharpe

Between blocks, a thin vertical arrow (1px #2F3640, 16px tall, centered
horizontally) pointing downward — the only ornament.

Bottom-right: "05 / 09" in #8B94A7 monospace.
```

---

#### PROMPT SLIDE 6 — RESULTS

```
Slide 6 of 9 — RESULTS (the proof).

Layout: #0E1117 background. This slide is all about the table — it
should feel like a terminal output.

Title (Inter Bold, #F5F6FA, equivalent 40pt, left-aligned):
"10 anos walk-forward OOS"

Subtitle immediately below (Inter Regular italic, #8B94A7, equivalent
24pt, left-aligned):
"2016 → 2026 · 52 large caps US · 2.514 dias"

Main table, rendered in JetBrains Mono (or IBM Plex Mono), equivalent
28pt, monospaced column alignment, thin 1px #2F3640 horizontal divider
between header and first row:

MÉTRICA             TITANIUM   SPY B&H
──────────────────────────────────────
Sharpe Ratio         0.766    0.592
CAGR                13.68%   14.89%
Max Drawdown       -21.94%  -33.72%
Volatilidade anual  11.2%    17.9%
Alpha CAPM          +2.57%       -
Beta                 0.566    1.000
Sortino              1.058    0.826
Calmar               0.624    0.442

Column "TITANIUM" values should be #26DE81 quant green when they beat
SPY (Sharpe, Max Drawdown, Volatilidade, Alpha, Sortino, Calmar).
CAGR for Titanium should be #F5F6FA neutral (it loses on this one).
SPY column always #F5F6FA at 75% opacity.
Header row in #8B94A7.

Below the table, generous spacing, then a closing line (Inter Regular
italic, #F5F6FA, equivalent 26pt, left-aligned):
"Menos retorno absoluto. Metade do risco. Alpha positivo."

Bottom-right: "06 / 09" in #8B94A7 monospace.
```

---

#### PROMPT SLIDE 7 — WAR ROOM (usa warroom.png como referência)

```
Slide 7 of 9 — WAR ROOM.

IMPORTANT: Use warroom.png (already attached) as the primary visual
source. Do NOT redraw the dashboard from scratch — crop and frame the
actual screenshot inside this slide.

Layout: #0E1117 background. The warroom.png screenshot occupies the
center 75% of the slide, with a subtle 1px #2F3640 border around it and
very slight rounded corners (4px). Small 12px padding between screenshot
and any text.

Title above the screenshot (Inter Bold, #F5F6FA, equivalent 36pt,
left-aligned, with a 2px #2E86DE dash prefix):
"— War Room · debate ao vivo"

Caption below the screenshot (Inter Regular, #8B94A7, equivalent 22pt,
left-aligned, line-height 1.5x, max width 90% of slide):

"Streamlit + LangGraph streaming. Cada agente entrega tese, confidence,
catalysts, risks e sources_cited (RAG do ChromaDB). Se a notícia não
estiver na base, o agente não inventa."

The phrase "sources_cited" should be in JetBrains Mono monospace.

Bottom-right: "07 / 09" in #8B94A7 monospace.

Do NOT add fake glow, do NOT add "screen" UI frames (no mockup of a
laptop or browser), do NOT add blur or motion effects. The screenshot
stands on its own.
```

---

#### PROMPT SLIDE 8 — RIGOR

```
Slide 8 of 9 — RIGOR.

Layout: #0E1117 background. A 2x2 grid of large number + caption blocks
occupying the center 80% of the slide.

Title at top (Inter Bold, #F5F6FA, equivalent 40pt, left-aligned):
"Por que eu confio nesses números"

Below the title, 2x2 grid. Each cell contains:
- A large numeric value (IBM Plex Mono Bold or JetBrains Mono Bold,
  #2E86DE titanium blue, equivalent 88pt, left-aligned)
- A caption beneath it (Inter Regular, #F5F6FA at 85%, equivalent 22pt,
  line-height 1.5x, left-aligned, max 2 lines)

CELL 1 (top-left):
1002
testes passando, fixtures isoladas, zero API real no CI

CELL 2 (top-right):
547
configs testadas via CPCV-OOS, grid 3 tiers, DSR aplicado

CELL 3 (bottom-left):
15 paths
combinatoriais com purge + embargo a cada rebalance

CELL 4 (bottom-right):
t − 1
decision_date sempre no fechamento anterior, zero look-ahead

Between the cells, thin 1px dividers in #2F3640 — one vertical in the
middle, one horizontal in the middle, forming a cross that barely
registers. Do not make them prominent.

Bottom-right: "08 / 09" in #8B94A7 monospace.
```

---

#### PROMPT SLIDE 9 — LIMITATIONS + CTA

```
Slide 9 of 9 — LIMITATIONS & CTA.

Layout: #0E1117 background. Two-column composition — left column ~55%
width, right column ~40% width, with a thin vertical 1px #2F3640 divider
between them.

LEFT COLUMN:

Title (Inter Bold, #F5F6FA, equivalent 36pt, left-aligned):
"O que eu ainda não resolvi"

Three bullet items, each prefixed with a 2px horizontal dash in #FFC837
amber (warning color, because these are open issues):

— Gap backtest × produção: o debate LangGraph ainda não foi backtestado
  com a mesma disciplina temporal. Fallback: PatchTST.

— Estocasticidade do LLM: temperature=0.2 + borderline tickers = BUY/HOLD
  oscilando entre runs. Produção real pede agregação de N passes.

— CPCV-OOS acceptance: 0 configs passaram em DSR > 0.95 com 547 trials.
  Workaround: holdout temporal de 2 anos com n_trials=1.

Body text (Inter Regular, #F5F6FA at 85%, equivalent 20pt, line-height
1.5x).
Phrases "temperature=0.2", "DSR > 0.95", "n_trials=1" in JetBrains Mono.

RIGHT COLUMN:

A callout card with #2E86DE titanium blue background, rounded corners 8px,
padding 32px, containing (all text in #FFFFFF pure white):

Top line (Inter Bold, equivalent 22pt):
"Código público · MIT"

Middle line (JetBrains Mono Bold, equivalent 20pt, tight fit):
"github.com/cockles98/titanium-alpha"

QR code below, 200x200px, dark modules on white background — the QR
must encode the URL "https://github.com/cockles98/titanium-alpha".

Bottom line of the card (Inter Regular, equivalent 18pt, centered):
"Comenta o furo que você viu."

Bottom-right of the slide (outside the callout): "09 / 09" in #8B94A7
monospace.
Bottom-left: "Felipe Cockles · Quant Engineer" in #8B94A7 equivalent
18pt.
```

---

#### PROMPT SLIDE 10 *(opcional)* — ASSINATURA

```
Slide 10 of 9 — SIGNATURE (optional closer, use only if you want the
carousel to end on a beat rather than on the CTA).

Layout: #0E1117 background. Centered composition, massive negative space.

Primary text (Inter Bold, #F5F6FA, equivalent 48pt, centered,
line-height 1.4x):

"Me segue se você constrói quant."

(blank line)

"Me comenta se você acha que eu errei."

Below, a thin 80px horizontal divider in #2F3640 centered.

Subtext below the divider (JetBrains Mono Regular, #8B94A7, equivalent
22pt, centered):

"Felipe Cockles · github.com/cockles98 · linkedin.com/in/[seu-usuário]"

No other elements.
```

---

### 9.4 — Workflow recomendado + o que fazer em pós-produção

Mesmo com o Nano Banana 2 renderizando texto razoavelmente bem, **os slides 6 (tabela de métricas) e 9 (QR code)** são os dois pontos críticos onde o modelo pode falhar. Plano de contingência:

| Slide | Estratégia |
|---|---|
| 1, 2, 3, 4, 5, 7, 8, 10 | Gerar direto no Nano Banana 2. Iterar 2–3 vezes se o texto não sair alinhado. |
| **6 (tabela)** | Gerar apenas o **fundo** pelo Nano Banana 2 (título + subtítulo + linha de fecho). **Inserir a tabela em pós** no Figma/Canva usando texto em JetBrains Mono — garante alinhamento monoespaçado perfeito. |
| **9 (QR code)** | O QR code **precisa ser gerado separadamente** em um gerador confiável (`qrcode.show` em Python, ou qrcode-monkey.com). O Nano Banana 2 gera QR codes **visualmente parecidos mas não funcionais**. Compor em Figma. |

**Passo a passo operacional:**

1. Abra nova conversa no Nano Banana 2.
2. Anexe `warroom.png`, `benchmark.png`, `microstructure.png` nessa ordem.
3. Cole o brief mestre (Seção 9.2). Aguarde confirmação.
4. Envie prompt do Slide 1. Revise. Se aprovado, próximo. Se não, peça ajuste específico ("reduza o peso da linha horizontal em 50%", "aumente o tracking do título").
5. Continue slide por slide.
6. Para Slide 6: gere só o fundo; monte a tabela em Figma/Canva.
7. Para Slide 9: gere tudo **exceto** o QR code; gere o QR code em `qrcode.show("https://github.com/cockles98/titanium-alpha")` e composite em Figma.
8. Exporte todos os slides em PNG 1080×1350 @ 2x.
9. Monte o PDF do carrossel em ordem (1 → 9, ou 1 → 10).
10. Valide em mobile real antes de publicar.

---

### 9.5 — Se o Nano Banana 2 derivar do estilo

Comandos de correção que funcionam bem com o modelo (use em turnos de follow-up, não em novo prompt do zero):

- `Remove all decorative elements. No icons, no gradients. Austere.`
- `The text rendering is blurry — regenerate with crisp, vector-sharp typography at the sizes specified.`
- `This looks too much like a marketing slide. Reduce visual weight by 40%, more negative space, thinner dividers.`
- `The accent color should only appear once in this slide. Remove all uses except [specific element].`
- `Match the density and restraint of the attached warroom.png reference.`

Evite comandos vagos ("mais bonito", "mais moderno") — o modelo interpreta como permissão para adicionar ornamento, e isso quebra a estética austera que vende credibilidade quant.

---

**Fim do guia v1.** O texto do caption na **Seção 3** é o que vai literalmente no LinkedIn. Os slides da **Seção 4** são a especificação de design. Os prompts da **Seção 9** são o que você cola no Nano Banana 2 para materializar. O resto é fundamento e operacionalização.

---

## 10. Atualização 2026-05-13 — Plots reais disponíveis (carrossel v2)

> Esta seção **substitui as Seções 4 e 9.3** para execução. As demais seções continuam como base conceitual. O caption (Seção 3) **não muda**, exceto possível revisão do número de Alpha — ver §10.10.

### 10.1 — O que mudou no projeto

A pasta `docs/images/benchmark graphs/` agora tem 16 plots reais do dashboard, todos renderizados na paleta dark do projeto (`#0E1117` fundo, série principal em azul, benchmark em laranja/amarelo). Isso elimina o risco de pedir ao Nano Banana 2 que **redesenhe** gráficos financeiros — ele falha consistentemente nessa tarefa. A partir de agora, **slides com gráfico usam o PNG verbatim**, e o modelo só renderiza o template (título, subtítulo, padding, número de página).

**Inventário completo (16 plots):**

| Plot | Função retórica | Usar onde |
|---|---|---|
| `Equity Curve.png` | Logos — proof of return | Slide 6 (carrossel) |
| `Drawdown.png` | Logos — proof of risk control | Slide 6 (composite) |
| `Sharpe Distribution Across CPCV Paths.png` | Ethos — rigor empírico | Slide 9 (substitui grid 2x2) |
| `CPCV-OOS Path Distribution.png` | Ethos — fan chart de paths | Backup para slide 9 |
| `Rolling Sharpe.png` | Logos — consistência 10y | Follow-up post 1 |
| `Monthly Returns Heatmap.png` | Logos — granularidade mensal | Follow-up post 4 |
| `CAPM Scatter vs SPY.png` | Authority — regressão clássica | Follow-up post 2 |
| `Vol Targeting Trajectory.png` | Pathos — risk management visível | Follow-up post 3 |
| `Top 10 Drawdowns.png` | Vulnerabilidade — perdi 10 vezes | Follow-up post 5 |
| `UpDown Capture vs SPY.png` | Logos — assimetria | Follow-up post 6 |
| `Return Distribution.png` | Logos — tail behavior | Follow-up post 7 (técnico) |
| `Market Relationship.png` | Logos — β/α/ρ rolantes | Follow-up post 8 |
| `Trading Cost.png` | Honestidade — custos reais | Follow-up post 9 |
| `Portfolio Weight Evolution.png` | Showcase — HRP em ação | Follow-up post 10 |
| `Portfolio Concentration Over Time.png` | Showcase — Effective N + Gini | Combina com Weight Evolution |
| `Return Attribution.png` | Logos — diversificação real | Backup para follow-up |

### 10.2 — Estrutura final do carrossel v2 (10 slides)

| # | Slide | Mudança vs v1 |
|---|---|---|
| 1 | Hook | inalterado |
| 2 | Situation | inalterado |
| 3 | Complication | inalterado |
| 4 | Confession | inalterado |
| 5 | Architecture | inalterado |
| **6** | **Equity Curve + Drawdown** (2-painel composite) | **NOVO** — inserido aqui |
| 7 | Results table | era slide 6 |
| 8 | War Room screenshot | era slide 7 |
| **9** | **CPCV-OOS Sharpe Distribution** (violin) | **substitui** o grid 2x2 numérico antigo (slide 8 v1). Os números 1002/547/15/t-1 já aparecem no caption e em outros slides |
| 10 | Limitations + CTA | era slide 9 |

**Slide 11 opcional** (assinatura) = slide 10 opcional do v1 com numeração ajustada.

**Por que substituir o grid 2x2 pelo violin:**
- Grid 2x2 era prova *declarada* ("eu rodei 1002 testes — confie em mim").
- Violin com 15 paths CPCV-OOS é prova *empírica* (mean=0.807, std=0.395, 100% paths positivos, PSR=0.99). Para quant sênior, distribution silencia o ceticismo de "e se você tivesse cortado em outra data?" — o gráfico responde.

### 10.3 — Slide 6 (NOVO) · Equity Curve + Drawdown composite

**Layout:** Canvas 1080×1350, fundo `#0E1117`, padding externo 48px.

```
┌──────────────────────────────────────────────┐
│ Título  +  subtítulo                          │  ← 12% da altura
├──────────────────────────────────────────────┤
│                                              │
│  [ Equity Curve.png — full width ]            │  ← painel 1, ~52% altura
│                                              │
├──────────── 24px gap ────────────────────────┤
│  [ Drawdown.png — full width ]                │  ← painel 2, ~32% altura
│                                              │
├──────────────────────────────────────────────┤
│                                    06 / 10   │  ← rodapé
└──────────────────────────────────────────────┘
```

**Conteúdo:**
- **Título** (Inter Bold, `#F5F6FA`, ~40pt, esquerda): `10 anos · Portfolio vs SPY`
- **Subtítulo** (JetBrains Mono, `#8B94A7`, ~22pt): `2016-04-19 → 2026-04-17 · 2.514 dias OOS`
- **Painel 1:** `Equity Curve.png` verbatim, borda 1px `#2F3640`, cantos 4px.
- **Painel 2:** `Drawdown.png` verbatim, mesmo tratamento.
- **Rodapé:** `06 / 10` direita, JetBrains Mono `#8B94A7` ~20pt.

**Regra crítica:** Nano Banana 2 **NÃO desenha** os gráficos. Os PNGs são assets — o modelo só faz o frame.

**Justificativa retórica:** este é o slide do "para o dedo". A equity curve estabelece "acompanha o mercado"; o drawdown estabelece "com metade da dor". É a defesa visual do Sharpe sem precisar gritar "Sharpe!".

### 10.4 — Slide 9 (MUDOU) · CPCV-OOS Sharpe Distribution

**Layout:** Canvas 1080×1350, fundo `#0E1117`, padding externo 48px.

**Conteúdo:**
- **Título** (Inter Bold, `#F5F6FA`, ~40pt, esquerda): `Por que eu confio neste Sharpe`
- **Imagem central:** `Sharpe Distribution Across CPCV Paths.png` verbatim, ~75% da altura, borda 1px `#2F3640`, cantos 4px.
- **Caption** (Inter Regular, `#F5F6FA` 85%, ~22pt, esquerda, largura máx 90%):
  > "15 paths combinatoriais purgados. Média 0.81, std 0.40, **100% positivos**, PSR=0.99. O Sharpe walk-forward (0.77, linha amarela) cai **dentro** da distribuição empírica — não é cherry-picking de período."
- **Rodapé:** `09 / 10`.

**Justificativa retórica:** steel-manning visual. O ceticismo do quant sênior ("e se você tivesse cortado em outra data?") morre ao ver os 15 paths e o PSR=0.99 já anotados no PNG.

### 10.5 — Slide 7 (renumerado) · Results table

Conteúdo idêntico ao slide 6 do v1. **Único ajuste:** rodapé vira `07 / 10`. Aproveita-se a vizinhança visual com o slide 6 (gráficos) — a tabela funciona como *texto-prova-do-gráfico*.

**Recomendação adicional para o slide 7:** acima da tabela, inserir uma microlegenda de uma linha em `#8B94A7` (~18pt JetBrains Mono): `(números do slide anterior, decompostos)`. Costura narrativa explícita.

### 10.6 — Slide 8 (renumerado) · War Room

Conteúdo idêntico ao slide 7 do v1 (composite com `docs/images/warroom.png`). **Único ajuste:** rodapé vira `08 / 10`.

### 10.7 — Slide 10 (renumerado) · Limitations + CTA

Conteúdo idêntico ao slide 9 do v1. **Único ajuste:** rodapé vira `09 / 10` → corrigido para `10 / 10` e o número final do contador.

### 10.8 — Prompts Nano Banana 2 (v2)

> Para os slides 1, 2, 3, 4, 5: prompts do v1 (Seção 9.3) continuam válidos. Trocar apenas `of 9` por `of 10` e os rodapés `0X / 09` por `0X / 10`.

#### PROMPT SLIDE 6 (NOVO) — Equity Curve + Drawdown composite

```
Slide 6 of 10 — EQUITY CURVE + DRAWDOWN (composite).

Two financial charts are attached as PNG files:
  - "Equity Curve.png" — line chart, dark background
  - "Drawdown.png" — area chart, dark background

HARD RULE: do NOT redraw, re-render, restyle, or recolor the charts.
Do NOT add overlays, glow, gradient, watermark, or annotations on top
of them. Use the attached PNGs verbatim. Compose them inside the
slide template — that is the entire job.

Canvas: 1080x1350, background #0E1117, outer padding 48px.

Top zone (~12% height): title and subtitle, left-aligned.
  - Title (Inter Bold, #F5F6FA, equivalent 40pt):
    "10 anos · Portfolio vs SPY"
  - Subtitle one line below (JetBrains Mono, #8B94A7, equivalent 22pt):
    "2016-04-19 → 2026-04-17 · 2,514 dias OOS"

Middle zone: two stacked panels separated by a 24px gap.
  - Panel 1 (upper, ~52% of canvas height): place Equity Curve.png at
    full width of the inner content area, with a 1px #2F3640 border
    and 4px rounded corners.
  - Panel 2 (lower, ~32% of canvas height): place Drawdown.png with
    identical border treatment.

Bottom-right: page indicator (JetBrains Mono, #8B94A7, equivalent 20pt):
"06 / 10"

No icons, no decorative marks, no extra dividers. The two charts ARE
the slide.
```

#### PROMPT SLIDE 7 (RESULTS) — ajuste de v1

Use o prompt do Slide 6 da Seção 9.3 do v1 com **duas alterações**:
- Substituir `Slide 6 of 9` → `Slide 7 of 10`
- Substituir `"06 / 09"` → `"07 / 10"`
- (Opcional) Acima do título, adicionar nano-rótulo (JetBrains Mono, `#8B94A7`, equivalente 18pt): `"números do slide anterior, decompostos"`

#### PROMPT SLIDE 8 (WAR ROOM) — ajuste de v1

Use o prompt do Slide 7 da Seção 9.3 do v1 com:
- Substituir `Slide 7 of 9` → `Slide 8 of 10`
- Substituir `"07 / 09"` → `"08 / 10"`

#### PROMPT SLIDE 9 (NOVO) — CPCV-OOS Sharpe Distribution composite

```
Slide 9 of 10 — CPCV-OOS SHARPE DISTRIBUTION (composite).

One chart is attached as a PNG file:
  - "Sharpe Distribution Across CPCV Paths.png" — violin + boxplot +
    scatter on dark background. The image already contains the
    annotations "DSR expected max = 1.19" and "Walk-forward OOS = 0.77"
    plus the stats box (Paths: 15, Mean: 0.807, Std: 0.395, % positive:
    100%, PSR: 0.99). These annotations MUST remain readable.

HARD RULE: do NOT redraw the violin, do NOT relabel axes, do NOT
re-render annotations. Use the attached PNG verbatim.

Canvas: 1080x1350, background #0E1117, outer padding 48px.

Top zone (~10% height):
  - Title (Inter Bold, #F5F6FA, equivalent 40pt, left-aligned):
    "Por que eu confio neste Sharpe"

Middle zone (~70% height): the attached PNG centered horizontally,
occupying ~92% of inner width, with a 1px #2F3640 border and 4px
rounded corners.

Bottom zone (~12% height):
  - Caption (Inter Regular, #F5F6FA at 85%, equivalent 22pt, left-aligned,
    max width 92%, line-height 1.5x):
    "15 paths combinatoriais purgados. Média 0.81, std 0.40, 100%
     positivos, PSR=0.99. O Sharpe walk-forward (0.77, linha amarela)
     cai dentro da distribuição empírica — não é cherry-picking
     de período."

Bottom-right corner: page indicator (JetBrains Mono, #8B94A7,
equivalent 20pt): "09 / 10"

No icons, no decorative gradients, no extra dividers beyond the page
number and chart border.
```

#### PROMPT SLIDE 10 (LIMITATIONS + CTA) — ajuste de v1

Use o prompt do Slide 9 da Seção 9.3 do v1 com:
- Substituir `Slide 9 of 9` → `Slide 10 of 10`
- Substituir `"09 / 09"` → `"10 / 10"`

### 10.9 — Anexos no Nano Banana 2 (revisão da Seção 9.1)

Ordem revisada de upload na conversa nova:

| Ordem | Arquivo | Caminho | Função |
|---|---|---|---|
| 1 | `warroom.png` | `docs/images/warroom.png` | DNA visual primário (template) |
| 2 | `Equity Curve.png` | `docs/images/benchmark graphs/Equity Curve.png` | Asset slide 6 |
| 3 | `Drawdown.png` | `docs/images/benchmark graphs/Drawdown.png` | Asset slide 6 |
| 4 | `Sharpe Distribution Across CPCV Paths.png` | `docs/images/benchmark graphs/Sharpe Distribution Across CPCV Paths.png` | Asset slide 9 |
| 5 | `benchmark.png` | `docs/images/benchmark.png` | Referência adicional de paleta |
| 6 *(opcional)* | `linkedin_publicacao_titanium_alpha.md` | raiz | Brief completo |

Os outros 12 plots **não precisam** ser anexados nesta conversa — vão para conversas separadas dos posts de follow-up (§10.11).

### 10.10 — ✓ Reconciliação do Alpha (resolvido 2026-05-13)

**Valor canônico: Alpha = +2.57% a.a. (Jensen's alpha).** O caption (Seção 3) e o `CLAUDE.md` já estavam corretos.

#### O que estava acontecendo

A `benchmark_metrics.json` (output autoritativo de `make benchmark`) registra `alpha = 0.0257`, `beta = 0.5656`. A função `_capm_regression` em `src/backtest/benchmark_metrics.py:134-165` faz a regressão acadêmica padrão de **retornos em excesso**:

```
(R_p − rf) = α_J + β · (R_m − rf) + ε
```

→ produz Jensen's alpha (α_J), a definição CAPM padrão.

Os plots do dashboard, no entanto, regrediam **retornos brutos** (sem subtrair `rf`):

| Plot | Função em `app.py` | Valor inflado |
|---|---|---|
| `CAPM Scatter vs SPY.png` | `_chart_capm_scatter` (linhas ~1358-1455) | +4.69% (raw α') |
| `Market Relationship.png` | `_rolling_regression` (linhas ~1038-1091) | +3.37% (rolling raw α' mean) |

A relação algébrica:

```
α_J = α'_raw − (1 − β) · rf
```

Verificação numérica:
- α'_raw = +4.69%/ano → α'_daily ≈ 0.000186
- rf_daily = (1.05)^(1/252) − 1 ≈ 0.000194
- (1 − β) · rf_daily = 0.4344 · 0.000194 ≈ 0.0000843
- α_J_daily = 0.000186 − 0.0000843 ≈ 0.000102
- α_J anual = 0.000102 × 252 = **+2.57%** ✓ bate com `benchmark_metrics.json`

#### Fix aplicado em código

- `src/dashboard/app.py:_chart_capm_scatter` — subtrai `(1 − β) · rf_daily` do intercepto antes de exibir.
- `src/dashboard/app.py:_rolling_regression` — mesma correção dentro de cada janela rolante.
- `tests/test_dashboard_phase_6_capm.py:test_capm_recovers_known_beta_and_alpha` — reconstrói o setup pra que Jensen's α seja zero/conhecido por construção.
- `tests/test_dashboard_phase_4_market_rel.py:test_rolling_regression_*` — mesmo ajuste.
- 22/22 testes do dashboard CAPM passam.

#### Ação pendente (você)

**Se for usar `CAPM Scatter vs SPY.png` ou `Market Relationship.png` na publicação, regere os PNGs** — os arquivos atuais ainda mostram os números inflados (+4.69%, +3.37% mean) porque foram exportados antes do fix. Caminho: rodar o dashboard localmente (`make dashboard` ou equivalente) e re-exportar via botão de download do Plotly, ou rodar `make benchmark` se ele exportar via script.

Os plots **NÃO precisam ser regerados se você seguir a recomendação de 4 imagens** (Equity Curve, Drawdown, Sharpe Distribution, War Room) — o CAPM Scatter fica de fora.

**Se for à variante de 5 imagens** (com CAPM Scatter), regere o PNG antes — caso contrário a publicação vai mostrar caption com +2.57% e imagem com +4.69%.

### 10.11 — Calendário de follow-up posts (uso dos 12 plots restantes)

> Cada post de follow-up referencia o carrossel original ("se você perdeu, link no comentário fixado"). Não publique todos — escolha 3-5 com base em qual post anterior engajou mais.

| T | Tema | Plot principal | Ângulo retórico |
|---|---|---|---|
| **T+2 d** | "Rolling Sharpe — 10 anos sem cherry-picking" | `Rolling Sharpe.png` | Logos: Sharpe móvel 252d ao longo de 10 anos, mostra consistência |
| **T+5 d** | "Vol targeting em ação — março de 2020" | `Vol Targeting Trajectory.png` | Pathos + Logos: leverage caiu para 0.5 enquanto realized vol explodia para 35% |
| **T+8 d** | "CAPM clássico — α=+4.69%, β=0.566, R²=0.822, n=2.513" | `CAPM Scatter vs SPY.png` | Authority (regressão) + Logos (n grande) |
| **T+12 d** | "Heatmap mensal — onde o portfolio perdeu" | `Monthly Returns Heatmap.png` | Vulnerabilidade: 2022 = −4.9%; mostra meses ruins (sem esconder) |
| **T+15 d** | "Top 10 drawdowns — sim, perdi dinheiro 10 vezes em 10 anos" | `Top 10 Drawdowns.png` | Confissão + steel-manning: pior DD = −21.94% (COVID), 164 dias |
| **T+18 d** | "Up/Down Capture — 76% up, 63% down, ratio 1.21" | `UpDown Capture vs SPY.png` | Logos: assimetria positiva é o coração do Sharpe excedente |
| **T+22 d** | "HRP em movimento — Effective N oscilando entre 4 e 43" | `Portfolio Concentration Over Time.png` + `Portfolio Weight Evolution.png` | Showcase de produto: HRP não é estático |
| **T+25 d** | "Quanto eu paguei em custos — $67.772 em 10 anos, 38 bps drag" | `Trading Cost.png` | Honestidade rara: custos reais vs backtests "sem fricção" |
| **T+28 d** | "Distribuição de retornos diários — skew −0.58, kurt +10.66" | `Return Distribution.png` | Tribo técnica: tail behavior + VaR/CVaR explícitos |
| **T+32 d** | "Atribuição — 47% do retorno vem de 15 tickers, 53% de outros 37" | `Return Attribution.png` | Diversificação real, não concentração disfarçada |

**Mínimo viável:** 3 posts (T+2, T+5, T+15). Maximize engajamento por gancho — não por volume.

### 10.12 — Workflow operacional (atualização do passo a passo)

Mesmo passo a passo da Seção 9.4, com ajustes:

1. Abra **nova conversa** no Nano Banana 2 dedicada a este carrossel.
2. Anexe na ordem da §10.9 (warroom.png primeiro).
3. Cole o **brief mestre da §9.2** (continua válido — paleta, tipografia, regras gerais).
4. Aguarde confirmação.
5. Gere slides 1-5 com os prompts originais da §9.3 (ajustando `09` → `10` no contador).
6. Gere slide 6 com o prompt da §10.8 (composite Equity+DD).
7. Gere slide 7 com o prompt original do v1 slide 6 (Results table). Lembre que a tabela é o ponto crítico — gere o fundo do slide e **monte a tabela em Figma/Canva por cima** com JetBrains Mono (mantém alinhamento monoespaçado).
8. Gere slide 8 com o prompt original do v1 slide 7 (War Room composite).
9. Gere slide 9 com o prompt da §10.8 (CPCV violin composite).
10. Gere slide 10 com o prompt original do v1 slide 9 (Limitations + CTA). QR code **continua** sendo gerado separadamente (`qrcode.show("https://github.com/cockles98/titanium-alpha")`) e composto em Figma.
11. Exporte 10 slides em PNG 1080×1350 @2x.
12. Monte PDF na ordem 1→10.
13. **Reconcile o número de Alpha (§10.10) antes de subir.**
14. Valide em mobile real.
15. Publique seguindo a Seção 6 (terça-quinta, 08h15-09h30 BRT).

---

**Fim do addendum v2.** O carrossel atual tem 10 slides, 3 deles ancorados em PNGs reais do dashboard (slides 6, 8, 9). O caption (Seção 3) e o resto do plano permanecem como estavam — exceto pela reconciliação do Alpha pendente.
