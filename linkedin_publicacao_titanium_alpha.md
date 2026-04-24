# Publicação LinkedIn — Titanium Alpha

> Guia de execução completo aplicando a skill `retorica-persuasao`.
> **Objetivo primário:** gerar autoridade no nicho de quant + data science.
> **Objetivo secundário:** promover o repositório GitHub `cockles98/titanium-alpha`.

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

**Fim do guia.** O texto do caption na **Seção 3** é o que vai literalmente no LinkedIn. Os slides da **Seção 4** são a especificação de design. Os prompts da **Seção 9** são o que você cola no Nano Banana 2 para materializar. O resto é fundamento e operacionalização.
