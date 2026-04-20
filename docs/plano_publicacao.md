# Plano de Publicação — Titanium Alpha
**Criado:** 2026-04-19
**Revisado:** 2026-04-19 (v2 — alinhado ao estado real do repositório)
**Objetivo:** Publicar o projeto como portfólio quant de alto impacto no GitHub
**Meta de resultado:** Projeto que demonstra autoridade no setor quant através de metodologia rigorosa + sistema multi-agente LLM funcionando ao vivo

---

## Decisões estratégicas pendentes

Antes de iniciar a Fase 0, **quatro decisões** precisam ser tomadas. Elas determinam o escopo real de várias tarefas abaixo:

| # | Decisão | Opções | Recomendação | Impacto |
|---|---|---|---|---|
| D1 | Provider LLM público | **(a)** Gemini como default (atual no código); **(b)** Anthropic Claude como default (atual no README) | **(a) Gemini** — compatível com o budget de 500 prompts/dia gratuitos; README deve ser ajustado | Fase 0 + Fase 4.1 (README) |
| D2 | Idioma da documentação interna | **(a)** Manter PT em `docs/`, `CLAUDE.md`, memory; README inglês; **(b)** Traduzir tudo para inglês | **(a)** — audiência quant internacional lê pelo README/notebooks; `docs/` em PT é autêntico e preserva o processo | Fase 4 |
| D3 | Publicar `.claude/agents/`? | **(a)** Sim, mostra o workflow agêntico de desenvolvimento (diferencial); **(b)** Não, mover para `.gitignore` | **(a)** — a presença dos agentes de dev é um fator de originalidade e demonstra adoção de Claude Code no workflow | Fase 2 + Fase 5.1 |
| D4 | Escopo dos notebooks de análise | **(a)** 1 notebook consolidado (`methodology_and_results.ipynb`); **(b)** 2 notebooks separados (methodology + results) | **(a)** — menos overhead de manutenção, narrativa mais linear | Fase 4.2 |

> **Ação requerida:** marcar as decisões acima com `[x]` antes de iniciar a Fase 0. Qualquer alternativa escolhida passa a ser a premissa do restante do plano.

**Decisões registradas (sessão 41, 2026-04-19):**
- [x] **D1 = (a) Gemini** — default público; README será ajustado na Fase 0.4
- [x] **D2 = (a) PT interno + EN README** — `docs/`, `CLAUDE.md`, memory em PT; README em inglês
- [x] **D3 = (a) Sim** — publicar `.claude/agents/`; Fase 2 completa
- [x] **D4 = (a) 1 notebook consolidado** — `methodology_and_results.ipynb` único

---

## Estado Atual (Baseline Verificada)

| Componente | Status | Observação |
|---|---|---|
| Walk-forward backtest | ✅ | Sharpe=0.710, CAGR=13.33%, MaxDD=-18.46% |
| CPCV-OOS fine-tuning | ✅ | 982 configs testadas (sessões 39-40) |
| PatchTST predictions | ✅ | Cache funcional, `predictions.parquet` + `forecast.parquet` existem |
| Dashboard (4 abas) | ✅ | Funciona; 17 chamadas `use_container_width` são dívida de deprecação |
| Testes | ✅ | 1002 passando |
| README.md | ✅ (com 1 inconsistência) | 391 linhas, Mermaid, badges, métricas, Quick Start. **Diz "Claude Sonnet agents" mas código defaulta para Gemini** |
| `.env.example` | ✅ | Existe com placeholders |
| CI/CD | ✅ | `.github/workflows/ci.yml` + `lint.yml` configurados |
| `.gitignore` | ✅ | Cobre `.env`, `data/outputs/`, `models/`, notebooks, `.agents/` |
| Pipeline de agentes | ⚠️ | `decisions.json` e `debate_history.json` de 2026-03-22 existem, mas os reports do Fundamentalista dizem "No news context available" → **RAG vazio na última execução** |
| War Room ao vivo | ⚠️ | Implementado; depende de agentes funcionando com modelo Gemini correto |
| ChromaDB / RAG | ⚠️ | Coleção provavelmente vazia (ver observação acima) |
| LICENSE file | ❌ | Badge MIT no README, **mas arquivo LICENSE não existe** |
| GIF demo do War Room | ❌ | Falta gravação |
| Repositório público | ❌ | Privado |
| `data/outputs/_temp_ohlcv.parquet` | ⚠️ | Artefato temporário de run anterior — limpar antes de publicar |

**Lição da sessão 40:** Parâmetros de concentração HRP (`max_weight`) não devem ser otimizados via CPCV-OOS — a restrição `2/n` é princípio de risco, não parâmetro livre. Fine-tuning via CPCV-OOS é válido apenas para parâmetros de timing (`rebalance_every`, `target_vol`).

---

## Cronograma de Alto Nível

```
Fase 0  │ Pre-flight: decisões + verificação de ambiente │ ~0.5 sessão
Fase 1  │ Agentes: debug + teste end-to-end              │ ~2 sessões
Fase 2  │ Refinamento dos agentes de desenvolvimento     │ ~1 sessão
Fase 3  │ Melhorias de produto                           │ ~1.5 sessões
Fase 4  │ Preparação para portfólio                      │ ~1.5 sessões
Fase 5  │ Publicação                                     │ ~0.5 sessão
────────┴───────────────────────────────────────────────────────────
Total estimado: 7 sessões
```

---

## Fase 0 — Pre-flight
**Objetivo:** eliminar erros bobos de ambiente **antes** de gastar tokens rodando o pipeline de agentes.
**Por que existe:** testar o debate com um nome de modelo errado ou API key inválida desperdiça tempo e dá a impressão falsa de que o código está quebrado.

### 0.1 — Validar o nome do modelo Gemini

**Contexto:** `src/agents/graph.py:66` usa `"gemini-3.1-flash-lite-preview"`. Este nome não bate com as convenções observadas no Google AI Studio (ex.: `gemini-2.5-flash-lite`, `gemini-2.0-flash-lite`). **Isto é o primeiro bloqueador a resolver.**

**Tarefas:**
- [x] Consultar https://ai.google.dev/gemini-api/docs/models para listar modelos atuais com tier gratuito de ≥500 req/dia
- [x] Atualizar `_DEFAULT_MODELS["gemini"]` em `src/agents/graph.py` para o nome correto
- [x] Criar smoke test: `python -c "from src.agents.graph import _create_llm; _create_llm(0.2).invoke('Reply with exactly: OK')"` — precisa retornar "OK"
- [x] Confirmar `GEMINI_KEY` está setada no `.env` (não `GOOGLE_API_KEY` — o código lê especificamente `GEMINI_KEY`)

**Critério:** smoke test responde com sucesso em < 3s. ✅ Validado (`gemini-3.1-flash-lite-preview` na página de pricing como free tier).

### 0.2 — Verificar predições e features

**Tarefas:**
- [x] Confirmar `data/outputs/predictions.parquet` existe e tem os 52 tickers + SPY (52 rows, cols=ticker/prob_up/expected_return/last_close)
- [x] Confirmar `data/outputs/forecast.parquet` (quantiles) existe
- [x] Confirmar `data/outputs/features.parquet` existe
- [ ] Se faltar algum: `make predict` para regenerar — **N/A, todos existem**

**Nota:** os 3 parquets são de 2026-03-17/24, anteriores à re-ingestão (0.3). Para Fase 1.2+ com dados atuais, rodar `make predict` antes. No entanto, para o pipeline de agentes (Fase 1.1), predictions são apenas insumo opcional — os agentes usam OHLCV do Postgres + RAG.

**Critério:** os três arquivos existem e carregam sem erro via Polars. ✅

### 0.3 — Verificar RAG (ChromaDB)

**Tarefas:**
- [x] Rodar `docker compose -f docker/docker-compose.yml up -d` para subir PostgreSQL + ChromaDB
- [x] Verificar se a coleção `financial_news` em ChromaDB tem documentos
- [x] DB limpo + re-ingest: OHLCV 12 anos (159.848 rows, 53 tickers, 2014-04-22 → 2026-04-17) + backfill Google News RSS (14.326 artigos, 53 tickers, 2025-01-01 → 2026-04-19)
- [x] Embedding: 14.326 artigos em ChromaDB `financial_news` (100% embedded)
- [x] Validar que `NEWSAPI_KEY` existe no `.env` — **N/A** (fallback para Google News RSS foi adotado, sem API key)

**Critério:** RAG retorna ≥ 1 documento para query "AAPL earnings". ✅ Retorna 3 hits (Apr 7-16, 2026).

### 0.4 — Alinhar README ao provider escolhido (D1)

**D1 = Gemini (escolhido):**
- [x] Ajustar `README.md` linha 123 e 361: "Four Gemini agents" + "Gemini 3.1-flash-lite (Google AI)"
- [x] Ajustar Quick Start: trocar `ANTHROPIC_API_KEY` por `GEMINI_KEY` em `.env.example`
- [x] Adicionar nota: "Anthropic provider also supported via `LLM_PROVIDER=anthropic`"

**Critério:** README e código concordam sobre qual LLM é o default. ✅

### 0.5 — Limpar artefatos temporários

**Tarefas:**
- [x] Remover `data/outputs/_temp_ohlcv.parquet`
- [x] Confirmar que `data/outputs/validation_6/` está referenciado em `CLAUDE.md` como regressão revertida (OK) e decidir se vale manter no repo público (recomendação: manter — é evidência honesta do rigor do processo)

**Critério:** diretório `data/outputs/` sem arquivos "_temp_". ✅

---

## Fase 1 — Teste e Validação do Pipeline de Agentes
**Objetivo:** War Room mostrando debate completo de todos os tickers com dados reais ao vivo.
**Por que primeiro:** É o componente mais diferenciado do projeto e atualmente não foi validado ponta-a-ponta. Sem isso, o projeto não tem o seu elemento mais impressionante funcionando.
**Dependência:** Fase 0 completa.

### 1.1 — Dry-run: 3 tickers

**Tarefas:**
- [x] Executar debate isolado para 3 tickers de setores distintos: `AAPL` (tech), `JPM` (finance), `XOM` (energy)
- [x] Validar que os 4 agentes completam (Technical → Fundamental → Bear → PM) sem cair no `_fallback_report`
- [x] Inspecionar `states[ticker]["reports"]` — cada report com `signal`, `confidence`, `reasoning`, `key_factors` preenchidos (3 analyst reports; PM emite `final_decision`)
- [x] Inspecionar `states[ticker]["news_context"]` — 5 hits por ticker (Apr 2026)
- [x] Corrigir erros de Pydantic validation — **zero erros** com Gemini 3.1-flash-lite-preview + `with_structured_output`

**Critério:** ✅ **MET** — 3 tickers × (3 reports + 1 final_decision) = 12 agent outputs, zero fallbacks, RAG cited 15× (5 per ticker). Tempo: 110s para 3 tickers (≈37s/ticker). Detalhes em `data/outputs/_phase1_dryrun.json`.

**Nota de comportamento:** todos os 3 tickers caíram em `HOLD` com confidence=0.25 e weight=0 porque `MIN_CONFIDENCE_FOR_ACTION=0.3` (em `src/agents/state.py:122`) força HOLD quando PM tem baixa confiança. Isso reflete o Bear Agent sendo agressivamente bearish (conf 0.85) vs Technical/Fundamental divididos. Pipeline funcional — o gate conservador é by-design.

### 1.2 — Execução completa: 52 tickers + SPY

**Tarefas:**
- [x] Rodar `make decide` (pipeline `DecisionEngine`: load OHLCV → debate → HRP → save)
- [x] Monitorar uso de prompts via logs (~260 prompts Gemini, 0 erros/fallbacks)
- [x] Registrar tempo total (22min 37s para 52 tickers = ~26s/ticker)
- [x] Validar que `decisions.json` tem 52 entries — **52 ✓**
- [x] Validar que `sum(weight) ≤ 1.0` e nenhum ticker > `max_weight` — sum=0.3150, max=0.0385=cap (2/52) ✓

**Critério:** 52 tickers com decisões reais (não-fallback), pesos somando ≤ 1.0, tempo registrado. ✅ **MET**

**Resultados (Session 41, 2026-04-19T22:32 UTC):**
- Distribuição: **4 BUY** (BAC, MS, PG, PEP — value/defensive), **45 HOLD**, **3 SELL** (META, ABBV, NFLX — conf=0.85 bearish)
- Invested fraction: **31.5%** (bem diversificado, não mostly-cash como previsto)
- Zero fallbacks, zero erros Pydantic, zero retries
- Gemini 3.1-flash-lite-preview consumiu ~260/500 prompts do budget diário

### 1.3 — Validação do War Room no dashboard

**Tarefas:**
- [x] `make run` → abrir `http://localhost:8501` → aba **War Room**
- [x] **Modo Replay:** selecionar AAPL, JPM, NVDA, TSLA individualmente; verificar bolhas sequenciais dos 4 agentes com efeito typing
- [x] **Modo Live:** clicar "Run Debate" em 1 ticker; validar streaming por nó do LangGraph (per-node callback)
- [x] Validar cartão final do Portfolio Manager (cor por ação, peso, confiança, dissent do Bear)
- [x] Validar aba **Performance**: tabela de decisões + donut de pesos refletindo `decisions.json`
- [x] **Bug corrigido:** `app.py:1396` lia `decision.get('weight')` (existe apenas em `decisions.json`, não no dict bruto do PM no modo Live) → fallback para `suggested_weight` adicionado

**Critério:** War Room exibe debate completo sem erros Python no terminal do Streamlit; screenshots para Fase 4 já são reaproveitáveis. ✅ **MET**

**Observações (Session 41, 2026-04-19):**
- Modo Replay funciona direto de `decisions.json` + `debate_history.json`
- Modo Live produz variância estocástica real entre cliques (ex.: PEP → BUY conf=0.55 em Run 4 vs HOLD conf=0.25 em Run 5 devido a `MIN_CONFIDENCE_FOR_ACTION=0.3`)
- Gemini 3.1-flash-lite-preview free tier retorna 503 UNAVAILABLE em picos de demanda (~3 de 5 runs em teste de variância) — fallback handling cobre isso

---

## Fase 2 — Refinamento dos Agentes de Desenvolvimento (`.claude/agents/`)
**Objetivo:** Agentes de desenvolvimento (Claude Code) mais informados sobre este projeto, acelerando as Fases 3-4.
**Por que agora:** Com os agentes do projeto funcionando, a Fase 3-4 usa os agentes de dev intensivamente.
**Dependência:** D3 = publicar (se D3 = não publicar, esta fase vira opcional e curta).

### 2.1 — `quant-reviewer.md`

- [ ] Adicionar regra explícita: separar parâmetros de **timing** (rebalance_every, target_vol — CPCV-OOS-otimizáveis) de **concentração** (max_weight — princípio de risco 2/N) — lição sessão 40
- [ ] Checklist de CPCV: purge days ≥ horizonte, embargo pós-purge, rf convertido geometricamente
- [ ] Flag automática: "max_weight aparece em grid search" → REPROVADO
- [ ] Diferenças < 0.02 Sharpe entre candidatos CPCV-OOS são ruído (DSR-dependente)
- [ ] Qualquer código que misture sinal de agentes + backtest → flag (backtest-produção gap)

### 2.2 — `architect.md`

- [ ] Codificar o modelo three-tier (BUY=HRP, HOLD=HRP×conf, SELL=0, cash implícito)
- [ ] Princípio: "agentes LangGraph não entram no loop de backtest — token cost proíbe" (custo)
- [ ] Constraint: estrutura de pastas sagrada (`src/data`, `src/models`, `src/agents`, etc.)
- [ ] Alerta de imports: `src/agents` não deve importar de `src/portfolio` (e vice-versa)
- [ ] Adicionar heurística: módulo com > 500 LOC pede subdivisão

### 2.3 — `docs-writer.md`

- [ ] Atualizar contexto: Fase 8 completa, 982 configs testadas, Sharpe=0.710 recorde
- [ ] Regra: **nunca** citar Sharpe ~2.7 (bug de look-ahead, corrigido sessão 37-38)
- [ ] Template padronizado para seção "Benchmark" (tabela + gráfico + rationale)
- [ ] Instrução: README público em inglês; `docs/` em PT (conforme D2)
- [ ] Template para seção "Methodology" do README (CPCV-OOS + DSR + HRP explicados em < 300 palavras)

### 2.4 — `test-writer.md`

- [ ] Padrões de mock para LangGraph (como mockar nós do grafo, `with_structured_output`)
- [ ] Padrões para testar agentes: mockar LLM calls com respostas Pydantic válidas
- [ ] Regra reforçada: testes **NUNCA** chamam API real (custo + flaky)
- [ ] Exemplos de fixtures Polars (projeto usa Polars, não Pandas)
- [ ] Teste-padrão: outputs parciais do LLM não quebram Pydantic (graceful fallback)

### 2.5 — `security-data.md`

- [ ] Regra: yfinance deve usar `yf.Ticker().history()` — **nunca** `yf.download()` (bug de thread-safety, sessão 37)
- [ ] Verificação: `financial_news` collection em ChromaDB existe e tem documentos
- [ ] Verificação: `decisions.json` schema (action ∈ {BUY,HOLD,SELL}, `sum(weights) ≤ 1.0`)
- [ ] Alerta: se `financial_news` vazia, `AgentReport.sources_cited` vai aparecer vazio → fundamentalista cai no fallback sem RAG

### 2.6 — Novo agente (opcional): `dashboard-reviewer.md`

- [ ] Avaliar criação de agente especializado em Streamlit: detecta `use_container_width` deprecado, checagem de session_state não reinicializado, threading issues em streaming per-node
- [ ] Se criado: aplicá-lo imediatamente na Fase 3.1

**Critério de conclusão da Fase 2:** Todos os agentes de dev atualizados com contexto pós-sessão 40; novo agente de dashboard decidido e (se sim) implementado.

---

## Fase 3 — Melhorias de Produto
**Objetivo:** Polir os componentes existentes para qualidade de portfólio público.

### 3.1 — Dashboard: polish e UX

- [ ] Corrigir 17 ocorrências de `use_container_width` → `width='stretch'` ou `width='content'` (deprecation Streamlit, deadline 2025-12-31 passou)
- [ ] Adicionar timestamp "Última atualização" em todas as abas (lê `decisions.json.timestamp`)
- [ ] Tratamento explícito para ausência de `decisions.json` (primeira execução): mensagem instrutiva + botão "Run Decision Pipeline"
- [ ] Botão "Exportar PDF" na aba Benchmark abrindo `benchmark_report.pdf`
- [ ] Loading states no War Room Live (progress bar por ticker, ETA estimado)
- [ ] Ajustar responsividade para screenshots (largura fixa 1440px para gravação de GIF)
- [ ] Adicionar pequena seção "About" no sidebar com link GitHub + metodologia

### 3.2 — Ingestão de notícias e RAG (prioritário — habilita Fundamentalista real)

- [ ] Verificar/reconfigurar `NEWSAPI_KEY` e fontes RSS no `.env`
- [ ] Rodar `make ingest` e validar contagem de artigos no ChromaDB (esperado: ≥ 500 para os 52 tickers, últimos 30 dias)
- [ ] Testar retrieval RAG para 5 tickers de setores diferentes; validar latência (< 500ms/query)
- [ ] Se NewsAPI inviável: documentar fallback para fontes RSS públicas gratuitas (Yahoo Finance, Seeking Alpha)
- [ ] Repetir Fase 1.2 (`make decide`) com RAG populado — comparar `sources_cited` antes/depois

### 3.3 — Testes: cobertura dos agentes

- [ ] `poetry run pytest --cov=src --cov-report=term-missing` → identificar módulos < 70%
- [ ] Adicionar testes para `src/agents/graph.py` com mocks de LLM (alvo: cobertura ≥ 70%)
- [ ] Adicionar teste de integração: `run_agent_debate` com mock LLM para 3 tickers → valida `decisions.json`
- [ ] Garantir que os 1002 testes existentes continuam passando

### 3.4 — (Stretch) Análise de feature importance do PatchTST

**Só executar se o cronograma estiver em dia após 3.1–3.3.**

- [ ] Notebook `notebooks/feature_importance.ipynb` (pode ser integrado ao notebook consolidado da Fase 4.2)
- [ ] Análise de impacto de RSI, Bollinger, Vol, VWAP, OBV em `prob_up` — SHAP ou permutation importance
- [ ] **Não retreinar o modelo** — apenas análise do modelo existente
- [ ] Documentar findings (mesmo "feature X é irrelevante" é resultado honesto que impressiona reviewers)

**Critério de conclusão da Fase 3:** Dashboard sem warnings de deprecação, RAG com dados reais, cobertura dos agentes ≥ 70%, 1002+ testes passando.

---

## Fase 4 — Preparação para Portfólio
**Objetivo:** Transformar o projeto técnico em vitrine profissional.

### 4.1 — README: refinamento (não rewrite)

**Contexto:** README atual já tem 391 linhas com Mermaid, tabela de 9 métricas, badges, Quick Start. O trabalho é refinar + adicionar demo visual, não reescrever.

**Tarefas:**
- [ ] Ajustar inconsistência LLM provider (se D1 = Gemini) — feito parcialmente na Fase 0.4
- [ ] **Gravar GIF do War Room** em Replay Mode, 3-5 tickers, ~30-40s, formato GIF otimizado (< 5MB)
  - Ferramenta sugerida: [ScreenToGif](https://www.screentogif.com/) ou [LICEcap](https://www.cockos.com/licecap/)
  - Resolução: 1440×900, 15fps, qualidade balanceada
  - Adicionar logo embed como badge no topo
- [ ] Screenshot da aba Benchmark (equity curve + drawdown + metrics)
- [ ] Screenshot da aba Microstructure (fan chart)
- [ ] Seção "Limitations" explicitamente documentando o backtest-produção gap (ganha credibilidade — é o que separa quant sério de entusiasta)
- [ ] Subseção "Stochastic debate" na Limitations: explicar que `make decide` é estocástico (temperature=0.2 analistas + 0.1 PM) — tickers borderline (ex.: PEP na Fase 1.2) oscilam entre BUY/HOLD em reruns; `decisions.json` é snapshot de uma run, não média. Tickers robustos (ex.: PG) reproduzem consistentemente no Live Debate. Gate de `MIN_CONFIDENCE_FOR_ACTION=0.3` é a fronteira que decide.
- [ ] Atualizar badge de CI com link para última run no GitHub Actions
- [ ] Adicionar citations: Lopez de Prado (HRP + CPCV), Bailey & Lopez de Prado (DSR), Nie et al. (PatchTST)

### 4.2 — Notebook consolidado: metodologia + resultados

Conforme D4 (recomendado: 1 notebook único).

Criar `notebooks/methodology_and_results.ipynb` com estrutura:

- [ ] **Section 1 — Problem:** por que equity selection é difícil (EMH, market microstructure)
- [ ] **Section 2 — Data:** 52 large caps US + SPY, 12 anos, ingestão thread-safe
- [ ] **Section 3 — CPCV-OOS:** diagrama dos 15 paths combinatoriais + purged/embargo
- [ ] **Section 4 — Deflated Sharpe:** curva DSR vs n_trials; por que 547 configs zeraram DSR acceptance; por que holdout temporal é a solução
- [ ] **Section 5 — HRP:** Ledoit-Wolf shrinkage, Ward linkage, confidence tilt sum-preserving
- [ ] **Section 6 — Walk-forward:** equity vs SPY, rolling Sharpe 252d, drawdown, turnover
- [ ] **Section 7 — Regime analysis:** bull (2019-2021), COVID (2020-03), bear (2022), recovery (2023-2024)
- [ ] **Section 8 — Lesson learned (sessão 40):** max_weight como restrição de risco, não parâmetro otimizável — gráfico comparativo validation_3 vs validation_6
- [ ] **Section 9 — Limitations & next steps:** backtest-produção gap + agentes não-backtestados

**Critério:** notebook executa end-to-end em < 5min, gera todas as figuras, sem output hardcoded.

### 4.3 — ARCHITECTURE.md: atualização

**Contexto:** arquivo já existe com 20KB.

- [ ] Atualizar com estado atual pós-Fases 1-8 + correções sessão 37-40
- [ ] Adicionar seção "Design Decisions" com trade-offs explícitos:
  - Ward vs Single linkage no HRP
  - `min(6%, 2/N)` max_weight cap
  - Three-tier weight model (BUY/HOLD/SELL) vs binário
  - Ex-ante volatility targeting vs ex-post
  - Option A (PatchTST fallback) vs Option B (agents no backtest) do design_gap
- [ ] Remover planos futuros não implementados (evita promessas quebradas)
- [ ] Garantir que os diagramas Mermaid renderizam (alguns servidores GitHub quebram Mermaid grande — quebrar em múltiplos diagramas se necessário)

### 4.4 — Publicação dos agentes de desenvolvimento (se D3 = sim)

- [ ] Adicionar seção no README explicando que `.claude/agents/` contém agentes de desenvolvimento usados no workflow (quant-reviewer, architect, docs-writer, test-writer, security-data)
- [ ] Verificar que nenhum agente menciona informação sensível (API keys, paths locais do Windows)
- [ ] Adicionar `.claude/README.md` curto descrevendo cada agente

**Critério de conclusão da Fase 4:** README com GIF do War Room + 2 screenshots, notebook consolidado executável, ARCHITECTURE.md atualizado, projeto visualmente apresentável.

---

## Fase 5 — Publicação
**Objetivo:** Repositório público com lançamento estratégico.

### 5.1 — Auditoria de segurança

- [ ] `git log --all --full-history -- .env` — confirmar que **nunca** foi commitado (histórico atual: limpo)
- [ ] `git log --all --full-history -- '*.env'` — mesma checagem genérica
- [ ] Grep por patterns de API keys: `rg -i "sk-[a-z0-9]{30,}|AIza[a-zA-Z0-9_-]{35}"` em `src/`, `tests/`, `notebooks/`
- [ ] Verificar que `.env.example` não tem valores reais, só placeholders
- [ ] Limpar output de notebooks: `jupyter nbconvert --clear-output notebooks/*.ipynb`
- [ ] Revisar `CLAUDE.md` e `memory/` — remover qualquer referência a infraestrutura privada

### 5.2 — LICENSE + metadados

- [ ] **Criar arquivo `LICENSE`** (MIT) — atualmente só existe o badge no README, não o arquivo
- [ ] Atualizar `pyproject.toml`: `version = "1.0.0"`, `description`, `authors`, `homepage`, `repository`
- [ ] Criar `CHANGELOG.md` com highlights das Fases 1-8 (condensado)
- [ ] Adicionar `CONTRIBUTING.md` básico: como rodar testes, padrões de commit, estrutura sagrada de pastas

### 5.3 — CI/CD: verificação final

- [ ] Rodar GitHub Actions local (act) ou em branch para validar `ci.yml` + `lint.yml`
- [ ] Atualizar badge de CI no README com a URL real pós-publicação
- [ ] Validar `docker compose -f docker/docker-compose.yml up -d` em máquina limpa (idealmente WSL2 / VM)
- [ ] Testar Quick Start completo em ambiente fresh: clone → install → ingest → predict → benchmark → decide → run

### 5.4 — Lançamento

- [ ] Tornar repositório público no GitHub
- [ ] Criar GitHub Release `v1.0.0` com changelog + GIF embed + link direto para notebooks
- [ ] GitHub topics: `quantitative-finance`, `algorithmic-trading`, `langgraph`, `portfolio-optimization`, `machine-learning`, `python`, `hrp`, `patchtst`, `cpcv`
- [ ] Post no LinkedIn: ângulo "multi-agent debate + quant rigoroso", incluir GIF + gráfico performance + link repo
- [ ] (Opcional) Post no r/algotrading, QuantStart, QuantConnect community — tom técnico, não promocional

---

## O que NÃO fazer (riscos de escopo)

| Tentação | Por que não fazer |
|---|---|
| Mais fine-tuning CPCV-OOS | Espaço de parâmetros esgotado (982 configs); risco de otimização enganosa (sessão 40) |
| Integrar agentes no walk-forward | Custo de tokens inviável (208 × 126 = ~26k prompts por rebalance); gap backtest-produção é **feature**, não bug |
| Adicionar mais tickers | 52 é suficiente e bem documentado; mais = mais ruído, mesmo Sharpe |
| Retreinar PatchTST sem CPCV-OOS | Look-ahead bias potencial; só com validação completa |
| Refatorar arquitetura existente | O que funciona, funciona; risco de quebrar 1002 testes |
| Traduzir toda a documentação interna | Overhead alto, ganho marginal — README em inglês já cobre audiência externa (D2) |
| Adicionar features novas ao PatchTST | Escopo criativo infinito; o foco é publicar o que está validado |

---

## Checklist de Publicação

```
[ ] D1-D4 decidas e marcadas
[ ] Fase 0 — Gemini model name validado e smoke test OK
[ ] Fase 0 — RAG populado com news reais
[ ] Fase 0 — README e código concordam sobre LLM provider
[ ] Fase 1 — Pipeline de agentes validado com 52 tickers (não-fallback)
[ ] Fase 1 — War Room ao vivo funcional no dashboard
[ ] Fase 2 — 5 agentes de dev refinados (+ dashboard-reviewer se criado)
[ ] Fase 3 — Dashboard sem warnings `use_container_width`
[ ] Fase 3 — Cobertura de testes dos agentes ≥ 70%
[ ] Fase 3 — RAG retornando sources reais citados pelo Fundamentalista
[ ] Fase 4 — GIF do War Room < 5MB, ~30s
[ ] Fase 4 — Notebook consolidado executa end-to-end
[ ] Fase 4 — ARCHITECTURE.md com seção Design Decisions
[ ] Fase 5 — LICENSE file existe (MIT)
[ ] Fase 5 — `git log --all -- .env` confirma zero commits
[ ] Fase 5 — GitHub Actions passando
[ ] Fase 5 — Repositório público + release v1.0.0
```

---

## Métricas de Sucesso do Portfólio

O projeto será considerado de alto impacto se:

- **Metodologia rigorosa:** CPCV-OOS + Deflated Sharpe + HRP explicados corretamente no README e notebook, com citações bibliográficas
- **Honestidade explícita:** backtest-produção gap documentado em destaque — isso é o que separa quant sério de entusiasta
- **Diferenciação real:** combinação PatchTST + LangGraph debate + HRP validada é genuinamente novel — nenhum projeto público equivalente
- **"Wow factor" visível:** War Room ao vivo (GIF + demo) é o elemento que prende o recrutador quant por > 30s
- **Números defensáveis:** Sharpe=0.710 com metodologia anti-overfitting documentada, sem claims de "outperformance" irreais
- **Workflow agêntico:** presença de `.claude/agents/` (se D3 = sim) demonstra adoção prática de ferramentas modernas de desenvolvimento

---

## Apêndice — Comandos úteis durante a execução

```bash
# Fase 0 — smoke test Gemini
poetry run python -c "from src.agents.graph import _create_llm; print(_create_llm(0.2).invoke('Reply exactly OK').content)"

# Fase 0 — smoke test RAG
poetry run python -c "from src.agents.rag import FinancialRAG; r=FinancialRAG(); print('docs:', r.collection.count())"

# Fase 1 — dry-run 3 tickers
poetry run python -c "from src.agents.graph import run_agent_debate; d,s=run_agent_debate(['AAPL','JPM','XOM']); print(len(d), 'decisions')"

# Fase 1 — pipeline completo
make decide

# Fase 3 — cobertura
poetry run pytest --cov=src --cov-report=term-missing --cov-report=html

# Fase 5 — auditoria secrets
rg -i "sk-[a-z0-9]{30,}|AIza[a-zA-Z0-9_-]{35}" src/ tests/ notebooks/
git log --all --full-history -- .env '*.env'
```
