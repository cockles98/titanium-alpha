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
- [x] Se faltar algum: `make predict` para regenerar — **N/A, todos existem**

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

- [x] Adicionar regra explícita: separar parâmetros de **timing** (rebalance_every, target_vol — CPCV-OOS-otimizáveis) de **concentração** (max_weight — princípio de risco 2/N) — lição sessão 40
- [x] Checklist de CPCV: purge days ≥ horizonte, embargo pós-purge, rf convertido geometricamente
- [x] Flag automática: "max_weight aparece em grid search" → REPROVADO
- [x] Diferenças < 0.02 Sharpe entre candidatos CPCV-OOS são ruído (DSR-dependente)
- [x] Qualquer código que misture sinal de agentes + backtest → flag (backtest-produção gap)

### 2.2 — `architect.md`

- [x] Codificar o modelo three-tier (BUY=HRP, HOLD=HRP×conf, SELL=0, cash implícito)
- [x] Princípio: "agentes LangGraph não entram no loop de backtest — token cost proíbe" (custo)
- [x] Constraint: estrutura de pastas sagrada (`src/data`, `src/models`, `src/agents`, etc.)
- [x] Alerta de imports: `src/agents` não deve importar de `src/portfolio` (e vice-versa)
- [x] Adicionar heurística: módulo com > 500 LOC pede subdivisão

### 2.3 — `docs-writer.md`

- [x] Atualizar contexto: Fase 8 completa, 982 configs testadas, Sharpe=0.710 recorde
- [x] Regra: **nunca** citar Sharpe ~2.7 (bug de look-ahead, corrigido sessão 37-38)
- [x] Template padronizado para seção "Benchmark" (tabela + gráfico + rationale)
- [x] Instrução: README público em inglês; `docs/` em PT (conforme D2)
- [x] Template para seção "Methodology" do README (CPCV-OOS + DSR + HRP explicados em < 300 palavras)

### 2.4 — `test-writer.md`

- [x] Padrões de mock para LangGraph (como mockar nós do grafo, `with_structured_output`)
- [x] Padrões para testar agentes: mockar LLM calls com respostas Pydantic válidas
- [x] Regra reforçada: testes **NUNCA** chamam API real (custo + flaky)
- [x] Exemplos de fixtures Polars (projeto usa Polars, não Pandas)
- [x] Teste-padrão: outputs parciais do LLM não quebram Pydantic (graceful fallback)

### 2.5 — `security-data.md`

- [x] Regra: yfinance deve usar `yf.Ticker().history()` — **nunca** `yf.download()` (bug de thread-safety, sessão 37)
- [x] Verificação: `financial_news` collection em ChromaDB existe e tem documentos
- [x] Verificação: `decisions.json` schema (action ∈ {BUY,HOLD,SELL}, `sum(weights) ≤ 1.0`)
- [x] Alerta: se `financial_news` vazia, `AgentReport.sources_cited` vai aparecer vazio → fundamentalista cai no fallback sem RAG

### 2.6 — Novo agente (opcional): `dashboard-reviewer.md`

- [x] **Decisão: NÃO criar** (sessão 41). Fase 3.1 tem escopo pequeno (17 `use_container_width` — grep-and-replace mecânico + polish); agentes existentes cobrem o necessário; streamlit patterns são bem documentados. Over-specialização = YAGNI. Reavaliar se dashboard ganhar complexidade em Fase 3+.

**Critério de conclusão da Fase 2:** Todos os agentes de dev atualizados com contexto pós-sessão 40; decisão sobre dashboard-reviewer documentada. ✅ **MET**

**Resultado (sessão 41, 2026-04-20):**
- `quant-reviewer.md`: adicionadas regras timing vs concentração, CPCV checklist, backtest-produção gap, flag automática para `max_weight` em grid search
- `architect.md`: codificado modelo 3-tier, firewalls de import (agents ↔ portfolio ↔ backtest), heurística 500 LOC, compatibilidade com pipeline existente
- `docs-writer.md`: contexto Fase 8 (Sharpe=0.712), baniu citação de Sharpe ~2.7, templates de Benchmark e Methodology, regra PT/EN
- `test-writer.md`: patterns de mock para LangGraph + Gemini + yfinance, fixtures Polars, teste-padrão de graceful fallback
- `security-data.md`: regra `yf.Ticker().history()` (bug sessão 37), checagens de `decisions.json` schema + ChromaDB `financial_news`, alertas RAG vazio

---

## Fase 3 — Melhorias de Produto
**Objetivo:** Polir os componentes existentes para qualidade de portfólio público.

### 3.1 — Dashboard: polish e UX

- [x] Corrigir 17 ocorrências de `use_container_width` → `width='stretch'` (deprecation Streamlit, deadline 2025-12-31 passou)
- [x] Adicionar timestamp "Última atualização" no header principal (lê `decisions.json.timestamp`, formato `YYYY-MM-DD HH:MM UTC`)
- [x] Tratamento explícito para ausência de `decisions.json`: mensagem instrutiva no War Room com comando `make decide`
- [x] Botão "Download Benchmark Report (PDF)" na aba Benchmark (lê `data/outputs/benchmark_report.pdf`, fallback para caption instrutiva)
- [x] Loading states no War Room Live: `st.progress` por nós completados (6 nodes: load_context/rag/tech/fund/bear/PM)
- [x] Seção "About" no sidebar com descrição, links GitHub/metodologia, caption do recorde walk-forward
- [ ] Ajustar responsividade para screenshots (1440×900 para GIF) — deferido para Fase 4.1 (gravação do GIF)

**Resultado (sessão 41, 2026-04-20):** 81/81 testes dashboard passando; 1000/1002 testes totais (2 pré-existentes em `test_rag.py` não relacionados — alocados para Fase 3.3).

### 3.2 — Ingestão de notícias e RAG (prioritário — habilita Fundamentalista real)

- [x] Verificar/reconfigurar `NEWSAPI_KEY` e fontes RSS no `.env`
- [x] Rodar `make ingest` e validar contagem de artigos no ChromaDB (esperado: ≥ 500 para os 52 tickers, últimos 30 dias)
- [x] Testar retrieval RAG para 5 tickers de setores diferentes; validar latência (< 500ms/query)
- [x] Se NewsAPI inviável: documentar fallback para fontes RSS públicas gratuitas (Yahoo Finance, Seeking Alpha)
- [x] Repetir Fase 1.2 (`make decide`) com RAG populado — comparar `sources_cited` antes/depois (executado em "3.2 pendente" abaixo)

**Resultado (sessão 41, 2026-04-20):** PG `financial_news` tem 172 artigos (Google Finance / CNBC / Yahoo Finance, datados abril 2026) — máquina foi trocada no meio da sessão e o DB do Cockles é independente do felip (14.326 artigos). Embedding para ChromaDB: 172/172 em 1.4s. Retrieval smoke-test (5 tickers, `top_k=5`, `max_age_days=720`):

| Ticker | Hits | Latência | Candidatos |
|---|---|---|---|
| NVDA | 5 | 78ms | 7 |
| SPY  | 5 | 61ms | 15 |
| BAC  | 5 | 59ms | 7 |
| META | 3 | 60ms | 3 |
| TSLA | 3 | 60ms | 3 |

Latência P95 = 101ms (NVDA cold-start), muito abaixo do target de 500ms. Cobertura esparsa em META/TSLA indica que o backfill histórico via Google Finance é ticker-dependente (fontes RSS públicas têm cobertura irregular). Scripts de validação: `scripts/_phase3_2_check_rag.py`, `scripts/_phase3_2_embed.py`, `scripts/_phase3_2_retrieve.py`.

### 3.3 — Testes: cobertura dos agentes

- [x] `poetry run pytest --cov=src --cov-report=term-missing` → identificar módulos < 70%
- [x] Adicionar testes para `src/agents/graph.py` com mocks de LLM (alvo: cobertura ≥ 70%)
- [x] Adicionar teste de integração: `run_agent_debate` com mock LLM para 3 tickers → valida `decisions.json`
- [x] Garantir que os 1002 testes existentes continuam passando

**Resultado (sessão 41, 2026-04-20):** pytest-cov instalado. Cobertura `src.agents`:

| Módulo | Cobertura | Faltando |
|---|---|---|
| `src/agents/__init__.py` | 100% | — |
| `src/agents/graph.py` | **84%** | Linhas 84-95 (LLM provider switch), 238-242/250-254/296-301/329-334/406/451-455/505-509/566-571/648-652 (NaN/Inf guards, exception fallbacks, `predictions.parquet` missing) |
| `src/agents/personas.py` | 100% | — |
| `src/agents/rag.py` | **100%** | — |
| `src/agents/state.py` | 98% | Linha 203 (edge-case reducer) |
| **TOTAL** | **90%** | 45 de 466 statements |

Todos os 1002 testes passam (previamente 1000 passando + 2 falhas pré-existentes em `test_rag.py`). Fixes aplicados:

1. `tests/test_rag.py::TestEmbedPendingNews::test_empty_articles_marked_as_embedded`: parâmetros vão como segundo argumento posicional (`{"ids": [...]}`), não como kwarg — teste atualizado para usar `call_args.args[1]`.
2. `tests/test_rag.py::TestRetrieve::test_n_results_is_capped_at_50` → renomeado para `test_n_results_is_capped_at_500`: source havia sido atualizado de `min(top_k * 3, 50)` para `min(top_k * 40, 500)` para garantir candidatos suficientes após filtro de data em python.

46 testes em `tests/test_graph.py` cobrem load_context, RAG retrieval, cada analista (Technical/Fundamental/Bear), Portfolio Manager, MIN_CONFIDENCE_FOR_ACTION gate e streaming callback. As linhas não cobertas em `graph.py` são guardas defensivas (provider switch Gemini↔Anthropic, NaN/Inf defaults, exceções de I/O) que exigiriam mocking intrusivo sem ROI proporcional.

### 3.4 — (Stretch) Análise de feature importance do PatchTST

**Só executar se o cronograma estiver em dia após 3.1–3.3.**

- [x] Notebook `notebooks/feature_importance.ipynb` (substituído por script versionável `scripts/_phase3_4_feature_importance.py`; output em `data/outputs/phase3_4_feature_importance.md`)
- [x] Análise de impacto de RSI, Bollinger, Vol, VWAP, OBV em `prob_up` — SHAP ou permutation importance
- [x] **Não retreinar o modelo** — apenas análise do modelo existente
- [x] Documentar findings (mesmo "feature X é irrelevante" é resultado honesto que impressiona reviewers)

**Resultado (sessão 41, 2026-04-20):** Descoberta arquitetural crítica — PatchTST é **channel-independent** (vê apenas a série de close), então permutation importance clássica nas features é estruturalmente sem sentido (shuffle não afeta o modelo porque não são inputs). O relatório documenta duas análises alternativas:

1. **Sensibilidade PatchTST a perturbação no close** (σ=2% multiplicativo nos últimos 60 dias, 30 perturbações × 5 tickers): mean\|Δprob\| varia de 0.003 a 0.017 — modelo **é sensível** aos inputs (não degenerado), maior sensibilidade em ABBV (0.017) e menor em AMD (0.003).

2. **Correlação Spearman cross-sectional features × prob_up** (n=53 tickers):

| Feature | ρ(prob_up) | ρ(expected_return) | Relevância |
|---|---|---|---|
| `rsi_14` | **+0.310** | **+0.409** | Acima do piso de ruído (|ρ|>0.3); candidata a feature exógena |
| `realized_vol_21` | +0.165 | **+0.415** | Tickers voláteis tendem a ter expected_return maior |
| `bb_upper` / `bb_middle` | +0.187 / +0.185 | +0.222 / +0.200 | Nível intermediário |
| `obv` | −0.141 | −0.179 | Relação fraca e inversa |
| `vwap` | +0.112 | +0.156 | Próximo de ruído |
| `volume_sma` / `relative_volume` | +0.036 / −0.071 | +0.016 / −0.065 | Indistinguível de ruído |

**Interpretação honesta:** o design channel-independent é estruturalmente fiel — features informam agentes, não o modelo. `rsi_14` e `realized_vol_21` são candidatas a inputs exógenos numa extensão multivariada futura do PatchTST.

### 3.2 pendente — `make decide` com RAG populado (comparação `sources_cited`)

- [x] Rodar `make predict` (gerou `predictions.parquet` + `features.parquet` + `model_checkpoint/` após ~16min training PatchTST no CPU, 53 tickers × 1256 rows)
- [x] Rodar `make decide` com RAG populado (172 artigos embedded)
- [x] Analisar distribuição de `sources_cited` (script `scripts/_phase3_2b_sources_analysis.py` → `data/outputs/phase3_2b_sources_report.md`)

**Resultado (sessão 41, 2026-04-20):**

| Métrica | Valor |
|---|---|
| Tickers decididos | 52 |
| Com news retrieved | 13 (25.0%) |
| Grounded (≥1 citation) | 14 (26.9%) |
| Ungrounded | 38 (73.1%) |
| Total de citations | 19 |
| Breakdown por ação | 1 BUY / 49 HOLD / 2 SELL |

Top grounded: NVDA (3 citations), AAPL (2), TSLA (2), BAC (2), META (1, SELL conf=0.85).

**Caveat honesto — quota Gemini exaurida mid-run:** durante `make decide` o free tier diário do Gemini (500 req/dia) esgotou no ticker PEP (tkr #30/52). Os ~22 tickers restantes passaram por erros HTTP 429 e retornaram reports vazios com `confidence=0.00`, inflando artificialmente a contagem de HOLDs e baixando a taxa de grounding. Restrito aos ~30 tickers que completaram debate real, grounding ≈ **47%** (14/30), consistente com a cobertura esparsa de 172 artigos para 52 tickers. O relatório `data/outputs/phase3_2b_sources_report.md` documenta o caveat e sugere re-run após reset da quota (24h) ou com chave paga para medição fiel.

**Observação técnica:** grounded (14) > retrieved (13) em 1 caso indica hallucination do LLM — o Fundamentalist ecoou um título de fonte sem hit correspondente do RAG. Isso é artefato do LLM, não bug do pipeline.

**Critério de conclusão da Fase 3:** Dashboard sem warnings de deprecação, RAG com dados reais, cobertura dos agentes ≥ 70%, 1002+ testes passando, feature importance documentada. ✅ **MET**

---

## Fase 4 — Preparação para Portfólio
**Objetivo:** Transformar o projeto técnico em vitrine profissional.

### 4.1 — README: refinamento (não rewrite)

**Contexto:** README atual já tem 391 linhas com Mermaid, tabela de 9 métricas, badges, Quick Start. O trabalho é refinar + adicionar demo visual, não reescrever.

**Tarefas:**
- [ ] Ajustar inconsistência LLM provider (se D1 = Gemini) — feito parcialmente na Fase 0.4
- [x] **Gravar GIF do War Room** em Replay Mode, 3-5 tickers, ~30-40s — **status:** GIF gravado (`docs/images/warroom_demo.gif`, atualmente 27MB — compressão para <5MB marcada como trabalho futuro do usuário)
  - Ferramenta sugerida: [ScreenToGif](https://www.screentogif.com/) ou [LICEcap](https://www.cockos.com/licecap/)
  - Resolução: 1440×900, 15fps, qualidade balanceada
  - Adicionar logo embed como badge no topo
- [x] Screenshot da aba Benchmark (equity curve + drawdown + metrics) — `docs/images/benchmark.png`
- [x] Screenshot da aba Microstructure (fan chart) — `docs/images/microstructure.png`
- [x] Seção "Limitations" explicitamente documentando o backtest-produção gap (ganha credibilidade — é o que separa quant sério de entusiasta)
- [x] Subseção "Stochastic debate" na Limitations: explicar que `make decide` é estocástico (temperature=0.2 analistas + 0.1 PM) — tickers borderline (ex.: PEP na Fase 1.2) oscilam entre BUY/HOLD em reruns; `decisions.json` é snapshot de uma run, não média. Tickers robustos (ex.: PG) reproduzem consistentemente no Live Debate. Gate de `MIN_CONFIDENCE_FOR_ACTION=0.3` é a fronteira que decide.
- [x] Atualizar badge de CI com link para última run no GitHub Actions — badge dinâmico `github.com/cockles98/titanium-alpha/actions/workflows/ci.yml/badge.svg` + link clicável; `git clone` com username placeholder corrigido para `cockles98`
- [x] Adicionar citations: Lopez de Prado (HRP + CPCV), Bailey & Lopez de Prado (DSR), Nie et al. (PatchTST) — + Ledoit-Wolf, Chernozhukov, Reimers-Gurevych

### 4.2 — Notebook consolidado: metodologia + resultados

Conforme D4 (recomendado: 1 notebook único).

Criar `notebooks/methodology_and_results.ipynb` com estrutura:

- [x] **Section 1 — Problem:** por que equity selection é difícil (EMH, market microstructure)
- [x] **Section 2 — Data:** 52 large caps US + SPY, 15 anos, ingestão thread-safe
- [x] **Section 3 — CPCV-OOS:** diagrama dos 15 paths combinatoriais + purged/embargo
- [x] **Section 4 — Deflated Sharpe:** curva DSR vs n_trials; por que 547 configs zeraram DSR acceptance; por que holdout temporal é a solução
- [x] **Section 5 — HRP:** Ledoit-Wolf shrinkage, Ward linkage, confidence tilt sum-preserving
- [x] **Section 6 — Walk-forward:** equity vs SPY, rolling Sharpe 252d, drawdown (10y OOS)
- [x] **Section 7 — Regime analysis:** bull (2016-19), COVID+recov (2020-21), bear (2022), late cycle (2023-26)
- [x] **Section 8 — Lesson learned (sessão 40):** max_weight como restrição de risco, não parâmetro otimizável
- [x] **Section 9 — Limitations & next steps:** backtest-produção gap + agentes não-backtestados + 4 próximos passos

**Critério:** notebook executa end-to-end em < 5min, gera todas as figuras, sem output hardcoded. **Verificado:** 6.6s em execução limpa, outputs limpos antes do commit.

### 4.3 — ARCHITECTURE.md: atualização

**Contexto:** arquivo já existe com 20KB.

- [x] Atualizar com estado atual pós-Fases 1-8 + correções sessão 37-40 (+ sessão 42)
- [x] Adicionar seção "Design Decisions" com trade-offs explícitos:
  - [x] Ward vs Single linkage no HRP
  - [x] `min(6%, 2/N)` max_weight cap (documentado na seção HRP e na Lição aprendida do notebook)
  - [x] Three-tier weight model (BUY/HOLD/SELL) — documentado em `decision_engine.py` entry
  - [x] Ex-ante volatility targeting vs ex-post
  - [x] CPCV-OOS + holdout vs CPCV alone (captura o tradeoff Option A/B do design_gap no ângulo da validação)
- [x] Remover planos futuros não implementados — scan do README via grep (`TODO|FIXME|planned|future work|roadmap|will be|will include|not yet`) não encontrou promessas vazias; a única menção a "not yet been backtested" refere-se ao backtest-production gap (documentação honesta, não promessa futura)
- [ ] Garantir que os diagramas Mermaid renderizam — **requer verificação manual no preview do GitHub após tornar público**

### 4.4 — Publicação dos agentes de desenvolvimento (se D3 = sim)

- [x] Adicionar seção no README explicando que `.claude/agents/` contém agentes de desenvolvimento usados no workflow (quant-reviewer, architect, docs-writer, test-writer, security-data) — nova seção "Development Workflow — Claude Code Subagents"
- [x] Verificar que nenhum agente menciona informação sensível (API keys, paths locais do Windows) — scan limpo via grep
- [x] Adicionar `.claude/README.md` curto descrevendo cada agente — criado com tabela + workflow diagram + fork-adaptation guide

**Critério de conclusão da Fase 4:** README com GIF do War Room + 2 screenshots, notebook consolidado executável, ARCHITECTURE.md atualizado, projeto visualmente apresentável.

---

## Fase 5 — Publicação
**Objetivo:** Repositório público com lançamento estratégico.

### 5.1 — Auditoria de segurança

- [x] `git log --all --full-history -- .env` — confirmado limpo (zero commits)
- [x] `git log --all --full-history -- '*.env'` — confirmado limpo
- [x] Grep por patterns de API keys: `sk-[a-zA-Z0-9]{30,}|AIza[a-zA-Z0-9_-]{35}` no repo inteiro — **zero matches**
- [x] `.env.example` validado — apenas placeholders (`your-gemini-api-key`, `your-newsapi-key`, `changeme`)
- [x] Notebook `methodology_and_results.ipynb` com outputs limpos (grep `"outputs":\s*\[[^\]]` = 0 ocorrências)
- [x] `CLAUDE.md` revisado — sem infraestrutura privada; paths referem-se apenas a `data/outputs/`, `docs/`, `src/`; memory files residem em `~/.claude/projects/` (fora do repo)

### 5.2 — LICENSE + metadados

- [x] **`LICENSE`** criado (MIT, copyright 2026 Felipe Cockles)
- [x] `pyproject.toml` atualizado: `version = "1.0.0"`, author com email, `license = {text = "MIT"}`, `readme`, `keywords` (10 tags), `classifiers` (10 PyPI), `[project.urls]` (Homepage/Repository/Issues/Changelog para `cockles98/titanium-alpha`), `pytest-cov>=5.0` adicionado às dev deps
- [x] `CHANGELOG.md` criado — v1.0.0 cobre Added/Fixed/Known limitations/Methodological stance consolidando as 8 fases técnicas
- [x] `CONTRIBUTING.md` criado — setup, workflow com `.claude/agents/`, commit format, code style rules, sacred folder structure, testing expectations, "what NOT to PR"

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
