# Titanium Alpha — Contexto do Projeto

## O que é este projeto
Sistema de fundo de hedge agêntico multi-estratégia que combina:
- PatchTST (Deep Learning para séries temporais financeiras)
- Multi-agente LangGraph (debate entre Analista Técnico, Fundamentalista, Bear, Portfolio Manager)
- RAG financeiro com ChromaDB
- Backtesting CPCV (Combinatorial Purged Cross-Validation)
- HRP (Hierarchical Risk Parity) para alocação

## Stack tecnológico
Python 3.10+, Polars, NeuralForecast (Nixtla), LangGraph, ChromaDB,
PostgreSQL, Streamlit, Plotly, Docker, Poetry

## Padrões obrigatórios de código
- Type hints em TODOS os métodos
- Docstrings Google Style em toda função/classe pública
- Módulos independentes — sem imports circulares
- Logging com loguru (nunca print() em produção)
- Erros sempre re-raised com contexto (nunca bare except)

## Estrutura de pastas (sagrada — não mudar sem perguntar)
src/data/          → ingestão e pipelines de dados
src/models/        → PatchTST, features, previsões
src/agents/        → LangGraph agents, RAG, personas
src/backtest/      → CPCV, simulação de custos
src/portfolio/     → HRP, decision engine
src/dashboard/     → Streamlit UI
src/utils/         → helpers compartilhados
tests/             → pytest, fixtures em conftest.py
notebooks/         → exploração (nunca importado por src/)
docs/              → documentação e referências
docker/            → Dockerfiles e compose

## Convenções de git
feat: nova funcionalidade
fix: correção de bug  
refactor: sem mudança de comportamento
test: adição/modificação de testes
docs: apenas documentação

## Contexto financeiro importante
- Ativos principais: SPY, NVDA, AAPL, QQQ
- Período de dados: últimos 5 anos
- Frequência: dados diários (OHLCV)
- Benchmark: SPY
- Métricas-chave: Sharpe Ratio anualizado (rf=0.05), Max Drawdown, CAGR

## Sessões implementadas

### Fase 1 — Infraestrutura e Dados (Semanas 1-2)

**Sessão 1 — Estrutura inicial** (concluída)
- Projeto inicializado com Poetry (pyproject.toml com todas as dependências)
- Estrutura de pastas criada: src/{data,models,agents,backtest,portfolio,dashboard,utils}, tests/, notebooks/, docs/, docker/
- CLAUDE.md com regras do projeto
- .env com variáveis de ambiente (sem secrets hardcoded)
- .gitignore configurado
- Makefile com targets: setup, ingest, test, lint, run, clean

**Sessão 2 — Docker e Bancos** (concluída)
- docker/docker-compose.yml com PostgreSQL 15 e ChromaDB (healthchecks, volumes persistentes)
- src/utils/db.py: get_postgres_engine() com connection pooling (SQLAlchemy) e get_chroma_client()
- tests/test_db.py: 15 testes (happy path, missing vars, overrides)

**Sessão 3 — Ingestão de Dados OHLCV** (concluída)
- src/data/ingestion.py: classe MarketDataIngester (yfinance → Polars → PostgreSQL)
  - Download com retry e backoff exponencial
  - Schema validation, upsert com ON CONFLICT
  - Suporte a SPY, NVDA, AAPL, QQQ (últimos 5 anos)
- tests/test_ingestion.py: 16 testes (download, schema, retry, save, NaN handling)
- tests/conftest.py: fixtures compartilhadas (mock_engine, sample_yf_dataframe)
- Validado pelo subagente security-data

**Sessão 4 — Ingestão de Notícias** (concluída)
- src/data/news_ingestion.py: classe NewsIngester (NewsAPI + RSS → Polars → PostgreSQL)
  - Fontes RSS: Yahoo Finance, Google Finance, CNBC
  - Ticker matching por keywords (SPY, NVDA, AAPL, QQQ)
  - _clean_html (BeautifulSoup), _truncate, _parse_date (multi-formato)
  - Schema financial_news com embedding_status para futuro RAG
  - Deduplicação por URL (ON CONFLICT), retry com backoff
- tests/test_news_ingestion.py: 37 testes (helpers, fetch, persistence, orchestration)
- Validado pelo subagente security-data

**Status dos testes:** 68 testes passando (2.47s)

**Checkpoint Semana 1** (validado em 2026-03-04)
- `make ingest` (OHLCV): 4 tickers, 5016 rows no PostgreSQL
- `make ingest` (News): 3 fontes RSS, 177 artigos fetched, 153 inseridos
- Deduplicação ON CONFLICT funcionando (168 duplicatas ignoradas na re-execução)
- Logs sem ERROR; único WARNING esperado: NEWSAPI_KEY não configurada
- Fix aplicado: substituído print() por logger nos blocos __main__ (encoding cp1252 no Windows)

**Sessão 5 — Feature Engineering** (concluída em 2026-03-04)
- src/models/features.py: 5 funções públicas + 2 helpers privados
  - `rsi()` — RSI (SMA-based), retorna Series nomeada
  - `bollinger_bands()` — SMA ± n*std, retorna DataFrame 3 colunas
  - `realized_volatility()` — log returns → rolling_std → √252 anualização
  - `volume_profile()` — volume_sma, relative_volume, vwap (cumulative), obv
  - `compute_all_features()` — orquestrador, 8 cols originais + 9 features = 17 cols
- tests/test_features.py: 30 testes (RSI, BB, RVol, VolumeProfile, validation, no-look-ahead)
- conftest.py: fixture `sample_ohlcv_df` (100 rows, random walk)
- Quant-reviewer: APROVADO (zero look-ahead bias, backward-only rolling windows)
  - Ressalvas tratadas: RSI docstring corrigida (SMA, não Wilder), RSI constante retorna null
  - Ressalvas aceitas (baixa severidade): VWAP cum_sum global, ddof=1 em BB std

**Sessão 6 — PatchTST com NeuralForecast** (concluída em 2026-03-04)
- src/models/patchtst_model.py: classe TitaniumForecaster
  - PatchTST (NeuralForecast v3.1.5) com MQLoss (quantis 0.1, 0.25, 0.5, 0.75, 0.9)
  - `fit()` — treina com split temporal (val_size), validação de min rows
  - `predict()` — previsão quantílica h=5 dias à frente
  - `predict_proba()` — P(up) por ticker (fração de quantis > close atual)
  - `save()`/`load()` — persistência com metadata JSON
  - Params: input_size=60, h=5, batch_size=32, freq="1bd"
- tests/test_patchtst_model.py: 31 testes (init, prepare, build, fit, predict, proba, save/load)
- conftest.py: fixture `sample_features_df` (compute_all_features → drop_nulls)
- models/checkpoints/.gitkeep criado
- **Decisão arquitetural:** PatchTST não suporta hist_exog_list (limitação NeuralForecast)
  - Features técnicas (RSI, BB, etc.) serão consumidas pelos agentes LangGraph
  - PatchTST opera apenas no close price (channel-independent, conforme paper original)
- Quant-reviewer: APROVADO (zero data leakage)
  - Ressalvas tratadas: sort defensivo em predict_proba, validação min rows no fit
  - Ressalvas aceitas (baixa): expected_return simples (ok para h=5)

**Sessão 7 — Pipeline de Previsões** (concluída em 2026-03-05)
- src/models/predict.py: classe PredictionPipeline
  - `load_ohlcv()` — carrega de PostgreSQL via Polars (read_database)
  - `compute_metrics()` — MAE/RMSE por ticker (forecast mediano vs actual)
  - `run()` — pipeline completo: PostgreSQL → features → PatchTST → Parquet
  - Outputs: predictions.parquet, forecast.parquet, metrics.parquet em data/outputs/
- tests/test_predict.py: 12 testes (init, load, metrics, run, parquet roundtrip)
- Makefile: target `make predict` adicionado
- data/outputs/.gitkeep criado

**Status dos testes:** 141 testes passando (10.95s)

### Fase 2 — Camada Agêntica (Semanas 3-4)

**Sessão 8 — Design dos Agentes** (concluída em 2026-03-05)
- Só planejamento — nenhum código escrito
- Topologia: pipeline linear START → load_context → Technical → Fundamental → Bear → PM → END
- LLM: claude-sonnet-4-6 (temp=0.2 analistas, temp=0.1 PM)
- Estado por ticker (loop externo), DAG puro sem ciclos
- Design completo em memory/agent_design.md

**Sessão 9 — State e Personas** (concluída em 2026-03-05)
- src/agents/state.py: 4 TypedDicts + 2 validators + factory
  - TickerPrediction, AgentReport, FinalDecision, InvestmentState
  - `make_empty_state(ticker)` — factory para estado inicial
  - `validate_report()` / `validate_decision()` — validação com regras de negócio
  - Constraints: confidence ∈ [0,1], weight ∈ [0,0.25], confidence < 0.3 → HOLD
- src/agents/personas.py: 4 system prompts + 2 Pydantic models
  - TECHNICAL_ANALYST: quantitativo, cita RSI/BB/volume exatos
  - FUNDAMENTALIST_ANALYST: complementa (não repete), foco narrativa/macro/news
  - BEAR_AGENT: devil's advocate, sempre cético, só critica, quantifica riscos
  - PORTFOLIO_MANAGER: sintetiza tudo, cap 25%/posição, HOLD se confiança < 0.3
  - AgentReportModel / FinalDecisionModel — Pydantic para structured output via LLM
  - PERSONA_REGISTRY — dict para acesso programático por chave
- tests/test_state.py: 26 testes (TypedDicts, factory, validators, edge cases)
- tests/test_personas.py: 32 testes (prompts, registry, Pydantic validation)

**Status dos testes:** 199 testes passando (11.63s)

**Sessão 10 — Grafo LangGraph** (concluída em 2026-03-05)
- src/agents/graph.py: pipeline LangGraph completo
  - `load_context()` — carrega predictions.parquet + forecast.parquet para o ticker
  - `technical_analyst()` → `fundamentalist_analyst()` → `bear_agent()` → `portfolio_manager()`
  - Cada nó: ChatAnthropic(claude-sonnet-4-6).with_structured_output() → Pydantic model
  - `build_investment_graph()` — monta StateGraph com 5 nós + edges lineares
  - `run_agent_debate(tickers)` — executa grafo 1x por ticker, retorna list[FinalDecision]
  - Prompt builders: _format_predictions, _format_features, _format_reports, _format_news
  - Logging estruturado via debate_log para dashboard Streamlit
- state.py atualizado: `Annotated[list, operator.add]` em reports e debate_log (reducer LangGraph)
- tests/test_graph.py: 29 testes (formatters, load_context, 4 nós, graph completo, run_agent_debate)
- Architect review: APROVADO
  - Fix aplicado: reducer operator.add nos campos de lista (evita perda de dados em futuras topologias paralelas)
  - Fix aplicado: removido parâmetro `model` morto de run_agent_debate
  - Aceito (baixa severidade): LLM instanciado por nó (ok para 4 nós), sem retry (add later)

**Status dos testes:** 228 testes passando (12.00s)

**Sessão 11 — Embeddings e ChromaDB** (concluída em 2026-03-05)
- src/agents/rag.py: classe FinancialRAG
  - `embed_pending_news()` — lê notícias pending do PostgreSQL, gera embeddings com sentence-transformers (all-MiniLM-L6-v2), upsert no ChromaDB, marca como 'embedded'
  - `retrieve(ticker, query, top_k=5, max_age_days=30)` — busca semântica no ChromaDB filtrada por ticker, reranking por data (recentes primeiro), tiebreak por distância
  - `_build_document()` — concatena title + summary para embedding
  - `_load_pending_news()` / `_mark_as_embedded()` — persistência PostgreSQL
  - `get_collection_count()` — utility para monitoramento
  - Proteção contra datas futuras no retrieve (exclui artigos com date > today)
  - Collection única "financial_news" com metadatas: ticker, date, source, url, title
  - IDs ChromaDB: "news_{pg_id}" (idempotente via upsert)
  - Batching de 64 artigos por vez para embedding
- tests/test_rag.py: 40 testes (init, build_document, load_pending, mark_embedded, embed_pending, retrieve, reranking, edge cases)
- Security-data review: APROVADO (0 ERROR, 2 WARNING baixa severidade)
  - Fix aplicado: filtro de datas futuras no retrieve()
  - Aceito (baixa severidade): não-atomicidade entre ChromaDB upsert e PostgreSQL mark (mitigado por idempotência do upsert)
  - Aceito (baixa severidade): filtragem de data client-side (ok para volume atual)

**Status dos testes:** 268 testes passando (11.72s)

**Sessão 12 — Integração RAG + Agentes** (concluída em 2026-03-05)
- graph.py: novo nó `rag_retrieval` entre load_context e technical
  - Topologia atualizada: START → load_context → rag_retrieval → technical → fundamental → bear → PM → END
  - Importação lazy do FinancialRAG dentro do nó (evita falha se ChromaDB indisponível)
  - Graceful degradation: try/except com fallback para news_context=[]
  - Query: "{ticker} financial outlook earnings market analysis", top_k=5, max_age_days=30
- personas.py: FUNDAMENTALIST_ANALYST reforçado com regras obrigatórias de citação
  - Seção "Source citation rules (MANDATORY)" adicionada
  - Formato: "Based on [SOURCE] ([DATE]): [claim]"
  - Proibição explícita de inventar notícias: "NEVER invent headlines, quotes, or events"
  - Se news_context vazio: instrui a declarar "No news context available"
- _format_news() atualizado: "(no recent news found for this ticker)" + "cite these sources"
- tests/test_graph.py: 36 testes (5 novos para rag_retrieval + 2 novos para news flow)
  - test_populates_news_context, test_graceful_degradation_on_error
  - test_includes_news_context_in_prompt (fundamentalist recebe RAG)
  - test_rag_news_flows_to_fundamentalist (integração completa)
- Quant-reviewer: APROVADO COM RESSALVAS
  - Ressalva tratada: adicionado teste de news_context fluindo até o fundamentalist
  - Aceito (média severidade): validação de citação depende do prompt compliance do LLM (sem guard programático)
  - Aceito (baixa severidade): mock path frágil (funcional enquanto import for local)

**Status dos testes:** 275 testes passando (20.24s)

### Fase 3 — Validação e Portfolio (Semanas 5-6)

**Sessão 13 — CPCV Implementation** (concluída em 2026-03-06)
- src/backtest/cpcv.py: classe CPCVBacktester
  - `__init__(n_splits=6, n_test_groups=2, embargo_days=10, h=5, input_size=60, rf=0.05)`
  - `generate_paths(n_samples)` — gera C(n_splits, n_test_groups) paths combinatoriais
  - `_split_into_groups(n_samples)` — particiona em n_splits grupos contíguos
  - `_apply_purge(train_indices, test_start, test_end)` — remove purge_window (h + input_size - 1 = 64) antes do test
  - `_apply_embargo(train_indices, test_end, n_samples)` — remove embargo_days após test
  - `_find_contiguous_blocks(indices)` — detecta blocos contíguos para avaliação per-block
  - `_evaluate_predictions(predictions, test_df, test_indices, path_id, test_groups)` — estratégia long/flat com retornos non-overlapping (cada h dias)
  - `_compute_sharpe(returns, rf, periods_per_year)` — Sharpe anualizado, ddof=1
  - `_compute_max_drawdown(cumulative_returns)` — peak-to-trough
  - `_compute_cagr(cumulative_returns, n_days)` — compound annual growth rate
  - `run(df, model_factory)` — pipeline completo: splits → model_factory → evaluate → aggregate
  - `_aggregate_results(fold_results)` — média, std, pct_positive do Sharpe
  - Dataclasses: FoldResult (por path) e BacktestResult (agregado)
  - Protocol ModelFactory: Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]
  - drop_nulls(subset=["close"]) na entrada para proteção contra NaN
  - Logging de warning quando model_factory retorna predictions vazias
- tests/test_cpcv.py: 78 testes (dataclasses, init validation, split, purge, embargo, paths, sharpe, drawdown, cagr, evaluate, run, aggregate, contiguous blocks)
- Architect review: APROVADO
  - Purge window = h + input_size - 1 (conservador, protege contra overlap do input window)
  - Per-ticker obrigatoriamente (loop externo)
  - model_factory desacoplado do TitaniumForecaster
  - Retornos non-overlapping (cada h dias) para evitar inflação de Sharpe
- Quant-reviewer: APROVADO COM RESSALVAS
  - Fix aplicado (#3): retornos computados per contiguous block (evita saltos entre test blocks)
  - Fix aplicado (#4): log warning quando predictions vazias
  - Fix aplicado (#5): drop_nulls(subset=["close"]) na entrada
  - Aceito (WARNING): purge window conservador (64 vs 5) — reduz train set mas mais seguro
  - Aceito (WARNING): variante simplificada do CPCV (sem recombinação de paths) — documentado na docstring
  - Aceito (INFO): sem custos de transação (ok para validação de sinal)

**Status dos testes:** 353 testes passando (16.43s)

**Sessão 14 — Simulação de Custos e Relatório** (concluída em 2026-03-06)
- src/backtest/cpcv.py: custos de transação adicionados
  - `TransactionCosts` frozen dataclass: slippage_bps=5.0, commission_bps=10.0, market_impact_bps=0.0
  - `CPCVBacktester.__init__` agora aceita `costs: TransactionCosts | None = None`
  - `costs=None` (default): zero custos, backward compatible com testes anteriores
  - Custos aplicados em cada mudança de posição (flat↔long)
  - Market impact: inversamente proporcional à liquidez `1/sqrt(relative_vol)`
  - Posição resetada a flat no início de cada bloco contíguo (quant-reviewer fix #1)
  - FoldResult novos campos: n_trades, total_costs, equity_curve
  - BacktestResult novos campos: mean_n_trades, mean_total_costs
- src/backtest/report.py: geração de relatório PDF (NOVO)
  - `BacktestReport(result, ticker, output_dir)` → `generate()` → PDF
  - Página 1: tabela de métricas por path + overlay de equity curves
  - Página 2: violinplot de Sharpe + bar chart de max drawdown
  - matplotlib (backend Agg) + seaborn
  - Output: `data/outputs/backtest_report_{TICKER}.pdf`
- docs/backtest_metrics.md: documentação completa das métricas
- tests/test_cpcv.py: 94 testes (+16 novos: TransactionCosts + CostsIntegration)
- tests/test_report.py: 15 testes (init, PDF generation, plots, edge cases)
- Dependências adicionadas: matplotlib>=3.10, seaborn>=0.13
- Architect review: APROVADO
  - TransactionCosts como dataclass frozen (agrupamento semântico, imutável)
  - costs=None preserva backward compat exata (zero custos)
  - equity_curve persistido em FoldResult (evita recomputação no report)
- Quant-reviewer: APROVADO COM RESSALVAS
  - Fix aplicado (#1): prev_position resetado a 0 no início de cada bloco contíguo
  - Fix aplicado (#3): market impact invertido para 1/sqrt(relative_vol) (alto volume → menor impacto)
  - Aceito (WARNING #2): exit cost no final do backtest não modelado (impacto marginal)
  - Aceito (INFO #5): avg_volume global vs per-block (baixa severidade)
  - Zero look-ahead bias confirmado

**Status dos testes:** 384 testes passando (18.76s)

**Sessão 15 — Hierarchical Risk Parity** (concluída em 2026-03-06)
- src/portfolio/hrp.py: classe HRPOptimizer (Lopez de Prado, 2016)
  - `HRPConfig` frozen dataclass: linkage_method, correlation_method, confidence_tilt_cap, min/max_weight
  - `HRPResult` dataclass: weights, raw_weights, cluster_order, linkage_matrix
  - `_compute_covariance()` — cov + corr (Pearson ou Spearman) via numpy
  - `_compute_distance()` — d = sqrt(0.5 * (1 - corr)), condensed form para scipy
  - `_cluster()` — scipy.cluster.hierarchy.linkage (single/complete/average/ward)
  - `_quasi_diagonalize()` — seriation recursiva do dendrograma
  - `_get_cluster_var()` — variância IVP do cluster (diagonal weights, full cov)
  - `_recursive_bisection()` — alocação inversamente proporcional à variância do cluster
  - `_apply_confidence_tilt()` — multiplier = 1 + cap * (conf - 0.5), clip + renorm
  - `_clip_and_normalise()` — clipping [min_weight, max_weight] + renormalização (sempre aplicado)
  - `optimize(returns, confidences)` — pipeline completo: cov → dist → cluster → seriate → bisect → tilt
  - Single-asset shortcut (N=1): retorna 100% sem clustering
  - Warning para observações < 120 e variância near-zero
- tests/test_hrp.py: 67 testes (config, init, cov, distance, cluster, seriation, bisection, cluster_var, tilt, optimize, edge cases)
- Dependências adicionadas: scipy>=1.11, numpy>=1.24
- Architect review: APROVADO
  - Single linkage (default, fiel ao LdP); ward configurável
  - Pearson default; Spearman configurável
  - Confidence tilt ±20% (range efetivo [0.90, 1.10])
  - Input: retornos DataFrame Polars (não cov matrix)
  - Zero imports de outros módulos src/ (independência)
- Quant-reviewer: APROVADO COM RESSALVAS
  - Fix aplicado: max_weight clipping aplicado sempre (com e sem confidences)
  - Fix aplicado: warning para ativos com variância near-zero (< 1e-8)
  - Fix aplicado: docstring imprecisa na quasi-diagonalização
  - Aceito (baixa severidade): sem validação de NaN na entrada (responsabilidade do caller)
  - Aceito (baixa severidade): sem filtragem de colunas não-numéricas (documentar)
  - Aceito (INFO): ativo com variância zero recebe peso dominante (matematicamente correto para HRP)

**Status dos testes:** 451 testes passando (384 anteriores + 67 novos)

**Sessão 16 — Pipeline Final de Decisão** (concluída em 2026-03-06)
- src/portfolio/decision_engine.py: classe DecisionEngine
  - `TickerDecision` frozen dataclass: ticker, action, weight, confidence, reasoning, dissenting_view
  - `DecisionOutput` dataclass: timestamp, tickers, decisions, hrp_raw/final_weights, cluster_order, metadata
  - `_load_ohlcv()` — carrega OHLCV via PredictionPipeline (lazy import, evita yfinance no import time)
  - `_compute_returns()` — log returns (`log().diff().over("ticker")`), pivot wide, trim lookback_days
  - `_run_debate()` — LangGraph agent debate com graceful degradation (retorna [] em falha)
  - `_extract_confidences()` — extrai {ticker: confidence}, missing=0.5 (neutro)
  - `_run_hrp()` — HRPOptimizer com confidence tilt
  - `_merge_actions()` — HOLD/SELL peso → 0, redistribuído proporcionalmente entre BUY
  - `_build_output()` — combina tudo com metadata (schema_version, hrp_config, n_observations)
  - `_save_json()` — persiste em decisions.json
  - `run()` — pipeline completo: OHLCV → returns → debate → HRP → merge → save
  - Validação: retornos < 2 obs → ValueError
  - DEFAULT_TICKERS definido localmente (evita import chain de yfinance)
  - lookback_days=504 (~2 anos) para estimativa de covariância
- tests/test_decision_engine.py: 31 testes (dataclasses, init, compute_returns, extract_confidences, merge_actions, debate, build_output, save_json, pipeline integration)
- Makefile: target `make decide` adicionado
- Architect review: APROVADO
  - Reutiliza PredictionPipeline.load_ohlcv() (DRY)
  - DecisionOutput como dataclass (consistente com HRPResult, FoldResult)
  - Graceful degradation: debate falha → HRP sem tilt
  - _merge_actions: HOLD/SELL → peso 0, BUY reescalado para soma=1
  - Log returns para HRP (correto para estimativa de covariância)
- Quant-reviewer: APROVADO COM RESSALVAS
  - Fix aplicado (P1): numeração de steps no log corrigida
  - Fix aplicado (P2): validação de retornos mínimos (< 2 obs → ValueError)
  - Fix aplicado (P3): exceções filtradas especificamente (ImportError, RuntimeError, ValueError, OSError)
  - Aceito (INFO): fixture mock_ohlcv inclui fins de semana (aceitável para testes)
  - Zero look-ahead bias confirmado, cálculo de log returns correto

**Status dos testes:** 482 testes passando (451 anteriores + 31 novos)

**Sessão 17 — Dashboard Streamlit** (concluída em 2026-03-06)
- src/dashboard/app.py: dashboard Streamlit com 3 abas
  - **Performance**: donut chart (HRP weights), bar chart (raw vs tilted), metric cards (BUY/HOLD/SELL/confidence/obs), decision table, cluster order
  - **War Room**: debate replay por ticker, chat bubbles com cor/ícone por agente (técnico=azul, fundamental=verde, bear=vermelho, PM=dourado), decisão final em destaque, dissenting view, timeline expandível
  - **Microstructure**: fan chart PatchTST com bandas de confiança (90% e 50% CI), prediction cards (P(up), expected return), seletor de ticker
  - Dark theme profissional com Plotly interativo
  - Data loaders com `@st.cache_data(ttl=300)` para decisions.json, debate_history.json, forecast.parquet, predictions.parquet
  - Graceful degradation: cada aba mostra `st.warning()` quando dados ausentes
- src/agents/graph.py: `run_agent_debate()` agora retorna `tuple[list[FinalDecision], dict[str, dict]]` (decisões + estados completos do debate)
- src/portfolio/decision_engine.py: `_run_debate()` atualizado para tuple, novo `_save_debate_history()` persiste reports/debate_log/predictions/news_context em `debate_history.json`
- tests/test_dashboard.py: 12 testes (agent styles, data loaders, chart builders)
- tests/test_decision_engine.py: 31 testes atualizados para novo retorno tuple
- Architect review: APROVADO
  - Dashboard lê apenas flat files (JSON/Parquet) — sem acesso PostgreSQL
  - Debate history persistido para War Room (reports individuais + debate_log)
  - Graceful degradation em todas as abas
  - `st.cache_data(ttl=300)` para performance
- Mudanças prerequisitas implementadas:
  - `run_agent_debate()` retorna estados completos (não mais descarta reports)
  - `decision_engine._save_debate_history()` persiste debate para consumo do dashboard

**Status dos testes:** 494 testes passando (482 anteriores + 12 dashboard)

**Sessão 18 — Streaming dos Agentes** (concluída em 2026-03-06)
- src/agents/graph.py: suporte a streaming per-node
  - `_NodeCallback = Callable[[str, str, dict[str, Any]], None]` — type alias para callback
  - `run_agent_debate()` aceita `on_node_complete` callback opcional
  - Quando callback presente: usa `graph.stream()` para updates per-node (load_context, rag, technical, fundamental, bear, PM)
  - Quando callback ausente: usa `graph.invoke()` (backward compatible)
  - Cada node_output passado ao callback com (ticker, node_name, partial_state)
- src/dashboard/app.py: War Room com 3 modos
  - **Replay mode**: `_replay_debate()` — animação com typing indicator (`st.empty()` + `time.sleep(delay)`)
  - **Live mode**: `_run_live_debate_thread()` em `threading.Thread` + `queue.Queue` para posting de eventos
    - Callback `on_node` posta `("node", ticker, node_name, output)` na fila
    - Worker posta `("done", decisions, states)` ou `("error", msg)` ao final
  - `_render_live_debate()` — drena fila, renderiza reports/decisions, auto-rerun com `st.rerun()`
  - **Static mode**: visualização padrão (sem animação)
  - `_NODE_TO_AGENT` mapping: load_context→context, rag_retrieval→rag, technical→technical, fundamental→fundamental, bear→bear, portfolio_manager→pm
  - Imports adicionados: queue, threading, time, datetime
- tests/test_dashboard.py: 21 testes (+9 novos)
  - TestNodeToAgent (2): analyst nodes + context nodes mapeados
  - TestReplayDebate (3): renders all reports, uses delay, empty reports
  - TestLiveDebateThread (4): done on success, error on failure, passes callback, callback posts node events
  - sys.modules patching para evitar import de langchain_anthropic (mesmo padrão de test_decision_engine)

**Status dos testes:** 503 testes passando (494 anteriores + 9 novos)

**Sessão 19 — README e Documentação** (concluída em 2026-03-07)
- README.md: README profissional em inglês para GitHub internacional
  - Badges: Python 3.10+, 503 testes, MIT, CI
  - Elevator pitch acessível para recrutadores não-quant
  - "Why This Matters" — valor de negócio (múltiplas perspectivas, CPCV, HRP)
  - Diagrama Mermaid completo do fluxo de dados (6 subgraphs)
  - Key Features table (8 componentes), Quick Start (5 comandos)
  - Project Structure tree, How It Works (7 steps + code snippet)
  - Test coverage table (503 testes por área), Tech Stack (17 tecnologias)
- ARCHITECTURE.md: documentação técnica completa
  - Data Flow Diagram (Mermaid flowchart — 5 camadas)
  - Module Reference — 15 módulos com classes, responsabilidades, dependências
  - Agent Topology (Mermaid sequence diagram — fluxo de debate por ticker)
  - Key Design Decisions table (9 decisões com rationale e alternativas)
  - Testing Strategy — 4 camadas (unit/integration/contract/validation), mock patterns
- Docstrings: audit completo de todos os 23 arquivos src/
  - 15 módulos já tinham docstrings completas (Google Style)
  - 8 __init__.py estavam vazios — preenchidos com docstrings de módulo
  - Funções de streaming da Sessão 18 já documentadas

**Sessão 20 — CI/CD e Organização Final** (concluída em 2026-03-07)
- .github/workflows/ci.yml: GitHub Actions CI
  - Roda pytest em cada push/PR para master
  - Python 3.12, Poetry install, variáveis de ambiente mock
  - Nenhuma API real chamada (todos os testes usam mocks)
- .github/workflows/lint.yml: GitHub Actions linting
  - Jobs separados para ruff (check) e mypy (type check)
  - ruff check src/ tests/ e mypy src/ --ignore-missing-imports
- docker/Dockerfile: imagem de produção para dashboard Streamlit
  - Base python:3.12-slim, deps de sistema (gcc, libpq-dev)
  - Poetry install --only main (sem dev deps)
  - Layer caching: pyproject.toml antes de src/
  - Healthcheck via /_stcore/health
  - ENTRYPOINT: streamlit run src/dashboard/app.py
- .env.example: template com todas as variáveis necessárias
  - ANTHROPIC_API_KEY, OPENAI_API_KEY, NEWSAPI_KEY
  - POSTGRES_HOST/PORT/DB/USER/PASSWORD
  - CHROMA_HOST/PORT
- .dockerignore: exclui tests, notebooks, docs, .git, cache dirs
- .gitignore: expandido com pytest_cache, mypy_cache, ruff_cache, coverage, logs, Docker volumes
- Security scan: zero API keys hardcoded no código (grep confirmado)
- Repositório organizado: sem arquivos temporários, sem outputs commitados

### Fase 4b — Benchmark Real (Sessões 21+)

**Sessão 21 — Configuração de Tickers** (concluída em 2026-03-07)
- config/tickers.json: 52 tickers US + SPY benchmark
- src/config.py: `load_ticker_config()`, `load_tickers()`, `load_benchmark()`
  - Fallback para 4 tickers originais se config/tickers.json não existir
  - Validação de campos obrigatórios (tickers, benchmark, market)
  - Deep copy do fallback (evita mutação)
- src/data/ingestion.py: `_resolve_tickers()` — tenta config, fallback para hardcoded
- src/models/predict.py: usa `_resolve_tickers()` em vez de `DEFAULT_TICKERS`
- src/portfolio/decision_engine.py: usa `_resolve_tickers()` com fallback
- src/agents/graph.py: usa `_resolve_tickers()` no `run_agent_debate()`
- tests/test_config.py: 13 testes (valid, minimal, missing keys, fallback, real config)

**Sessão 22 — HRP max_weight Dinâmico** (concluída em 2026-03-07)
- src/portfolio/decision_engine.py: `__init__` calcula `HRPConfig(max_weight=min(0.25, 2/n))`
  - 4 tickers → max_weight=0.25 (backward compat)
  - 50 tickers → max_weight=0.04 (4%)
  - Config explícito nunca sobrescrito
- tests/test_decision_engine.py: 3 testes novos (dynamic 4/50 tickers + explicit config)

**Sessão 23 — Download Paralelo de Dados** (concluída em 2026-03-07)
- src/data/ingestion.py: `MarketDataIngester` com novos parâmetros
  - `start_date`/`end_date`: datas explícitas (necessário para walk-forward)
  - `max_workers=5`: threads paralelas para download
  - `_download_batch()`: `ThreadPoolExecutor` com stagger de 0.5s, falha parcial tolerada
  - `run(parallel=True)`: paralelo por default, `parallel=False` para backward compat
  - Tickers que falham são skipados (log error), não travam o pipeline
  - Falha total (nenhum ticker baixado) → `RuntimeError`
- tests/test_ingestion.py: 8 testes novos (parallel, partial failure, date range)

**Sessão 24 — Walk-Forward Backtester** (concluída em 2026-03-07)
- src/backtest/walk_forward.py: classe WalkForwardBacktester (NOVO — ~620 linhas)
  - `ModelFactory` Protocol: `train(pl.DataFrame) -> None`, `predict(pl.DataFrame) -> dict[str, float]`
  - `NaiveModelFactory`: momentum-based, confidence em [0.05, 0.95] (para validação rápida)
  - `WalkForwardConfig` frozen dataclass: retrain_every=126, rebalance_every=5, lookback_days=504, initial_capital=1M, costs, min_rebalance_delta=0.02, rf=0.05
  - `RebalanceRecord`: date, weights, turnover, costs, retrained
  - `WalkForwardResult`: equity_curve, daily_returns, rebalance_history, metrics, config, metadata
  - `run()` — loop principal com dois ciclos: retrain (lento, 126 dias) + rebalance (rápido, 5 dias)
  - Weight drift via per-asset dollar tracking (holdings dict, não constant-mix)
  - HRP com confidence tilt integrado (max_weight=min(0.25, 2/n))
  - Benchmark buy-and-hold computado em paralelo
  - `_compute_daily_returns()` — simple returns wide-format
  - `_compute_log_returns_for_hrp()` — log returns filtrados por data e lookback
  - `_apply_costs()` — turnover-based com TransactionCosts
  - min_rebalance_delta: skip rebalance se turnover < threshold
  - Zero look-ahead bias: todos os dados filtrados por `<= current_date`
- tests/test_walk_forward.py: 38 testes (config, dataclasses, NaiveModelFactory, returns, costs, run, validation, drift, benchmark, look-ahead)
- Quant-reviewer: APROVADO COM RESSALVAS
  - Fix aplicado: constant-mix bug → per-asset dollar holdings tracking
  - Fix aplicado: total_retrains counter independente
  - Fix aplicado: market_impact_bps documentado como limitação (volume per-asset não disponível)
  - Aceito (INFO): weight drift test usa random noise para variância HRP realista

**Status dos testes:** 408 testes passando (17.06s, módulos disponíveis)

**Sessão 25 — Métricas de Portfolio vs Benchmark** (concluída em 2026-03-07)
- src/backtest/benchmark_metrics.py: módulo completo de métricas (NOVO)
  - `compute_benchmark_metrics()` — função principal, retorna dict com 16 métricas
  - **Retorno**: cagr, total_return, benchmark_total_return
  - **Risco**: annualized_volatility, max_drawdown, max_drawdown_duration_days, calmar_ratio
  - **Risk-adjusted**: sharpe_ratio, sortino_ratio
  - **vs Benchmark**: information_ratio, alpha (Jensen's), beta (CAPM), tracking_error, hit_rate_monthly
  - **Turnover**: avg_annual_turnover, avg_positions
  - Helpers: `_cumulative_from_returns()`, `_compute_drawdown_series()`, `_max_drawdown_duration()`, `_compute_monthly_returns()`, `_capm_regression()` (OLS)
  - Sortino: downside deviation per Sortino (1994) — E[min(r,0)²] sobre TODAS obs
  - CAPM alpha: anualização linear (alpha_daily * 252) — aproximação documentada
  - rf_daily = rf / 252 — aproximação linear (convenção padrão)
- src/backtest/walk_forward.py: integração automática
  - `run()` agora chama `compute_benchmark_metrics()` e preenche `result.metrics`
  - Import lazy dentro do método (evita circular)
- tests/test_benchmark_metrics.py: 40 testes (helpers, métricas individuais, edge cases, integração)
- Quant-reviewer: APROVADO APÓS FIX
  - Fix aplicado (ERROR): Sortino downside deviation — dividia por len(negative) em vez de n
  - Aceito (WARNING): alpha annualization linear (convenção padrão)
  - Aceito (WARNING): rf decomposition linear (diferença <2%)
  - Zero look-ahead bias confirmado

**Status dos testes:** 448 testes passando (18.15s, módulos disponíveis)

**Sessão 26 — Relatório PDF do Benchmark** (concluída em 2026-03-07)
- src/backtest/benchmark_report.py: classe BenchmarkReport (NOVO)
  - `generate()` → PDF com 6 páginas via matplotlib + PdfPages
  - **Página 1**: Equity curve portfolio vs benchmark (escala log)
  - **Página 2**: Drawdown chart (área preenchida vermelha)
  - **Página 3**: Tabela de métricas (16 métricas formatadas, cores alternadas)
  - **Página 4**: Rolling Sharpe 252 dias (portfolio vs benchmark)
  - **Página 5**: Heatmap de pesos HRP (top 15 ativos por peso médio)
  - **Página 6**: Turnover por rebalanceo (bar chart, cores diferenciadas para retrain)
  - Graceful degradation: dados insuficientes → mensagem no plot
  - Helpers: `_compute_drawdown()`, `_rolling_sharpe()`, `_format_metrics_rows()`
  - Output: `data/outputs/benchmark_report.pdf`
- tests/test_benchmark_report.py: 19 testes (helpers, init, PDF generation, edge cases)

**Status dos testes:** 467 testes passando (25.06s, módulos disponíveis)

**Sessão 27 — Integração e Pipeline Completo** (concluída em 2026-03-07)
- src/backtest/run_benchmark.py: script orquestrador (NOVO)
  - `run_us_benchmark()` — pipeline 7 passos: config → OHLCV → OOS filter → model_factory → walk-forward → PDF → save
  - `_load_ohlcv_from_postgres()` — carrega OHLCV do PostgreSQL para todos os tickers + benchmark
  - `_filter_oos_period()` — filtra últimos n_years de dados
  - `_resolve_model_factory()` — PatchTST (com fallback) ou NaiveModelFactory
  - `_PatchTSTModelFactory` — adapter para TitaniumForecaster (ModelFactory protocol)
  - `_save_outputs()` — salva equity.parquet, metrics.json, weights.parquet
  - Aceita `ohlcv` pré-carregado (para testes e uso programático)
  - PDF failure é non-fatal (try/except com warning)
  - CLI: `python -m src.backtest.run_benchmark [--naive]`
- WalkForwardConfig: rebalance_every=5, retrain_every=126, costs(5bps+10bps), min_delta=0.02
- Makefile: targets `benchmark` e `benchmark-fast` adicionados
- Output: `data/outputs/benchmark_{equity,metrics,weights,report}.*`
- tests/test_run_benchmark.py: 15 testes (filter, model_factory, save, e2e integration)

**Status dos testes:** 482 testes passando (33.11s, módulos disponíveis)

**Sessão 28 — Dashboard: Aba de Benchmark** (concluída em 2026-03-07)
- src/dashboard/app.py: nova aba "Benchmark" (tab 0, antes de Performance)
  - **Equity curve**: Plotly interativo, portfolio vs benchmark, toggle log scale
  - **Drawdown chart**: área preenchida vermelha com drawdown percentual
  - **Tabela de métricas**: 16 métricas com formatação condicional (verde/vermelho)
  - **Rolling Sharpe**: slider de janela (60-504 dias), portfolio vs benchmark
  - **Weight heatmap**: top 15 ativos por peso médio, evolução ao longo do tempo
  - Graceful degradation: mensagem informativa quando dados ausentes
- Data loaders: `load_benchmark_equity()`, `load_benchmark_metrics()`, `load_benchmark_weights()`
  - Todos com `@st.cache_data(ttl=300)` e return None se arquivo ausente
  - Lê: benchmark_equity.parquet, benchmark_metrics.json, benchmark_weights.parquet
- Helpers: `_format_metric_value()`, `_metric_color()`, `_chart_benchmark_equity()`,
  `_chart_benchmark_drawdown()`, `_chart_rolling_sharpe()`, `_chart_weight_heatmap()`
- tests/test_dashboard.py: 18 testes novos (loaders, formatting, charts)
- Dashboard agora tem 4 abas: Benchmark, Performance, War Room, Microstructure

**Status dos testes:** 500 testes passando (32.02s, módulos disponíveis)

### Fase 4b Benchmark — Completa (Sessões 21-28)

### Fase 5 — Melhorias do Benchmark (Sessões 29+)

**Sessão 29 — CPCV-OOS Parameter Validator** (concluída em 2026-03-08)
- src/backtest/cpcv_oos.py: módulo de validação CPCV-OOS (NOVO — ~760 linhas)
  - `CPCVParameterValidator`: valida configs walk-forward via CPCV no período OOS
  - Divide OOS em `n_splits=6` blocos temporais, gera `C(n_splits, n_test_groups)` paths
  - `_PurgedModelFactory`: wrapper que exclui test+embargo dates do `train()` do modelo
  - `copy.deepcopy(model_factory)` por path — previne contaminação cross-path
  - `_evaluate_path()` → `tuple[float, list[float]]` — (Sharpe, test returns)
  - `_get_test_dates()` / `_get_train_dates()` — extração de datas com embargo
  - `_filter_ohlcv_by_dates()` — helper de filtragem
  - `validate()` — roda backtester em cada path, agrega Sharpes, computa DSR
  - `grid_search()` — valida múltiplos configs, ranked por DSR
  - Aceitação: `pct_positive >= 0.6667 AND dsr_pvalue > 0.95`
  - `deflated_sharpe_ratio()` — Bailey & Lopez de Prado (2014)
    - Euler-Mascheroni para E[max(Z)]
    - Lo (2002) V[SR] = (1 + 0.5*SR² - skew*SR + (kurt-3)/4*SR²) / T
    - Skewness/kurtosis empíricos computados dos retornos de teste reais
    - n_observations: contagem real do path mais longo (não estimativa)
  - `ValidationResult` dataclass: mean_sharpe, std_sharpe, pct_positive, per_path_sharpe, deflated_sharpe, p_value, accepted, metadata
  - Math helpers: `_normal_cdf`, `_inv_normal_cdf` (Abramowitz & Stegun), `_compute_sharpe`, `_skewness`, `_kurtosis`, `_std_list`
- tests/test_cpcv_oos.py: 66 testes (math helpers, PurgedModelFactory, DSR, splits, embargo, evaluate, validate, grid_search, integração)
- Quant-reviewer: APROVADO
  - 3 ERRORs corrigidos: train/test separation (PurgedModelFactory), Lo(2002) +0.5*SR², cross-path contamination (deepcopy)
  - 4 WARNINGs corrigidos: empirical skew/kurt, n_obs real, DSR threshold 0.95, kurtosis docstring

**Status dos testes:** 566 testes passando (37s, módulos disponíveis)

**Sessão 30 — Fixes Estruturais no Walk-Forward** (concluída em 2026-03-08)
- src/backtest/walk_forward.py: 2 fixes estruturais
  - **Fix `_compute_log_returns_for_hrp`**: `drop_nulls()` → truncar leading rows incompletas + `fill_null(0.0)` para nulls interiores
    - Antes: `drop_nulls()` descartava TODA row se qualquer ticker tinha null (com 52 tickers eliminava muitos dias)
    - Agora: descarta leading rows onde algum ticker não tem dado (período antes do IPO), mantém o restante
    - `all_present = pl.all_horizontal(is_not_null)` para encontrar primeira data completa
    - Evita deflação artificial de variância (bias no HRP que over-alocava tickers com histórico curto)
  - **Fix `NaiveModelFactory.predict`**: scaling proporcional ao lookback
    - Antes: `ret * 10` (hardcoded) saturava clamp com lookback > 5
    - Agora: `scaling = 50.0 / max(self.lookback, 1)` → lookback=5 gera scaling=10 (backward compat), lookback=63 gera scaling=0.79
    - Evita binarização (tudo 0.95 ou 0.05) com lookback longo
- tests/test_walk_forward.py: 6 testes novos
  - Scaling: backward compat (lookback=5), not saturated (lookback=63), edge case (lookback=1), range check
  - fill_null: preserva rows com dados parciais, zero nulls na saída
- Quant-reviewer: APROVADO COM RESSALVAS
  - Fix extra aplicado: substituído `fill_null(0.0)` puro por truncar leading + fill interior
  - Aceito (INFO): scaling `50/lookback` é ad-hoc mas adequado para modelo de validação

**Status dos testes:** 572 testes passando (36s, módulos disponíveis)

**Sessão 31 — HRP Ward Linkage + Ledoit-Wolf Shrinkage** (concluída em 2026-03-08)
- src/portfolio/hrp.py: Ledoit-Wolf shrinkage adicionado
  - `HRPConfig.shrinkage: bool = False` — nova opção (backward compatible)
  - `_compute_covariance()`: quando `shrinkage=True`, usa `sklearn.covariance.LedoitWolf` em vez de `np.cov`
  - Import lazy do sklearn (dentro do bloco `if shrinkage`)
  - Correlação derivada da covariância shrunk via `cov / outer(std, std)` (matematicamente correto)
  - Interação shrinkage+spearman: cov shrunk + corr Spearman (rank-based, não se beneficia de regularização)
  - Log do coeficiente de shrinkage (`lw.shrinkage_`) para diagnóstico
  - Ward linkage já era suportado — apenas testado para validar clusters mais balanceados
- pyproject.toml: `scikit-learn>=1.3` adicionado às dependências
- tests/test_hrp.py: 13 testes novos (80 total, era 67)
  - TestLedoitWolfShrinkage (11): PSD validity, corr diagonal, bounded, sum=1, backward compat, convergence, confidences, deterministic, spearman, few_obs_many_assets
  - TestWardLinkage (2): ward mais balanceado que single (dispersão de pesos), linkage matrix válida
- Quant-reviewer: APROVADO COM RESSALVAS
  - Zero look-ahead bias (shrinkage é estimador, não parâmetro)
  - Zero overfitting risk (coeficiente analítico, não otimizado)
  - Aceito (INFO): warning de near-zero variance inspeciona após shrinkage (seguro, perde utilidade diagnóstica)

**Status dos testes:** 585 testes passando (34s, módulos disponíveis)

**Sessão 32 — Volatility Targeting** (concluída em 2026-03-09)
- src/backtest/walk_forward.py: vol targeting overlay no loop principal
  - `WalkForwardConfig` novos campos: `target_vol` (float|None=None), `vol_lookback` (int=63), `max_leverage` (float=1.0), `min_leverage` (float=0.5)
  - `target_vol=None` (default): nenhum vol targeting aplicado (backward compat)
  - Lógica: `leverage = target_vol / realized_vol`, clamped a `[min_leverage, max_leverage]`
  - Realized vol: `std(returns[-vol_lookback:]) * sqrt(252)`, ddof=1
  - Cash handling: `cash = portfolio_value - sum(scaled_holdings)`, cash earns 0 (simplificação documentada)
  - Scale: `holdings[t] *= leverage * portfolio_value / invested` (proporcional)
  - Aplicado APÓS rebalance, ANTES de retornos diários (zero look-ahead)
  - Subsume drawdown killswitch: `target_vol=0.10, min_leverage=0.0` = killswitch conservador
  - Validação: `vol_lookback >= 2`, `max_leverage >= min_leverage` (quando target_vol definido)
- tests/test_walk_forward.py: 15 testes novos (59 total, era 44)
  - TestVolatilityTargeting (15): backward compat, config defaults/custom, exposure reduction, max/min leverage, killswitch, warmup, vol reduction, holdings non-negative, matching vol, vol_lookback=2 edge, vol_lookback=1 raises, max<min raises, realized_vol=0
- Quant-reviewer: APROVADO COM RESSALVAS
  - Zero look-ahead bias (usa returns_port passados, antes de append do dia corrente)
  - Fix aplicado: validação vol_lookback>=2 e max_leverage>=min_leverage
  - Fix aplicado: testes para edge cases (vol_lookback=2, realized_vol=0)
  - Aceito (WARNING): cash earns nothing (~rf*cash_frac understated)
  - Aceito (INFO): custos aplicados sobre turnover full antes de vol scaling

**Status dos testes:** 601 testes passando (42s, módulos disponíveis)

**Sessão 33 — Drawdown Killswitch** (concluída em 2026-03-09)
- src/backtest/walk_forward.py: killswitch overlay no loop principal
  - `KillswitchConfig` frozen dataclass: `max_drawdown_pct=-0.15`, `recovery_threshold_pct=-0.05`, `ramp_up_days=21`
  - `WalkForwardConfig.killswitch: KillswitchConfig | None = None` (backward compat)
  - Trigger: `dd = portfolio_value / peak_value - 1 <= max_drawdown_pct` → liquidate tudo, go to cash
  - Em cash: `port_ret=0`, skip rebalance/retrain/vol targeting, holdings all 0
  - Recovery: usa **benchmark** drawdown (não portfolio) como proxy — evita bug lógico de DD constante em cash
  - `days_recovering` conta dias consecutivos com `bench_dd >= recovery_threshold_pct`
  - Re-entry: quando `days_recovering >= ramp_up_days`, sai de cash, força rebalance
  - `peak_value = portfolio_value` no re-entry — evita re-trigger imediato do pico pré-crash
  - Exit costs: aplicados via `_apply_costs()` sobre turnover de liquidação
  - Validação: `ramp_up_days >= 1` (evita ZeroDivisionError)
- tests/test_walk_forward.py: 15 testes novos (74 total, era 59)
  - TestKillswitchConfig (3): defaults, custom, frozen
  - TestDrawdownKillswitch (12): backward compat, trigger, cash stability, exit costs, benchmark recovery, ramp, vol targeting interaction, config fields, portfolio positive, ramp_up_days=0 raises, no re-trigger
- Quant-reviewer: APROVADO (após 2 rounds)
  - Fix aplicado (ERROR): ramp_up_days >= 1 validation
  - Fix aplicado (ERROR): docstring corrigida — ramp é wait period, não exposição gradual
  - Fix aplicado (WARNING): peak_value resetado no re-entry (evita oscilação)
  - Aceito (cosmetic): variável `ramp` poderia ser renomeada para `recovery_progress`

**Status dos testes:** 616 testes passando (36s, módulos disponíveis)

## O que NUNCA fazer
- Nunca hardcode API keys (usar .env + python-dotenv)
- Nunca usar Pandas (usar Polars — é a escolha do projeto)
- Nunca fazer train/test split simples (sempre CPCV)
- Nunca rodar código que acessa APIs reais nos testes (usar mocks)
- Nunca commitar notebooks com output (limpar antes)