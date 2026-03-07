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

### Projeto completo — Todas as 20 sessões concluídas

## O que NUNCA fazer
- Nunca hardcode API keys (usar .env + python-dotenv)
- Nunca usar Pandas (usar Polars — é a escolha do projeto)
- Nunca fazer train/test split simples (sempre CPCV)
- Nunca rodar código que acessa APIs reais nos testes (usar mocks)
- Nunca commitar notebooks com output (limpar antes)