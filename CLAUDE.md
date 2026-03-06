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

### Próxima sessão: Sessão 13 — CPCV Implementation (src/backtest/cpcv.py)

## O que NUNCA fazer
- Nunca hardcode API keys (usar .env + python-dotenv)
- Nunca usar Pandas (usar Polars — é a escolha do projeto)
- Nunca fazer train/test split simples (sempre CPCV)
- Nunca rodar código que acessa APIs reais nos testes (usar mocks)
- Nunca commitar notebooks com output (limpar antes)