# Titanium Alpha — Contexto do Projeto

## O que é este projeto
Sistema de fundo de hedge agêntico multi-estratégia que combina:
- PatchTST (Deep Learning para séries temporais financeiras)
- Multi-agente LangGraph (debate entre Analista Técnico, Fundamentalista, Bear, Portfolio Manager)
- RAG financeiro com ChromaDB
- CPCV-OOS (Combinatorial Purged Cross-Validation com Deflated Sharpe Ratio)
- HRP (Hierarchical Risk Parity com Ledoit-Wolf shrinkage) para alocação
- Walk-forward backtest com vol targeting e drawdown killswitch

## Stack tecnológico
Python 3.10+, Polars, NeuralForecast (Nixtla), LangGraph, ChromaDB,
PostgreSQL, Streamlit, Plotly, Docker, Poetry, scikit-learn, scipy

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
src/backtest/      → CPCV, CPCV-OOS, walk-forward, benchmark metrics, reports
src/portfolio/     → HRP, decision engine
src/dashboard/     → Streamlit UI (4 abas)
src/utils/         → helpers compartilhados
tests/             → pytest, fixtures em conftest.py
notebooks/         → exploração (nunca importado por src/)
docs/              → documentação e referências
docker/            → Dockerfiles e compose
config/            → tickers.json (52 US large caps + SPY)

## Convenções de git
feat: nova funcionalidade
fix: correção de bug
refactor: sem mudança de comportamento
test: adição/modificação de testes
docs: apenas documentação

## Contexto financeiro importante
- Universo: 52 large caps US + SPY benchmark (config/tickers.json)
- Período de dados: últimos 5 anos (OHLCV diário via yfinance)
- Benchmark: SPY buy-and-hold
- Métricas-chave: Sharpe Ratio anualizado (rf=0.05), Max Drawdown, CAGR

## Configuração validada (CPCV-OOS)
- NaiveModelFactory(lookback=1) — momentum ultra-short-term
- rebalance_every=1 (diário), retrain_every=126 (semestral)
- lookback_days=63 (~3 meses de covariância)
- target_vol=0.15, vol_lookback=21
- HRPConfig: confidence_tilt_cap=1.0, max_weight=0.15
- TransactionCosts: slippage=5bps, commission=10bps
- min_rebalance_delta=0.01
- Resultados: Sharpe ~2.7, CAGR ~45%, MaxDD ~-20%

## Limitação conhecida: gap backtest-produção
O Sharpe validado (~2.7) reflete o sinal PatchTST sozinho. O pipeline de produção
(make decide) usa debate LangGraph, que nunca foi backtestado. Quando os agentes
falham, o sistema cai para BUY com confidence=0.5 para todos os tickers,
descartando o prob_up do PatchTST. Ver docs/design_gap_backtest_vs_production.md.

## Histórico de implementação

### Fase 1 — Infraestrutura e Dados (Sessões 1-7)
- Estrutura do projeto, Docker (PostgreSQL + ChromaDB), ingestão OHLCV e notícias
- Feature engineering (RSI, Bollinger, Vol, VWAP, OBV) — zero look-ahead bias
- PatchTST com MQLoss (5 quantis), CDF interpolation para prob_up contínuo
- Pipeline de predições (PostgreSQL → features → PatchTST → Parquet)

### Fase 2 — Camada Agêntica (Sessões 8-12)
- Design e implementação dos 4 agentes LangGraph (pipeline linear)
- State management com TypedDicts + reducers, Pydantic structured output
- RAG financeiro: sentence-transformers → ChromaDB → semantic retrieval
- Integração RAG + Agentes (fundamentalist cita fontes, graceful degradation)

### Fase 3 — Validação e Portfolio (Sessões 13-16)
- CPCV: 15 paths combinatoriais, purge+embargo, custos de transação
- HRP: Lopez de Prado 2016, confidence tilt, clipping, report PDF
- Decision engine: OHLCV → returns → debate → HRP → merge → save

### Fase 4 — Dashboard e Documentação (Sessões 17-20)
- Dashboard Streamlit 4 abas: Benchmark, Performance, War Room, Microstructure
- Streaming per-node com callbacks, replay animado, live debate
- CI/CD (GitHub Actions), Dockerfile, README, ARCHITECTURE.md

### Fase 4b — Benchmark Real (Sessões 21-28)
- Config de 52 tickers, HRP max_weight dinâmico, download paralelo
- Walk-forward backtester: two-cycle (retrain lento + rebalance rápido)
- 16 métricas portfolio vs benchmark, relatório PDF 6 páginas
- Pipeline orquestrador (run_benchmark.py), aba Benchmark no dashboard

### Fase 5 — Melhorias do Benchmark (Sessões 29-35)
- CPCV-OOS: validação de parâmetros com Deflated Sharpe Ratio
- Fixes: log returns fill_null, NaiveModelFactory scaling
- HRP: Ledoit-Wolf shrinkage, Ward linkage
- Volatility targeting (15%, 21d), drawdown killswitch (-15%, benchmark recovery)
- Grid search: 17+ configs × 8 momentum factories
- Bug fix: custos de transação desaparecendo do port_ret
- Bug fix: prob_up discreto → CDF interpolation contínua
- Bug fix: fan chart sort alfabético → sort por nível de quantil

**Status atual:** 720+ testes passando | Fases 1-5 completas

## O que NUNCA fazer
- Nunca hardcode API keys (usar .env + python-dotenv)
- Nunca usar Pandas (usar Polars — é a escolha do projeto)
- Nunca fazer train/test split simples (sempre CPCV)
- Nunca rodar código que acessa APIs reais nos testes (usar mocks)
- Nunca commitar notebooks com output (limpar antes)
