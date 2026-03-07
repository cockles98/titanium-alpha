# Titanium Alpha

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Tests 503](https://img.shields.io/badge/Tests-503%20passing-brightgreen?logo=pytest&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![CI](https://img.shields.io/badge/CI-passing-brightgreen?logo=github-actions&logoColor=white)

An **agentic multi-strategy hedge fund system** that uses AI agents to debate investment decisions the way a real trading desk operates. Four specialized agents -- a Technical Analyst, a Fundamentalist, a Devil's Advocate, and a Portfolio Manager -- analyse deep learning forecasts, financial news, and market data, then argue their positions before committing capital. The system validates every strategy with rigorous walk-forward backtesting and allocates risk using the same algorithms employed by institutional quant funds.

---

## Why This Matters

Traditional quantitative trading systems rely on a single model making a single prediction. When that model is wrong, there is no safety net.

Titanium Alpha takes a fundamentally different approach:

- **Multiple perspectives reduce blind spots.** A technical analyst may see a bullish RSI divergence while the bear agent identifies an earnings risk. The portfolio manager weighs both views before deciding -- mimicking how the best hedge fund teams actually operate.

- **Deep learning captures patterns that rules cannot.** PatchTST (a transformer architecture purpose-built for time series) forecasts 5-day returns with quantile uncertainty bands, so the system knows *how confident* its predictions are, not just what they predict.

- **Memory matters.** A RAG pipeline embeds financial news into ChromaDB, giving agents access to recent events -- earnings surprises, macro shifts, sector rotations -- so decisions are grounded in reality, not just price charts.

- **Every strategy is stress-tested before deployment.** Combinatorial Purged Cross-Validation (CPCV) generates 15 non-overlapping backtest paths with embargo periods and transaction costs, eliminating the look-ahead bias and overfitting that plague naive backtests.

- **Risk allocation is mathematically principled.** Hierarchical Risk Parity (Lopez de Prado, 2016) replaces fragile mean-variance optimization with a clustering-based approach that does not require inverting an unstable covariance matrix.

The result is an end-to-end system where every component -- from data ingestion to portfolio allocation -- is production-grade, fully tested, and designed to make better decisions under uncertainty.

---

## Architecture

```mermaid
flowchart TB
    subgraph Data Layer
        YF[yfinance API] -->|OHLCV| PG[(PostgreSQL)]
        RSS[RSS Feeds] -->|News Articles| PG
    end

    subgraph Feature Engineering
        PG -->|OHLCV| FE[Feature Engine<br/>RSI, Bollinger, Vol, VWAP, OBV]
        FE --> PTST[PatchTST<br/>5-day Quantile Forecasts]
        PTST --> PRED[predictions.parquet<br/>forecast.parquet]
    end

    subgraph RAG Pipeline
        PG -->|News| EMB[Sentence Transformers<br/>all-MiniLM-L6-v2]
        EMB --> CHROMA[(ChromaDB)]
        CHROMA -->|Semantic Retrieval| RAG[RAG Context<br/>top-k=5, 30-day window]
    end

    subgraph Agent Debate - LangGraph
        PRED --> LC[Load Context]
        RAG --> RR[RAG Retrieval]
        LC --> RR
        RR --> TA[Technical Analyst<br/>RSI, Bollinger, Volume]
        TA --> FA[Fundamentalist<br/>News, Macro, Earnings]
        FA --> BA[Bear Agent<br/>Devil's Advocate]
        BA --> PM[Portfolio Manager<br/>Final Synthesis]
        PM --> DEC{BUY / HOLD / SELL<br/>+ confidence}
    end

    subgraph Portfolio Allocation
        PG -->|OHLCV| RET[Log Returns<br/>504-day lookback]
        DEC -->|Confidences| HRP[HRP Optimizer<br/>Lopez de Prado 2016]
        RET --> HRP
        HRP --> MERGE[Merge Actions + Weights<br/>Redistribute HOLD/SELL to BUY]
        MERGE --> JSON[decisions.json]
    end

    subgraph Validation
        PTST --> CPCV[CPCV Backtester<br/>15 paths, embargo, costs]
        CPCV --> PDF[Backtest Report PDF<br/>Sharpe, Drawdown, CAGR]
    end

    subgraph Dashboard - Streamlit
        JSON --> TAB1[Performance Tab<br/>Weights, Decisions, Metrics]
        JSON --> TAB2[War Room Tab<br/>Agent Debate Replay]
        PRED --> TAB3[Microstructure Tab<br/>Fan Charts, P-up]
    end
```

---

## Key Features

| Component | Description |
|---|---|
| **PatchTST Forecaster** | Transformer-based 5-day return forecasting with quantile uncertainty (10th, 25th, 50th, 75th, 90th percentiles). Produces P(up) probability per ticker. |
| **Multi-Agent Debate** | Four Claude Sonnet agents with distinct personas debate each ticker through a LangGraph pipeline. Structured output via Pydantic ensures machine-readable decisions. |
| **Financial RAG** | News articles embedded with sentence-transformers, stored in ChromaDB, retrieved by semantic similarity with date-aware reranking. Agents cite sources -- never hallucinate news. |
| **CPCV Backtesting** | 15 combinatorial paths with purge windows (64 days), embargo periods (10 days), and configurable transaction costs (slippage + commission + market impact). |
| **HRP Allocation** | Hierarchical Risk Parity with confidence tilt from agent debate. Single/complete/average/ward linkage. Automatic weight clipping and renormalization. |
| **Streamlit Dashboard** | Three-tab interface: portfolio performance (donut + bar charts), war room (agent debate replay with chat bubbles), and microstructure (fan charts with confidence intervals). |
| **Feature Engineering** | RSI, Bollinger Bands, realized volatility, VWAP, OBV, relative volume -- all implemented in Polars with zero look-ahead bias (verified by quant reviewer). |
| **Transaction Cost Model** | Slippage, commission, and liquidity-aware market impact (`1/sqrt(relative_volume)`) applied per position change in backtests. |

---

## Quick Start

```bash
# 1. Clone and install dependencies
git clone https://github.com/your-username/titanium-alpha.git
cd titanium-alpha && poetry install --no-root

# 2. Start PostgreSQL and ChromaDB
docker compose -f docker/docker-compose.yml up -d

# 3. Ingest market data and news
make ingest

# 4. Run the full decision pipeline (predictions + debate + HRP)
make predict && make decide

# 5. Launch the dashboard
make run
```

> **Prerequisites:** Python 3.10+, [Poetry](https://python-poetry.org/), Docker, and an `.env` file with database credentials and API keys. See `.env.example` for required variables.

---

## Project Structure

```
titanium-alpha/
|-- src/
|   |-- data/               Data ingestion (yfinance OHLCV + RSS/NewsAPI)
|   |-- models/             PatchTST forecaster, feature engineering, prediction pipeline
|   |-- agents/             LangGraph debate graph, personas, state management, RAG
|   |-- backtest/           CPCV backtester, transaction costs, PDF report generator
|   |-- portfolio/          HRP optimizer, decision engine (final pipeline)
|   |-- dashboard/          Streamlit app (3 tabs: Performance, War Room, Microstructure)
|   |-- utils/              Database connections (PostgreSQL, ChromaDB)
|-- tests/                  503 tests (pytest), fixtures in conftest.py
|-- docker/                 docker-compose.yml (PostgreSQL 15 + ChromaDB)
|-- docs/                   Backtest metrics reference, research notes
|-- notebooks/              Exploration only (never imported by src/)
|-- data/outputs/           Pipeline artifacts (predictions, forecasts, decisions)
|-- models/checkpoints/     Saved PatchTST model weights
|-- Makefile                setup, ingest, predict, decide, test, lint, run, clean
|-- pyproject.toml          Poetry config, ruff, mypy, pytest settings
```

---

## How It Works

The `DecisionEngine` orchestrates the entire pipeline in seven steps:

```python
from src.portfolio.decision_engine import DecisionEngine

engine = DecisionEngine()
output = engine.run()

# output.decisions -> per-ticker BUY/HOLD/SELL with HRP weights
# output.hrp_final_weights -> {"SPY": 0.28, "NVDA": 0.22, "AAPL": 0.25, "QQQ": 0.25}
# output.metadata -> schema version, HRP config, number of observations
```

**Pipeline steps:**

1. **Load OHLCV** from PostgreSQL (SPY, NVDA, AAPL, QQQ -- 5 years of daily data)
2. **Compute log returns** in wide format, trimmed to a 504-day (~2 year) lookback window
3. **Run agent debate** -- four Claude agents analyse PatchTST forecasts and RAG-retrieved news, producing BUY/HOLD/SELL with confidence scores per ticker
4. **Extract confidences** from the debate (missing tickers default to 0.5 = neutral)
5. **Run HRP** with confidence tilt (agents shift allocation by up to 20%)
6. **Merge actions and weights** -- HOLD/SELL tickers get weight 0, redistributed proportionally to BUY tickers
7. **Save** `decisions.json` and `debate_history.json` for dashboard consumption

Graceful degradation is built in: if the agent debate fails (no API key, network error), the pipeline falls back to pure HRP allocation without confidence tilt.

---

## Testing

The test suite covers every module with 503 tests running in under 20 seconds:

```bash
make test
# or directly:
poetry run pytest tests/ -v --tb=short
```

**Test coverage highlights:**

| Area | Tests | What is validated |
|---|---|---|
| Data ingestion | 53 | Download, schema, retry, upsert, dedup, NaN handling |
| Feature engineering | 30 | RSI, Bollinger, volatility, volume, no look-ahead bias |
| PatchTST model | 31 | Init, prepare, build, fit, predict, save/load |
| Prediction pipeline | 12 | Load, metrics (MAE/RMSE), Parquet roundtrip |
| Agent state + personas | 58 | TypedDicts, Pydantic models, validation, registry |
| LangGraph graph | 36 | Node execution, RAG integration, full pipeline |
| RAG (ChromaDB) | 40 | Embedding, retrieval, reranking, edge cases |
| CPCV backtest | 94 | Splits, purge, embargo, Sharpe, drawdown, costs |
| Backtest report | 15 | PDF generation, plots, edge cases |
| HRP optimizer | 67 | Covariance, clustering, bisection, tilt, clipping |
| Decision engine | 31 | Returns, merge, debate, JSON output, integration |
| Dashboard | 21 | Loaders, chart builders, agent styles, streaming/replay |
| DB utilities | 15 | Connection pooling, env vars, overrides |

All tests use mocks for external dependencies (APIs, databases, LLMs). No real API calls are made during testing.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Type hints, modern syntax |
| DataFrames | Polars | Fast columnar processing (no Pandas) |
| Deep Learning | NeuralForecast (Nixtla) | PatchTST with quantile loss |
| Agents | LangGraph + LangChain | Multi-agent orchestration |
| LLM | Claude Sonnet (Anthropic) | Structured output for agent reasoning |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 for news embedding |
| Vector Store | ChromaDB | Semantic search over financial news |
| Database | PostgreSQL 15 | OHLCV + news persistence |
| Portfolio | scipy + numpy | HRP clustering and optimization |
| Backtesting | Custom CPCV | Combinatorial purged cross-validation |
| Dashboard | Streamlit + Plotly | Interactive 3-tab interface |
| Logging | loguru | Structured logging (never print()) |
| Config | python-dotenv | Environment-based configuration |
| Packaging | Poetry | Dependency management |
| Containers | Docker Compose | PostgreSQL + ChromaDB services |
| Linting | ruff + mypy | Style enforcement + strict type checking |
| Testing | pytest + pytest-mock | 503 tests, all mocked |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
