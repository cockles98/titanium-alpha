# Titanium Alpha

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Tests 720+](https://img.shields.io/badge/Tests-720%2B%20passing-brightgreen?logo=pytest&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![CI](https://img.shields.io/badge/CI-passing-brightgreen?logo=github-actions&logoColor=white)

An **agentic multi-strategy hedge fund system** that uses AI agents to debate investment decisions the way a real trading desk operates. Four specialized agents -- a Technical Analyst, a Fundamentalist, a Devil's Advocate, and a Portfolio Manager -- analyse deep learning forecasts, financial news, and market data, then argue their positions before committing capital. The system validates every strategy through **CPCV-OOS parameter optimization with Deflated Sharpe Ratio**, a **walk-forward backtest across 52 S&P 500 constituents** with daily rebalancing, volatility targeting, and drawdown killswitch -- then allocates risk using **Hierarchical Risk Parity with Ledoit-Wolf shrinkage**.

---

## Why This Matters

Traditional quantitative trading systems rely on a single model making a single prediction. When that model is wrong, there is no safety net.

Titanium Alpha takes a fundamentally different approach:

- **Multiple perspectives reduce blind spots.** A technical analyst may see a bullish RSI divergence while the bear agent identifies an earnings risk. The portfolio manager weighs both views before deciding -- mimicking how the best hedge fund teams actually operate.

- **Deep learning captures patterns that rules cannot.** PatchTST (a transformer architecture purpose-built for time series) forecasts 5-day returns with quantile uncertainty bands. CDF interpolation produces continuous P(up) probabilities per ticker rather than crude discrete counts.

- **Memory matters.** A RAG pipeline embeds financial news into ChromaDB, giving agents access to recent events -- earnings surprises, macro shifts, sector rotations -- so decisions are grounded in reality, not just price charts.

- **Every strategy parameter is validated before deployment.** CPCV-OOS (Combinatorial Purged Cross-Validation Out-of-Sample) with **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014) tests 17+ configurations and 8 momentum factories across 15 non-overlapping paths. Only parameters with `pct_positive >= 66.7%` and `DSR p-value > 0.95` are accepted.

- **Walk-forward benchmark with risk overlays.** A full temporal simulation across 52 US large-cap stocks with **daily rebalancing**, semi-annual PatchTST retraining, **volatility targeting** (15% annualized, 21-day lookback), and **drawdown killswitch** (-15% threshold, benchmark-based recovery). Transaction costs (slippage + commission) applied on every position change. 16 portfolio-vs-benchmark metrics computed automatically.

- **Risk allocation is mathematically principled.** Hierarchical Risk Parity (Lopez de Prado, 2016) with **Ledoit-Wolf covariance shrinkage**, confidence tilt from model signals, and dynamic weight caps (`min(15%, 2/N)`) that scale automatically with the number of assets.

The result is an end-to-end system where every component -- from data ingestion to portfolio allocation -- is production-grade, fully tested, and designed to make better decisions under uncertainty.

---

## Key Results (CPCV-OOS Validated)

| Metric | Value | Notes |
|---|---|---|
| **Sharpe Ratio** | ~2.7 | Annualized, out-of-sample validated |
| **CAGR** | ~45% | Compound annual growth rate |
| **Max Drawdown** | ~-20% | Peak-to-trough |
| **Volatility Targeting** | 15% annualized | Crushes tail kurtosis from ~26 to ~9.4 |
| **Cost Tolerance** | Up to ~30 bps | Sharpe OOS > 1.5 at 30 bps total costs |
| **Universe** | 52 US large caps | S&P 500 constituents across 9 sectors |

> **Note:** These results reflect the PatchTST signal alone. The multi-agent debate layer has not yet been backtested independently. See [docs/design_gap_backtest_vs_production.md](docs/design_gap_backtest_vs_production.md) for details.

---

## Architecture

```mermaid
flowchart TB
    subgraph Data Layer
        YF[yfinance API<br/>52 tickers + SPY] -->|OHLCV parallel| PG[(PostgreSQL)]
        RSS[RSS Feeds] -->|News Articles| PG
        CFG[config/tickers.json<br/>52 US large caps] -.->|ticker list| YF
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
        PG -->|OHLCV| RET[Log Returns<br/>63-day lookback]
        DEC -->|Confidences| HRP[HRP Optimizer<br/>Ledoit-Wolf + Ward]
        RET --> HRP
        HRP --> MERGE[Merge Actions + Weights<br/>Redistribute HOLD/SELL to BUY]
        MERGE --> JSON[decisions.json]
    end

    subgraph CPCV-OOS Validation
        PTST --> CPCV[CPCV Backtester<br/>15 paths, embargo, costs]
        CPCV --> DSR[Deflated Sharpe Ratio<br/>Bailey & Lopez de Prado 2014]
        DSR --> GRID[Grid Search<br/>17 configs x 8 factories]
        GRID --> VALID[Validated Parameters]
    end

    subgraph Walk-Forward Benchmark
        PG -->|OHLCV 52 tickers| WF[Walk-Forward Backtester<br/>daily rebalance, vol targeting, killswitch]
        VALID -.->|optimised config| WF
        WF -->|equity curve| BM[Benchmark Metrics<br/>16 metrics vs SPY]
        BM --> PDF2[Benchmark Report PDF<br/>6 pages]
        WF -->|weights history| PQT[benchmark_equity.parquet<br/>benchmark_weights.parquet<br/>benchmark_metrics.json]
    end

    subgraph Dashboard - Streamlit
        PQT --> TAB0[Benchmark Tab<br/>Equity, Drawdown, Rolling Sharpe, Heatmap]
        JSON --> TAB1[Performance Tab<br/>Weights, Decisions, Metrics]
        JSON --> TAB2[War Room Tab<br/>Agent Debate Replay]
        PRED --> TAB3[Microstructure Tab<br/>Fan Charts, CDF P-up]
    end
```

---

## Key Features

| Component | Description |
|---|---|
| **PatchTST Forecaster** | Transformer-based 5-day return forecasting with quantile uncertainty (10th, 25th, 50th, 75th, 90th percentiles). CDF interpolation produces continuous P(up) probability per ticker. Supports both old and new NeuralForecast column naming conventions. |
| **Multi-Agent Debate** | Four Claude Sonnet agents with distinct personas debate each ticker through a LangGraph pipeline. Structured output via Pydantic ensures machine-readable decisions. Live streaming mode with per-node callbacks. |
| **Financial RAG** | News articles embedded with sentence-transformers, stored in ChromaDB, retrieved by semantic similarity with date-aware reranking. Agents cite sources -- never hallucinate news. |
| **CPCV-OOS Validation** | Grid search across 17+ configurations and 8 momentum factories. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) with empirical skewness/kurtosis. Acceptance criteria: `pct_positive >= 66.7%` and `DSR p-value > 0.95`. |
| **Walk-Forward Benchmark** | Full temporal simulation across 52 US large-cap stocks with **daily rebalancing** (1 day), semi-annual PatchTST retraining (126 days), **volatility targeting** (15% annualized, 21-day lookback), and **drawdown killswitch** (-15% threshold, benchmark-based recovery). Per-asset dollar tracking for weight drift. Compared against SPY buy-and-hold. |
| **Benchmark Metrics** | 16 portfolio-vs-benchmark metrics: CAGR, Sharpe, Sortino, Information Ratio, Jensen's alpha (CAPM OLS), beta, max drawdown, max drawdown duration, Calmar ratio, tracking error, monthly hit rate, avg turnover, and more. |
| **HRP Allocation** | Hierarchical Risk Parity with **Ledoit-Wolf covariance shrinkage**, confidence tilt from model signals (cap=1.0), single/complete/average/ward linkage. Dynamic weight caps (`min(15%, 2/N)`) that scale with the number of assets. |
| **Transaction Cost Model** | Slippage (5 bps), commission (10 bps), and liquidity-aware market impact (`1/sqrt(relative_volume)`) applied per position change. Turnover threshold (`min_rebalance_delta=0.01`) to skip dust rebalances. |
| **Streamlit Dashboard** | Four-tab interface: benchmark (equity curve, drawdown, rolling Sharpe, weight heatmap, CPCV-OOS results), portfolio performance (donut + bar charts), war room (agent debate replay with live streaming), and microstructure (fan charts with quantile bands and last-close reference line). |
| **Feature Engineering** | RSI, Bollinger Bands, realized volatility, VWAP, OBV, relative volume -- all implemented in Polars with zero look-ahead bias (verified by quant reviewer). |

---

## Quick Start

```bash
# 1. Clone and install dependencies
git clone https://github.com/your-username/titanium-alpha.git
cd titanium-alpha && poetry install --no-root

# 2. Start PostgreSQL and ChromaDB
docker compose -f docker/docker-compose.yml up -d

# 3. Configure environment variables
cp .env.example .env  # then fill in API keys

# 4. Ingest market data (52 US tickers, parallel download)
make ingest

# 5. Run PatchTST predictions
make predict

# 6. Run walk-forward benchmark vs S&P 500
make benchmark          # PatchTST model (production)
make benchmark-fast     # NaiveModelFactory (quick validation)

# 7. Validate parameters with CPCV-OOS
make validate           # Full grid search (17 configs x 8 factories)
make validate-fast      # Tier 1 only (7 configs)

# 8. Run agent debate + portfolio allocation (requires ANTHROPIC_API_KEY)
make decide

# 9. Launch the dashboard
make run
```

> **Prerequisites:** Python 3.10+, [Poetry](https://python-poetry.org/), Docker, and an `.env` file with database credentials and API keys. See `.env.example` for required variables.

---

## Project Structure

```
titanium-alpha/
|-- src/
|   |-- config.py           Ticker configuration loader (52 US + SPY benchmark)
|   |-- data/               Data ingestion (yfinance OHLCV + RSS/NewsAPI, parallel download)
|   |-- models/             PatchTST forecaster, feature engineering, prediction pipeline
|   |-- agents/             LangGraph debate graph, personas, state management, RAG
|   |-- backtest/           CPCV, CPCV-OOS validator, walk-forward backtester, benchmark metrics
|   |-- portfolio/          HRP optimizer (Ledoit-Wolf, ward linkage), decision engine
|   |-- dashboard/          Streamlit app (4 tabs: Benchmark, Performance, War Room, Microstructure)
|   |-- utils/              Database connections (PostgreSQL, ChromaDB)
|-- config/                 tickers.json (52 US large caps, SPY benchmark)
|-- tests/                  720+ tests (pytest), fixtures in conftest.py
|-- docker/                 docker-compose.yml (PostgreSQL 15 + ChromaDB)
|-- docs/                   Architecture, backtest metrics, design gap analysis, research notes
|-- notebooks/              Exploration only (never imported by src/)
|-- data/outputs/           Pipeline artifacts (predictions, decisions, equity curves, reports)
|-- models/checkpoints/     Saved PatchTST model weights
|-- Makefile                setup, ingest, predict, decide, benchmark, validate, test, lint, run
|-- pyproject.toml          Poetry config, ruff, mypy, pytest settings
```

---

## How It Works

### Decision Pipeline

The `DecisionEngine` orchestrates the live decision pipeline in seven steps:

```python
from src.portfolio.decision_engine import DecisionEngine

engine = DecisionEngine()
output = engine.run()

# output.decisions -> per-ticker BUY/HOLD/SELL with HRP weights
# output.hrp_final_weights -> {"AAPL": 0.04, "NVDA": 0.03, ...}  (52 tickers)
# output.metadata -> schema version, HRP config, number of observations
```

**Pipeline steps:**

1. **Load OHLCV** from PostgreSQL (52 US large caps from `config/tickers.json`)
2. **Compute log returns** in wide format, trimmed to a 63-day lookback window
3. **Run agent debate** -- four Claude agents analyse PatchTST forecasts and RAG-retrieved news, producing BUY/HOLD/SELL with confidence scores per ticker
4. **Extract confidences** from the debate (missing tickers default to 0.5 = neutral)
5. **Run HRP** with Ledoit-Wolf shrinkage and confidence tilt (cap=1.0), dynamic `max_weight = min(15%, 2/N)`
6. **Merge actions and weights** -- HOLD/SELL tickers get weight 0, redistributed proportionally to BUY tickers
7. **Save** `decisions.json` and `debate_history.json` for dashboard consumption

Graceful degradation is built in: if the agent debate fails (no API key, network error), the pipeline falls back to pure HRP allocation with uniform confidence (0.5). See [design gap analysis](docs/design_gap_backtest_vs_production.md) for limitations of this fallback.

### Walk-Forward Benchmark

The `WalkForwardBacktester` simulates the strategy historically against SPY buy-and-hold:

```python
from src.backtest.run_benchmark import run_us_benchmark

result = run_us_benchmark(use_patchtst=True)  # or --naive for quick validation

# result.equity_curve -> daily portfolio_value vs benchmark_value
# result.metrics -> 16 metrics (Sharpe, Sortino, alpha, beta, max DD, ...)
# result.rebalance_history -> every rebalance with weights, turnover, costs
```

**Benchmark configuration (CPCV-OOS validated):**

| Parameter | Value | Rationale |
|---|---|---|
| Universe | 52 US large caps + SPY | S&P 500 constituents across 9 sectors |
| Rebalance | Daily (1 trading day) | Ultra-short-term alpha capture |
| Retrain PatchTST | Semi-annual (126 trading days) | Separates slow/fast cycles |
| Lookback | 63 days (~3 months) | Short covariance window captures regime changes |
| Costs | 5 bps slippage + 10 bps commission | Conservative for US large caps |
| Vol targeting | 15% annualized, 21-day lookback | Crushes tail kurtosis from ~26 to ~9.4 |
| Drawdown killswitch | -15% trigger, benchmark recovery | Liquidates to cash, re-enters after benchmark stabilises |
| HRP | Ledoit-Wolf shrinkage, confidence tilt cap=1.0, max_weight=15% | Regularised covariance, strong signal amplification |
| Min rebalance delta | 1% turnover | Skip dust rebalances |
| Capital | $1,000,000 | Institutional standard |

Outputs: `benchmark_equity.parquet`, `benchmark_metrics.json`, `benchmark_weights.parquet`, and a 6-page PDF report (equity curve, drawdown, metrics table, rolling Sharpe, weight heatmap, turnover chart).

### CPCV-OOS Parameter Validation

Before deploying any configuration, parameters are validated through Combinatorial Purged Cross-Validation:

```python
from src.backtest.run_validation import run_improvement_validation

results = run_improvement_validation(subset="all")  # tier1 + tier2 + tier3 + tier4
# results -> validation_results.json, validation_summary.md
```

The validator:
1. Divides the OOS period into 6 temporal blocks
2. Generates C(6,2) = 15 combinatorial paths with purge + embargo
3. Runs the walk-forward backtester on each path independently
4. Computes Deflated Sharpe Ratio (corrects for multiple testing)
5. Accepts only configurations with `pct_positive >= 66.7%` AND `DSR p-value > 0.95`

---

## Testing

The test suite covers every module with 720+ tests running in under 65 seconds:

```bash
make test
# or directly:
poetry run pytest tests/ -v --tb=short
```

**Test coverage highlights:**

| Area | Tests | What is validated |
|---|---|---|
| Data ingestion | 61 | Download, schema, retry, upsert, parallel, partial failure, date range |
| Feature engineering | 30 | RSI, Bollinger, volatility, volume, no look-ahead bias |
| PatchTST model | 43 | Init, prepare, build, fit, predict, CDF interpolation, NF naming conventions |
| Prediction pipeline | 12 | Load, metrics (MAE/RMSE), Parquet roundtrip |
| Ticker config | 13 | Config loading, fallback, validation, real config |
| Agent state + personas | 58 | TypedDicts, Pydantic models, validation, registry |
| LangGraph graph | 36 | Node execution, RAG integration, full pipeline |
| RAG (ChromaDB) | 40 | Embedding, retrieval, reranking, edge cases |
| CPCV backtest | 94 | Splits, purge, embargo, Sharpe, drawdown, costs |
| CPCV-OOS validator | 66 | DSR math, purged factory, grid search, integration |
| CPCV report | 15 | PDF generation, plots, edge cases |
| Walk-forward backtest | 74 | Config, returns, costs, drift, vol targeting, killswitch, look-ahead bias |
| Benchmark metrics | 40 | Sharpe, Sortino, alpha, beta, drawdown, hit rate, edge cases |
| Benchmark report | 19 | PDF 6-page generation, helpers, edge cases |
| Run benchmark | 15 | Filter, model factory, save, e2e integration |
| Run validation | 43 | Config builders, output savers, HRP integration, pipeline mocked |
| HRP optimizer | 80 | Covariance, clustering, bisection, tilt, Ledoit-Wolf shrinkage, ward linkage |
| Decision engine | 34 | Returns, merge, debate, dynamic max_weight, JSON output |
| Dashboard | 47 | Loaders, charts, agent styles, streaming, benchmark tab, fan chart sort |
| DB utilities | 15 | Connection pooling, env vars, overrides |

All tests use mocks for external dependencies (APIs, databases, LLMs). No real API calls are made during testing.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Type hints, modern syntax |
| DataFrames | Polars | Fast columnar processing (no Pandas) |
| Deep Learning | NeuralForecast (Nixtla) | PatchTST with MQLoss (5 quantiles) |
| Agents | LangGraph + LangChain | Multi-agent orchestration |
| LLM | Claude Sonnet (Anthropic) | Structured output for agent reasoning |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 for news embedding |
| Vector Store | ChromaDB | Semantic search over financial news |
| Database | PostgreSQL 15 | OHLCV + news persistence |
| Portfolio | scipy + numpy + scikit-learn | HRP clustering, Ledoit-Wolf shrinkage |
| Backtesting | Custom CPCV + CPCV-OOS + Walk-Forward | Cross-validation + parameter validation + temporal benchmark |
| Reporting | matplotlib + seaborn | 6-page benchmark PDF (equity, drawdown, metrics, heatmap) |
| Dashboard | Streamlit + Plotly | Interactive 4-tab interface |
| Logging | loguru | Structured logging (never print()) |
| Config | python-dotenv + JSON | Environment variables + ticker configuration |
| Packaging | Poetry | Dependency management |
| Containers | Docker Compose | PostgreSQL + ChromaDB services |
| Linting | ruff + mypy | Style enforcement + strict type checking |
| CI | GitHub Actions | Automated test + lint on push/PR |
| Testing | pytest | 720+ tests, all mocked |

---

## Known Limitations

- **Backtest-production gap:** The validated Sharpe (~2.7) reflects PatchTST signal alone. The multi-agent debate pipeline has not been backtested independently. See [design gap analysis](docs/design_gap_backtest_vs_production.md).
- **Agent fallback:** When LangGraph agents are unavailable (no API key), the decision engine defaults to uniform BUY with confidence 0.5 -- discarding PatchTST signals entirely.
- **Cost sensitivity:** Strategy degrades above ~30 bps total costs (Sharpe drops below 1.5 at 50 bps).
- **No short selling:** The system only goes long or flat (no short positions).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
