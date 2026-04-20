# Changelog

All notable changes to Titanium Alpha are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] -- 2026-04-20

First public release. All core components validated end-to-end on 15 years of
market data (2011-2026) and 1002 passing tests.

### Added
- **Walk-forward benchmark** with two-cycle retrain (126 days) + rebalance (15
  days), volatility targeting at 10% annualised, transaction costs (5bps
  slippage + 10bps commission), and 16-metric portfolio vs SPY report.
  10-year OOS champion: Sharpe=0.766, CAGR=13.68%, MaxDD=-21.94%, Beta=0.566,
  Alpha=+2.57%.
- **CPCV-OOS parameter tuning** with Deflated Sharpe Ratio (Bailey & Lopez de
  Prado 2014). 982 configurations tested across three tiers; champion config
  committed in `run_benchmark.py`.
- **PatchTST forecaster** (NeuralForecast) with multi-quantile loss (5
  quantiles), CDF interpolation for continuous `prob_up`, CDF rearrangement
  for monotonic quantiles, and NaN/Inf guards in the predict path.
- **LangGraph 4-agent debate** (Technical Analyst, Fundamentalist, Bear,
  Portfolio Manager) with Pydantic-structured outputs and streaming per-node
  callbacks. Gemini (default, free tier) or Anthropic Claude via
  `LLM_PROVIDER=anthropic`.
- **Financial RAG** with sentence-transformers embeddings (`all-MiniLM-L6-v2`)
  and ChromaDB vector store; P95 retrieval latency = 101ms on 172 articles.
- **HRP allocator** with Ward linkage, Ledoit-Wolf covariance shrinkage,
  sum-preserving confidence tilt, and waterfilling weight optimiser.
- **DecisionEngine** with three-tier weight model (BUY=HRP, HOLD=HRP*conf,
  SELL=0); cash implicit; fallback to `predictions.parquet` if agents fail.
- **Streamlit dashboard** with four tabs: Benchmark, Performance, War Room
  (live + replay modes), Microstructure (fan chart).
- **Data pipeline** with thread-safe `yf.Ticker().history()` ingestion for
  52 US large caps + SPY, 15 years OHLCV (~199k rows), news ingestion with
  Google News RSS fallback.
- **`.claude/agents/`** with 5 development subagents (architect,
  quant-reviewer, security-data, test-writer, docs-writer) encoding project
  rules and historical incidents.
- **Docker Compose** for PostgreSQL + ChromaDB.
- **CI/CD** via GitHub Actions (`ci.yml` + `lint.yml`).
- **Methodology notebook** (`notebooks/methodology_and_results.ipynb`)
  executing end-to-end in 6.6s with 9 sections covering problem framing,
  data, CPCV-OOS, DSR, HRP, walk-forward, regime analysis, the max_weight
  lesson, and limitations.

### Fixed
- **Thread-safety bug in `yf.download()`** (pre-session 37): 22 of 52 tickers
  were receiving data from alphabetically-adjacent tickers due to a yfinance
  thread-safety defect. Migrated to `yf.Ticker().history()`; 15 years of data
  re-ingested.
- **Look-ahead bias in early Sharpe estimates**: the infamous Sharpe ~2.7 from
  sessions 1-35 was an artefact of look-ahead in rolling windows and fill
  strategies. Corrected to the current Sharpe=0.766 with full CPCV-OOS and
  out-of-sample walk-forward.
- **Geometric `rf` conversion** across `benchmark_metrics`, `cpcv`,
  `cpcv_oos`, and `walk_forward` (was arithmetic, biased Sharpe upward).
- **Transaction costs silently dropping** from `port_ret` in walk-forward
  (session 36).
- **`prob_up` discretisation**: replaced step function with CDF interpolation
  across 5 quantiles.
- **Fan chart sort order**: was alphabetical, now sorted by quantile level.
- **Volatility targeting applied ex-ante** (pre-allocation) instead of ex-post
  (post-allocation) to avoid leverage lag.
- **HRP `tilt` sum-preserving**: previously naive multiplication could drift
  the sum of weights away from 1.
- **`walk_forward` initial state**: portfolio starts 100% cash (institutional
  convention) rather than fully invested.
- **Bankruptcy safeguard** in walk-forward (guards against NaV<=0 chains).
- **Dashboard** (`app.py:1396`): `decision.get('weight')` fallback to
  `suggested_weight` for live debate mode (weight key only exists after HRP
  merge in replay mode).
- **Streamlit deprecations**: 17 `use_container_width=True` replaced with
  `width='stretch'` (deadline 2025-12-31).

### Known limitations
- **Backtest-production gap**: the production `make decide` pipeline uses the
  LangGraph debate, which is **not** part of the walk-forward validation (cost
  of ~26k LLM calls per rebalance is infeasible). See
  `docs/design_gap_backtest_vs_production.md`.
- **Stochastic debate**: `make decide` is non-deterministic (temperature=0.2
  analysts + 0.1 PM). `decisions.json` is a single-run snapshot; borderline
  tickers flip between BUY/HOLD in reruns. The `MIN_CONFIDENCE_FOR_ACTION=0.3`
  gate is the deciding boundary.
- **RAG coverage depends on news availability**: with 172 articles across 52
  tickers, grounding coverage is ~47% on tickers with real debate. Meta and
  Tesla sparse in Google News RSS; citations will vary with the backfill.

### Methodological stance
- The project favours **honest reporting over marketing numbers**. MaxDD,
  Beta, Alpha, Tracking Error are published alongside Sharpe; the
  backtest-production gap is documented in the README; the session 40
  regression (max_weight=0.10 reverted at Sharpe=0.462) is preserved as
  evidence of anti-overfitting discipline.
- `max_weight` is treated as a **risk constraint** (`min(6%, 2/N)`) not an
  optimisation parameter; fine-tuning via CPCV-OOS is valid only for timing
  parameters (`rebalance_every`, `target_vol`).

[1.0.0]: https://github.com/cockles98/titanium-alpha/releases/tag/v1.0.0
