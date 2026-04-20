# Contributing to Titanium Alpha

Thank you for considering a contribution. Titanium Alpha is a research-oriented
project and the bar for new code is deliberately high: we care about
statistical integrity and look-ahead bias far more than feature count. Please
read this guide before opening a PR.

## Getting set up

```bash
git clone https://github.com/cockles98/titanium-alpha.git
cd titanium-alpha
poetry install --with dev
cp .env.example .env     # fill in GEMINI_KEY + optional NEWSAPI_KEY
docker compose -f docker/docker-compose.yml up -d
poetry run pytest        # must stay at 1002+ passing
```

## Development workflow

1. Branch from `master` with a descriptive name (`feat/<topic>`,
   `fix/<bug>`, `refactor/<scope>`, `test/<scope>`, `docs/<scope>`).
2. Use the Claude Code subagents in `.claude/agents/` when making changes:
   - **architect** before any new module or cross-system integration.
   - **quant-reviewer** after any change to model, backtest, or feature
     engineering code. This is mandatory for financial logic.
   - **security-data** after touching any ingestion / transformation script.
   - **test-writer** after any new function or class in `src/`.
   - **docs-writer** after a module or phase completes.
3. Keep commits small and semantic. Hooks and CI run `ruff` + `mypy` + `pytest`.
4. Open a PR. The description should state (a) what changed, (b) why, and (c)
   any metric impact (Sharpe, CAGR, MaxDD, Beta).

## Commit message format

```
<type>: <short imperative sentence>

<body, if needed — explain "why" not "what">
```

`<type>` is one of `feat`, `fix`, `refactor`, `test`, `docs`, `perf`, `chore`.

## Code style rules

- **Type hints on every public method.** `mypy --strict` is enforced.
- **Docstrings on every public function / class** (Google style).
- **No Pandas.** The project uses **Polars** end-to-end. Fixtures and
  intermediate frames must be `pl.DataFrame`.
- **No `print()` in production code.** Use `loguru` (`from loguru import
  logger`).
- **Never swallow exceptions.** Re-raise with context if you catch.
- **No bare `except:`**. Catch specific exception classes.
- **No real API calls in tests.** Mock `yfinance`, Gemini, Anthropic,
  NewsAPI. Tests run on CI without network access.
- **No look-ahead bias.** Rolling windows must use closed="left" or
  equivalent shifting; `fill_null` must not propagate future values
  backward.

## Sacred folder structure

Do not rearrange or cross these boundaries without discussion:

```
src/data/          ingestion + persistence
src/models/        PatchTST + features
src/agents/        LangGraph + RAG
src/backtest/      CPCV, CPCV-OOS, walk-forward, benchmark metrics / report
src/portfolio/     HRP + decision engine
src/dashboard/     Streamlit app
src/utils/         shared helpers
tests/             pytest
notebooks/         exploration only (never imported by src/)
docs/              documentation (Portuguese)
docker/            Dockerfiles + compose
config/            tickers.json
```

Import firewalls:
- `src/agents` must not import from `src/portfolio` (and vice versa).
- `src/backtest` must not import from `src/agents` (the backtest-production
  gap is a deliberate design choice documented in
  `docs/design_gap_backtest_vs_production.md`).

## Testing expectations

- **Coverage floor: 80% per module** (70% for LangGraph-heavy agent code).
- **Every new function** gets at least a happy-path test and one edge-case
  test (empty input, NaN, single-row DataFrame).
- **No flaky tests.** If a test depends on time or randomness, seed it.
- Run the full suite before pushing:
  ```bash
  poetry run pytest --cov=src --cov-report=term-missing
  ```

## What *not* to PR

- Further CPCV-OOS parameter sweeps. The parameter space has been saturated
  (982 configs). New tuning requests should come with a hypothesis, not a
  grid search.
- Integration of the LangGraph debate into the walk-forward loop. The
  token-cost gap is a documented design decision, not a bug.
- Additional tickers beyond the current 52. More tickers dilute signal.
- Pandas refactors. The project uses Polars by choice.
- Backwards-compatibility shims or feature flags around internal APIs. Rename
  freely and update callers.

## Reporting issues

For bugs, open a GitHub issue with (a) a minimal reproduction, (b) the
expected vs actual behaviour, (c) the output of `poetry env info` and `git
rev-parse HEAD`. For methodology questions, open a **Discussion** rather than
an issue.

## License

By contributing, you agree that your contribution will be licensed under the
MIT License (see `LICENSE`).
