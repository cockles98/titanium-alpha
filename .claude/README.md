# `.claude/` -- Development Agents

This directory holds the **Claude Code subagents** that guard quality on the Titanium Alpha codebase. Each one is a purpose-built reviewer or generator that the primary assistant delegates to during development; none of them run at runtime or are shipped to end users. The agents are versioned in git so that anyone who forks the repository inherits the same review pipeline.

The agents were iteratively refined during Phase 2 of the publication plan (session 41) against the specific failure modes we hit during Phases 1-8 -- look-ahead bias, overfitted grid search champions, silent thread-safety regressions in `yfinance`, stale documentation, and missing tests. The rules encoded in each prompt are not generic best-practice platitudes -- they reference concrete incidents and fixes from the session history, and are expected to catch those same issues before they re-enter the code.

All files are in Portuguese (the language of the internal project notes) apart from docstrings and public-facing documentation, which stay in English per decision D2 of `docs/plano_publicacao.md`.

## Agents

| File | Role | Invoked when |
|---|---|---|
| [`agents/architect.md`](agents/architect.md) | Reviews architecture and interface design **before** any new module, class or cross-system integration is written. Guards the "sacred" folder structure (`src/data/`, `src/models/`, `src/agents/`, `src/backtest/`, `src/portfolio/`, `src/dashboard/`, `src/utils/`). Proposes public signatures first, flags unnecessary coupling, verifies existing patterns. Tools: Read, Glob, Grep. Model: Opus 4.6. |
| [`agents/quant-reviewer.md`](agents/quant-reviewer.md) | **Mandatory** review after any change to model logic, backtest, or feature engineering. Checks for look-ahead bias (rolling windows, `shift()`, `fill_null` from future periods), overfitting (Deflated Sharpe Ratio when `n_trials > 1`, annualized -> daily Sharpe conversion), statistical integrity (ddof, geometric `rf` conversion), transaction cost integrity (costs don't silently disappear from `port_ret`). Tools: Read, Bash, Grep. Model: Opus 4.6. |
| [`agents/security-data.md`](agents/security-data.md) | Validates data pipelines after every ingestion/transformation/persistence script. Enforces `yf.Ticker().history()` over the thread-unsafe `yf.download()` (incident: Phase 7, 22/52 tickers received adjacent-ticker data), ensures SPY is added explicitly (it is not in `config/tickers.json`), checks that `.env` never contains real keys in diffs. Validates OHLCV schema, no negative prices, no future-dated rows. Tools: Read, Bash, Glob. Model: Sonnet 4.6. |
| [`agents/test-writer.md`](agents/test-writer.md) | Writes `pytest` tests for new code in `src/`. Rules: tests **never** call real APIs (yfinance, Gemini, Anthropic, NewsAPI) -- always mocked; fixtures return `pl.DataFrame` (never Pandas); target coverage >=80% per module (>=70% for LangGraph-heavy agent modules). Runs the full suite and reports before finishing. Tools: Read, Write, Bash. Model: Sonnet 4.6. |
| [`agents/docs-writer.md`](agents/docs-writer.md) | Generates and updates technical documentation at the end of each module or phase. Enforces the metric hygiene rules (never cite the old Sharpe ~2.7 that was a look-ahead artifact; always report Sharpe + CAGR + MaxDD + Beta together; always compare against SPY buy-and-hold). Language split: README.md / ARCHITECTURE.md / notebooks in English, `docs/` / `CLAUDE.md` / memory files in Portuguese. Tools: Read, Write, Glob. Model: Sonnet 4.6. |

## How they fit into the workflow

```
new feature or fix
        |
        v
  architect  -->  validates structure + interface
        |
        v
  implement code in src/
        |
        v
  quant-reviewer  -->  MANDATORY if financial / model / backtest code changed
        |
        v
  security-data  -->  if data pipeline touched
        |
        v
  test-writer  -->  writes tests, runs suite
        |
        v
  docs-writer  -->  updates README / ARCHITECTURE / module docs
```

None of these agents have write access to production configuration or secrets -- they only read the repo, run the test suite, and write markdown/tests back to the tree. They are safe to invoke from any branch and leave a deterministic paper trail (they write files under `src/`, `tests/`, `docs/`, or `.claude/`; they never touch `.env`, `data/`, or `docker/`).

## Reusing them in your own fork

The prompts are written specifically for Titanium Alpha -- they reference our file layout, our champion config, our historical incidents, and our test count. If you fork the repo for your own quant project, treat them as a starting template rather than a drop-in: keep the structural rules (invoke-before-implement for the architect, mandatory review for any model change, no-real-API in tests) but rewrite the specific numeric anchors (champion Sharpe, baseline metrics, known bugs) to match your own history.
