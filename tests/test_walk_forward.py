"""Tests for ``src.backtest.walk_forward``.

Uses synthetic OHLCV data (3 tickers + benchmark, 600 days) to validate
the walk-forward loop without any external dependencies.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import polars as pl
import pytest

from src.backtest.cpcv import TransactionCosts
from src.backtest.walk_forward import (
    ModelFactory,
    NaiveModelFactory,
    RebalanceRecord,
    WalkForwardBacktester,
    WalkForwardConfig,
    WalkForwardResult,
)
from src.portfolio.hrp import HRPConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv(
    tickers: list[str],
    n_days: int = 600,
    start: date = date(2020, 1, 2),
    base_price: float = 100.0,
    daily_return: float = 0.0005,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic OHLCV data for testing.

    Each ticker gets a drifting price series with pseudo-random noise
    to create realistic variance and non-trivial HRP allocations.
    """
    import random

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for t_idx, ticker in enumerate(tickers):
        price = base_price + t_idx * 10
        for day in range(n_days):
            d = start + timedelta(days=day)
            # Drift + noise (different volatility per ticker)
            vol = 0.005 + t_idx * 0.003
            noise = rng.gauss(0, vol)
            ret = daily_return + (t_idx - 1) * 0.0001 + noise
            price *= (1 + ret)
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": price * 0.999,
                "high": price * 1.002,
                "low": price * 0.998,
                "close": price,
                "volume": 1_000_000 + t_idx * 100_000,
            })
    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Date)
    )


@pytest.fixture()
def ohlcv_3t() -> pl.DataFrame:
    """3 tradeable tickers + SPY benchmark, 600 days."""
    return _make_ohlcv(["AAPL", "MSFT", "GOOG", "SPY"], n_days=600)


@pytest.fixture()
def ohlcv_1t() -> pl.DataFrame:
    """Single ticker + SPY benchmark, 600 days."""
    return _make_ohlcv(["AAPL", "SPY"], n_days=600)


@pytest.fixture()
def short_ohlcv() -> pl.DataFrame:
    """Too short for default warmup (504 days)."""
    return _make_ohlcv(["AAPL", "SPY"], n_days=100)


# ---------------------------------------------------------------------------
# TestWalkForwardConfig
# ---------------------------------------------------------------------------


class TestWalkForwardConfig:
    def test_defaults(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.retrain_every == 126
        assert cfg.rebalance_every == 5
        assert cfg.lookback_days == 504
        assert cfg.initial_capital == 1_000_000.0
        assert cfg.costs is None
        assert cfg.min_rebalance_delta == 0.0
        assert cfg.trading_days_per_year == 252
        assert cfg.rf == 0.05

    def test_custom(self) -> None:
        costs = TransactionCosts(slippage_bps=10, commission_bps=20)
        cfg = WalkForwardConfig(
            retrain_every=63,
            rebalance_every=21,
            costs=costs,
            initial_capital=500_000.0,
        )
        assert cfg.retrain_every == 63
        assert cfg.rebalance_every == 21
        assert cfg.costs is costs
        assert cfg.initial_capital == 500_000.0

    def test_frozen(self) -> None:
        cfg = WalkForwardConfig()
        with pytest.raises(AttributeError):
            cfg.retrain_every = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestRebalanceRecord
# ---------------------------------------------------------------------------


class TestRebalanceRecord:
    def test_creation(self) -> None:
        rec = RebalanceRecord(
            date=date(2023, 6, 1),
            weights={"AAPL": 0.5, "MSFT": 0.5},
            turnover=0.3,
            costs=150.0,
            retrained=True,
        )
        assert rec.date == date(2023, 6, 1)
        assert rec.weights == {"AAPL": 0.5, "MSFT": 0.5}
        assert rec.turnover == 0.3
        assert rec.costs == 150.0
        assert rec.retrained is True


# ---------------------------------------------------------------------------
# TestWalkForwardResult
# ---------------------------------------------------------------------------


class TestWalkForwardResult:
    def test_defaults(self) -> None:
        res = WalkForwardResult(
            equity_curve=pl.DataFrame(),
            daily_returns=pl.DataFrame(),
            rebalance_history=[],
        )
        assert res.metrics == {}
        assert isinstance(res.config, WalkForwardConfig)
        assert res.metadata == {}


# ---------------------------------------------------------------------------
# TestNaiveModelFactory
# ---------------------------------------------------------------------------


class TestNaiveModelFactory:
    def test_implements_protocol(self) -> None:
        factory = NaiveModelFactory()
        assert isinstance(factory, ModelFactory)

    def test_train_is_noop(self) -> None:
        factory = NaiveModelFactory()
        df = _make_ohlcv(["AAPL"], n_days=10)
        factory.train(df)  # Should not raise

    def test_predict_returns_confidences(self) -> None:
        factory = NaiveModelFactory(lookback=5)
        df = _make_ohlcv(["AAPL", "MSFT"], n_days=20)
        scores = factory.predict(df)
        assert "AAPL" in scores
        assert "MSFT" in scores
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_predict_positive_momentum_above_half(self) -> None:
        """Upward-drifting series should produce confidence > 0.5."""
        factory = NaiveModelFactory(lookback=5)
        df = _make_ohlcv(["AAPL"], n_days=20, daily_return=0.01)
        scores = factory.predict(df)
        assert scores["AAPL"] > 0.5

    def test_predict_single_row_returns_neutral(self) -> None:
        factory = NaiveModelFactory()
        df = _make_ohlcv(["AAPL"], n_days=1)
        scores = factory.predict(df)
        assert scores["AAPL"] == 0.5

    def test_scaling_backward_compat_lookback5(self) -> None:
        """lookback=5 → scaling=10.0, same as the original hardcoded value."""
        factory = NaiveModelFactory(lookback=5)
        df = _make_ohlcv(["AAPL"], n_days=20, daily_return=0.01)
        scores = factory.predict(df)
        # With positive momentum, conf should be > 0.5
        assert scores["AAPL"] > 0.5

    def test_scaling_lookback63_not_saturated(self) -> None:
        """lookback=63 with moderate return should NOT saturate to 0.95."""
        factory = NaiveModelFactory(lookback=63)
        # Moderate daily return → ~4% over 63 days
        df = _make_ohlcv(["AAPL"], n_days=100, daily_return=0.0006)
        scores = factory.predict(df)
        conf = scores["AAPL"]
        # Should be nuanced, not saturated
        assert 0.5 < conf < 0.95, f"Expected nuanced confidence, got {conf}"

    def test_scaling_lookback1(self) -> None:
        """Edge case: lookback=1 → scaling=50.0, but still clamped."""
        factory = NaiveModelFactory(lookback=1)
        df = _make_ohlcv(["AAPL"], n_days=10, daily_return=0.01)
        scores = factory.predict(df)
        assert 0.05 <= scores["AAPL"] <= 0.95

    def test_scaling_produces_range(self) -> None:
        """With lookback=63, different returns should produce varied confidences."""
        factory = NaiveModelFactory(lookback=63)
        # Strong positive
        df_up = _make_ohlcv(["AAPL"], n_days=100, daily_return=0.002)
        # Negative
        df_dn = _make_ohlcv(["AAPL"], n_days=100, daily_return=-0.002)
        conf_up = factory.predict(df_up)["AAPL"]
        conf_dn = factory.predict(df_dn)["AAPL"]
        assert conf_up > 0.5
        assert conf_dn < 0.5
        # Both should be within bounds (not saturated for moderate returns)
        assert conf_up < 0.95 and conf_dn > 0.05  # both unsaturated


# ---------------------------------------------------------------------------
# TestComputeDailyReturns
# ---------------------------------------------------------------------------


class TestComputeDailyReturns:
    def test_returns_wide_format(self, ohlcv_3t: pl.DataFrame) -> None:
        bt = WalkForwardBacktester()
        wide = bt._compute_daily_returns(ohlcv_3t, ["AAPL", "MSFT"])
        assert "date" in wide.columns
        assert "AAPL" in wide.columns
        assert "MSFT" in wide.columns
        # No ticker column
        assert "ticker" not in wide.columns

    def test_returns_no_nulls(self, ohlcv_3t: pl.DataFrame) -> None:
        bt = WalkForwardBacktester()
        wide = bt._compute_daily_returns(ohlcv_3t, ["AAPL"])
        assert wide.null_count().sum_horizontal()[0] == 0


# ---------------------------------------------------------------------------
# TestComputeLogReturnsForHRP
# ---------------------------------------------------------------------------


class TestComputeLogReturnsForHRP:
    def test_respects_max_date(self, ohlcv_3t: pl.DataFrame) -> None:
        bt = WalkForwardBacktester()
        cutoff = date(2020, 7, 1)
        lr = bt._compute_log_returns_for_hrp(
            ohlcv_3t, ["AAPL", "MSFT"], cutoff, lookback=50
        )
        # Should have at most 50 rows
        assert lr.height <= 50
        # Should only have ticker columns (no date)
        assert "date" not in lr.columns

    def test_respects_lookback(self, ohlcv_3t: pl.DataFrame) -> None:
        bt = WalkForwardBacktester()
        cutoff = date(2021, 6, 1)
        lr = bt._compute_log_returns_for_hrp(
            ohlcv_3t, ["AAPL"], cutoff, lookback=30
        )
        assert lr.height <= 30

    def test_fill_null_preserves_rows_with_partial_data(self) -> None:
        """When one ticker has missing dates, rows should NOT be dropped."""
        # AAPL has 100 days, MSFT has only 50 (starting later)
        from datetime import timedelta
        import random

        rng = random.Random(42)
        rows: list[dict[str, Any]] = []
        base = date(2020, 1, 1)

        # AAPL: 100 days
        price = 100.0
        for d in range(100):
            price *= 1 + rng.gauss(0.0003, 0.01)
            rows.append({
                "date": base + timedelta(days=d),
                "ticker": "AAPL",
                "open": price, "high": price, "low": price,
                "close": price, "volume": 1_000_000,
            })

        # MSFT: only days 50-99 (50 days)
        price = 200.0
        for d in range(50, 100):
            price *= 1 + rng.gauss(0.0003, 0.01)
            rows.append({
                "date": base + timedelta(days=d),
                "ticker": "MSFT",
                "open": price, "high": price, "low": price,
                "close": price, "volume": 1_000_000,
            })

        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
        bt = WalkForwardBacktester()
        lr = bt._compute_log_returns_for_hrp(
            df, ["AAPL", "MSFT"], base + timedelta(days=99), lookback=200
        )
        # With the improved logic, leading rows where MSFT is null are
        # dropped (to avoid variance deflation), so we get ~49 rows
        # (MSFT's coverage).  This is better than the old drop_nulls()
        # which could drop ALL rows with any interior null.
        assert lr.height > 0, f"Expected some rows, got {lr.height}"
        # No nulls
        assert lr.null_count().sum_horizontal()[0] == 0

    def test_fill_null_no_nulls_in_output(self, ohlcv_3t: pl.DataFrame) -> None:
        """Log returns output should have zero null values."""
        bt = WalkForwardBacktester()
        cutoff = date(2021, 6, 1)
        lr = bt._compute_log_returns_for_hrp(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], cutoff, lookback=200
        )
        assert lr.null_count().sum_horizontal()[0] == 0


# ---------------------------------------------------------------------------
# TestApplyCosts
# ---------------------------------------------------------------------------


class TestApplyCosts:
    def test_no_costs_returns_zero(self) -> None:
        old = {"A": 0.5, "B": 0.5}
        new = {"A": 0.3, "B": 0.7}
        turnover, cost = WalkForwardBacktester._apply_costs(
            old, new, 1_000_000, None
        )
        assert turnover == pytest.approx(0.4)  # |0.3-0.5| + |0.7-0.5|
        assert cost == 0.0

    def test_with_costs(self) -> None:
        old = {"A": 0.5, "B": 0.5}
        new = {"A": 0.3, "B": 0.7}
        costs = TransactionCosts(slippage_bps=5, commission_bps=10)
        turnover, cost = WalkForwardBacktester._apply_costs(
            old, new, 1_000_000, costs
        )
        assert turnover == pytest.approx(0.4)
        # cost = 0.4 * 1M * (5+10)/10000 = 0.4 * 1M * 0.0015 = 600
        assert cost == pytest.approx(600.0)

    def test_no_change_no_cost(self) -> None:
        w = {"A": 0.5, "B": 0.5}
        costs = TransactionCosts(slippage_bps=5, commission_bps=10)
        turnover, cost = WalkForwardBacktester._apply_costs(w, w, 1_000_000, costs)
        assert turnover == 0.0
        assert cost == 0.0

    def test_new_ticker_added(self) -> None:
        old = {"A": 1.0}
        new = {"A": 0.5, "B": 0.5}
        turnover, cost = WalkForwardBacktester._apply_costs(
            old, new, 100_000, None
        )
        assert turnover == pytest.approx(1.0)  # |0.5-1| + |0.5-0|


# ---------------------------------------------------------------------------
# TestWalkForwardRun — core integration tests
# ---------------------------------------------------------------------------


class TestWalkForwardRun:
    def test_basic_run_produces_result(self, ohlcv_3t: pl.DataFrame) -> None:
        cfg = WalkForwardConfig(
            lookback_days=100,
            retrain_every=50,
            rebalance_every=10,
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t,
            tickers=["AAPL", "MSFT", "GOOG"],
            benchmark_ticker="SPY",
            model_factory=NaiveModelFactory(),
        )

        assert isinstance(result, WalkForwardResult)
        assert result.equity_curve.height > 0
        assert result.daily_returns.height > 0
        assert len(result.rebalance_history) > 0

    def test_equity_curve_has_correct_columns(
        self, ohlcv_3t: pl.DataFrame
    ) -> None:
        cfg = WalkForwardConfig(lookback_days=100, rebalance_every=10)
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t,
            tickers=["AAPL", "MSFT", "GOOG"],
            benchmark_ticker="SPY",
        )
        assert set(result.equity_curve.columns) == {
            "date", "portfolio_value", "benchmark_value",
        }

    def test_daily_returns_has_correct_columns(
        self, ohlcv_3t: pl.DataFrame
    ) -> None:
        cfg = WalkForwardConfig(lookback_days=100, rebalance_every=10)
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t,
            tickers=["AAPL", "MSFT", "GOOG"],
            benchmark_ticker="SPY",
        )
        assert set(result.daily_returns.columns) == {
            "date", "portfolio_return", "benchmark_return",
        }

    def test_portfolio_starts_at_initial_capital(
        self, ohlcv_3t: pl.DataFrame
    ) -> None:
        capital = 500_000.0
        cfg = WalkForwardConfig(lookback_days=100, initial_capital=capital)
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t,
            tickers=["AAPL", "MSFT", "GOOG"],
            benchmark_ticker="SPY",
        )
        first_val = result.equity_curve["portfolio_value"][0]
        # First day has a return applied, so it won't be exactly capital,
        # but the benchmark should start near capital too
        first_bench = result.equity_curve["benchmark_value"][0]
        # Both should be close to initial capital (within one day's move)
        assert abs(first_val / capital - 1) < 0.05
        assert abs(first_bench / capital - 1) < 0.05

    def test_rebalance_frequency(self, ohlcv_3t: pl.DataFrame) -> None:
        cfg = WalkForwardConfig(
            lookback_days=100,
            rebalance_every=20,
            retrain_every=200,
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t,
            tickers=["AAPL", "MSFT", "GOOG"],
            benchmark_ticker="SPY",
        )
        n_active_days = result.equity_curve.height
        expected_rebalances = n_active_days // 20 + 1  # +1 for initial
        actual = len(result.rebalance_history)
        # Allow ±2 tolerance (boundary effects)
        assert abs(actual - expected_rebalances) <= 2

    def test_retrain_happens(self, ohlcv_3t: pl.DataFrame) -> None:
        """At least one retrain should occur (initial + periodic)."""
        cfg = WalkForwardConfig(
            lookback_days=100,
            rebalance_every=10,
            retrain_every=50,
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t,
            tickers=["AAPL", "MSFT", "GOOG"],
            benchmark_ticker="SPY",
        )
        n_retrains = sum(
            1 for r in result.rebalance_history if r.retrained
        )
        assert n_retrains >= 1
        # Retrain should happen less than every rebalance
        assert n_retrains < len(result.rebalance_history)

    def test_single_ticker(self, ohlcv_1t: pl.DataFrame) -> None:
        cfg = WalkForwardConfig(lookback_days=100, rebalance_every=10)
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_1t,
            tickers=["AAPL"],
            benchmark_ticker="SPY",
        )
        assert result.equity_curve.height > 0
        # Single ticker: weight should always be 1.0
        for rec in result.rebalance_history:
            assert rec.weights.get("AAPL", 0) == pytest.approx(1.0, abs=0.01)

    def test_metadata_populated(self, ohlcv_3t: pl.DataFrame) -> None:
        cfg = WalkForwardConfig(lookback_days=100)
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t,
            tickers=["AAPL", "MSFT", "GOOG"],
            benchmark_ticker="SPY",
        )
        assert result.metadata["n_tickers"] == 3
        assert result.metadata["benchmark_ticker"] == "SPY"
        assert result.metadata["n_trading_days"] > 0
        assert result.metadata["n_rebalances"] > 0

    def test_default_model_factory(self, ohlcv_3t: pl.DataFrame) -> None:
        """When model_factory is None, uses NaiveModelFactory."""
        cfg = WalkForwardConfig(lookback_days=100, rebalance_every=50)
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t,
            tickers=["AAPL", "MSFT", "GOOG"],
            benchmark_ticker="SPY",
            model_factory=None,
        )
        assert result.equity_curve.height > 0


# ---------------------------------------------------------------------------
# TestCosts
# ---------------------------------------------------------------------------


class TestCosts:
    def test_costs_reduce_equity(self, ohlcv_3t: pl.DataFrame) -> None:
        """Portfolio with costs should end lower than without."""
        cfg_no_cost = WalkForwardConfig(
            lookback_days=100, rebalance_every=10, costs=None
        )
        cfg_with_cost = WalkForwardConfig(
            lookback_days=100,
            rebalance_every=10,
            costs=TransactionCosts(slippage_bps=50, commission_bps=50),
        )
        factory = NaiveModelFactory()

        bt_no = WalkForwardBacktester(config=cfg_no_cost)
        bt_yes = WalkForwardBacktester(config=cfg_with_cost)

        res_no = bt_no.run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", factory
        )
        res_yes = bt_yes.run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", factory
        )

        final_no = res_no.equity_curve["portfolio_value"][-1]
        final_yes = res_yes.equity_curve["portfolio_value"][-1]
        assert final_yes < final_no

    def test_rebalance_records_have_costs(
        self, ohlcv_3t: pl.DataFrame
    ) -> None:
        cfg = WalkForwardConfig(
            lookback_days=100,
            rebalance_every=10,
            costs=TransactionCosts(slippage_bps=5, commission_bps=10),
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", NaiveModelFactory()
        )
        total_costs = sum(r.costs for r in result.rebalance_history)
        assert total_costs > 0


# ---------------------------------------------------------------------------
# TestMinRebalanceDelta
# ---------------------------------------------------------------------------


class TestMinRebalanceDelta:
    def test_high_threshold_fewer_rebalances(
        self, ohlcv_3t: pl.DataFrame
    ) -> None:
        cfg_always = WalkForwardConfig(
            lookback_days=100,
            rebalance_every=10,
            min_rebalance_delta=0.0,
        )
        cfg_threshold = WalkForwardConfig(
            lookback_days=100,
            rebalance_every=10,
            min_rebalance_delta=2.0,  # Impossibly high → no rebalances
        )
        factory = NaiveModelFactory()
        bt = WalkForwardBacktester

        res_always = bt(config=cfg_always).run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", factory
        )
        res_thresh = bt(config=cfg_threshold).run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", factory
        )
        assert len(res_thresh.rebalance_history) < len(
            res_always.rebalance_history
        )


# ---------------------------------------------------------------------------
# TestLookAheadBias
# ---------------------------------------------------------------------------


class TestLookAheadBias:
    def test_future_data_does_not_affect_early_weights(self) -> None:
        """Adding future data should not change weights computed on
        earlier dates — proving no look-ahead."""
        tickers = ["A", "B", "SPY"]

        # Scenario 1: 400 days of data
        ohlcv_short = _make_ohlcv(tickers, n_days=400)

        # Scenario 2: same 400 days + 200 extra future days with
        # wildly different returns
        ohlcv_long = _make_ohlcv(tickers, n_days=400)
        future_rows: list[dict[str, Any]] = []
        for t in tickers:
            for d in range(400, 600):
                dt = date(2020, 1, 2) + timedelta(days=d)
                future_rows.append({
                    "date": dt,
                    "ticker": t,
                    "open": 999.0,
                    "high": 999.0,
                    "low": 999.0,
                    "close": 999.0,
                    "volume": 999_999,
                })
        future_df = pl.DataFrame(future_rows).with_columns(
            pl.col("date").cast(pl.Date)
        )
        ohlcv_long = pl.concat([ohlcv_long, future_df])

        cfg = WalkForwardConfig(
            lookback_days=100,
            rebalance_every=20,
            retrain_every=100,
        )
        factory = NaiveModelFactory()

        res_short = WalkForwardBacktester(config=cfg).run(
            ohlcv_short, ["A", "B"], "SPY", factory
        )
        res_long = WalkForwardBacktester(config=cfg).run(
            ohlcv_long, ["A", "B"], "SPY", factory
        )

        # Compare weights from first few rebalances (within original 400 days)
        n_compare = min(5, len(res_short.rebalance_history))
        for i in range(n_compare):
            ws = res_short.rebalance_history[i].weights
            wl = res_long.rebalance_history[i].weights
            for ticker in ["A", "B"]:
                assert ws.get(ticker, 0) == pytest.approx(
                    wl.get(ticker, 0), abs=1e-10
                ), (
                    f"Look-ahead detected at rebalance {i} for {ticker}: "
                    f"short={ws.get(ticker)} != long={wl.get(ticker)}"
                )


# ---------------------------------------------------------------------------
# TestBenchmark
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_benchmark_is_buy_and_hold(self, ohlcv_3t: pl.DataFrame) -> None:
        """Benchmark equity should equal SPY cumulative return × capital."""
        cfg = WalkForwardConfig(lookback_days=100)
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", NaiveModelFactory()
        )

        # Verify benchmark = buy-and-hold SPY
        bench_returns = result.daily_returns["benchmark_return"]
        # Manually compound
        expected = cfg.initial_capital
        for r in bench_returns.to_list():
            expected *= (1 + r)

        actual = result.equity_curve["benchmark_value"][-1]
        assert actual == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_insufficient_data_raises(self, short_ohlcv: pl.DataFrame) -> None:
        cfg = WalkForwardConfig(lookback_days=504)  # needs > 504 days
        bt = WalkForwardBacktester(config=cfg)
        with pytest.raises(ValueError, match="warmup"):
            bt.run(short_ohlcv, ["AAPL"], "SPY")

    def test_missing_benchmark_raises(self, ohlcv_3t: pl.DataFrame) -> None:
        cfg = WalkForwardConfig(lookback_days=100)
        bt = WalkForwardBacktester(config=cfg)
        with pytest.raises(ValueError, match="Benchmark ticker"):
            bt.run(ohlcv_3t, ["AAPL"], "NONEXISTENT")

    def test_no_tradeable_tickers_raises(
        self, ohlcv_3t: pl.DataFrame
    ) -> None:
        cfg = WalkForwardConfig(lookback_days=100)
        bt = WalkForwardBacktester(config=cfg)
        with pytest.raises(ValueError, match="No tradeable tickers"):
            bt.run(ohlcv_3t, ["FAKE1", "FAKE2"], "SPY")


# ---------------------------------------------------------------------------
# TestWeightsSumToOne
# ---------------------------------------------------------------------------


class TestWeightsSumToOne:
    def test_all_rebalance_weights_sum_to_one(
        self, ohlcv_3t: pl.DataFrame
    ) -> None:
        cfg = WalkForwardConfig(lookback_days=100, rebalance_every=10)
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", NaiveModelFactory()
        )
        for rec in result.rebalance_history:
            total = sum(rec.weights.values())
            assert total == pytest.approx(1.0, abs=0.01), (
                f"Weights don't sum to 1 on {rec.date}: {rec.weights} "
                f"(sum={total})"
            )


# ---------------------------------------------------------------------------
# TestConfigPreserved
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestWeightDrift
# ---------------------------------------------------------------------------


class TestWeightDrift:
    def test_weights_drift_between_rebalances(self) -> None:
        """Between rebalances, effective weights should drift with prices.

        Asset A rises ~2%/day, B is flat.  With drift-adjusted holdings
        A's effective weight grows over time.  We verify the portfolio
        value changes (holdings are tracked per-asset, not constant-mix).
        """
        import random

        rng = random.Random(42)
        rows: list[dict[str, Any]] = []
        start = date(2020, 1, 2)
        price_a = 100.0
        price_b = 100.0
        price_spy = 100.0
        for day in range(250):
            d = start + timedelta(days=day)
            # Add noise so HRP sees real variance and allocates to both
            price_a *= 1.02 + rng.gauss(0, 0.005)
            price_b *= 1.00 + rng.gauss(0, 0.005)
            price_spy *= 1.001 + rng.gauss(0, 0.003)
            for ticker, price in [("A", price_a), ("B", price_b), ("SPY", price_spy)]:
                rows.append({
                    "date": d, "ticker": ticker,
                    "open": price, "high": price * 1.001, "low": price * 0.999,
                    "close": price, "volume": 1_000_000,
                })
        ohlcv = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))

        # Long rebalance interval so drift accumulates
        cfg = WalkForwardConfig(
            lookback_days=50,
            rebalance_every=200,  # effectively no rebalance after initial
            retrain_every=200,
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(ohlcv, ["A", "B"], "SPY", NaiveModelFactory())

        # Only 1 rebalance (initial) should exist
        assert len(result.rebalance_history) == 1

        # Verify the portfolio value changed (drift was applied, not stuck)
        final_port = result.equity_curve["portfolio_value"][-1]
        assert final_port != cfg.initial_capital, "Portfolio value should change"
        # A rises ~2%/day so portfolio should grow significantly
        assert final_port > cfg.initial_capital * 2, "Portfolio should grow with A"

        # Verify daily returns were recorded
        assert result.daily_returns.height > 0, "Should have daily returns"


# ---------------------------------------------------------------------------
# TestConfigPreserved
# ---------------------------------------------------------------------------


class TestConfigPreserved:
    def test_result_contains_config(self, ohlcv_3t: pl.DataFrame) -> None:
        cfg = WalkForwardConfig(
            lookback_days=100, rebalance_every=15, rf=0.03
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", NaiveModelFactory()
        )
        assert result.config is cfg
        assert result.config.rf == 0.03
        assert result.config.rebalance_every == 15
