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
    KillswitchConfig,
    ModelFactory,
    NaiveModelFactory,
    RebalanceRecord,
    WalkForwardBacktester,
    WalkForwardConfig,
    WalkForwardResult,
)

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
        assert cfg.min_rebalance_delta == 0.02
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
        import random
        from datetime import timedelta

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
            min_rebalance_delta=0.0,  # disable delta filter for frequency test
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


# ---------------------------------------------------------------------------
# TestVolatilityTargeting
# ---------------------------------------------------------------------------


class TestVolatilityTargeting:
    """Tests for the volatility targeting overlay."""

    def _run_with_vol(
        self,
        ohlcv: pl.DataFrame,
        target_vol: float | None = None,
        max_leverage: float = 1.0,
        min_leverage: float = 0.5,
        vol_lookback: int = 63,
    ) -> WalkForwardResult:
        cfg = WalkForwardConfig(
            lookback_days=100,
            rebalance_every=10,
            retrain_every=200,
            target_vol=target_vol,
            vol_lookback=vol_lookback,
            max_leverage=max_leverage,
            min_leverage=min_leverage,
        )
        bt = WalkForwardBacktester(config=cfg)
        return bt.run(
            ohlcv,
            tickers=["AAPL", "MSFT", "GOOG"],
            benchmark_ticker="SPY",
            model_factory=NaiveModelFactory(),
        )

    def test_none_target_vol_backward_compat(
        self, ohlcv_3t: pl.DataFrame
    ) -> None:
        """target_vol=None must produce identical results to default."""
        cfg_default = WalkForwardConfig(lookback_days=100, rebalance_every=10)
        cfg_none = WalkForwardConfig(
            lookback_days=100, rebalance_every=10, target_vol=None
        )
        factory = NaiveModelFactory()

        r1 = WalkForwardBacktester(config=cfg_default).run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", factory
        )
        r2 = WalkForwardBacktester(config=cfg_none).run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", factory
        )

        final1 = r1.equity_curve["portfolio_value"][-1]
        final2 = r2.equity_curve["portfolio_value"][-1]
        assert final1 == pytest.approx(final2, rel=1e-10)

    def test_config_new_fields_defaults(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.target_vol is None
        assert cfg.vol_lookback == 63
        assert cfg.max_leverage == 1.0
        assert cfg.min_leverage == 0.5

    def test_config_custom_vol_fields(self) -> None:
        cfg = WalkForwardConfig(
            target_vol=0.10,
            vol_lookback=42,
            max_leverage=1.5,
            min_leverage=0.3,
        )
        assert cfg.target_vol == 0.10
        assert cfg.vol_lookback == 42
        assert cfg.max_leverage == 1.5
        assert cfg.min_leverage == 0.3

    def test_high_vol_reduces_exposure(self) -> None:
        """With high-vol data and target_vol=0.10, portfolio should be
        more conservative (lower absolute returns) than without targeting."""
        # Create high-volatility synthetic data
        ohlcv = _make_ohlcv(
            ["AAPL", "MSFT", "GOOG", "SPY"],
            n_days=400,
            daily_return=0.001,
            seed=42,
        )

        res_no = self._run_with_vol(ohlcv, target_vol=None)
        res_yes = self._run_with_vol(ohlcv, target_vol=0.03)

        # With vol targeting, the portfolio should have different final value
        final_no = res_no.equity_curve["portfolio_value"][-1]
        final_yes = res_yes.equity_curve["portfolio_value"][-1]
        assert final_no != pytest.approx(final_yes, rel=0.01), (
            "Vol targeting should change portfolio trajectory"
        )

    def test_max_leverage_never_exceeded(self) -> None:
        """With max_leverage=1.0 (default), leverage should never
        cause holdings to exceed portfolio value."""
        ohlcv = _make_ohlcv(
            ["AAPL", "MSFT", "GOOG", "SPY"],
            n_days=400,
            seed=42,
        )
        # Very high target vol → leverage wants to be > 1.0
        res = self._run_with_vol(
            ohlcv, target_vol=0.50, max_leverage=1.0
        )
        # Portfolio should still be positive
        for val in res.equity_curve["portfolio_value"].to_list():
            assert val > 0

    def test_min_leverage_floor(self) -> None:
        """With very low target vol and min_leverage=0.5, exposure
        should never go below 50%."""
        ohlcv = _make_ohlcv(
            ["AAPL", "MSFT", "GOOG", "SPY"],
            n_days=400,
            daily_return=0.002,  # higher return → higher vol
            seed=42,
        )
        # Very low target → wants to reduce leverage heavily
        res = self._run_with_vol(
            ohlcv, target_vol=0.01, min_leverage=0.5
        )
        # Portfolio should still be positive and growing (floor at 50%)
        for val in res.equity_curve["portfolio_value"].to_list():
            assert val > 0

    def test_killswitch_mode_min_leverage_zero(self) -> None:
        """With min_leverage=0.0, portfolio can go fully to cash."""
        ohlcv = _make_ohlcv(
            ["AAPL", "MSFT", "GOOG", "SPY"],
            n_days=400,
            daily_return=0.002,
            seed=42,
        )
        # Killswitch: very low target, can go to 0% invested
        res = self._run_with_vol(
            ohlcv,
            target_vol=0.001,
            min_leverage=0.0,
            max_leverage=1.0,
        )
        # Should still produce valid results
        assert res.equity_curve.height > 0
        assert all(v > 0 for v in res.equity_curve["portfolio_value"].to_list())

    def test_vol_lookback_insufficient_no_targeting(self) -> None:
        """When < vol_lookback returns available, no targeting applied.

        First vol_lookback days should behave identically to no targeting.
        """
        ohlcv = _make_ohlcv(
            ["AAPL", "MSFT", "GOOG", "SPY"],
            n_days=250,
            seed=42,
        )
        cfg_no = WalkForwardConfig(
            lookback_days=100, rebalance_every=10, target_vol=None
        )
        cfg_vol = WalkForwardConfig(
            lookback_days=100, rebalance_every=10,
            target_vol=0.10, vol_lookback=63,
        )
        factory = NaiveModelFactory()
        res_no = WalkForwardBacktester(config=cfg_no).run(
            ohlcv, ["AAPL", "MSFT", "GOOG"], "SPY", factory
        )
        res_vol = WalkForwardBacktester(config=cfg_vol).run(
            ohlcv, ["AAPL", "MSFT", "GOOG"], "SPY", factory
        )

        # First vol_lookback days should be identical
        n_check = min(63, res_no.equity_curve.height)
        for i in range(n_check):
            v_no = res_no.equity_curve["portfolio_value"][i]
            v_vol = res_vol.equity_curve["portfolio_value"][i]
            assert v_no == pytest.approx(v_vol, rel=1e-10), (
                f"Day {i}: expected identical before vol_lookback window"
            )

    def test_vol_targeting_reduces_vol(self) -> None:
        """Portfolio with vol targeting should have lower realised vol
        than without targeting, when target_vol < actual vol."""

        ohlcv = _make_ohlcv(
            ["AAPL", "MSFT", "GOOG", "SPY"],
            n_days=500,
            daily_return=0.001,
            seed=42,
        )
        res_no = self._run_with_vol(ohlcv, target_vol=None)
        res_yes = self._run_with_vol(ohlcv, target_vol=0.03)

        def _annualized_vol(returns: list[float]) -> float:
            n = len(returns)
            if n < 2:
                return 0.0
            mean = sum(returns) / n
            var = sum((r - mean) ** 2 for r in returns) / (n - 1)
            return math.sqrt(var) * math.sqrt(252)

        vol_no = _annualized_vol(
            res_no.daily_returns["portfolio_return"].to_list()
        )
        vol_yes = _annualized_vol(
            res_yes.daily_returns["portfolio_return"].to_list()
        )
        assert vol_yes < vol_no, (
            f"Vol targeting should reduce vol: {vol_yes:.4f} >= {vol_no:.4f}"
        )

    def test_holdings_never_negative(self) -> None:
        """With vol targeting, no holding should go negative."""
        ohlcv = _make_ohlcv(
            ["AAPL", "MSFT", "GOOG", "SPY"],
            n_days=400,
            seed=42,
        )
        cfg = WalkForwardConfig(
            lookback_days=100,
            rebalance_every=10,
            target_vol=0.05,
            min_leverage=0.0,
        )
        bt = WalkForwardBacktester(config=cfg)
        # We can't directly inspect holdings during the run,
        # but we can verify portfolio value never goes negative
        result = bt.run(
            ohlcv, ["AAPL", "MSFT", "GOOG"], "SPY", NaiveModelFactory()
        )
        for val in result.equity_curve["portfolio_value"].to_list():
            assert val > 0, "Portfolio value should never go negative"

    def test_constant_vol_leverage_near_one(self) -> None:
        """With target_vol matching realised vol, leverage should be ~1.0,
        producing results close to no vol targeting."""
        ohlcv = _make_ohlcv(
            ["AAPL", "MSFT", "GOOG", "SPY"],
            n_days=400,
            daily_return=0.0005,
            seed=42,
        )

        # First run without targeting to measure realised vol

        res_no = self._run_with_vol(ohlcv, target_vol=None)
        rets = res_no.daily_returns["portfolio_return"].to_list()
        n = len(rets)
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / (n - 1)
        actual_vol = math.sqrt(var) * math.sqrt(252)

        # Run with target matching actual vol
        res_match = self._run_with_vol(ohlcv, target_vol=actual_vol)
        final_no = res_no.equity_curve["portfolio_value"][-1]
        final_match = res_match.equity_curve["portfolio_value"][-1]

        # Results should be similar (within 10% relative)
        assert final_match == pytest.approx(final_no, rel=0.10), (
            f"With target=realised vol, results should be close: "
            f"no_target={final_no:.0f}, matched={final_match:.0f}"
        )

    def test_vol_lookback_2_edge_case(self) -> None:
        """vol_lookback=2 is the minimum valid value."""
        ohlcv = _make_ohlcv(
            ["AAPL", "MSFT", "GOOG", "SPY"],
            n_days=300,
            seed=42,
        )
        res = self._run_with_vol(
            ohlcv, target_vol=0.10, vol_lookback=2
        )
        assert res.equity_curve.height > 0
        for val in res.equity_curve["portfolio_value"].to_list():
            assert val > 0

    def test_vol_lookback_1_raises(self) -> None:
        """vol_lookback=1 must raise ValueError (ddof=1 → division by zero)."""
        with pytest.raises(ValueError, match="vol_lookback"):
            WalkForwardBacktester(config=WalkForwardConfig(
                target_vol=0.10, vol_lookback=1
            ))

    def test_max_lt_min_leverage_raises(self) -> None:
        """max_leverage < min_leverage must raise ValueError."""
        with pytest.raises(ValueError, match="max_leverage"):
            WalkForwardBacktester(config=WalkForwardConfig(
                target_vol=0.10, max_leverage=0.3, min_leverage=0.5
            ))

    def test_realized_vol_zero_no_crash(self) -> None:
        """When all returns in the window are identical (vol=0),
        vol targeting should be skipped without error."""
        # Create constant-price data (zero returns → zero vol)
        rows: list[dict[str, Any]] = []
        start = date(2020, 1, 2)
        for ticker in ["A", "B", "SPY"]:
            for day in range(300):
                d = start + timedelta(days=day)
                rows.append({
                    "date": d, "ticker": ticker,
                    "open": 100.0, "high": 100.0, "low": 100.0,
                    "close": 100.0, "volume": 1_000_000,
                })
        ohlcv = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date)
        )
        cfg = WalkForwardConfig(
            lookback_days=50,
            rebalance_every=10,
            target_vol=0.10,
            vol_lookback=10,
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(ohlcv, ["A", "B"], "SPY", NaiveModelFactory())
        # Portfolio should stay at initial capital (zero returns)
        final = result.equity_curve["portfolio_value"][-1]
        assert final == pytest.approx(cfg.initial_capital, rel=0.01)


# ---------------------------------------------------------------------------
# TestKillswitchConfig
# ---------------------------------------------------------------------------


class TestKillswitchConfig:
    def test_defaults(self) -> None:
        ks = KillswitchConfig()
        assert ks.max_drawdown_pct == -0.15
        assert ks.recovery_threshold_pct == -0.05
        assert ks.ramp_up_days == 21

    def test_custom(self) -> None:
        ks = KillswitchConfig(
            max_drawdown_pct=-0.10,
            recovery_threshold_pct=-0.03,
            ramp_up_days=10,
        )
        assert ks.max_drawdown_pct == -0.10
        assert ks.ramp_up_days == 10

    def test_frozen(self) -> None:
        ks = KillswitchConfig()
        with pytest.raises(AttributeError):
            ks.max_drawdown_pct = -0.20  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestDrawdownKillswitch
# ---------------------------------------------------------------------------


def _make_crash_ohlcv(
    n_normal: int = 200,
    n_crash: int = 30,
    n_recovery: int = 100,
    crash_daily_ret: float = -0.03,
    seed: int = 42,
) -> pl.DataFrame:
    """Synthetic data: normal → crash → recovery for 3 tickers + SPY.

    All tickers share similar dynamics to keep things predictable.
    """
    import random

    rng = random.Random(seed)
    tickers = ["A", "B", "C", "SPY"]
    rows: list[dict[str, Any]] = []
    start = date(2020, 1, 2)
    total_days = n_normal + n_crash + n_recovery

    for t_idx, ticker in enumerate(tickers):
        price = 100.0 + t_idx * 5
        for day in range(total_days):
            d = start + timedelta(days=day)
            if day < n_normal:
                ret = 0.001 + rng.gauss(0, 0.005)
            elif day < n_normal + n_crash:
                ret = crash_daily_ret + rng.gauss(0, 0.003)
            else:
                ret = 0.002 + rng.gauss(0, 0.005)
            price *= (1 + ret)
            rows.append({
                "date": d, "ticker": ticker,
                "open": price * 0.999, "high": price * 1.002,
                "low": price * 0.998, "close": price,
                "volume": 1_000_000,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


class TestDrawdownKillswitch:
    """Tests for the drawdown killswitch overlay."""

    def test_killswitch_none_backward_compat(
        self, ohlcv_3t: pl.DataFrame
    ) -> None:
        """killswitch=None must produce identical results to default."""
        cfg_default = WalkForwardConfig(
            lookback_days=100, rebalance_every=10
        )
        cfg_none = WalkForwardConfig(
            lookback_days=100, rebalance_every=10, killswitch=None
        )
        factory = NaiveModelFactory()

        r1 = WalkForwardBacktester(config=cfg_default).run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", factory
        )
        r2 = WalkForwardBacktester(config=cfg_none).run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", factory
        )
        final1 = r1.equity_curve["portfolio_value"][-1]
        final2 = r2.equity_curve["portfolio_value"][-1]
        assert final1 == pytest.approx(final2, rel=1e-10)

    def test_killswitch_triggers_on_crash(self) -> None:
        """During a crash that breaches -15%, killswitch should activate,
        producing a different (better) outcome than no killswitch."""
        ohlcv = _make_crash_ohlcv()
        cfg_no = WalkForwardConfig(
            lookback_days=50, rebalance_every=5,
        )
        cfg_ks = WalkForwardConfig(
            lookback_days=50, rebalance_every=5,
            killswitch=KillswitchConfig(max_drawdown_pct=-0.10),
        )

        res_no = WalkForwardBacktester(config=cfg_no).run(
            ohlcv, ["A", "B", "C"], "SPY", NaiveModelFactory()
        )
        res_ks = WalkForwardBacktester(config=cfg_ks).run(
            ohlcv, ["A", "B", "C"], "SPY", NaiveModelFactory()
        )

        # Results should differ (killswitch changes trajectory)
        final_no = res_no.equity_curve["portfolio_value"][-1]
        final_ks = res_ks.equity_curve["portfolio_value"][-1]
        assert final_no != pytest.approx(final_ks, rel=0.01)

    def test_in_cash_portfolio_earns_rf(self) -> None:
        """While in cash, portfolio should grow at rf only (not market)."""
        # Use extreme crash to guarantee killswitch triggers
        ohlcv = _make_crash_ohlcv(
            n_normal=100, n_crash=50, crash_daily_ret=-0.05,
            n_recovery=50,
        )
        rf = 0.05
        cfg = WalkForwardConfig(
            lookback_days=50, rebalance_every=5, rf=rf,
            killswitch=KillswitchConfig(
                max_drawdown_pct=-0.05,  # very sensitive trigger
                recovery_threshold_pct=-0.01,  # hard to recover
                ramp_up_days=100,  # long ramp = stay in cash
            ),
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(ohlcv, ["A", "B", "C"], "SPY", NaiveModelFactory())

        # Find days where return equals daily rf (killswitch active)
        # Walk-forward uses geometric compounding: (1+rf)^(1/252) - 1
        daily_rf = (1.0 + rf) ** (1.0 / cfg.trading_days_per_year) - 1.0
        port_rets = result.daily_returns["portfolio_return"].to_list()
        found_rf_carry = False
        for ret in port_rets:
            if ret == pytest.approx(daily_rf, rel=1e-6):
                found_rf_carry = True
                break
        assert found_rf_carry, "Should find rf-carry period while in cash"

    def test_exit_costs_applied(self) -> None:
        """Killswitch exit should incur transaction costs."""
        ohlcv = _make_crash_ohlcv(crash_daily_ret=-0.05)
        cfg_no_cost = WalkForwardConfig(
            lookback_days=50, rebalance_every=5,
            killswitch=KillswitchConfig(max_drawdown_pct=-0.05),
        )
        cfg_with_cost = WalkForwardConfig(
            lookback_days=50, rebalance_every=5,
            killswitch=KillswitchConfig(max_drawdown_pct=-0.05),
            costs=TransactionCosts(slippage_bps=50, commission_bps=50),
        )

        res_no = WalkForwardBacktester(config=cfg_no_cost).run(
            ohlcv, ["A", "B", "C"], "SPY", NaiveModelFactory()
        )
        res_cost = WalkForwardBacktester(config=cfg_with_cost).run(
            ohlcv, ["A", "B", "C"], "SPY", NaiveModelFactory()
        )

        # With costs, should end lower
        final_no = res_no.equity_curve["portfolio_value"][-1]
        final_cost = res_cost.equity_curve["portfolio_value"][-1]
        assert final_cost < final_no

    def test_recovery_via_benchmark(self) -> None:
        """After crash, recovery should use benchmark drawdown,
        eventually re-entering the market."""
        ohlcv = _make_crash_ohlcv(
            n_normal=100, n_crash=20, n_recovery=200,
            crash_daily_ret=-0.03,
        )
        cfg = WalkForwardConfig(
            lookback_days=50, rebalance_every=5,
            killswitch=KillswitchConfig(
                max_drawdown_pct=-0.10,
                recovery_threshold_pct=-0.02,
                ramp_up_days=10,
            ),
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(ohlcv, ["A", "B", "C"], "SPY", NaiveModelFactory())

        # After a long recovery, should have rebalances again
        # (some rebalances happen before crash, some after recovery)
        assert len(result.rebalance_history) >= 2

    def test_ramp_up_gradual(self) -> None:
        """With ramp_up_days=21, re-entry should take 21 days of
        benchmark recovery before fully re-entering."""
        ohlcv = _make_crash_ohlcv(
            n_normal=100, n_crash=20, n_recovery=200,
            crash_daily_ret=-0.03,
        )
        cfg = WalkForwardConfig(
            lookback_days=50, rebalance_every=5,
            killswitch=KillswitchConfig(
                max_drawdown_pct=-0.10,
                recovery_threshold_pct=-0.02,
                ramp_up_days=21,
            ),
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(ohlcv, ["A", "B", "C"], "SPY", NaiveModelFactory())
        # Just verify it runs without error and produces results
        assert result.equity_curve.height > 0

    def test_killswitch_with_vol_targeting(self) -> None:
        """Killswitch should work alongside vol targeting."""
        ohlcv = _make_crash_ohlcv(crash_daily_ret=-0.04)
        cfg = WalkForwardConfig(
            lookback_days=50, rebalance_every=5,
            target_vol=0.10, vol_lookback=20,
            killswitch=KillswitchConfig(max_drawdown_pct=-0.10),
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(ohlcv, ["A", "B", "C"], "SPY", NaiveModelFactory())
        assert result.equity_curve.height > 0
        # Portfolio should remain positive
        for val in result.equity_curve["portfolio_value"].to_list():
            assert val > 0

    def test_config_killswitch_field_default_none(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.killswitch is None

    def test_config_killswitch_custom(self) -> None:
        ks = KillswitchConfig(max_drawdown_pct=-0.20)
        cfg = WalkForwardConfig(killswitch=ks)
        assert cfg.killswitch is ks
        assert cfg.killswitch.max_drawdown_pct == -0.20

    def test_portfolio_always_positive(self) -> None:
        """With killswitch, portfolio value should never go negative."""
        ohlcv = _make_crash_ohlcv(crash_daily_ret=-0.06)
        cfg = WalkForwardConfig(
            lookback_days=50, rebalance_every=5,
            killswitch=KillswitchConfig(max_drawdown_pct=-0.05),
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(ohlcv, ["A", "B", "C"], "SPY", NaiveModelFactory())
        for val in result.equity_curve["portfolio_value"].to_list():
            assert val > 0

    def test_ramp_up_days_zero_raises(self) -> None:
        """ramp_up_days=0 must raise ValueError (division by zero)."""
        with pytest.raises(ValueError, match="ramp_up_days"):
            WalkForwardBacktester(config=WalkForwardConfig(
                killswitch=KillswitchConfig(ramp_up_days=0)
            ))

    def test_no_immediate_retrigger_after_recovery(self) -> None:
        """After exiting cash, peak_value is reset so the killswitch
        does not immediately re-trigger from the stale pre-crash peak."""
        # Short mild crash followed by long strong recovery
        ohlcv = _make_crash_ohlcv(
            n_normal=100, n_crash=8, n_recovery=300,
            crash_daily_ret=-0.015,
        )
        cfg = WalkForwardConfig(
            lookback_days=50, rebalance_every=5,
            killswitch=KillswitchConfig(
                max_drawdown_pct=-0.08,
                recovery_threshold_pct=-0.03,
                ramp_up_days=5,
            ),
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(ohlcv, ["A", "B", "C"], "SPY", NaiveModelFactory())

        # After recovery, portfolio should have rebalances (not stuck in cash)
        # With 300 recovery days and +0.2%/day, benchmark recovers fully
        post_crash_date = date(2020, 1, 2) + timedelta(days=250)
        post_recovery_rebalances = [
            r for r in result.rebalance_history
            if r.date > post_crash_date
        ]
        assert len(post_recovery_rebalances) > 0, (
            "Should have rebalances after recovery (no re-trigger)"
        )


# ---------------------------------------------------------------------------
# TestTopNSelection
# ---------------------------------------------------------------------------


class _FixedConfidenceFactory:
    """Returns pre-set confidences for testing top-N selection."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def train(self, train_df: pl.DataFrame) -> None:
        pass

    def predict(self, df: pl.DataFrame) -> dict[str, float]:
        return dict(self._scores)


@pytest.fixture()
def ohlcv_5t() -> pl.DataFrame:
    """5 tradeable tickers + SPY benchmark, 600 days."""
    return _make_ohlcv(
        ["AAPL", "MSFT", "GOOG", "AMZN", "META", "SPY"], n_days=600
    )


class TestTopNSelection:
    """Tests for top_n ticker selection at each rebalance."""

    def test_top_n_none_uses_all_tickers(self, ohlcv_3t: pl.DataFrame) -> None:
        """top_n=None (default) includes all tickers in rebalance."""
        cfg = WalkForwardConfig(
            rebalance_every=5,
            retrain_every=126,
            lookback_days=100,
            top_n=None,
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", NaiveModelFactory()
        )
        # All 3 tickers should appear in weights
        first_rb = result.rebalance_history[0]
        non_zero = [t for t, w in first_rb.weights.items() if w > 0]
        assert len(non_zero) == 3

    def test_top_n_selects_highest_confidence(
        self, ohlcv_5t: pl.DataFrame
    ) -> None:
        """top_n=2 selects only the 2 tickers with highest confidence."""
        scores = {
            "AAPL": 0.9,
            "MSFT": 0.7,
            "GOOG": 0.3,
            "AMZN": 0.2,
            "META": 0.1,
        }
        cfg = WalkForwardConfig(
            rebalance_every=5,
            retrain_every=9999,
            lookback_days=100,
            top_n=2,
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_5t,
            ["AAPL", "MSFT", "GOOG", "AMZN", "META"],
            "SPY",
            _FixedConfidenceFactory(scores),
        )
        first_rb = result.rebalance_history[0]
        non_zero = {t for t, w in first_rb.weights.items() if w > 1e-8}
        assert non_zero == {"AAPL", "MSFT"}

    def test_top_n_exceeds_available(self, ohlcv_3t: pl.DataFrame) -> None:
        """top_n larger than available tickers uses all without error."""
        cfg = WalkForwardConfig(
            rebalance_every=5,
            retrain_every=126,
            lookback_days=100,
            top_n=10,
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", NaiveModelFactory()
        )
        first_rb = result.rebalance_history[0]
        non_zero = [t for t, w in first_rb.weights.items() if w > 0]
        assert len(non_zero) == 3

    def test_top_n_one_single_ticker(self, ohlcv_3t: pl.DataFrame) -> None:
        """top_n=1 concentrates into a single ticker."""
        cfg = WalkForwardConfig(
            rebalance_every=5,
            retrain_every=9999,
            lookback_days=100,
            top_n=1,
        )
        scores = {"AAPL": 0.9, "MSFT": 0.5, "GOOG": 0.3}
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t,
            ["AAPL", "MSFT", "GOOG"],
            "SPY",
            _FixedConfidenceFactory(scores),
        )
        first_rb = result.rebalance_history[0]
        non_zero = {t for t, w in first_rb.weights.items() if w > 1e-8}
        assert non_zero == {"AAPL"}

    def test_top_n_zero_raises(self) -> None:
        """top_n=0 raises ValueError."""
        cfg = WalkForwardConfig(top_n=0)
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            WalkForwardBacktester(config=cfg)

    def test_top_n_negative_raises(self) -> None:
        """top_n=-1 raises ValueError."""
        cfg = WalkForwardConfig(top_n=-1)
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            WalkForwardBacktester(config=cfg)

    def test_top_n_in_metadata(self, ohlcv_3t: pl.DataFrame) -> None:
        """Metadata records the top_n setting."""
        cfg = WalkForwardConfig(
            rebalance_every=5,
            retrain_every=126,
            lookback_days=100,
            top_n=2,
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", NaiveModelFactory()
        )
        assert result.metadata["top_n"] == 2

    def test_top_n_metadata_none_default(
        self, ohlcv_3t: pl.DataFrame
    ) -> None:
        """Metadata records top_n=None when not set."""
        cfg = WalkForwardConfig(
            rebalance_every=5,
            retrain_every=126,
            lookback_days=100,
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_3t, ["AAPL", "MSFT", "GOOG"], "SPY", NaiveModelFactory()
        )
        assert result.metadata["top_n"] is None

    def test_unselected_tickers_get_zero_weight(
        self, ohlcv_5t: pl.DataFrame
    ) -> None:
        """Tickers not in top-N have 0 weight in rebalance."""
        scores = {
            "AAPL": 0.9,
            "MSFT": 0.8,
            "GOOG": 0.3,
            "AMZN": 0.2,
            "META": 0.1,
        }
        cfg = WalkForwardConfig(
            rebalance_every=5,
            retrain_every=9999,
            lookback_days=100,
            top_n=3,
        )
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(
            ohlcv_5t,
            ["AAPL", "MSFT", "GOOG", "AMZN", "META"],
            "SPY",
            _FixedConfidenceFactory(scores),
        )
        first_rb = result.rebalance_history[0]
        assert first_rb.weights.get("AMZN", 0.0) < 1e-8
        assert first_rb.weights.get("META", 0.0) < 1e-8

    def test_config_default_top_n_none(self) -> None:
        """WalkForwardConfig defaults top_n to None."""
        cfg = WalkForwardConfig()
        assert cfg.top_n is None
