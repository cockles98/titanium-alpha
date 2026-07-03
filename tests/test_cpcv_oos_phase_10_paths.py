"""Phase 10 additions to cpcv_oos — per-path equity collection and persistence."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.backtest.cpcv_oos import (
    CPCVParameterValidator,
    ValidationResult,
    save_cpcv_paths,
)
from src.backtest.walk_forward import (
    NaiveModelFactory,
    WalkForwardConfig,
    WalkForwardResult,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv() -> pl.DataFrame:
    start = date(2022, 1, 1)
    rows: list[dict] = []
    for t in ("AAA", "BBB", "SPY"):
        for i in range(220):
            rows.append({
                "date": start + timedelta(days=i),
                "ticker": t,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 + i * 0.05,
                "volume": 1_000_000,
            })
    return pl.DataFrame(rows)


@pytest.fixture
def tickers() -> list[str]:
    return ["AAA", "BBB"]


def _mock_wf_result(
    dates: list[date], returns: list[float]
) -> WalkForwardResult:
    return WalkForwardResult(
        equity_curve=pl.DataFrame({
            "date": dates,
            "portfolio_value": [100.0 * (1.0 + 0.001 * i) for i in range(len(dates))],
            "benchmark_value": [100.0] * len(dates),
        }),
        daily_returns=pl.DataFrame({
            "date": dates,
            "portfolio_return": returns,
            "benchmark_return": [0.0] * len(dates),
        }),
        rebalance_history=[],
    )


# ---------------------------------------------------------------------------
# _evaluate_path_with_equity returns a 3-tuple
# ---------------------------------------------------------------------------


class TestEvaluatePathWithEquity:
    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_returns_tuple_of_three(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=4, n_test_groups=1
        )
        all_dates = validator._trading_dates
        mock_bt_cls.return_value.run.return_value = _mock_wf_result(
            all_dates, [0.001] * len(all_dates)
        )
        config = WalkForwardConfig(lookback_days=50)
        sharpe, rets, eq = validator._evaluate_path_with_equity(
            0, (1,), config, NaiveModelFactory()
        )
        assert isinstance(sharpe, float)
        assert isinstance(rets, list)
        assert isinstance(eq, pl.DataFrame)
        assert set(eq.columns) == {"date", "equity"}

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_equity_base_normalised_to_one(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=4, n_test_groups=1
        )
        all_dates = validator._trading_dates
        mock_bt_cls.return_value.run.return_value = _mock_wf_result(
            all_dates, [0.001] * len(all_dates)
        )
        config = WalkForwardConfig(lookback_days=50)
        _s, _r, eq = validator._evaluate_path_with_equity(
            0, (1,), config, NaiveModelFactory()
        )
        assert eq is not None
        assert eq.height > 0
        assert abs(float(eq["equity"][0]) - 1.0) < 1e-12

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_backtester_failure_returns_none_equity(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=4, n_test_groups=1
        )
        mock_bt_cls.return_value.run.side_effect = RuntimeError("boom")
        config = WalkForwardConfig(lookback_days=50)
        sharpe, rets, eq = validator._evaluate_path_with_equity(
            0, (0,), config, NaiveModelFactory()
        )
        assert sharpe == 0.0
        assert rets == []
        assert eq is None


# ---------------------------------------------------------------------------
# validate(collect_equity=True) populates per_path_equity
# ---------------------------------------------------------------------------


class TestValidateCollectEquity:
    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_collect_false_leaves_per_path_equity_none(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=3, n_test_groups=1
        )
        all_dates = validator._trading_dates
        mock_bt_cls.return_value.run.return_value = _mock_wf_result(
            all_dates, [0.001] * len(all_dates)
        )
        result = validator.validate(
            WalkForwardConfig(lookback_days=50),
            model_factory=NaiveModelFactory(),
        )
        assert result.per_path_equity is None

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_collect_true_stores_one_frame_per_path(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=3, n_test_groups=1
        )
        all_dates = validator._trading_dates
        mock_bt_cls.return_value.run.return_value = _mock_wf_result(
            all_dates, [0.001] * len(all_dates)
        )
        result = validator.validate(
            WalkForwardConfig(lookback_days=50),
            model_factory=NaiveModelFactory(),
            collect_equity=True,
        )
        assert result.per_path_equity is not None
        assert len(result.per_path_equity) == len(result.per_path_sharpe)


# ---------------------------------------------------------------------------
# ValidationResult.to_paths_frame + save_cpcv_paths
# ---------------------------------------------------------------------------


def _make_validation_result_with_equity(
    n_paths: int = 3, n_days: int = 10
) -> ValidationResult:
    start = date(2023, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    per_path_equity: list[pl.DataFrame | None] = []
    per_path_sharpe: list[float] = []
    for pid in range(n_paths):
        eq_values = [1.0 + 0.001 * i + 0.0005 * pid for i in range(n_days)]
        per_path_equity.append(pl.DataFrame({"date": dates, "equity": eq_values}))
        per_path_sharpe.append(0.5 + 0.1 * pid)

    return ValidationResult(
        config=WalkForwardConfig(lookback_days=50),
        mean_sharpe=sum(per_path_sharpe) / n_paths,
        std_sharpe=0.1,
        pct_positive=1.0,
        per_path_sharpe=per_path_sharpe,
        deflated_sharpe=0.5,
        p_value=0.5,
        accepted=True,
        per_path_equity=per_path_equity,
    )


def test_to_paths_frame_produces_long_format():
    result = _make_validation_result_with_equity(n_paths=3, n_days=10)
    frame = result.to_paths_frame(config_name="testcfg")
    assert set(frame.columns) == {"config", "path_id", "date", "equity", "sharpe"}
    assert frame["config"].unique().to_list() == ["testcfg"]
    assert frame["path_id"].n_unique() == 3
    # Each path contributes 10 rows → 30 rows total.
    assert frame.height == 30


def test_to_paths_frame_preserves_per_path_sharpe():
    result = _make_validation_result_with_equity(n_paths=3, n_days=5)
    frame = result.to_paths_frame()
    per_path = (
        frame.group_by("path_id")
        .agg(pl.col("sharpe").first())
        .sort("path_id")
    )
    assert per_path["sharpe"].to_list() == pytest.approx(
        result.per_path_sharpe
    )


def test_to_paths_frame_empty_when_no_equity():
    result = _make_validation_result_with_equity(n_paths=2, n_days=5)
    result.per_path_equity = None
    frame = result.to_paths_frame()
    assert frame.height == 0
    assert set(frame.columns) == {"config", "path_id", "date", "equity", "sharpe"}


def test_save_cpcv_paths_writes_expected_schema(tmp_path: Path):
    result = _make_validation_result_with_equity(n_paths=3, n_days=6)
    out = tmp_path / "cpcv_paths.parquet"
    path = save_cpcv_paths(result, out, config_name="champion")
    assert path.exists()
    reloaded = pl.read_parquet(path)
    assert set(reloaded.columns) == {"config", "path_id", "date", "equity", "sharpe"}
    assert reloaded.height == 18  # 3 paths × 6 days
    assert reloaded["config"].unique().to_list() == ["champion"]


def test_save_cpcv_paths_raises_without_equity():
    result = _make_validation_result_with_equity(n_paths=1, n_days=5)
    result.per_path_equity = None
    with pytest.raises(ValueError, match="collect_equity"):
        save_cpcv_paths(result, Path("ignored.parquet"))
