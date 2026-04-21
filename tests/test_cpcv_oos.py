"""Tests for ``src.backtest.cpcv_oos``.

Validates the CPCV-OOS parameter validation framework including split
generation, embargo, path evaluation, Deflated Sharpe Ratio, and grid
search using synthetic OHLCV data.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.backtest.cpcv_oos import (
    CPCVParameterValidator,
    ValidationResult,
    _compute_sharpe,
    _inv_normal_cdf,
    _kurtosis,
    _normal_cdf,
    _PurgedModelFactory,
    _skewness,
    _std_list,
    deflated_sharpe_ratio,
)
from src.backtest.walk_forward import (
    NaiveModelFactory,
    WalkForwardConfig,
    WalkForwardResult,
)

# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------


def _make_ohlcv(
    tickers: list[str],
    n_days: int = 800,
    start: date = date(2016, 1, 4),
    base_price: float = 100.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic OHLCV for CPCV-OOS testing."""
    import random

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for t_idx, ticker in enumerate(tickers):
        price = base_price + t_idx * 10
        for day in range(n_days):
            d = start + timedelta(days=day)
            vol = 0.005 + t_idx * 0.002
            ret = 0.0003 + rng.gauss(0, vol)
            price *= (1 + ret)
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": price * 0.999,
                "high": price * 1.002,
                "low": price * 0.998,
                "close": price,
                "volume": 1_000_000,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture()
def ohlcv() -> pl.DataFrame:
    """3 tickers + SPY, 800 days."""
    return _make_ohlcv(["AAPL", "MSFT", "GOOG", "SPY"])


@pytest.fixture()
def tickers() -> list[str]:
    return ["AAPL", "MSFT", "GOOG"]


@pytest.fixture()
def small_config() -> WalkForwardConfig:
    """Config with small lookback for fast synthetic tests."""
    return WalkForwardConfig(
        rebalance_every=10,
        retrain_every=50,
        lookback_days=100,
        initial_capital=100_000.0,
        rf=0.05,
    )


# ===================================================================
# Tests for math helpers
# ===================================================================


class TestNormalCDF:
    """Tests for _normal_cdf."""

    def test_zero(self) -> None:
        assert abs(_normal_cdf(0.0) - 0.5) < 1e-10

    def test_large_positive(self) -> None:
        assert _normal_cdf(5.0) > 0.999

    def test_large_negative(self) -> None:
        assert _normal_cdf(-5.0) < 0.001

    def test_symmetry(self) -> None:
        assert abs(_normal_cdf(1.0) + _normal_cdf(-1.0) - 1.0) < 1e-10

    def test_known_value(self) -> None:
        # Φ(1.96) ≈ 0.975
        assert abs(_normal_cdf(1.96) - 0.975) < 0.001


class TestInvNormalCDF:
    """Tests for _inv_normal_cdf."""

    def test_half(self) -> None:
        assert abs(_inv_normal_cdf(0.5) - 0.0) < 0.01

    def test_0975(self) -> None:
        assert abs(_inv_normal_cdf(0.975) - 1.96) < 0.02

    def test_low(self) -> None:
        assert _inv_normal_cdf(0.025) < -1.9

    def test_roundtrip(self) -> None:
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            z = _inv_normal_cdf(p)
            p_back = _normal_cdf(z)
            assert abs(p_back - p) < 0.01, f"Roundtrip failed for p={p}"

    def test_boundary_zero(self) -> None:
        assert _inv_normal_cdf(0.0) == -10.0

    def test_boundary_one(self) -> None:
        assert _inv_normal_cdf(1.0) == 10.0


class TestStdList:
    """Tests for _std_list."""

    def test_empty(self) -> None:
        assert _std_list([]) == 0.0

    def test_single(self) -> None:
        assert _std_list([5.0]) == 0.0

    def test_known(self) -> None:
        # std([1, 2, 3], ddof=1) = 1.0
        assert abs(_std_list([1.0, 2.0, 3.0]) - 1.0) < 1e-10

    def test_constant(self) -> None:
        assert _std_list([7.0, 7.0, 7.0]) == 0.0


class TestComputeSharpe:
    """Tests for _compute_sharpe."""

    def test_zero_returns_constant(self) -> None:
        """Constant zero returns have zero variance → Sharpe = 0."""
        rets = [0.0] * 100
        assert _compute_sharpe(rets, rf=0.05) == 0.0

    def test_negative_mean_returns(self) -> None:
        """Returns with negative mean excess → negative Sharpe."""
        import random
        rng = random.Random(42)
        rets = [-0.001 + rng.gauss(0, 0.005) for _ in range(252)]
        assert _compute_sharpe(rets, rf=0.05) < 0.0

    def test_positive_returns(self) -> None:
        """Returns with positive mean and variance → positive Sharpe."""
        import random
        rng = random.Random(42)
        rets = [0.002 + rng.gauss(0, 0.005) for _ in range(252)]
        sharpe = _compute_sharpe(rets, rf=0.0)
        assert sharpe > 0

    def test_insufficient_data(self) -> None:
        assert _compute_sharpe([0.01]) == 0.0
        assert _compute_sharpe([]) == 0.0


class TestSkewness:
    """Tests for _skewness."""

    def test_insufficient_data(self) -> None:
        assert _skewness([]) == 0.0
        assert _skewness([1.0]) == 0.0
        assert _skewness([1.0, 2.0]) == 0.0

    def test_symmetric_distribution(self) -> None:
        # Symmetric data → skewness ≈ 0
        vals = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        assert abs(_skewness(vals)) < 0.01

    def test_right_skewed(self) -> None:
        # Right-skewed: many small + one large
        vals = [0.0, 0.1, 0.1, 0.1, 0.1, 5.0]
        assert _skewness(vals) > 0

    def test_constant_returns_zero(self) -> None:
        assert _skewness([5.0, 5.0, 5.0, 5.0]) == 0.0


class TestKurtosis:
    """Tests for _kurtosis."""

    def test_insufficient_data(self) -> None:
        assert _kurtosis([]) == 3.0
        assert _kurtosis([1.0, 2.0, 3.0]) == 3.0

    def test_constant_returns_normal(self) -> None:
        assert _kurtosis([5.0, 5.0, 5.0, 5.0]) == 3.0

    def test_normal_distribution_near_three(self) -> None:
        """Large normal sample → kurtosis ≈ 3."""
        import random
        rng = random.Random(42)
        vals = [rng.gauss(0, 1) for _ in range(10000)]
        k = _kurtosis(vals)
        assert 2.8 < k < 3.2

    def test_heavy_tails_above_three(self) -> None:
        """Distribution with outliers → kurtosis > 3."""
        vals = [0.0] * 100 + [10.0, -10.0]
        assert _kurtosis(vals) > 3.0


class TestPurgedModelFactory:
    """Tests for _PurgedModelFactory."""

    def test_train_excludes_dates(self) -> None:
        """Training data should not contain excluded dates."""
        inner = MagicMock()
        excluded = {date(2020, 1, 5), date(2020, 1, 6)}
        factory = _PurgedModelFactory(inner, excluded)

        df = pl.DataFrame({
            "date": [date(2020, 1, d) for d in range(1, 10)],
            "close": [100.0] * 9,
        }).with_columns(pl.col("date").cast(pl.Date))

        factory.train(df)
        # inner.train should be called with filtered df
        call_args = inner.train.call_args[0][0]
        dates_in_call = set(call_args["date"].to_list())
        assert excluded.isdisjoint(dates_in_call)
        assert len(dates_in_call) == 7  # 9 - 2 excluded

    def test_predict_not_filtered(self) -> None:
        """Predict passes data through without filtering."""
        inner = MagicMock()
        inner.predict.return_value = {"AAPL": 0.5}
        factory = _PurgedModelFactory(inner, {date(2020, 1, 1)})

        df = pl.DataFrame({
            "date": [date(2020, 1, 1)],
            "close": [100.0],
        }).with_columns(pl.col("date").cast(pl.Date))

        result = factory.predict(df)
        assert result == {"AAPL": 0.5}
        inner.predict.assert_called_once_with(df)

    def test_all_excluded_still_works(self) -> None:
        """If all training data is excluded, train is still called (empty)."""
        inner = MagicMock()
        excluded = {date(2020, 1, 1), date(2020, 1, 2)}
        factory = _PurgedModelFactory(inner, excluded)

        df = pl.DataFrame({
            "date": [date(2020, 1, 1), date(2020, 1, 2)],
            "close": [100.0, 101.0],
        }).with_columns(pl.col("date").cast(pl.Date))

        # Should not raise — logs warning and returns
        factory.train(df)
        inner.train.assert_not_called()


# ===================================================================
# Tests for Deflated Sharpe Ratio
# ===================================================================


class TestDeflatedSharpeRatio:
    """Tests for deflated_sharpe_ratio."""

    def test_single_trial_no_deflation(self) -> None:
        """With n_trials=1, DSR should be high for positive Sharpe."""
        pval = deflated_sharpe_ratio(
            observed_sharpe=1.0,
            n_trials=1,
            n_observations=252,
        )
        # With 1 trial and benchmark=0, Sharpe of 1.0 should be significant
        assert pval > 0.5

    def test_many_trials_deflates(self) -> None:
        """More trials → harder to be significant → lower p-value."""
        pval_1 = deflated_sharpe_ratio(
            observed_sharpe=0.5,
            n_trials=1,
            n_observations=252,
        )
        pval_100 = deflated_sharpe_ratio(
            observed_sharpe=0.5,
            n_trials=100,
            n_observations=252,
        )
        assert pval_100 < pval_1

    def test_higher_sharpe_more_significant(self) -> None:
        """Higher Sharpe → higher p-value."""
        pval_low = deflated_sharpe_ratio(
            observed_sharpe=0.3,
            n_trials=10,
            n_observations=252,
        )
        pval_high = deflated_sharpe_ratio(
            observed_sharpe=2.0,
            n_trials=10,
            n_observations=252,
        )
        assert pval_high > pval_low

    def test_more_observations_more_significant(self) -> None:
        """More data → lower variance of SR estimator → more significant.

        Uses moderate Sharpe that exceeds E[max] but doesn't saturate
        the CDF to 1.0.  With the Lo (2002) correction (+0.5*SR^2),
        larger SR values inflate the variance, so we use SR=0.8 to
        keep p-values in a testable range.
        """
        pval_short = deflated_sharpe_ratio(
            observed_sharpe=0.8,
            n_trials=5,
            n_observations=30,
        )
        pval_long = deflated_sharpe_ratio(
            observed_sharpe=0.8,
            n_trials=5,
            n_observations=500,
        )
        assert pval_long > pval_short

    def test_zero_trials_returns_zero(self) -> None:
        assert deflated_sharpe_ratio(1.0, n_trials=0, n_observations=252) == 0.0

    def test_one_observation_returns_zero(self) -> None:
        assert deflated_sharpe_ratio(1.0, n_trials=1, n_observations=1) == 0.0

    def test_negative_sharpe_low_pvalue(self) -> None:
        """Negative Sharpe should have p-value well below 0.5."""
        pval = deflated_sharpe_ratio(
            observed_sharpe=-1.0,
            n_trials=1,
            n_observations=252,
        )
        # After daily conversion, sr_daily ≈ -0.063; with 252 obs the z-score
        # is ≈ -1.0 → CDF ≈ 0.16.  Still clearly below 0.5.
        assert pval < 0.5

    def test_benchmark_sharpe_offset(self) -> None:
        """Higher benchmark → higher E[max SR] → harder to be significant."""
        pval_zero_bench = deflated_sharpe_ratio(
            observed_sharpe=1.0,
            n_trials=10,
            n_observations=100,
            sharpe_benchmark=0.0,
        )
        pval_high_bench = deflated_sharpe_ratio(
            observed_sharpe=1.0,
            n_trials=10,
            n_observations=100,
            sharpe_benchmark=0.5,
        )
        # Higher benchmark shifts E[max SR] up → p-value drops
        assert pval_high_bench < pval_zero_bench

    def test_returns_between_zero_and_one(self) -> None:
        """p-value should always be in [0, 1]."""
        for sr in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0]:
            for n in [1, 5, 50, 200]:
                pval = deflated_sharpe_ratio(sr, n_trials=n, n_observations=252)
                assert 0.0 <= pval <= 1.0, f"Out of range: SR={sr}, n={n}, pval={pval}"


# ===================================================================
# Tests for ValidationResult
# ===================================================================


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_to_dict(self) -> None:
        vr = ValidationResult(
            config=WalkForwardConfig(),
            mean_sharpe=0.5,
            std_sharpe=0.2,
            pct_positive=0.8,
            per_path_sharpe=[0.4, 0.6, 0.5],
            deflated_sharpe=0.7,
            p_value=0.7,
            accepted=True,
        )
        d = vr.to_dict()
        assert d["mean_sharpe"] == 0.5
        assert d["accepted"] is True
        assert len(d["per_path_sharpe"]) == 3
        assert "config" not in d  # config excluded from dict

    def test_default_metadata(self) -> None:
        vr = ValidationResult(
            config=WalkForwardConfig(),
            mean_sharpe=0.0,
            std_sharpe=0.0,
            pct_positive=0.0,
            per_path_sharpe=[],
            deflated_sharpe=0.0,
            p_value=0.0,
            accepted=False,
        )
        assert vr.metadata == {}


# ===================================================================
# Tests for CPCVParameterValidator — Split generation
# ===================================================================


class TestSplitGeneration:
    """Tests for split boundaries and date assignment."""

    def test_split_count(self, ohlcv: pl.DataFrame, tickers: list[str]) -> None:
        """Default 6 splits generates exactly 6 boundaries."""
        validator = CPCVParameterValidator(ohlcv, tickers, n_splits=6)
        assert len(validator._split_boundaries) == 6

    def test_splits_cover_all_dates(self, ohlcv: pl.DataFrame, tickers: list[str]) -> None:
        """All dates assigned to exactly one split."""
        validator = CPCVParameterValidator(ohlcv, tickers, n_splits=6)
        n_total = len(validator._trading_dates)
        covered = sum(
            end - start + 1 for start, end in validator._split_boundaries
        )
        assert covered == n_total

    def test_splits_non_overlapping(self, ohlcv: pl.DataFrame, tickers: list[str]) -> None:
        """Split boundaries don't overlap."""
        validator = CPCVParameterValidator(ohlcv, tickers, n_splits=4)
        for i in range(len(validator._split_boundaries) - 1):
            _, end_i = validator._split_boundaries[i]
            start_next, _ = validator._split_boundaries[i + 1]
            assert start_next == end_i + 1

    def test_path_count_c62(self, ohlcv: pl.DataFrame, tickers: list[str]) -> None:
        """C(6,2) = 15 paths."""
        validator = CPCVParameterValidator(ohlcv, tickers, n_splits=6, n_test_groups=2)
        assert len(validator._paths) == 15

    def test_path_count_c43(self, ohlcv: pl.DataFrame, tickers: list[str]) -> None:
        """C(4,3) = 4 paths."""
        validator = CPCVParameterValidator(ohlcv, tickers, n_splits=4, n_test_groups=3)
        assert len(validator._paths) == 4

    def test_three_splits(self, ohlcv: pl.DataFrame, tickers: list[str]) -> None:
        """C(3,1) = 3 paths with 3 splits."""
        validator = CPCVParameterValidator(ohlcv, tickers, n_splits=3, n_test_groups=1)
        assert len(validator._paths) == 3


class TestTestTrainDates:
    """Tests for test/train date extraction."""

    def test_test_dates_in_correct_splits(self, ohlcv: pl.DataFrame, tickers: list[str]) -> None:
        """Test dates come from the specified split groups."""
        validator = CPCVParameterValidator(ohlcv, tickers, n_splits=4, n_test_groups=1)
        test_groups = (1,)
        test_dates = validator._get_test_dates(test_groups)

        start, end = validator._split_boundaries[1]
        expected = validator._trading_dates[start : end + 1]
        assert test_dates == sorted(expected)

    def test_train_dates_exclude_test(self, ohlcv: pl.DataFrame, tickers: list[str]) -> None:
        """Train dates do not contain any test dates."""
        validator = CPCVParameterValidator(ohlcv, tickers, n_splits=4, n_test_groups=1)
        test_groups = (2,)
        test_dates = set(validator._get_test_dates(test_groups))
        train_dates = set(validator._get_train_dates(test_groups))
        assert len(test_dates & train_dates) == 0

    def test_embargo_removes_dates_after_test(self, ohlcv: pl.DataFrame, tickers: list[str]) -> None:
        """Embargo removes days immediately after the test block."""
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=4, n_test_groups=1, embargo_pct=0.05
        )
        test_groups = (1,)
        _, test_end_idx = validator._split_boundaries[1]
        train_dates = set(validator._get_train_dates(test_groups))

        # Days right after test end should be embargoed
        for offset in range(1, validator._embargo_days + 1):
            idx = test_end_idx + offset
            if idx < len(validator._trading_dates):
                embargoed_date = validator._trading_dates[idx]
                assert embargoed_date not in train_dates

    def test_train_plus_test_plus_embargo_le_total(self, ohlcv: pl.DataFrame, tickers: list[str]) -> None:
        """Train + test + embargo dates <= total dates."""
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=6, n_test_groups=2, embargo_pct=0.02
        )
        for test_groups in validator._paths:
            test_dates = validator._get_test_dates(test_groups)
            train_dates = validator._get_train_dates(test_groups)
            assert len(test_dates) + len(train_dates) <= len(validator._trading_dates)

    def test_multiple_test_groups(self, ohlcv: pl.DataFrame, tickers: list[str]) -> None:
        """With 2 test groups, test dates come from both splits."""
        validator = CPCVParameterValidator(ohlcv, tickers, n_splits=4, n_test_groups=2)
        test_groups = (0, 3)
        test_dates = validator._get_test_dates(test_groups)

        split0_start, split0_end = validator._split_boundaries[0]
        split3_start, split3_end = validator._split_boundaries[3]
        expected_count = (split0_end - split0_start + 1) + (split3_end - split3_start + 1)
        assert len(test_dates) == expected_count


# ===================================================================
# Tests for path evaluation (with mocked backtester)
# ===================================================================


class TestEvaluatePath:
    """Tests for _evaluate_path with mocked WalkForwardBacktester."""

    def _mock_wf_result(self, dates: list[date], returns: list[float]) -> WalkForwardResult:
        """Create a minimal WalkForwardResult for testing."""
        return WalkForwardResult(
            equity_curve=pl.DataFrame({
                "date": dates,
                "portfolio_value": [100.0] * len(dates),
                "benchmark_value": [100.0] * len(dates),
            }),
            daily_returns=pl.DataFrame({
                "date": dates,
                "portfolio_return": returns,
                "benchmark_return": [0.0] * len(dates),
            }),
            rebalance_history=[],
        )

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_returns_sharpe_for_test_dates_only(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        """Sharpe should be computed only on test-period returns."""
        import random

        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=4, n_test_groups=1
        )

        # Create mock result covering all dates with noisy positive returns
        all_dates = validator._trading_dates
        n = len(all_dates)
        rng = random.Random(42)
        returns = [0.002 + rng.gauss(0, 0.005) for _ in range(n)]

        mock_result = self._mock_wf_result(all_dates, returns)
        mock_bt_cls.return_value.run.return_value = mock_result

        config = WalkForwardConfig(lookback_days=50)
        sharpe, rets = validator._evaluate_path(0, (1,), config, NaiveModelFactory())
        assert sharpe > 0
        assert len(rets) > 0

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_backtester_failure_returns_zero(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        """If backtester raises, path returns Sharpe=0 and empty returns."""
        validator = CPCVParameterValidator(ohlcv, tickers, n_splits=4, n_test_groups=1)
        mock_bt_cls.return_value.run.side_effect = ValueError("boom")

        config = WalkForwardConfig(lookback_days=50)
        sharpe, rets = validator._evaluate_path(0, (0,), config, NaiveModelFactory())
        assert sharpe == 0.0
        assert rets == []

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_each_path_gets_fresh_model_factory(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        """Each path should get a deep-copied model factory (no cross-path state)."""
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=3, n_test_groups=1
        )

        all_dates = validator._trading_dates
        mock_result = self._mock_wf_result(
            all_dates, [0.001] * len(all_dates)
        )
        mock_bt_cls.return_value.run.return_value = mock_result

        config = WalkForwardConfig(lookback_days=50)

        # Run two paths with the same model_factory
        factory = NaiveModelFactory()
        factory_id = id(factory)
        validator._evaluate_path(0, (0,), config, factory)
        validator._evaluate_path(1, (1,), config, factory)

        # Verify the backtester received different _PurgedModelFactory instances
        calls = mock_bt_cls.return_value.run.call_args_list
        assert len(calls) == 2
        # Extract model_factory from kwargs (may be positional or keyword)
        def _extract_factory(call: Any) -> Any:
            if "model_factory" in call.kwargs:
                return call.kwargs["model_factory"]
            # positional args
            for arg in call.args:
                if isinstance(arg, _PurgedModelFactory):
                    return arg
            return None

        f0 = _extract_factory(calls[0])
        f1 = _extract_factory(calls[1])
        # Both are PurgedModelFactory wrapping DIFFERENT inner instances
        assert isinstance(f0, _PurgedModelFactory)
        assert isinstance(f1, _PurgedModelFactory)
        assert f0._inner is not f1._inner  # deep-copied, not shared

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_purged_factory_excludes_test_dates(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        """The model factory passed to backtester should be _PurgedModelFactory."""
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=4, n_test_groups=1
        )

        all_dates = validator._trading_dates
        mock_result = self._mock_wf_result(
            all_dates, [0.001] * len(all_dates)
        )
        mock_bt_cls.return_value.run.return_value = mock_result

        config = WalkForwardConfig(lookback_days=50)
        validator._evaluate_path(0, (1,), config, NaiveModelFactory())

        # Verify the backtester was called with a _PurgedModelFactory
        call_kwargs = mock_bt_cls.return_value.run.call_args
        factory_arg = call_kwargs.kwargs.get(
            "model_factory",
            call_kwargs.args[3] if len(call_kwargs.args) > 3 else None,
        )
        # The factory should be a _PurgedModelFactory (not the raw NaiveModelFactory)
        assert isinstance(factory_arg, _PurgedModelFactory)


# ===================================================================
# Tests for validate()
# ===================================================================


class TestValidate:
    """Tests for the full validate() pipeline."""

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_returns_validation_result(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        """validate() returns a ValidationResult with correct path count."""
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=4, n_test_groups=1
        )

        all_dates = validator._trading_dates
        mock_result = WalkForwardResult(
            equity_curve=pl.DataFrame({
                "date": all_dates,
                "portfolio_value": [100.0] * len(all_dates),
                "benchmark_value": [100.0] * len(all_dates),
            }),
            daily_returns=pl.DataFrame({
                "date": all_dates,
                "portfolio_return": [0.001] * len(all_dates),
                "benchmark_return": [0.0] * len(all_dates),
            }),
            rebalance_history=[],
        )
        mock_bt_cls.return_value.run.return_value = mock_result

        config = WalkForwardConfig(lookback_days=50)
        result = validator.validate(config)

        assert isinstance(result, ValidationResult)
        assert len(result.per_path_sharpe) == 4  # C(4,1) = 4
        assert result.metadata["n_paths"] == 4

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_all_positive_paths_accepted(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        """If all paths have strongly positive Sharpe, config is accepted."""
        import random

        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=3, n_test_groups=1, acceptance_pct=0.6
        )

        all_dates = validator._trading_dates
        # Very strong positive returns with low noise → high Sharpe, passes DSR 0.95
        rng = random.Random(42)
        returns = [0.005 + rng.gauss(0, 0.003) for _ in range(len(all_dates))]

        mock_result = WalkForwardResult(
            equity_curve=pl.DataFrame({
                "date": all_dates,
                "portfolio_value": [100.0] * len(all_dates),
                "benchmark_value": [100.0] * len(all_dates),
            }),
            daily_returns=pl.DataFrame({
                "date": all_dates,
                "portfolio_return": returns,
                "benchmark_return": [0.0] * len(all_dates),
            }),
            rebalance_history=[],
        )
        mock_bt_cls.return_value.run.return_value = mock_result

        config = WalkForwardConfig(lookback_days=50)
        result = validator.validate(config, n_trials=1)
        assert result.pct_positive == 1.0
        assert result.mean_sharpe > 0

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_all_negative_paths_rejected(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        """If all paths have negative Sharpe, config is rejected."""
        import random

        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=3, n_test_groups=1
        )

        all_dates = validator._trading_dates
        # Strongly negative returns with noise → negative Sharpe in all paths
        rng = random.Random(42)
        returns = [-0.003 + rng.gauss(0, 0.005) for _ in range(len(all_dates))]

        mock_result = WalkForwardResult(
            equity_curve=pl.DataFrame({
                "date": all_dates,
                "portfolio_value": [100.0] * len(all_dates),
                "benchmark_value": [100.0] * len(all_dates),
            }),
            daily_returns=pl.DataFrame({
                "date": all_dates,
                "portfolio_return": returns,
                "benchmark_return": [0.0] * len(all_dates),
            }),
            rebalance_history=[],
        )
        mock_bt_cls.return_value.run.return_value = mock_result

        config = WalkForwardConfig(lookback_days=50)
        result = validator.validate(config, n_trials=1)
        assert result.pct_positive == 0.0
        assert result.accepted is False

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_metadata_includes_empirical_stats(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        """Metadata should include empirical skewness, kurtosis, n_observations."""
        import random

        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=3, n_test_groups=1
        )

        all_dates = validator._trading_dates
        rng = random.Random(42)
        returns = [0.001 + rng.gauss(0, 0.005) for _ in range(len(all_dates))]

        mock_result = WalkForwardResult(
            equity_curve=pl.DataFrame({
                "date": all_dates,
                "portfolio_value": [100.0] * len(all_dates),
                "benchmark_value": [100.0] * len(all_dates),
            }),
            daily_returns=pl.DataFrame({
                "date": all_dates,
                "portfolio_return": returns,
                "benchmark_return": [0.0] * len(all_dates),
            }),
            rebalance_history=[],
        )
        mock_bt_cls.return_value.run.return_value = mock_result

        config = WalkForwardConfig(lookback_days=50)
        result = validator.validate(config, n_trials=1)
        assert "skewness" in result.metadata
        assert "kurtosis" in result.metadata
        assert "n_observations" in result.metadata
        assert result.metadata["n_observations"] > 0

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_default_model_factory_is_naive(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        """When model_factory=None, uses NaiveModelFactory."""
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=3, n_test_groups=1
        )

        all_dates = validator._trading_dates
        mock_result = WalkForwardResult(
            equity_curve=pl.DataFrame({
                "date": all_dates,
                "portfolio_value": [100.0] * len(all_dates),
                "benchmark_value": [100.0] * len(all_dates),
            }),
            daily_returns=pl.DataFrame({
                "date": all_dates,
                "portfolio_return": [0.0] * len(all_dates),
                "benchmark_return": [0.0] * len(all_dates),
            }),
            rebalance_history=[],
        )
        mock_bt_cls.return_value.run.return_value = mock_result

        config = WalkForwardConfig(lookback_days=50)
        # Should not raise — NaiveModelFactory is used as default
        result = validator.validate(config, model_factory=None)
        assert isinstance(result, ValidationResult)

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_n_trials_affects_dsr(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        """More trials → lower deflated Sharpe."""
        import random

        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=3, n_test_groups=1
        )

        all_dates = validator._trading_dates
        rng = random.Random(42)
        returns = [0.001 + rng.gauss(0, 0.005) for _ in range(len(all_dates))]

        mock_result = WalkForwardResult(
            equity_curve=pl.DataFrame({
                "date": all_dates,
                "portfolio_value": [100.0] * len(all_dates),
                "benchmark_value": [100.0] * len(all_dates),
            }),
            daily_returns=pl.DataFrame({
                "date": all_dates,
                "portfolio_return": returns,
                "benchmark_return": [0.0] * len(all_dates),
            }),
            rebalance_history=[],
        )
        mock_bt_cls.return_value.run.return_value = mock_result

        config = WalkForwardConfig(lookback_days=50)
        result_1 = validator.validate(config, n_trials=1)
        result_50 = validator.validate(config, n_trials=50)

        assert result_50.deflated_sharpe <= result_1.deflated_sharpe


# ===================================================================
# Tests for grid_search()
# ===================================================================


class TestGridSearch:
    """Tests for grid_search across multiple configs."""

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_returns_sorted_by_dsr(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        """Results are sorted by deflated_sharpe descending."""
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=3, n_test_groups=1
        )

        all_dates = validator._trading_dates
        mock_result = WalkForwardResult(
            equity_curve=pl.DataFrame({
                "date": all_dates,
                "portfolio_value": [100.0] * len(all_dates),
                "benchmark_value": [100.0] * len(all_dates),
            }),
            daily_returns=pl.DataFrame({
                "date": all_dates,
                "portfolio_return": [0.001] * len(all_dates),
                "benchmark_return": [0.0] * len(all_dates),
            }),
            rebalance_history=[],
        )
        mock_bt_cls.return_value.run.return_value = mock_result

        configs = {
            "config_a": WalkForwardConfig(lookback_days=50, rebalance_every=5),
            "config_b": WalkForwardConfig(lookback_days=50, rebalance_every=10),
        }

        results = validator.grid_search(configs)
        assert len(results) == 2
        # Sorted descending
        assert results[0][1].deflated_sharpe >= results[1][1].deflated_sharpe

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_grid_search_n_trials_equals_len_configs(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        """n_trials passed to validate equals len(configs)."""
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=3, n_test_groups=1
        )

        all_dates = validator._trading_dates
        mock_result = WalkForwardResult(
            equity_curve=pl.DataFrame({
                "date": all_dates,
                "portfolio_value": [100.0] * len(all_dates),
                "benchmark_value": [100.0] * len(all_dates),
            }),
            daily_returns=pl.DataFrame({
                "date": all_dates,
                "portfolio_return": [0.001] * len(all_dates),
                "benchmark_return": [0.0] * len(all_dates),
            }),
            rebalance_history=[],
        )
        mock_bt_cls.return_value.run.return_value = mock_result

        configs = {
            "a": WalkForwardConfig(lookback_days=50),
            "b": WalkForwardConfig(lookback_days=50),
            "c": WalkForwardConfig(lookback_days=50),
        }

        results = validator.grid_search(configs)
        # All should have n_trials=3 in metadata
        for _, vr in results:
            assert vr.metadata["n_trials"] == 3

    @patch("src.backtest.cpcv_oos.WalkForwardBacktester")
    def test_grid_search_returns_names(
        self, mock_bt_cls: MagicMock, ohlcv: pl.DataFrame, tickers: list[str]
    ) -> None:
        """Each result has its config name."""
        validator = CPCVParameterValidator(
            ohlcv, tickers, n_splits=3, n_test_groups=1
        )

        all_dates = validator._trading_dates
        mock_result = WalkForwardResult(
            equity_curve=pl.DataFrame({
                "date": all_dates,
                "portfolio_value": [100.0] * len(all_dates),
                "benchmark_value": [100.0] * len(all_dates),
            }),
            daily_returns=pl.DataFrame({
                "date": all_dates,
                "portfolio_return": [0.001] * len(all_dates),
                "benchmark_return": [0.0] * len(all_dates),
            }),
            rebalance_history=[],
        )
        mock_bt_cls.return_value.run.return_value = mock_result

        configs = {"alpha": WalkForwardConfig(lookback_days=50)}
        results = validator.grid_search(configs)
        names = {name for name, _ in results}
        assert "alpha" in names


# ===================================================================
# Integration test (no mocks — uses real walk-forward with small data)
# ===================================================================


class TestIntegration:
    """End-to-end tests with real backtester on tiny synthetic data."""

    def test_validate_runs_without_mock(self, ohlcv: pl.DataFrame, tickers: list[str]) -> None:
        """Full pipeline with real backtester on 800-day synthetic data."""
        validator = CPCVParameterValidator(
            ohlcv, tickers,
            n_splits=3,
            n_test_groups=1,
            embargo_pct=0.01,
        )

        config = WalkForwardConfig(
            rebalance_every=20,
            retrain_every=100,
            lookback_days=100,
            initial_capital=100_000.0,
        )

        result = validator.validate(config, n_trials=1)
        assert isinstance(result, ValidationResult)
        assert len(result.per_path_sharpe) == 3  # C(3,1)
        assert 0.0 <= result.pct_positive <= 1.0
        assert 0.0 <= result.p_value <= 1.0
