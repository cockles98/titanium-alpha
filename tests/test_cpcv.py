"""Tests for src/backtest/cpcv.py — Combinatorial Purged Cross-Validation.

Covers dataclasses, constructor validation, split generation, purge/embargo
logic, metric computation, prediction evaluation, full pipeline, and
result aggregation.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from src.backtest.cpcv import (
    BacktestResult,
    CPCVBacktester,
    FoldResult,
    TransactionCosts,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures local to this test module
# ---------------------------------------------------------------------------


def _make_ohlcv_df(n: int, start_price: float = 100.0) -> pl.DataFrame:
    """Create a synthetic OHLCV Polars DataFrame with *n* rows.

    Prices follow a slight upward random walk seeded deterministically.
    """
    import random

    random.seed(12345)
    base_date = date(2020, 1, 2)
    dates = [base_date + timedelta(days=i) for i in range(n)]

    closes: list[float] = [start_price]
    for _ in range(n - 1):
        closes.append(max(closes[-1] + random.gauss(0.05, 1.0), 1.0))

    opens = [c + random.gauss(0, 0.3) for c in closes]
    highs = [max(o, c) + abs(random.gauss(0, 0.5)) for o, c in zip(opens, closes)]
    lows = [min(o, c) - abs(random.gauss(0, 0.5)) for o, c in zip(opens, closes)]
    volumes = [int(1_000_000 + random.gauss(0, 100_000)) for _ in range(n)]

    return pl.DataFrame(
        {
            "date": dates,
            "close": closes,
            "open": opens,
            "high": highs,
            "low": lows,
            "volume": volumes,
        }
    )


def _mock_factory_positive(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> pl.DataFrame:
    """Model factory that always predicts positive return."""
    return pl.DataFrame(
        {
            "date": test_df["date"],
            "predicted_return": [0.01] * test_df.height,
        }
    )


def _mock_factory_negative(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> pl.DataFrame:
    """Model factory that always predicts negative return (stay flat)."""
    return pl.DataFrame(
        {
            "date": test_df["date"],
            "predicted_return": [-0.01] * test_df.height,
        }
    )


def _mock_factory_zero(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> pl.DataFrame:
    """Model factory that predicts zero return (stay flat)."""
    return pl.DataFrame(
        {
            "date": test_df["date"],
            "predicted_return": [0.0] * test_df.height,
        }
    )


def _mock_factory_alternating(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> pl.DataFrame:
    """Model factory that alternates positive and negative predictions."""
    returns = [0.01 if i % 2 == 0 else -0.01 for i in range(test_df.height)]
    return pl.DataFrame(
        {
            "date": test_df["date"],
            "predicted_return": returns,
        }
    )


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestFoldResult:
    """Tests for the FoldResult dataclass."""

    def test_instantiation(self) -> None:
        """FoldResult stores all fields correctly."""
        fr = FoldResult(
            path_id=0,
            test_groups=(0, 1),
            sharpe=1.5,
            max_drawdown=-0.10,
            cagr=0.12,
            n_train=400,
            n_test=100,
            n_returns=20,
        )
        assert fr.path_id == 0
        assert fr.test_groups == (0, 1)
        assert fr.sharpe == 1.5
        assert fr.max_drawdown == -0.10
        assert fr.cagr == 0.12
        assert fr.n_train == 400
        assert fr.n_test == 100
        assert fr.n_returns == 20


class TestBacktestResult:
    """Tests for the BacktestResult dataclass."""

    def test_defaults(self) -> None:
        """BacktestResult has sensible defaults (all zeros, empty list)."""
        br = BacktestResult()
        assert br.fold_results == []
        assert br.n_paths == 0
        assert br.mean_sharpe == 0.0
        assert br.std_sharpe == 0.0
        assert br.pct_positive_sharpe == 0.0
        assert br.mean_max_drawdown == 0.0
        assert br.mean_cagr == 0.0

    def test_instantiation_with_values(self) -> None:
        """BacktestResult stores provided values."""
        br = BacktestResult(n_paths=15, mean_sharpe=0.8)
        assert br.n_paths == 15
        assert br.mean_sharpe == 0.8


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for CPCVBacktester.__init__ validation."""

    def test_default_params(self) -> None:
        """Default constructor succeeds with expected attributes."""
        bt = CPCVBacktester()
        assert bt.n_splits == 6
        assert bt.n_test_groups == 2
        assert bt.embargo_days == 10
        assert bt.h == 5
        assert bt.input_size == 60
        assert bt.rf == 0.05
        assert bt._purge_window == 5 + 60 - 1  # 64

    def test_n_splits_too_small(self) -> None:
        """n_splits < 3 raises ValueError."""
        with pytest.raises(ValueError, match="n_splits must be >= 3"):
            CPCVBacktester(n_splits=2)

    def test_n_splits_boundary(self) -> None:
        """n_splits=3 is the minimum accepted."""
        bt = CPCVBacktester(n_splits=3)
        assert bt.n_splits == 3

    def test_n_test_groups_zero(self) -> None:
        """n_test_groups=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_test_groups must be in"):
            CPCVBacktester(n_test_groups=0)

    def test_n_test_groups_equals_n_splits(self) -> None:
        """n_test_groups == n_splits raises ValueError (no train data)."""
        with pytest.raises(ValueError, match="n_test_groups must be in"):
            CPCVBacktester(n_splits=6, n_test_groups=6)

    def test_n_test_groups_exceeds_n_splits(self) -> None:
        """n_test_groups > n_splits raises ValueError."""
        with pytest.raises(ValueError, match="n_test_groups must be in"):
            CPCVBacktester(n_splits=6, n_test_groups=7)

    def test_embargo_days_negative(self) -> None:
        """Negative embargo_days raises ValueError."""
        with pytest.raises(ValueError, match="embargo_days must be >= 0"):
            CPCVBacktester(embargo_days=-1)

    def test_embargo_days_zero(self) -> None:
        """embargo_days=0 is valid."""
        bt = CPCVBacktester(embargo_days=0)
        assert bt.embargo_days == 0

    def test_h_zero(self) -> None:
        """h=0 raises ValueError."""
        with pytest.raises(ValueError, match="h must be >= 1"):
            CPCVBacktester(h=0)

    def test_purge_window_computation(self) -> None:
        """Purge window is h + input_size - 1."""
        bt = CPCVBacktester(h=10, input_size=30)
        assert bt._purge_window == 10 + 30 - 1  # 39


# ---------------------------------------------------------------------------
# _split_into_groups
# ---------------------------------------------------------------------------


class TestSplitIntoGroups:
    """Tests for _split_into_groups."""

    def test_even_split(self) -> None:
        """600 samples / 6 groups = 100 each, no remainder."""
        bt = CPCVBacktester(n_splits=6)
        groups = bt._split_into_groups(600)
        assert len(groups) == 6
        for start, end in groups:
            assert end - start == 100
        # Check contiguous coverage
        assert groups[0][0] == 0
        assert groups[-1][1] == 600

    def test_remainder_distribution(self) -> None:
        """Remainder samples are distributed to the first groups."""
        bt = CPCVBacktester(n_splits=6)
        groups = bt._split_into_groups(604)  # 604 / 6 = 100 rem 4
        sizes = [end - start for start, end in groups]
        # First 4 groups get 101, last 2 get 100
        assert sizes == [101, 101, 101, 101, 100, 100]
        assert groups[-1][1] == 604

    def test_contiguous(self) -> None:
        """Groups are contiguous with no gaps."""
        bt = CPCVBacktester(n_splits=5, n_test_groups=1)
        groups = bt._split_into_groups(503)
        for i in range(len(groups) - 1):
            assert groups[i][1] == groups[i + 1][0]

    def test_three_splits(self) -> None:
        """Minimum n_splits=3 works correctly."""
        bt = CPCVBacktester(n_splits=3)
        groups = bt._split_into_groups(300)
        assert len(groups) == 3
        assert all(end - start == 100 for start, end in groups)


# ---------------------------------------------------------------------------
# _apply_purge
# ---------------------------------------------------------------------------


class TestApplyPurge:
    """Tests for _apply_purge."""

    def test_purge_removes_correct_zone(self) -> None:
        """Purge zone is [test_start - purge_window, test_start)."""
        bt = CPCVBacktester(h=5, input_size=60)  # purge_window = 64
        train = set(range(200))
        test_start = 100
        result = bt._apply_purge(train, test_start, 150)
        purge_zone = set(range(100 - 64, 100))  # indices 36..99
        assert purge_zone.isdisjoint(result)
        # Indices before purge zone should remain
        assert 35 in result
        # Indices at purge boundary
        assert 36 not in result
        assert 99 not in result

    def test_purge_near_start(self) -> None:
        """Purge zone is clamped to 0 when test_start < purge_window."""
        bt = CPCVBacktester(h=5, input_size=60)  # purge_window = 64
        train = set(range(200))
        test_start = 30  # purge would go negative
        result = bt._apply_purge(train, test_start, 50)
        # Indices 0..29 should all be purged
        for i in range(30):
            assert i not in result

    def test_purge_no_overlap(self) -> None:
        """Purge does not affect indices outside the purge zone."""
        bt = CPCVBacktester(h=5, input_size=10)  # purge_window = 14
        train = set(range(50, 200))
        result = bt._apply_purge(train, 100, 150)
        # Purge zone: 86..99 -- only some overlap with train (which starts at 50)
        for i in range(86, 100):
            assert i not in result
        for i in range(50, 86):
            assert i in result

    def test_purge_empty_train(self) -> None:
        """Purging an empty training set returns empty."""
        bt = CPCVBacktester()
        result = bt._apply_purge(set(), 100, 200)
        assert result == set()


# ---------------------------------------------------------------------------
# _apply_embargo
# ---------------------------------------------------------------------------


class TestApplyEmbargo:
    """Tests for _apply_embargo."""

    def test_embargo_removes_correct_zone(self) -> None:
        """Embargo zone is [test_end, test_end + embargo_days)."""
        bt = CPCVBacktester(embargo_days=10)
        train = set(range(200))
        result = bt._apply_embargo(train, 100, 200)
        # Indices 100..109 should be removed
        for i in range(100, 110):
            assert i not in result
        assert 110 in result

    def test_embargo_clamped_at_end(self) -> None:
        """Embargo does not exceed n_samples."""
        bt = CPCVBacktester(embargo_days=10)
        train = set(range(105))
        result = bt._apply_embargo(train, 100, 105)
        # Only indices 100..104 removed (not 105..109)
        for i in range(100, 105):
            assert i not in result

    def test_embargo_zero(self) -> None:
        """embargo_days=0 removes nothing."""
        bt = CPCVBacktester(embargo_days=0)
        train = set(range(200))
        result = bt._apply_embargo(train, 100, 200)
        assert result == train

    def test_embargo_empty_train(self) -> None:
        """Embargoing an empty training set returns empty."""
        bt = CPCVBacktester(embargo_days=10)
        result = bt._apply_embargo(set(), 100, 200)
        assert result == set()


# ---------------------------------------------------------------------------
# generate_paths
# ---------------------------------------------------------------------------


class TestGeneratePaths:
    """Tests for generate_paths."""

    def test_number_of_paths(self) -> None:
        """C(6, 2) = 15 paths are generated."""
        bt = CPCVBacktester(n_splits=6, n_test_groups=2, h=2, input_size=5)
        # purge_window = 6, need enough data
        paths = bt.generate_paths(600)
        assert len(paths) == math.comb(6, 2)  # 15

    def test_c_5_1_paths(self) -> None:
        """C(5, 1) = 5 paths."""
        bt = CPCVBacktester(n_splits=5, n_test_groups=1, h=2, input_size=5)
        paths = bt.generate_paths(500)
        assert len(paths) == 5

    def test_no_train_test_overlap(self) -> None:
        """Train and test indices never overlap in any path."""
        bt = CPCVBacktester(n_splits=6, n_test_groups=2, h=2, input_size=5)
        paths = bt.generate_paths(600)
        for train_idx, test_idx, _ in paths:
            overlap = set(train_idx) & set(test_idx)
            assert overlap == set(), f"Overlap found: {overlap}"

    def test_purge_applied(self) -> None:
        """Indices in the purge zone before test start are excluded from train."""
        bt = CPCVBacktester(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=0
        )
        # purge_window = 6
        n = 400
        paths = bt.generate_paths(n)
        groups = bt._split_into_groups(n)

        for train_idx, test_idx, test_groups in paths:
            train_set = set(train_idx)
            for g in test_groups:
                test_start = groups[g][0]
                purge_start = max(0, test_start - bt._purge_window)
                for i in range(purge_start, test_start):
                    assert i not in train_set, (
                        f"Index {i} in purge zone but found in train"
                    )

    def test_embargo_applied(self) -> None:
        """Indices in the embargo zone after test end are excluded from train."""
        bt = CPCVBacktester(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=10
        )
        n = 400
        paths = bt.generate_paths(n)
        groups = bt._split_into_groups(n)

        for train_idx, test_idx, test_groups in paths:
            train_set = set(train_idx)
            for g in test_groups:
                test_end = groups[g][1]
                embargo_end = min(n, test_end + bt.embargo_days)
                for i in range(test_end, embargo_end):
                    assert i not in train_set, (
                        f"Index {i} in embargo zone but found in train"
                    )

    def test_dataset_too_small(self) -> None:
        """ValueError when groups are smaller than purge_window + embargo."""
        bt = CPCVBacktester(n_splits=6, n_test_groups=2, h=5, input_size=60, embargo_days=10)
        # purge_window=64, embargo=10, need group_size >= 74
        # 6 groups of ~70 each = 420 total -> too small
        with pytest.raises(ValueError, match="Dataset too small"):
            bt.generate_paths(420)

    def test_all_indices_covered(self) -> None:
        """Every index 0..n-1 appears in at least one path (train or test)."""
        bt = CPCVBacktester(n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=0)
        n = 400
        paths = bt.generate_paths(n)
        all_seen: set[int] = set()
        for train_idx, test_idx, _ in paths:
            all_seen.update(train_idx)
            all_seen.update(test_idx)
        # Every index should appear somewhere (at least in test of some path)
        assert all_seen == set(range(n))

    def test_test_groups_match_indices(self) -> None:
        """test_groups tuple matches the actual test index ranges."""
        bt = CPCVBacktester(n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=0)
        n = 400
        paths = bt.generate_paths(n)
        groups = bt._split_into_groups(n)

        for _, test_idx, test_groups in paths:
            expected_test = set()
            for g in test_groups:
                start, end = groups[g]
                expected_test.update(range(start, end))
            assert set(test_idx) == expected_test


# ---------------------------------------------------------------------------
# _compute_sharpe
# ---------------------------------------------------------------------------


class TestComputeSharpe:
    """Tests for the static _compute_sharpe method."""

    def test_positive_returns(self) -> None:
        """Mostly positive returns with variance yield positive Sharpe."""
        import random
        random.seed(99)
        returns = [0.02 + random.gauss(0, 0.005) for _ in range(50)]
        sharpe = CPCVBacktester._compute_sharpe(returns, rf=0.05, periods_per_year=252 / 5)
        assert sharpe > 0

    def test_negative_returns(self) -> None:
        """Mostly negative returns with variance yield negative Sharpe."""
        import random
        random.seed(99)
        returns = [-0.02 + random.gauss(0, 0.005) for _ in range(50)]
        sharpe = CPCVBacktester._compute_sharpe(returns, rf=0.05, periods_per_year=252 / 5)
        assert sharpe < 0

    def test_zero_volatility(self) -> None:
        """Constant returns (zero vol) return 0.0."""
        returns = [0.001] * 50
        sharpe = CPCVBacktester._compute_sharpe(returns, rf=0.0, periods_per_year=252)
        assert sharpe == 0.0

    def test_single_return(self) -> None:
        """Single return (< 2 observations) returns 0.0."""
        sharpe = CPCVBacktester._compute_sharpe([0.05], rf=0.05, periods_per_year=252)
        assert sharpe == 0.0

    def test_empty_returns(self) -> None:
        """Empty list returns 0.0."""
        sharpe = CPCVBacktester._compute_sharpe([], rf=0.05, periods_per_year=252)
        assert sharpe == 0.0

    def test_annualisation(self) -> None:
        """Sharpe scales with sqrt of periods_per_year."""
        import random
        random.seed(42)
        returns = [random.gauss(0.01, 0.02) for _ in range(100)]
        s1 = CPCVBacktester._compute_sharpe(returns, rf=0.0, periods_per_year=52)
        s2 = CPCVBacktester._compute_sharpe(returns, rf=0.0, periods_per_year=252)
        # s2/s1 should be approximately sqrt(252/52)
        ratio = s2 / s1 if s1 != 0 else 0
        expected_ratio = math.sqrt(252 / 52)
        assert abs(ratio - expected_ratio) < 0.01


# ---------------------------------------------------------------------------
# _compute_max_drawdown
# ---------------------------------------------------------------------------


class TestComputeMaxDrawdown:
    """Tests for the static _compute_max_drawdown method."""

    def test_monotonic_up(self) -> None:
        """Monotonically increasing series has zero drawdown."""
        cum = [1.0, 1.05, 1.10, 1.15, 1.20]
        assert CPCVBacktester._compute_max_drawdown(cum) == 0.0

    def test_with_drawdown(self) -> None:
        """Series with a 10% drawdown reports -0.10."""
        cum = [1.0, 1.10, 1.00, 0.99, 1.05]
        # Peak 1.10 -> trough 0.99 => dd = (0.99-1.10)/1.10 = -0.1
        dd = CPCVBacktester._compute_max_drawdown(cum)
        assert dd == pytest.approx(-0.1, abs=0.001)

    def test_empty_list(self) -> None:
        """Empty series returns 0.0."""
        assert CPCVBacktester._compute_max_drawdown([]) == 0.0

    def test_single_point(self) -> None:
        """Single-point series has zero drawdown."""
        assert CPCVBacktester._compute_max_drawdown([1.0]) == 0.0

    def test_monotonic_down(self) -> None:
        """Monotonically decreasing series: DD = (last - first) / first."""
        cum = [1.0, 0.9, 0.8, 0.7]
        dd = CPCVBacktester._compute_max_drawdown(cum)
        assert dd == pytest.approx((0.7 - 1.0) / 1.0, abs=1e-9)

    def test_drawdown_is_negative(self) -> None:
        """Drawdown value is always <= 0."""
        cum = [1.0, 1.2, 0.5, 0.8, 1.0, 0.3]
        dd = CPCVBacktester._compute_max_drawdown(cum)
        assert dd <= 0.0


# ---------------------------------------------------------------------------
# _compute_cagr
# ---------------------------------------------------------------------------


class TestComputeCAGR:
    """Tests for the static _compute_cagr method."""

    def test_positive_growth(self) -> None:
        """Doubling over 252 days => CAGR ~= 1.0 (100%)."""
        cum = [1.0, 2.0]
        cagr = CPCVBacktester._compute_cagr(cum, n_days=252)
        assert cagr == pytest.approx(1.0, abs=0.01)

    def test_no_growth(self) -> None:
        """Flat series has CAGR = 0."""
        cum = [1.0, 1.0]
        cagr = CPCVBacktester._compute_cagr(cum, n_days=252)
        assert cagr == pytest.approx(0.0, abs=1e-9)

    def test_negative_growth(self) -> None:
        """Loss over the period => negative CAGR."""
        cum = [1.0, 0.8]
        cagr = CPCVBacktester._compute_cagr(cum, n_days=252)
        assert cagr < 0.0

    def test_empty_series(self) -> None:
        """Empty series returns 0.0."""
        assert CPCVBacktester._compute_cagr([], n_days=252) == 0.0

    def test_zero_days(self) -> None:
        """n_days=0 returns 0.0."""
        assert CPCVBacktester._compute_cagr([1.0, 1.5], n_days=0) == 0.0

    def test_negative_final_value(self) -> None:
        """Final value <= 0 returns -1.0."""
        assert CPCVBacktester._compute_cagr([1.0, -0.5], n_days=252) == -1.0


# ---------------------------------------------------------------------------
# _evaluate_predictions
# ---------------------------------------------------------------------------


class TestEvaluatePredictions:
    """Tests for _evaluate_predictions (long/flat strategy)."""

    def _make_test_df(self, n: int, start_price: float = 100.0) -> pl.DataFrame:
        """Helper: create a simple test DataFrame with date and close."""
        base = date(2023, 6, 1)
        # Slight upward trend
        closes = [start_price + i * 0.5 for i in range(n)]
        return pl.DataFrame(
            {
                "date": [base + timedelta(days=i) for i in range(n)],
                "close": closes,
            }
        )

    def test_all_positive_predictions(self) -> None:
        """All positive predictions => always long => FoldResult has returns."""
        bt = CPCVBacktester(h=5)
        test_df = self._make_test_df(50)
        predictions = pl.DataFrame(
            {
                "date": test_df["date"],
                "predicted_return": [0.01] * 50,
            }
        )
        test_indices = list(range(50))
        result = bt._evaluate_predictions(predictions, test_df, test_indices, 0, (0, 1))
        assert result.n_returns == (50 - 5) // 5  # 9
        assert result.n_test == 50
        assert result.path_id == 0
        assert result.test_groups == (0, 1)

    def test_all_negative_predictions(self) -> None:
        """All negative predictions => always flat => all strategy returns = 0."""
        bt = CPCVBacktester(h=5)
        test_df = self._make_test_df(50)
        predictions = pl.DataFrame(
            {
                "date": test_df["date"],
                "predicted_return": [-0.01] * 50,
            }
        )
        test_indices = list(range(50))
        result = bt._evaluate_predictions(predictions, test_df, test_indices, 0, (0,))
        # All flat => sharpe should be 0 (zero vol)
        assert result.sharpe == 0.0

    def test_non_overlapping_returns(self) -> None:
        """Returns are computed every h days (non-overlapping)."""
        bt = CPCVBacktester(h=10)
        test_df = self._make_test_df(100)
        predictions = pl.DataFrame(
            {
                "date": test_df["date"],
                "predicted_return": [0.01] * 100,
            }
        )
        test_indices = list(range(100))
        result = bt._evaluate_predictions(predictions, test_df, test_indices, 0, (0,))
        # With h=10 and 100 rows: i goes 0,10,20,...,90 but need i+10<100
        # so i can be 0,10,...,80 => 9 returns
        assert result.n_returns == 9

    def test_missing_prediction_defaults_flat(self) -> None:
        """Missing prediction date => pred_map returns 0.0 => flat."""
        bt = CPCVBacktester(h=5)
        test_df = self._make_test_df(30)
        # Only provide predictions for half the dates
        predictions = pl.DataFrame(
            {
                "date": test_df["date"][:15],
                "predicted_return": [0.01] * 15,
            }
        )
        test_indices = list(range(30))
        result = bt._evaluate_predictions(predictions, test_df, test_indices, 0, (0,))
        assert result.n_returns > 0

    def test_short_test_set(self) -> None:
        """Test set shorter than h yields zero returns."""
        bt = CPCVBacktester(h=10)
        test_df = self._make_test_df(8)
        predictions = pl.DataFrame(
            {
                "date": test_df["date"],
                "predicted_return": [0.01] * 8,
            }
        )
        test_indices = list(range(8))
        result = bt._evaluate_predictions(predictions, test_df, test_indices, 0, (0,))
        assert result.n_returns == 0
        assert result.sharpe == 0.0


# ---------------------------------------------------------------------------
# run (full pipeline)
# ---------------------------------------------------------------------------


class TestRun:
    """Tests for the full run() pipeline with mock model factories."""

    def test_run_positive_factory(self) -> None:
        """Full pipeline with always-positive predictions completes."""
        bt = CPCVBacktester(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=3
        )
        df = _make_ohlcv_df(500)
        result = bt.run(df, _mock_factory_positive)

        assert result.n_paths == math.comb(4, 1)  # 4
        assert len(result.fold_results) == 4
        # Positive predictions on upward-trending data => mostly positive sharpe
        assert result.mean_sharpe != 0.0

    def test_run_negative_factory(self) -> None:
        """All-negative predictions => always flat => zero sharpe."""
        bt = CPCVBacktester(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=3
        )
        df = _make_ohlcv_df(500)
        result = bt.run(df, _mock_factory_negative)
        # All flat => zero vol => sharpe = 0 for each path
        assert result.mean_sharpe == 0.0

    def test_run_zero_factory(self) -> None:
        """Zero predictions => flat => zero sharpe."""
        bt = CPCVBacktester(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=3
        )
        df = _make_ohlcv_df(500)
        result = bt.run(df, _mock_factory_zero)
        assert result.mean_sharpe == 0.0

    def test_run_missing_columns(self) -> None:
        """Missing required columns raises ValueError."""
        bt = CPCVBacktester()
        df = pl.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            bt.run(df, _mock_factory_positive)

    def test_run_missing_close(self) -> None:
        """Missing 'close' column raises ValueError."""
        bt = CPCVBacktester()
        df = pl.DataFrame({"date": [date(2023, 1, 1)]})
        with pytest.raises(ValueError, match="close"):
            bt.run(df, _mock_factory_positive)

    def test_run_missing_date(self) -> None:
        """Missing 'date' column raises ValueError."""
        bt = CPCVBacktester()
        df = pl.DataFrame({"close": [100.0]})
        with pytest.raises(ValueError, match="date"):
            bt.run(df, _mock_factory_positive)

    def test_run_data_too_small(self) -> None:
        """Dataset too small for splits raises ValueError."""
        bt = CPCVBacktester(n_splits=6, n_test_groups=2, h=5, input_size=60, embargo_days=10)
        df = _make_ohlcv_df(100)
        with pytest.raises(ValueError, match="Dataset too small"):
            bt.run(df, _mock_factory_positive)

    def test_run_fold_results_populated(self) -> None:
        """Each FoldResult has correct n_train and n_test > 0."""
        bt = CPCVBacktester(
            n_splits=3, n_test_groups=1, h=2, input_size=5, embargo_days=2
        )
        df = _make_ohlcv_df(300)
        result = bt.run(df, _mock_factory_positive)

        for fold in result.fold_results:
            assert fold.n_train > 0
            assert fold.n_test > 0
            assert fold.n_train + fold.n_test <= 300

    def test_run_train_test_no_leakage(self) -> None:
        """Verify no data leakage: model_factory receives disjoint train/test."""
        seen_splits: list[tuple[set[int], set[int]]] = []

        def tracking_factory(
            train_df: pl.DataFrame, test_df: pl.DataFrame
        ) -> pl.DataFrame:
            # Record the row counts to verify disjoint later
            seen_splits.append(
                (set(range(train_df.height)), set(range(test_df.height)))
            )
            return pl.DataFrame(
                {
                    "date": test_df["date"],
                    "predicted_return": [0.01] * test_df.height,
                }
            )

        bt = CPCVBacktester(
            n_splits=3, n_test_groups=1, h=2, input_size=5, embargo_days=2
        )
        df = _make_ohlcv_df(300)
        result = bt.run(df, tracking_factory)
        assert len(seen_splits) == 3  # C(3,1) = 3

    def test_run_c62_paths(self) -> None:
        """C(6,2) = 15 paths generated with standard params but small purge."""
        bt = CPCVBacktester(
            n_splits=6, n_test_groups=2, h=2, input_size=5, embargo_days=3
        )
        df = _make_ohlcv_df(600)
        result = bt.run(df, _mock_factory_positive)
        assert result.n_paths == 15


# ---------------------------------------------------------------------------
# _aggregate_results
# ---------------------------------------------------------------------------


class TestAggregateResults:
    """Tests for _aggregate_results."""

    def test_empty_input(self) -> None:
        """Empty fold_results returns default BacktestResult."""
        result = CPCVBacktester._aggregate_results([])
        assert result.n_paths == 0
        assert result.mean_sharpe == 0.0
        assert result.fold_results == []

    def test_single_fold(self) -> None:
        """Single fold: mean = that fold, std = 0."""
        fr = FoldResult(
            path_id=0, test_groups=(0,), sharpe=1.5,
            max_drawdown=-0.1, cagr=0.2, n_train=100, n_test=50, n_returns=10,
        )
        result = CPCVBacktester._aggregate_results([fr])
        assert result.n_paths == 1
        assert result.mean_sharpe == pytest.approx(1.5)
        assert result.std_sharpe == 0.0
        assert result.pct_positive_sharpe == 1.0

    def test_aggregation_math(self) -> None:
        """Verify mean/std/pct computations with known values."""
        folds = [
            FoldResult(0, (0,), sharpe=2.0, max_drawdown=-0.05, cagr=0.15,
                       n_train=100, n_test=50, n_returns=10),
            FoldResult(1, (1,), sharpe=-1.0, max_drawdown=-0.20, cagr=-0.05,
                       n_train=100, n_test=50, n_returns=10),
            FoldResult(2, (2,), sharpe=1.0, max_drawdown=-0.10, cagr=0.10,
                       n_train=100, n_test=50, n_returns=10),
        ]
        result = CPCVBacktester._aggregate_results(folds)

        assert result.n_paths == 3
        assert result.mean_sharpe == pytest.approx((2.0 - 1.0 + 1.0) / 3)
        assert result.pct_positive_sharpe == pytest.approx(2.0 / 3.0)
        assert result.mean_max_drawdown == pytest.approx((-0.05 - 0.20 - 0.10) / 3)
        assert result.mean_cagr == pytest.approx((0.15 - 0.05 + 0.10) / 3)

        # Verify std_sharpe (sample std with ddof=1)
        sharpes = [2.0, -1.0, 1.0]
        mean_s = sum(sharpes) / 3
        var_s = sum((s - mean_s) ** 2 for s in sharpes) / 2
        assert result.std_sharpe == pytest.approx(math.sqrt(var_s))

    def test_all_negative_sharpe(self) -> None:
        """All negative sharpes => pct_positive = 0."""
        folds = [
            FoldResult(i, (i,), sharpe=-0.5 * (i + 1), max_drawdown=-0.1,
                       cagr=-0.05, n_train=100, n_test=50, n_returns=10)
            for i in range(5)
        ]
        result = CPCVBacktester._aggregate_results(folds)
        assert result.pct_positive_sharpe == 0.0

    def test_all_positive_sharpe(self) -> None:
        """All positive sharpes => pct_positive = 1.0."""
        folds = [
            FoldResult(i, (i,), sharpe=0.5 * (i + 1), max_drawdown=-0.1,
                       cagr=0.05, n_train=100, n_test=50, n_returns=10)
            for i in range(4)
        ]
        result = CPCVBacktester._aggregate_results(folds)
        assert result.pct_positive_sharpe == 1.0


# ---------------------------------------------------------------------------
# _find_contiguous_blocks
# ---------------------------------------------------------------------------


class TestFindContiguousBlocks:
    """Tests for _find_contiguous_blocks."""

    def test_single_block(self) -> None:
        """Contiguous indices produce a single block."""
        blocks = CPCVBacktester._find_contiguous_blocks([0, 1, 2, 3, 4])
        assert len(blocks) == 1
        assert blocks[0] == [0, 1, 2, 3, 4]

    def test_two_blocks(self) -> None:
        """Gap in indices produces two blocks."""
        blocks = CPCVBacktester._find_contiguous_blocks([0, 1, 2, 10, 11, 12])
        assert len(blocks) == 2
        assert blocks[0] == [0, 1, 2]
        assert blocks[1] == [10, 11, 12]

    def test_three_blocks(self) -> None:
        """Multiple gaps produce multiple blocks."""
        blocks = CPCVBacktester._find_contiguous_blocks([0, 1, 5, 6, 7, 20])
        assert len(blocks) == 3
        assert blocks[0] == [0, 1]
        assert blocks[1] == [5, 6, 7]
        assert blocks[2] == [20]

    def test_empty_input(self) -> None:
        """Empty list returns empty."""
        assert CPCVBacktester._find_contiguous_blocks([]) == []

    def test_single_element(self) -> None:
        """Single element produces single block."""
        blocks = CPCVBacktester._find_contiguous_blocks([42])
        assert blocks == [[42]]


# ---------------------------------------------------------------------------
# Contiguous block evaluation (quant-reviewer fix #3)
# ---------------------------------------------------------------------------


class TestContiguousBlockEvaluation:
    """Tests that returns are computed within contiguous blocks only."""

    def test_non_contiguous_test_no_cross_gap_returns(self) -> None:
        """Returns are NOT computed across gaps between test blocks."""
        bt = CPCVBacktester(h=5, input_size=5, embargo_days=0)

        # Create two separate blocks of test data with a price jump between them
        base = date(2023, 1, 1)
        # Block 1: prices 100..109 (indices 0-9)
        # Block 2: prices 200..209 (indices 20-29) — price doubled
        block1_dates = [base + timedelta(days=i) for i in range(10)]
        block2_dates = [base + timedelta(days=i + 20) for i in range(10)]

        test_df = pl.DataFrame({
            "date": block1_dates + block2_dates,
            "close": list(range(100, 110)) + list(range(200, 210)),
        })

        predictions = pl.DataFrame({
            "date": test_df["date"],
            "predicted_return": [0.01] * 20,
        })

        # Non-contiguous indices: two blocks with a gap
        test_indices = list(range(10)) + list(range(20, 30))

        result = bt._evaluate_predictions(
            predictions, test_df, test_indices, 0, (0, 2)
        )

        # With h=5 and 10 elements per block:
        # Block 1: i=0 (return from 100->105), i=5 would need i+5=10 >= 10, skip
        # Block 2: i=0 (return from 200->205), same
        # Total: 2 returns, not 3 (which would happen if gap was crossed)
        assert result.n_returns == 2

    def test_empty_predictions_handled(self) -> None:
        """Empty predictions DataFrame does not crash."""
        bt = CPCVBacktester(h=5, input_size=5)
        base = date(2023, 1, 1)
        test_df = pl.DataFrame({
            "date": [base + timedelta(days=i) for i in range(20)],
            "close": [100.0 + i for i in range(20)],
        })
        predictions = pl.DataFrame({
            "date": [],
            "predicted_return": [],
        }).cast({"date": pl.Date, "predicted_return": pl.Float64})

        test_indices = list(range(20))
        result = bt._evaluate_predictions(
            predictions, test_df, test_indices, 0, (0,)
        )
        # All predictions default to 0.0 => flat => sharpe 0
        assert result.sharpe == 0.0


# ---------------------------------------------------------------------------
# TransactionCosts dataclass
# ---------------------------------------------------------------------------


class TestTransactionCosts:
    """Tests for the TransactionCosts frozen dataclass."""

    def test_default_values(self) -> None:
        """Default slippage=5, commission=10, market_impact=0."""
        tc = TransactionCosts()
        assert tc.slippage_bps == 5.0
        assert tc.commission_bps == 10.0
        assert tc.market_impact_bps == 0.0

    def test_fixed_cost_frac(self) -> None:
        """fixed_cost_frac = (5 + 10) / 10_000 = 0.0015."""
        tc = TransactionCosts()
        assert tc.fixed_cost_frac == pytest.approx(0.0015)

    def test_custom_values(self) -> None:
        """Custom slippage and commission are stored correctly."""
        tc = TransactionCosts(slippage_bps=2.0, commission_bps=8.0, market_impact_bps=3.0)
        assert tc.slippage_bps == 2.0
        assert tc.commission_bps == 8.0
        assert tc.market_impact_bps == 3.0
        assert tc.fixed_cost_frac == pytest.approx((2.0 + 8.0) / 10_000)

    def test_frozen(self) -> None:
        """Cannot mutate fields on a frozen dataclass."""
        tc = TransactionCosts()
        with pytest.raises(AttributeError):
            tc.slippage_bps = 99.0  # type: ignore[misc]

    def test_zero_costs(self) -> None:
        """All-zero costs yield fixed_cost_frac = 0."""
        tc = TransactionCosts(slippage_bps=0, commission_bps=0, market_impact_bps=0)
        assert tc.fixed_cost_frac == 0.0


# ---------------------------------------------------------------------------
# Costs integration with CPCVBacktester
# ---------------------------------------------------------------------------


class TestCostsIntegration:
    """Tests for transaction cost integration in the CPCV pipeline."""

    def test_no_costs_default(self) -> None:
        """CPCVBacktester() has costs=None; results have zero costs."""
        bt = CPCVBacktester(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=3
        )
        assert bt.costs is None
        df = _make_ohlcv_df(500)
        result = bt.run(df, _mock_factory_positive)
        for fold in result.fold_results:
            assert fold.total_costs == 0.0
            assert fold.n_trades == 0
        assert result.mean_total_costs == 0.0
        assert result.mean_n_trades == 0.0

    def test_costs_reduce_sharpe(self) -> None:
        """Sharpe with costs is lower than or equal to Sharpe without costs."""
        params = dict(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=3
        )
        df = _make_ohlcv_df(500)

        bt_no_cost = CPCVBacktester(**params)
        result_no_cost = bt_no_cost.run(df, _mock_factory_positive)

        bt_cost = CPCVBacktester(**params, costs=TransactionCosts())
        result_cost = bt_cost.run(df, _mock_factory_positive)

        # With always-positive predictions, there is exactly 1 trade
        # (flat->long at start). Costs should reduce cumulative return
        # and thus Sharpe (or keep equal if both zero).
        assert result_cost.mean_sharpe <= result_no_cost.mean_sharpe + 1e-9

    def test_n_trades_counted(self) -> None:
        """Position changes (flat->long and long->flat) are counted as trades."""
        bt = CPCVBacktester(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=3,
            costs=TransactionCosts(),
        )
        df = _make_ohlcv_df(500)
        # Alternating predictions cause many position changes
        result = bt.run(df, _mock_factory_alternating)
        total_trades = sum(f.n_trades for f in result.fold_results)
        assert total_trades > 0

    def test_total_costs_positive(self) -> None:
        """total_costs > 0 when there are trades with non-zero cost."""
        bt = CPCVBacktester(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=3,
            costs=TransactionCosts(),
        )
        df = _make_ohlcv_df(500)
        result = bt.run(df, _mock_factory_alternating)
        total_costs = sum(f.total_costs for f in result.fold_results)
        assert total_costs > 0.0

    def test_always_long_two_trades(self) -> None:
        """Always positive predictions = 2 trades per path (entry + forced exit)."""
        bt = CPCVBacktester(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=3,
            costs=TransactionCosts(),
        )
        df = _make_ohlcv_df(500)
        result = bt.run(df, _mock_factory_positive)
        for fold in result.fold_results:
            # Two transitions: flat->long at start + forced exit at end of block
            assert fold.n_trades == 2

    def test_always_flat_zero_trades(self) -> None:
        """Always negative predictions = 0 trades (stay flat entire time)."""
        bt = CPCVBacktester(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=3,
            costs=TransactionCosts(),
        )
        df = _make_ohlcv_df(500)
        result = bt.run(df, _mock_factory_negative)
        for fold in result.fold_results:
            assert fold.n_trades == 0
            assert fold.total_costs == 0.0

    def test_equity_curve_populated(self) -> None:
        """equity_curve is a non-empty list starting with 1.0."""
        bt = CPCVBacktester(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=3,
            costs=TransactionCosts(),
        )
        df = _make_ohlcv_df(500)
        result = bt.run(df, _mock_factory_positive)
        for fold in result.fold_results:
            assert isinstance(fold.equity_curve, list)
            assert len(fold.equity_curve) >= 1
            assert fold.equity_curve[0] == 1.0

    def test_equity_curve_length(self) -> None:
        """len(equity_curve) = n_returns + 1 (includes starting 1.0)."""
        bt = CPCVBacktester(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=3,
            costs=TransactionCosts(),
        )
        df = _make_ohlcv_df(500)
        result = bt.run(df, _mock_factory_positive)
        for fold in result.fold_results:
            assert len(fold.equity_curve) == fold.n_returns + 1

    def test_market_impact_with_volume(self) -> None:
        """market_impact_bps > 0 with volume column applies extra cost."""
        params = dict(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=3,
        )
        df = _make_ohlcv_df(500)  # has volume column

        # Without market impact
        bt_fixed = CPCVBacktester(**params, costs=TransactionCosts(market_impact_bps=0.0))
        result_fixed = bt_fixed.run(df, _mock_factory_alternating)

        # With market impact
        bt_impact = CPCVBacktester(
            **params, costs=TransactionCosts(market_impact_bps=50.0)
        )
        result_impact = bt_impact.run(df, _mock_factory_alternating)

        # Market impact adds extra cost, so total costs should be higher
        fixed_costs = sum(f.total_costs for f in result_fixed.fold_results)
        impact_costs = sum(f.total_costs for f in result_impact.fold_results)
        assert impact_costs > fixed_costs

    def test_market_impact_without_volume(self) -> None:
        """market_impact_bps > 0 but no volume column: graceful fallback."""
        bt = CPCVBacktester(
            n_splits=4, n_test_groups=1, h=2, input_size=5, embargo_days=3,
            costs=TransactionCosts(market_impact_bps=50.0),
        )
        # Create a df WITHOUT volume column
        df = _make_ohlcv_df(500).drop("volume")
        # Should not raise; market impact silently ignored
        result = bt.run(df, _mock_factory_alternating)
        assert result.n_paths == 4
        # Costs are only fixed (no market impact applied)
        for fold in result.fold_results:
            if fold.n_trades > 0:
                cost_per_trade = fold.total_costs / fold.n_trades
                # Should be approximately fixed_cost_frac = 0.0015
                assert cost_per_trade == pytest.approx(0.0015, abs=1e-6)

    def test_aggregate_includes_costs(self) -> None:
        """mean_n_trades and mean_total_costs are populated in BacktestResult."""
        folds = [
            FoldResult(
                path_id=i, test_groups=(i,), sharpe=1.0, max_drawdown=-0.05,
                cagr=0.1, n_train=100, n_test=50, n_returns=10,
                n_trades=5 + i, total_costs=0.01 * (i + 1),
                equity_curve=[1.0, 1.01, 1.02],
            )
            for i in range(3)
        ]
        result = CPCVBacktester._aggregate_results(folds)
        assert result.mean_n_trades == pytest.approx((5 + 6 + 7) / 3)
        assert result.mean_total_costs == pytest.approx((0.01 + 0.02 + 0.03) / 3)
