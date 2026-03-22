"""Combinatorial Purged Cross-Validation (CPCV) backtester.

Implements the CPCV methodology formalised by Marcos Lopez de Prado to
validate time-series prediction models without look-ahead bias.

Key features:

- **Combinatorial splits**: generates ``C(n_splits, n_test_groups)`` distinct
  train/test paths from *n_splits* contiguous temporal groups.
- **Purging**: removes from the training set any samples whose forward label
  (of length ``h``) overlaps with the test period.
- **Extended purge**: also removes ``input_size - 1`` additional samples so
  that the model's look-back window cannot peek into the test data.
- **Embargo**: adds a gap of ``embargo_days`` trading days after each test
  block before resuming training.
- **Non-overlapping returns**: evaluates every ``h`` days to prevent
  autocorrelation from inflating the Sharpe ratio.

Usage::

    from src.backtest.cpcv import CPCVBacktester

    backtester = CPCVBacktester(n_splits=6, embargo_days=10)
    result = backtester.run(df, model_factory=my_factory)
    print(result.mean_sharpe, result.pct_positive_sharpe)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Protocol

import polars as pl
from loguru import logger


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransactionCosts:
    """Per-trade cost components (one-way, as fraction of trade value).

    All values are in basis points (1 bp = 0.01%).  The total fixed cost
    per trade (entry or exit) is ``(slippage_bps + commission_bps) / 10_000``.

    Attributes:
        slippage_bps: Bid-ask spread cost per trade.
        commission_bps: Brokerage fee per trade.
        market_impact_bps: Volume-dependent impact cost.  Applied only
            when a ``volume`` column is present in the test data.
    """

    slippage_bps: float = 5.0
    commission_bps: float = 10.0
    market_impact_bps: float = 0.0

    @property
    def fixed_cost_frac(self) -> float:
        """Total fixed cost per trade as a decimal fraction."""
        return (self.slippage_bps + self.commission_bps) / 10_000


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------


class ModelFactory(Protocol):
    """Callable that trains a model and returns predictions on the test set.

    Args:
        train_df: Training data (OHLCV + features).
        test_df: Test data (OHLCV + features).

    Returns:
        DataFrame with at least ``date`` and ``predicted_return`` columns.
    """

    def __call__(
        self, train_df: pl.DataFrame, test_df: pl.DataFrame
    ) -> pl.DataFrame: ...


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Metrics for a single CPCV train/test path.

    Attributes:
        path_id: Sequential identifier for this path.
        test_groups: Tuple of group indices used as the test set.
        sharpe: Annualised Sharpe ratio (excess return / volatility).
        max_drawdown: Maximum peak-to-trough drawdown (negative value).
        cagr: Compound annual growth rate.
        n_train: Number of training samples after purge + embargo.
        n_test: Number of test samples.
        n_returns: Number of non-overlapping returns used for Sharpe.
    """

    path_id: int
    test_groups: tuple[int, ...]
    sharpe: float
    max_drawdown: float
    cagr: float
    n_train: int
    n_test: int
    n_returns: int
    n_trades: int = 0
    total_costs: float = 0.0
    equity_curve: list[float] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Aggregated results across all CPCV paths.

    Attributes:
        fold_results: Per-path metrics.
        n_paths: Total number of combinatorial paths evaluated.
        mean_sharpe: Mean Sharpe ratio across paths.
        std_sharpe: Standard deviation of Sharpe across paths.
        pct_positive_sharpe: Fraction of paths with Sharpe > 0.
        mean_max_drawdown: Mean maximum drawdown across paths.
        mean_cagr: Mean CAGR across paths.
    """

    fold_results: list[FoldResult] = field(default_factory=list)
    n_paths: int = 0
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    pct_positive_sharpe: float = 0.0
    mean_max_drawdown: float = 0.0
    mean_cagr: float = 0.0
    mean_n_trades: float = 0.0
    mean_total_costs: float = 0.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ANNUALISATION_FACTOR = 252
_MIN_RETURNS_WARNING = 20


# ---------------------------------------------------------------------------
# CPCVBacktester
# ---------------------------------------------------------------------------


class CPCVBacktester:
    """Combinatorial Purged Cross-Validation backtester.

    Args:
        n_splits: Number of contiguous temporal groups to partition the
            data into.
        n_test_groups: Number of groups used as the test set in each
            combinatorial path.  Defaults to 2 (i.e. ``C(n_splits, 2)``
            paths).
        embargo_days: Number of trading days to skip after each test
            block before resuming training.
        h: Forecast horizon in trading days.  Must match the model's
            prediction horizon.
        input_size: Model look-back window.  Used to extend the purge
            zone so the model's input window cannot overlap with test
            data.
        rf: Risk-free rate (annualised) for Sharpe ratio computation.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_groups: int = 2,
        embargo_days: int = 10,
        h: int = 5,
        input_size: int = 60,
        rf: float = 0.05,
        costs: TransactionCosts | None = None,
    ) -> None:
        if n_splits < 3:
            raise ValueError(f"n_splits must be >= 3, got {n_splits}")
        if n_test_groups < 1 or n_test_groups >= n_splits:
            raise ValueError(
                f"n_test_groups must be in [1, {n_splits - 1}], "
                f"got {n_test_groups}"
            )
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")
        if h < 1:
            raise ValueError(f"h must be >= 1, got {h}")

        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_days = embargo_days
        self.h = h
        self.input_size = input_size
        self.rf = rf
        self.costs = costs

        self._purge_window = h + input_size - 1

        n_paths = math.comb(n_splits, n_test_groups)
        logger.info(
            "CPCVBacktester: n_splits={}, n_test_groups={}, "
            "embargo_days={}, h={}, input_size={}, purge_window={}, "
            "n_paths={}",
            n_splits,
            n_test_groups,
            embargo_days,
            h,
            input_size,
            self._purge_window,
            n_paths,
        )

    # ------------------------------------------------------------------
    # Split generation
    # ------------------------------------------------------------------

    def _split_into_groups(
        self, n_samples: int
    ) -> list[tuple[int, int]]:
        """Partition *n_samples* into ``n_splits`` contiguous groups.

        Args:
            n_samples: Total number of rows in the dataset.

        Returns:
            List of ``(start_idx, end_idx)`` pairs (end exclusive).
        """
        group_size = n_samples // self.n_splits
        remainder = n_samples % self.n_splits

        groups: list[tuple[int, int]] = []
        start = 0
        for i in range(self.n_splits):
            size = group_size + (1 if i < remainder else 0)
            groups.append((start, start + size))
            start += size
        return groups

    def _apply_purge(
        self,
        train_indices: set[int],
        test_start: int,
        test_end: int,
    ) -> set[int]:
        """Remove training indices whose forward label overlaps the test set.

        The purge zone extends ``_purge_window`` days before ``test_start``
        so that neither the model's input window (``input_size``) nor
        the label (``h``) can leak into the test period.

        Args:
            train_indices: Set of training row indices.
            test_start: First index of the test block.
            test_end: Last index (exclusive) of the test block.

        Returns:
            Pruned set of training indices.
        """
        purge_start = max(0, test_start - self._purge_window)
        purge_end = test_start
        purge_zone = set(range(purge_start, purge_end))
        return train_indices - purge_zone

    def _apply_embargo(
        self,
        train_indices: set[int],
        test_end: int,
        n_samples: int,
    ) -> set[int]:
        """Remove ``embargo_days`` indices after the test block.

        Args:
            train_indices: Set of training row indices.
            test_end: First index after the test block.
            n_samples: Total dataset size (for bounds checking).

        Returns:
            Pruned set of training indices.
        """
        embargo_end = min(n_samples, test_end + self.embargo_days)
        embargo_zone = set(range(test_end, embargo_end))
        return train_indices - embargo_zone

    def generate_paths(
        self, n_samples: int
    ) -> list[tuple[list[int], list[int], tuple[int, ...]]]:
        """Generate all combinatorial train/test paths with purge + embargo.

        Args:
            n_samples: Total number of rows in the dataset.

        Returns:
            List of ``(train_indices, test_indices, test_group_ids)``
            tuples sorted by path id.

        Raises:
            ValueError: If the dataset is too small for the requested
                number of splits and purge window.
        """
        groups = self._split_into_groups(n_samples)
        min_group_size = min(end - start for start, end in groups)

        if min_group_size < self._purge_window + self.embargo_days:
            raise ValueError(
                f"Dataset too small: smallest group has {min_group_size} "
                f"samples but purge_window={self._purge_window} + "
                f"embargo_days={self.embargo_days} = "
                f"{self._purge_window + self.embargo_days} needed. "
                f"Provide more data or reduce n_splits."
            )

        all_indices = set(range(n_samples))
        paths: list[tuple[list[int], list[int], tuple[int, ...]]] = []

        for test_combo in combinations(range(self.n_splits), self.n_test_groups):
            # Build test indices from selected groups
            test_indices: set[int] = set()
            for g in test_combo:
                start, end = groups[g]
                test_indices.update(range(start, end))

            # Start with all non-test indices as training
            train_indices = all_indices - test_indices

            # Apply purge and embargo for each test block
            for g in test_combo:
                start, end = groups[g]
                train_indices = self._apply_purge(train_indices, start, end)
                train_indices = self._apply_embargo(
                    train_indices, end, n_samples
                )

            train_sorted = sorted(train_indices)
            test_sorted = sorted(test_indices)

            paths.append((train_sorted, test_sorted, test_combo))

        logger.info(
            "Generated {} CPCV paths from {} groups ({} samples)",
            len(paths),
            len(groups),
            n_samples,
        )
        return paths

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sharpe(
        returns: list[float],
        rf: float,
        periods_per_year: float,
    ) -> float:
        """Compute annualised Sharpe ratio from a list of periodic returns.

        Args:
            returns: List of non-overlapping periodic returns.
            rf: Annualised risk-free rate.
            periods_per_year: Number of return periods in a year
                (e.g. 252/h for h-day returns).

        Returns:
            Annualised Sharpe ratio.  Returns 0.0 if volatility is zero.
        """
        if len(returns) < 2:
            return 0.0

        # CORREÇÃO: Equivalência geométrica em vez de divisão linear
        rf_per_period = (1.0 + rf) ** (1.0 / periods_per_year) - 1.0
        excess = [r - rf_per_period for r in returns]

        mean_excess = sum(excess) / len(excess)
        var = sum((e - mean_excess) ** 2 for e in excess) / (len(excess) - 1)
        std = math.sqrt(var) if var > 0 else 0.0

        if std == 0.0:
            return 0.0

        return (mean_excess / std) * math.sqrt(periods_per_year)

    @staticmethod
    def _compute_max_drawdown(cumulative_returns: list[float]) -> float:
        """Compute maximum drawdown from a cumulative return series.

        Args:
            cumulative_returns: List of cumulative (compounded) return values,
                starting from 1.0.

        Returns:
            Maximum drawdown as a negative float (e.g. -0.15 for 15% DD).
            Returns 0.0 if the series is empty or monotonically increasing.
        """
        if not cumulative_returns:
            return 0.0

        peak = cumulative_returns[0]
        max_dd = 0.0
        for val in cumulative_returns:
            if val > peak:
                peak = val
            dd = (val - peak) / peak if peak != 0 else 0.0
            if dd < max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _compute_cagr(
        cumulative_returns: list[float],
        n_days: int,
    ) -> float:
        """Compute compound annual growth rate.

        Args:
            cumulative_returns: Cumulative return series starting from 1.0.
            n_days: Number of trading days in the period.

        Returns:
            CAGR as a decimal (e.g. 0.12 for 12% annual).
        """
        if not cumulative_returns or n_days <= 0:
            return 0.0

        final = cumulative_returns[-1]
        if final <= 0:
            return -1.0

        years = n_days / _ANNUALISATION_FACTOR
        if years <= 0:
            return 0.0

        return final ** (1.0 / years) - 1.0

    @staticmethod
    def _find_contiguous_blocks(
        indices: list[int],
    ) -> list[list[int]]:
        """Split a sorted list of indices into contiguous blocks.

        Args:
            indices: Sorted list of integer indices.

        Returns:
            List of lists, each containing a contiguous run of indices.
        """
        if not indices:
            return []

        blocks: list[list[int]] = [[indices[0]]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                blocks[-1].append(indices[i])
            else:
                blocks.append([indices[i]])
        return blocks

    def _evaluate_predictions(
        self,
        predictions: pl.DataFrame,
        test_df: pl.DataFrame,
        test_indices: list[int],
        path_id: int,
        test_groups: tuple[int, ...],
    ) -> FoldResult:
        """Evaluate model predictions against test actuals.

        Computes non-overlapping h-day returns from a simple long/flat
        strategy: go long when ``predicted_return > 0``, stay flat
        otherwise.  Returns are computed **within each contiguous test
        block** to avoid spurious returns that span temporal gaps.

        When ``self.costs`` is set, transaction costs (slippage +
        commission + optional market impact) are deducted on each
        position change (flat→long or long→flat).

        Args:
            predictions: DataFrame with ``date`` and ``predicted_return``.
            test_df: Test slice with ``date`` and ``close`` columns.
            test_indices: Original sorted test indices for block detection.
            path_id: Path identifier for the result.
            test_groups: Group indices forming the test set.

        Returns:
            FoldResult with computed metrics.
        """
        test_sorted = test_df.sort("date")
        n_test = test_sorted.height

        # Build date → predicted_return lookup
        pred_map: dict = {}
        if predictions.height > 0:
            for row in predictions.iter_rows(named=True):
                pred_map[row["date"]] = row["predicted_return"]
        else:
            logger.warning(
                "Path {}: model_factory returned empty predictions",
                path_id + 1,
            )

        # Transaction cost setup
        has_costs = self.costs is not None
        fixed_cost = self.costs.fixed_cost_frac if has_costs else 0.0
        has_volume = "volume" in test_sorted.columns
        has_market_impact = (
            has_costs
            and self.costs is not None
            and self.costs.market_impact_bps > 0
            and has_volume
        )

        if has_costs and self.costs is not None and self.costs.market_impact_bps > 0 and not has_volume:
            logger.info(
                "Path {}: market_impact_bps={} but no 'volume' column; "
                "market impact ignored",
                path_id + 1,
                self.costs.market_impact_bps,
            )

        # Precompute average volume for market impact scaling
        avg_volume = 0.0
        if has_market_impact:
            vol_series = test_sorted["volume"].cast(pl.Float64)
            avg_volume = vol_series.mean() or 1.0  # type: ignore[assignment]

        # Split test indices into contiguous blocks to avoid computing
        # returns across temporal gaps (quant-reviewer issue #3).
        blocks = self._find_contiguous_blocks(test_indices)

        strategy_returns: list[float] = []
        cumulative: list[float] = [1.0]
        total_days = 0
        row_offset = 0
        n_trades = 0
        total_cost_sum = 0.0
        prev_position = 0  # 0=flat, 1=long

        # Taxa livre de risco proporcional ao horizonte h
        periods_per_year = _ANNUALISATION_FACTOR / self.h
        rf_per_period = (1.0 + self.rf) ** (1.0 / periods_per_year) - 1.0

        for block in blocks:
            block_len = len(block)
            block_slice = test_sorted.slice(row_offset, block_len)
            row_offset += block_len

            closes = block_slice["close"].to_list()
            dates = block_slice["date"].to_list()
            volumes = (
                block_slice["volume"].cast(pl.Float64).to_list()
                if has_market_impact
                else None
            )
            total_days += block_len

            # Reset position at each block boundary — between
            # non-contiguous blocks the position is implicitly closed
            # (quant-reviewer fix #1).
            prev_position = 0

            i = 0
            while i + self.h < len(closes):
                entry_date = dates[i]
                pred_ret = pred_map.get(entry_date, 0.0)
                signal = 1 if pred_ret > 0 else 0

                actual_return = (closes[i + self.h] - closes[i]) / closes[i]

                # Compute trade cost on position change
                trade_cost = 0.0
                if has_costs and signal != prev_position:
                    trade_cost = fixed_cost
                    if has_market_impact and volumes is not None and avg_volume > 0:
                        relative_vol = volumes[i] / avg_volume
                        # Impact inversely proportional to liquidity:
                        # high volume → lower impact (quant-reviewer fix #3).
                        impact = (self.costs.market_impact_bps / 10_000) / math.sqrt(max(relative_vol, 1e-9))  # type: ignore[union-attr]
                        trade_cost += impact
                    n_trades += 1
                    total_cost_sum += trade_cost

                # Se comprado, ganha o retorno do ativo. Se flat, ganha a taxa livre de risco.
                gross_return = actual_return if signal == 1 else rf_per_period
                strategy_return = gross_return - trade_cost
                strategy_returns.append(strategy_return)
                cumulative.append(cumulative[-1] * (1.0 + strategy_return))

                prev_position = signal
                i += self.h  # Non-overlapping
            
            # NOVO: Se o bloco terminou posicionado, cobra o custo da saída forçada
            if has_costs and prev_position != 0:
                trade_cost = fixed_cost
                if has_market_impact and volumes is not None and avg_volume > 0:
                    # Estima impacto usando o volume da entrada correspondente
                    last_vol = volumes[max(0, i - self.h)] 
                    relative_vol = last_vol / avg_volume
                    impact = (self.costs.market_impact_bps / 10_000) / math.sqrt(max(relative_vol, 1e-9))
                    trade_cost += impact
                
                n_trades += 1
                total_cost_sum += trade_cost
                
                # Desconta o custo do último retorno gerado no bloco
                if strategy_returns:
                    strategy_returns[-1] -= trade_cost
                    # Recalcula a última equidade
                    prev_cum = cumulative[-2] if len(cumulative) > 1 else 1.0
                    cumulative[-1] = prev_cum * (1.0 + strategy_returns[-1])

        if len(strategy_returns) < _MIN_RETURNS_WARNING:
            logger.warning(
                "Path {}: only {} non-overlapping returns (< {} recommended)",
                path_id,
                len(strategy_returns),
                _MIN_RETURNS_WARNING,
            )

        periods_per_year = _ANNUALISATION_FACTOR / self.h
        sharpe = self._compute_sharpe(
            strategy_returns, self.rf, periods_per_year
        )
        max_dd = self._compute_max_drawdown(cumulative)
        cagr = self._compute_cagr(cumulative, total_days)

        return FoldResult(
            path_id=path_id,
            test_groups=test_groups,
            sharpe=sharpe,
            max_drawdown=max_dd,
            cagr=cagr,
            n_train=0,  # Filled by caller
            n_test=n_test,
            n_returns=len(strategy_returns),
            n_trades=n_trades,
            total_costs=total_cost_sum,
            equity_curve=cumulative,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pl.DataFrame,
        model_factory: Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame],
    ) -> BacktestResult:
        """Run CPCV across all combinatorial paths.

        For each path, trains the model on the purged training set and
        evaluates on the test set using non-overlapping h-day returns.

        Args:
            df: Full OHLCV+features DataFrame for a **single ticker**,
                sorted by date.  Must contain at least ``date`` and
                ``close`` columns.
            model_factory: Callable ``(train_df, test_df) -> predictions_df``.
                The returned DataFrame must have ``date`` and
                ``predicted_return`` columns.

        Returns:
            BacktestResult with per-path metrics and aggregate statistics.

        Raises:
            ValueError: If required columns are missing or the dataset
                is too small.
        """
        required_cols = {"date", "close"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df_sorted = df.sort("date").drop_nulls(subset=["close"])
        n_samples = df_sorted.height

        paths = self.generate_paths(n_samples)
        fold_results: list[FoldResult] = []

        for path_id, (train_idx, test_idx, test_groups) in enumerate(paths):
            logger.info(
                "Path {}/{}: test_groups={}, train={}, test={}",
                path_id + 1,
                len(paths),
                test_groups,
                len(train_idx),
                len(test_idx),
            )

            train_df = df_sorted[train_idx]
            test_df = df_sorted[test_idx]

            predictions = model_factory(train_df, test_df)

            fold = self._evaluate_predictions(
                predictions, test_df, test_idx, path_id, test_groups
            )
            fold.n_train = len(train_idx)
            fold_results.append(fold)

            logger.info(
                "Path {} result: Sharpe={:.3f}, MaxDD={:.3f}, "
                "CAGR={:.3f}, n_returns={}",
                path_id + 1,
                fold.sharpe,
                fold.max_drawdown,
                fold.cagr,
                fold.n_returns,
            )

        result = self._aggregate_results(fold_results)
        logger.info(
            "CPCV complete: mean_Sharpe={:.3f} +/- {:.3f}, "
            "positive_Sharpe={:.1%}, mean_MaxDD={:.3f}",
            result.mean_sharpe,
            result.std_sharpe,
            result.pct_positive_sharpe,
            result.mean_max_drawdown,
        )
        return result

    @staticmethod
    def _aggregate_results(
        fold_results: list[FoldResult],
    ) -> BacktestResult:
        """Aggregate per-path metrics into a BacktestResult.

        Args:
            fold_results: List of FoldResult from each path.

        Returns:
            BacktestResult with summary statistics.
        """
        if not fold_results:
            return BacktestResult()

        sharpes = [f.sharpe for f in fold_results]
        drawdowns = [f.max_drawdown for f in fold_results]
        cagrs = [f.cagr for f in fold_results]
        trades = [f.n_trades for f in fold_results]
        costs = [f.total_costs for f in fold_results]

        n = len(sharpes)
        mean_sharpe = sum(sharpes) / n
        var_sharpe = (
            sum((s - mean_sharpe) ** 2 for s in sharpes) / (n - 1)
            if n > 1
            else 0.0
        )
        std_sharpe = math.sqrt(var_sharpe)
        pct_positive = sum(1 for s in sharpes if s > 0) / n

        return BacktestResult(
            fold_results=fold_results,
            n_paths=n,
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            pct_positive_sharpe=pct_positive,
            mean_max_drawdown=sum(drawdowns) / n,
            mean_cagr=sum(cagrs) / n,
            mean_n_trades=sum(trades) / n,
            mean_total_costs=sum(costs) / n,
        )
