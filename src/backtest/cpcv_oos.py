"""CPCV-OOS parameter validator for walk-forward strategy validation.

Applies Combinatorial Purged Cross-Validation to the **walk-forward
backtester** itself, producing a distribution of Sharpe ratios across
``C(n_splits, n_test_groups)`` non-overlapping temporal paths.  This
prevents overfitting when tuning strategy parameters: a configuration
is accepted only if it shows consistent positive Sharpe across most
paths *and* survives the Deflated Sharpe Ratio correction for multiple
testing.

Key concepts:

- **Splits**: the OOS period is divided into ``n_splits`` contiguous
  temporal blocks.
- **Paths**: each combination of ``n_test_groups`` blocks is a test
  set; the remaining blocks form the calibration set.
- **Embargo**: a configurable gap between adjacent train/test blocks
  to prevent information leakage.
- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014): adjusts
  the observed Sharpe for the number of configurations tried, producing
  a p-value that guards against selection bias.

Usage::

    from src.backtest.cpcv_oos import CPCVParameterValidator

    validator = CPCVParameterValidator(ohlcv=df, tickers=tickers)
    result = validator.validate(config=my_config, model_factory=factory)
    print(result.accepted, result.deflated_sharpe)

References:
    Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe
    Ratio." *Journal of Portfolio Management*.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations
from typing import Any

import polars as pl
from loguru import logger

from src.backtest.walk_forward import (
    ModelFactory,
    NaiveModelFactory,
    WalkForwardBacktester,
    WalkForwardConfig,
)


# ---------------------------------------------------------------------------
# Purged model factory wrapper
# ---------------------------------------------------------------------------


class _PurgedModelFactory:
    """Wraps a ModelFactory to exclude test dates during training.

    The predict() method delegates directly — only train() is filtered.
    This enforces CPCV train/test separation without modifying the
    WalkForwardBacktester.

    Args:
        inner: The real model factory to delegate to.
        excluded_dates: Dates that must NOT be used for training
            (test dates + embargo dates from the CPCV path).
    """

    def __init__(
        self, inner: ModelFactory, excluded_dates: set[date]
    ) -> None:
        self._inner = inner
        self._excluded = excluded_dates

    def train(self, df: pl.DataFrame) -> None:
        """Train the inner model on data excluding test/embargo dates."""
        filtered = df.filter(~pl.col("date").is_in(self._excluded))
        if filtered.height == 0:
            logger.warning(
                "_PurgedModelFactory: all training data excluded "
                "({} rows filtered)", df.height,
            )
            return
        self._inner.train(filtered)

    def predict(self, df: pl.DataFrame) -> dict[str, float]:
        """Predict using the inner model (no filtering)."""
        return self._inner.predict(df)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_ACCEPTANCE_PCT = 0.6667  # 10/15 paths for C(6,2)
_DEFAULT_DSR_THRESHOLD = 0.95  # 5% significance level


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_benchmark: float = 0.0,
) -> float:
    """Compute the Deflated Sharpe Ratio p-value.

    Returns the probability that the observed Sharpe ratio is significant
    after correcting for multiple testing.  A **higher** value means the
    Sharpe is more likely genuine (not a result of selection bias).

    The expected maximum Sharpe from ``n_trials`` independent draws is
    estimated via the Euler-Mascheroni approximation for the max of
    standard normal variates.

    Args:
        observed_sharpe: The Sharpe ratio of the selected strategy.
        n_trials: Number of strategy configurations tested.
        n_observations: Number of return observations used to compute
            the Sharpe.
        skewness: Skewness of the return distribution (0 = normal).
        kurtosis: Kurtosis of the return distribution (3 = normal).
        sharpe_benchmark: Reference Sharpe to test against (usually 0).

    Returns:
        P-value in ``[0, 1]``.  Values above 0.95 indicate the Sharpe
        is significant at the 5% level after deflation.
    """
    if n_trials < 1 or n_observations < 2:
        return 0.0

    # Expected maximum Sharpe from n_trials independent normal draws.
    # E[max(SR)] = SR_benchmark + σ(SR) * E[max(Z_1, ..., Z_n)]
    # where E[max(Z)] uses the Euler-Mascheroni approximation:
    # E[max(Z)] ≈ (1 - γ) * Φ^{-1}(1 - 1/n) + γ * Φ^{-1}(1 - 1/(n*e))
    euler_mascheroni = 0.5772156649
    if n_trials == 1:
        e_max_z = 0.0
    else:
        z = _inv_normal_cdf(1.0 - 1.0 / n_trials)
        e_max_z = (
            (1.0 - euler_mascheroni) * z
            + euler_mascheroni * _inv_normal_cdf(1.0 - 1.0 / (n_trials * math.e))
        )

    # Variance of the Sharpe estimator (Lo, 2002)
    # V[SR] ≈ (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / T
    sr = observed_sharpe
    v_sr = (
        1.0
        + 0.5 * sr * sr
        - skewness * sr
        + (kurtosis - 3.0) / 4.0 * sr * sr
    ) / n_observations

    if v_sr <= 0:
        e_max_sr = sharpe_benchmark + e_max_z
        return 1.0 if observed_sharpe > e_max_sr else 0.0

    std_sr = math.sqrt(v_sr)

    # E[max SR] = benchmark + σ(SR) * E[max(Z)]
    e_max_sr = sharpe_benchmark + std_sr * e_max_z

    # PSR = Φ((SR_obs - E[max SR]) / σ(SR))
    z_score = (observed_sharpe - e_max_sr) / std_sr
    return _normal_cdf(z_score)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _inv_normal_cdf(p: float) -> float:
    """Approximate inverse normal CDF (Abramowitz & Stegun, 26.2.23).

    Accurate to ~4.5e-4 for 0.5 < p < 1.  Uses symmetry for p < 0.5.

    Args:
        p: Probability in ``(0, 1)``.

    Returns:
        The z-score corresponding to cumulative probability ``p``.
    """
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    if p < 0.5:
        return -_inv_normal_cdf(1.0 - p)

    # Rational approximation constants
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


# ---------------------------------------------------------------------------
# Sharpe helper
# ---------------------------------------------------------------------------


def _compute_sharpe(
    returns: list[float],
    rf: float = 0.05,
    trading_days: int = 252,
) -> float:
    """Annualised Sharpe from a list of daily simple returns.

    Args:
        returns: Daily simple returns.
        rf: Annual risk-free rate.
        trading_days: Trading days per year.

    Returns:
        Annualised Sharpe ratio.  0.0 if insufficient data.
    """
    n = len(returns)
    if n < 2:
        return 0.0

    rf_daily = rf / trading_days
    excess = [r - rf_daily for r in returns]
    mean_ex = sum(excess) / n
    var = sum((e - mean_ex) ** 2 for e in excess) / (n - 1)
    if var <= 0:
        return 0.0
    return (mean_ex / math.sqrt(var)) * math.sqrt(trading_days)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Output of a single CPCV-OOS validation run.

    Attributes:
        config: The walk-forward configuration tested.
        mean_sharpe: Mean Sharpe across all CPCV paths.
        std_sharpe: Standard deviation of Sharpe across paths.
        pct_positive: Fraction of paths with Sharpe > 0.
        per_path_sharpe: Sharpe ratio for each individual path.
        deflated_sharpe: DSR p-value adjusted for multiple testing.
        p_value: Same as ``deflated_sharpe`` (alias for clarity).
        accepted: Whether the configuration passes acceptance criteria.
        metadata: Additional information (n_paths, path details, etc.).
    """

    config: WalkForwardConfig
    mean_sharpe: float
    std_sharpe: float
    pct_positive: float
    per_path_sharpe: list[float]
    deflated_sharpe: float
    p_value: float
    accepted: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary for JSON output.

        Returns:
            Dictionary with all fields.
        """
        return {
            "mean_sharpe": self.mean_sharpe,
            "std_sharpe": self.std_sharpe,
            "pct_positive": self.pct_positive,
            "per_path_sharpe": self.per_path_sharpe,
            "deflated_sharpe": self.deflated_sharpe,
            "p_value": self.p_value,
            "accepted": self.accepted,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# CPCV-OOS Validator
# ---------------------------------------------------------------------------


class CPCVParameterValidator:
    """Validate walk-forward configurations via CPCV on the OOS period.

    Splits the full OOS data into ``n_splits`` contiguous temporal blocks,
    generates ``C(n_splits, n_test_groups)`` combinatorial paths, and runs
    a walk-forward backtest on each path.  The Sharpe distribution across
    paths is used to assess whether a configuration is robust or overfit.

    Args:
        ohlcv: Full OHLCV data (long format: date, ticker, open, high,
            low, close, volume).
        tickers: Tradeable ticker symbols.
        benchmark_ticker: Buy-and-hold benchmark ticker.
        n_splits: Number of temporal splits.
        n_test_groups: Number of splits used as test in each path.
        embargo_pct: Fraction of total trading days used as embargo
            between adjacent train/test blocks.
        acceptance_pct: Minimum fraction of paths with Sharpe > 0 for
            a configuration to be accepted.
    """

    def __init__(
        self,
        ohlcv: pl.DataFrame,
        tickers: list[str],
        benchmark_ticker: str = "SPY",
        n_splits: int = 6,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
        acceptance_pct: float = _DEFAULT_ACCEPTANCE_PCT,
    ) -> None:
        self.ohlcv = ohlcv
        self.tickers = tickers
        self.benchmark_ticker = benchmark_ticker
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.acceptance_pct = acceptance_pct

        # Unique sorted trading dates across all tickers
        self._trading_dates: list[date] = (
            ohlcv.select("date")
            .unique()
            .sort("date")["date"]
            .to_list()
        )
        n_dates = len(self._trading_dates)
        self._embargo_days = max(1, int(n_dates * embargo_pct))

        # Pre-compute split boundaries
        self._split_boundaries = self._compute_split_boundaries()
        self._paths = list(
            combinations(range(self.n_splits), self.n_test_groups)
        )

        logger.info(
            "CPCVParameterValidator: {} dates, {} splits, {} paths, "
            "embargo={} days",
            n_dates,
            self.n_splits,
            len(self._paths),
            self._embargo_days,
        )

    # ------------------------------------------------------------------
    # Split generation
    # ------------------------------------------------------------------

    def _compute_split_boundaries(self) -> list[tuple[int, int]]:
        """Compute start/end indices for each temporal split.

        Returns:
            List of ``(start_idx, end_idx)`` tuples into
            ``self._trading_dates``.  Indices are inclusive.
        """
        n = len(self._trading_dates)
        boundaries: list[tuple[int, int]] = []
        base_size = n // self.n_splits
        remainder = n % self.n_splits

        start = 0
        for i in range(self.n_splits):
            size = base_size + (1 if i < remainder else 0)
            end = start + size - 1
            boundaries.append((start, end))
            start = end + 1

        return boundaries

    def _get_test_dates(
        self, test_groups: tuple[int, ...]
    ) -> list[date]:
        """Get the ordered list of test dates for a path.

        Args:
            test_groups: Indices of splits used as test.

        Returns:
            Sorted list of dates in the test period.
        """
        test_dates: list[date] = []
        for g in test_groups:
            start, end = self._split_boundaries[g]
            test_dates.extend(self._trading_dates[start : end + 1])
        return sorted(test_dates)

    def _get_train_dates(
        self, test_groups: tuple[int, ...]
    ) -> list[date]:
        """Get train dates with embargo applied.

        Removes embargo_days after the end of each test block to prevent
        information leakage from the test period into training.

        Args:
            test_groups: Indices of splits used as test.

        Returns:
            Sorted list of dates in the training period.
        """
        test_indices: set[int] = set()
        embargo_indices: set[int] = set()

        for g in test_groups:
            start, end = self._split_boundaries[g]
            for idx in range(start, end + 1):
                test_indices.add(idx)
            # Embargo after each test block
            for idx in range(end + 1, min(end + 1 + self._embargo_days, len(self._trading_dates))):
                embargo_indices.add(idx)

        train_indices = sorted(
            set(range(len(self._trading_dates))) - test_indices - embargo_indices
        )
        return [self._trading_dates[i] for i in train_indices]

    def _filter_ohlcv_by_dates(
        self, dates: list[date]
    ) -> pl.DataFrame:
        """Filter OHLCV to only include rows on the given dates.

        Args:
            dates: Dates to keep.

        Returns:
            Filtered OHLCV DataFrame.
        """
        date_set = set(dates)
        return self.ohlcv.filter(pl.col("date").is_in(date_set))

    # ------------------------------------------------------------------
    # Single path evaluation
    # ------------------------------------------------------------------

    def _evaluate_path(
        self,
        path_id: int,
        test_groups: tuple[int, ...],
        config: WalkForwardConfig,
        model_factory: ModelFactory,
    ) -> tuple[float, list[float]]:
        """Run walk-forward on a single CPCV path and return the Sharpe.

        The walk-forward backtester runs on the **full OHLCV** data
        (so the model can look back ``lookback_days`` into the past for
        covariance estimation and return computation), but the model is
        trained only on **train dates** via ``_PurgedModelFactory``.
        The Sharpe is computed **only on the test dates**.

        Args:
            path_id: Integer ID for logging.
            test_groups: Which splits constitute the test period.
            config: Walk-forward configuration to evaluate.
            model_factory: Model factory for the backtester.

        Returns:
            Tuple of (annualised Sharpe on test period, list of test
            daily returns).
        """
        test_dates = self._get_test_dates(test_groups)
        if len(test_dates) < 2:
            logger.warning("Path {}: fewer than 2 test dates", path_id)
            return 0.0, []

        # Deep-copy the model factory to prevent cross-path state
        # contamination.  Each CPCV path must start with a fresh model.
        fresh_factory = copy.deepcopy(model_factory)

        # Compute excluded dates = all dates NOT in train set
        # (i.e. test dates + embargo dates).  The purged factory ensures
        # the model never trains on these.
        train_dates = self._get_train_dates(test_groups)
        train_date_set = set(train_dates)
        excluded_dates = set(self._trading_dates) - train_date_set

        # Wrap the model factory to enforce train/test separation
        purged_factory = _PurgedModelFactory(fresh_factory, excluded_dates)

        # Run walk-forward on the full data (backtester needs continuous
        # dates for return computation and weight drift).
        try:
            backtester = WalkForwardBacktester(config=config)
            result = backtester.run(
                ohlcv=self.ohlcv,
                tickers=self.tickers,
                benchmark_ticker=self.benchmark_ticker,
                model_factory=purged_factory,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("Path {}: backtester failed: {}", path_id, exc)
            return 0.0, []

        # Extract portfolio returns for test dates only
        test_date_set = set(test_dates)
        test_returns = (
            result.daily_returns
            .filter(pl.col("date").is_in(test_date_set))
            .sort("date")
        )

        if test_returns.height < 2:
            logger.warning(
                "Path {}: only {} test returns", path_id, test_returns.height
            )
            return 0.0, []

        port_rets = test_returns["portfolio_return"].to_list()
        sharpe = _compute_sharpe(
            port_rets,
            rf=config.rf,
            trading_days=config.trading_days_per_year,
        )

        logger.debug(
            "Path {} (test splits {}): {} test days, Sharpe={:.3f}",
            path_id,
            test_groups,
            len(port_rets),
            sharpe,
        )
        return sharpe, port_rets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        config: WalkForwardConfig,
        model_factory: ModelFactory | None = None,
        baseline_sharpe: float | None = None,
        n_trials: int = 1,
    ) -> ValidationResult:
        """Validate a single walk-forward configuration via CPCV-OOS.

        Runs the walk-forward backtester on each of the ``C(n_splits,
        n_test_groups)`` paths and aggregates Sharpe ratios.

        Args:
            config: Walk-forward configuration to evaluate.
            model_factory: Model factory for predictions.  Defaults to
                ``NaiveModelFactory()`` if ``None``.
            baseline_sharpe: Reference Sharpe for the DSR test.  If
                ``None``, uses 0.0 (tests whether Sharpe > 0).
            n_trials: Number of configurations tested so far (for DSR
                correction).  Pass 1 if this is the only config.

        Returns:
            ValidationResult with per-path Sharpe and acceptance verdict.
        """
        if model_factory is None:
            model_factory = NaiveModelFactory()

        if baseline_sharpe is None:
            baseline_sharpe = 0.0

        logger.info(
            "Validating config: rebalance_every={}, retrain_every={}, "
            "lookback_days={}, n_trials={}",
            config.rebalance_every,
            config.retrain_every,
            config.lookback_days,
            n_trials,
        )

        per_path_sharpe: list[float] = []
        all_test_returns: list[float] = []
        max_path_obs = 0
        for path_id, test_groups in enumerate(self._paths):
            sharpe, path_returns = self._evaluate_path(
                path_id, test_groups, config, model_factory
            )
            per_path_sharpe.append(sharpe)
            all_test_returns.extend(path_returns)
            if len(path_returns) > max_path_obs:
                max_path_obs = len(path_returns)

        # Aggregate
        n_paths = len(per_path_sharpe)
        mean_sharpe = sum(per_path_sharpe) / n_paths if n_paths > 0 else 0.0
        std_sharpe = _std_list(per_path_sharpe)
        n_positive = sum(1 for s in per_path_sharpe if s > 0)
        pct_positive = n_positive / n_paths if n_paths > 0 else 0.0

        # Use the longest individual path's observation count for DSR
        # (conservative: more obs = tighter SR variance estimate)
        n_obs = max(2, max_path_obs)

        # Compute empirical skewness and kurtosis from actual test returns
        skew = _skewness(all_test_returns)
        kurt = _kurtosis(all_test_returns)

        dsr_pvalue = deflated_sharpe_ratio(
            observed_sharpe=mean_sharpe,
            n_trials=max(1, n_trials),
            n_observations=n_obs,
            skewness=skew,
            kurtosis=kurt,
            sharpe_benchmark=baseline_sharpe,
        )

        accepted = (
            pct_positive >= self.acceptance_pct
            and dsr_pvalue > _DEFAULT_DSR_THRESHOLD
        )

        logger.info(
            "Validation result: mean_sharpe={:.3f}, std={:.3f}, "
            "pct_positive={:.1%}, DSR_pvalue={:.3f}, accepted={}",
            mean_sharpe,
            std_sharpe,
            pct_positive,
            dsr_pvalue,
            accepted,
        )

        return ValidationResult(
            config=config,
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            pct_positive=pct_positive,
            per_path_sharpe=per_path_sharpe,
            deflated_sharpe=dsr_pvalue,
            p_value=dsr_pvalue,
            accepted=accepted,
            metadata={
                "n_paths": n_paths,
                "n_positive": n_positive,
                "n_trials": n_trials,
                "embargo_days": self._embargo_days,
                "paths": [list(p) for p in self._paths],
                "baseline_sharpe": baseline_sharpe,
                "n_observations": n_obs,
                "skewness": skew,
                "kurtosis": kurt,
            },
        )

    def grid_search(
        self,
        configs: dict[str, WalkForwardConfig],
        model_factory: ModelFactory | None = None,
        baseline_sharpe: float | None = None,
    ) -> list[tuple[str, ValidationResult]]:
        """Validate multiple configurations and return ranked results.

        Each configuration is validated via CPCV-OOS.  The Deflated
        Sharpe Ratio accounts for the total number of configurations
        tested (``n_trials = len(configs)``).

        Args:
            configs: Named configurations to evaluate.
            model_factory: Model factory (shared across configs).
            baseline_sharpe: Reference Sharpe for DSR.

        Returns:
            List of ``(name, ValidationResult)`` sorted by
            ``deflated_sharpe`` descending.
        """
        n_trials = len(configs)
        results: list[tuple[str, ValidationResult]] = []

        for i, (name, config) in enumerate(configs.items()):
            logger.info(
                "Grid search [{}/{}]: {}",
                i + 1,
                n_trials,
                name,
            )
            result = self.validate(
                config=config,
                model_factory=model_factory,
                baseline_sharpe=baseline_sharpe,
                n_trials=n_trials,
            )
            results.append((name, result))

        # Sort by deflated_sharpe descending
        results.sort(key=lambda x: x[1].deflated_sharpe, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _std_list(values: list[float]) -> float:
    """Sample standard deviation (ddof=1) from a list."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var) if var > 0 else 0.0


def _skewness(values: list[float]) -> float:
    """Sample skewness (Fisher's definition) from a list.

    Returns 0.0 for fewer than 3 observations or zero variance.
    """
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    if m2 <= 0:
        return 0.0
    m3 = sum((v - mean) ** 3 for v in values) / n
    return m3 / (m2 ** 1.5)


def _kurtosis(values: list[float]) -> float:
    """Sample kurtosis (full, not excess) from a list.

    Returns 3.0 (normal) for fewer than 4 observations or zero variance.
    The returned value is the **full kurtosis** (normal = 3.0), matching
    the convention used in the DSR formula where ``(kurtosis - 3)``
    converts to excess kurtosis internally.
    """
    n = len(values)
    if n < 4:
        return 3.0
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    if m2 <= 0:
        return 3.0
    m4 = sum((v - mean) ** 4 for v in values) / n
    return m4 / (m2 ** 2)
