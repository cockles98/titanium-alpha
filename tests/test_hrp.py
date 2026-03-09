"""Tests for the Hierarchical Risk Parity optimizer."""

from __future__ import annotations

import math
import random

import numpy as np
import polars as pl
import pytest

from src.portfolio.hrp import HRPConfig, HRPOptimizer, HRPResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def four_asset_returns() -> pl.DataFrame:
    """Synthetic daily returns for 4 assets (SPY, NVDA, AAPL, QQQ).

    250 rows of random returns with realistic correlations:
    - SPY and QQQ are highly correlated (~0.9)
    - NVDA and AAPL moderately correlated (~0.6)
    """
    random.seed(42)
    np.random.seed(42)
    n = 250

    # Generate correlated returns via Cholesky decomposition
    corr_target = np.array([
        [1.0, 0.5, 0.5, 0.9],  # SPY
        [0.5, 1.0, 0.6, 0.5],  # NVDA
        [0.5, 0.6, 1.0, 0.5],  # AAPL
        [0.9, 0.5, 0.5, 1.0],  # QQQ
    ])
    vols = np.array([0.01, 0.02, 0.015, 0.012])  # daily vol
    cov_target = np.outer(vols, vols) * corr_target
    L = np.linalg.cholesky(cov_target)

    z = np.random.randn(n, 4)
    returns = z @ L.T

    return pl.DataFrame({
        "SPY": returns[:, 0].tolist(),
        "NVDA": returns[:, 1].tolist(),
        "AAPL": returns[:, 2].tolist(),
        "QQQ": returns[:, 3].tolist(),
    })


@pytest.fixture()
def two_asset_returns() -> pl.DataFrame:
    """Minimal 2-asset returns for edge case testing."""
    np.random.seed(123)
    n = 60
    return pl.DataFrame({
        "A": np.random.randn(n).tolist(),
        "B": np.random.randn(n).tolist(),
    })


@pytest.fixture()
def single_asset_returns() -> pl.DataFrame:
    """Single-asset returns."""
    np.random.seed(99)
    return pl.DataFrame({
        "SPY": np.random.randn(100).tolist(),
    })


@pytest.fixture()
def default_optimizer() -> HRPOptimizer:
    """HRP optimizer with default config."""
    return HRPOptimizer()


# ---------------------------------------------------------------------------
# HRPConfig tests
# ---------------------------------------------------------------------------


class TestHRPConfig:
    """Tests for HRPConfig dataclass."""

    def test_default_values(self) -> None:
        config = HRPConfig()
        assert config.linkage_method == "single"
        assert config.correlation_method == "pearson"
        assert config.shrinkage is False
        assert config.confidence_tilt_cap == 0.20
        assert config.min_weight == 0.0
        assert config.max_weight == 0.25

    def test_custom_values(self) -> None:
        config = HRPConfig(
            linkage_method="ward",
            correlation_method="spearman",
            confidence_tilt_cap=0.30,
            max_weight=0.50,
        )
        assert config.linkage_method == "ward"
        assert config.correlation_method == "spearman"
        assert config.confidence_tilt_cap == 0.30
        assert config.max_weight == 0.50

    def test_frozen(self) -> None:
        config = HRPConfig()
        with pytest.raises(AttributeError):
            config.linkage_method = "ward"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HRPResult tests
# ---------------------------------------------------------------------------


class TestHRPResult:
    """Tests for HRPResult dataclass."""

    def test_empty_defaults(self) -> None:
        result = HRPResult()
        assert result.weights == {}
        assert result.raw_weights == {}
        assert result.cluster_order == []
        assert result.linkage_matrix == []

    def test_populated(self) -> None:
        result = HRPResult(
            weights={"SPY": 0.5, "NVDA": 0.5},
            raw_weights={"SPY": 0.5, "NVDA": 0.5},
            cluster_order=["SPY", "NVDA"],
            linkage_matrix=[[0, 1, 0.5, 2]],
        )
        assert sum(result.weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


class TestHRPOptimizerInit:
    """Tests for HRPOptimizer initialization and validation."""

    def test_default_config(self, default_optimizer: HRPOptimizer) -> None:
        assert default_optimizer.config.linkage_method == "single"

    def test_custom_config(self) -> None:
        config = HRPConfig(linkage_method="ward")
        opt = HRPOptimizer(config)
        assert opt.config.linkage_method == "ward"

    def test_invalid_linkage_method(self) -> None:
        with pytest.raises(ValueError, match="linkage_method"):
            HRPOptimizer(HRPConfig(linkage_method="invalid"))

    def test_invalid_correlation_method(self) -> None:
        with pytest.raises(ValueError, match="correlation_method"):
            HRPOptimizer(HRPConfig(correlation_method="kendall"))

    def test_negative_tilt_cap(self) -> None:
        with pytest.raises(ValueError, match="confidence_tilt_cap"):
            HRPOptimizer(HRPConfig(confidence_tilt_cap=-0.1))

    def test_zero_max_weight(self) -> None:
        with pytest.raises(ValueError, match="max_weight"):
            HRPOptimizer(HRPConfig(max_weight=0.0))

    def test_valid_linkage_methods(self) -> None:
        for method in ("single", "complete", "average", "ward"):
            opt = HRPOptimizer(HRPConfig(linkage_method=method))
            assert opt.config.linkage_method == method


# ---------------------------------------------------------------------------
# Covariance computation
# ---------------------------------------------------------------------------


class TestComputeCovariance:
    """Tests for _compute_covariance."""

    def test_shapes(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        cov, corr = default_optimizer._compute_covariance(four_asset_returns)
        assert cov.shape == (4, 4)
        assert corr.shape == (4, 4)

    def test_cov_symmetric(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        cov, _ = default_optimizer._compute_covariance(four_asset_returns)
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_corr_diagonal_ones(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        _, corr = default_optimizer._compute_covariance(four_asset_returns)
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(4))

    def test_corr_bounded(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        _, corr = default_optimizer._compute_covariance(four_asset_returns)
        assert np.all(corr >= -1.0)
        assert np.all(corr <= 1.0)

    def test_cov_positive_diagonal(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        cov, _ = default_optimizer._compute_covariance(four_asset_returns)
        assert np.all(np.diag(cov) > 0)

    def test_spearman_correlation(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        opt = HRPOptimizer(HRPConfig(correlation_method="spearman"))
        _, corr = opt._compute_covariance(four_asset_returns)
        assert corr.shape == (4, 4)
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(4))

    def test_two_assets(
        self, default_optimizer: HRPOptimizer, two_asset_returns: pl.DataFrame
    ) -> None:
        cov, corr = default_optimizer._compute_covariance(two_asset_returns)
        assert cov.shape == (2, 2)
        assert corr.shape == (2, 2)


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------


class TestComputeDistance:
    """Tests for _compute_distance."""

    def test_non_negative(self, default_optimizer: HRPOptimizer) -> None:
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        dist = default_optimizer._compute_distance(corr)
        assert np.all(dist >= 0)

    def test_perfect_correlation_zero_distance(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        corr = np.array([[1.0, 1.0], [1.0, 1.0]])
        dist = default_optimizer._compute_distance(corr)
        assert dist[0] == pytest.approx(0.0)

    def test_zero_correlation_max_distance(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        corr = np.array([[1.0, 0.0], [0.0, 1.0]])
        dist = default_optimizer._compute_distance(corr)
        expected = math.sqrt(0.5)
        assert dist[0] == pytest.approx(expected)

    def test_negative_correlation(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
        dist = default_optimizer._compute_distance(corr)
        assert dist[0] == pytest.approx(1.0)

    def test_condensed_form_length(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        n = 4
        corr = np.eye(n)
        dist = default_optimizer._compute_distance(corr)
        # Condensed form: n*(n-1)/2
        assert len(dist) == n * (n - 1) // 2


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class TestCluster:
    """Tests for _cluster."""

    def test_linkage_shape(self, default_optimizer: HRPOptimizer) -> None:
        n = 4
        corr = np.eye(n)
        dist = default_optimizer._compute_distance(corr)
        link = default_optimizer._cluster(dist)
        assert link.shape == (n - 1, 4)

    def test_ward_linkage(self) -> None:
        opt = HRPOptimizer(HRPConfig(linkage_method="ward"))
        n = 4
        corr = np.eye(n)
        dist = opt._compute_distance(corr)
        link = opt._cluster(dist)
        assert link.shape == (n - 1, 4)


# ---------------------------------------------------------------------------
# Quasi-diagonalisation
# ---------------------------------------------------------------------------


class TestQuasiDiagonalize:
    """Tests for _quasi_diagonalize."""

    def test_contains_all_indices(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        n = 4
        corr = np.eye(n)
        dist = default_optimizer._compute_distance(corr)
        link = default_optimizer._cluster(dist)
        order = default_optimizer._quasi_diagonalize(link, n)
        assert sorted(order) == list(range(n))

    def test_no_duplicates(self, default_optimizer: HRPOptimizer) -> None:
        n = 4
        corr = np.eye(n)
        dist = default_optimizer._compute_distance(corr)
        link = default_optimizer._cluster(dist)
        order = default_optimizer._quasi_diagonalize(link, n)
        assert len(order) == len(set(order))

    def test_two_assets(self, default_optimizer: HRPOptimizer) -> None:
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        dist = default_optimizer._compute_distance(corr)
        link = default_optimizer._cluster(dist)
        order = default_optimizer._quasi_diagonalize(link, 2)
        assert sorted(order) == [0, 1]


# ---------------------------------------------------------------------------
# Recursive bisection
# ---------------------------------------------------------------------------


class TestRecursiveBisection:
    """Tests for _recursive_bisection."""

    def test_weights_sum_to_one(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        cov = np.array([
            [0.04, 0.01],
            [0.01, 0.09],
        ])
        weights = default_optimizer._recursive_bisection(cov, [0, 1])
        total = sum(weights.values())
        assert total == pytest.approx(1.0)

    def test_lower_var_gets_more_weight(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        # Asset 0 has lower variance → should get more weight
        cov = np.array([
            [0.01, 0.0],
            [0.0, 0.09],
        ])
        weights = default_optimizer._recursive_bisection(cov, [0, 1])
        assert weights[0] > weights[1]

    def test_equal_variance_equal_weight(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        cov = np.array([
            [0.04, 0.0],
            [0.0, 0.04],
        ])
        weights = default_optimizer._recursive_bisection(cov, [0, 1])
        assert weights[0] == pytest.approx(weights[1], abs=1e-10)

    def test_four_assets(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        cov = np.diag([0.01, 0.04, 0.02, 0.08])
        weights = default_optimizer._recursive_bisection(
            cov, [0, 1, 2, 3]
        )
        assert len(weights) == 4
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_single_asset(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        cov = np.array([[0.04]])
        weights = default_optimizer._recursive_bisection(cov, [0])
        assert weights[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cluster variance
# ---------------------------------------------------------------------------


class TestGetClusterVar:
    """Tests for _get_cluster_var."""

    def test_single_asset(self) -> None:
        cov = np.array([[0.04]])
        var = HRPOptimizer._get_cluster_var(cov, [0])
        assert var == pytest.approx(0.04)

    def test_uncorrelated_assets(self) -> None:
        cov = np.diag([0.04, 0.09])
        var = HRPOptimizer._get_cluster_var(cov, [0, 1])
        # IVP variance: w0 = (1/0.04)/(1/0.04+1/0.09), etc.
        inv0 = 1.0 / 0.04
        inv1 = 1.0 / 0.09
        total_inv = inv0 + inv1
        w0 = inv0 / total_inv
        w1 = inv1 / total_inv
        expected = w0**2 * 0.04 + w1**2 * 0.09
        assert var == pytest.approx(expected)

    def test_positive(self) -> None:
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        var = HRPOptimizer._get_cluster_var(cov, [0, 1])
        assert var > 0


# ---------------------------------------------------------------------------
# Confidence tilt
# ---------------------------------------------------------------------------


class TestConfidenceTilt:
    """Tests for _apply_confidence_tilt."""

    def test_neutral_confidence_no_change(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        weights = {"SPY": 0.5, "NVDA": 0.5}
        confidences = {"SPY": 0.5, "NVDA": 0.5}
        tilted = default_optimizer._apply_confidence_tilt(weights, confidences)
        assert tilted["SPY"] == pytest.approx(0.5)
        assert tilted["NVDA"] == pytest.approx(0.5)

    def test_high_confidence_gets_boost(self) -> None:
        opt = HRPOptimizer(HRPConfig(max_weight=1.0))
        weights = {"A": 0.5, "B": 0.5}
        confidences = {"A": 1.0, "B": 0.0}
        tilted = opt._apply_confidence_tilt(weights, confidences)
        # A should get more than B after tilt
        assert tilted["A"] > tilted["B"]

    def test_sum_to_one(self, default_optimizer: HRPOptimizer) -> None:
        weights = {"SPY": 0.3, "NVDA": 0.4, "AAPL": 0.3}
        confidences = {"SPY": 0.9, "NVDA": 0.2, "AAPL": 0.7}
        tilted = default_optimizer._apply_confidence_tilt(weights, confidences)
        assert sum(tilted.values()) == pytest.approx(1.0)

    def test_missing_confidence_uses_neutral(self) -> None:
        opt = HRPOptimizer(HRPConfig(max_weight=1.0))
        weights = {"A": 0.5, "B": 0.5}
        confidences = {"A": 1.0}  # B missing
        tilted = opt._apply_confidence_tilt(weights, confidences)
        # B gets neutral tilt (0.5), A gets boosted
        assert tilted["A"] > tilted["B"]

    def test_clipping_max_weight(self) -> None:
        config = HRPConfig(max_weight=0.25)
        opt = HRPOptimizer(config)
        # Give very uneven weights to trigger clipping
        weights = {"A": 0.8, "B": 0.1, "C": 0.1}
        confidences = {"A": 1.0, "B": 0.0, "C": 0.0}
        tilted = opt._apply_confidence_tilt(weights, confidences)
        # After clipping and renormalisation, should sum to 1
        assert sum(tilted.values()) == pytest.approx(1.0)

    def test_clipping_min_weight(self) -> None:
        config = HRPConfig(min_weight=0.05, max_weight=0.50)
        opt = HRPOptimizer(config)
        weights = {"A": 0.01, "B": 0.99}
        confidences = {"A": 0.0, "B": 1.0}
        tilted = opt._apply_confidence_tilt(weights, confidences)
        # After clipping, A should be at least min_weight (pre-renorm)
        assert sum(tilted.values()) == pytest.approx(1.0)

    def test_confidence_clamped_to_01(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        weights = {"A": 0.5, "B": 0.5}
        confidences = {"A": 2.0, "B": -1.0}  # out of range
        tilted = default_optimizer._apply_confidence_tilt(weights, confidences)
        assert sum(tilted.values()) == pytest.approx(1.0)

    def test_zero_tilt_cap_no_change(self) -> None:
        config = HRPConfig(confidence_tilt_cap=0.0, max_weight=1.0)
        opt = HRPOptimizer(config)
        weights = {"A": 0.3, "B": 0.7}
        confidences = {"A": 1.0, "B": 0.0}
        tilted = opt._apply_confidence_tilt(weights, confidences)
        # With cap=0, multiplier is always 1.0
        assert tilted["A"] == pytest.approx(0.3)
        assert tilted["B"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Full pipeline — optimize()
# ---------------------------------------------------------------------------


class TestOptimize:
    """Integration tests for the full HRP pipeline."""

    def test_four_assets_weights_sum_to_one(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        result = default_optimizer.optimize(four_asset_returns)
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_four_assets_all_positive_weights(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        result = default_optimizer.optimize(four_asset_returns)
        for w in result.weights.values():
            assert w >= 0

    def test_four_assets_correct_tickers(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        result = default_optimizer.optimize(four_asset_returns)
        assert set(result.weights.keys()) == {"SPY", "NVDA", "AAPL", "QQQ"}

    def test_raw_weights_sum_to_one(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        result = default_optimizer.optimize(four_asset_returns)
        assert sum(result.raw_weights.values()) == pytest.approx(1.0)

    def test_cluster_order_has_all_tickers(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        result = default_optimizer.optimize(four_asset_returns)
        assert sorted(result.cluster_order) == ["AAPL", "NVDA", "QQQ", "SPY"]

    def test_linkage_matrix_shape(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        result = default_optimizer.optimize(four_asset_returns)
        # n-1 merges for n=4 assets, each merge = [left, right, dist, count]
        assert len(result.linkage_matrix) == 3
        for row in result.linkage_matrix:
            assert len(row) == 4

    def test_with_confidences(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        confidences = {"SPY": 0.9, "NVDA": 0.3, "AAPL": 0.7, "QQQ": 0.5}
        result = default_optimizer.optimize(four_asset_returns, confidences)
        assert sum(result.weights.values()) == pytest.approx(1.0)
        # Weights should differ from raw weights due to tilt
        assert result.weights != result.raw_weights

    def test_without_confidences_clips_max_weight(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        """Without confidences, max_weight clipping still applies."""
        result = default_optimizer.optimize(four_asset_returns)
        for w in result.weights.values():
            # After clipping + renorm, no weight exceeds max_weight
            # (unless renorm pushes it back — but sum must be 1.0)
            assert w >= 0
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_no_clipping_matches_raw(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        """With max_weight=1.0 (no effective clipping), weights match raw."""
        opt = HRPOptimizer(HRPConfig(max_weight=1.0))
        result = opt.optimize(four_asset_returns)
        for ticker in result.weights:
            assert result.weights[ticker] == pytest.approx(
                result.raw_weights[ticker]
            )

    def test_two_assets(
        self, default_optimizer: HRPOptimizer, two_asset_returns: pl.DataFrame
    ) -> None:
        result = default_optimizer.optimize(two_asset_returns)
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert len(result.weights) == 2

    def test_single_asset(
        self, default_optimizer: HRPOptimizer, single_asset_returns: pl.DataFrame
    ) -> None:
        result = default_optimizer.optimize(single_asset_returns)
        assert result.weights == {"SPY": 1.0}
        assert result.linkage_matrix == []

    def test_too_few_observations(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        df = pl.DataFrame({"A": [0.01]})
        with pytest.raises(ValueError, match="observations"):
            default_optimizer.optimize(df)

    def test_warning_few_observations(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        """With < 120 observations should still work but log warning."""
        np.random.seed(77)
        df = pl.DataFrame({
            "A": np.random.randn(50).tolist(),
            "B": np.random.randn(50).tolist(),
        })
        result = default_optimizer.optimize(df)
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_ward_linkage(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        opt = HRPOptimizer(HRPConfig(linkage_method="ward"))
        result = opt.optimize(four_asset_returns)
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_spearman_correlation(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        opt = HRPOptimizer(HRPConfig(correlation_method="spearman"))
        result = opt.optimize(four_asset_returns)
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_high_vol_asset_gets_less_weight(
        self, default_optimizer: HRPOptimizer
    ) -> None:
        """Asset with higher variance should get lower weight (HRP property)."""
        np.random.seed(55)
        n = 200
        df = pl.DataFrame({
            "LOW_VOL": (np.random.randn(n) * 0.005).tolist(),
            "HIGH_VOL": (np.random.randn(n) * 0.05).tolist(),
        })
        result = default_optimizer.optimize(df)
        assert result.weights["LOW_VOL"] > result.weights["HIGH_VOL"]

    def test_confidence_boost_increases_weight(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        """High confidence on one asset should increase its weight vs raw."""
        opt = HRPOptimizer(HRPConfig(max_weight=1.0))
        # First get raw weights (no confidence)
        raw_result = opt.optimize(four_asset_returns)

        # Then with high confidence on SPY
        conf_result = opt.optimize(
            four_asset_returns,
            confidences={"SPY": 1.0, "NVDA": 0.5, "AAPL": 0.5, "QQQ": 0.5},
        )
        assert conf_result.weights["SPY"] > raw_result.weights["SPY"]

    def test_deterministic(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        """Running twice with same input should give identical results."""
        r1 = default_optimizer.optimize(four_asset_returns)
        r2 = default_optimizer.optimize(four_asset_returns)
        for ticker in r1.weights:
            assert r1.weights[ticker] == pytest.approx(r2.weights[ticker])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_constant_returns_asset(self) -> None:
        """An asset with zero variance should still work."""
        np.random.seed(88)
        n = 100
        df = pl.DataFrame({
            "CONST": [0.0] * n,
            "VOLATILE": np.random.randn(n).tolist(),
        })
        opt = HRPOptimizer()
        result = opt.optimize(df)
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_identical_returns(self) -> None:
        """Two assets with perfectly correlated returns."""
        np.random.seed(44)
        n = 100
        base = np.random.randn(n)
        df = pl.DataFrame({
            "A": base.tolist(),
            "B": base.tolist(),
        })
        opt = HRPOptimizer()
        result = opt.optimize(df)
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_many_assets(self) -> None:
        """Stress test with 20 assets."""
        np.random.seed(99)
        n = 200
        tickers = [f"ASSET_{i}" for i in range(20)]
        data = {t: np.random.randn(n).tolist() for t in tickers}
        df = pl.DataFrame(data)
        opt = HRPOptimizer(HRPConfig(max_weight=1.0))
        result = opt.optimize(df)
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert len(result.weights) == 20

    def test_result_serialisable(
        self, default_optimizer: HRPOptimizer, four_asset_returns: pl.DataFrame
    ) -> None:
        """HRPResult should be JSON-serialisable (for decision_engine)."""
        import json

        result = default_optimizer.optimize(four_asset_returns)
        # All fields should be basic Python types
        data = {
            "weights": result.weights,
            "raw_weights": result.raw_weights,
            "cluster_order": result.cluster_order,
            "linkage_matrix": result.linkage_matrix,
        }
        json_str = json.dumps(data)
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# Ledoit-Wolf shrinkage tests
# ---------------------------------------------------------------------------


class TestLedoitWolfShrinkage:
    """Tests for Ledoit-Wolf shrinkage covariance estimator."""

    def test_shrinkage_produces_valid_cov(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        """Shrinkage covariance must be symmetric positive semi-definite."""
        opt = HRPOptimizer(HRPConfig(shrinkage=True, max_weight=1.0))
        cov, corr = opt._compute_covariance(four_asset_returns)
        # Symmetric
        np.testing.assert_array_almost_equal(cov, cov.T)
        # Positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)

    def test_shrinkage_corr_diagonal_ones(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        """Correlation diagonal must be 1.0 even with shrinkage."""
        opt = HRPOptimizer(HRPConfig(shrinkage=True, max_weight=1.0))
        _, corr = opt._compute_covariance(four_asset_returns)
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(4))

    def test_shrinkage_corr_bounded(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        """Correlation values must be in [-1, 1] with shrinkage."""
        opt = HRPOptimizer(HRPConfig(shrinkage=True, max_weight=1.0))
        _, corr = opt._compute_covariance(four_asset_returns)
        assert np.all(corr >= -1.0 - 1e-10)
        assert np.all(corr <= 1.0 + 1e-10)

    def test_shrinkage_weights_sum_to_one(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        """Full pipeline with shrinkage must produce valid weights."""
        opt = HRPOptimizer(HRPConfig(shrinkage=True, max_weight=1.0))
        result = opt.optimize(four_asset_returns)
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert all(w >= 0 for w in result.weights.values())

    def test_shrinkage_all_tickers_present(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        """Shrinkage result must include all tickers."""
        opt = HRPOptimizer(HRPConfig(shrinkage=True))
        result = opt.optimize(four_asset_returns)
        assert set(result.weights.keys()) == {"SPY", "NVDA", "AAPL", "QQQ"}

    def test_shrinkage_false_backward_compat(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        """shrinkage=False must produce identical results to default."""
        opt_default = HRPOptimizer(HRPConfig(max_weight=1.0))
        opt_no_shrink = HRPOptimizer(
            HRPConfig(shrinkage=False, max_weight=1.0)
        )
        r1 = opt_default.optimize(four_asset_returns)
        r2 = opt_no_shrink.optimize(four_asset_returns)
        for ticker in r1.weights:
            assert r1.weights[ticker] == pytest.approx(r2.weights[ticker])

    def test_shrinkage_convergence_large_sample(self) -> None:
        """With n_obs >> n_assets, shrinkage and sample cov should converge."""
        np.random.seed(42)
        n = 2000  # large sample
        data = {
            "A": np.random.randn(n).tolist(),
            "B": np.random.randn(n).tolist(),
        }
        df = pl.DataFrame(data)
        opt_sample = HRPOptimizer(HRPConfig(max_weight=1.0))
        opt_shrink = HRPOptimizer(
            HRPConfig(shrinkage=True, max_weight=1.0)
        )
        r_sample = opt_sample.optimize(df)
        r_shrink = opt_shrink.optimize(df)
        # With large sample, weights should be close (within 5%)
        for ticker in r_sample.weights:
            assert r_sample.weights[ticker] == pytest.approx(
                r_shrink.weights[ticker], abs=0.05
            )

    def test_shrinkage_with_confidences(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        """Shrinkage + confidence tilt must produce valid weights."""
        opt = HRPOptimizer(HRPConfig(shrinkage=True))
        confidences = {"SPY": 0.9, "NVDA": 0.3, "AAPL": 0.7, "QQQ": 0.5}
        result = opt.optimize(four_asset_returns, confidences)
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert result.weights != result.raw_weights

    def test_shrinkage_deterministic(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        """Shrinkage must be deterministic."""
        opt = HRPOptimizer(HRPConfig(shrinkage=True, max_weight=1.0))
        r1 = opt.optimize(four_asset_returns)
        r2 = opt.optimize(four_asset_returns)
        for ticker in r1.weights:
            assert r1.weights[ticker] == pytest.approx(r2.weights[ticker])

    def test_shrinkage_with_spearman(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        """Shrinkage cov + Spearman correlation must work."""
        opt = HRPOptimizer(HRPConfig(
            shrinkage=True,
            correlation_method="spearman",
            max_weight=1.0,
        ))
        result = opt.optimize(four_asset_returns)
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_shrinkage_few_obs_many_assets(self) -> None:
        """Shrinkage must work when n_obs is close to n_assets (p ~ n)."""
        np.random.seed(42)
        n_obs = 10
        n_assets = 8
        data = {
            f"A{i}": np.random.randn(n_obs).tolist()
            for i in range(n_assets)
        }
        df = pl.DataFrame(data)
        opt = HRPOptimizer(HRPConfig(shrinkage=True, max_weight=1.0))
        result = opt.optimize(df)
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert all(w >= 0 for w in result.weights.values())
        assert len(result.weights) == n_assets


# ---------------------------------------------------------------------------
# Ward linkage tests
# ---------------------------------------------------------------------------


class TestWardLinkage:
    """Tests for ward linkage producing more balanced clusters."""

    def test_ward_more_balanced_than_single(self) -> None:
        """Ward linkage should produce less dispersed weights than single.

        With many assets and a chaining-prone correlation structure,
        single linkage tends to isolate individual assets while ward
        creates more balanced clusters.
        """
        np.random.seed(42)
        n = 250
        n_assets = 20
        # Create block-diagonal structure (groups of correlated assets)
        data = {}
        for group in range(4):
            base = np.random.randn(n)
            for i in range(5):
                ticker = f"ASSET_{group * 5 + i}"
                noise = np.random.randn(n) * 0.3
                data[ticker] = (base + noise).tolist()
        df = pl.DataFrame(data)

        opt_single = HRPOptimizer(HRPConfig(
            linkage_method="single", max_weight=1.0,
        ))
        opt_ward = HRPOptimizer(HRPConfig(
            linkage_method="ward", max_weight=1.0,
        ))
        r_single = opt_single.optimize(df)
        r_ward = opt_ward.optimize(df)

        # Measure dispersion via std of weights
        w_single = np.array(list(r_single.weights.values()))
        w_ward = np.array(list(r_ward.weights.values()))
        assert w_ward.std() < w_single.std()

    def test_ward_produces_valid_linkage(
        self, four_asset_returns: pl.DataFrame
    ) -> None:
        """Ward linkage matrix must have correct shape and valid entries."""
        opt = HRPOptimizer(HRPConfig(linkage_method="ward"))
        result = opt.optimize(four_asset_returns)
        assert len(result.linkage_matrix) == 3  # n-1 for 4 assets
        for row in result.linkage_matrix:
            assert len(row) == 4
            assert row[2] >= 0  # distance >= 0
            assert row[3] >= 2  # cluster size >= 2
