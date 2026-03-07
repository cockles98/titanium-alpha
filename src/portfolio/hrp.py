"""Hierarchical Risk Parity (HRP) portfolio optimizer.

Implements the HRP algorithm formalised by Marcos Lopez de Prado (2016)
for allocating portfolio weights based on the covariance structure of
asset returns, optionally tilted by agent-supplied confidence scores.

Pipeline::

    returns DataFrame (cols=tickers, rows=dates)
        │
        ├─ _compute_covariance  → (cov, corr)
        ├─ _compute_distance    → condensed distance vector
        ├─ _cluster             → scipy linkage matrix
        ├─ _quasi_diagonalize   → seriated leaf order
        ├─ _recursive_bisection → raw HRP weights
        └─ _apply_confidence_tilt → tilted + clipped weights
        │
        v
    HRPResult(weights, raw_weights, cluster_order, linkage_matrix)

Usage::

    from src.portfolio.hrp import HRPOptimizer

    optimizer = HRPOptimizer()
    result = optimizer.optimize(returns_df, confidences={"SPY": 0.8, "NVDA": 0.6})
    print(result.weights)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from loguru import logger
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HRPConfig:
    """Immutable configuration for the HRP optimizer.

    Attributes:
        linkage_method: Agglomerative clustering linkage method.
            ``"single"`` follows the original Lopez de Prado paper;
            ``"ward"`` creates more balanced clusters.
        correlation_method: ``"pearson"`` (default, consistent with HRP
            literature) or ``"spearman"`` (more robust to outliers).
        confidence_tilt_cap: Maximum adjustment factor applied to raw
            HRP weights based on agent confidence.  A cap of 0.20 means
            weights are scaled by at most ±10% (at confidence extremes
            0.0 and 1.0).
        min_weight: Floor per asset after tilt (before renormalisation).
        max_weight: Cap per asset after tilt (before renormalisation).
            Aligned with ``MAX_SINGLE_WEIGHT = 0.25`` from the agent
            state definitions.
    """

    linkage_method: str = "single"
    correlation_method: str = "pearson"
    confidence_tilt_cap: float = 0.20
    min_weight: float = 0.0
    max_weight: float = 0.25


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class HRPResult:
    """Output of the HRP optimization.

    Attributes:
        weights: Final portfolio weights after confidence tilt and
            clipping.  Keys are ticker symbols; values sum to 1.0.
        raw_weights: HRP weights before the confidence tilt.
        cluster_order: Ticker symbols reordered by the dendrogram
            seriation (quasi-diagonalisation order).
        linkage_matrix: The scipy linkage matrix, stored for
            visualisation or reporting.
    """

    weights: dict[str, float] = field(default_factory=dict)
    raw_weights: dict[str, float] = field(default_factory=dict)
    cluster_order: list[str] = field(default_factory=list)
    linkage_matrix: list[list[float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_ASSETS = 1
_MIN_OBSERVATIONS = 2
_RECOMMENDED_OBSERVATIONS = 120


# ---------------------------------------------------------------------------
# HRPOptimizer
# ---------------------------------------------------------------------------


class HRPOptimizer:
    """Hierarchical Risk Parity optimizer (Lopez de Prado, 2016).

    Args:
        config: Optional configuration overrides.  Uses sensible
            defaults when ``None``.
    """

    def __init__(self, config: HRPConfig | None = None) -> None:
        self.config = config or HRPConfig()

        valid_linkage = {"single", "complete", "average", "ward"}
        if self.config.linkage_method not in valid_linkage:
            raise ValueError(
                f"linkage_method must be one of {sorted(valid_linkage)}, "
                f"got '{self.config.linkage_method}'"
            )

        valid_corr = {"pearson", "spearman"}
        if self.config.correlation_method not in valid_corr:
            raise ValueError(
                f"correlation_method must be one of {sorted(valid_corr)}, "
                f"got '{self.config.correlation_method}'"
            )

        if self.config.confidence_tilt_cap < 0:
            raise ValueError(
                f"confidence_tilt_cap must be >= 0, "
                f"got {self.config.confidence_tilt_cap}"
            )

        if self.config.max_weight <= 0:
            raise ValueError(
                f"max_weight must be > 0, got {self.config.max_weight}"
            )

        logger.info(
            "HRPOptimizer: linkage={}, correlation={}, tilt_cap={}, "
            "max_weight={}",
            self.config.linkage_method,
            self.config.correlation_method,
            self.config.confidence_tilt_cap,
            self.config.max_weight,
        )

    # ------------------------------------------------------------------
    # Covariance / correlation
    # ------------------------------------------------------------------

    def _compute_covariance(
        self, returns: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute covariance and correlation matrices from returns.

        Args:
            returns: DataFrame where each column is a ticker's return
                series.

        Returns:
            Tuple of ``(cov_matrix, corr_matrix)`` as numpy 2-D arrays.
        """
        arr = returns.to_numpy()

        cov = np.cov(arr, rowvar=False, ddof=1)
        # Ensure 2-D for single-asset edge case
        cov = np.atleast_2d(cov)

        # Warn about near-zero variance assets (quant-reviewer fix)
        diag = np.diag(cov)
        tickers = returns.columns
        for i, var in enumerate(diag):
            if var < 1e-8:
                logger.warning(
                    "Asset '{}' has near-zero variance ({:.2e}); "
                    "HRP will over-allocate to it",
                    tickers[i],
                    var,
                )

        if self.config.correlation_method == "spearman":
            from scipy.stats import spearmanr

            corr, _ = spearmanr(arr)
            corr = np.atleast_2d(corr)
        else:
            # Pearson from covariance
            std = np.sqrt(np.diag(cov))
            std[std == 0] = 1.0  # avoid division by zero
            corr = cov / np.outer(std, std)
            # Clip for numerical stability
            np.clip(corr, -1.0, 1.0, out=corr)

        return cov, corr

    # ------------------------------------------------------------------
    # Distance + clustering
    # ------------------------------------------------------------------

    def _compute_distance(self, corr: np.ndarray) -> np.ndarray:
        """Compute distance matrix from correlation (Lopez de Prado).

        Uses the standard HRP distance: ``d = sqrt(0.5 * (1 - corr))``.

        Args:
            corr: Correlation matrix (n × n).

        Returns:
            Condensed distance vector suitable for ``scipy.cluster``.
        """
        dist = np.sqrt(0.5 * (1.0 - corr))
        # Ensure exact zeros on diagonal (numerical safety)
        np.fill_diagonal(dist, 0.0)
        return squareform(dist, checks=False)

    def _cluster(self, dist_condensed: np.ndarray) -> np.ndarray:
        """Hierarchical agglomerative clustering.

        Args:
            dist_condensed: Condensed distance vector.

        Returns:
            Scipy linkage matrix (shape ``(n-1, 4)``).
        """
        return linkage(dist_condensed, method=self.config.linkage_method)

    # ------------------------------------------------------------------
    # Quasi-diagonalisation (seriation)
    # ------------------------------------------------------------------

    @staticmethod
    def _quasi_diagonalize(link: np.ndarray, n_assets: int) -> list[int]:
        """Reorder assets according to the dendrogram leaf order.

        Performs a recursive traversal of the linkage tree to produce
        a quasi-diagonal covariance matrix.  The leaf order follows
        scipy's linkage convention (left child, then right child).

        Args:
            link: Scipy linkage matrix.
            n_assets: Number of original assets (leaf nodes).

        Returns:
            List of original asset indices in seriated order.
        """
        # Start from the last merge (root)
        root = 2 * n_assets - 2

        def _recurse(node_id: int) -> list[int]:
            if node_id < n_assets:
                return [node_id]

            row = int(node_id - n_assets)
            left = int(link[row, 0])
            right = int(link[row, 1])

            left_leaves = _recurse(left)
            right_leaves = _recurse(right)

            return left_leaves + right_leaves

        return _recurse(root)

    # ------------------------------------------------------------------
    # Recursive bisection
    # ------------------------------------------------------------------

    @staticmethod
    def _get_cluster_var(
        cov: np.ndarray, indices: list[int]
    ) -> float:
        """Compute the variance of an inverse-variance-weighted cluster.

        This is the HRP cluster variance: the variance of the
        minimum-variance portfolio within the cluster (using diagonal
        elements only, as per Lopez de Prado).

        Args:
            cov: Full covariance matrix.
            indices: Asset indices belonging to the cluster.

        Returns:
            Cluster variance (scalar).
        """
        sub_cov = cov[np.ix_(indices, indices)]
        diag = np.diag(sub_cov)
        # Inverse-variance weights within cluster
        inv_diag = 1.0 / np.where(diag > 0, diag, 1e-10)
        w = inv_diag / inv_diag.sum()
        return float(w @ sub_cov @ w)

    def _recursive_bisection(
        self, cov: np.ndarray, sorted_indices: list[int]
    ) -> dict[int, float]:
        """Allocate weights via recursive bisection of the seriated assets.

        At each step, the seriated list is split in half and the
        allocation is distributed inversely proportional to each
        sub-cluster's variance.

        Args:
            cov: Full covariance matrix.
            sorted_indices: Asset indices in quasi-diagonal order.

        Returns:
            Mapping ``{asset_index: weight}``.
        """
        weights = {i: 1.0 for i in sorted_indices}

        clusters: list[list[int]] = [sorted_indices]

        while clusters:
            next_clusters: list[list[int]] = []
            for cluster in clusters:
                if len(cluster) == 1:
                    continue

                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                var_left = self._get_cluster_var(cov, left)
                var_right = self._get_cluster_var(cov, right)

                total_var = var_left + var_right
                if total_var == 0:
                    alpha = 0.5
                else:
                    # Allocate more to the lower-variance cluster
                    alpha = 1.0 - var_left / total_var

                for i in left:
                    weights[i] *= alpha
                for i in right:
                    weights[i] *= (1.0 - alpha)

                if len(left) > 1:
                    next_clusters.append(left)
                if len(right) > 1:
                    next_clusters.append(right)

            clusters = next_clusters

        return weights

    # ------------------------------------------------------------------
    # Clipping + normalisation
    # ------------------------------------------------------------------

    def _clip_and_normalise(
        self, weights: dict[str, float]
    ) -> dict[str, float]:
        """Clip weights to ``[min_weight, max_weight]`` and renormalise.

        Args:
            weights: Unnormalised or pre-tilt weights.

        Returns:
            Clipped and normalised weights summing to 1.0.
        """
        clipped: dict[str, float] = {}
        for ticker, w in weights.items():
            clipped[ticker] = max(
                self.config.min_weight,
                min(self.config.max_weight, w),
            )

        total = sum(clipped.values())
        if total > 0:
            return {k: v / total for k, v in clipped.items()}

        # Fallback: equal weight
        n = len(clipped)
        return {k: 1.0 / n for k in clipped}

    # ------------------------------------------------------------------
    # Confidence tilt
    # ------------------------------------------------------------------

    def _apply_confidence_tilt(
        self,
        weights: dict[str, float],
        confidences: dict[str, float],
    ) -> dict[str, float]:
        """Adjust HRP weights by agent confidence scores.

        Each weight is multiplied by ``(1 + cap * (confidence - 0.5))``,
        then clipped to ``[min_weight, max_weight]`` and renormalised
        to sum to 1.0.

        Args:
            weights: Raw HRP weights ``{ticker: weight}``.
            confidences: Agent confidence ``{ticker: float}`` in [0, 1].

        Returns:
            Tilted and normalised weights.
        """
        cap = self.config.confidence_tilt_cap
        tilted: dict[str, float] = {}

        for ticker, w in weights.items():
            conf = confidences.get(ticker, 0.5)  # neutral if missing
            conf = max(0.0, min(1.0, conf))  # clamp to [0, 1]
            multiplier = 1.0 + cap * (conf - 0.5)
            tilted[ticker] = w * multiplier

        return self._clip_and_normalise(tilted)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        returns: pl.DataFrame,
        confidences: dict[str, float] | None = None,
    ) -> HRPResult:
        """Run the full HRP pipeline.

        Args:
            returns: Polars DataFrame where each column is a ticker's
                daily return series.  Must have at least 1 ticker and
                ``>= 2`` rows.
            confidences: Optional ``{ticker: confidence}`` from agent
                decisions.  Values in ``[0, 1]``.  When ``None``, no
                confidence tilt is applied.

        Returns:
            HRPResult with final weights, raw weights, cluster order,
            and linkage matrix.

        Raises:
            ValueError: If the DataFrame has fewer than the minimum
                required assets or observations.
        """
        tickers = returns.columns
        n_assets = len(tickers)
        n_obs = returns.height

        if n_assets < _MIN_ASSETS:
            raise ValueError(
                f"Need at least {_MIN_ASSETS} asset(s), got {n_assets}"
            )

        if n_obs < _MIN_OBSERVATIONS:
            raise ValueError(
                f"Need at least {_MIN_OBSERVATIONS} observations, "
                f"got {n_obs}"
            )

        if n_obs < _RECOMMENDED_OBSERVATIONS:
            logger.warning(
                "Only {} observations (< {} recommended); covariance "
                "estimates may be noisy",
                n_obs,
                _RECOMMENDED_OBSERVATIONS,
            )

        # --- Single asset shortcut ---
        if n_assets == 1:
            w = {tickers[0]: 1.0}
            logger.info("Single asset — returning 100% weight to {}", tickers[0])
            return HRPResult(
                weights=w,
                raw_weights=w.copy(),
                cluster_order=list(tickers),
                linkage_matrix=[],
            )

        # --- Step 1: Covariance and correlation ---
        cov, corr = self._compute_covariance(returns)

        # --- Step 2: Distance matrix ---
        dist_condensed = self._compute_distance(corr)

        # --- Step 3: Hierarchical clustering ---
        link = self._cluster(dist_condensed)

        # --- Step 4: Quasi-diagonalise ---
        sorted_indices = self._quasi_diagonalize(link, n_assets)
        cluster_order = [tickers[i] for i in sorted_indices]

        logger.info("Cluster order: {}", cluster_order)

        # --- Step 5: Recursive bisection ---
        idx_weights = self._recursive_bisection(cov, sorted_indices)

        raw_weights: dict[str, float] = {}
        for idx, w in idx_weights.items():
            raw_weights[tickers[idx]] = w

        # --- Step 6: Confidence tilt ---
        if confidences is not None:
            final_weights = self._apply_confidence_tilt(raw_weights, confidences)
            logger.info(
                "Confidence tilt applied: raw={}, tilted={}",
                {k: round(v, 4) for k, v in raw_weights.items()},
                {k: round(v, 4) for k, v in final_weights.items()},
            )
        else:
            # Still apply clipping and normalisation for consistency
            final_weights = self._clip_and_normalise(raw_weights)

        # Store linkage as list[list[float]] for serialisation
        link_list = link.tolist()

        return HRPResult(
            weights=final_weights,
            raw_weights=raw_weights,
            cluster_order=cluster_order,
            linkage_matrix=link_list,
        )
