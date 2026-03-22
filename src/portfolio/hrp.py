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
        shrinkage: When ``True``, use Ledoit-Wolf shrinkage estimator
            instead of sample covariance.  Produces more stable weights
            and lower turnover, especially when ``n_obs`` is close to
            ``n_assets``.
        confidence_tilt_cap: Maximum adjustment factor applied to raw
            HRP weights based on agent confidence. The maximum tilt 
            depends on the spread of confidences around their 
            cross-sectional mean.
        min_weight: Floor per asset after tilt (before renormalisation).
        max_weight: Cap per asset after tilt (before renormalisation).
            Aligned with ``MAX_SINGLE_WEIGHT = 0.25`` from the agent
            state definitions.
    """

    linkage_method: str = "single"
    correlation_method: str = "pearson"
    shrinkage: bool = False
    confidence_tilt_cap: float = 0.20
    min_weight: float = 0.0
    max_weight: float = 0.25
    turnover_threshold: float = 0.02  # Só rebalanceia o ativo se a mudança for > 2%


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
            "HRPOptimizer: linkage={}, correlation={}, shrinkage={}, "
            "tilt_cap={}, max_weight={}",
            self.config.linkage_method,
            self.config.correlation_method,
            self.config.shrinkage,
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

        When ``config.shrinkage`` is ``True``, uses the Ledoit-Wolf
        shrinkage estimator (``sklearn.covariance.LedoitWolf``) instead
        of the sample covariance.  This produces a better-conditioned
        covariance matrix, especially when the number of observations is
        close to the number of assets.

        Args:
            returns: DataFrame where each column is a ticker's return
                series.

        Returns:
            Tuple of ``(cov_matrix, corr_matrix)`` as numpy 2-D arrays.
        """
        arr = returns.to_numpy()

        if self.config.shrinkage:
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf().fit(arr)
            cov = lw.covariance_
            logger.info(
                "Ledoit-Wolf shrinkage applied (shrinkage_={:.4f})",
                lw.shrinkage_,
            )
        else:
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
        inv_diag = 1.0 / np.where(diag > 0, diag, 1e-6)
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
    # Confidence tilt
    # ------------------------------------------------------------------

    def _apply_confidence_tilt(
        self,
        weights: dict[str, float],
        confidences: dict[str, float],
    ) -> dict[str, float]:
        """Ajusta os pesos do HRP baseado na confiança transversal do agente.

        Usa a média ponderada pelos pesos como ponto neutro, garantindo que
        sum(tilted) == sum(weights) exatamente (sum-preserving tilt).
        """
        cap = self.config.confidence_tilt_cap
        tilted: dict[str, float] = {}

        # 1. Resolver confiança por ativo (clamp [0,1], default 0.5 = neutro)
        resolved: dict[str, float] = {}
        for ticker in weights:
            raw = confidences.get(ticker, 0.5)
            resolved[ticker] = max(0.0, min(1.0, raw))

        # 2. Média ponderada pelos pesos (ponto neutro sum-preserving)
        #    sum(w_i * (1 + cap*(c_i - wmean))) = sum(w_i) exatamente
        #    quando wmean = sum(w_i * c_i) / sum(w_i).
        w_sum = sum(weights.values())
        if w_sum > 1e-10:
            wmean_conf = sum(
                weights[t] * resolved[t] for t in weights
            ) / w_sum
        else:
            wmean_conf = 0.5

        # 3. Aplicar o multiplicador centrado na média ponderada
        for ticker, w in weights.items():
            multiplier = 1.0 + cap * (resolved[ticker] - wmean_conf)
            tilted[ticker] = w * max(0.0, multiplier)

        return tilted

    # ------------------------------------------------------------------
    # Constraints & normalise
    # ------------------------------------------------------------------

    def _apply_constraints_and_normalise(
        self, 
        target_weights: dict[str, float],
        previous_weights: dict[str, float] | None = None
    ) -> dict[str, float]:
        """
        Otimizador unificado via Waterfilling com limites dinâmicos.
        Garante soma = 1.0, respeita min/max_weight globais e aplica 
        latching (inércia) travando os limites de ativos que não romperam o threshold.
        """
        tickers = list(target_weights.keys())
        n_assets = len(tickers)

        # Guard: max_weight deve ser >= 1/n para o problema ser feasível
        effective_max = max(self.config.max_weight, 1.0 / n_assets)

        # 1. Definição dos Limites Dinâmicos (Dynamic Bounds)
        bounds: dict[str, tuple[float, float]] = {}
        tau = getattr(self.config, 'turnover_threshold', 0.0)

        for t in tickers:
            w_target = target_weights[t]
            w_prev = previous_weights.get(t) if previous_weights else None

            # Se a mudança modular estiver dentro do threshold, trava o ativo no peso anterior
            if w_prev is not None and abs(w_target - w_prev) < tau:
                bounds[t] = (w_prev, w_prev)
            else:
                bounds[t] = (self.config.min_weight, effective_max)

        # 2. Normalização inicial (Baseline)
        total_raw = sum(target_weights.values())
        if total_raw <= 0:
            return {t: 1.0 / n_assets for t in tickers}

        w = {t: val / total_raw for t, val in target_weights.items()}

        # 3. Loop de redistribuição iterativa (Waterfilling com limites individuais)
        #    Runs until all weights are within bounds and sum to 1.0.
        #    The final normalization is folded INTO the loop to avoid
        #    the classic bug where post-loop renorm pushes capped weights
        #    above max_weight.
        max_iterations = 100
        for iteration in range(max_iterations):
            out_of_bounds = False
            total_excess = 0.0
            free_weight_sum = 0.0
            free_assets = []

            for t, val in w.items():
                lo, hi = bounds[t]

                if val > hi + 1e-8:
                    total_excess += val - hi
                    w[t] = hi
                    out_of_bounds = True
                elif val < lo - 1e-8:
                    total_excess += val - lo
                    w[t] = lo
                    out_of_bounds = True
                else:
                    free_weight_sum += val
                    free_assets.append(t)

            if not out_of_bounds or not free_assets:
                break

            # Redistribute excess to free assets (proportional)
            if free_weight_sum > 0:
                for t in free_assets:
                    w[t] = max(0.0, w[t] + total_excess * (w[t] / free_weight_sum))
            else:
                for t in free_assets:
                    w[t] += total_excess / len(free_assets)

        # 4. Final normalization — only scale FREE assets to avoid
        #    pushing capped assets above max_weight.
        total = sum(w.values())
        if abs(total - 1.0) > 1e-8:
            capped = {t for t in tickers if abs(w[t] - bounds[t][1]) < 1e-8
                       or abs(w[t] - bounds[t][0]) < 1e-8}
            free = [t for t in tickers if t not in capped]
            free_sum = sum(w[t] for t in free)
            deficit = 1.0 - total
            if free and free_sum > 1e-10:
                for t in free:
                    w[t] += deficit * (w[t] / free_sum)
            else:
                # All assets capped — uniform scale as last resort
                w = {t: v / total for t, v in w.items()}
            logger.debug("Post-waterfill adjust: deficit was {:.6f}", deficit)

        return w
    

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        returns: pl.DataFrame,
        confidences: dict[str, float] | None = None,
        previous_weights: dict[str, float] | None = None,
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
            final_weights = raw_weights.copy()

        # --- Step 7: Otimização final com restrições e inércia ---
        final_weights = self._apply_constraints_and_normalise(
            target_weights=final_weights,
            previous_weights=previous_weights
        )

        # Store linkage as list[list[float]] for serialisation
        link_list = link.tolist()

        return HRPResult(
            weights=final_weights,
            raw_weights=raw_weights,
            cluster_order=cluster_order,
            linkage_matrix=link_list,
        )
