"""Decision engine — orchestrates the full investment pipeline.

Combines PatchTST predictions, LangGraph multi-agent debate, and
Hierarchical Risk Parity (HRP) allocation into a single pipeline
that outputs actionable portfolio decisions.

Pipeline::

    PostgreSQL (OHLCV) ──► daily returns (wide format)
                                │
    predictions.parquet ──► LangGraph debate ──► confidences
                                │                    │
                                └────► HRPOptimizer ◄┘
                                           │
                                    DecisionOutput
                                           │
                                    decisions.json

Usage::

    from src.portfolio.decision_engine import DecisionEngine

    engine = DecisionEngine()
    output = engine.run()
    print(output.decisions)

"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from sqlalchemy import Engine

from src.agents.state import FinalDecision
from src.config import load_tickers
from src.portfolio.hrp import HRPConfig, HRPOptimizer, HRPResult

# Default tickers — used when config/tickers.json is missing.
DEFAULT_TICKERS = ("SPY", "NVDA", "AAPL", "QQQ")


def _resolve_tickers(tickers: list[str] | None = None) -> list[str]:
    """Resolve tickers from argument, config file, or hardcoded fallback."""
    if tickers is not None:
        return tickers
    try:
        return load_tickers()
    except Exception:
        return list(DEFAULT_TICKERS)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TickerDecision:
    """Investment decision for a single ticker.

    Attributes:
        ticker: Asset symbol (e.g. ``"SPY"``).
        action: One of ``"BUY"``, ``"HOLD"``, ``"SELL"``.
        weight: Final portfolio weight from HRP.  Reduced for HOLD
            (weight * confidence), 0.0 for SELL.
        confidence: Agent confidence in ``[0, 1]``, 0.5 if no debate.
        reasoning: Portfolio Manager's synthesis.
        dissenting_view: Bear Agent's main objections.
    """

    ticker: str
    action: str
    weight: float
    confidence: float
    reasoning: str
    dissenting_view: str


@dataclass
class DecisionOutput:
    """Full output of the decision pipeline.

    Attributes:
        timestamp: ISO 8601 UTC timestamp.
        tickers: List of tickers analysed.
        decisions: Per-ticker decisions.
        hrp_raw_weights: HRP weights before confidence tilt.
        hrp_final_weights: HRP weights after confidence tilt.
        cluster_order: Dendrogram seriation order from HRP.
        metadata: Pipeline metadata for reproducibility.
    """

    timestamp: str
    tickers: list[str]
    decisions: list[TickerDecision]
    hrp_raw_weights: dict[str, float]
    hrp_final_weights: dict[str, float]
    cluster_order: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ClassificationConfig:
    """Configuration for percentile-based ticker classification.

    When PatchTST fallback is used (no debate), prob_up values are
    classified relative to the cross-sectional distribution rather
    than fixed absolute thresholds (which are unreachable given
    PatchTST's narrow output range ~[0.47, 0.57]).

    Attributes:
        sell_percentile: Bottom percentile fraction -> SELL.
            Default 0.15 (bottom 15%).
        hold_percentile: Percentile boundary between HOLD and BUY.
            Tickers in [sell_percentile, hold_percentile) -> HOLD.
            Default 0.40 (15-40% -> HOLD, top 60% -> BUY).
        min_tickers_for_percentile: Minimum number of fallback tickers
            required to use percentile classification.  Below this,
            falls back to absolute thresholds.  Default 5.
    """

    sell_percentile: float = 0.15
    hold_percentile: float = 0.40
    min_tickers_for_percentile: int = 5


# ---------------------------------------------------------------------------
# DecisionEngine
# ---------------------------------------------------------------------------


_DEFAULT_LOOKBACK_DAYS = 756  # ~3 years of trading days (CPCV-OOS validated)

# PatchTST fallback classification thresholds (absolute, used when
# too few tickers are available for percentile-based classification)
_PROB_UP_BUY = 0.5   # prob_up >= 0.5 -> BUY
_PROB_UP_SELL = 0.3   # prob_up < 0.3 -> SELL; [0.3, 0.5) -> HOLD


class DecisionEngine:
    """Orchestrates predictions, agent debate, and HRP allocation.

    Args:
        tickers: Asset symbols to process. Defaults to
            ``DEFAULT_TICKERS``.
        output_dir: Directory for JSON output.
        engine: SQLAlchemy engine for PostgreSQL. If ``None``, creates
            one from environment variables.
        hrp_config: HRP optimizer configuration. Uses defaults when
            ``None``.
        lookback_days: Number of trading days of returns to feed
            into HRP covariance estimation.
        classification_config: Percentile-based classification
            settings.  Uses defaults when ``None``.
    """

    def __init__(
        self,
        *,
        tickers: list[str] | None = None,
        output_dir: str = "data/outputs",
        engine: Engine | None = None,
        hrp_config: HRPConfig | None = None,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
        classification_config: ClassificationConfig | None = None,
    ) -> None:
        self.tickers = _resolve_tickers(tickers)
        self.output_dir = Path(output_dir)

        # Dynamic max_weight when no explicit config is provided
        # Uses CPCV-OOS validated params (ward + Ledoit-Wolf shrinkage)
        if hrp_config is None:
            n = len(self.tickers)
            dynamic_max = min(0.25, 2.0 / n)
            self.hrp_config = HRPConfig(
                linkage_method="ward",
                shrinkage=True,
                max_weight=dynamic_max,
            )
            logger.info(
                "HRP config: ward/shrinkage max_weight={:.4f} for {} tickers",
                dynamic_max,
                n,
            )
        else:
            self.hrp_config = hrp_config

        if engine is not None:
            self.engine = engine
        else:
            from src.utils.db import get_postgres_engine

            self.engine = get_postgres_engine()
        self.lookback_days = lookback_days
        self.classification_config = (
            classification_config or ClassificationConfig()
        )

        logger.info(
            "DecisionEngine: tickers={}, lookback={}, output={}",
            self.tickers,
            self.lookback_days,
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_ohlcv(self) -> pl.DataFrame:
        """Load OHLCV data from PostgreSQL via PredictionPipeline.

        Returns:
            Raw OHLCV DataFrame with columns: date, ticker, open, high,
            low, close, volume, adj_close.

        Raises:
            ValueError: If no data found for configured tickers.
        """
        from src.models.predict import PredictionPipeline

        pipeline = PredictionPipeline(
            engine=self.engine, tickers=self.tickers
        )
        return pipeline.load_ohlcv()

    def _compute_returns(self, ohlcv: pl.DataFrame) -> pl.DataFrame:
        """Compute daily log returns and pivot to wide format for HRP.

        Args:
            ohlcv: Raw OHLCV DataFrame (long format with ``ticker``
                column).

        Returns:
            Wide DataFrame where each column is a ticker's daily log
            return series, trimmed to ``lookback_days`` most recent
            rows.
        """
        returns = (
            ohlcv.sort(["ticker", "date"])
            .with_columns(
                pl.col("close")
                .log()
                .diff()
                .over("ticker")
                .alias("log_return")
            )
            .drop_nulls(subset=["log_return"])
        )

        # Pivot to wide format: rows=dates, cols=tickers
        wide = returns.pivot(
            on="ticker",
            index="date",
            values="log_return",
        ).sort("date")

        # Align tickers: skip leading rows where any ticker is missing,
        # then fill remaining interior nulls with 0.0.  This matches
        # the walk-forward backtester logic and avoids the old
        # drop_nulls() which was too aggressive (dropped entire rows
        # if ANY ticker had a gap, silently shrinking the sample).
        ticker_cols = [c for c in wide.columns if c != "date"]
        all_present = pl.all_horizontal(
            pl.col(c).is_not_null() for c in ticker_cols
        )
        first_complete_idx = (
            wide.with_row_index("_idx").filter(all_present)["_idx"]
        )
        if first_complete_idx.len() > 0:
            wide = wide.slice(int(first_complete_idx[0]))
        wide = wide.fill_null(0.0)

        # Trim to lookback window
        if wide.height > self.lookback_days:
            wide = wide.tail(self.lookback_days)

        result = wide.select(ticker_cols)

        logger.info(
            "Returns matrix: {} obs × {} tickers (lookback={})",
            result.height,
            result.width,
            self.lookback_days,
        )
        return result

    # ------------------------------------------------------------------
    # Agent debate
    # ------------------------------------------------------------------

    def _run_debate(
        self,
    ) -> tuple[list[FinalDecision], dict[str, dict]]:
        """Run the LangGraph multi-agent debate.

        Returns:
            Tuple of:
                - List of ``FinalDecision`` per ticker (empty on failure).
                - Dict of full graph states per ticker (empty on failure).
        """
        try:
            from src.agents.graph import run_agent_debate

            decisions, full_states = run_agent_debate(
                tickers=self.tickers
            )
            logger.info(
                "Agent debate complete: {} decisions",
                len(decisions),
            )
            return decisions, full_states
        except Exception as exc:
            logger.warning(
                "Agent debate failed ({}); proceeding with HRP "
                "without confidence tilt",
                exc,
            )
            return [], {}

    def _save_debate_history(
        self, full_states: dict[str, dict]
    ) -> Path | None:
        """Persist agent debate states for dashboard consumption.

        Saves reports, debate_log, and predictions per ticker to
        ``debate_history.json``.

        Args:
            full_states: Dict mapping ticker to full LangGraph state.

        Returns:
            Path to the saved file, or None if states are empty.
        """
        if not full_states:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "debate_history.json"

        # Extract serialisable subset of each state
        history: dict[str, Any] = {}
        for ticker, state in full_states.items():
            history[ticker] = {
                "reports": state.get("reports", []),
                "debate_log": state.get("debate_log", []),
                "predictions": state.get("predictions", {}),
                "news_context": state.get("news_context", []),
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info("Debate history saved to {}", path)
        return path

    def _extract_confidences(
        self,
        decisions: list[FinalDecision],
        fallback: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Extract ``{ticker: confidence}`` from agent decisions.

        Tickers not present in decisions use ``fallback`` prob_up
        if available, otherwise default to 0.5 (neutral).

        Args:
            decisions: Agent debate results.
            fallback: PatchTST prob_up mapping for missing tickers.

        Returns:
            Confidence mapping for all configured tickers.
        """
        conf_map: dict[str, float] = {}
        for d in decisions:
            conf_map[d["ticker"]] = d["confidence"]

        # Fill missing tickers with fallback or neutral
        for ticker in self.tickers:
            if ticker not in conf_map:
                if fallback is not None and ticker in fallback:
                    conf_map[ticker] = fallback[ticker]
                else:
                    conf_map[ticker] = 0.5

        return conf_map

    def _load_predictions(self) -> dict[str, float] | None:
        """Load PatchTST prob_up predictions as fallback confidences.

        Reads ``predictions.parquet`` from the output directory and
        returns a mapping of ``{ticker: prob_up}`` filtered to
        ``self.tickers``.

        Returns:
            Mapping of ticker to prob_up, or ``None`` if file is
            missing or unreadable.
        """
        path = self.output_dir / "predictions.parquet"
        try:
            df = pl.read_parquet(path)
            preds: dict[str, float] = {}
            for row in df.iter_rows(named=True):
                ticker = row.get("ticker")
                prob_up = row.get("prob_up")
                if ticker in self.tickers and prob_up is not None:
                    preds[ticker] = float(prob_up)
            if preds:
                logger.info(
                    "Loaded {} PatchTST predictions as fallback",
                    len(preds),
                )
                return preds
            return None
        except FileNotFoundError:
            logger.info(
                "predictions.parquet not found — no fallback available"
            )
            return None
        except Exception as exc:
            logger.warning(
                "Failed to load predictions.parquet: {}", exc
            )
            return None

    @staticmethod
    def _compute_percentile_thresholds(
        prob_ups: dict[str, float],
        config: ClassificationConfig,
    ) -> tuple[float, float]:
        """Compute dynamic SELL and BUY thresholds from percentiles.

        Args:
            prob_ups: Mapping of ticker to prob_up for tickers that
                need fallback classification.
            config: Classification configuration with percentile levels.

        Returns:
            Tuple of ``(sell_threshold, buy_threshold)`` derived from
            the cross-sectional distribution of prob_up values.
        """
        values = np.array(list(prob_ups.values()))
        sell_thresh = float(
            np.percentile(values, config.sell_percentile * 100)
        )
        buy_thresh = float(
            np.percentile(values, config.hold_percentile * 100)
        )
        return sell_thresh, buy_thresh

    def _classify_tickers(
        self,
        debate: list[FinalDecision],
        fallback: dict[str, float] | None = None,
    ) -> tuple[list[str], list[str], list[str], tuple[float, float]]:
        """Classify tickers into BUY, HOLD, and SELL groups.

        Uses debate actions when available.  For tickers without debate
        results, uses PatchTST ``prob_up`` thresholds when ``fallback``
        is provided, otherwise defaults to BUY.

        When enough fallback tickers are available
        (>= ``min_tickers_for_percentile``), thresholds are computed
        from the cross-sectional percentile distribution of prob_up
        values instead of fixed absolute values.

        Args:
            debate: Agent debate results (may be empty).
            fallback: PatchTST prob_up mapping for classification of
                tickers missing from the debate.

        Returns:
            Tuple of ``(buy_tickers, hold_tickers, sell_tickers,
            thresholds)`` where ``thresholds`` is
            ``(sell_threshold, buy_threshold)``.
        """
        decision_map: dict[str, FinalDecision] = {}
        for d in debate:
            decision_map[d["ticker"]] = d

        # Determine which tickers need fallback classification
        fallback_tickers: dict[str, float] = {}
        if fallback is not None:
            for ticker in self.tickers:
                if ticker not in decision_map and ticker in fallback:
                    fallback_tickers[ticker] = fallback[ticker]

        # Compute thresholds: percentile-based when enough data
        cfg = self.classification_config
        if len(fallback_tickers) >= cfg.min_tickers_for_percentile:
            sell_thresh, buy_thresh = (
                self._compute_percentile_thresholds(
                    fallback_tickers, cfg
                )
            )
            logger.info(
                "Percentile thresholds: SELL<{:.4f}, BUY>={:.4f} "
                "({} fallback tickers, p_sell={}, p_hold={})",
                sell_thresh,
                buy_thresh,
                len(fallback_tickers),
                cfg.sell_percentile,
                cfg.hold_percentile,
            )
        else:
            sell_thresh = _PROB_UP_SELL
            buy_thresh = _PROB_UP_BUY

        buy: list[str] = []
        hold: list[str] = []
        sell: list[str] = []

        for ticker in self.tickers:
            if ticker in decision_map:
                action = decision_map[ticker]["action"]
                if action == "BUY":
                    buy.append(ticker)
                elif action == "HOLD":
                    hold.append(ticker)
                else:
                    sell.append(ticker)
            elif ticker in fallback_tickers:
                prob_up = fallback_tickers[ticker]
                if prob_up >= buy_thresh:
                    buy.append(ticker)
                elif prob_up < sell_thresh:
                    sell.append(ticker)
                else:
                    hold.append(ticker)
            else:
                buy.append(ticker)

        logger.info(
            "Classification: {} BUY, {} HOLD, {} SELL",
            len(buy),
            len(hold),
            len(sell),
        )
        return buy, hold, sell, (sell_thresh, buy_thresh)

    # ------------------------------------------------------------------
    # HRP allocation
    # ------------------------------------------------------------------

    def _run_hrp(
        self,
        returns: pl.DataFrame,
        confidences: dict[str, float] | None,
    ) -> HRPResult:
        """Run HRP optimizer on the returns matrix.

        Args:
            returns: Wide DataFrame of daily log returns.
            confidences: Agent confidence scores per ticker, or
                ``None`` for no tilt.

        Returns:
            HRPResult with weights, cluster order, and linkage matrix.
        """
        optimizer = HRPOptimizer(config=self.hrp_config)
        return optimizer.optimize(returns, confidences=confidences)

    # ------------------------------------------------------------------
    # Merge actions + weights
    # ------------------------------------------------------------------

    def _merge_actions(
        self,
        debate: list[FinalDecision],
        final_weights: dict[str, float],
        sell_tickers: list[str],
        *,
        hold_tickers: list[str] | None = None,
        confidences: dict[str, float] | None = None,
    ) -> list[TickerDecision]:
        """Build TickerDecision list from pre-computed weights.

        Weight computation (BUY = HRP, HOLD = HRP * confidence,
        SELL = 0) is handled in ``run()``.  This method only
        assembles ``TickerDecision`` objects.

        Args:
            debate: Agent debate results (may be empty).
            final_weights: Pre-computed portfolio weights.
            sell_tickers: Tickers classified as SELL.
            hold_tickers: Tickers classified as HOLD.
            confidences: Confidence mapping for non-debate tickers.

        Returns:
            List of ``TickerDecision`` for each ticker.
        """
        decision_map: dict[str, FinalDecision] = {}
        for d in debate:
            decision_map[d["ticker"]] = d

        _hold = set(hold_tickers) if hold_tickers else set()
        _sell = set(sell_tickers)

        result: list[TickerDecision] = []
        for ticker in self.tickers:
            if ticker in decision_map:
                d = decision_map[ticker]
                result.append(
                    TickerDecision(
                        ticker=ticker,
                        action=d["action"],
                        weight=round(final_weights.get(ticker, 0.0), 6),
                        confidence=d["confidence"],
                        reasoning=d["reasoning"],
                        dissenting_view=d["dissenting_view"],
                    )
                )
            else:
                if ticker in _sell:
                    action = "SELL"
                elif ticker in _hold:
                    action = "HOLD"
                else:
                    action = "BUY"
                conf = (
                    confidences.get(ticker, 0.5)
                    if confidences is not None
                    else 0.5
                )
                reasoning = (
                    f"PatchTST signal (prob_up={conf:.2f})"
                    if confidences is not None and ticker in confidences
                    else "No agent debate available"
                )
                result.append(
                    TickerDecision(
                        ticker=ticker,
                        action=action,
                        weight=round(final_weights.get(ticker, 0.0), 6),
                        confidence=conf,
                        reasoning=reasoning,
                        dissenting_view="",
                    )
                )

        return result

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _build_output(
        self,
        decisions: list[TickerDecision],
        hrp_result: HRPResult,
        n_observations: int,
        *,
        invested_fraction: float = 1.0,
        confidence_source: str = "none",
        n_buy: int = 0,
        n_hold: int = 0,
        n_sell: int = 0,
        classification_method: str = "absolute",
        sell_threshold: float = _PROB_UP_SELL,
        buy_threshold: float = _PROB_UP_BUY,
    ) -> DecisionOutput:
        """Build the final DecisionOutput.

        Args:
            decisions: Merged ticker decisions.
            hrp_result: HRP optimization result.
            n_observations: Number of return observations used.
            invested_fraction: Sum of final weights (< 1.0 when
                HOLD/SELL tickers are present).
            confidence_source: Origin of confidences used for HRP
                tilt (``"debate"``, ``"debate+patchtst_fallback"``,
                ``"patchtst"``, or ``"none"``).
            n_buy: Number of BUY tickers.
            n_hold: Number of HOLD tickers.
            n_sell: Number of SELL tickers.
            classification_method: How tickers were classified
                (``"percentile"`` or ``"absolute"``).
            sell_threshold: Computed SELL prob_up threshold.
            buy_threshold: Computed BUY prob_up threshold.

        Returns:
            Complete DecisionOutput ready for serialisation.
        """
        config = self.hrp_config

        metadata: dict[str, Any] = {
            "schema_version": "1.2",
            "hrp_config": {
                "linkage_method": config.linkage_method,
                "correlation_method": config.correlation_method,
                "shrinkage": config.shrinkage,
                "confidence_tilt_cap": config.confidence_tilt_cap,
                "min_weight": config.min_weight,
                "max_weight": config.max_weight,
            },
            "n_observations": n_observations,
            "lookback_days": self.lookback_days,
            "invested_fraction": round(invested_fraction, 6),
            "confidence_source": confidence_source,
            "n_buy": n_buy,
            "n_hold": n_hold,
            "n_sell": n_sell,
            "classification": {
                "method": classification_method,
                "sell_threshold": round(sell_threshold, 6),
                "buy_threshold": round(buy_threshold, 6),
                "sell_percentile": self.classification_config.sell_percentile,
                "hold_percentile": self.classification_config.hold_percentile,
            },
        }

        return DecisionOutput(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tickers=list(self.tickers),
            decisions=decisions,
            hrp_raw_weights={
                k: round(v, 6)
                for k, v in hrp_result.raw_weights.items()
            },
            hrp_final_weights={
                k: round(v, 6)
                for k, v in hrp_result.weights.items()
            },
            cluster_order=hrp_result.cluster_order,
            metadata=metadata,
        )

    def _save_json(self, output: DecisionOutput) -> Path:
        """Save DecisionOutput to a JSON file.

        Args:
            output: The decision output to persist.

        Returns:
            Path to the saved JSON file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "decisions.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("Decisions saved to {}", path)
        return path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> DecisionOutput:
        """Execute the full decision pipeline.

        Steps:
            1. Load OHLCV from PostgreSQL
            2. Compute daily log returns (wide format)
            3. Run LangGraph agent debate (graceful degradation)
            4. Load PatchTST predictions as fallback
            5. Extract confidences (debate with fallback prob_up)
            6. Classify tickers: BUY / HOLD / SELL
            7. Filter returns + confidences to investable (BUY + HOLD)
            8. Run HRP on investable subset
            9. HOLD scaling + max_weight enforcement
            10. Merge actions + save

        Returns:
            DecisionOutput with weights, actions, and metadata.

        Raises:
            ValueError: If OHLCV data is missing or insufficient.
        """
        # Step 1-2: Load and compute returns
        logger.info("Step 1: Loading OHLCV data")
        ohlcv = self._load_ohlcv()

        logger.info("Step 2: Computing returns matrix")
        returns = self._compute_returns(ohlcv)
        n_observations = returns.height

        if n_observations < 2:
            raise ValueError(
                f"Need at least 2 return observations for HRP, "
                f"got {n_observations}"
            )

        # Step 3: Agent debate
        logger.info("Step 3: Running agent debate")
        debate_results, debate_states = self._run_debate()
        self._save_debate_history(debate_states)

        # Step 4: Load predictions as fallback
        fallback = self._load_predictions()

        # Step 5: Extract confidences
        if debate_results:
            confidences = self._extract_confidences(
                debate_results, fallback=fallback
            )
            confidence_source = (
                "debate+patchtst_fallback" if fallback else "debate"
            )
        elif fallback:
            confidences = self._extract_confidences(
                [], fallback=fallback
            )
            confidence_source = "patchtst"
        else:
            confidences = None
            confidence_source = "none"
        logger.info("Confidence source: {}", confidence_source)

        # Step 6: Classify tickers (percentile-based when enough data)
        buy_tickers, hold_tickers, sell_tickers, (sell_thresh, buy_thresh) = (
            self._classify_tickers(debate_results, fallback=fallback)
        )

        # Step 7: Filter to investable (BUY + HOLD)
        investable = buy_tickers + hold_tickers
        available = [c for c in investable if c in returns.columns]

        if available:
            returns_subset = returns.select(available)
            # Only pass BUY confidences to HRP — HOLD tickers default
            # to 0.5 (neutral) inside HRP's tilt to minimise the
            # interaction with Step 9 HOLD scaling.  A small residual
            # tilt may remain when BUY wmean > 0.5 (capped at ~20%
            # of the wmean deviation).
            conf_subset = (
                {
                    t: confidences[t]
                    for t in available
                    if t not in hold_tickers
                }
                if confidences is not None
                else None
            )

            # Step 8: Run HRP on investable subset
            logger.info(
                "Step 8: Running HRP on {} investable tickers",
                len(available),
            )
            hrp_result = self._run_hrp(returns_subset, conf_subset)
        else:
            # All SELL — empty HRP result
            logger.warning("No investable tickers — all SELL")
            hrp_result = HRPResult(
                weights={},
                raw_weights={},
                cluster_order=[],
                linkage_matrix=[],
            )

        # Step 9: Compute final weights with HOLD scaling
        debate_set = {d["ticker"] for d in debate_results}
        final_weights: dict[str, float] = {}
        for ticker in self.tickers:
            if ticker in sell_tickers:
                final_weights[ticker] = 0.0
            elif ticker in hold_tickers:
                hrp_w = hrp_result.weights.get(ticker, 0.0)
                conf = (
                    confidences.get(ticker, 0.5)
                    if confidences is not None
                    else 0.5
                )
                if ticker not in debate_set:
                    # Fallback HOLD: ramp conf from [sell_thresh,
                    # buy_thresh) to [0, 1) so the boundary is smooth.
                    ramp_range = buy_thresh - sell_thresh
                    if ramp_range > 0:
                        conf = (conf - sell_thresh) / ramp_range
                    else:
                        conf = 0.5
                    conf = max(0.0, min(conf, 1.0))
                final_weights[ticker] = hrp_w * conf
            else:
                # BUY — HRP weight directly
                final_weights[ticker] = hrp_result.weights.get(
                    ticker, 0.0
                )

        # Re-enforce max_weight with the same feasibility guard
        # that HRP uses internally: max_weight >= 1/n_investable,
        # otherwise a small investable set gets clipped to near-zero.
        n_investable = max(len(available), 1)
        effective_max = max(
            self.hrp_config.max_weight, 1.0 / n_investable
        )
        for ticker in investable:
            if final_weights.get(ticker, 0.0) > effective_max:
                final_weights[ticker] = effective_max

        # Step 10: Merge + save
        logger.info("Step 10: Merging actions and weights")
        decisions = self._merge_actions(
            debate_results,
            final_weights,
            sell_tickers,
            hold_tickers=hold_tickers,
            confidences=confidences,
        )

        invested_fraction = sum(final_weights.values())
        # Determine classification method used
        n_fallback_classified = sum(
            1
            for t in self.tickers
            if t not in debate_set
            and fallback is not None
            and t in fallback
        )
        classification_method = (
            "percentile"
            if n_fallback_classified
            >= self.classification_config.min_tickers_for_percentile
            else "absolute"
        )
        output = self._build_output(
            decisions,
            hrp_result,
            n_observations,
            invested_fraction=invested_fraction,
            confidence_source=confidence_source,
            n_buy=len(buy_tickers),
            n_hold=len(hold_tickers),
            n_sell=len(sell_tickers),
            classification_method=classification_method,
            sell_threshold=sell_thresh,
            buy_threshold=buy_thresh,
        )
        self._save_json(output)

        # Summary log
        for d in decisions:
            logger.info(
                "{}: {} weight={:.4f} conf={:.2f}",
                d.ticker,
                d.action,
                d.weight,
                d.confidence,
            )
        logger.info(
            "Invested fraction: {:.4f} ({} BUY, {} HOLD, {} SELL)",
            invested_fraction,
            len(buy_tickers),
            len(hold_tickers),
            len(sell_tickers),
        )

        return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``make decide``."""
    engine_inst = DecisionEngine()
    engine_inst.run()


if __name__ == "__main__":
    main()
