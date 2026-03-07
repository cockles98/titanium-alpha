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

import polars as pl
from loguru import logger
from sqlalchemy import Engine

from src.agents.state import FinalDecision
from src.portfolio.hrp import HRPConfig, HRPOptimizer, HRPResult

# Default tickers (mirrors src.data.ingestion.DEFAULT_TICKERS to avoid
# importing yfinance at module level).
DEFAULT_TICKERS = ("SPY", "NVDA", "AAPL", "QQQ")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TickerDecision:
    """Investment decision for a single ticker.

    Attributes:
        ticker: Asset symbol (e.g. ``"SPY"``).
        action: One of ``"BUY"``, ``"HOLD"``, ``"SELL"``.
        weight: Final portfolio weight from HRP (0.0 for HOLD/SELL).
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


# ---------------------------------------------------------------------------
# DecisionEngine
# ---------------------------------------------------------------------------


_DEFAULT_LOOKBACK_DAYS = 504  # ~2 years of trading days


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
    """

    def __init__(
        self,
        *,
        tickers: list[str] | None = None,
        output_dir: str = "data/outputs",
        engine: Engine | None = None,
        hrp_config: HRPConfig | None = None,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        self.tickers = tickers or list(DEFAULT_TICKERS)
        self.output_dir = Path(output_dir)
        self.hrp_config = hrp_config

        if engine is not None:
            self.engine = engine
        else:
            from src.utils.db import get_postgres_engine

            self.engine = get_postgres_engine()
        self.lookback_days = lookback_days

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

        # Drop any rows with nulls (tickers with different date ranges)
        wide = wide.drop_nulls()

        # Trim to lookback window
        if wide.height > self.lookback_days:
            wide = wide.tail(self.lookback_days)

        # Remove date column — HRP only needs the numeric matrix
        ticker_cols = [c for c in wide.columns if c != "date"]
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
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
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
        self, decisions: list[FinalDecision]
    ) -> dict[str, float]:
        """Extract ``{ticker: confidence}`` from agent decisions.

        Tickers not present in decisions get confidence 0.5 (neutral).

        Args:
            decisions: Agent debate results.

        Returns:
            Confidence mapping for all configured tickers.
        """
        conf_map: dict[str, float] = {}
        for d in decisions:
            conf_map[d["ticker"]] = d["confidence"]

        # Fill missing tickers with neutral confidence
        for ticker in self.tickers:
            if ticker not in conf_map:
                conf_map[ticker] = 0.5

        return conf_map

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
        hrp_result: HRPResult,
    ) -> list[TickerDecision]:
        """Combine agent actions with HRP weights.

        HOLD and SELL tickers have their HRP weight redistributed
        proportionally among BUY tickers. If no debate occurred, all
        tickers default to BUY with HRP weights.

        Args:
            debate: Agent debate results (may be empty).
            hrp_result: HRP optimization result.

        Returns:
            List of ``TickerDecision`` for each ticker.
        """
        # Build lookup from debate
        decision_map: dict[str, FinalDecision] = {}
        for d in debate:
            decision_map[d["ticker"]] = d

        # Classify tickers
        buy_tickers: list[str] = []
        non_buy_tickers: list[str] = []

        for ticker in self.tickers:
            if ticker in decision_map:
                action = decision_map[ticker]["action"]
                if action == "BUY":
                    buy_tickers.append(ticker)
                else:
                    non_buy_tickers.append(ticker)
            else:
                # No debate result → default to BUY
                buy_tickers.append(ticker)

        # Redistribute non-BUY weight to BUY tickers proportionally
        final_weights: dict[str, float] = {}

        if buy_tickers:
            # Sum of HRP weights for BUY tickers
            buy_weight_sum = sum(
                hrp_result.weights.get(t, 0.0) for t in buy_tickers
            )

            if buy_weight_sum > 0:
                # Scale BUY tickers to sum to 1.0
                for t in buy_tickers:
                    final_weights[t] = (
                        hrp_result.weights.get(t, 0.0) / buy_weight_sum
                    )
            else:
                # Equal weight fallback
                for t in buy_tickers:
                    final_weights[t] = 1.0 / len(buy_tickers)

            for t in non_buy_tickers:
                final_weights[t] = 0.0
        else:
            # All tickers are HOLD/SELL — equal weight (cash-like)
            for t in self.tickers:
                final_weights[t] = 0.0

        # Build TickerDecision list
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
                result.append(
                    TickerDecision(
                        ticker=ticker,
                        action="BUY",
                        weight=round(final_weights.get(ticker, 0.0), 6),
                        confidence=0.5,
                        reasoning="No agent debate available",
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
    ) -> DecisionOutput:
        """Build the final DecisionOutput.

        Args:
            decisions: Merged ticker decisions.
            hrp_result: HRP optimization result.
            n_observations: Number of return observations used.

        Returns:
            Complete DecisionOutput ready for serialisation.
        """
        config = self.hrp_config or HRPConfig()

        metadata: dict[str, Any] = {
            "schema_version": "1.0",
            "hrp_config": {
                "linkage_method": config.linkage_method,
                "correlation_method": config.correlation_method,
                "confidence_tilt_cap": config.confidence_tilt_cap,
                "min_weight": config.min_weight,
                "max_weight": config.max_weight,
            },
            "n_observations": n_observations,
            "lookback_days": self.lookback_days,
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
            4. Extract confidences from debate
            5. Run HRP optimizer with confidence tilt
            6. Merge agent actions with HRP weights
            7. Save to ``decisions.json``

        Returns:
            DecisionOutput with weights, actions, and metadata.

        Raises:
            ValueError: If OHLCV data is missing or insufficient.
            FileNotFoundError: If ``predictions.parquet`` is missing
                (required by agent debate).
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

        # Step 3b: Persist debate history for dashboard
        self._save_debate_history(debate_states)

        # Step 4: Extract confidences
        if debate_results:
            confidences = self._extract_confidences(debate_results)
            logger.info("Confidences: {}", confidences)
        else:
            confidences = None
            logger.info("No debate results — HRP without tilt")

        # Step 5: HRP allocation
        logger.info("Step 5: Running HRP optimizer")
        hrp_result = self._run_hrp(returns, confidences)
        logger.info(
            "HRP weights: {}",
            {k: round(v, 4) for k, v in hrp_result.weights.items()},
        )

        # Step 6: Merge actions + weights
        logger.info("Step 6: Merging actions and weights")
        decisions = self._merge_actions(debate_results, hrp_result)

        # Step 7: Build and save output
        output = self._build_output(decisions, hrp_result, n_observations)
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
