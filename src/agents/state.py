"""Shared state definitions for the LangGraph investment agent graph.

Defines the TypedDicts that flow through the multi-agent pipeline:
    START → load_context → TechnicalAnalyst → FundamentalistAnalyst
          → BearAgent → PortfolioManager → END

Each agent reads previous reports from ``InvestmentState.reports`` and
appends its own, building a structured debate that the PortfolioManager
synthesises into a ``FinalDecision``.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


# ---------------------------------------------------------------------------
# Sub-state types
# ---------------------------------------------------------------------------


class TickerPrediction(TypedDict):
    """PatchTST quantile forecast for a single ticker.

    Attributes:
        ticker: Asset symbol (e.g. ``"SPY"``).
        prob_up: Probability of price increase over the forecast horizon.
        expected_return: Simple expected return (mean of quantile deltas).
        quantiles: Mapping of quantile labels to forecast values,
            e.g. ``{"q0.1": 95.0, "q0.5": 100.0, "q0.9": 105.0}``.
    """

    ticker: str
    prob_up: float
    expected_return: float
    quantiles: dict[str, float]


class AgentReport(TypedDict):
    """Structured output produced by an analyst agent.

    Attributes:
        agent: Identifier for the analyst (``"technical"``,
            ``"fundamental"``, or ``"bear"``).
        ticker: Asset symbol analysed.
        signal: Directional view — ``"bullish"``, ``"bearish"``,
            or ``"neutral"``.
        confidence: Agent's self-assessed confidence in ``[0.0, 1.0]``.
        reasoning: Multi-paragraph analysis text.
        key_factors: Bullet-point list of the most important drivers.
        sources_cited: URLs or reference strings backing the analysis.
    """

    agent: str
    ticker: str
    signal: str
    confidence: float
    reasoning: str
    key_factors: list[str]
    sources_cited: list[str]


class FinalDecision(TypedDict):
    """Portfolio Manager's investment decision for a ticker.

    Attributes:
        ticker: Asset symbol.
        action: One of ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
        confidence: Decision confidence in ``[0.0, 1.0]``.
        suggested_weight: Target portfolio weight in ``[0.0, 0.25]``.
        reasoning: Synthesis of all agent reports.
        dissenting_view: Summary of the Bear Agent's main objections.
    """

    ticker: str
    action: str
    confidence: float
    suggested_weight: float
    reasoning: str
    dissenting_view: str


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class InvestmentState(TypedDict):
    """LangGraph shared state flowing through the investment agent graph.

    Attributes:
        ticker: Asset symbol being analysed in this graph run.
        predictions: PatchTST quantile forecast for the ticker.
        technical_features: Latest indicator values (RSI, Bollinger Bands,
            realized volatility, volume profile metrics).
        news_context: Top-k news snippets from RAG retrieval
            (empty until Session 11).
        reports: Accumulated analyst reports — each agent appends one.
        final_decision: Set by the PortfolioManager node; ``None``
            until that node executes.
        debate_log: Human-readable log entries for the Streamlit dashboard.
    """

    ticker: str
    predictions: TickerPrediction
    technical_features: dict[str, float]
    news_context: list[str]
    reports: Annotated[list[AgentReport], operator.add]
    final_decision: FinalDecision | None
    debate_log: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Factories / helpers
# ---------------------------------------------------------------------------

VALID_SIGNALS = frozenset({"bullish", "bearish", "neutral"})
VALID_ACTIONS = frozenset({"BUY", "SELL", "HOLD"})
MAX_SINGLE_WEIGHT = 0.25


def make_empty_state(ticker: str) -> InvestmentState:
    """Create a blank ``InvestmentState`` ready for graph invocation.

    Args:
        ticker: Asset symbol to analyse.

    Returns:
        A fully initialised state with empty collections and ``None``
        decision.
    """
    return InvestmentState(
        ticker=ticker,
        predictions=TickerPrediction(
            ticker=ticker,
            prob_up=0.0,
            expected_return=0.0,
            quantiles={},
        ),
        technical_features={},
        news_context=[],
        reports=[],
        final_decision=None,
        debate_log=[],
    )


def validate_report(report: AgentReport) -> list[str]:
    """Validate an ``AgentReport`` and return a list of error messages.

    Returns an empty list when the report is valid.

    Args:
        report: The agent report to validate.

    Returns:
        List of human-readable validation error strings.
    """
    errors: list[str] = []

    if report.get("signal") not in VALID_SIGNALS:
        errors.append(
            f"Invalid signal '{report.get('signal')}'; "
            f"expected one of {sorted(VALID_SIGNALS)}"
        )

    confidence = report.get("confidence")
    if confidence is None or not (0.0 <= confidence <= 1.0):
        errors.append(
            f"Confidence must be in [0.0, 1.0], got {confidence}"
        )

    if not report.get("reasoning"):
        errors.append("Reasoning must not be empty")

    return errors


def validate_decision(decision: FinalDecision) -> list[str]:
    """Validate a ``FinalDecision`` and return a list of error messages.

    Returns an empty list when the decision is valid.

    Args:
        decision: The portfolio manager decision to validate.

    Returns:
        List of human-readable validation error strings.
    """
    errors: list[str] = []

    action = decision.get("action")
    if action not in VALID_ACTIONS:
        errors.append(
            f"Invalid action '{action}'; expected one of {sorted(VALID_ACTIONS)}"
        )

    confidence = decision.get("confidence")
    if confidence is None or not (0.0 <= confidence <= 1.0):
        errors.append(
            f"Confidence must be in [0.0, 1.0], got {confidence}"
        )

    weight = decision.get("suggested_weight")
    if weight is None or not (0.0 <= weight <= MAX_SINGLE_WEIGHT):
        errors.append(
            f"Weight must be in [0.0, {MAX_SINGLE_WEIGHT}], got {weight}"
        )

    if confidence is not None and confidence < 0.3 and action != "HOLD":
        errors.append(
            f"Action must be HOLD when confidence < 0.3 (got {action} "
            f"with confidence={confidence})"
        )

    return errors
