"""LangGraph investment agent pipeline.

Implements a linear multi-agent graph that analyses a single ticker:

    START → load_context → rag_retrieval → technical → fundamental
          → bear → portfolio_manager → END

Each analyst node calls an LLM via ``.with_structured_output()``
and appends its report to ``InvestmentState.reports``.  The Portfolio Manager
synthesises all reports into a ``FinalDecision``.

The LLM provider is selected via the ``LLM_PROVIDER`` env var
(``"gemini"`` or ``"anthropic"``; defaults to ``"gemini"``).

Usage::

    from src.agents.graph import run_agent_debate

    decisions, full_states = run_agent_debate(
        tickers=["SPY", "NVDA"],
        on_node_complete=lambda t, n, o: print(f"{t}/{n}"),
    )
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import polars as pl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from loguru import logger

load_dotenv()

# Imports below must follow load_dotenv() so that env-dependent modules
# (e.g. LLM client initialisation inside personas/state) see the loaded values.
from src.agents.personas import (  # noqa: E402
    BEAR_AGENT,
    FUNDAMENTALIST_ANALYST,
    PORTFOLIO_MANAGER,
    TECHNICAL_ANALYST,
    AgentReportModel,
    FinalDecisionModel,
)
from src.agents.state import (  # noqa: E402
    MIN_CONFIDENCE_FOR_ACTION,
    AgentReport,
    FinalDecision,
    InvestmentState,
    TickerPrediction,
    make_empty_state,
    validate_decision,
    validate_report,
)

# ---------------------------------------------------------------------------
# LLM provider config
# ---------------------------------------------------------------------------

_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
_DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-3.1-flash-lite-preview",
    "anthropic": "claude-sonnet-4-6",
}
_DEFAULT_MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(_LLM_PROVIDER, "gemini-3.1-flash-lite-preview"))
_ANALYST_TEMPERATURE = 0.2
_PM_TEMPERATURE = 0.1
_PREDICTIONS_DIR = Path("data/outputs")


def _create_llm(temperature: float) -> Any:
    """Create an LLM instance based on ``LLM_PROVIDER`` env var.

    Args:
        temperature: Sampling temperature for the model.

    Returns:
        A LangChain chat model with ``.with_structured_output()`` support.
    """
    if _LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=_DEFAULT_MODEL,
            temperature=temperature,
            google_api_key=os.environ.get("GEMINI_KEY"),
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(  # type: ignore[call-arg]
        model=_DEFAULT_MODEL,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _format_predictions(predictions: TickerPrediction) -> str:
    """Format PatchTST predictions as a readable string for LLM context.

    Args:
        predictions: Ticker prediction data from PatchTST.

    Returns:
        Human-readable summary of predictions.
    """
    quantile_lines = "\n".join(
        f"  {k}: {v:.2f}" for k, v in sorted(predictions.get("quantiles", {}).items())
    )
    return (
        f"Ticker: {predictions['ticker']}\n"
        f"Probability of price increase (5-day): {predictions['prob_up']:.2%}\n"
        f"Expected return: {predictions['expected_return']:.4f}\n"
        f"Quantile forecasts:\n{quantile_lines or '  (none available)'}"
    )


def _format_features(features: dict[str, float]) -> str:
    """Format technical features as a readable string for LLM context.

    Args:
        features: Dictionary of indicator name to value.

    Returns:
        Human-readable feature summary.
    """
    if not features:
        return "(no technical features available)"
    lines = [f"  {k}: {v:.4f}" for k, v in sorted(features.items())]
    return "Technical indicators:\n" + "\n".join(lines)


def _format_reports(reports: list[AgentReport]) -> str:
    """Format previous agent reports for inclusion in LLM prompt.

    Args:
        reports: List of reports from previous agents.

    Returns:
        Formatted string with each report clearly separated.
    """
    if not reports:
        return "(no previous reports)"
    sections: list[str] = []
    for r in reports:
        factors = ", ".join(r.get("key_factors", []))
        sections.append(
            f"--- {r['agent'].upper()} ANALYST REPORT ---\n"
            f"Signal: {r['signal']} (confidence: {r['confidence']:.2f})\n"
            f"Key factors: {factors}\n"
            f"Reasoning: {r['reasoning']}\n"
        )
    return "\n".join(sections)


def _format_news(news_context: list[str]) -> str:
    """Format RAG news context for LLM prompt.

    Args:
        news_context: List of news snippets from RAG retrieval.

    Returns:
        Formatted news context string.
    """
    if not news_context:
        return "(no recent news found for this ticker)"
    return "Recent news (cite these sources in your analysis):\n" + "\n".join(
        f"- {n}" for n in news_context
    )


def _format_debate_entry(agent_name: str, report: Mapping[str, Any]) -> str:
    """Create a structured debate log entry for the Streamlit dashboard.

    Args:
        agent_name: Human-readable name of the agent.
        report: The agent's report as a dict.

    Returns:
        Formatted log entry string.
    """
    signal = report.get("signal", report.get("action", "N/A"))
    confidence = report.get("confidence", 0.0)
    return f"[{agent_name}] {signal} (confidence: {confidence:.2f})"


# ---------------------------------------------------------------------------
# Node: load_context
# ---------------------------------------------------------------------------


def load_context(state: InvestmentState) -> dict[str, Any]:
    """Load PatchTST predictions and technical features for the ticker.

    Reads from ``data/outputs/predictions.parquet``,
    ``data/outputs/forecast.parquet``, and
    ``data/outputs/features.parquet``.  If files are missing, uses
    the values already present in the state (allows pre-population
    for testing).

    Args:
        state: Current graph state.

    Returns:
        State update dict with ``predictions`` and ``technical_features``.
    """
    ticker = state["ticker"]
    predictions = state["predictions"]
    features = dict(state.get("technical_features", {}))

    predictions_path = _PREDICTIONS_DIR / "predictions.parquet"
    forecast_path = _PREDICTIONS_DIR / "forecast.parquet"
    features_path = _PREDICTIONS_DIR / "features.parquet"

    # Load predictions if file exists and state is empty
    if predictions_path.exists() and not predictions.get("quantiles"):
        try:
            pred_df = pl.read_parquet(predictions_path)
            row = pred_df.filter(pl.col("ticker") == ticker)
            if row.height > 0:
                raw_prob = float(row["prob_up"][0])
                raw_er = float(row["expected_return"][0])
                if not math.isfinite(raw_prob):
                    logger.warning(
                        "NaN/Inf prob_up for {} in predictions.parquet — defaulting to 0.5 (neutral)",
                        ticker,
                    )
                    raw_prob = 0.5
                if not math.isfinite(raw_er):
                    logger.warning(
                        "NaN/Inf expected_return for {} in predictions.parquet — defaulting to 0.0",
                        ticker,
                    )
                    raw_er = 0.0
                predictions = TickerPrediction(
                    ticker=ticker,
                    prob_up=raw_prob,
                    expected_return=raw_er,
                    quantiles=predictions.get("quantiles", {}),
                )
            else:
                logger.warning(
                    "{} not found in predictions.parquet — using defaults", ticker
                )
        except Exception:
            logger.warning("Could not load predictions.parquet for {}", ticker)

    # Load quantiles from forecast
    if forecast_path.exists() and not predictions.get("quantiles"):
        try:
            fc_df = pl.read_parquet(forecast_path)
            fc_id_col = "ticker" if "ticker" in fc_df.columns else "unique_id"
            fc_rows = fc_df.filter(pl.col(fc_id_col) == ticker)
            # Sort by time so last row = farthest horizon (h-day)
            if "ds" in fc_rows.columns:
                fc_rows = fc_rows.sort("ds")
            elif "date" in fc_rows.columns:
                fc_rows = fc_rows.sort("date")
            if fc_rows.height > 0:
                quantile_cols = [
                    c for c in fc_rows.columns
                    if c.startswith("PatchTST-") and c not in (fc_id_col, "date", "ds")
                ]
                # Use LAST forecast row (h-day horizon) to match prob_up/
                # expected_return, which are also computed from tail(1) in
                # TitaniumForecaster._compute_prob_up.
                quantiles_raw = {
                    c.replace("PatchTST-", ""): float(fc_rows[c][-1])
                    for c in quantile_cols
                }
                quantiles = {
                    k: v for k, v in quantiles_raw.items() if math.isfinite(v)
                }
                if len(quantiles) < len(quantiles_raw):
                    logger.warning(
                        "{} of {} quantiles were NaN/Inf for {} — dropped",
                        len(quantiles_raw) - len(quantiles),
                        len(quantiles_raw),
                        ticker,
                    )
                predictions = TickerPrediction(
                    ticker=predictions["ticker"],
                    prob_up=predictions["prob_up"],
                    expected_return=predictions["expected_return"],
                    quantiles=quantiles,
                )
            else:
                logger.warning(
                    "{} not found in forecast.parquet — no quantiles loaded",
                    ticker,
                )
        except Exception:
            logger.warning("Could not load forecast.parquet for {}", ticker)

    # Load technical features if file exists and state is empty
    if features_path.exists() and not features:
        try:
            feat_df = pl.read_parquet(features_path)
            feat_id_col = "ticker" if "ticker" in feat_df.columns else "unique_id"
            feat_row = feat_df.filter(pl.col(feat_id_col) == ticker)
            if feat_row.height > 0:
                indicator_cols = [
                    c for c in feat_row.columns if c not in (feat_id_col, "date", "ds")
                ]
                features_raw = {
                    c: float(feat_row[c][0])
                    for c in indicator_cols
                    if feat_row[c][0] is not None
                }
                features = {
                    k: v for k, v in features_raw.items() if math.isfinite(v)
                }
                if len(features) < len(features_raw):
                    logger.warning(
                        "{} of {} features were NaN/Inf for {} — dropped",
                        len(features_raw) - len(features),
                        len(features_raw),
                        ticker,
                    )
            else:
                logger.warning(
                    "{} not found in features.parquet — no indicators loaded",
                    ticker,
                )
        except Exception:
            logger.warning("Could not load features.parquet for {}", ticker)

    log_entry = (
        f"[Context] Loaded {ticker}: prob_up={predictions['prob_up']:.2%}, "
        f"features={len(features)} indicators"
    )
    logger.info(log_entry)

    return {
        "predictions": predictions,
        "technical_features": features,
        "debate_log": [log_entry],
    }


# ---------------------------------------------------------------------------
# Node: rag_retrieval
# ---------------------------------------------------------------------------


def rag_retrieval(state: InvestmentState) -> dict[str, Any]:
    """Retrieve relevant news context from ChromaDB via FinancialRAG.

    Queries the RAG system with a ticker-specific query to populate
    ``news_context`` for downstream agents (primarily the Fundamentalist).
    Gracefully degrades to empty context if ChromaDB is unavailable.

    Args:
        state: Current graph state with ticker set.

    Returns:
        State update dict with ``news_context`` and debate log entry.
    """
    ticker = state["ticker"]
    news: list[str] = []

    try:
        from src.agents.rag import FinancialRAG

        rag = FinancialRAG()
        query = f"{ticker} financial outlook earnings market analysis"
        news = rag.retrieve(ticker, query, top_k=5, max_age_days=30)
    except Exception as exc:
        logger.warning(
            "RAG retrieval failed for {} (graceful degradation): {}", ticker, exc
        )

    log_entry = f"[RAG] Retrieved {len(news)} news snippets for {ticker}"
    logger.info(log_entry)

    return {
        "news_context": news,
        "debate_log": [log_entry],
    }


# ---------------------------------------------------------------------------
# Node: technical_analyst
# ---------------------------------------------------------------------------


def _fallback_report(agent: str, ticker: str, reason: str) -> AgentReport:
    """Build a neutral fallback report when an LLM call fails.

    Args:
        agent: Agent name (``"technical"``, ``"fundamental"``, ``"bear"``).
        ticker: Ticker symbol.
        reason: Short description of the failure.

    Returns:
        Neutral ``AgentReport`` so the pipeline can continue.
    """
    return {
        "agent": agent,
        "ticker": ticker,
        "signal": "neutral",
        "confidence": 0.0,
        "reasoning": f"Agent unavailable: {reason}",
        "key_factors": ["LLM call failed — fallback to neutral"],
        "sources_cited": [],
    }


def technical_analyst(state: InvestmentState) -> dict[str, Any]:
    """Run the Technical Analyst agent.

    Analyses PatchTST forecasts and technical indicators (RSI, Bollinger
    Bands, realized volatility, volume profile) to produce a structured
    ``AgentReport``.

    Args:
        state: Current graph state with predictions and features.

    Returns:
        State update with appended report and debate log entry.
    """
    try:
        llm = _create_llm(temperature=_ANALYST_TEMPERATURE)
        structured_llm = llm.with_structured_output(AgentReportModel)

        user_content = (
            f"Analyse {state['ticker']}.\n\n"
            f"{_format_predictions(state['predictions'])}\n\n"
            f"{_format_features(state['technical_features'])}"
        )

        response: AgentReportModel = structured_llm.invoke([
            SystemMessage(content=TECHNICAL_ANALYST),
            HumanMessage(content=user_content),
        ])

        report: AgentReport = response.model_dump()  # type: ignore[assignment]
        report["agent"] = "technical"
        report["ticker"] = state["ticker"]

        errors = validate_report(report)
        if errors:
            logger.warning("Technical report validation: {}", errors)

    except Exception as exc:
        logger.error("Technical analyst LLM failed for {}: {}", state["ticker"], exc)
        report = _fallback_report("technical", state["ticker"], str(exc))

    log_entry = _format_debate_entry("Technical Analyst", report)
    logger.info(log_entry)

    return {
        "reports": [report],
        "debate_log": [log_entry],
    }


# ---------------------------------------------------------------------------
# Node: fundamentalist_analyst
# ---------------------------------------------------------------------------


def fundamentalist_analyst(state: InvestmentState) -> dict[str, Any]:
    """Run the Fundamental Analyst agent.

    Assesses macro context, news sentiment, and fundamental drivers.
    Reads the Technical Analyst's report to complement (not repeat).

    Args:
        state: Current graph state with predictions and prior reports.

    Returns:
        State update with appended report and debate log entry.
    """
    try:
        llm = _create_llm(temperature=_ANALYST_TEMPERATURE)
        structured_llm = llm.with_structured_output(AgentReportModel)

        user_content = (
            f"Analyse {state['ticker']}.\n\n"
            f"{_format_predictions(state['predictions'])}\n\n"
            f"Previous reports:\n{_format_reports(state['reports'])}\n\n"
            f"{_format_news(state['news_context'])}"
        )

        response: AgentReportModel = structured_llm.invoke([
            SystemMessage(content=FUNDAMENTALIST_ANALYST),
            HumanMessage(content=user_content),
        ])

        report: AgentReport = response.model_dump()  # type: ignore[assignment]
        report["agent"] = "fundamental"
        report["ticker"] = state["ticker"]

        errors = validate_report(report)
        if errors:
            logger.warning("Fundamental report validation: {}", errors)

    except Exception as exc:
        logger.error("Fundamentalist LLM failed for {}: {}", state["ticker"], exc)
        report = _fallback_report("fundamental", state["ticker"], str(exc))

    log_entry = _format_debate_entry("Fundamental Analyst", report)
    logger.info(log_entry)

    return {
        "reports": [report],
        "debate_log": [log_entry],
    }


# ---------------------------------------------------------------------------
# Node: bear_agent
# ---------------------------------------------------------------------------


def bear_agent(state: InvestmentState) -> dict[str, Any]:
    """Run the Bear (Devil's Advocate) agent.

    Critiques all previous reports, identifies risks, and challenges
    assumptions.  Never outputs a bullish signal.

    Args:
        state: Current graph state with predictions and prior reports.

    Returns:
        State update with appended report and debate log entry.
    """
    try:
        llm = _create_llm(temperature=_ANALYST_TEMPERATURE)
        structured_llm = llm.with_structured_output(AgentReportModel)

        user_content = (
            f"Critique the analysis for {state['ticker']}.\n\n"
            f"{_format_predictions(state['predictions'])}\n\n"
            f"Previous reports:\n{_format_reports(state['reports'])}"
        )

        response: AgentReportModel = structured_llm.invoke([
            SystemMessage(content=BEAR_AGENT),
            HumanMessage(content=user_content),
        ])

        report: AgentReport = response.model_dump()  # type: ignore[assignment]
        report["agent"] = "bear"
        report["ticker"] = state["ticker"]

        # Bear agent must never output bullish — enforce structurally
        if report.get("signal") == "bullish":
            logger.warning(
                "Bear agent output 'bullish' for {} — clamping to 'neutral'",
                state["ticker"],
            )
            report["signal"] = "neutral"

        errors = validate_report(report)
        if errors:
            logger.warning("Bear report validation: {}", errors)

    except Exception as exc:
        logger.error("Bear agent LLM failed for {}: {}", state["ticker"], exc)
        report = _fallback_report("bear", state["ticker"], str(exc))
        report["signal"] = "bearish"  # bear fallback is bearish, not neutral

    log_entry = _format_debate_entry("Bear Agent", report)
    logger.info(log_entry)

    return {
        "reports": [report],
        "debate_log": [log_entry],
    }


# ---------------------------------------------------------------------------
# Node: portfolio_manager
# ---------------------------------------------------------------------------


def portfolio_manager(state: InvestmentState) -> dict[str, Any]:
    """Run the Portfolio Manager agent.

    Synthesises all analyst reports and PatchTST predictions into a
    final ``FinalDecision`` with action, confidence, and weight.

    Args:
        state: Current graph state with all reports.

    Returns:
        State update with ``final_decision`` and debate log entry.
    """
    try:
        llm = _create_llm(temperature=_PM_TEMPERATURE)
        structured_llm = llm.with_structured_output(FinalDecisionModel)

        user_content = (
            f"Make your final decision for {state['ticker']}.\n\n"
            f"{_format_predictions(state['predictions'])}\n\n"
            f"All analyst reports:\n{_format_reports(state['reports'])}"
        )

        response: FinalDecisionModel = structured_llm.invoke([
            SystemMessage(content=PORTFOLIO_MANAGER),
            HumanMessage(content=user_content),
        ])

        decision: FinalDecision = response.model_dump()  # type: ignore[assignment]
        decision["ticker"] = state["ticker"]

        # Enforce: confidence < MIN_CONFIDENCE_FOR_ACTION → action=HOLD, weight=0.0
        confidence = decision.get("confidence", 0.0)
        if confidence < MIN_CONFIDENCE_FOR_ACTION:
            if decision.get("action") != "HOLD":
                logger.warning(
                    "PM decision for {} has confidence={:.2f} but action={} — forcing HOLD",
                    state["ticker"],
                    confidence,
                    decision["action"],
                )
                decision["action"] = "HOLD"
            if decision.get("suggested_weight", 0.0) > 0.0:
                logger.warning(
                    "PM decision for {} has confidence={:.2f} but weight={:.2f} — forcing 0.0",
                    state["ticker"],
                    confidence,
                    decision["suggested_weight"],
                )
                decision["suggested_weight"] = 0.0

        # Enforce: SELL → weight=0.0 (3-tier model: SELL=0)
        if decision.get("action") == "SELL" and decision.get("suggested_weight", 0.0) > 0.0:
            logger.warning(
                "PM decision for {} is SELL but weight={:.2f} — forcing 0.0",
                state["ticker"],
                decision["suggested_weight"],
            )
            decision["suggested_weight"] = 0.0

        errors = validate_decision(decision)
        if errors:
            logger.warning("Decision validation errors for {}: {}", state["ticker"], errors)

    except Exception as exc:
        logger.error("Portfolio Manager LLM failed for {}: {}", state["ticker"], exc)
        decision = {
            "ticker": state["ticker"],
            "action": "HOLD",
            "confidence": 0.0,
            "suggested_weight": 0.0,
            "reasoning": f"Portfolio Manager unavailable: {exc}",
            "dissenting_view": "Unable to synthesise — all reports should be reviewed manually.",
        }

    log_entry = _format_debate_entry("Portfolio Manager", decision)
    logger.info(log_entry)

    return {
        "final_decision": decision,
        "debate_log": [log_entry],
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_investment_graph() -> Any:
    """Build and compile the LangGraph investment analysis pipeline.

    Returns:
        Compiled StateGraph ready for ``invoke()``.  The graph follows
        a linear topology::

            START → load_context → rag_retrieval → technical
                  → fundamental → bear → portfolio_manager → END
    """
    graph = StateGraph(InvestmentState)

    graph.add_node("load_context", load_context)
    graph.add_node("rag_retrieval", rag_retrieval)
    graph.add_node("technical", technical_analyst)
    graph.add_node("fundamental", fundamentalist_analyst)
    graph.add_node("bear", bear_agent)
    graph.add_node("portfolio_manager", portfolio_manager)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "rag_retrieval")
    graph.add_edge("rag_retrieval", "technical")
    graph.add_edge("technical", "fundamental")
    graph.add_edge("fundamental", "bear")
    graph.add_edge("bear", "portfolio_manager")
    graph.add_edge("portfolio_manager", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_NodeCallback = Callable[[str, str, dict[str, Any]], None]


def run_agent_debate(
    tickers: list[str] | None = None,
    on_node_complete: _NodeCallback | None = None,
) -> tuple[list[FinalDecision], dict[str, dict]]:
    """Run the full agent debate for each ticker.

    Builds the LangGraph pipeline and invokes it once per ticker.
    Each invocation produces a structured debate and a final investment
    decision.

    Args:
        tickers: List of ticker symbols to analyse.  Defaults to
            ``["SPY", "NVDA", "AAPL", "QQQ"]``.
        on_node_complete: Optional callback invoked after each graph
            node finishes.  Signature:
            ``(ticker, node_name, node_output) -> None``.
            Useful for live dashboard updates.

    Returns:
        Tuple of:
            - List of ``FinalDecision`` dicts, one per ticker.
            - Dict mapping ticker to the full graph state (reports,
              debate_log, predictions, etc.) for dashboard consumption.
    """
    from src.data.ingestion import _resolve_tickers

    tickers = _resolve_tickers(tickers)
    graph = build_investment_graph()
    decisions: list[FinalDecision] = []
    full_states: dict[str, dict] = {}

    for ticker in tickers:
        logger.info("=" * 60)
        logger.info("Starting agent debate for {}", ticker)
        logger.info("=" * 60)

        try:
            initial_state = make_empty_state(ticker)

            if on_node_complete is not None:
                # Stream mode: get per-node updates for live UI.
                # Seed with initial state; fresh lists for reducer fields
                # (reports/debate_log use operator.add — we must accumulate
                # manually because graph.stream() yields raw node deltas).
                result: dict[str, Any] = {
                    **initial_state,
                    "reports": [],
                    "debate_log": [],
                }
                for node_output in graph.stream(initial_state, stream_mode="updates"):
                    for node_name, partial in node_output.items():
                        for key, value in partial.items():
                            if key in ("reports", "debate_log") and isinstance(value, list):
                                result[key].extend(value)
                            else:
                                result[key] = value
                        on_node_complete(ticker, node_name, partial)
            else:
                result = graph.invoke(initial_state)

            full_states[ticker] = dict(result)

            decision = result.get("final_decision")
            if decision:
                decisions.append(decision)
                logger.info(
                    "Decision for {}: {} (confidence={:.2f}, weight={:.2f})",
                    ticker,
                    decision["action"],
                    decision["confidence"],
                    decision["suggested_weight"],
                )
            else:
                logger.error("No decision produced for {}", ticker)

            # Log full debate
            for entry in result.get("debate_log", []):
                logger.debug(entry)

        except Exception as exc:
            logger.error(
                "Agent debate failed for {} (skipping): {}", ticker, exc
            )

    logger.info("Agent debate complete | {} decisions produced", len(decisions))
    return decisions, full_states


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    tickers_arg = sys.argv[1:] if len(sys.argv) > 1 else None
    decisions, _full_states = run_agent_debate(tickers=tickers_arg)
    for d in decisions:
        logger.info(json.dumps(d, indent=2))
