"""LangGraph investment agent pipeline.

Implements a linear multi-agent graph that analyses a single ticker:

    START → load_context → rag_retrieval → technical → fundamental
          → bear → portfolio_manager → END

Each analyst node calls Claude via ``ChatAnthropic.with_structured_output()``
and appends its report to ``InvestmentState.reports``.  The Portfolio Manager
synthesises all reports into a ``FinalDecision``.

Usage::

    from src.agents.graph import run_agent_debate

    decisions = run_agent_debate(
        tickers=["SPY", "NVDA"],
        predictions_path="data/outputs/predictions.parquet",
        forecast_path="data/outputs/forecast.parquet",
        features_path="data/outputs/features.parquet",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import polars as pl
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.agents.personas import (
    BEAR_AGENT,
    FUNDAMENTALIST_ANALYST,
    PORTFOLIO_MANAGER,
    TECHNICAL_ANALYST,
    AgentReportModel,
    FinalDecisionModel,
)
from src.agents.state import (
    AgentReport,
    FinalDecision,
    InvestmentState,
    TickerPrediction,
    make_empty_state,
    validate_decision,
    validate_report,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "claude-sonnet-4-6"
_ANALYST_TEMPERATURE = 0.2
_PM_TEMPERATURE = 0.1
_PREDICTIONS_DIR = Path("data/outputs")


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


def _format_debate_entry(agent_name: str, report: dict[str, Any]) -> str:
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

    Reads from ``data/outputs/predictions.parquet`` and
    ``data/outputs/forecast.parquet``.  If files are missing, uses
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

    # Load predictions if file exists and state is empty
    if predictions_path.exists() and not predictions.get("quantiles"):
        try:
            pred_df = pl.read_parquet(predictions_path)
            row = pred_df.filter(pl.col("ticker") == ticker)
            if row.height > 0:
                predictions = TickerPrediction(
                    ticker=ticker,
                    prob_up=float(row["prob_up"][0]),
                    expected_return=float(row["expected_return"][0]),
                    quantiles=predictions.get("quantiles", {}),
                )
        except Exception:
            logger.warning("Could not load predictions.parquet for {}", ticker)

    # Load quantiles from forecast
    if forecast_path.exists() and not predictions.get("quantiles"):
        try:
            fc_df = pl.read_parquet(forecast_path)
            fc_rows = fc_df.filter(pl.col("unique_id") == ticker)
            if fc_rows.height > 0:
                quantile_cols = [
                    c for c in fc_rows.columns if c.startswith("PatchTST-q")
                ]
                # Use first forecast row for quantile snapshot
                quantiles = {
                    c.replace("PatchTST-", ""): float(fc_rows[c][0])
                    for c in quantile_cols
                }
                predictions = TickerPrediction(
                    ticker=predictions["ticker"],
                    prob_up=predictions["prob_up"],
                    expected_return=predictions["expected_return"],
                    quantiles=quantiles,
                )
        except Exception:
            logger.warning("Could not load forecast.parquet for {}", ticker)

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
    llm = ChatAnthropic(model=_DEFAULT_MODEL, temperature=_ANALYST_TEMPERATURE)
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
    llm = ChatAnthropic(model=_DEFAULT_MODEL, temperature=_ANALYST_TEMPERATURE)
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
    llm = ChatAnthropic(model=_DEFAULT_MODEL, temperature=_ANALYST_TEMPERATURE)
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

    errors = validate_report(report)
    if errors:
        logger.warning("Bear report validation: {}", errors)

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
    llm = ChatAnthropic(model=_DEFAULT_MODEL, temperature=_PM_TEMPERATURE)
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

    errors = validate_decision(decision)
    if errors:
        logger.warning("Decision validation errors for {}: {}", state["ticker"], errors)

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
    from src.data.ingestion import DEFAULT_TICKERS

    tickers = tickers or DEFAULT_TICKERS
    graph = build_investment_graph()
    decisions: list[FinalDecision] = []
    full_states: dict[str, dict] = {}

    for ticker in tickers:
        logger.info("=" * 60)
        logger.info("Starting agent debate for {}", ticker)
        logger.info("=" * 60)

        initial_state = make_empty_state(ticker)

        if on_node_complete is not None:
            # Stream mode: get per-node updates for live UI
            result: dict[str, Any] = {}
            for node_output in graph.stream(initial_state):
                # node_output is {node_name: partial_state_update}
                for node_name, partial in node_output.items():
                    result.update(partial)
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

    logger.info("Agent debate complete | {} decisions produced", len(decisions))
    return decisions, full_states


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    tickers_arg = sys.argv[1:] if len(sys.argv) > 1 else None
    results = run_agent_debate(tickers=tickers_arg)
    for d in results:
        logger.info(json.dumps(d, indent=2))
