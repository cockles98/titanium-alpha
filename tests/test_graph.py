"""Tests for src/agents/graph.py — LangGraph investment pipeline.

All LLM calls are mocked via _create_llm patches.  No real API calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

# Ensure src.agents.rag is resolvable by mock.patch() even when heavy
# dependencies (sentence_transformers) are not installed.
if "src.agents.rag" not in sys.modules:
    try:
        import src.agents.rag  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        import src.agents as _agents_pkg

        _rag_stub = MagicMock()
        sys.modules["src.agents.rag"] = _rag_stub
        _agents_pkg.rag = _rag_stub  # type: ignore[attr-defined]

from src.agents.graph import (
    _format_debate_entry,
    _format_features,
    _format_news,
    _format_predictions,
    _format_reports,
    bear_agent,
    build_investment_graph,
    fundamentalist_analyst,
    load_context,
    portfolio_manager,
    rag_retrieval,
    run_agent_debate,
    technical_analyst,
)
from src.agents.personas import AgentReportModel, FinalDecisionModel
from src.agents.state import (
    AgentReport,
    InvestmentState,
    TickerPrediction,
    make_empty_state,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_prediction() -> TickerPrediction:
    return TickerPrediction(
        ticker="SPY",
        prob_up=0.72,
        expected_return=0.015,
        quantiles={"q0.1": 445.0, "q0.5": 450.0, "q0.9": 455.0},
    )


@pytest.fixture()
def sample_features() -> dict[str, float]:
    return {
        "rsi_14": 65.0,
        "bb_upper": 460.0,
        "bb_middle": 450.0,
        "bb_lower": 440.0,
        "realized_vol_21": 0.18,
        "relative_volume": 1.15,
        "obv": 5_000_000.0,
    }


@pytest.fixture()
def sample_state(
    sample_prediction: TickerPrediction,
    sample_features: dict[str, float],
) -> InvestmentState:
    return InvestmentState(
        ticker="SPY",
        predictions=sample_prediction,
        technical_features=sample_features,
        news_context=[],
        reports=[],
        final_decision=None,
        debate_log=[],
    )


@pytest.fixture()
def mock_technical_report() -> AgentReportModel:
    return AgentReportModel(
        agent="technical",
        ticker="SPY",
        signal="bullish",
        confidence=0.75,
        reasoning="RSI at 65 shows strong momentum. BB middle at 450 as support.",
        key_factors=["RSI above midline", "Volume 1.15x SMA"],
        sources_cited=[],
    )


@pytest.fixture()
def mock_fundamental_report() -> AgentReportModel:
    return AgentReportModel(
        agent="fundamental",
        ticker="SPY",
        signal="bullish",
        confidence=0.65,
        reasoning="Macro environment supportive. No immediate headwinds.",
        key_factors=["Stable rates", "Strong earnings season"],
        sources_cited=[],
    )


@pytest.fixture()
def mock_bear_report() -> AgentReportModel:
    return AgentReportModel(
        agent="bear",
        ticker="SPY",
        signal="bearish",
        confidence=0.55,
        reasoning="Model trained on limited data. False breakout risk exists.",
        key_factors=["Overfitting risk", "Extended trend"],
        sources_cited=[],
    )


@pytest.fixture()
def mock_decision() -> FinalDecisionModel:
    return FinalDecisionModel(
        ticker="SPY",
        action="BUY",
        confidence=0.7,
        suggested_weight=0.15,
        reasoning="Technical and fundamental agree. Bear risk moderate.",
        dissenting_view="Bear warns of false breakout and overfitting risk.",
    )


def _make_mock_structured_llm(return_value: AgentReportModel | FinalDecisionModel) -> MagicMock:
    """Create a mock LLM that returns a structured output."""
    mock_llm_instance = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = return_value
    mock_llm_instance.with_structured_output.return_value = mock_structured
    return mock_llm_instance


# ---------------------------------------------------------------------------
# TestFormatHelpers
# ---------------------------------------------------------------------------


class TestFormatHelpers:
    """Prompt formatting functions."""

    def test_format_predictions(self, sample_prediction: TickerPrediction) -> None:
        result = _format_predictions(sample_prediction)
        assert "SPY" in result
        assert "72.00%" in result
        assert "q0.5" in result

    def test_format_predictions_empty_quantiles(self) -> None:
        pred = TickerPrediction(
            ticker="AAPL", prob_up=0.5, expected_return=0.0, quantiles={}
        )
        result = _format_predictions(pred)
        assert "none available" in result

    def test_format_features(self, sample_features: dict[str, float]) -> None:
        result = _format_features(sample_features)
        assert "rsi_14" in result
        assert "65.0000" in result

    def test_format_features_empty(self) -> None:
        result = _format_features({})
        assert "no technical features" in result

    def test_format_reports_empty(self) -> None:
        result = _format_reports([])
        assert "no previous reports" in result

    def test_format_reports_with_data(self) -> None:
        report = AgentReport(
            agent="technical",
            ticker="SPY",
            signal="bullish",
            confidence=0.8,
            reasoning="Strong trend.",
            key_factors=["RSI high"],
            sources_cited=[],
        )
        result = _format_reports([report])
        assert "TECHNICAL" in result
        assert "bullish" in result
        assert "Strong trend" in result

    def test_format_news_empty(self) -> None:
        result = _format_news([])
        assert "no recent news found" in result

    def test_format_news_with_items(self) -> None:
        result = _format_news(["Market rally continues", "Fed holds rates"])
        assert "Market rally" in result
        assert "Fed holds" in result
        assert "cite these sources" in result

    def test_format_debate_entry(self) -> None:
        report = {"signal": "bullish", "confidence": 0.75}
        result = _format_debate_entry("Technical Analyst", report)
        assert "[Technical Analyst]" in result
        assert "bullish" in result
        assert "0.75" in result

    def test_format_debate_entry_with_action(self) -> None:
        decision = {"action": "BUY", "confidence": 0.7}
        result = _format_debate_entry("Portfolio Manager", decision)
        assert "BUY" in result


# ---------------------------------------------------------------------------
# TestLoadContext
# ---------------------------------------------------------------------------


class TestLoadContext:
    """load_context node."""

    def test_preserves_pre_populated_state(
        self, sample_state: InvestmentState
    ) -> None:
        result = load_context(sample_state)
        # Predictions already have quantiles, should be preserved
        assert result["predictions"]["prob_up"] == 0.72

    def test_adds_debate_log_entry(self, sample_state: InvestmentState) -> None:
        result = load_context(sample_state)
        assert len(result["debate_log"]) == 1
        assert "[Context]" in result["debate_log"][0]

    def test_reads_parquet_when_state_empty(self, tmp_path: Path) -> None:
        # Create mock parquet files
        pred_df = pl.DataFrame({
            "ticker": ["SPY"],
            "prob_up": [0.8],
            "expected_return": [0.02],
        })
        pred_df.write_parquet(tmp_path / "predictions.parquet")

        # Multi-row forecast: h=3 horizons. load_context must take the
        # LAST row (farthest horizon) to match prob_up computation.
        fc_df = pl.DataFrame({
            "unique_id": ["SPY", "SPY", "SPY"],
            "ds": [1, 2, 3],
            "PatchTST-q0.1": [445.0, 444.0, 443.0],
            "PatchTST-q0.5": [450.0, 451.0, 452.0],
            "PatchTST-q0.9": [455.0, 457.0, 460.0],
        })
        fc_df.write_parquet(tmp_path / "forecast.parquet")

        state = make_empty_state("SPY")

        with patch("src.agents.graph._PREDICTIONS_DIR", tmp_path):
            result = load_context(state)

        assert result["predictions"]["prob_up"] == 0.8
        assert "q0.5" in result["predictions"]["quantiles"]
        # Must be from the last row (ds=3), not the first (ds=1)
        assert result["predictions"]["quantiles"]["q0.1"] == 443.0
        assert result["predictions"]["quantiles"]["q0.5"] == 452.0
        assert result["predictions"]["quantiles"]["q0.9"] == 460.0

    def test_loads_features_from_parquet(self, tmp_path: Path) -> None:
        feat_df = pl.DataFrame({
            "ticker": ["SPY"],
            "rsi_14": [65.0],
            "bb_upper": [460.0],
            "bb_middle": [450.0],
            "bb_lower": [440.0],
            "realized_vol_21": [0.18],
        })
        feat_df.write_parquet(tmp_path / "features.parquet")

        state = make_empty_state("SPY")

        with patch("src.agents.graph._PREDICTIONS_DIR", tmp_path):
            result = load_context(state)

        assert result["technical_features"]["rsi_14"] == 65.0
        assert result["technical_features"]["bb_upper"] == 460.0
        assert len(result["technical_features"]) == 5

    def test_handles_missing_files(self) -> None:
        state = make_empty_state("SPY")
        with patch(
            "src.agents.graph._PREDICTIONS_DIR", Path("/nonexistent")
        ):
            result = load_context(state)
        # Should not crash, uses neutral prior (0.5)
        assert result["predictions"]["prob_up"] == 0.5

    def test_nan_prob_up_defaults_to_neutral(self, tmp_path: Path) -> None:
        """NaN prob_up in parquet must be replaced with 0.5 (neutral), not propagated."""
        pred_df = pl.DataFrame({
            "ticker": ["SPY"],
            "prob_up": [float("nan")],
            "expected_return": [0.01],
        })
        pred_df.write_parquet(tmp_path / "predictions.parquet")
        state = make_empty_state("SPY")

        with patch("src.agents.graph._PREDICTIONS_DIR", tmp_path):
            result = load_context(state)

        assert result["predictions"]["prob_up"] == 0.5
        assert result["predictions"]["expected_return"] == 0.01

    def test_nan_features_filtered_out(self, tmp_path: Path) -> None:
        """NaN/Inf feature values must be dropped, not passed to LLM."""
        feat_df = pl.DataFrame({
            "ticker": ["SPY"],
            "rsi_14": [65.0],
            "bb_upper": [float("nan")],
            "realized_vol_21": [float("inf")],
            "obv": [5_000_000.0],
        })
        feat_df.write_parquet(tmp_path / "features.parquet")
        state = make_empty_state("SPY")

        with patch("src.agents.graph._PREDICTIONS_DIR", tmp_path):
            result = load_context(state)

        assert "rsi_14" in result["technical_features"]
        assert "obv" in result["technical_features"]
        assert "bb_upper" not in result["technical_features"]
        assert "realized_vol_21" not in result["technical_features"]
        assert len(result["technical_features"]) == 2

    def test_nan_quantiles_filtered_out(self, tmp_path: Path) -> None:
        """NaN quantile values must be dropped from the quantiles dict."""
        pred_df = pl.DataFrame({
            "ticker": ["SPY"],
            "prob_up": [0.7],
            "expected_return": [0.01],
        })
        pred_df.write_parquet(tmp_path / "predictions.parquet")

        fc_df = pl.DataFrame({
            "unique_id": ["SPY"],
            "ds": [1],
            "PatchTST-q0.1": [445.0],
            "PatchTST-q0.5": [float("nan")],
            "PatchTST-q0.9": [460.0],
        })
        fc_df.write_parquet(tmp_path / "forecast.parquet")
        state = make_empty_state("SPY")

        with patch("src.agents.graph._PREDICTIONS_DIR", tmp_path):
            result = load_context(state)

        quantiles = result["predictions"]["quantiles"]
        assert "q0.1" in quantiles
        assert "q0.9" in quantiles
        assert "q0.5" not in quantiles


# ---------------------------------------------------------------------------
# TestRAGRetrieval
# ---------------------------------------------------------------------------


class TestRAGRetrieval:
    """rag_retrieval node."""

    def test_populates_news_context(self, sample_state: InvestmentState) -> None:
        """Should populate news_context with RAG results."""
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = [
            "[2026-03-01 | Reuters] NVDA beats earnings",
            "[2026-02-28 | CNBC] AI chip demand surges",
        ]
        with patch("src.agents.rag.FinancialRAG", return_value=mock_rag):
            result = rag_retrieval(sample_state)

        assert len(result["news_context"]) == 2
        assert "NVDA beats earnings" in result["news_context"][0]

    def test_adds_debate_log_entry(self, sample_state: InvestmentState) -> None:
        """Should add a debate log entry with count."""
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = ["news1", "news2", "news3"]
        with patch("src.agents.rag.FinancialRAG", return_value=mock_rag):
            result = rag_retrieval(sample_state)

        assert len(result["debate_log"]) == 1
        assert "[RAG]" in result["debate_log"][0]
        assert "3" in result["debate_log"][0]

    def test_graceful_degradation_on_error(
        self, sample_state: InvestmentState
    ) -> None:
        """Should return empty context when RAG fails."""
        with patch(
            "src.agents.rag.FinancialRAG",
            side_effect=RuntimeError("ChromaDB down"),
        ):
            result = rag_retrieval(sample_state)

        assert result["news_context"] == []
        assert "0" in result["debate_log"][0]

    def test_calls_retrieve_with_ticker(
        self, sample_state: InvestmentState
    ) -> None:
        """Should query RAG with the ticker from state."""
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = []
        with patch("src.agents.rag.FinancialRAG", return_value=mock_rag):
            rag_retrieval(sample_state)

        mock_rag.retrieve.assert_called_once()
        call_args = mock_rag.retrieve.call_args
        assert call_args[0][0] == "SPY"  # ticker
        assert call_args[1]["top_k"] == 5
        assert call_args[1]["max_age_days"] == 30

    def test_empty_retrieve_returns_empty_list(
        self, sample_state: InvestmentState
    ) -> None:
        """Should handle empty retrieve results."""
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = []
        with patch("src.agents.rag.FinancialRAG", return_value=mock_rag):
            result = rag_retrieval(sample_state)

        assert result["news_context"] == []


# ---------------------------------------------------------------------------
# TestTechnicalAnalyst
# ---------------------------------------------------------------------------


class TestTechnicalAnalyst:
    """technical_analyst node."""

    @patch("src.agents.graph._create_llm")
    def test_returns_single_report(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
        mock_technical_report: AgentReportModel,
    ) -> None:
        mock_chat_cls.return_value = _make_mock_structured_llm(
            mock_technical_report
        )

        result = technical_analyst(sample_state)

        # With operator.add reducer, node returns [report] (not accumulated)
        assert len(result["reports"]) == 1
        assert result["reports"][0]["agent"] == "technical"

    @patch("src.agents.graph._create_llm")
    def test_returns_debate_log_entry(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
        mock_technical_report: AgentReportModel,
    ) -> None:
        mock_chat_cls.return_value = _make_mock_structured_llm(
            mock_technical_report
        )

        result = technical_analyst(sample_state)

        assert len(result["debate_log"]) == 1
        assert "Technical Analyst" in result["debate_log"][0]

    @patch("src.agents.graph._create_llm")
    def test_forces_agent_field(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
    ) -> None:
        """Even if LLM returns wrong agent name, we override it."""
        report = AgentReportModel(
            agent="wrong_name",
            ticker="SPY",
            signal="bullish",
            confidence=0.5,
            reasoning="Test.",
            key_factors=[],
        )
        mock_chat_cls.return_value = _make_mock_structured_llm(report)

        result = technical_analyst(sample_state)
        assert result["reports"][0]["agent"] == "technical"


# ---------------------------------------------------------------------------
# TestFundamentalAnalyst
# ---------------------------------------------------------------------------


class TestFundamentalAnalyst:
    """fundamentalist_analyst node."""

    @patch("src.agents.graph._create_llm")
    def test_returns_single_report(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
        mock_fundamental_report: AgentReportModel,
    ) -> None:
        # Add a technical report first (simulating graph accumulation)
        tech_report = AgentReport(
            agent="technical", ticker="SPY", signal="bullish",
            confidence=0.75, reasoning="Strong.", key_factors=[], sources_cited=[],
        )
        sample_state["reports"] = [tech_report]

        mock_chat_cls.return_value = _make_mock_structured_llm(
            mock_fundamental_report
        )

        result = fundamentalist_analyst(sample_state)

        # Node returns only its own report; reducer accumulates
        assert len(result["reports"]) == 1
        assert result["reports"][0]["agent"] == "fundamental"

    @patch("src.agents.graph._create_llm")
    def test_includes_previous_reports_in_prompt(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
        mock_fundamental_report: AgentReportModel,
    ) -> None:
        tech_report = AgentReport(
            agent="technical", ticker="SPY", signal="bullish",
            confidence=0.75, reasoning="RSI strong.", key_factors=["RSI"],
            sources_cited=[],
        )
        sample_state["reports"] = [tech_report]

        mock_llm = _make_mock_structured_llm(mock_fundamental_report)
        mock_chat_cls.return_value = mock_llm

        fundamentalist_analyst(sample_state)

        # Verify the prompt includes previous report data
        call_args = mock_llm.with_structured_output.return_value.invoke.call_args
        messages = call_args[0][0]
        user_msg = messages[1].content
        assert "TECHNICAL" in user_msg
        assert "RSI strong" in user_msg

    @patch("src.agents.graph._create_llm")
    def test_includes_news_context_in_prompt(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
        mock_fundamental_report: AgentReportModel,
    ) -> None:
        """RAG news context should appear in the fundamentalist's prompt."""
        sample_state["news_context"] = [
            "[2026-03-01 | Reuters] NVDA beats earnings expectations",
            "[2026-02-28 | CNBC] AI chip demand surges globally",
        ]

        mock_llm = _make_mock_structured_llm(mock_fundamental_report)
        mock_chat_cls.return_value = mock_llm

        fundamentalist_analyst(sample_state)

        call_args = mock_llm.with_structured_output.return_value.invoke.call_args
        messages = call_args[0][0]
        user_msg = messages[1].content
        assert "NVDA beats earnings" in user_msg
        assert "AI chip demand" in user_msg
        assert "cite these sources" in user_msg


# ---------------------------------------------------------------------------
# TestBearAgent
# ---------------------------------------------------------------------------


class TestBearAgent:
    """bear_agent node."""

    @patch("src.agents.graph._create_llm")
    def test_returns_single_report(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
        mock_bear_report: AgentReportModel,
    ) -> None:
        sample_state["reports"] = [
            AgentReport(
                agent="technical", ticker="SPY", signal="bullish",
                confidence=0.75, reasoning="Strong.", key_factors=[],
                sources_cited=[],
            ),
            AgentReport(
                agent="fundamental", ticker="SPY", signal="bullish",
                confidence=0.65, reasoning="Supportive.", key_factors=[],
                sources_cited=[],
            ),
        ]
        mock_chat_cls.return_value = _make_mock_structured_llm(
            mock_bear_report
        )

        result = bear_agent(sample_state)

        # Node returns only its own report; reducer accumulates
        assert len(result["reports"]) == 1
        assert result["reports"][0]["agent"] == "bear"

    @patch("src.agents.graph._create_llm")
    def test_forces_agent_field(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
    ) -> None:
        report = AgentReportModel(
            agent="analyst",
            ticker="SPY",
            signal="bearish",
            confidence=0.6,
            reasoning="Risky.",
            key_factors=["Risk"],
        )
        mock_chat_cls.return_value = _make_mock_structured_llm(report)

        result = bear_agent(sample_state)
        assert result["reports"][0]["agent"] == "bear"

    @patch("src.agents.graph._create_llm")
    def test_clamps_bullish_to_neutral(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
    ) -> None:
        """Bear agent must never output bullish — should be clamped to neutral."""
        bullish_report = AgentReportModel(
            agent="bear",
            ticker="SPY",
            signal="bullish",
            confidence=0.6,
            reasoning="Actually everything looks great.",
            key_factors=["No risks found"],
        )
        mock_chat_cls.return_value = _make_mock_structured_llm(bullish_report)

        result = bear_agent(sample_state)
        assert result["reports"][0]["signal"] == "neutral"


# ---------------------------------------------------------------------------
# TestPortfolioManager
# ---------------------------------------------------------------------------


class TestPortfolioManager:
    """portfolio_manager node."""

    @patch("src.agents.graph._create_llm")
    def test_produces_final_decision(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
        mock_decision: FinalDecisionModel,
    ) -> None:
        sample_state["reports"] = [
            AgentReport(
                agent="technical", ticker="SPY", signal="bullish",
                confidence=0.75, reasoning="Strong.", key_factors=[],
                sources_cited=[],
            ),
        ]
        mock_chat_cls.return_value = _make_mock_structured_llm(mock_decision)

        result = portfolio_manager(sample_state)

        assert result["final_decision"] is not None
        assert result["final_decision"]["action"] == "BUY"
        assert result["final_decision"]["ticker"] == "SPY"

    @patch("src.agents.graph._create_llm")
    def test_adds_debate_log(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
        mock_decision: FinalDecisionModel,
    ) -> None:
        mock_chat_cls.return_value = _make_mock_structured_llm(mock_decision)

        result = portfolio_manager(sample_state)

        assert any("Portfolio Manager" in e for e in result["debate_log"])

    @patch("src.agents.graph._create_llm")
    def test_enforces_hold_on_low_confidence(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
    ) -> None:
        """LLM returning BUY with confidence < 0.3 must be forced to HOLD/0.0."""
        bad_decision = FinalDecisionModel(
            ticker="SPY",
            action="BUY",
            confidence=0.15,
            suggested_weight=0.10,
            reasoning="Uncertain but buying anyway.",
            dissenting_view="Bear sees major risk.",
        )
        mock_chat_cls.return_value = _make_mock_structured_llm(bad_decision)

        result = portfolio_manager(sample_state)

        assert result["final_decision"]["action"] == "HOLD"
        assert result["final_decision"]["suggested_weight"] == 0.0
        assert result["final_decision"]["confidence"] == 0.15

    @patch("src.agents.graph._create_llm")
    def test_preserves_valid_high_confidence_decision(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
        mock_decision: FinalDecisionModel,
    ) -> None:
        """Decisions with confidence >= 0.3 must NOT be overridden."""
        mock_chat_cls.return_value = _make_mock_structured_llm(mock_decision)

        result = portfolio_manager(sample_state)

        assert result["final_decision"]["action"] == "BUY"
        assert result["final_decision"]["suggested_weight"] == 0.15

    @patch("src.agents.graph._create_llm")
    def test_boundary_confidence_030_allows_action(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
    ) -> None:
        """Exact boundary confidence=0.3 must NOT trigger HOLD enforcement."""
        boundary_decision = FinalDecisionModel(
            ticker="SPY",
            action="BUY",
            confidence=0.3,
            suggested_weight=0.10,
            reasoning="Borderline but acting.",
            dissenting_view="Close call.",
        )
        mock_chat_cls.return_value = _make_mock_structured_llm(boundary_decision)

        result = portfolio_manager(sample_state)

        assert result["final_decision"]["action"] == "BUY"
        assert result["final_decision"]["suggested_weight"] == 0.10
        assert result["final_decision"]["confidence"] == 0.3

    @patch("src.agents.graph._create_llm")
    def test_enforces_zero_weight_on_sell(
        self,
        mock_chat_cls: MagicMock,
        sample_state: InvestmentState,
    ) -> None:
        """LLM returning SELL with non-zero weight must be forced to 0.0."""
        sell_decision = FinalDecisionModel(
            ticker="SPY",
            action="SELL",
            confidence=0.8,
            suggested_weight=0.15,
            reasoning="Exiting position.",
            dissenting_view="Could rebound.",
        )
        mock_chat_cls.return_value = _make_mock_structured_llm(sell_decision)

        result = portfolio_manager(sample_state)

        assert result["final_decision"]["action"] == "SELL"
        assert result["final_decision"]["suggested_weight"] == 0.0
        assert result["final_decision"]["confidence"] == 0.8


# ---------------------------------------------------------------------------
# TestBuildInvestmentGraph
# ---------------------------------------------------------------------------


class TestBuildInvestmentGraph:
    """Graph construction and compilation."""

    def test_graph_compiles(self) -> None:
        graph = build_investment_graph()
        assert graph is not None

    @patch("src.agents.rag.FinancialRAG")
    @patch("src.agents.graph._create_llm")
    def test_full_graph_execution(
        self,
        mock_chat_cls: MagicMock,
        mock_rag_cls: MagicMock,
        sample_state: InvestmentState,
        mock_technical_report: AgentReportModel,
        mock_fundamental_report: AgentReportModel,
        mock_bear_report: AgentReportModel,
        mock_decision: FinalDecisionModel,
    ) -> None:
        """Full graph run with mocked LLM returning different responses."""
        mock_rag_cls.return_value.retrieve.return_value = []

        responses = iter([
            mock_technical_report,
            mock_fundamental_report,
            mock_bear_report,
            mock_decision,
        ])

        def make_llm(*args: object, **kwargs: object) -> MagicMock:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = next(responses)
            mock_llm.with_structured_output.return_value = mock_structured
            return mock_llm

        mock_chat_cls.side_effect = make_llm

        graph = build_investment_graph()
        result = graph.invoke(sample_state)

        # Should have 3 analyst reports
        assert len(result["reports"]) == 3
        agents = [r["agent"] for r in result["reports"]]
        assert agents == ["technical", "fundamental", "bear"]

        # Should have a final decision
        assert result["final_decision"] is not None
        assert result["final_decision"]["action"] == "BUY"

        # Should have debate log entries (context + rag + 4 agents)
        assert len(result["debate_log"]) == 6

    @patch("src.agents.rag.FinancialRAG")
    @patch("src.agents.graph._create_llm")
    def test_graph_state_accumulates(
        self,
        mock_chat_cls: MagicMock,
        mock_rag_cls: MagicMock,
        sample_state: InvestmentState,
        mock_technical_report: AgentReportModel,
        mock_fundamental_report: AgentReportModel,
        mock_bear_report: AgentReportModel,
        mock_decision: FinalDecisionModel,
    ) -> None:
        """Verify state flows correctly through all nodes."""
        mock_rag_cls.return_value.retrieve.return_value = []

        responses = iter([
            mock_technical_report,
            mock_fundamental_report,
            mock_bear_report,
            mock_decision,
        ])

        def make_llm(*args: object, **kwargs: object) -> MagicMock:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = next(responses)
            mock_llm.with_structured_output.return_value = mock_structured
            return mock_llm

        mock_chat_cls.side_effect = make_llm

        graph = build_investment_graph()
        result = graph.invoke(sample_state)

        # Verify each report has correct ticker
        for report in result["reports"]:
            assert report["ticker"] == "SPY"

        # Decision should have ticker
        assert result["final_decision"]["ticker"] == "SPY"

    @patch("src.agents.rag.FinancialRAG")
    @patch("src.agents.graph._create_llm")
    def test_rag_news_flows_to_fundamentalist(
        self,
        mock_chat_cls: MagicMock,
        mock_rag_cls: MagicMock,
        sample_state: InvestmentState,
        mock_technical_report: AgentReportModel,
        mock_bear_report: AgentReportModel,
        mock_decision: FinalDecisionModel,
    ) -> None:
        """RAG news should appear in the fundamentalist's LLM prompt."""
        mock_rag_cls.return_value.retrieve.return_value = [
            "[2026-03-01 | Reuters] SPY hits all-time high",
        ]

        fundamental_with_sources = AgentReportModel(
            agent="fundamental",
            ticker="SPY",
            signal="bullish",
            confidence=0.7,
            reasoning="Based on Reuters (2026-03-01): SPY hits all-time high.",
            key_factors=["All-time high"],
            sources_cited=["Reuters (2026-03-01)"],
        )

        responses = iter([
            mock_technical_report,
            fundamental_with_sources,
            mock_bear_report,
            mock_decision,
        ])

        def make_llm(*args: object, **kwargs: object) -> MagicMock:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = next(responses)
            mock_llm.with_structured_output.return_value = mock_structured
            return mock_llm

        mock_chat_cls.side_effect = make_llm

        graph = build_investment_graph()
        result = graph.invoke(sample_state)

        # RAG news should be in the state
        assert len(result["news_context"]) == 1
        assert "SPY hits all-time high" in result["news_context"][0]

        # Fundamental report should cite sources
        fundamental = [r for r in result["reports"] if r["agent"] == "fundamental"][0]
        assert len(fundamental["sources_cited"]) > 0


# ---------------------------------------------------------------------------
# TestRunAgentDebate
# ---------------------------------------------------------------------------


class TestRunAgentDebate:
    """run_agent_debate() orchestrator."""

    @patch("src.agents.rag.FinancialRAG")
    @patch("src.agents.graph._create_llm")
    def test_produces_decisions_for_all_tickers(
        self,
        mock_chat_cls: MagicMock,
        mock_rag_cls: MagicMock,
        mock_technical_report: AgentReportModel,
        mock_fundamental_report: AgentReportModel,
        mock_bear_report: AgentReportModel,
        mock_decision: FinalDecisionModel,
    ) -> None:
        mock_rag_cls.return_value.retrieve.return_value = []

        # Shared iterator: each ChatAnthropic() call gets the next response
        responses = iter([
            mock_technical_report,
            mock_fundamental_report,
            mock_bear_report,
            mock_decision,
        ])

        def make_llm(*args: object, **kwargs: object) -> MagicMock:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = next(responses)
            mock_llm.with_structured_output.return_value = mock_structured
            return mock_llm

        mock_chat_cls.side_effect = make_llm

        decisions, full_states = run_agent_debate(tickers=["SPY"])

        assert len(decisions) == 1
        assert decisions[0]["action"] == "BUY"
        assert "SPY" in full_states

    @patch("src.agents.rag.FinancialRAG")
    @patch("src.agents.graph._create_llm")
    def test_multiple_tickers(
        self,
        mock_chat_cls: MagicMock,
        mock_rag_cls: MagicMock,
        mock_technical_report: AgentReportModel,
        mock_fundamental_report: AgentReportModel,
        mock_bear_report: AgentReportModel,
    ) -> None:
        mock_rag_cls.return_value.retrieve.return_value = []

        spy_decision = FinalDecisionModel(
            ticker="SPY", action="BUY", confidence=0.7,
            suggested_weight=0.15, reasoning="Strong.",
            dissenting_view="Moderate risk.",
        )
        nvda_decision = FinalDecisionModel(
            ticker="NVDA", action="HOLD", confidence=0.4,
            suggested_weight=0.05, reasoning="Mixed signals.",
            dissenting_view="High volatility.",
        )

        # 8 responses: 4 per ticker
        responses = iter([
            mock_technical_report, mock_fundamental_report,
            mock_bear_report, spy_decision,
            mock_technical_report, mock_fundamental_report,
            mock_bear_report, nvda_decision,
        ])

        def make_llm(*args: object, **kwargs: object) -> MagicMock:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = next(responses)
            mock_llm.with_structured_output.return_value = mock_structured
            return mock_llm

        mock_chat_cls.side_effect = make_llm

        decisions, full_states = run_agent_debate(tickers=["SPY", "NVDA"])

        assert len(decisions) == 2
        assert "SPY" in full_states
        assert "NVDA" in full_states

    @patch("src.agents.rag.FinancialRAG")
    @patch("src.agents.graph._create_llm")
    def test_hold_decision_for_low_confidence(
        self,
        mock_chat_cls: MagicMock,
        mock_rag_cls: MagicMock,
        mock_technical_report: AgentReportModel,
        mock_fundamental_report: AgentReportModel,
        mock_bear_report: AgentReportModel,
    ) -> None:
        mock_rag_cls.return_value.retrieve.return_value = []

        hold_decision = FinalDecisionModel(
            ticker="QQQ", action="HOLD", confidence=0.2,
            suggested_weight=0.0, reasoning="Too uncertain.",
            dissenting_view="Bear sees significant risk.",
        )

        responses = iter([
            mock_technical_report,
            mock_fundamental_report,
            mock_bear_report,
            hold_decision,
        ])

        def make_llm(*args: object, **kwargs: object) -> MagicMock:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = next(responses)
            mock_llm.with_structured_output.return_value = mock_structured
            return mock_llm

        mock_chat_cls.side_effect = make_llm

        decisions, full_states = run_agent_debate(tickers=["QQQ"])

        assert len(decisions) == 1
        assert decisions[0]["action"] == "HOLD"
        assert decisions[0]["confidence"] == 0.2
        assert "QQQ" in full_states

    @patch("src.agents.rag.FinancialRAG")
    @patch("src.agents.graph._create_llm")
    def test_streaming_callback_receives_correct_node_names(
        self,
        mock_chat_cls: MagicMock,
        mock_rag_cls: MagicMock,
        mock_technical_report: AgentReportModel,
        mock_fundamental_report: AgentReportModel,
        mock_bear_report: AgentReportModel,
        mock_decision: FinalDecisionModel,
    ) -> None:
        """on_node_complete must receive actual graph node names, not state keys."""
        mock_rag_cls.return_value.retrieve.return_value = []

        responses = iter([
            mock_technical_report,
            mock_fundamental_report,
            mock_bear_report,
            mock_decision,
        ])

        def make_llm(*args: object, **kwargs: object) -> MagicMock:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = next(responses)
            mock_llm.with_structured_output.return_value = mock_structured
            return mock_llm

        mock_chat_cls.side_effect = make_llm

        callback_calls: list[tuple[str, str]] = []

        def on_node(ticker: str, node: str, output: dict) -> None:
            callback_calls.append((ticker, node))

        decisions, full_states = run_agent_debate(
            tickers=["SPY"], on_node_complete=on_node
        )

        # Must produce a valid decision
        assert len(decisions) == 1
        assert decisions[0]["action"] == "BUY"

        # Callback must have been called with real node names
        node_names = [name for _, name in callback_calls]
        expected_nodes = [
            "load_context", "rag_retrieval", "technical",
            "fundamental", "bear", "portfolio_manager",
        ]
        assert node_names == expected_nodes

        # Full state must be correctly accumulated
        assert len(full_states["SPY"]["reports"]) == 3
        assert len(full_states["SPY"]["debate_log"]) == 6
