"""Tests for src/agents/personas.py — system prompts and Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.personas import (
    BEAR_AGENT,
    FUNDAMENTALIST_ANALYST,
    PERSONA_REGISTRY,
    PORTFOLIO_MANAGER,
    TECHNICAL_ANALYST,
    AgentReportModel,
    FinalDecisionModel,
)


# ---------------------------------------------------------------------------
# TestSystemPrompts
# ---------------------------------------------------------------------------


class TestSystemPrompts:
    """Validate that system prompts contain required content."""

    def test_technical_mentions_rsi(self) -> None:
        assert "RSI" in TECHNICAL_ANALYST

    def test_technical_mentions_bollinger(self) -> None:
        assert "Bollinger" in TECHNICAL_ANALYST

    def test_technical_mentions_volume(self) -> None:
        assert "volume" in TECHNICAL_ANALYST.lower()

    def test_technical_mentions_volatility(self) -> None:
        assert "volatility" in TECHNICAL_ANALYST.lower()

    def test_fundamental_complements(self) -> None:
        assert "complement" in FUNDAMENTALIST_ANALYST.lower() or \
               "not repeat" in FUNDAMENTALIST_ANALYST.lower()

    def test_fundamental_mentions_macro(self) -> None:
        assert "macro" in FUNDAMENTALIST_ANALYST.lower()

    def test_fundamental_mentions_news(self) -> None:
        assert "news" in FUNDAMENTALIST_ANALYST.lower()

    def test_bear_is_sceptical(self) -> None:
        prompt_lower = BEAR_AGENT.lower()
        assert "sceptical" in prompt_lower or "skeptical" in prompt_lower

    def test_bear_mentions_tail_risk(self) -> None:
        assert "tail risk" in BEAR_AGENT.lower() or \
               "tail-risk" in BEAR_AGENT.lower()

    def test_bear_no_recommendations(self) -> None:
        assert "only critique" in BEAR_AGENT.lower()

    def test_bear_never_bullish(self) -> None:
        assert 'never output "bullish"' in BEAR_AGENT.lower() or \
               "never" in BEAR_AGENT.lower()

    def test_pm_mentions_hold_rule(self) -> None:
        assert "0.3" in PORTFOLIO_MANAGER
        assert "HOLD" in PORTFOLIO_MANAGER

    def test_pm_max_weight(self) -> None:
        assert "0.25" in PORTFOLIO_MANAGER or "25%" in PORTFOLIO_MANAGER

    def test_pm_mentions_stop_loss(self) -> None:
        assert "stop-loss" in PORTFOLIO_MANAGER.lower() or \
               "stop loss" in PORTFOLIO_MANAGER.lower()

    def test_all_prompts_non_empty(self) -> None:
        for name, prompt in PERSONA_REGISTRY.items():
            assert len(prompt) > 100, f"Prompt '{name}' too short"

    def test_all_prompts_mention_json(self) -> None:
        for name, prompt in PERSONA_REGISTRY.items():
            assert "json" in prompt.lower(), \
                f"Prompt '{name}' should mention JSON output"


# ---------------------------------------------------------------------------
# TestPersonaRegistry
# ---------------------------------------------------------------------------


class TestPersonaRegistry:
    """PERSONA_REGISTRY completeness."""

    def test_has_four_agents(self) -> None:
        assert len(PERSONA_REGISTRY) == 4

    def test_expected_keys(self) -> None:
        expected = {"technical", "fundamental", "bear", "portfolio_manager"}
        assert set(PERSONA_REGISTRY.keys()) == expected

    def test_values_are_strings(self) -> None:
        for prompt in PERSONA_REGISTRY.values():
            assert isinstance(prompt, str)


# ---------------------------------------------------------------------------
# TestAgentReportModel
# ---------------------------------------------------------------------------


class TestAgentReportModel:
    """Pydantic model for analyst structured output."""

    def test_valid_report(self) -> None:
        report = AgentReportModel(
            agent="technical",
            ticker="SPY",
            signal="bullish",
            confidence=0.75,
            reasoning="Strong momentum with RSI at 65.",
            key_factors=["RSI above midline", "High volume"],
            sources_cited=[],
        )
        assert report.agent == "technical"
        assert report.confidence == 0.75

    def test_confidence_out_of_range_high(self) -> None:
        with pytest.raises(ValidationError):
            AgentReportModel(
                agent="technical",
                ticker="SPY",
                signal="bullish",
                confidence=1.5,
                reasoning="Test.",
                key_factors=[],
            )

    def test_confidence_out_of_range_low(self) -> None:
        with pytest.raises(ValidationError):
            AgentReportModel(
                agent="technical",
                ticker="SPY",
                signal="bullish",
                confidence=-0.1,
                reasoning="Test.",
                key_factors=[],
            )

    def test_empty_reasoning_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentReportModel(
                agent="technical",
                ticker="SPY",
                signal="bullish",
                confidence=0.5,
                reasoning="",
                key_factors=[],
            )

    def test_sources_default_empty(self) -> None:
        report = AgentReportModel(
            agent="bear",
            ticker="NVDA",
            signal="bearish",
            confidence=0.6,
            reasoning="High model risk.",
            key_factors=["Overfitting"],
        )
        assert report.sources_cited == []

    def test_to_dict(self) -> None:
        report = AgentReportModel(
            agent="fundamental",
            ticker="AAPL",
            signal="neutral",
            confidence=0.5,
            reasoning="Mixed signals.",
            key_factors=["Macro uncertainty"],
            sources_cited=["https://example.com"],
        )
        d = report.model_dump()
        assert isinstance(d, dict)
        assert d["agent"] == "fundamental"
        assert d["sources_cited"] == ["https://example.com"]


# ---------------------------------------------------------------------------
# TestFinalDecisionModel
# ---------------------------------------------------------------------------


class TestFinalDecisionModel:
    """Pydantic model for Portfolio Manager structured output."""

    def test_valid_decision(self) -> None:
        decision = FinalDecisionModel(
            ticker="SPY",
            action="BUY",
            confidence=0.7,
            suggested_weight=0.15,
            reasoning="Strong consensus across analysts.",
            dissenting_view="Bear warns of false breakout risk.",
        )
        assert decision.action == "BUY"
        assert decision.suggested_weight == 0.15

    def test_weight_exceeds_max(self) -> None:
        with pytest.raises(ValidationError):
            FinalDecisionModel(
                ticker="SPY",
                action="BUY",
                confidence=0.8,
                suggested_weight=0.30,
                reasoning="Test.",
                dissenting_view="N/A",
            )

    def test_weight_negative(self) -> None:
        with pytest.raises(ValidationError):
            FinalDecisionModel(
                ticker="SPY",
                action="BUY",
                confidence=0.5,
                suggested_weight=-0.1,
                reasoning="Test.",
                dissenting_view="N/A",
            )

    def test_confidence_boundary(self) -> None:
        """Exactly 0.0 and 1.0 should be valid."""
        for conf in [0.0, 1.0]:
            decision = FinalDecisionModel(
                ticker="SPY",
                action="HOLD",
                confidence=conf,
                suggested_weight=0.0,
                reasoning="Boundary test.",
                dissenting_view="N/A",
            )
            assert decision.confidence == conf

    def test_weight_boundary_025(self) -> None:
        """Exactly 0.25 should be valid."""
        decision = FinalDecisionModel(
            ticker="SPY",
            action="BUY",
            confidence=0.9,
            suggested_weight=0.25,
            reasoning="Max weight.",
            dissenting_view="N/A",
        )
        assert decision.suggested_weight == 0.25

    def test_empty_reasoning_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FinalDecisionModel(
                ticker="SPY",
                action="BUY",
                confidence=0.5,
                suggested_weight=0.1,
                reasoning="",
                dissenting_view="N/A",
            )

    def test_to_dict(self) -> None:
        decision = FinalDecisionModel(
            ticker="NVDA",
            action="SELL",
            confidence=0.6,
            suggested_weight=0.0,
            reasoning="Bearish momentum.",
            dissenting_view="Technical still bullish.",
        )
        d = decision.model_dump()
        assert d["action"] == "SELL"
        assert d["ticker"] == "NVDA"
