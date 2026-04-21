"""Tests for src/agents/state.py — TypedDicts, factories, and validators."""

from __future__ import annotations

import pytest

from src.agents.state import (
    MAX_SINGLE_WEIGHT,
    VALID_ACTIONS,
    VALID_SIGNALS,
    AgentReport,
    FinalDecision,
    InvestmentState,
    TickerPrediction,
    make_empty_state,
    validate_decision,
    validate_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_report() -> AgentReport:
    """Valid AgentReport for testing."""
    return AgentReport(
        agent="technical",
        ticker="SPY",
        signal="bullish",
        confidence=0.75,
        reasoning="RSI at 65 with strong volume confirmation.",
        key_factors=["RSI above midline", "Volume 1.2x SMA"],
        sources_cited=[],
    )


@pytest.fixture()
def sample_decision() -> FinalDecision:
    """Valid FinalDecision for testing."""
    return FinalDecision(
        ticker="SPY",
        action="BUY",
        confidence=0.7,
        suggested_weight=0.15,
        reasoning="Technical and fundamental agree, bear risk moderate.",
        dissenting_view="Bear warns of potential false breakout.",
    )


# ---------------------------------------------------------------------------
# TestTickerPrediction
# ---------------------------------------------------------------------------


class TestTickerPrediction:
    """TickerPrediction TypedDict construction."""

    def test_create_with_all_fields(self) -> None:
        pred = TickerPrediction(
            ticker="NVDA",
            prob_up=0.8,
            expected_return=0.012,
            quantiles={"q0.1": 95.0, "q0.5": 100.0, "q0.9": 105.0},
        )
        assert pred["ticker"] == "NVDA"
        assert pred["prob_up"] == 0.8
        assert len(pred["quantiles"]) == 3

    def test_quantiles_empty(self) -> None:
        pred = TickerPrediction(
            ticker="SPY", prob_up=0.5, expected_return=0.0, quantiles={}
        )
        assert pred["quantiles"] == {}


# ---------------------------------------------------------------------------
# TestAgentReport
# ---------------------------------------------------------------------------


class TestAgentReport:
    """AgentReport TypedDict construction."""

    def test_valid_report(self, sample_report: AgentReport) -> None:
        assert sample_report["agent"] == "technical"
        assert sample_report["signal"] == "bullish"
        assert sample_report["confidence"] == 0.75

    def test_sources_can_be_empty(self, sample_report: AgentReport) -> None:
        assert sample_report["sources_cited"] == []

    def test_key_factors_list(self, sample_report: AgentReport) -> None:
        assert isinstance(sample_report["key_factors"], list)
        assert len(sample_report["key_factors"]) == 2


# ---------------------------------------------------------------------------
# TestFinalDecision
# ---------------------------------------------------------------------------


class TestFinalDecision:
    """FinalDecision TypedDict construction."""

    def test_valid_decision(self, sample_decision: FinalDecision) -> None:
        assert sample_decision["action"] == "BUY"
        assert sample_decision["suggested_weight"] == 0.15

    def test_hold_decision(self) -> None:
        decision = FinalDecision(
            ticker="AAPL",
            action="HOLD",
            confidence=0.2,
            suggested_weight=0.0,
            reasoning="Low confidence due to conflicting signals.",
            dissenting_view="N/A",
        )
        assert decision["action"] == "HOLD"
        assert decision["confidence"] == 0.2


# ---------------------------------------------------------------------------
# TestInvestmentState
# ---------------------------------------------------------------------------


class TestInvestmentState:
    """InvestmentState TypedDict construction."""

    def test_full_state(self, sample_report: AgentReport) -> None:
        state = InvestmentState(
            ticker="SPY",
            predictions=TickerPrediction(
                ticker="SPY",
                prob_up=0.7,
                expected_return=0.01,
                quantiles={"q0.5": 450.0},
            ),
            technical_features={"rsi": 65.0, "bb_upper": 460.0},
            news_context=[],
            reports=[sample_report],
            final_decision=None,
            debate_log=["[Technical] bullish SPY"],
        )
        assert state["ticker"] == "SPY"
        assert len(state["reports"]) == 1
        assert state["final_decision"] is None

    def test_reports_accumulate(self, sample_report: AgentReport) -> None:
        state = make_empty_state("QQQ")
        state["reports"].append(sample_report)
        assert len(state["reports"]) == 1

        bear_report = AgentReport(
            agent="bear",
            ticker="QQQ",
            signal="bearish",
            confidence=0.6,
            reasoning="Model risk is high.",
            key_factors=["Overfitting risk"],
            sources_cited=[],
        )
        state["reports"].append(bear_report)
        assert len(state["reports"]) == 2


# ---------------------------------------------------------------------------
# TestMakeEmptyState
# ---------------------------------------------------------------------------


class TestMakeEmptyState:
    """Factory function for blank states."""

    def test_creates_valid_state(self) -> None:
        state = make_empty_state("NVDA")
        assert state["ticker"] == "NVDA"
        assert state["predictions"]["ticker"] == "NVDA"
        assert state["reports"] == []
        assert state["debate_log"] == []
        assert state["final_decision"] is None
        assert state["news_context"] == []

    def test_empty_technical_features(self) -> None:
        state = make_empty_state("AAPL")
        assert state["technical_features"] == {}

    def test_predictions_neutral_prior(self) -> None:
        state = make_empty_state("SPY")
        assert state["predictions"]["prob_up"] == 0.5
        assert state["predictions"]["expected_return"] == 0.0
        assert state["predictions"]["quantiles"] == {}


# ---------------------------------------------------------------------------
# TestValidateReport
# ---------------------------------------------------------------------------


class TestValidateReport:
    """Report validation logic."""

    def test_valid_report_no_errors(self, sample_report: AgentReport) -> None:
        assert validate_report(sample_report) == []

    def test_invalid_signal(self, sample_report: AgentReport) -> None:
        sample_report["signal"] = "very_bullish"
        errors = validate_report(sample_report)
        assert len(errors) == 1
        assert "signal" in errors[0].lower()

    def test_confidence_too_high(self, sample_report: AgentReport) -> None:
        sample_report["confidence"] = 1.5
        errors = validate_report(sample_report)
        assert any("confidence" in e.lower() for e in errors)

    def test_confidence_negative(self, sample_report: AgentReport) -> None:
        sample_report["confidence"] = -0.1
        errors = validate_report(sample_report)
        assert any("confidence" in e.lower() for e in errors)

    def test_empty_reasoning(self, sample_report: AgentReport) -> None:
        sample_report["reasoning"] = ""
        errors = validate_report(sample_report)
        assert any("reasoning" in e.lower() for e in errors)

    def test_all_valid_signals(self) -> None:
        for signal in VALID_SIGNALS:
            report = AgentReport(
                agent="technical",
                ticker="SPY",
                signal=signal,
                confidence=0.5,
                reasoning="Test.",
                key_factors=[],
                sources_cited=[],
            )
            assert validate_report(report) == []


# ---------------------------------------------------------------------------
# TestValidateDecision
# ---------------------------------------------------------------------------


class TestValidateDecision:
    """Decision validation logic."""

    def test_valid_decision_no_errors(
        self, sample_decision: FinalDecision
    ) -> None:
        assert validate_decision(sample_decision) == []

    def test_invalid_action(self, sample_decision: FinalDecision) -> None:
        sample_decision["action"] = "SHORT"
        errors = validate_decision(sample_decision)
        assert any("action" in e.lower() for e in errors)

    def test_weight_exceeds_max(self, sample_decision: FinalDecision) -> None:
        sample_decision["suggested_weight"] = 0.30
        errors = validate_decision(sample_decision)
        assert any("weight" in e.lower() for e in errors)

    def test_low_confidence_must_hold(self) -> None:
        decision = FinalDecision(
            ticker="SPY",
            action="BUY",
            confidence=0.2,
            suggested_weight=0.1,
            reasoning="Test.",
            dissenting_view="N/A",
        )
        errors = validate_decision(decision)
        assert any("hold" in e.lower() for e in errors)

    def test_low_confidence_weight_must_be_zero(self) -> None:
        """HOLD with confidence<0.3 but non-zero weight must be flagged."""
        decision = FinalDecision(
            ticker="SPY",
            action="HOLD",
            confidence=0.2,
            suggested_weight=0.15,
            reasoning="Low confidence but weight was not zeroed.",
            dissenting_view="N/A",
        )
        errors = validate_decision(decision)
        assert any("weight" in e.lower() and "0.0" in e for e in errors)

    def test_low_confidence_hold_is_valid(self) -> None:
        decision = FinalDecision(
            ticker="SPY",
            action="HOLD",
            confidence=0.2,
            suggested_weight=0.0,
            reasoning="Low confidence.",
            dissenting_view="N/A",
        )
        errors = validate_decision(decision)
        # Weight 0.0 is within range, action is HOLD — valid
        assert errors == []

    def test_all_valid_actions(self) -> None:
        for action in VALID_ACTIONS:
            # SELL must have weight=0.0 (3-tier model: SELL=0)
            weight = 0.0 if action == "SELL" else 0.1
            decision = FinalDecision(
                ticker="SPY",
                action=action,
                confidence=0.5,
                suggested_weight=weight,
                reasoning="Test.",
                dissenting_view="N/A",
            )
            assert validate_decision(decision) == []

    def test_sell_with_nonzero_weight(self) -> None:
        """SELL action with non-zero weight must be flagged."""
        decision = FinalDecision(
            ticker="SPY",
            action="SELL",
            confidence=0.8,
            suggested_weight=0.15,
            reasoning="Exiting position.",
            dissenting_view="N/A",
        )
        errors = validate_decision(decision)
        assert any("sell" in e.lower() and "weight" in e.lower() for e in errors)

    def test_max_weight_boundary(self) -> None:
        decision = FinalDecision(
            ticker="SPY",
            action="BUY",
            confidence=0.8,
            suggested_weight=MAX_SINGLE_WEIGHT,
            reasoning="Strong conviction.",
            dissenting_view="Moderate risk.",
        )
        assert validate_decision(decision) == []

    def test_confidence_boundary_030(self) -> None:
        """Confidence exactly 0.3 should allow non-HOLD actions."""
        decision = FinalDecision(
            ticker="SPY",
            action="BUY",
            confidence=0.3,
            suggested_weight=0.1,
            reasoning="Borderline.",
            dissenting_view="N/A",
        )
        assert validate_decision(decision) == []
