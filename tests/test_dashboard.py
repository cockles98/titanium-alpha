"""Tests for src/dashboard/app.py — data loaders, chart builders, and streaming."""

from __future__ import annotations

import json
import queue
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.dashboard.app import (
    AGENT_STYLES,
    _NODE_TO_AGENT,
    _chart_quantile_fan,
    _chart_weight_comparison,
    _chart_weight_donut,
    _replay_debate,
    _run_live_debate_thread,
    load_debate_history,
    load_decisions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_decisions() -> dict[str, Any]:
    return {
        "timestamp": "2026-03-06T12:00:00+00:00",
        "tickers": ["SPY", "NVDA", "AAPL", "QQQ"],
        "decisions": [
            {
                "ticker": "SPY",
                "action": "BUY",
                "weight": 0.35,
                "confidence": 0.8,
                "reasoning": "Strong momentum with tech rebound",
                "dissenting_view": "Overvalued relative to earnings",
            },
            {
                "ticker": "NVDA",
                "action": "BUY",
                "weight": 0.30,
                "confidence": 0.7,
                "reasoning": "AI growth driver",
                "dissenting_view": "High PE ratio",
            },
            {
                "ticker": "AAPL",
                "action": "HOLD",
                "weight": 0.0,
                "confidence": 0.4,
                "reasoning": "Uncertain outlook",
                "dissenting_view": "iPhone saturation",
            },
            {
                "ticker": "QQQ",
                "action": "BUY",
                "weight": 0.35,
                "confidence": 0.6,
                "reasoning": "Tech sector rebound",
                "dissenting_view": "Rate risk remains",
            },
        ],
        "hrp_raw_weights": {"SPY": 0.28, "NVDA": 0.27, "AAPL": 0.22, "QQQ": 0.23},
        "hrp_final_weights": {"SPY": 0.35, "NVDA": 0.30, "AAPL": 0.0, "QQQ": 0.35},
        "cluster_order": ["SPY", "QQQ", "NVDA", "AAPL"],
        "metadata": {
            "schema_version": "1.0",
            "n_observations": 504,
            "hrp_config": {
                "linkage_method": "single",
                "correlation_method": "pearson",
            },
        },
    }


@pytest.fixture()
def sample_debate_history() -> dict[str, Any]:
    return {
        "SPY": {
            "reports": [
                {
                    "agent": "technical",
                    "ticker": "SPY",
                    "signal": "bullish",
                    "confidence": 0.8,
                    "reasoning": "RSI shows momentum",
                    "key_factors": ["RSI > 60", "Above BB upper"],
                    "sources_cited": [],
                },
                {
                    "agent": "fundamental",
                    "ticker": "SPY",
                    "signal": "bullish",
                    "confidence": 0.7,
                    "reasoning": "Earnings beat expectations",
                    "key_factors": ["Q4 earnings +12%"],
                    "sources_cited": ["Reuters 2026-03-01"],
                },
                {
                    "agent": "bear",
                    "ticker": "SPY",
                    "signal": "bearish",
                    "confidence": 0.6,
                    "reasoning": "Overvalued, rate risk",
                    "key_factors": ["PE > 25", "Fed hawkish"],
                    "sources_cited": [],
                },
            ],
            "debate_log": [
                "[Context] Loaded SPY: prob_up=65.00%, features=9 indicators",
                "[Technical Analyst] bullish (confidence: 0.80)",
                "[Fundamental Analyst] bullish (confidence: 0.70)",
                "[Bear Agent] bearish (confidence: 0.60)",
                "[Portfolio Manager] BUY (confidence: 0.80)",
            ],
            "predictions": {"ticker": "SPY", "prob_up": 0.65, "expected_return": 0.012},
            "news_context": ["Reuters: SPY hits all-time high"],
        },
    }


@pytest.fixture()
def sample_forecast_rows() -> list[dict[str, Any]]:
    return [
        {
            "unique_id": "SPY",
            "ds": "2026-03-07",
            "PatchTST-q0.1": 490.0,
            "PatchTST-q0.25": 495.0,
            "PatchTST-q0.5": 500.0,
            "PatchTST-q0.75": 505.0,
            "PatchTST-q0.9": 510.0,
        },
        {
            "unique_id": "SPY",
            "ds": "2026-03-10",
            "PatchTST-q0.1": 491.0,
            "PatchTST-q0.25": 496.0,
            "PatchTST-q0.5": 501.0,
            "PatchTST-q0.75": 506.0,
            "PatchTST-q0.9": 511.0,
        },
        {
            "unique_id": "SPY",
            "ds": "2026-03-11",
            "PatchTST-q0.1": 492.0,
            "PatchTST-q0.25": 497.0,
            "PatchTST-q0.5": 502.0,
            "PatchTST-q0.75": 507.0,
            "PatchTST-q0.9": 512.0,
        },
    ]


# ---------------------------------------------------------------------------
# TestAgentStyles
# ---------------------------------------------------------------------------


class TestAgentStyles:
    def test_all_agents_have_styles(self) -> None:
        required = {"technical", "fundamental", "bear", "pm"}
        assert required == set(AGENT_STYLES.keys())

    def test_style_fields(self) -> None:
        for agent, style in AGENT_STYLES.items():
            assert "color" in style, f"{agent} missing color"
            assert "icon" in style, f"{agent} missing icon"
            assert "label" in style, f"{agent} missing label"


# ---------------------------------------------------------------------------
# TestDataLoaders
# ---------------------------------------------------------------------------


class TestDataLoaders:
    def test_load_decisions_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.dashboard.app.DATA_DIR", tmp_path)
        # Clear streamlit cache
        load_decisions.clear()
        result = load_decisions()
        assert result is None

    def test_load_decisions_valid(
        self,
        tmp_path: Path,
        sample_decisions: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("src.dashboard.app.DATA_DIR", tmp_path)
        (tmp_path / "decisions.json").write_text(
            json.dumps(sample_decisions), encoding="utf-8"
        )
        load_decisions.clear()
        result = load_decisions()
        assert result is not None
        assert result["tickers"] == ["SPY", "NVDA", "AAPL", "QQQ"]

    def test_load_debate_history_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.dashboard.app.DATA_DIR", tmp_path)
        load_debate_history.clear()
        result = load_debate_history()
        assert result is None

    def test_load_debate_history_valid(
        self,
        tmp_path: Path,
        sample_debate_history: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("src.dashboard.app.DATA_DIR", tmp_path)
        (tmp_path / "debate_history.json").write_text(
            json.dumps(sample_debate_history), encoding="utf-8"
        )
        load_debate_history.clear()
        result = load_debate_history()
        assert result is not None
        assert "SPY" in result
        assert len(result["SPY"]["reports"]) == 3


# ---------------------------------------------------------------------------
# TestCharts
# ---------------------------------------------------------------------------


class TestCharts:
    def test_weight_donut(self, sample_decisions: dict[str, Any]) -> None:
        fig = _chart_weight_donut(sample_decisions)
        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].type == "pie"

    def test_weight_donut_empty(self) -> None:
        fig = _chart_weight_donut({"hrp_final_weights": {}})
        assert fig is not None

    def test_weight_comparison(self, sample_decisions: dict[str, Any]) -> None:
        fig = _chart_weight_comparison(sample_decisions)
        assert fig is not None
        assert len(fig.data) == 2  # raw + tilted bars

    def test_quantile_fan_chart(
        self, sample_forecast_rows: list[dict[str, Any]]
    ) -> None:
        fig = _chart_quantile_fan(sample_forecast_rows, "SPY")
        assert fig is not None
        # 2 bands (4 traces) + 1 median = 5
        assert len(fig.data) == 5

    def test_quantile_fan_empty(self) -> None:
        fig = _chart_quantile_fan([], "SPY")
        assert fig is not None
        assert len(fig.data) == 0

    def test_quantile_fan_no_quantiles(self) -> None:
        rows = [{"ds": "2026-03-07", "close": 500.0}]
        fig = _chart_quantile_fan(rows, "SPY")
        assert fig is not None
        # No quantile columns → no traces
        assert len(fig.data) == 0


# ---------------------------------------------------------------------------
# TestNodeToAgent
# ---------------------------------------------------------------------------


class TestNodeToAgent:
    def test_analyst_nodes_mapped(self) -> None:
        assert _NODE_TO_AGENT["technical"] == "technical"
        assert _NODE_TO_AGENT["fundamental"] == "fundamental"
        assert _NODE_TO_AGENT["bear"] == "bear"
        assert _NODE_TO_AGENT["portfolio_manager"] == "pm"

    def test_context_nodes_mapped(self) -> None:
        assert "load_context" in _NODE_TO_AGENT
        assert "rag_retrieval" in _NODE_TO_AGENT


# ---------------------------------------------------------------------------
# TestReplayDebate
# ---------------------------------------------------------------------------


class TestReplayDebate:
    def test_replay_renders_all_reports(self) -> None:
        """_replay_debate should call _render_agent_report for each report."""
        reports = [
            {
                "agent": "technical",
                "signal": "bullish",
                "confidence": 0.8,
                "reasoning": "RSI shows momentum",
                "key_factors": ["RSI > 60"],
                "sources_cited": [],
            },
            {
                "agent": "bear",
                "signal": "bearish",
                "confidence": 0.6,
                "reasoning": "Overvalued",
                "key_factors": ["PE > 25"],
                "sources_cited": [],
            },
        ]
        with (
            patch("src.dashboard.app._render_agent_report") as mock_render,
            patch("src.dashboard.app.st") as mock_st,
            patch("src.dashboard.app.time") as mock_time,
        ):
            mock_placeholder = MagicMock()
            mock_st.empty.return_value = mock_placeholder
            _replay_debate(reports, delay=0.0)

        assert mock_render.call_count == 2

    def test_replay_uses_delay(self) -> None:
        """_replay_debate should sleep between reports."""
        reports = [
            {
                "agent": "technical",
                "signal": "bullish",
                "confidence": 0.8,
                "reasoning": "Test",
                "key_factors": [],
                "sources_cited": [],
            },
        ]
        with (
            patch("src.dashboard.app._render_agent_report"),
            patch("src.dashboard.app.st") as mock_st,
            patch("src.dashboard.app.time") as mock_time,
        ):
            mock_st.empty.return_value = MagicMock()
            _replay_debate(reports, delay=2.5)

        mock_time.sleep.assert_called_with(2.5)

    def test_replay_empty_reports(self) -> None:
        """_replay_debate with empty list does nothing."""
        with (
            patch("src.dashboard.app._render_agent_report") as mock_render,
            patch("src.dashboard.app.st"),
            patch("src.dashboard.app.time"),
        ):
            _replay_debate([], delay=0.0)
        mock_render.assert_not_called()


# ---------------------------------------------------------------------------
# TestLiveDebateThread
# ---------------------------------------------------------------------------


class TestLiveDebateThread:
    """Tests for _run_live_debate_thread.

    Uses sys.modules patching because src.agents.graph has heavy deps
    (langchain_anthropic) that may not be installed in test env.
    """

    @staticmethod
    def _make_fake_graph_module(
        side_effect: Any = None,
        return_value: Any = None,
    ) -> tuple[types.ModuleType, MagicMock]:
        """Create a fake src.agents.graph module with a mock run_agent_debate."""
        fake = types.ModuleType("src.agents.graph")
        mock_fn = MagicMock(side_effect=side_effect, return_value=return_value)
        fake.run_agent_debate = mock_fn  # type: ignore[attr-defined]
        return fake, mock_fn

    def test_posts_done_on_success(self) -> None:
        """Worker thread should post ('done', ...) on success."""
        q: queue.Queue = queue.Queue()
        fake_mod, mock_fn = self._make_fake_graph_module(
            return_value=(
                [{"ticker": "SPY", "action": "BUY", "confidence": 0.8}],
                {"SPY": {"reports": []}},
            )
        )

        with patch.dict(sys.modules, {"src.agents.graph": fake_mod}):
            _run_live_debate_thread(["SPY"], q)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        assert len(events) == 1
        assert events[0][0] == "done"

    def test_posts_error_on_failure(self) -> None:
        """Worker thread should post ('error', ...) on exception."""
        q: queue.Queue = queue.Queue()
        fake_mod, _ = self._make_fake_graph_module(
            side_effect=RuntimeError("LLM unavailable"),
        )

        with patch.dict(sys.modules, {"src.agents.graph": fake_mod}):
            _run_live_debate_thread(["SPY"], q)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        assert len(events) == 1
        assert events[0][0] == "error"
        assert "LLM unavailable" in events[0][1]

    def test_passes_callback_to_debate(self) -> None:
        """Worker thread should pass on_node_complete callback."""
        q: queue.Queue = queue.Queue()
        fake_mod, mock_fn = self._make_fake_graph_module(
            return_value=([], {}),
        )

        with patch.dict(sys.modules, {"src.agents.graph": fake_mod}):
            _run_live_debate_thread(["SPY"], q)

        call_kwargs = mock_fn.call_args
        assert call_kwargs.kwargs.get("on_node_complete") is not None

    def test_callback_posts_node_events(self) -> None:
        """The on_node callback should post ('node', ...) to queue."""
        q: queue.Queue = queue.Queue()

        def fake_debate(tickers, on_node_complete=None):
            if on_node_complete:
                on_node_complete("SPY", "technical", {"reports": [{"agent": "technical"}]})
                on_node_complete("SPY", "bear", {"reports": [{"agent": "bear"}]})
            return ([], {})

        fake_mod, _ = self._make_fake_graph_module(side_effect=fake_debate)

        with patch.dict(sys.modules, {"src.agents.graph": fake_mod}):
            _run_live_debate_thread(["SPY"], q)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        # 2 node events + 1 done event
        assert len(events) == 3
        assert events[0][0] == "node"
        assert events[0][1] == "SPY"
        assert events[0][2] == "technical"
        assert events[1][0] == "node"
        assert events[1][2] == "bear"
        assert events[2][0] == "done"
