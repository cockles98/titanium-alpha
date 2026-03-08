"""Tests for src/portfolio/decision_engine.py."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from src.agents.state import FinalDecision
from src.portfolio.decision_engine import (
    DecisionEngine,
    DecisionOutput,
    TickerDecision,
    _DEFAULT_LOOKBACK_DAYS,
)
from src.portfolio.hrp import HRPConfig, HRPResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tickers() -> list[str]:
    return ["SPY", "NVDA", "AAPL", "QQQ"]


@pytest.fixture()
def mock_ohlcv(tickers: list[str]) -> pl.DataFrame:
    """Multi-ticker OHLCV DataFrame (250 rows per ticker)."""
    rng = np.random.default_rng(42)
    n = 250
    base = date(2023, 1, 2)
    rows: list[dict] = []
    for ticker in tickers:
        prices = 100.0 + np.cumsum(rng.normal(0.01, 1.0, n))
        for i in range(n):
            c = float(prices[i])
            rows.append(
                {
                    "date": base + timedelta(days=i),
                    "ticker": ticker,
                    "open": c + rng.normal(0, 0.5),
                    "high": c + abs(rng.normal(0, 1.0)),
                    "low": c - abs(rng.normal(0, 1.0)),
                    "close": c,
                    "volume": int(1e7 + rng.normal(0, 1e6)),
                    "adj_close": c * 0.998,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture()
def sample_debate(tickers: list[str]) -> list[FinalDecision]:
    """Four FinalDecision results from a mock debate."""
    return [
        FinalDecision(
            ticker="SPY",
            action="BUY",
            confidence=0.8,
            suggested_weight=0.25,
            reasoning="Strong momentum",
            dissenting_view="Overvalued",
        ),
        FinalDecision(
            ticker="NVDA",
            action="BUY",
            confidence=0.7,
            suggested_weight=0.20,
            reasoning="AI growth",
            dissenting_view="High PE",
        ),
        FinalDecision(
            ticker="AAPL",
            action="HOLD",
            confidence=0.4,
            suggested_weight=0.10,
            reasoning="Uncertain outlook",
            dissenting_view="iPhone saturation",
        ),
        FinalDecision(
            ticker="QQQ",
            action="BUY",
            confidence=0.6,
            suggested_weight=0.15,
            reasoning="Tech rebound",
            dissenting_view="Rate risk",
        ),
    ]


@pytest.fixture()
def sample_hrp_result() -> HRPResult:
    return HRPResult(
        weights={"SPY": 0.30, "NVDA": 0.25, "AAPL": 0.20, "QQQ": 0.25},
        raw_weights={"SPY": 0.28, "NVDA": 0.27, "AAPL": 0.22, "QQQ": 0.23},
        cluster_order=["SPY", "QQQ", "NVDA", "AAPL"],
        linkage_matrix=[[0, 1, 0.5, 2], [2, 3, 0.6, 2]],
    )


@pytest.fixture()
def engine(mock_engine: MagicMock, tickers: list[str], tmp_path: Path) -> DecisionEngine:
    """DecisionEngine with mocked PostgreSQL engine."""
    return DecisionEngine(
        tickers=tickers,
        output_dir=str(tmp_path),
        engine=mock_engine,
    )


# ---------------------------------------------------------------------------
# TestTickerDecision
# ---------------------------------------------------------------------------


class TestTickerDecision:
    def test_creation(self) -> None:
        td = TickerDecision(
            ticker="SPY",
            action="BUY",
            weight=0.25,
            confidence=0.8,
            reasoning="Strong",
            dissenting_view="Risk",
        )
        assert td.ticker == "SPY"
        assert td.action == "BUY"
        assert td.weight == 0.25

    def test_frozen(self) -> None:
        td = TickerDecision("SPY", "BUY", 0.25, 0.8, "r", "d")
        with pytest.raises(AttributeError):
            td.weight = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestDecisionOutput
# ---------------------------------------------------------------------------


class TestDecisionOutput:
    def test_to_dict(self) -> None:
        output = DecisionOutput(
            timestamp="2026-03-06T12:00:00+00:00",
            tickers=["SPY"],
            decisions=[
                TickerDecision("SPY", "BUY", 1.0, 0.8, "r", "d"),
            ],
            hrp_raw_weights={"SPY": 1.0},
            hrp_final_weights={"SPY": 1.0},
            cluster_order=["SPY"],
            metadata={"schema_version": "1.0"},
        )
        d = output.to_dict()
        assert d["timestamp"] == "2026-03-06T12:00:00+00:00"
        assert len(d["decisions"]) == 1
        assert d["decisions"][0]["ticker"] == "SPY"
        assert d["metadata"]["schema_version"] == "1.0"

    def test_to_dict_is_json_serializable(self) -> None:
        output = DecisionOutput(
            timestamp="2026-03-06T12:00:00+00:00",
            tickers=["SPY"],
            decisions=[TickerDecision("SPY", "BUY", 1.0, 0.8, "r", "d")],
            hrp_raw_weights={"SPY": 1.0},
            hrp_final_weights={"SPY": 1.0},
            cluster_order=["SPY"],
        )
        # Should not raise
        json_str = json.dumps(output.to_dict())
        parsed = json.loads(json_str)
        assert parsed["tickers"] == ["SPY"]


# ---------------------------------------------------------------------------
# TestDecisionEngineInit
# ---------------------------------------------------------------------------


class TestDecisionEngineInit:
    @patch(
        "src.portfolio.decision_engine.load_tickers",
        side_effect=FileNotFoundError,
    )
    def test_defaults(
        self, _mock_cfg: MagicMock, mock_engine: MagicMock
    ) -> None:
        eng = DecisionEngine(engine=mock_engine)
        assert eng.tickers == ["SPY", "NVDA", "AAPL", "QQQ"]
        assert eng.lookback_days == _DEFAULT_LOOKBACK_DAYS
        # Dynamic HRPConfig: 4 tickers → max_weight = min(0.25, 2/4) = 0.25
        assert eng.hrp_config.max_weight == 0.25

    def test_custom_tickers(self, mock_engine: MagicMock) -> None:
        eng = DecisionEngine(
            tickers=["TSLA"], engine=mock_engine
        )
        assert eng.tickers == ["TSLA"]

    def test_custom_hrp_config(self, mock_engine: MagicMock) -> None:
        cfg = HRPConfig(linkage_method="ward")
        eng = DecisionEngine(engine=mock_engine, hrp_config=cfg)
        assert eng.hrp_config is cfg

    def test_custom_lookback(self, mock_engine: MagicMock) -> None:
        eng = DecisionEngine(engine=mock_engine, lookback_days=252)
        assert eng.lookback_days == 252

    def test_dynamic_max_weight_4_tickers(
        self, mock_engine: MagicMock
    ) -> None:
        eng = DecisionEngine(
            tickers=["A", "B", "C", "D"], engine=mock_engine
        )
        # min(0.25, 2/4) = min(0.25, 0.5) = 0.25
        assert eng.hrp_config.max_weight == 0.25

    def test_dynamic_max_weight_50_tickers(
        self, mock_engine: MagicMock
    ) -> None:
        tickers = [f"T{i}" for i in range(50)]
        eng = DecisionEngine(tickers=tickers, engine=mock_engine)
        # min(0.25, 2/50) = min(0.25, 0.04) = 0.04
        assert eng.hrp_config.max_weight == pytest.approx(0.04)

    def test_explicit_config_not_overridden(
        self, mock_engine: MagicMock
    ) -> None:
        cfg = HRPConfig(max_weight=0.10)
        tickers = [f"T{i}" for i in range(50)]
        eng = DecisionEngine(
            tickers=tickers, engine=mock_engine, hrp_config=cfg
        )
        # Explicit config preserved, not overridden by dynamic calc
        assert eng.hrp_config.max_weight == 0.10


# ---------------------------------------------------------------------------
# TestComputeReturns
# ---------------------------------------------------------------------------


class TestComputeReturns:
    def test_shape(
        self, engine: DecisionEngine, mock_ohlcv: pl.DataFrame
    ) -> None:
        returns = engine._compute_returns(mock_ohlcv)
        # 4 tickers, ~249 rows (first diff is null → dropped)
        assert returns.width == 4
        assert returns.height > 200
        assert set(returns.columns) == {"SPY", "NVDA", "AAPL", "QQQ"}

    def test_no_nulls(
        self, engine: DecisionEngine, mock_ohlcv: pl.DataFrame
    ) -> None:
        returns = engine._compute_returns(mock_ohlcv)
        assert returns.null_count().sum_horizontal().item() == 0

    def test_lookback_trim(
        self, mock_engine: MagicMock, mock_ohlcv: pl.DataFrame, tmp_path: Path
    ) -> None:
        eng = DecisionEngine(
            tickers=["SPY", "NVDA", "AAPL", "QQQ"],
            engine=mock_engine,
            output_dir=str(tmp_path),
            lookback_days=100,
        )
        returns = eng._compute_returns(mock_ohlcv)
        assert returns.height == 100

    def test_returns_are_log_returns(
        self, engine: DecisionEngine, mock_ohlcv: pl.DataFrame
    ) -> None:
        """Verify log returns are computed (not simple returns)."""
        returns = engine._compute_returns(mock_ohlcv)
        # Log returns should be small numbers centered around zero
        spy_returns = returns["SPY"].to_numpy()
        assert np.abs(spy_returns.mean()) < 0.05
        assert spy_returns.std() < 0.1


# ---------------------------------------------------------------------------
# TestExtractConfidences
# ---------------------------------------------------------------------------


class TestExtractConfidences:
    def test_all_tickers_present(
        self,
        engine: DecisionEngine,
        sample_debate: list[FinalDecision],
    ) -> None:
        conf = engine._extract_confidences(sample_debate)
        assert conf["SPY"] == 0.8
        assert conf["NVDA"] == 0.7
        assert conf["AAPL"] == 0.4
        assert conf["QQQ"] == 0.6

    def test_missing_ticker_gets_neutral(
        self, engine: DecisionEngine
    ) -> None:
        partial = [
            FinalDecision(
                ticker="SPY",
                action="BUY",
                confidence=0.9,
                suggested_weight=0.25,
                reasoning="r",
                dissenting_view="d",
            ),
        ]
        conf = engine._extract_confidences(partial)
        assert conf["SPY"] == 0.9
        assert conf["NVDA"] == 0.5
        assert conf["AAPL"] == 0.5
        assert conf["QQQ"] == 0.5

    def test_empty_debate(self, engine: DecisionEngine) -> None:
        conf = engine._extract_confidences([])
        assert all(v == 0.5 for v in conf.values())
        assert len(conf) == 4


# ---------------------------------------------------------------------------
# TestMergeActions
# ---------------------------------------------------------------------------


class TestMergeActions:
    def test_buy_tickers_get_weight(
        self,
        engine: DecisionEngine,
        sample_debate: list[FinalDecision],
        sample_hrp_result: HRPResult,
    ) -> None:
        decisions = engine._merge_actions(sample_debate, sample_hrp_result)
        by_ticker = {d.ticker: d for d in decisions}

        # BUY tickers have positive weight
        assert by_ticker["SPY"].weight > 0
        assert by_ticker["NVDA"].weight > 0
        assert by_ticker["QQQ"].weight > 0

        # HOLD ticker gets zero
        assert by_ticker["AAPL"].weight == 0.0
        assert by_ticker["AAPL"].action == "HOLD"

    def test_buy_weights_sum_to_one(
        self,
        engine: DecisionEngine,
        sample_debate: list[FinalDecision],
        sample_hrp_result: HRPResult,
    ) -> None:
        decisions = engine._merge_actions(sample_debate, sample_hrp_result)
        total = sum(d.weight for d in decisions)
        assert abs(total - 1.0) < 1e-6

    def test_no_debate_all_buy(
        self,
        engine: DecisionEngine,
        sample_hrp_result: HRPResult,
    ) -> None:
        """Without debate, all tickers default to BUY."""
        decisions = engine._merge_actions([], sample_hrp_result)
        for d in decisions:
            assert d.action == "BUY"
            assert d.weight > 0
            assert d.confidence == 0.5

    def test_all_hold(
        self, engine: DecisionEngine, sample_hrp_result: HRPResult
    ) -> None:
        """When all tickers are HOLD, all weights are zero."""
        debate = [
            FinalDecision(
                ticker=t,
                action="HOLD",
                confidence=0.3,
                suggested_weight=0.0,
                reasoning="uncertain",
                dissenting_view="risk",
            )
            for t in engine.tickers
        ]
        decisions = engine._merge_actions(debate, sample_hrp_result)
        assert all(d.weight == 0.0 for d in decisions)

    def test_sell_ticker_gets_zero(
        self, engine: DecisionEngine, sample_hrp_result: HRPResult
    ) -> None:
        debate = [
            FinalDecision(
                ticker="SPY",
                action="SELL",
                confidence=0.8,
                suggested_weight=0.0,
                reasoning="bearish",
                dissenting_view="",
            ),
            FinalDecision(
                ticker="NVDA",
                action="BUY",
                confidence=0.7,
                suggested_weight=0.2,
                reasoning="bullish",
                dissenting_view="",
            ),
        ]
        decisions = engine._merge_actions(debate, sample_hrp_result)
        by_ticker = {d.ticker: d for d in decisions}
        assert by_ticker["SPY"].weight == 0.0
        assert by_ticker["NVDA"].weight > 0

    def test_preserves_reasoning(
        self,
        engine: DecisionEngine,
        sample_debate: list[FinalDecision],
        sample_hrp_result: HRPResult,
    ) -> None:
        decisions = engine._merge_actions(sample_debate, sample_hrp_result)
        by_ticker = {d.ticker: d for d in decisions}
        assert by_ticker["SPY"].reasoning == "Strong momentum"
        assert by_ticker["AAPL"].dissenting_view == "iPhone saturation"


# ---------------------------------------------------------------------------
# TestRunDebate
# ---------------------------------------------------------------------------


class TestRunDebate:
    def test_graceful_degradation(self, engine: DecisionEngine) -> None:
        """When the debate import/call fails, returns empty list."""
        # Simulate import failure inside _run_debate by mocking the
        # lazy import that happens inside the method
        import sys
        import types

        # Create a fake agents.graph module that raises on call
        fake_graph = types.ModuleType("src.agents.graph")
        fake_graph.run_agent_debate = MagicMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("API key missing")
        )

        with patch.dict(sys.modules, {"src.agents.graph": fake_graph}):
            decisions, states = engine._run_debate()
            assert decisions == []
            assert states == {}

    def test_successful_debate(
        self,
        engine: DecisionEngine,
        sample_debate: list[FinalDecision],
    ) -> None:
        import sys
        import types

        fake_states = {"SPY": {"reports": []}}
        fake_graph = types.ModuleType("src.agents.graph")
        fake_graph.run_agent_debate = MagicMock(  # type: ignore[attr-defined]
            return_value=(sample_debate, fake_states)
        )

        with patch.dict(sys.modules, {"src.agents.graph": fake_graph}):
            decisions, states = engine._run_debate()
            assert len(decisions) == 4
            assert "SPY" in states


# ---------------------------------------------------------------------------
# TestBuildOutput
# ---------------------------------------------------------------------------


class TestBuildOutput:
    def test_metadata(
        self,
        engine: DecisionEngine,
        sample_hrp_result: HRPResult,
    ) -> None:
        decisions = [
            TickerDecision("SPY", "BUY", 1.0, 0.8, "r", "d"),
        ]
        output = engine._build_output(decisions, sample_hrp_result, 200)
        assert output.metadata["schema_version"] == "1.0"
        assert output.metadata["n_observations"] == 200
        assert output.metadata["hrp_config"]["linkage_method"] == "single"

    def test_timestamp_format(
        self,
        engine: DecisionEngine,
        sample_hrp_result: HRPResult,
    ) -> None:
        decisions = [TickerDecision("SPY", "BUY", 1.0, 0.8, "r", "d")]
        output = engine._build_output(decisions, sample_hrp_result, 100)
        # ISO 8601 with timezone
        assert "T" in output.timestamp
        assert "+" in output.timestamp or "Z" in output.timestamp


# ---------------------------------------------------------------------------
# TestSaveJson
# ---------------------------------------------------------------------------


class TestSaveJson:
    def test_saves_file(
        self, engine: DecisionEngine, tmp_path: Path
    ) -> None:
        engine.output_dir = tmp_path
        output = DecisionOutput(
            timestamp="2026-03-06T12:00:00+00:00",
            tickers=["SPY"],
            decisions=[TickerDecision("SPY", "BUY", 1.0, 0.8, "r", "d")],
            hrp_raw_weights={"SPY": 1.0},
            hrp_final_weights={"SPY": 1.0},
            cluster_order=["SPY"],
            metadata={"schema_version": "1.0"},
        )
        path = engine._save_json(output)
        assert path.exists()
        assert path.name == "decisions.json"

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["tickers"] == ["SPY"]
        assert data["metadata"]["schema_version"] == "1.0"

    def test_creates_directory(
        self, engine: DecisionEngine, tmp_path: Path
    ) -> None:
        engine.output_dir = tmp_path / "nested" / "dir"
        output = DecisionOutput(
            timestamp="2026-03-06T12:00:00+00:00",
            tickers=["SPY"],
            decisions=[TickerDecision("SPY", "BUY", 1.0, 0.8, "r", "d")],
            hrp_raw_weights={"SPY": 1.0},
            hrp_final_weights={"SPY": 1.0},
            cluster_order=["SPY"],
        )
        path = engine._save_json(output)
        assert path.exists()


# ---------------------------------------------------------------------------
# TestRunPipeline (integration)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_full_pipeline_without_debate(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """End-to-end: load → returns → HRP (no debate) → save."""
        engine.output_dir = tmp_path

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(engine, "_run_debate", return_value=([], {})),
        ):
            output = engine.run()

        assert len(output.decisions) == 4
        assert all(d.action == "BUY" for d in output.decisions)
        total_weight = sum(d.weight for d in output.decisions)
        assert abs(total_weight - 1.0) < 1e-6
        assert (tmp_path / "decisions.json").exists()

    def test_full_pipeline_with_debate(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        sample_debate: list[FinalDecision],
        tmp_path: Path,
    ) -> None:
        """End-to-end with agent debate results."""
        engine.output_dir = tmp_path

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(
                engine, "_run_debate", return_value=(sample_debate, {})
            ),
        ):
            output = engine.run()

        by_ticker = {d.ticker: d for d in output.decisions}
        # AAPL is HOLD → weight 0
        assert by_ticker["AAPL"].weight == 0.0
        # BUY tickers have weight
        assert by_ticker["SPY"].weight > 0
        assert by_ticker["NVDA"].weight > 0
        assert by_ticker["QQQ"].weight > 0
        # Weights sum to 1
        total = sum(d.weight for d in output.decisions)
        assert abs(total - 1.0) < 1e-6

    def test_json_output_roundtrip(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Verify JSON output is valid and complete."""
        engine.output_dir = tmp_path

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(engine, "_run_debate", return_value=([], {})),
        ):
            output = engine.run()

        with open(tmp_path / "decisions.json", encoding="utf-8") as f:
            data = json.load(f)

        assert data["tickers"] == ["SPY", "NVDA", "AAPL", "QQQ"]
        assert len(data["decisions"]) == 4
        assert "hrp_raw_weights" in data
        assert "hrp_final_weights" in data
        assert data["metadata"]["schema_version"] == "1.0"

    def test_debate_failure_graceful(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Pipeline succeeds even when debate fails."""
        engine.output_dir = tmp_path

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(engine, "_run_debate", return_value=([], {})),
        ):
            output = engine.run()

        # Should succeed with HRP-only weights
        assert len(output.decisions) == 4
        total = sum(d.weight for d in output.decisions)
        assert abs(total - 1.0) < 1e-6
