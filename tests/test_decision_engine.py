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
    _DEFAULT_LOOKBACK_DAYS,
    _PROB_UP_BUY,
    _PROB_UP_SELL,
    ClassificationConfig,
    DecisionEngine,
    DecisionOutput,
    TickerDecision,
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
            confidence=0.25,
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
        assert conf["AAPL"] == 0.25
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
    ) -> None:
        # Pre-computed weights: AAPL is HOLD (weight * 0.25)
        final_weights = {
            "SPY": 0.30,
            "NVDA": 0.25,
            "AAPL": 0.20 * 0.25,  # HOLD scaling
            "QQQ": 0.25,
        }
        decisions = engine._merge_actions(
            sample_debate, final_weights, sell_tickers=[]
        )
        by_ticker = {d.ticker: d for d in decisions}

        assert by_ticker["SPY"].weight > 0
        assert by_ticker["NVDA"].weight > 0
        assert by_ticker["QQQ"].weight > 0

        # HOLD now has reduced but positive weight
        assert by_ticker["AAPL"].weight > 0
        assert by_ticker["AAPL"].action == "HOLD"

    def test_weights_sum_lte_one(
        self,
        engine: DecisionEngine,
        sample_debate: list[FinalDecision],
    ) -> None:
        final_weights = {
            "SPY": 0.30,
            "NVDA": 0.25,
            "AAPL": 0.20 * 0.25,
            "QQQ": 0.25,
        }
        decisions = engine._merge_actions(
            sample_debate, final_weights, sell_tickers=[]
        )
        total = sum(d.weight for d in decisions)
        # With HOLD scaling, total <= 1.0 (implicit cash)
        assert total <= 1.0 + 1e-4
        assert total > 0

    def test_no_debate_all_buy(
        self,
        engine: DecisionEngine,
        sample_hrp_result: HRPResult,
    ) -> None:
        """Without debate, all tickers default to BUY."""
        final_weights = dict(sample_hrp_result.weights)
        decisions = engine._merge_actions(
            [], final_weights, sell_tickers=[]
        )
        for d in decisions:
            assert d.action == "BUY"
            assert d.weight > 0
            assert d.confidence == 0.5

    def test_all_hold(
        self, engine: DecisionEngine, sample_hrp_result: HRPResult
    ) -> None:
        """When all tickers are HOLD, weights are small but > 0."""
        debate = [
            FinalDecision(
                ticker=t,
                action="HOLD",
                confidence=0.25,
                suggested_weight=0.0,
                reasoning="uncertain",
                dissenting_view="risk",
            )
            for t in engine.tickers
        ]
        # HOLD scaling: HRP weight * confidence
        final_weights = {
            t: sample_hrp_result.weights[t] * 0.25
            for t in engine.tickers
        }
        decisions = engine._merge_actions(
            debate, final_weights, sell_tickers=[]
        )
        assert all(d.weight > 0 for d in decisions)
        total = sum(d.weight for d in decisions)
        assert total < 1.0

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
        final_weights = {
            "SPY": 0.0,
            "NVDA": 0.25,
            "AAPL": 0.20,
            "QQQ": 0.25,
        }
        decisions = engine._merge_actions(
            debate, final_weights, sell_tickers=["SPY"]
        )
        by_ticker = {d.ticker: d for d in decisions}
        assert by_ticker["SPY"].weight == 0.0
        assert by_ticker["NVDA"].weight > 0

    def test_preserves_reasoning(
        self,
        engine: DecisionEngine,
        sample_debate: list[FinalDecision],
    ) -> None:
        final_weights = {
            "SPY": 0.30,
            "NVDA": 0.25,
            "AAPL": 0.05,
            "QQQ": 0.25,
        }
        decisions = engine._merge_actions(
            sample_debate, final_weights, sell_tickers=[]
        )
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
        output = engine._build_output(
            decisions, sample_hrp_result, 200,
            invested_fraction=0.85,
            confidence_source="debate",
            n_buy=3, n_hold=1, n_sell=0,
        )
        assert output.metadata["schema_version"] == "1.2"
        assert output.metadata["n_observations"] == 200
        assert output.metadata["hrp_config"]["linkage_method"] == "ward"
        assert output.metadata["invested_fraction"] == 0.85
        assert output.metadata["confidence_source"] == "debate"
        assert output.metadata["n_buy"] == 3
        assert output.metadata["n_hold"] == 1
        assert output.metadata["n_sell"] == 0

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
        # AAPL is HOLD → reduced but positive weight
        assert by_ticker["AAPL"].weight > 0
        assert by_ticker["AAPL"].action == "HOLD"
        # BUY tickers have weight
        assert by_ticker["SPY"].weight > 0
        assert by_ticker["NVDA"].weight > 0
        assert by_ticker["QQQ"].weight > 0
        # Total <= 1.0 (implicit cash due to HOLD scaling)
        total = sum(d.weight for d in output.decisions)
        assert total <= 1.0 + 1e-4
        assert total > 0
        # New metadata fields
        assert "invested_fraction" in output.metadata
        assert "confidence_source" in output.metadata
        assert output.metadata["n_hold"] == 1

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
        assert data["metadata"]["schema_version"] == "1.2"

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


# ---------------------------------------------------------------------------
# TestLoadPredictions
# ---------------------------------------------------------------------------


class TestLoadPredictions:
    def test_load_predictions_success(
        self, engine: DecisionEngine, tmp_path: Path
    ) -> None:
        engine.output_dir = tmp_path
        df = pl.DataFrame(
            {
                "ticker": ["SPY", "NVDA", "AAPL", "QQQ"],
                "prob_up": [0.7, 0.6, 0.3, 0.8],
            }
        )
        df.write_parquet(tmp_path / "predictions.parquet")

        result = engine._load_predictions()
        assert result is not None
        assert result["SPY"] == pytest.approx(0.7)
        assert result["AAPL"] == pytest.approx(0.3)
        assert len(result) == 4

    def test_load_predictions_missing_file(
        self, engine: DecisionEngine, tmp_path: Path
    ) -> None:
        engine.output_dir = tmp_path
        result = engine._load_predictions()
        assert result is None

    def test_load_predictions_partial(
        self, engine: DecisionEngine, tmp_path: Path
    ) -> None:
        """Only tickers in self.tickers are returned."""
        engine.output_dir = tmp_path
        df = pl.DataFrame(
            {
                "ticker": ["SPY", "TSLA"],  # TSLA not in engine.tickers
                "prob_up": [0.65, 0.9],
            }
        )
        df.write_parquet(tmp_path / "predictions.parquet")

        result = engine._load_predictions()
        assert result is not None
        assert "SPY" in result
        assert "TSLA" not in result
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestClassifyTickers
# ---------------------------------------------------------------------------


class TestClassifyTickers:
    def test_classify_tickers_mixed(
        self,
        engine: DecisionEngine,
        sample_debate: list[FinalDecision],
    ) -> None:
        buy, hold, sell, _ = engine._classify_tickers(sample_debate)
        assert set(buy) == {"SPY", "NVDA", "QQQ"}
        assert hold == ["AAPL"]
        assert sell == []

    def test_classify_tickers_no_debate(
        self, engine: DecisionEngine
    ) -> None:
        """Empty debate → all tickers default to BUY."""
        buy, hold, sell, _ = engine._classify_tickers([])
        assert set(buy) == {"SPY", "NVDA", "AAPL", "QQQ"}
        assert hold == []
        assert sell == []

    def test_classify_tickers_with_sell(
        self, engine: DecisionEngine
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
        buy, hold, sell, _ = engine._classify_tickers(debate)
        assert "SPY" in sell
        assert "NVDA" in buy
        # AAPL, QQQ have no debate → BUY
        assert "AAPL" in buy
        assert "QQQ" in buy


# ---------------------------------------------------------------------------
# TestExtractConfidencesWithFallback
# ---------------------------------------------------------------------------


class TestExtractConfidencesWithFallback:
    def test_fallback_fills_missing(
        self, engine: DecisionEngine
    ) -> None:
        partial_debate = [
            FinalDecision(
                ticker="SPY",
                action="BUY",
                confidence=0.8,
                suggested_weight=0.25,
                reasoning="r",
                dissenting_view="d",
            ),
        ]
        fallback = {"NVDA": 0.65, "AAPL": 0.3, "QQQ": 0.75}
        conf = engine._extract_confidences(
            partial_debate, fallback=fallback
        )
        assert conf["SPY"] == 0.8  # from debate
        assert conf["NVDA"] == 0.65  # from fallback
        assert conf["AAPL"] == 0.3  # from fallback
        assert conf["QQQ"] == 0.75  # from fallback

    def test_no_debate_all_fallback(
        self, engine: DecisionEngine
    ) -> None:
        fallback = {
            "SPY": 0.7,
            "NVDA": 0.6,
            "AAPL": 0.3,
            "QQQ": 0.8,
        }
        conf = engine._extract_confidences([], fallback=fallback)
        assert conf["SPY"] == 0.7
        assert conf["QQQ"] == 0.8

    def test_fallback_partial_coverage(
        self, engine: DecisionEngine
    ) -> None:
        """Tickers missing from both debate and fallback get 0.5."""
        fallback = {"SPY": 0.65}
        conf = engine._extract_confidences([], fallback=fallback)
        assert conf["SPY"] == 0.65
        assert conf["NVDA"] == 0.5  # no fallback, neutral
        assert conf["QQQ"] == 0.5


# ---------------------------------------------------------------------------
# TestHoldReducedWeight
# ---------------------------------------------------------------------------


class TestHoldReducedWeight:
    def test_hold_gets_reduced_weight(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        sample_debate: list[FinalDecision],
        tmp_path: Path,
    ) -> None:
        """HOLD tickers get weight = HRP_weight * confidence."""
        engine.output_dir = tmp_path

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(
                engine,
                "_run_debate",
                return_value=(sample_debate, {}),
            ),
        ):
            output = engine.run()

        by_ticker = {d.ticker: d for d in output.decisions}
        aapl = by_ticker["AAPL"]
        assert aapl.action == "HOLD"
        assert aapl.weight > 0
        # HOLD weight should be much smaller than BUY weights
        buy_weights = [
            d.weight
            for d in output.decisions
            if d.action == "BUY"
        ]
        assert aapl.weight < min(buy_weights)


# ---------------------------------------------------------------------------
# TestInvestedFraction
# ---------------------------------------------------------------------------


class TestInvestedFraction:
    def test_invested_fraction_in_metadata(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        sample_debate: list[FinalDecision],
        tmp_path: Path,
    ) -> None:
        """Metadata includes invested_fraction < 1.0 when HOLD present."""
        engine.output_dir = tmp_path

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(
                engine,
                "_run_debate",
                return_value=(sample_debate, {}),
            ),
        ):
            output = engine.run()

        frac = output.metadata["invested_fraction"]
        assert 0 < frac < 1.0
        # Should match sum of weights
        total = sum(d.weight for d in output.decisions)
        assert abs(frac - total) < 1e-4

    def test_invested_fraction_one_without_debate(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Without debate, all BUY → invested_fraction ≈ 1.0."""
        engine.output_dir = tmp_path

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(engine, "_run_debate", return_value=([], {})),
        ):
            output = engine.run()

        assert output.metadata["invested_fraction"] == pytest.approx(
            1.0, abs=1e-6
        )
        assert output.metadata["confidence_source"] == "none"
        assert output.metadata["n_buy"] == 4
        assert output.metadata["n_hold"] == 0
        assert output.metadata["n_sell"] == 0


# ---------------------------------------------------------------------------
# TestClassifyTickersWithFallback
# ---------------------------------------------------------------------------


class TestClassifyTickersWithFallback:
    """Tests for _classify_tickers when debate is empty but fallback is given.

    Verifies that prob_up thresholds (_PROB_UP_BUY=0.5, _PROB_UP_SELL=0.3)
    drive the BUY/HOLD/SELL classification when the LangGraph debate returns
    no results and PatchTST predictions are used as the fallback source.
    """

    def test_buy_at_exact_threshold(self, engine: DecisionEngine) -> None:
        """prob_up == _PROB_UP_BUY (0.5) -> BUY (boundary inclusive)."""
        fallback = {
            "SPY": _PROB_UP_BUY,   # exactly 0.5 -- BUY
            "NVDA": 0.8,
            "AAPL": 0.4,
            "QQQ": 0.1,
        }
        buy, hold, sell, _ = engine._classify_tickers([], fallback=fallback)
        assert "SPY" in buy
        assert "NVDA" in buy

    def test_hold_range(self, engine: DecisionEngine) -> None:
        """0.3 <= prob_up < 0.5 -> HOLD for all four tickers."""
        fallback = {
            "SPY": 0.49,   # just below BUY
            "NVDA": 0.30,  # exactly SELL threshold -> HOLD (not SELL)
            "AAPL": 0.35,
            "QQQ": 0.45,
        }
        buy, hold, sell, _ = engine._classify_tickers([], fallback=fallback)
        assert buy == []
        assert sell == []
        assert set(hold) == {"SPY", "NVDA", "AAPL", "QQQ"}

    def test_sell_below_threshold(self, engine: DecisionEngine) -> None:
        """prob_up < _PROB_UP_SELL (0.3) -> SELL."""
        fallback = {
            "SPY": 0.8,
            "NVDA": 0.6,
            "AAPL": 0.29,   # just below 0.3 -> SELL
            "QQQ": 0.0,     # zero -> SELL
        }
        buy, hold, sell, _ = engine._classify_tickers([], fallback=fallback)
        assert "AAPL" in sell
        assert "QQQ" in sell
        assert "SPY" in buy
        assert "NVDA" in buy

    def test_mixed_classification(self, engine: DecisionEngine) -> None:
        """Mixed prob_up values produce correct BUY/HOLD/SELL split."""
        fallback = {
            "SPY": 0.8,    # BUY
            "NVDA": 0.6,   # BUY
            "AAPL": 0.2,   # SELL  (< 0.3)
            "QQQ": 0.45,   # HOLD  ([0.3, 0.5))
        }
        buy, hold, sell, _ = engine._classify_tickers([], fallback=fallback)
        assert set(buy) == {"SPY", "NVDA"}
        assert hold == ["QQQ"]
        assert sell == ["AAPL"]

    def test_debate_takes_priority_over_fallback(
        self, engine: DecisionEngine, sample_debate: list[FinalDecision]
    ) -> None:
        """Tickers in the debate keep their debate action even if fallback disagrees."""
        # sample_debate: SPY=BUY, NVDA=BUY, AAPL=HOLD, QQQ=BUY
        # Fallback tries to override SPY to SELL -- should be ignored
        fallback = {
            "SPY": 0.1,   # would be SELL if debate absent
            "NVDA": 0.1,  # would be SELL if debate absent
            "AAPL": 0.8,  # would be BUY if debate absent
            "QQQ": 0.1,   # would be SELL if debate absent
        }
        buy, hold, sell, _ = engine._classify_tickers(
            sample_debate, fallback=fallback
        )
        assert "SPY" in buy    # debate wins (BUY)
        assert "NVDA" in buy   # debate wins (BUY)
        assert "AAPL" in hold  # debate wins (HOLD)
        assert "QQQ" in buy    # debate wins (BUY)
        assert sell == []

    def test_empty_fallback_entry_defaults_to_buy(
        self, engine: DecisionEngine
    ) -> None:
        """Ticker absent from both debate and fallback defaults to BUY."""
        # Only SPY has fallback; NVDA/AAPL/QQQ have none -> BUY
        fallback = {"SPY": 0.1}  # SPY -> SELL; others -> BUY (no fallback)
        buy, hold, sell, _ = engine._classify_tickers([], fallback=fallback)
        assert "SPY" in sell
        assert "NVDA" in buy
        assert "AAPL" in buy
        assert "QQQ" in buy


# ---------------------------------------------------------------------------
# TestMergeActionsWithFallback
# ---------------------------------------------------------------------------


class TestMergeActionsWithFallback:
    """Tests for _merge_actions when non-debate tickers carry fallback info.

    Validates that:
    - action reflects hold_tickers / sell_tickers rather than a hardcoded BUY
    - confidence comes from the confidences dict (not hardcoded 0.5)
    - reasoning includes 'PatchTST signal (prob_up=X.XX)'
    """

    def test_sell_action_for_non_debate_ticker(
        self, engine: DecisionEngine
    ) -> None:
        """Non-debate ticker in sell_tickers gets action=SELL and weight=0."""
        final_weights = {"SPY": 0.5, "NVDA": 0.5, "AAPL": 0.0, "QQQ": 0.0}
        confidences = {"SPY": 0.7, "NVDA": 0.65, "AAPL": 0.2, "QQQ": 0.45}
        decisions = engine._merge_actions(
            [],  # no debate
            final_weights,
            sell_tickers=["AAPL"],
            hold_tickers=["QQQ"],
            confidences=confidences,
        )
        by_ticker = {d.ticker: d for d in decisions}
        assert by_ticker["AAPL"].action == "SELL"
        assert by_ticker["AAPL"].weight == 0.0

    def test_hold_action_for_non_debate_ticker(
        self, engine: DecisionEngine
    ) -> None:
        """Non-debate ticker in hold_tickers gets action=HOLD."""
        final_weights = {"SPY": 0.4, "NVDA": 0.4, "AAPL": 0.0, "QQQ": 0.1}
        confidences = {"SPY": 0.7, "NVDA": 0.65, "AAPL": 0.15, "QQQ": 0.42}
        decisions = engine._merge_actions(
            [],  # no debate
            final_weights,
            sell_tickers=["AAPL"],
            hold_tickers=["QQQ"],
            confidences=confidences,
        )
        by_ticker = {d.ticker: d for d in decisions}
        assert by_ticker["QQQ"].action == "HOLD"

    def test_fallback_confidence_used_not_hardcoded(
        self, engine: DecisionEngine
    ) -> None:
        """Confidence on a non-debate ticker must come from confidences dict."""
        final_weights = {"SPY": 0.5, "NVDA": 0.5, "AAPL": 0.0, "QQQ": 0.0}
        confidences = {
            "SPY": 0.73,   # unique values so we can detect hardcoding
            "NVDA": 0.61,
            "AAPL": 0.22,
            "QQQ": 0.44,
        }
        decisions = engine._merge_actions(
            [],
            final_weights,
            sell_tickers=["AAPL", "QQQ"],
            confidences=confidences,
        )
        by_ticker = {d.ticker: d for d in decisions}
        assert by_ticker["AAPL"].confidence == pytest.approx(0.22)
        assert by_ticker["QQQ"].confidence == pytest.approx(0.44)
        assert by_ticker["SPY"].confidence == pytest.approx(0.73)
        assert by_ticker["NVDA"].confidence == pytest.approx(0.61)

    def test_reasoning_contains_patchtst_signal(
        self, engine: DecisionEngine
    ) -> None:
        """Non-debate tickers with confidences show PatchTST signal in reasoning."""
        final_weights = {"SPY": 0.5, "NVDA": 0.4, "AAPL": 0.0, "QQQ": 0.1}
        confidences = {"SPY": 0.80, "NVDA": 0.60, "AAPL": 0.20, "QQQ": 0.45}
        decisions = engine._merge_actions(
            [],
            final_weights,
            sell_tickers=["AAPL"],
            hold_tickers=["QQQ"],
            confidences=confidences,
        )
        by_ticker = {d.ticker: d for d in decisions}
        assert "PatchTST signal" in by_ticker["SPY"].reasoning
        assert "prob_up=0.80" in by_ticker["SPY"].reasoning
        assert "PatchTST signal" in by_ticker["AAPL"].reasoning
        assert "prob_up=0.20" in by_ticker["AAPL"].reasoning
        assert "PatchTST signal" in by_ticker["QQQ"].reasoning
        assert "prob_up=0.45" in by_ticker["QQQ"].reasoning

    def test_no_confidences_defaults_buy_half(
        self, engine: DecisionEngine
    ) -> None:
        """When confidences=None and no debate, action=BUY and confidence=0.5 (legacy path)."""
        final_weights = {"SPY": 0.25, "NVDA": 0.25, "AAPL": 0.25, "QQQ": 0.25}
        decisions = engine._merge_actions(
            [],
            final_weights,
            sell_tickers=[],
            confidences=None,
        )
        for d in decisions:
            assert d.action == "BUY"
            assert d.confidence == 0.5
            assert d.reasoning == "No agent debate available"


# ---------------------------------------------------------------------------
# TestPipelineWithPatchTSTFallback (integration)
# ---------------------------------------------------------------------------


class TestPipelineWithPatchTSTFallback:
    """Integration tests: full pipeline using PatchTST fallback (no debate).

    When _run_debate returns empty but predictions.parquet exists, the
    engine must classify tickers by prob_up thresholds and reflect those
    classifications in the final DecisionOutput.
    """

    @staticmethod
    def _write_predictions(tmp_path: Path, prob_ups: dict) -> None:
        """Write predictions.parquet to tmp_path."""
        df = pl.DataFrame(
            {
                "ticker": list(prob_ups.keys()),
                "prob_up": [float(v) for v in prob_ups.values()],
            }
        )
        df.write_parquet(tmp_path / "predictions.parquet")

    def test_aapl_sell_below_threshold(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """AAPL prob_up=0.2 < 0.3 -> SELL with weight=0."""
        engine.output_dir = tmp_path
        self._write_predictions(
            tmp_path,
            {"SPY": 0.8, "NVDA": 0.6, "AAPL": 0.2, "QQQ": 0.45},
        )

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(engine, "_run_debate", return_value=([], {})),
        ):
            output = engine.run()

        by_ticker = {d.ticker: d for d in output.decisions}
        assert by_ticker["AAPL"].action == "SELL"
        assert by_ticker["AAPL"].weight == pytest.approx(0.0)

    def test_qqq_hold_in_range(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """QQQ prob_up=0.45 in [0.3, 0.5) -> HOLD."""
        engine.output_dir = tmp_path
        self._write_predictions(
            tmp_path,
            {"SPY": 0.8, "NVDA": 0.6, "AAPL": 0.2, "QQQ": 0.45},
        )

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(engine, "_run_debate", return_value=([], {})),
        ):
            output = engine.run()

        by_ticker = {d.ticker: d for d in output.decisions}
        assert by_ticker["QQQ"].action == "HOLD"

    def test_spy_nvda_buy_above_threshold(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """SPY prob_up=0.8 and NVDA prob_up=0.6 -> BUY with positive weight."""
        engine.output_dir = tmp_path
        self._write_predictions(
            tmp_path,
            {"SPY": 0.8, "NVDA": 0.6, "AAPL": 0.2, "QQQ": 0.45},
        )

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(engine, "_run_debate", return_value=([], {})),
        ):
            output = engine.run()

        by_ticker = {d.ticker: d for d in output.decisions}
        assert by_ticker["SPY"].action == "BUY"
        assert by_ticker["SPY"].weight > 0
        assert by_ticker["NVDA"].action == "BUY"
        assert by_ticker["NVDA"].weight > 0

    def test_full_mixed_pipeline(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Full pipeline with mixed prob_up: correct counts and confidence_source."""
        engine.output_dir = tmp_path
        self._write_predictions(
            tmp_path,
            {"SPY": 0.8, "NVDA": 0.6, "AAPL": 0.2, "QQQ": 0.45},
        )

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(engine, "_run_debate", return_value=([], {})),
        ):
            output = engine.run()

        meta = output.metadata
        assert meta["confidence_source"] == "patchtst"
        assert meta["n_buy"] == 2   # SPY, NVDA
        assert meta["n_hold"] == 1  # QQQ
        assert meta["n_sell"] == 1  # AAPL
        # SELL has zero weight; investable fraction < 1
        assert meta["invested_fraction"] < 1.0

    def test_sell_ticker_excluded_from_hrp(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """SELL ticker weight is exactly zero even if HRP would otherwise allocate to it."""
        engine.output_dir = tmp_path
        self._write_predictions(
            tmp_path,
            {"SPY": 0.8, "NVDA": 0.6, "AAPL": 0.2, "QQQ": 0.45},
        )

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(engine, "_run_debate", return_value=([], {})),
        ):
            output = engine.run()

        by_ticker = {d.ticker: d for d in output.decisions}
        # HRP runs only on investable subset (SPY, NVDA, QQQ);
        # AAPL must not appear with positive weight
        assert output.hrp_final_weights.get("AAPL", 0.0) == pytest.approx(0.0)
        assert by_ticker["AAPL"].weight == pytest.approx(0.0)

    def test_confidence_source_patchtst_in_json(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """decisions.json reflects confidence_source='patchtst' when only fallback is used."""
        engine.output_dir = tmp_path
        self._write_predictions(
            tmp_path,
            {"SPY": 0.8, "NVDA": 0.6, "AAPL": 0.2, "QQQ": 0.45},
        )

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(engine, "_run_debate", return_value=([], {})),
        ):
            engine.run()

        with open(tmp_path / "decisions.json", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["confidence_source"] == "patchtst"
        decisions_map = {d["ticker"]: d for d in data["decisions"]}
        assert decisions_map["AAPL"]["action"] == "SELL"
        assert decisions_map["AAPL"]["weight"] == pytest.approx(0.0)
        assert decisions_map["QQQ"]["action"] == "HOLD"
        assert decisions_map["SPY"]["action"] == "BUY"
        assert decisions_map["NVDA"]["action"] == "BUY"


# ---------------------------------------------------------------------------
# TestHoldRampSmoothing
# ---------------------------------------------------------------------------


class TestHoldRampSmoothing:
    """Verify the linear ramp that smooths the BUY/HOLD cliff for fallback."""

    def test_near_buy_boundary_gets_high_weight(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """prob_up=0.49 (just below BUY) should get ~95% of HRP weight,
        not 49% as raw conf would give."""
        engine.output_dir = tmp_path
        # SPY BUY, NVDA BUY, AAPL HOLD near boundary, QQQ BUY
        df = pl.DataFrame(
            {"ticker": ["SPY", "NVDA", "AAPL", "QQQ"],
             "prob_up": [0.8, 0.7, 0.49, 0.6]}
        )
        df.write_parquet(tmp_path / "predictions.parquet")

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(engine, "_run_debate", return_value=([], {})),
        ):
            output = engine.run()

        by_ticker = {d.ticker: d for d in output.decisions}
        aapl = by_ticker["AAPL"]
        assert aapl.action == "HOLD"
        # Ramp: (0.49 - 0.3) / (0.5 - 0.3) = 0.95
        # So weight should be ≈ 95% of its HRP weight, not 49%.
        buy_weights = [
            d.weight for d in output.decisions if d.action == "BUY"
        ]
        avg_buy = sum(buy_weights) / len(buy_weights)
        # HOLD near boundary should be close to average BUY weight
        assert aapl.weight > 0.7 * avg_buy

    def test_near_sell_boundary_gets_low_weight(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """prob_up=0.31 (just above SELL) should get ~5% of HRP weight."""
        engine.output_dir = tmp_path
        df = pl.DataFrame(
            {"ticker": ["SPY", "NVDA", "AAPL", "QQQ"],
             "prob_up": [0.8, 0.7, 0.31, 0.6]}
        )
        df.write_parquet(tmp_path / "predictions.parquet")

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(engine, "_run_debate", return_value=([], {})),
        ):
            output = engine.run()

        by_ticker = {d.ticker: d for d in output.decisions}
        aapl = by_ticker["AAPL"]
        assert aapl.action == "HOLD"
        # Ramp: (0.31 - 0.3) / (0.5 - 0.3) = 0.05
        buy_weights = [
            d.weight for d in output.decisions if d.action == "BUY"
        ]
        avg_buy = sum(buy_weights) / len(buy_weights)
        assert aapl.weight < 0.15 * avg_buy

    def test_debate_hold_not_ramped(
        self,
        engine: DecisionEngine,
        mock_ohlcv: pl.DataFrame,
        sample_debate: list[FinalDecision],
        tmp_path: Path,
    ) -> None:
        """Debate HOLD uses raw confidence, not the ramp."""
        engine.output_dir = tmp_path

        with (
            patch.object(engine, "_load_ohlcv", return_value=mock_ohlcv),
            patch.object(
                engine, "_run_debate",
                return_value=(sample_debate, {}),
            ),
        ):
            output = engine.run()

        by_ticker = {d.ticker: d for d in output.decisions}
        aapl = by_ticker["AAPL"]
        assert aapl.action == "HOLD"
        assert aapl.confidence == 0.25
        # Debate HOLD: raw conf=0.25, NOT ramped.
        # Weight = hrp_w * 0.25 (small).
        buy_weights = [
            d.weight for d in output.decisions if d.action == "BUY"
        ]
        avg_buy = sum(buy_weights) / len(buy_weights)
        assert aapl.weight < 0.4 * avg_buy


# ---------------------------------------------------------------------------
# TestPercentileClassification
# ---------------------------------------------------------------------------


class TestPercentileClassification:
    """Tests for percentile-based ticker classification.

    Verifies that when enough fallback tickers are available
    (>= min_tickers_for_percentile), thresholds are computed from
    the cross-sectional distribution instead of fixed absolutes.
    """

    @pytest.fixture()
    def engine_10(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> DecisionEngine:
        """DecisionEngine with 10 tickers (>= min_tickers_for_percentile)."""
        tickers = [f"T{i}" for i in range(10)]
        return DecisionEngine(
            tickers=tickers,
            output_dir=str(tmp_path),
            engine=mock_engine,
        )

    def test_compute_percentile_thresholds(self) -> None:
        """Direct test of threshold computation."""
        prob_ups = {f"T{i}": 0.48 + i * 0.01 for i in range(10)}
        config = ClassificationConfig()
        sell_thresh, buy_thresh = (
            DecisionEngine._compute_percentile_thresholds(
                prob_ups, config
            )
        )
        # 15th percentile of [0.48..0.57], 40th percentile
        assert 0.48 <= sell_thresh < 0.50
        assert 0.50 <= buy_thresh < 0.54
        assert sell_thresh < buy_thresh

    def test_narrow_range_produces_sells(
        self, engine_10: DecisionEngine
    ) -> None:
        """Even with narrow prob_up range [0.48, 0.57], percentile
        produces SELLs — the key fix for PatchTST's narrow output."""
        fallback = {f"T{i}": 0.48 + i * 0.01 for i in range(10)}
        buy, hold, sell, _ = engine_10._classify_tickers(
            [], fallback=fallback
        )
        # Absolute thresholds would give 0 SELL (all > 0.3).
        # Percentile must produce at least 1 SELL.
        assert len(sell) >= 1
        assert len(buy) + len(hold) + len(sell) == 10
        # SELL tickers have lowest prob_ups
        sell_probs = [fallback[t] for t in sell]
        buy_probs = [fallback[t] for t in buy]
        if sell_probs and buy_probs:
            assert max(sell_probs) <= min(buy_probs)

    def test_absolute_fallback_when_few_tickers(
        self, engine: DecisionEngine
    ) -> None:
        """With 4 tickers (< min=5), absolute thresholds are used."""
        fallback = {
            "SPY": 0.52,
            "NVDA": 0.53,
            "AAPL": 0.48,
            "QQQ": 0.55,
        }
        _, _, _, (sell_thresh, buy_thresh) = engine._classify_tickers(
            [], fallback=fallback
        )
        assert sell_thresh == _PROB_UP_SELL
        assert buy_thresh == _PROB_UP_BUY

    def test_all_same_prob_up_no_sells(
        self, engine_10: DecisionEngine
    ) -> None:
        """When all prob_ups are identical, all BUY (no differentiation)."""
        fallback = {f"T{i}": 0.52 for i in range(10)}
        buy, hold, sell, _ = engine_10._classify_tickers(
            [], fallback=fallback
        )
        assert len(buy) == 10
        assert len(sell) == 0
        assert len(hold) == 0

    def test_custom_classification_config(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Custom ClassificationConfig changes the split."""
        tickers = [f"T{i}" for i in range(10)]
        cfg = ClassificationConfig(
            sell_percentile=0.30,
            hold_percentile=0.70,
        )
        eng = DecisionEngine(
            tickers=tickers,
            output_dir=str(tmp_path),
            engine=mock_engine,
            classification_config=cfg,
        )
        fallback = {f"T{i}": 0.48 + i * 0.01 for i in range(10)}
        buy, hold, sell, _ = eng._classify_tickers(
            [], fallback=fallback
        )
        # 30/70 split: ~3 SELL, ~4 HOLD, ~3 BUY
        assert len(sell) >= 2
        assert len(buy) >= 2
        assert len(hold) >= 2
        assert len(buy) + len(hold) + len(sell) == 10

    def test_debate_excluded_from_percentile_pool(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Debate tickers don't contribute to percentile computation."""
        tickers = [f"T{i}" for i in range(6)]
        eng = DecisionEngine(
            tickers=tickers,
            output_dir=str(tmp_path),
            engine=mock_engine,
        )
        debate = [
            FinalDecision(
                ticker="T0",
                action="BUY",
                confidence=0.9,
                suggested_weight=0.2,
                reasoning="r",
                dissenting_view="d",
            ),
        ]
        fallback = {
            "T0": 0.10,  # debate overrides — not in percentile pool
            "T1": 0.48,
            "T2": 0.50,
            "T3": 0.52,
            "T4": 0.54,
            "T5": 0.56,
        }
        buy, hold, sell, (sell_thresh, _) = eng._classify_tickers(
            debate, fallback=fallback
        )
        assert "T0" in buy  # from debate, not percentile
        # sell_thresh from [0.48..0.56], not dragged down by 0.10
        assert sell_thresh > 0.40

    def test_percentile_classification_counts(
        self, engine_10: DecisionEngine
    ) -> None:
        """Default config (15/40): bottom 15% SELL, 15-40% HOLD, top 60% BUY."""
        fallback = {f"T{i}": 0.48 + i * 0.01 for i in range(10)}
        buy, hold, sell, _ = engine_10._classify_tickers(
            [], fallback=fallback
        )
        assert len(buy) >= 5  # top 60% of 10
        assert len(sell) >= 1  # bottom 15% of 10
        assert len(buy) + len(hold) + len(sell) == 10

    def test_schema_1_2_with_classification_metadata(
        self,
        mock_engine: MagicMock,
        tmp_path: Path,
    ) -> None:
        """_build_output metadata includes classification block."""
        eng = DecisionEngine(
            tickers=["SPY"],
            output_dir=str(tmp_path),
            engine=mock_engine,
        )
        hrp_result = HRPResult(
            weights={"SPY": 1.0},
            raw_weights={"SPY": 1.0},
            cluster_order=["SPY"],
            linkage_matrix=[],
        )
        decisions = [
            TickerDecision("SPY", "BUY", 1.0, 0.8, "r", "d"),
        ]
        output = eng._build_output(
            decisions,
            hrp_result,
            200,
            classification_method="percentile",
            sell_threshold=0.485,
            buy_threshold=0.516,
        )
        meta = output.metadata
        assert meta["schema_version"] == "1.2"
        assert meta["classification"]["method"] == "percentile"
        assert meta["classification"]["sell_threshold"] == 0.485
        assert meta["classification"]["buy_threshold"] == 0.516
        assert meta["classification"]["sell_percentile"] == 0.15
        assert meta["classification"]["hold_percentile"] == 0.40
