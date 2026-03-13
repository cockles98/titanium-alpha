"""Tests for ``src.backtest.run_validation``.

Validates the CPCV-OOS validation pipeline: config grid building,
momentum factory variants, output generation, and end-to-end integration.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.backtest.cpcv import TransactionCosts
from src.backtest.cpcv_oos import ValidationResult
from src.backtest.run_validation import (
    _base_config,
    _save_per_path_parquet,
    _save_results_json,
    _save_summary_md,
    build_all_configs,
    build_momentum_factories,
    build_tier1_configs,
    build_tier2_configs,
    run_improvement_validation,
)
from src.backtest.walk_forward import (
    KillswitchConfig,
    NaiveModelFactory,
    WalkForwardConfig,
)
from src.portfolio.hrp import HRPConfig


# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------


def _make_ohlcv(
    tickers: list[str],
    n_days: int = 800,
    start: date = date(2016, 1, 4),
    base_price: float = 100.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic OHLCV for validation testing."""
    import random

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for t_idx, ticker in enumerate(tickers):
        price = base_price + t_idx * 10
        for day in range(n_days):
            d = start + timedelta(days=day)
            vol = 0.005 + t_idx * 0.002
            ret = 0.0003 + rng.gauss(0, vol)
            price *= (1 + ret)
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": price * 0.999,
                "high": price * 1.002,
                "low": price * 0.998,
                "close": price,
                "volume": 1_000_000,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture()
def ohlcv() -> pl.DataFrame:
    """3 tickers + SPY, 800 days."""
    return _make_ohlcv(["AAPL", "MSFT", "GOOG", "SPY"])


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


class TestBaseConfig:
    """Tests for _base_config helper."""

    def test_default_values(self) -> None:
        """Baseline config has expected defaults."""
        cfg = _base_config()
        assert cfg.rebalance_every == 5
        assert cfg.retrain_every == 126
        assert cfg.lookback_days == 504
        assert cfg.costs is not None
        assert cfg.min_rebalance_delta == 0.02
        assert cfg.rf == 0.05

    def test_overrides(self) -> None:
        """Overrides are applied correctly."""
        cfg = _base_config(rebalance_every=10, target_vol=0.10)
        assert cfg.rebalance_every == 10
        assert cfg.target_vol == 0.10
        assert cfg.lookback_days == 504  # unchanged


class TestBuildTier1:
    """Tests for build_tier1_configs."""

    def test_has_baseline(self) -> None:
        configs = build_tier1_configs()
        assert "baseline" in configs

    def test_has_vol_targets(self) -> None:
        configs = build_tier1_configs()
        assert "vol_target_08" in configs
        assert "vol_target_10" in configs
        assert "vol_target_12" in configs
        assert configs["vol_target_10"].target_vol == 0.10

    def test_has_hrp_variants(self) -> None:
        configs = build_tier1_configs()
        assert "ward_linkage" in configs
        assert "shrinkage" in configs
        assert "ward_shrinkage" in configs

    def test_ward_linkage_config(self) -> None:
        configs = build_tier1_configs(n_tickers=50)
        hrp = configs["ward_linkage"].hrp_config
        assert hrp is not None
        assert hrp.linkage_method == "ward"
        assert hrp.max_weight == pytest.approx(0.04)  # min(0.25, 2/50)

    def test_shrinkage_config(self) -> None:
        configs = build_tier1_configs(n_tickers=50)
        hrp = configs["shrinkage"].hrp_config
        assert hrp is not None
        assert hrp.shrinkage is True
        assert hrp.max_weight == pytest.approx(0.04)

    def test_ward_shrinkage_config(self) -> None:
        configs = build_tier1_configs(n_tickers=50)
        hrp = configs["ward_shrinkage"].hrp_config
        assert hrp is not None
        assert hrp.linkage_method == "ward"
        assert hrp.shrinkage is True
        assert hrp.max_weight == pytest.approx(0.04)

    def test_hrp_max_weight_matches_dynamic_default(self) -> None:
        """HRP configs use the same dynamic max_weight as baseline."""
        configs = build_tier1_configs(n_tickers=4)
        hrp = configs["ward_linkage"].hrp_config
        assert hrp is not None
        assert hrp.max_weight == pytest.approx(0.25)  # min(0.25, 2/4) = 0.25

    def test_count(self) -> None:
        """Tier 1 has expected number of configs."""
        configs = build_tier1_configs()
        assert len(configs) == 7  # baseline + 3 vol + 3 hrp


class TestBuildTier2:
    """Tests for build_tier2_configs."""

    def test_has_rebalance_variants(self) -> None:
        configs = build_tier2_configs()
        assert "rebalance_10d" in configs
        assert "rebalance_21d" in configs

    def test_has_tilt_variants(self) -> None:
        configs = build_tier2_configs()
        assert "no_tilt" in configs
        assert "tilt_010" in configs
        assert "tilt_030" in configs

    def test_has_lookback_variants(self) -> None:
        configs = build_tier2_configs()
        assert "lookback_252" in configs
        assert "lookback_756" in configs

    def test_has_killswitch_variants(self) -> None:
        configs = build_tier2_configs()
        assert "killswitch_15" in configs
        assert "killswitch_20" in configs
        assert "killswitch_25" in configs

    def test_killswitch_config(self) -> None:
        configs = build_tier2_configs()
        ks = configs["killswitch_15"].killswitch
        assert ks is not None
        assert ks.max_drawdown_pct == -0.15

    def test_count(self) -> None:
        """Tier 2 has expected number of configs."""
        configs = build_tier2_configs()
        assert len(configs) == 10  # 2 rebal + 3 tilt + 2 lookback + 3 ks


class TestBuildAllConfigs:
    """Tests for build_all_configs."""

    def test_all_includes_all_tiers(self) -> None:
        configs = build_all_configs("all")
        assert "baseline" in configs  # tier1
        assert "rebalance_10d" in configs  # tier2
        assert "rebalance_1d" in configs  # tier3
        assert "t4_vol_lookback_21" in configs  # tier4
        assert len(configs) == 39  # 7 + 10 + 13 + 9

    def test_tier1_only(self) -> None:
        configs = build_all_configs("tier1")
        assert "baseline" in configs
        assert "rebalance_10d" not in configs
        assert len(configs) == 7

    def test_tier2_only(self) -> None:
        configs = build_all_configs("tier2")
        assert "baseline" in configs  # auto-added if not present
        assert "rebalance_10d" in configs
        assert len(configs) == 11  # 10 tier2 + 1 baseline


class TestBuildMomentumFactories:
    """Tests for build_momentum_factories."""

    def test_count(self) -> None:
        factories = build_momentum_factories()
        assert len(factories) == 8

    def test_lookback_values(self) -> None:
        factories = build_momentum_factories()
        assert factories["momentum_1d"].lookback == 1
        assert factories["momentum_5d"].lookback == 5
        assert factories["momentum_21d"].lookback == 21
        assert factories["momentum_63d"].lookback == 63
        assert factories["momentum_126d"].lookback == 126

    def test_all_are_naive_factory(self) -> None:
        factories = build_momentum_factories()
        for f in factories.values():
            assert isinstance(f, NaiveModelFactory)


# ---------------------------------------------------------------------------
# Output savers
# ---------------------------------------------------------------------------


def _make_validation_result(
    mean_sharpe: float = 0.5,
    accepted: bool = True,
) -> ValidationResult:
    """Create a mock ValidationResult for output tests."""
    return ValidationResult(
        config=_base_config(),
        mean_sharpe=mean_sharpe,
        std_sharpe=0.1,
        pct_positive=0.8,
        per_path_sharpe=[0.5, 0.6, -0.1, 0.4, 0.3],
        deflated_sharpe=0.97,
        p_value=0.97,
        accepted=accepted,
    )


class TestSaveResultsJson:
    """Tests for _save_results_json."""

    def test_creates_file(self, tmp_path: Path) -> None:
        results = [("baseline", _make_validation_result())]
        path = _save_results_json(results, tmp_path)
        assert path.exists()
        assert path.name == "validation_results.json"

    def test_json_structure(self, tmp_path: Path) -> None:
        results = [
            ("baseline", _make_validation_result(0.5)),
            ("ward", _make_validation_result(0.6)),
        ]
        path = _save_results_json(results, tmp_path)
        data = json.loads(path.read_text())
        assert data["n_configs"] == 2
        assert "baseline" in data["configs"]
        assert "ward" in data["configs"]
        assert "generated_at" in data

    def test_result_fields(self, tmp_path: Path) -> None:
        results = [("test", _make_validation_result(0.42))]
        path = _save_results_json(results, tmp_path)
        data = json.loads(path.read_text())
        cfg = data["configs"]["test"]
        assert cfg["mean_sharpe"] == pytest.approx(0.42)
        assert "per_path_sharpe" in cfg
        assert "accepted" in cfg


class TestSaveSummaryMd:
    """Tests for _save_summary_md."""

    def test_creates_file(self, tmp_path: Path) -> None:
        results = [("baseline", _make_validation_result())]
        path = _save_summary_md(results, None, tmp_path)
        assert path.exists()
        assert path.name == "validation_summary.md"

    def test_contains_table_headers(self, tmp_path: Path) -> None:
        results = [("baseline", _make_validation_result())]
        path = _save_summary_md(results, None, tmp_path)
        content = path.read_text()
        assert "Config" in content
        assert "Mean Sharpe" in content
        assert "DSR p-value" in content

    def test_contains_config_names(self, tmp_path: Path) -> None:
        results = [
            ("baseline", _make_validation_result()),
            ("ward_shrinkage", _make_validation_result()),
        ]
        path = _save_summary_md(results, None, tmp_path)
        content = path.read_text()
        assert "baseline" in content
        assert "ward_shrinkage" in content

    def test_accepted_formatting(self, tmp_path: Path) -> None:
        results = [
            ("accepted_cfg", _make_validation_result(accepted=True)),
            ("rejected_cfg", _make_validation_result(accepted=False)),
        ]
        path = _save_summary_md(results, None, tmp_path)
        content = path.read_text()
        assert "YES" in content
        assert "| no |" in content

    def test_momentum_section(self, tmp_path: Path) -> None:
        config_results = [("baseline", _make_validation_result())]
        mom_results = [("momentum_21d", _make_validation_result())]
        path = _save_summary_md(config_results, mom_results, tmp_path)
        content = path.read_text()
        assert "Momentum Lookback" in content
        assert "momentum_21d" in content


class TestSavePerPathParquet:
    """Tests for _save_per_path_parquet."""

    def test_creates_file(self, tmp_path: Path) -> None:
        results = [("baseline", _make_validation_result())]
        path = _save_per_path_parquet(results, None, tmp_path)
        assert path.exists()
        assert path.name == "validation_per_path.parquet"

    def test_parquet_structure(self, tmp_path: Path) -> None:
        results = [("baseline", _make_validation_result())]
        path = _save_per_path_parquet(results, None, tmp_path)
        df = pl.read_parquet(path)
        assert "config" in df.columns
        assert "path_id" in df.columns
        assert "sharpe" in df.columns
        assert "accepted" in df.columns

    def test_row_count(self, tmp_path: Path) -> None:
        """Each config contributes len(per_path_sharpe) rows."""
        r1 = _make_validation_result()
        r1_paths = len(r1.per_path_sharpe)
        results = [("a", r1), ("b", _make_validation_result())]
        path = _save_per_path_parquet(results, None, tmp_path)
        df = pl.read_parquet(path)
        assert df.height == r1_paths * 2

    def test_includes_momentum(self, tmp_path: Path) -> None:
        config_results = [("baseline", _make_validation_result())]
        mom_results = [("momentum_21d", _make_validation_result())]
        path = _save_per_path_parquet(config_results, mom_results, tmp_path)
        df = pl.read_parquet(path)
        configs = df["config"].unique().to_list()
        assert "baseline" in configs
        assert "momentum_21d" in configs


# ---------------------------------------------------------------------------
# WalkForwardConfig.hrp_config integration
# ---------------------------------------------------------------------------


class TestHRPConfigInWalkForward:
    """Tests that hrp_config field works in WalkForwardConfig."""

    def test_default_none(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.hrp_config is None

    def test_custom_hrp_config(self) -> None:
        hrp = HRPConfig(linkage_method="ward", shrinkage=True)
        cfg = WalkForwardConfig(hrp_config=hrp)
        assert cfg.hrp_config is not None
        assert cfg.hrp_config.linkage_method == "ward"
        assert cfg.hrp_config.shrinkage is True

    def test_backward_compat(self) -> None:
        """Existing code using WalkForwardConfig() still works."""
        cfg = WalkForwardConfig(rebalance_every=10)
        assert cfg.rebalance_every == 10
        assert cfg.hrp_config is None


# ---------------------------------------------------------------------------
# Integration test (mocked validator)
# ---------------------------------------------------------------------------


class TestRunImprovementValidation:
    """Integration tests for run_improvement_validation."""

    def _make_mock_result(
        self,
        config: WalkForwardConfig,
        mean_sharpe: float = 0.3,
    ) -> ValidationResult:
        """Create a ValidationResult for mocking."""
        return ValidationResult(
            config=config,
            mean_sharpe=mean_sharpe,
            std_sharpe=0.15,
            pct_positive=0.73,
            per_path_sharpe=[mean_sharpe + i * 0.01 for i in range(15)],
            deflated_sharpe=0.80,
            p_value=0.80,
            accepted=False,
        )

    @patch("src.backtest.run_validation.CPCVParameterValidator")
    def test_returns_dict(
        self, mock_validator_cls: MagicMock, ohlcv: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Pipeline returns dict of name → ValidationResult."""
        mock_validator = mock_validator_cls.return_value
        mock_validator.grid_search.return_value = [
            ("baseline", self._make_mock_result(_base_config())),
        ]
        mock_validator.validate.return_value = self._make_mock_result(
            _base_config()
        )

        results = run_improvement_validation(
            ohlcv=ohlcv,
            tickers=["AAPL", "MSFT", "GOOG"],
            output_dir=str(tmp_path),
            subset="tier1",
        )
        assert isinstance(results, dict)
        assert "baseline" in results

    @patch("src.backtest.run_validation.CPCVParameterValidator")
    def test_creates_output_files(
        self, mock_validator_cls: MagicMock, ohlcv: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Pipeline creates all 3 output files."""
        mock_validator = mock_validator_cls.return_value
        mock_validator.grid_search.return_value = [
            ("baseline", self._make_mock_result(_base_config())),
        ]
        mock_validator.validate.return_value = self._make_mock_result(
            _base_config()
        )

        run_improvement_validation(
            ohlcv=ohlcv,
            tickers=["AAPL", "MSFT", "GOOG"],
            output_dir=str(tmp_path),
            subset="tier1",
        )

        assert (tmp_path / "validation_results.json").exists()
        assert (tmp_path / "validation_summary.md").exists()
        assert (tmp_path / "validation_per_path.parquet").exists()

    @patch("src.backtest.run_validation.CPCVParameterValidator")
    def test_grid_search_called_with_configs(
        self, mock_validator_cls: MagicMock, ohlcv: pl.DataFrame, tmp_path: Path
    ) -> None:
        """grid_search is called with the built configs."""
        mock_validator = mock_validator_cls.return_value
        mock_validator.grid_search.return_value = []
        mock_validator.validate.return_value = self._make_mock_result(
            _base_config()
        )

        run_improvement_validation(
            ohlcv=ohlcv,
            tickers=["AAPL", "MSFT", "GOOG"],
            output_dir=str(tmp_path),
            subset="tier1",
        )

        mock_validator.grid_search.assert_called_once()
        call_args = mock_validator.grid_search.call_args
        configs = call_args.kwargs.get("configs") or call_args[1].get("configs")
        assert "baseline" in configs

    @patch("src.backtest.run_validation.CPCVParameterValidator")
    def test_momentum_factories_validated(
        self, mock_validator_cls: MagicMock, ohlcv: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Momentum factories are validated via individual validate() calls."""
        mock_validator = mock_validator_cls.return_value
        mock_validator.grid_search.return_value = []
        mock_validator.validate.return_value = self._make_mock_result(
            _base_config()
        )

        run_improvement_validation(
            ohlcv=ohlcv,
            tickers=["AAPL", "MSFT", "GOOG"],
            output_dir=str(tmp_path),
            subset="tier1",
        )

        # 8 momentum factories → 8 validate() calls
        assert mock_validator.validate.call_count == 8

    @patch("src.backtest.run_validation.CPCVParameterValidator")
    def test_includes_momentum_in_results(
        self, mock_validator_cls: MagicMock, ohlcv: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Momentum results are included in the final output."""
        mock_validator = mock_validator_cls.return_value
        mock_validator.grid_search.return_value = [
            ("baseline", self._make_mock_result(_base_config())),
        ]
        mock_validator.validate.return_value = self._make_mock_result(
            _base_config()
        )

        results = run_improvement_validation(
            ohlcv=ohlcv,
            tickers=["AAPL", "MSFT", "GOOG"],
            output_dir=str(tmp_path),
            subset="tier1",
        )

        # Should include both config grid and momentum factories
        assert "momentum_5d" in results
        assert "momentum_21d" in results
        assert "momentum_63d" in results
        assert "momentum_126d" in results

    @patch("src.backtest.run_validation.CPCVParameterValidator")
    def test_n_trials_consistent_across_grid_and_momentum(
        self, mock_validator_cls: MagicMock, ohlcv: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Both grid_search and momentum validate() use the same n_trials."""
        mock_validator = mock_validator_cls.return_value
        mock_validator.grid_search.return_value = []
        mock_validator.validate.return_value = self._make_mock_result(
            _base_config()
        )

        run_improvement_validation(
            ohlcv=ohlcv,
            tickers=["AAPL", "MSFT", "GOOG"],
            output_dir=str(tmp_path),
            subset="tier1",
        )

        # n_trials = len(tier1_configs) + len(factories) + len(god_combos)
        # For tier1: 7 + 8 + 0 = 15
        grid_call = mock_validator.grid_search.call_args
        grid_n_trials = grid_call.kwargs.get("n_trials")
        assert grid_n_trials == 15

        validate_call = mock_validator.validate.call_args
        validate_n_trials = validate_call.kwargs.get("n_trials")
        assert validate_n_trials == 15
