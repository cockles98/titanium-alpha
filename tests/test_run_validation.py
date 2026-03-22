"""Tests for ``src.backtest.run_validation``.

Validates the 3-tier CPCV-OOS validation pipeline: Trial construction,
parameter fingerprinting, dedup logic, tier builders, output generation,
and the legacy API.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.backtest.cpcv import TransactionCosts
from src.backtest.cpcv_oos import ValidationResult
from src.backtest.run_validation import (
    Trial,
    _dedup,
    _dyn_maxw,
    _hrp,
    _param_fp,
    _save_paths_parquet,
    _save_summary_md,
    _save_tier_json,
    _t,
    _trial_from_params,
    _trial_params,
    _wf,
    build_tier1,
    build_tier2,
    build_tier3,
    run_improvement_validation,
    run_tier,
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
            price *= 1 + ret
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
# Helpers
# ---------------------------------------------------------------------------


class TestDynMaxW:
    def test_small_n(self) -> None:
        assert _dyn_maxw(4) == 0.25  # min(0.25, 2/4) = 0.25

    def test_large_n(self) -> None:
        assert _dyn_maxw(50) == pytest.approx(0.04)  # 2/50

    def test_n_zero(self) -> None:
        """Handles n=0 gracefully."""
        assert _dyn_maxw(0) == 0.25  # min(0.25, 2/1)


class TestHRPHelper:
    def test_baseline_defaults(self) -> None:
        h = _hrp(50)
        assert h.linkage_method == "single"
        assert h.shrinkage is False
        assert h.max_weight == pytest.approx(0.04)

    def test_overrides(self) -> None:
        h = _hrp(50, linkage_method="ward", shrinkage=True)
        assert h.linkage_method == "ward"
        assert h.shrinkage is True


class TestWFHelper:
    def test_baseline_defaults(self) -> None:
        cfg = _wf()
        assert cfg.rebalance_every == 5
        assert cfg.retrain_every == 126
        assert cfg.lookback_days == 504
        assert cfg.rf == 0.05

    def test_overrides(self) -> None:
        cfg = _wf(rebalance_every=1, target_vol=0.12)
        assert cfg.rebalance_every == 1
        assert cfg.target_vol == 0.12
        assert cfg.lookback_days == 504  # unchanged


class TestTrialShorthand:
    def test_creates_trial(self) -> None:
        t = _t("test", mom=10, rebalance_every=3)
        assert t.name == "test"
        assert t.factory.lookback == 10
        assert t.wf_config.rebalance_every == 3


# ---------------------------------------------------------------------------
# Parameter fingerprinting & dedup
# ---------------------------------------------------------------------------


class TestTrialParams:
    def test_extracts_all_keys(self) -> None:
        trial = _t("test")
        params = _trial_params(trial, 50)
        expected_keys = {
            "momentum_lookback", "rebalance_every", "retrain_every",
            "lookback_days", "linkage_method", "shrinkage", "correlation_method",
            "confidence_tilt_cap", "max_weight", "min_weight", "turnover_threshold",
            "target_vol", "vol_lookback", "min_leverage", "max_leverage",
            "killswitch_max_dd", "killswitch_recovery", "killswitch_ramp",
            "min_rebalance_delta", "slippage_bps", "commission_bps",
            "market_impact_bps",
        }
        assert set(params.keys()) == expected_keys

    def test_includes_market_impact(self) -> None:
        """market_impact_bps is present in fingerprint (prevents dedup bugs)."""
        trial = _t("test")
        params = _trial_params(trial, 50)
        assert "market_impact_bps" in params
        assert params["market_impact_bps"] == 0.0

    def test_killswitch_params(self) -> None:
        trial = _t("ks", killswitch=KillswitchConfig(max_drawdown_pct=-0.15))
        params = _trial_params(trial, 50)
        assert params["killswitch_max_dd"] == -0.15
        assert params["killswitch_recovery"] is not None


class TestParamFingerprint:
    def test_deterministic(self) -> None:
        t1 = _t("a")
        t2 = _t("b")  # same params, different name
        fp1 = _param_fp(_trial_params(t1, 50))
        fp2 = _param_fp(_trial_params(t2, 50))
        assert fp1 == fp2

    def test_different_params(self) -> None:
        t1 = _t("a", mom=1)
        t2 = _t("b", mom=21)
        fp1 = _param_fp(_trial_params(t1, 50))
        fp2 = _param_fp(_trial_params(t2, 50))
        assert fp1 != fp2

    def test_market_impact_differentiates(self) -> None:
        """Configs with different market_impact_bps have different fingerprints."""
        p1 = _trial_params(_t("a"), 50)
        p2 = {**p1, "market_impact_bps": 5.0}
        assert _param_fp(p1) != _param_fp(p2)


class TestDedup:
    def test_removes_duplicates(self) -> None:
        trials = [_t("a"), _t("b")]  # same params
        unique = _dedup(trials, 50)
        assert len(unique) == 1

    def test_keeps_different(self) -> None:
        trials = [_t("a", mom=1), _t("b", mom=21)]
        unique = _dedup(trials, 50)
        assert len(unique) == 2

    def test_cross_tier_dedup(self) -> None:
        existing_fp = _param_fp(_trial_params(_t("old"), 50))
        trials = [_t("new")]  # same params as "old"
        unique = _dedup(trials, 50, existing_fps={existing_fp})
        assert len(unique) == 0


class TestTrialFromParams:
    def test_roundtrip(self) -> None:
        """Params extracted from a Trial can reconstruct an equivalent Trial."""
        original = _t("orig", mom=3, rebalance_every=10)
        params = _trial_params(original, 50)
        rebuilt = _trial_from_params("rebuilt", params, 50)
        rebuilt_params = _trial_params(rebuilt, 50)
        assert _param_fp(params) == _param_fp(rebuilt_params)

    def test_with_killswitch(self) -> None:
        original = _t("ks", killswitch=KillswitchConfig(max_drawdown_pct=-0.20))
        params = _trial_params(original, 50)
        rebuilt = _trial_from_params("rebuilt", params, 50)
        assert rebuilt.wf_config.killswitch is not None
        assert rebuilt.wf_config.killswitch.max_drawdown_pct == -0.20


# ---------------------------------------------------------------------------
# Tier builders
# ---------------------------------------------------------------------------


class TestBuildTier1:
    def test_has_baseline(self) -> None:
        trials = build_tier1(50)
        names = [t.name for t in trials]
        assert "baseline" in names

    def test_has_momentum_sweeps(self) -> None:
        trials = build_tier1(50)
        names = {t.name for t in trials}
        assert "s_mom001" in names
        assert "s_mom021" in names
        assert "s_mom126" in names

    def test_has_hrp_variants(self) -> None:
        trials = build_tier1(50)
        names = {t.name for t in trials}
        assert "s_ward" in names
        assert "s_shrink" in names
        assert "s_ward_shrink" in names

    def test_no_duplicates(self) -> None:
        trials = build_tier1(50)
        fps = [_param_fp(_trial_params(t, 50)) for t in trials]
        assert len(fps) == len(set(fps))

    def test_count_within_budget(self) -> None:
        """Tier 1 fits in 18h at ~375s/config (<=172 configs)."""
        trials = build_tier1(50)
        assert len(trials) <= 180  # small margin


class TestBuildTier2:
    def test_has_factorial_combos(self) -> None:
        trials = build_tier2(50)
        names = {t.name for t in trials}
        # 3-way mom x rebal x tilt (rebal shifted to sweet spot 5-15)
        assert "t2a_m01_r05_t020" in names

    def test_has_rebalance_fine_grain(self) -> None:
        trials = build_tier2(50)
        names = {t.name for t in trials}
        assert any("t2f_rb" in n for n in names)

    def test_no_duplicates(self) -> None:
        trials = build_tier2(50)
        fps = [_param_fp(_trial_params(t, 50)) for t in trials]
        assert len(fps) == len(set(fps))

    def test_god_bases_not_mutated(self) -> None:
        """god_bases dicts must not be mutated during tier build."""
        # Run twice — if mutation occurs, second run will differ
        t1 = build_tier2(50)
        t2 = build_tier2(50)
        assert len(t1) == len(t2)
        names1 = [t.name for t in t1]
        names2 = [t.name for t in t2]
        assert names1 == names2


class TestBuildTier3:
    def test_fallback_when_no_results(self) -> None:
        """Without prior results, falls back to predefined grid."""
        trials = build_tier3(50)
        assert len(trials) > 0


# ---------------------------------------------------------------------------
# Output savers
# ---------------------------------------------------------------------------


class TestSaveTierJson:
    def test_creates_file(self, tmp_path: Path) -> None:
        _save_tier_json(1, {"baseline": {"mean_sharpe": 0.5}}, 1, tmp_path)
        path = tmp_path / "validation_tier1_results.json"
        assert path.exists()

    def test_json_structure(self, tmp_path: Path) -> None:
        configs = {"baseline": {"mean_sharpe": 0.5, "accepted": True}}
        _save_tier_json(1, configs, 1, tmp_path)
        data = json.loads((tmp_path / "validation_tier1_results.json").read_text())
        assert data["tier"] == 1
        assert data["completed"] == 1
        assert "baseline" in data["configs"]

    def test_atomic_write(self, tmp_path: Path) -> None:
        """No .tmp file should remain after save."""
        _save_tier_json(1, {}, 0, tmp_path)
        assert not (tmp_path / "validation_tier1_results.tmp").exists()


class TestSaveSummaryMd:
    def test_creates_file(self, tmp_path: Path) -> None:
        configs = {"baseline": {"mean_sharpe": 0.5, "std_sharpe": 0.1,
                                "pct_positive": 0.8, "deflated_sharpe": 0.9, "accepted": True}}
        _save_summary_md(1, configs, tmp_path)
        path = tmp_path / "validation_tier1_summary.md"
        assert path.exists()

    def test_contains_rankings(self, tmp_path: Path) -> None:
        configs = {"baseline": {"mean_sharpe": 0.5, "std_sharpe": 0.1,
                                "pct_positive": 0.8, "deflated_sharpe": 0.9, "accepted": True}}
        _save_summary_md(1, configs, tmp_path)
        content = (tmp_path / "validation_tier1_summary.md").read_text()
        assert "baseline" in content
        assert "YES" in content


class TestSavePathsParquet:
    def test_creates_file(self, tmp_path: Path) -> None:
        configs = {"baseline": {"per_path_sharpe": [0.5, 0.6, 0.3], "accepted": True}}
        _save_paths_parquet(1, configs, tmp_path)
        path = tmp_path / "validation_tier1_paths.parquet"
        assert path.exists()

    def test_parquet_structure(self, tmp_path: Path) -> None:
        configs = {"baseline": {"per_path_sharpe": [0.5, 0.6], "accepted": True}}
        _save_paths_parquet(1, configs, tmp_path)
        df = pl.read_parquet(tmp_path / "validation_tier1_paths.parquet")
        assert "config" in df.columns
        assert "path_id" in df.columns
        assert "sharpe" in df.columns
        assert df.height == 2


# ---------------------------------------------------------------------------
# WalkForwardConfig.hrp_config integration
# ---------------------------------------------------------------------------


class TestHRPConfigInWalkForward:
    def test_default_none(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.hrp_config is None

    def test_custom_hrp_config(self) -> None:
        hrp = HRPConfig(linkage_method="ward", shrinkage=True)
        cfg = WalkForwardConfig(hrp_config=hrp)
        assert cfg.hrp_config is not None
        assert cfg.hrp_config.linkage_method == "ward"
        assert cfg.hrp_config.shrinkage is True


# ---------------------------------------------------------------------------
# Integration: run_tier with mocked validator
# ---------------------------------------------------------------------------


class TestRunTier:
    def _mock_validator(self) -> MagicMock:
        validator = MagicMock(spec=["validate"])
        validator.validate.return_value = ValidationResult(
            config=_wf(),
            mean_sharpe=0.5,
            std_sharpe=0.1,
            pct_positive=0.8,
            per_path_sharpe=[0.5, 0.6, -0.1, 0.4, 0.3],
            deflated_sharpe=0.97,
            p_value=0.97,
            accepted=True,
        )
        return validator

    def test_runs_all_trials(self, tmp_path: Path) -> None:
        trials = [_t("a", mom=1), _t("b", mom=21)]
        validator = self._mock_validator()
        results = run_tier(1, trials, validator, 50, tmp_path)
        assert len(results) == 2
        assert "a" in results
        assert "b" in results
        assert validator.validate.call_count == 2

    def test_resume_skips_existing(self, tmp_path: Path) -> None:
        """Pre-existing results are not re-run."""
        # Pre-populate results for trial "a"
        _save_tier_json(1, {"a": {"params": _trial_params(_t("a", mom=1), 50),
                                  "mean_sharpe": 0.5}}, 2, tmp_path)
        trials = [_t("a", mom=1), _t("b", mom=21)]
        validator = self._mock_validator()
        results = run_tier(1, trials, validator, 50, tmp_path)
        # Only "b" should be newly validated
        assert validator.validate.call_count == 1

    def test_creates_output_files(self, tmp_path: Path) -> None:
        trials = [_t("test")]
        validator = self._mock_validator()
        run_tier(1, trials, validator, 50, tmp_path)
        assert (tmp_path / "validation_tier1_results.json").exists()
        assert (tmp_path / "validation_tier1_summary.md").exists()
        assert (tmp_path / "validation_tier1_paths.parquet").exists()


# ---------------------------------------------------------------------------
# Legacy API
# ---------------------------------------------------------------------------


class TestRunImprovementValidation:
    @patch("src.backtest.run_validation.CPCVParameterValidator")
    def test_returns_dict(
        self, mock_cls: MagicMock, ohlcv: pl.DataFrame, tmp_path: Path
    ) -> None:
        mock_validator = mock_cls.return_value
        mock_validator.validate.return_value = ValidationResult(
            config=_wf(),
            mean_sharpe=0.5,
            std_sharpe=0.1,
            pct_positive=0.8,
            per_path_sharpe=[0.5] * 15,
            deflated_sharpe=0.9,
            p_value=0.9,
            accepted=True,
        )
        results = run_improvement_validation(
            ohlcv=ohlcv,
            tickers=["AAPL", "MSFT", "GOOG"],
            output_dir=str(tmp_path),
        )
        assert isinstance(results, dict)
