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
    HoldoutResult,
    Trial,
    _dedup,
    _dyn_maxw,
    _holdout_cutoff,
    _hrp,
    _param_fp,
    _save_holdout_results,
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
    filter_before_holdout,
    identify_champion,
    prepare_holdout_ohlcv,
    run_holdout_validation,
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
        assert h.linkage_method == "ward"
        assert h.shrinkage is True
        assert h.max_weight == pytest.approx(0.04)

    def test_overrides(self) -> None:
        h = _hrp(50, linkage_method="single", shrinkage=False)
        assert h.linkage_method == "single"
        assert h.shrinkage is False


class TestWFHelper:
    def test_baseline_defaults(self) -> None:
        cfg = _wf()
        assert cfg.rebalance_every == 13
        assert cfg.retrain_every == 126
        assert cfg.lookback_days == 756
        assert cfg.rf == 0.05

    def test_overrides(self) -> None:
        cfg = _wf(rebalance_every=1, target_vol=0.12)
        assert cfg.rebalance_every == 1
        assert cfg.target_vol == 0.12
        assert cfg.lookback_days == 756  # unchanged


class TestTrialShorthand:
    def test_creates_trial(self) -> None:
        t = _t(50, "test", rebalance_every=3)
        assert t.name == "test"
        assert t.factory.lookback == 5  # fixed momentum (PatchTST-safe)
        assert t.wf_config.rebalance_every == 3

    def test_default_hrp_is_ward_shrink(self) -> None:
        t = _t(50, "test")
        assert t.wf_config.hrp_config is not None
        assert t.wf_config.hrp_config.linkage_method == "ward"
        assert t.wf_config.hrp_config.shrinkage is True


# ---------------------------------------------------------------------------
# Parameter fingerprinting & dedup
# ---------------------------------------------------------------------------


class TestTrialParams:
    def test_extracts_all_keys(self) -> None:
        trial = _t(50, "test")
        params = _trial_params(trial, 50)
        expected_keys = {
            "momentum_lookback", "rebalance_every", "retrain_every",
            "lookback_days", "linkage_method", "shrinkage", "correlation_method",
            "confidence_tilt_cap", "max_weight", "min_weight", "turnover_threshold",
            "target_vol", "vol_lookback", "min_leverage", "max_leverage",
            "killswitch_max_dd", "killswitch_recovery", "killswitch_ramp",
            "min_rebalance_delta", "slippage_bps", "commission_bps",
            "market_impact_bps", "top_n",
        }
        assert set(params.keys()) == expected_keys

    def test_includes_market_impact(self) -> None:
        """market_impact_bps is present in fingerprint (prevents dedup bugs)."""
        trial = _t(50, "test")
        params = _trial_params(trial, 50)
        assert "market_impact_bps" in params
        assert params["market_impact_bps"] == 0.0

    def test_killswitch_params(self) -> None:
        trial = _t(50, "ks", killswitch=KillswitchConfig(max_drawdown_pct=-0.15))
        params = _trial_params(trial, 50)
        assert params["killswitch_max_dd"] == -0.15
        assert params["killswitch_recovery"] is not None


class TestParamFingerprint:
    def test_deterministic(self) -> None:
        t1 = _t(50, "a")
        t2 = _t(50, "b")  # same params, different name
        fp1 = _param_fp(_trial_params(t1, 50))
        fp2 = _param_fp(_trial_params(t2, 50))
        assert fp1 == fp2

    def test_different_params(self) -> None:
        t1 = _t(50, "a", rebalance_every=1)
        t2 = _t(50, "b", rebalance_every=21)
        fp1 = _param_fp(_trial_params(t1, 50))
        fp2 = _param_fp(_trial_params(t2, 50))
        assert fp1 != fp2

    def test_market_impact_differentiates(self) -> None:
        """Configs with different market_impact_bps have different fingerprints."""
        p1 = _trial_params(_t(50, "a"), 50)
        p2 = {**p1, "market_impact_bps": 5.0}
        assert _param_fp(p1) != _param_fp(p2)


class TestDedup:
    def test_removes_duplicates(self) -> None:
        trials = [_t(50, "a"), _t(50, "b")]  # same params
        unique = _dedup(trials, 50)
        assert len(unique) == 1

    def test_keeps_different(self) -> None:
        trials = [_t(50, "a", rebalance_every=1), _t(50, "b", rebalance_every=21)]
        unique = _dedup(trials, 50)
        assert len(unique) == 2

    def test_cross_tier_dedup(self) -> None:
        existing_fp = _param_fp(_trial_params(_t(50, "old"), 50))
        trials = [_t(50, "new")]  # same params as "old"
        unique = _dedup(trials, 50, existing_fps={existing_fp})
        assert len(unique) == 0


class TestTrialFromParams:
    def test_roundtrip(self) -> None:
        """Params extracted from a Trial can reconstruct an equivalent Trial."""
        original = _t(50, "orig", rebalance_every=10)
        params = _trial_params(original, 50)
        rebuilt = _trial_from_params("rebuilt", params, 50)
        rebuilt_params = _trial_params(rebuilt, 50)
        assert _param_fp(params) == _param_fp(rebuilt_params)

    def test_with_killswitch(self) -> None:
        original = _t(50, "ks", killswitch=KillswitchConfig(max_drawdown_pct=-0.20))
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

    def test_has_topn_sweeps(self) -> None:
        trials = build_tier1(50)
        names = {t.name for t in trials}
        assert "s_topn010" in names
        assert "s_topn020" in names
        assert "s_topn040" in names

    def test_has_hrp_variants(self) -> None:
        trials = build_tier1(50)
        names = {t.name for t in trials}
        assert "s_single" in names
        assert "s_noshrink" in names
        assert "s_single_noshrink" in names

    def test_no_patchtst_unsafe_params(self) -> None:
        """No config should vary retrain_every, lookback_days, or momentum."""
        trials = build_tier1(50)
        for t in trials:
            assert t.wf_config.retrain_every == 126, f"{t.name} varies retrain_every"
            assert t.wf_config.lookback_days == 756, f"{t.name} varies lookback_days"
            assert t.factory.lookback == 5, f"{t.name} varies momentum"

    def test_no_duplicates(self) -> None:
        trials = build_tier1(50)
        fps = [_param_fp(_trial_params(t, 50)) for t in trials]
        assert len(fps) == len(set(fps))

    def test_count_within_budget(self) -> None:
        """Tier 1 fits in 18h at ~375s/config (<=172 configs)."""
        trials = build_tier1(50)
        assert len(trials) <= 180  # small margin


class TestBuildTier2:
    def test_has_hrp_structure_combos(self) -> None:
        trials = build_tier2(50)
        names = {t.name for t in trials}
        # HRP structure factorial at winner rb
        assert "t2b_SY_p_r10" in names

    def test_has_rebalance_fine_grain(self) -> None:
        trials = build_tier2(50)
        names = {t.name for t in trials}
        assert any("t2a_rb" in n for n in names)

    def test_no_topn_in_tier2(self) -> None:
        """Tier 1 showed top_n hurts — Tier 2 should not use it."""
        trials = build_tier2(50)
        for t in trials:
            assert t.wf_config.top_n is None, f"{t.name} uses top_n={t.wf_config.top_n}"

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
        trials = [_t(50, "a", rebalance_every=1), _t(50, "b", rebalance_every=21)]
        validator = self._mock_validator()
        results = run_tier(1, trials, validator, 50, tmp_path)
        assert len(results) == 2
        assert "a" in results
        assert "b" in results
        assert validator.validate.call_count == 2

    def test_resume_skips_existing(self, tmp_path: Path) -> None:
        """Pre-existing results are not re-run."""
        # Pre-populate results for trial "a"
        _save_tier_json(1, {"a": {"params": _trial_params(_t(50, "a", rebalance_every=1), 50),
                                  "mean_sharpe": 0.5}}, 2, tmp_path)
        trials = [_t(50, "a", rebalance_every=1), _t(50, "b", rebalance_every=21)]
        validator = self._mock_validator()
        results = run_tier(1, trials, validator, 50, tmp_path)
        # Only "b" should be newly validated
        assert validator.validate.call_count == 1

    def test_creates_output_files(self, tmp_path: Path) -> None:
        trials = [_t(50, "test")]
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


# ---------------------------------------------------------------------------
# Holdout validation
# ---------------------------------------------------------------------------


def _make_long_ohlcv(
    tickers: list[str],
    n_days: int = 2600,
    start: date = date(2014, 1, 2),
) -> pl.DataFrame:
    """Generate synthetic OHLCV covering ~10 years for holdout tests."""
    return _make_ohlcv(tickers, n_days=n_days, start=start)


class TestHoldoutCutoff:
    def test_basic(self) -> None:
        df = pl.DataFrame({"date": [date(2016, 1, 4), date(2026, 3, 15)]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        cutoff = _holdout_cutoff(df, holdout_years=2)
        assert cutoff == date(2024, 3, 15)

    def test_leap_year(self) -> None:
        """Feb 29 in a leap year falls back to Feb 28."""
        df = pl.DataFrame({"date": [date(2014, 1, 1), date(2024, 2, 29)]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        cutoff = _holdout_cutoff(df, holdout_years=1)
        assert cutoff == date(2023, 2, 28)

    def test_one_year(self) -> None:
        df = pl.DataFrame({"date": [date(2016, 1, 1), date(2026, 6, 1)]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        assert _holdout_cutoff(df, holdout_years=1) == date(2025, 6, 1)

    def test_holdout_exceeds_data_span(self) -> None:
        """Holdout larger than data span raises ValueError."""
        df = pl.DataFrame({
            "date": [date(2024, 1, 1), date(2026, 1, 1)]
        }).with_columns(pl.col("date").cast(pl.Date))
        with pytest.raises(ValueError, match="exceeds data span"):
            _holdout_cutoff(df, holdout_years=5)

    def test_holdout_years_zero_raises(self) -> None:
        df = pl.DataFrame({"date": [date(2026, 1, 1)]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        with pytest.raises(ValueError, match="must be >= 1"):
            _holdout_cutoff(df, holdout_years=0)


class TestFilterBeforeHoldout:
    def test_removes_holdout_period(self) -> None:
        ohlcv = _make_long_ohlcv(["AAPL", "SPY"], n_days=2600)
        filtered = filter_before_holdout(ohlcv, holdout_years=2)
        cutoff = _holdout_cutoff(ohlcv, holdout_years=2)
        assert filtered["date"].max() < cutoff

    def test_preserves_earlier_data(self) -> None:
        ohlcv = _make_long_ohlcv(["AAPL", "SPY"], n_days=2600)
        filtered = filter_before_holdout(ohlcv, holdout_years=2)
        assert filtered.height > 0
        assert filtered["date"].min() == ohlcv["date"].min()


class TestPrepareHoldoutOhlcv:
    def test_includes_lookback_buffer(self) -> None:
        ohlcv = _make_long_ohlcv(["AAPL", "SPY"], n_days=2600)
        holdout_df, holdout_start = prepare_holdout_ohlcv(
            ohlcv, holdout_years=2, lookback_days=504,
        )
        # Data should start well before holdout
        assert holdout_df["date"].min() < holdout_start

    def test_returns_correct_holdout_start(self) -> None:
        ohlcv = _make_long_ohlcv(["AAPL", "SPY"], n_days=2600)
        _, holdout_start = prepare_holdout_ohlcv(
            ohlcv, holdout_years=2, lookback_days=504,
        )
        cutoff = _holdout_cutoff(ohlcv, holdout_years=2)
        assert holdout_start == cutoff

    def test_larger_lookback_starts_earlier(self) -> None:
        ohlcv = _make_long_ohlcv(["AAPL", "SPY"], n_days=2600)
        df_small, _ = prepare_holdout_ohlcv(ohlcv, holdout_years=2, lookback_days=252)
        df_large, _ = prepare_holdout_ohlcv(ohlcv, holdout_years=2, lookback_days=756)
        assert df_large["date"].min() <= df_small["date"].min()


class TestIdentifyChampion:
    def test_picks_best_by_sharpe(self, tmp_path: Path) -> None:
        tier_data = {
            "configs": {
                "config_a": {
                    "mean_sharpe": 0.6,
                    "deflated_sharpe": 0.05,
                    "params": {"rebalance_every": 5, "lookback_days": 504},
                },
                "config_b": {
                    "mean_sharpe": 0.8,
                    "deflated_sharpe": 0.08,
                    "params": {"rebalance_every": 13, "lookback_days": 756},
                },
            }
        }
        (tmp_path / "validation_tier1_results.json").write_text(
            json.dumps(tier_data), encoding="utf-8"
        )
        name, params, sharpe, dsr = identify_champion(tmp_path)
        assert name == "config_b"
        assert sharpe == pytest.approx(0.8)
        assert params["rebalance_every"] == 13

    def test_picks_across_multiple_tiers(self, tmp_path: Path) -> None:
        """Champion selection works across tier1 + tier2 results."""
        tier1 = {"configs": {"t1_a": {
            "mean_sharpe": 0.6, "deflated_sharpe": 0.05,
            "params": {"rebalance_every": 5, "lookback_days": 504},
        }}}
        tier2 = {"configs": {"t2_b": {
            "mean_sharpe": 0.9, "deflated_sharpe": 0.09,
            "params": {"rebalance_every": 13, "lookback_days": 756},
        }}}
        (tmp_path / "validation_tier1_results.json").write_text(
            json.dumps(tier1), encoding="utf-8"
        )
        (tmp_path / "validation_tier2_results.json").write_text(
            json.dumps(tier2), encoding="utf-8"
        )
        name, _, sharpe, _ = identify_champion(tmp_path)
        assert name == "t2_b"
        assert sharpe == pytest.approx(0.9)

    def test_empty_results_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="No tier results"):
            identify_champion(tmp_path)


class TestRunHoldoutValidation:
    @patch("src.backtest.run_validation.WalkForwardBacktester")
    @patch("src.backtest.run_validation.identify_champion")
    def test_returns_holdout_result(
        self, mock_champion: MagicMock, mock_bt_cls: MagicMock, tmp_path: Path,
    ) -> None:
        ohlcv = _make_long_ohlcv(["AAPL", "MSFT", "SPY"], n_days=2600)

        # Mock champion
        mock_champion.return_value = (
            "test_champion",
            {"rebalance_every": 5, "lookback_days": 504},
            0.7,
            0.06,
        )

        # Mock walk-forward result (needs variance for non-zero Sharpe)
        import random
        rng = random.Random(42)
        n_holdout = 500
        mock_result = MagicMock()
        mock_result.daily_returns = pl.DataFrame({
            "date": [date(2024, 1, 2) + timedelta(days=i) for i in range(n_holdout)],
            "portfolio_return": [0.001 + rng.gauss(0, 0.005) for _ in range(n_holdout)],
            "benchmark_return": [0.0005 + rng.gauss(0, 0.003) for _ in range(n_holdout)],
        }).with_columns(pl.col("date").cast(pl.Date))
        mock_result.equity_curve = pl.DataFrame({
            "date": [date(2024, 1, 2) + timedelta(days=i) for i in range(n_holdout)],
            "portfolio_value": [1_000_000 + i * 100 for i in range(n_holdout)],
            "benchmark_value": [1_000_000 + i * 50 for i in range(n_holdout)],
        }).with_columns(pl.col("date").cast(pl.Date))
        mock_result.metrics = {"sharpe_ratio": 0.7, "cagr": 0.12}
        mock_bt_cls.return_value.run.return_value = mock_result

        result = run_holdout_validation(
            ohlcv=ohlcv,
            tickers=["AAPL", "MSFT"],
            holdout_years=2,
            output_dir=str(tmp_path),
        )

        assert isinstance(result, HoldoutResult)
        assert result.champion_name == "test_champion"
        assert result.holdout_sharpe > 0
        assert result.holdout_n_obs > 0
        assert isinstance(result.holdout_accepted, bool)

    @patch("src.backtest.run_validation.WalkForwardBacktester")
    @patch("src.backtest.run_validation.identify_champion")
    def test_dsr_with_n_trials_1(
        self, mock_champion: MagicMock, mock_bt_cls: MagicMock, tmp_path: Path,
    ) -> None:
        """DSR with n_trials=1 should give higher p-value than grid search."""
        ohlcv = _make_long_ohlcv(["AAPL", "SPY"], n_days=2600)

        mock_champion.return_value = (
            "champ", {"rebalance_every": 5, "lookback_days": 504}, 0.7, 0.06,
        )

        # Positive returns → positive Sharpe → DSR should be meaningful
        n_holdout = 504
        returns = [0.001 + 0.0001 * (i % 5) for i in range(n_holdout)]
        mock_result = MagicMock()
        mock_result.daily_returns = pl.DataFrame({
            "date": [date(2024, 1, 2) + timedelta(days=i) for i in range(n_holdout)],
            "portfolio_return": returns,
            "benchmark_return": [0.0005] * n_holdout,
        }).with_columns(pl.col("date").cast(pl.Date))
        mock_result.equity_curve = pl.DataFrame({
            "date": [date(2024, 1, 2)],
            "portfolio_value": [1_000_000.0],
            "benchmark_value": [1_000_000.0],
        }).with_columns(pl.col("date").cast(pl.Date))
        mock_result.metrics = {}
        mock_bt_cls.return_value.run.return_value = mock_result

        result = run_holdout_validation(
            ohlcv=ohlcv,
            tickers=["AAPL"],
            holdout_years=2,
            output_dir=str(tmp_path),
        )

        # With n_trials=1 and strongly positive returns, DSR should pass threshold
        assert result.holdout_dsr > 0.95
        assert result.holdout_accepted is True

    @patch("src.backtest.run_validation.WalkForwardBacktester")
    @patch("src.backtest.run_validation.identify_champion")
    def test_saves_output_files(
        self, mock_champion: MagicMock, mock_bt_cls: MagicMock, tmp_path: Path,
    ) -> None:
        ohlcv = _make_long_ohlcv(["AAPL", "SPY"], n_days=2600)
        mock_champion.return_value = (
            "champ", {"rebalance_every": 5, "lookback_days": 504}, 0.7, 0.06,
        )
        n_holdout = 300
        mock_result = MagicMock()
        mock_result.daily_returns = pl.DataFrame({
            "date": [date(2024, 1, 2) + timedelta(days=i) for i in range(n_holdout)],
            "portfolio_return": [0.001] * n_holdout,
            "benchmark_return": [0.0005] * n_holdout,
        }).with_columns(pl.col("date").cast(pl.Date))
        mock_result.equity_curve = pl.DataFrame({
            "date": [date(2024, 1, 2)],
            "portfolio_value": [1_000_000.0],
            "benchmark_value": [1_000_000.0],
        }).with_columns(pl.col("date").cast(pl.Date))
        mock_result.metrics = {"sharpe_ratio": 0.5}
        mock_bt_cls.return_value.run.return_value = mock_result

        run_holdout_validation(
            ohlcv=ohlcv, tickers=["AAPL"],
            holdout_years=2, output_dir=str(tmp_path),
        )

        assert (tmp_path / "validation_holdout_results.json").exists()
        assert (tmp_path / "validation_holdout_equity.parquet").exists()
        assert (tmp_path / "validation_holdout_summary.md").exists()

        # Verify JSON content
        with open(tmp_path / "validation_holdout_results.json") as f:
            data = json.load(f)
        assert "holdout_sharpe" in data
        assert "holdout_dsr" in data
        assert "holdout_accepted" in data


class TestSaveHoldoutResults:
    def test_creates_files(self, tmp_path: Path) -> None:
        result = HoldoutResult(
            champion_name="test",
            champion_params={"rebalance_every": 5},
            grid_search_sharpe=0.7,
            grid_search_dsr=0.06,
            holdout_sharpe=0.65,
            holdout_dsr=0.90,
            holdout_accepted=False,
            holdout_n_obs=500,
            holdout_metrics={"sharpe_ratio": 0.65},
            wf_result=None,
            metadata={"holdout_years": 2},
        )
        _save_holdout_results(result, tmp_path)
        assert (tmp_path / "validation_holdout_results.json").exists()
        assert (tmp_path / "validation_holdout_summary.md").exists()
        # No equity parquet when wf_result is None
        assert not (tmp_path / "validation_holdout_equity.parquet").exists()

# ---------------------------------------------------------------------------
# Additional edge cases and gap coverage (appended)
# ---------------------------------------------------------------------------


class TestFilterBeforeHoldoutEdgeCases:
    def test_cutoff_date_itself_is_excluded(self) -> None:
        """The row whose date equals the cutoff (strict '<') must not appear."""
        base = date(2026, 1, 15)
        cutoff_date = date(2024, 1, 15)  # 2 years back
        df = pl.DataFrame({
            "date": [date(2023, 6, 1), cutoff_date, date(2025, 3, 1), base],
            "ticker": ["X", "X", "X", "X"],
            "close": [100.0, 101.0, 102.0, 103.0],
        }).with_columns(pl.col("date").cast(pl.Date))
        filtered = filter_before_holdout(df, holdout_years=2)
        assert cutoff_date not in filtered["date"].to_list()
        assert date(2023, 6, 1) in filtered["date"].to_list()

    def test_result_has_no_holdout_dates(self) -> None:
        """All rows in the output predate the holdout cutoff."""
        ohlcv = _make_long_ohlcv(["AAPL", "SPY"], n_days=2600)
        cutoff = _holdout_cutoff(ohlcv, holdout_years=2)
        filtered = filter_before_holdout(ohlcv, holdout_years=2)
        max_filtered = filtered["date"].max()
        assert max_filtered < cutoff, (
            f"date {max_filtered} should be strictly before cutoff {cutoff}"
        )


class TestPrepareHoldoutOhlcvEdgeCases:
    def test_holdout_period_included_in_output(self) -> None:
        """The returned DataFrame must extend through the end of the full dataset."""
        ohlcv = _make_long_ohlcv(["AAPL", "SPY"], n_days=2600)
        holdout_df, holdout_start = prepare_holdout_ohlcv(
            ohlcv, holdout_years=2, lookback_days=504,
        )
        assert holdout_df["date"].max() >= holdout_start

    def test_oversized_lookback_clamped_to_data_start(self) -> None:
        """When the buffer window precedes available data the function still
        returns rows -- the filter returns everything from data start."""
        ohlcv = _make_long_ohlcv(["AAPL", "SPY"], n_days=400, start=date(2023, 1, 2))
        holdout_df, holdout_start = prepare_holdout_ohlcv(
            ohlcv, holdout_years=1, lookback_days=756,
        )
        assert holdout_df.height > 0
        assert holdout_df["date"].min() >= ohlcv["date"].min()


class TestIdentifyChampionEdgeCases:
    def test_empty_configs_dict_raises(self, tmp_path: Path) -> None:
        """A tier file present but with an empty configs dict still raises."""
        (tmp_path / "validation_tier1_results.json").write_text(
            json.dumps({"configs": {}}), encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="No tier results"):
            identify_champion(tmp_path)

    def test_returns_four_tuple_of_correct_types(self, tmp_path: Path) -> None:
        """Return value is (str, dict, float, float)."""
        tier_data = {
            "configs": {
                "only_cfg": {
                    "mean_sharpe": 0.55,
                    "deflated_sharpe": 0.07,
                    "params": {"rebalance_every": 7},
                }
            }
        }
        (tmp_path / "validation_tier1_results.json").write_text(
            json.dumps(tier_data), encoding="utf-8"
        )
        name, params, sharpe, dsr = identify_champion(tmp_path)
        assert isinstance(name, str)
        assert isinstance(params, dict)
        assert isinstance(sharpe, float)
        assert isinstance(dsr, float)
        assert name == "only_cfg"
        assert sharpe == pytest.approx(0.55)


class TestRunHoldoutValidationEdgeCases:
    @patch("src.backtest.run_validation.WalkForwardBacktester")
    @patch("src.backtest.run_validation.identify_champion")
    def test_insufficient_holdout_obs_returns_early(
        self, mock_champion: MagicMock, mock_bt_cls: MagicMock, tmp_path: Path,
    ) -> None:
        """When daily_returns filtered to holdout period has fewer than 2 rows the
        function returns holdout_sharpe=0 and marks metadata with error flag."""
        ohlcv = _make_long_ohlcv(["AAPL", "SPY"], n_days=2600)
        mock_champion.return_value = (
            "champ", {"rebalance_every": 5, "lookback_days": 504}, 0.7, 0.06,
        )
        mock_result = MagicMock()
        mock_result.daily_returns = pl.DataFrame({
            "date": [date(2020, 1, 2)],
            "portfolio_return": [0.001],
            "benchmark_return": [0.0005],
        }).with_columns(pl.col("date").cast(pl.Date))
        mock_result.equity_curve = pl.DataFrame({
            "date": [date(2020, 1, 2)],
            "portfolio_value": [1_000_000.0],
            "benchmark_value": [1_000_000.0],
        }).with_columns(pl.col("date").cast(pl.Date))
        mock_result.metrics = {}
        mock_bt_cls.return_value.run.return_value = mock_result

        result = run_holdout_validation(
            ohlcv=ohlcv, tickers=["AAPL"],
            holdout_years=2, output_dir=str(tmp_path),
        )

        assert isinstance(result, HoldoutResult)
        assert result.holdout_sharpe == pytest.approx(0.0)
        assert result.holdout_dsr == pytest.approx(0.0)
        assert result.holdout_accepted is False
        assert result.holdout_n_obs < 2
        assert result.metadata.get("error") == "insufficient_data"

    @patch("src.backtest.run_validation.WalkForwardBacktester")
    @patch("src.backtest.run_validation.identify_champion")
    def test_champion_params_propagated_to_result(
        self, mock_champion: MagicMock, mock_bt_cls: MagicMock, tmp_path: Path,
    ) -> None:
        """HoldoutResult must carry the exact champion name and grid-search metrics."""
        ohlcv = _make_long_ohlcv(["AAPL", "SPY"], n_days=2600)
        mock_champion.return_value = (
            "my_best_config",
            {"rebalance_every": 13, "lookback_days": 756, "momentum_lookback": 21},
            0.83,
            0.09,
        )
        n = 200
        mock_result = MagicMock()
        mock_result.daily_returns = pl.DataFrame({
            "date": [date(2024, 1, 2) + timedelta(days=i) for i in range(n)],
            "portfolio_return": [0.0008] * n,
            "benchmark_return": [0.0004] * n,
        }).with_columns(pl.col("date").cast(pl.Date))
        mock_result.equity_curve = pl.DataFrame({
            "date": [date(2024, 1, 2)],
            "portfolio_value": [1_000_000.0],
            "benchmark_value": [1_000_000.0],
        }).with_columns(pl.col("date").cast(pl.Date))
        mock_result.metrics = {}
        mock_bt_cls.return_value.run.return_value = mock_result

        result = run_holdout_validation(
            ohlcv=ohlcv, tickers=["AAPL"],
            holdout_years=2, output_dir=str(tmp_path),
        )

        assert result.champion_name == "my_best_config"
        assert result.champion_params["rebalance_every"] == 13
        assert result.grid_search_sharpe == pytest.approx(0.83)
        assert result.grid_search_dsr == pytest.approx(0.09)

    @patch("src.backtest.run_validation.WalkForwardBacktester")
    @patch("src.backtest.run_validation.identify_champion")
    def test_walk_forward_backtester_called_with_correct_tickers(
        self, mock_champion: MagicMock, mock_bt_cls: MagicMock, tmp_path: Path,
    ) -> None:
        """WalkForwardBacktester.run() must receive the tickers list and
        benchmark_ticker passed to run_holdout_validation."""
        ohlcv = _make_long_ohlcv(["AAPL", "MSFT", "GOOG", "SPY"], n_days=2600)
        mock_champion.return_value = (
            "champ", {"rebalance_every": 5, "lookback_days": 504}, 0.7, 0.06,
        )
        n = 300
        mock_result = MagicMock()
        mock_result.daily_returns = pl.DataFrame({
            "date": [date(2024, 1, 2) + timedelta(days=i) for i in range(n)],
            "portfolio_return": [0.001] * n,
            "benchmark_return": [0.0005] * n,
        }).with_columns(pl.col("date").cast(pl.Date))
        mock_result.equity_curve = pl.DataFrame({
            "date": [date(2024, 1, 2)],
            "portfolio_value": [1_000_000.0],
            "benchmark_value": [1_000_000.0],
        }).with_columns(pl.col("date").cast(pl.Date))
        mock_result.metrics = {}
        mock_bt_cls.return_value.run.return_value = mock_result

        run_holdout_validation(
            ohlcv=ohlcv,
            tickers=["AAPL", "MSFT", "GOOG"],
            benchmark_ticker="SPY",
            holdout_years=2,
            output_dir=str(tmp_path),
        )

        call_args = mock_bt_cls.return_value.run.call_args
        assert call_args.args[1] == ["AAPL", "MSFT", "GOOG"]
        assert call_args.args[2] == "SPY"


class TestSaveHoldoutResultsEdgeCases:
    def test_json_contains_all_keys_and_values(self, tmp_path: Path) -> None:
        """Every field of HoldoutResult is serialised to JSON."""
        result = HoldoutResult(
            champion_name="champ",
            champion_params={"rebalance_every": 13},
            grid_search_sharpe=0.75,
            grid_search_dsr=0.07,
            holdout_sharpe=0.68,
            holdout_dsr=0.88,
            holdout_accepted=False,
            holdout_n_obs=480,
            holdout_metrics={"cagr": 0.11},
            wf_result=None,
            metadata={"holdout_years": 2, "holdout_start": "2024-01-01"},
        )
        _save_holdout_results(result, tmp_path)
        with open(tmp_path / "validation_holdout_results.json") as fh:
            data = json.load(fh)
        expected_keys = {
            "champion_name", "champion_params",
            "grid_search_sharpe", "grid_search_dsr",
            "holdout_sharpe", "holdout_dsr",
            "holdout_accepted", "holdout_n_obs",
            "holdout_metrics", "metadata",
        }
        assert set(data.keys()) >= expected_keys
        assert data["champion_name"] == "champ"
        assert data["holdout_sharpe"] == pytest.approx(0.68)
        assert data["holdout_accepted"] is False
        assert data["holdout_n_obs"] == 480

    def test_markdown_accepted_verdict(self, tmp_path: Path) -> None:
        """When holdout_accepted=True, the markdown contains ACCEPTED."""
        result = HoldoutResult(
            champion_name="ward_shrink_rb13",
            champion_params={},
            grid_search_sharpe=0.8,
            grid_search_dsr=0.08,
            holdout_sharpe=0.72,
            holdout_dsr=0.96,
            holdout_accepted=True,
            holdout_n_obs=504,
            metadata={"holdout_years": 2, "holdout_start": "2024-01-01"},
        )
        _save_holdout_results(result, tmp_path)
        content = (tmp_path / "validation_holdout_summary.md").read_text()
        assert "ward_shrink_rb13" in content
        assert "ACCEPTED" in content

    def test_markdown_rejected_verdict(self, tmp_path: Path) -> None:
        """When holdout_accepted=False, the markdown contains REJECTED."""
        result = HoldoutResult(
            champion_name="weak_config",
            champion_params={},
            grid_search_sharpe=0.3,
            grid_search_dsr=0.02,
            holdout_sharpe=-0.1,
            holdout_dsr=0.20,
            holdout_accepted=False,
            holdout_n_obs=200,
            metadata={"holdout_years": 1},
        )
        _save_holdout_results(result, tmp_path)
        content = (tmp_path / "validation_holdout_summary.md").read_text()
        assert "REJECTED" in content

    def test_equity_parquet_written_when_wf_result_present(
        self, tmp_path: Path
    ) -> None:
        """When wf_result is not None, its equity_curve is written to Parquet."""
        mock_wf = MagicMock()
        mock_wf.equity_curve = pl.DataFrame({
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "portfolio_value": [1_000_000.0, 1_001_000.0],
            "benchmark_value": [1_000_000.0, 1_000_500.0],
        }).with_columns(pl.col("date").cast(pl.Date))
        result = HoldoutResult(
            champion_name="champ",
            champion_params={},
            grid_search_sharpe=0.7,
            grid_search_dsr=0.06,
            holdout_sharpe=0.65,
            holdout_dsr=0.91,
            holdout_accepted=True,
            holdout_n_obs=400,
            wf_result=mock_wf,
            metadata={"holdout_years": 2, "holdout_start": "2024-01-02"},
        )
        _save_holdout_results(result, tmp_path)
        parquet_path = tmp_path / "validation_holdout_equity.parquet"
        assert parquet_path.exists()
        equity_df = pl.read_parquet(parquet_path)
        assert "date" in equity_df.columns
        assert "portfolio_value" in equity_df.columns
        assert equity_df.height == 2

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        """_save_holdout_results must create the directory if it does not exist."""
        nested = tmp_path / "deep" / "nested" / "dir"
        assert not nested.exists()
        result = HoldoutResult(
            champion_name="x",
            champion_params={},
            grid_search_sharpe=0.5,
            grid_search_dsr=0.05,
            holdout_sharpe=0.4,
            holdout_dsr=0.6,
            holdout_accepted=False,
            holdout_n_obs=100,
            metadata={},
        )
        _save_holdout_results(result, nested)
        assert (nested / "validation_holdout_results.json").exists()
