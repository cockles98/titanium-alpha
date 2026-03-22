"""Comprehensive fine-tuning grid search for walk-forward strategy parameters.

Three-tier systematic approach designed for ~18h per tier:

  Tier 1 (~165 configs): Single-axis sweeps + key 2-way interactions
  Tier 2 (~165 configs): Multi-axis factorial combinations (3/4-way)
  Tier 3 (~170 configs): Adaptive champion refinement from Tier 1+2 winners

Features:

  - Resume: skips already-computed configs on restart
  - Incremental saves: persists every 5 configs (crash-safe)
  - ETA: rolling-average time estimation
  - Cross-tier dedup: same parameter fingerprint never runs twice

Time budget (NaiveModelFactory, C(6,2)=15 CPCV-OOS paths):

  - ~375s per config validation (25s x 15 paths)
  - ~172 configs per 18h

Usage::

    python -m src.backtest.run_validation --tier 1        # Day 1
    python -m src.backtest.run_validation --tier 2        # Day 2
    python -m src.backtest.run_validation --tier 3        # Day 3
    python -m src.backtest.run_validation --tier all       # Run all sequentially
    python -m src.backtest.run_validation --estimate       # Time 1 config
    python -m src.backtest.run_validation --dry-run        # Print grid, no execution
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from src.backtest.cpcv import TransactionCosts
from src.backtest.cpcv_oos import CPCVParameterValidator, ValidationResult
from src.backtest.walk_forward import (
    KillswitchConfig,
    NaiveModelFactory,
    WalkForwardConfig,
)
from src.portfolio.hrp import HRPConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path("data/outputs")
_BASE_COSTS = TransactionCosts(slippage_bps=5.0, commission_bps=10.0)
_SAVE_EVERY = 5


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """Single config + factory pair for CPCV-OOS validation."""

    name: str
    wf_config: WalkForwardConfig
    factory: NaiveModelFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dyn_maxw(n: int) -> float:
    """Dynamic max_weight = min(0.25, 2/n)."""
    return min(0.25, 2.0 / max(n, 1))


def _hrp(n: int, **kw: Any) -> HRPConfig:
    """HRPConfig with baseline defaults + overrides."""
    d: dict[str, Any] = {
        "linkage_method": "single",
        "correlation_method": "pearson",
        "shrinkage": False,
        "confidence_tilt_cap": 0.20,
        "min_weight": 0.0,
        "max_weight": _dyn_maxw(n),
        "turnover_threshold": 0.02,
    }
    d.update(kw)
    return HRPConfig(**d)


def _wf(**kw: Any) -> WalkForwardConfig:
    """WalkForwardConfig with baseline defaults + overrides."""
    d: dict[str, Any] = {
        "rebalance_every": 5,
        "retrain_every": 126,
        "lookback_days": 504,
        "initial_capital": 1_000_000.0,
        "costs": _BASE_COSTS,
        "min_rebalance_delta": 0.02,
        "trading_days_per_year": 252,
        "rf": 0.05,
    }
    d.update(kw)
    return WalkForwardConfig(**d)


def _t(name: str, mom: int = 5, **wf_kw: Any) -> Trial:
    """Shorthand: create Trial with baseline defaults + overrides."""
    return Trial(name=name, wf_config=_wf(**wf_kw), factory=NaiveModelFactory(lookback=mom))


def _trial_params(trial: Trial, n: int) -> dict[str, Any]:
    """Extract all tunable parameters into a flat dict for fingerprinting."""
    cfg = trial.wf_config
    hrp = cfg.hrp_config or HRPConfig(max_weight=_dyn_maxw(n))
    ks = cfg.killswitch
    costs = cfg.costs or TransactionCosts()
    return {
        "momentum_lookback": trial.factory.lookback,
        "rebalance_every": cfg.rebalance_every,
        "retrain_every": cfg.retrain_every,
        "lookback_days": cfg.lookback_days,
        "linkage_method": hrp.linkage_method,
        "shrinkage": hrp.shrinkage,
        "correlation_method": hrp.correlation_method,
        "confidence_tilt_cap": hrp.confidence_tilt_cap,
        "max_weight": round(hrp.max_weight, 6),
        "min_weight": hrp.min_weight,
        "turnover_threshold": hrp.turnover_threshold,
        "target_vol": cfg.target_vol,
        "vol_lookback": cfg.vol_lookback,
        "min_leverage": cfg.min_leverage,
        "max_leverage": cfg.max_leverage,
        "killswitch_max_dd": ks.max_drawdown_pct if ks else None,
        "killswitch_recovery": ks.recovery_threshold_pct if ks else None,
        "killswitch_ramp": ks.ramp_up_days if ks else None,
        "min_rebalance_delta": cfg.min_rebalance_delta,
        "slippage_bps": costs.slippage_bps,
        "commission_bps": costs.commission_bps,
        "market_impact_bps": costs.market_impact_bps,
    }


def _param_fp(params: dict[str, Any]) -> str:
    """Canonical JSON string for dedup (parameter fingerprint)."""
    norm: dict[str, Any] = {}
    for k in sorted(params):
        v = params[k]
        norm[k] = round(v, 6) if isinstance(v, float) else v
    return json.dumps(norm, sort_keys=True)


def _dedup(
    trials: list[Trial], n: int, existing_fps: set[str] | None = None,
) -> list[Trial]:
    """Remove duplicate trials by parameter fingerprint."""
    seen = set(existing_fps) if existing_fps else set()
    unique: list[Trial] = []
    for t in trials:
        fp = _param_fp(_trial_params(t, n))
        if fp not in seen:
            seen.add(fp)
            unique.append(t)
    return unique


def _trial_from_params(name: str, params: dict[str, Any], n: int) -> Trial:
    """Reconstruct a Trial from a saved parameter dict."""
    hrp = _hrp(
        n,
        linkage_method=params.get("linkage_method", "single"),
        correlation_method=params.get("correlation_method", "pearson"),
        shrinkage=params.get("shrinkage", False),
        confidence_tilt_cap=params.get("confidence_tilt_cap", 0.20),
        max_weight=params.get("max_weight", _dyn_maxw(n)),
        min_weight=params.get("min_weight", 0.0),
        turnover_threshold=params.get("turnover_threshold", 0.02),
    )
    ks = None
    if params.get("killswitch_max_dd") is not None:
        ks = KillswitchConfig(
            max_drawdown_pct=params["killswitch_max_dd"],
            recovery_threshold_pct=params.get("killswitch_recovery", -0.05),
            ramp_up_days=params.get("killswitch_ramp", 21),
        )
    costs = TransactionCosts(
        slippage_bps=params.get("slippage_bps", 5.0),
        commission_bps=params.get("commission_bps", 10.0),
        market_impact_bps=params.get("market_impact_bps", 0.0),
    )
    wf = WalkForwardConfig(
        rebalance_every=params.get("rebalance_every", 5),
        retrain_every=params.get("retrain_every", 126),
        lookback_days=params.get("lookback_days", 504),
        initial_capital=1_000_000.0,
        costs=costs,
        min_rebalance_delta=params.get("min_rebalance_delta", 0.02),
        trading_days_per_year=252,
        rf=0.05,
        target_vol=params.get("target_vol"),
        vol_lookback=params.get("vol_lookback", 63),
        max_leverage=params.get("max_leverage", 1.0),
        min_leverage=params.get("min_leverage", 0.5),
        killswitch=ks,
        hrp_config=hrp,
    )
    factory = NaiveModelFactory(lookback=params.get("momentum_lookback", 5))
    return Trial(name=name, wf_config=wf, factory=factory)


# ---------------------------------------------------------------------------
# Tier 1: Single-axis sweeps + 2-way interactions (~165 configs)
# ---------------------------------------------------------------------------


def build_tier1(n: int) -> list[Trial]:
    """Tier 1: isolate marginal effect of each dimension, then test key pairs."""
    trials: list[Trial] = []

    # === BASELINE ===
    trials.append(_t("baseline"))

    # === A. SINGLE-AXIS SWEEPS ===

    # A1. Momentum lookback (7 new, baseline=5 already above)
    for lb in [1, 2, 3, 10, 21, 63, 126]:
        trials.append(_t(f"s_mom{lb:03d}", mom=lb))

    # A2. Rebalance frequency (7 new)
    for rb in [1, 2, 3, 7, 10, 15, 21]:
        trials.append(_t(f"s_rebal{rb:03d}", rebalance_every=rb))

    # A3. Covariance lookback (6 new)
    for cov in [63, 126, 252, 378, 630, 756]:
        trials.append(_t(f"s_cov{cov:03d}", lookback_days=cov))

    # A4. Retrain frequency (5 new)
    for rt in [21, 42, 63, 189, 252]:
        trials.append(_t(f"s_retrain{rt:03d}", retrain_every=rt))

    # A5. HRP linkage x shrinkage (3 new)
    trials.append(_t("s_ward", hrp_config=_hrp(n, linkage_method="ward")))
    trials.append(_t("s_shrink", hrp_config=_hrp(n, shrinkage=True)))
    trials.append(_t("s_ward_shrink", hrp_config=_hrp(n, linkage_method="ward", shrinkage=True)))

    # A6. Confidence tilt cap (8 new)
    for tc in [0.0, 0.05, 0.10, 0.15, 0.30, 0.50, 0.75, 1.0]:
        trials.append(_t(f"s_tilt{int(tc * 100):03d}", hrp_config=_hrp(n, confidence_tilt_cap=tc)))

    # A7. Target volatility (6 new)
    for tv in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        trials.append(_t(f"s_vol{int(tv * 100):03d}", target_vol=tv))

    # A8. Max weight (6 new)
    for mw in [0.04, 0.06, 0.08, 0.10, 0.15, 0.25]:
        trials.append(_t(f"s_maxw{int(mw * 100):03d}", hrp_config=_hrp(n, max_weight=mw)))

    # A9. Min rebalance delta (6 new)
    for d in [0.005, 0.01, 0.015, 0.03, 0.04, 0.05]:
        trials.append(_t(f"s_delta{int(d * 1000):03d}", min_rebalance_delta=d))

    # A10. Killswitch threshold (6 new)
    for dd in [0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        trials.append(_t(f"s_ks{int(dd * 100):03d}", killswitch=KillswitchConfig(max_drawdown_pct=-dd)))

    # A11. Correlation method (1 new)
    trials.append(_t("s_spearman", hrp_config=_hrp(n, correlation_method="spearman")))

    # A12. HRP turnover threshold (5 new)
    for to in [0.0, 0.005, 0.01, 0.03, 0.05]:
        trials.append(_t(f"s_to{int(to * 1000):03d}", hrp_config=_hrp(n, turnover_threshold=to)))

    # A13. Vol lookback (4 new, with target_vol=0.12 to activate)
    for vlb in [21, 42, 84, 126]:
        trials.append(_t(f"s_vlb{vlb:03d}", target_vol=0.12, vol_lookback=vlb))

    # A14. Min leverage (4 new, with target_vol=0.12)
    for ml in [0.0, 0.2, 0.3, 0.7]:
        trials.append(_t(f"s_mlev{int(ml * 10):02d}", target_vol=0.12, min_leverage=ml))

    # A15. Killswitch recovery threshold (3 new, with max_dd=-0.15)
    for rec in [0.02, 0.08, 0.10]:
        trials.append(_t(
            f"s_ksrec{int(rec * 100):03d}",
            killswitch=KillswitchConfig(max_drawdown_pct=-0.15, recovery_threshold_pct=-rec),
        ))

    # A16. Killswitch ramp days (4 new, with max_dd=-0.15)
    for rd in [5, 10, 15, 42]:
        trials.append(_t(
            f"s_ksramp{rd:03d}",
            killswitch=KillswitchConfig(max_drawdown_pct=-0.15, ramp_up_days=rd),
        ))

    # === B. 2-WAY INTERACTIONS ===

    # B1. Momentum x Rebalance (16)
    for mom in [1, 3, 10, 21]:
        for rb in [1, 3, 10, 21]:
            trials.append(_t(f"x_mom{mom:03d}_rebal{rb:03d}", mom=mom, rebalance_every=rb))

    # B2. Momentum x Covariance (6)
    for mom in [1, 21]:
        for cov in [63, 252, 756]:
            trials.append(_t(f"x_mom{mom:03d}_cov{cov:03d}", mom=mom, lookback_days=cov))

    # B3. Rebalance x Tilt (12)
    for rb in [1, 3, 10]:
        for tc in [0.0, 0.10, 0.50, 1.0]:
            trials.append(_t(
                f"x_rebal{rb:03d}_tilt{int(tc * 100):03d}",
                rebalance_every=rb,
                hrp_config=_hrp(n, confidence_tilt_cap=tc),
            ))

    # B4. Ward+Shrinkage x Tilt (4)
    for tc in [0.0, 0.10, 0.50, 1.0]:
        trials.append(_t(
            f"x_wardshrink_tilt{int(tc * 100):03d}",
            hrp_config=_hrp(n, linkage_method="ward", shrinkage=True, confidence_tilt_cap=tc),
        ))

    # B5. Momentum x HRP variant (6)
    for mom in [1, 21]:
        for hname, hkw in [
            ("ward", {"linkage_method": "ward"}),
            ("shrink", {"shrinkage": True}),
            ("wardshrink", {"linkage_method": "ward", "shrinkage": True}),
        ]:
            trials.append(_t(f"x_mom{mom:03d}_{hname}", mom=mom, hrp_config=_hrp(n, **hkw)))

    # B6. Vol target x Killswitch (4)
    for tv in [0.10, 0.15]:
        for dd in [0.15, 0.20]:
            trials.append(_t(
                f"x_vol{int(tv * 100):03d}_ks{int(dd * 100):03d}",
                target_vol=tv,
                killswitch=KillswitchConfig(max_drawdown_pct=-dd),
            ))

    # B7. Rebalance x Covariance (9)
    for rb in [1, 3, 10]:
        for cov in [63, 252, 756]:
            trials.append(_t(f"x_rebal{rb:03d}_cov{cov:03d}", rebalance_every=rb, lookback_days=cov))

    # B8. Momentum x Max weight (6)
    for mom in [1, 21]:
        for mw in [0.04, 0.10, 0.25]:
            trials.append(_t(
                f"x_mom{mom:03d}_maxw{int(mw * 100):03d}",
                mom=mom,
                hrp_config=_hrp(n, max_weight=mw),
            ))

    # B9. Rebalance x Delta (6)
    for rb in [1, 3]:
        for d in [0.01, 0.03, 0.05]:
            trials.append(_t(
                f"x_rebal{rb:03d}_delta{int(d * 1000):03d}",
                rebalance_every=rb,
                min_rebalance_delta=d,
            ))

    # B10. Retrain x Covariance (6)
    for rt in [21, 63]:
        for cov in [63, 126, 252]:
            trials.append(_t(f"x_retrain{rt:03d}_cov{cov:03d}", retrain_every=rt, lookback_days=cov))

    # === C. STRATEGIC 3/4-WAY COMBOS ===

    # C1. Fast signal + fast rebalance + short memory
    for cov in [63, 252]:
        trials.append(_t(f"x3_mom001_rebal001_cov{cov:03d}", mom=1, rebalance_every=1, lookback_days=cov))

    # C2. Fast signal + fast rebalance + high tilt
    for tc in [0.50, 1.0]:
        trials.append(_t(
            f"x3_mom001_rebal001_tilt{int(tc * 100):03d}",
            mom=1, rebalance_every=1,
            hrp_config=_hrp(n, confidence_tilt_cap=tc),
        ))

    # C3. Medium combos
    trials.append(_t("x3_mom003_rebal003_cov252", mom=3, rebalance_every=3, lookback_days=252))
    trials.append(_t("x3_mom010_rebal010_cov252", mom=10, rebalance_every=10, lookback_days=252))
    trials.append(_t("x3_mom021_rebal010_cov063", mom=21, rebalance_every=10, lookback_days=63))

    # C4. Fast + ward_shrink
    trials.append(_t(
        "x3_mom001_rebal001_wardshrink",
        mom=1, rebalance_every=1,
        hrp_config=_hrp(n, linkage_method="ward", shrinkage=True),
    ))

    # C5. 4-way godmode
    trials.append(_t(
        "x4_godmode",
        mom=1, rebalance_every=1, lookback_days=63,
        hrp_config=_hrp(n, confidence_tilt_cap=1.0),
    ))

    # C6. Godmode + risk overlay
    trials.append(_t(
        "x4_godmode_ks015",
        mom=1, rebalance_every=1, lookback_days=63,
        hrp_config=_hrp(n, confidence_tilt_cap=1.0),
        killswitch=KillswitchConfig(max_drawdown_pct=-0.15),
    ))
    trials.append(_t(
        "x4_godmode_vol012",
        mom=1, rebalance_every=1, lookback_days=63,
        hrp_config=_hrp(n, confidence_tilt_cap=1.0),
        target_vol=0.12,
    ))

    return _dedup(trials, n)


# ---------------------------------------------------------------------------
# Tier 2: Multi-axis factorial combinations (~165 configs)
# ---------------------------------------------------------------------------


def build_tier2(n: int) -> list[Trial]:
    """Tier 2: systematic 3/4-way factorials focused on rebal=5-15, cov=126/756 sweet spot."""
    trials: list[Trial] = []

    # === A. 3-WAY: Momentum x Rebalance x Tilt (27) ===
    for mom in [1, 5, 21]:
        for rb in [5, 10, 15]:
            for tc in [0.20, 0.60, 1.0]:
                trials.append(_t(
                    f"t2a_m{mom:02d}_r{rb:02d}_t{int(tc * 100):03d}",
                    mom=mom, rebalance_every=rb,
                    hrp_config=_hrp(n, confidence_tilt_cap=tc),
                ))

    # === B. 3-WAY: Momentum x Rebalance x Covariance (27) ===
    for mom in [1, 5, 21]:
        for rb in [5, 10, 15]:
            for cov in [63, 126, 756]:
                trials.append(_t(
                    f"t2b_m{mom:02d}_r{rb:02d}_c{cov:03d}",
                    mom=mom, rebalance_every=rb, lookback_days=cov,
                ))

    # === C. 3-WAY: Momentum x Tilt x Covariance (27) ===
    for mom in [1, 5, 21]:
        for tc in [0.20, 0.60, 1.0]:
            for cov in [63, 126, 756]:
                trials.append(_t(
                    f"t2c_m{mom:02d}_t{int(tc * 100):03d}_c{cov:03d}",
                    mom=mom, lookback_days=cov,
                    hrp_config=_hrp(n, confidence_tilt_cap=tc),
                ))

    # === D. 3-WAY: HRP structure in winner region (8) ===
    for link in ["single", "ward"]:
        for shrk in [False, True]:
            for tc in [0.20, 1.0]:
                lk = "W" if link == "ward" else "S"
                sk = "Y" if shrk else "N"
                trials.append(_t(
                    f"t2d_{lk}{sk}_t{int(tc * 100):03d}",
                    rebalance_every=10, lookback_days=756,
                    hrp_config=_hrp(n, linkage_method=link, shrinkage=shrk, confidence_tilt_cap=tc),
                ))

    # === E. 4-WAY: Momentum x Rebalance x Tilt x Covariance (36) ===
    for mom in [1, 5, 21]:
        for rb in [5, 10, 15]:
            for tc in [0.60, 1.0]:
                for cov in [126, 756]:
                    trials.append(_t(
                        f"t2e_m{mom:02d}_r{rb:02d}_t{int(tc * 100):03d}_c{cov:03d}",
                        mom=mom, rebalance_every=rb, lookback_days=cov,
                        hrp_config=_hrp(n, confidence_tilt_cap=tc),
                    ))

    # === F. Rebalance fine-grain around sweet spot (12) ===
    for rb in [8, 9, 11, 12, 13, 14]:
        for cov in [126, 756]:
            trials.append(_t(
                f"t2f_rb{rb:03d}_c{cov:03d}",
                rebalance_every=rb, lookback_days=cov,
            ))

    # === G. Covariance exploration between poles (12) ===
    for cov in [100, 150, 189, 504, 630, 840]:
        for rb in [10, 15]:
            trials.append(_t(
                f"t2g_rb{rb:03d}_c{cov:03d}",
                rebalance_every=rb, lookback_days=cov,
            ))

    # === H. HRP structure combos in winner region (12) ===
    for link in ["single", "ward"]:
        for shrk in [False, True]:
            for corr in ["pearson", "spearman"]:
                lk = "W" if link == "ward" else "S"
                sk = "Y" if shrk else "N"
                trials.append(_t(
                    f"t2h_{lk}{sk}_{corr[0]}_c756",
                    rebalance_every=10, lookback_days=756,
                    hrp_config=_hrp(n, linkage_method=link, shrinkage=shrk, correlation_method=corr),
                ))
    for link in ["single", "ward"]:
        for shrk in [False, True]:
            lk = "W" if link == "ward" else "S"
            sk = "Y" if shrk else "N"
            trials.append(_t(
                f"t2h_{lk}{sk}_p_c126",
                rebalance_every=10, lookback_days=126,
                hrp_config=_hrp(n, linkage_method=link, shrinkage=shrk),
            ))

    # === I. Max_weight x Tilt x Rebalance (18) ===
    for mw in [0.04, 0.10]:
        for tc in [0.20, 0.50, 1.0]:
            for rb in [5, 10, 15]:
                trials.append(_t(
                    f"t2i_mw{int(mw * 100):02d}_t{int(tc * 100):03d}_r{rb:02d}",
                    rebalance_every=rb,
                    hrp_config=_hrp(n, max_weight=mw, confidence_tilt_cap=tc),
                ))

    # === J. Turnover management: Rebalance x Delta x TO threshold (27) ===
    for rb in [5, 10, 15]:
        for d in [0.01, 0.02, 0.03]:
            for to in [0.0, 0.02, 0.05]:
                trials.append(_t(
                    f"t2j_r{rb:02d}_d{int(d * 1000):03d}_to{int(to * 1000):03d}",
                    rebalance_every=rb,
                    min_rebalance_delta=d,
                    hrp_config=_hrp(n, turnover_threshold=to),
                ))

    return _dedup(trials, n)


# ---------------------------------------------------------------------------
# Tier 3: Adaptive champion refinement (~170 configs)
# ---------------------------------------------------------------------------


def _load_all_results(output_dir: Path = _OUTPUT_DIR) -> list[dict[str, Any]]:
    """Load all previous tier results as a flat list sorted by DSR."""
    results: list[dict[str, Any]] = []
    for tier_num in [1, 2, 3]:
        path = output_dir / f"validation_tier{tier_num}_results.json"
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt file {} — skipping", path)
            continue
        for name, r in data.get("configs", {}).items():
            results.append({"name": name, **r})
    results.sort(key=lambda x: x.get("deflated_sharpe", 0), reverse=True)
    return results


def _load_existing_fps(output_dir: Path = _OUTPUT_DIR) -> set[str]:
    """Load parameter fingerprints from all completed validations."""
    fps: set[str] = set()
    for r in _load_all_results(output_dir):
        params = r.get("params")
        if params:
            fps.add(_param_fp(params))
    return fps


def build_tier3(n: int) -> list[Trial]:
    """Tier 3: adaptive refinement from Tier 1+2 results."""
    prev = _load_all_results()
    if not prev:
        logger.warning("No previous results found; using fallback Tier 3 grid")
        return _build_tier3_fallback(n)

    trials: list[Trial] = []

    # === A. NARROW SWEEP AROUND TOP-5 CHAMPIONS (~60 configs) ===
    for rank, champ in enumerate(prev[:5]):
        p = champ["params"]
        tag = f"t3r{rank}"

        # Rebalance: +/-1, +/-2 (most impactful dimension)
        rb = p["rebalance_every"]
        for delta in [-2, -1, 1, 2]:
            new_rb = max(1, rb + delta)
            if new_rb != rb:
                new_p = {**p, "rebalance_every": new_rb}
                trials.append(_trial_from_params(f"{tag}_rb{new_rb:03d}", new_p, n))

        # Covariance: +/-25, +/-50 (fine-grain around pole)
        cov = p["lookback_days"]
        for delta in [-50, -25, 25, 50]:
            new_cov = max(21, cov + delta)
            if new_cov != cov:
                new_p = {**p, "lookback_days": new_cov}
                trials.append(_trial_from_params(f"{tag}_cov{new_cov:03d}", new_p, n))

        # HRP structure variants
        for link in ["single", "ward"]:
            for shrk in [False, True]:
                if link != p.get("linkage_method") or shrk != p.get("shrinkage"):
                    new_p = {**p, "linkage_method": link, "shrinkage": shrk}
                    lk = "w" if link == "ward" else "s"
                    sk = "y" if shrk else "n"
                    trials.append(_trial_from_params(f"{tag}_hrp_{lk}{sk}", new_p, n))

        # Correlation method swap
        corr = p.get("correlation_method", "pearson")
        alt_corr = "spearman" if corr == "pearson" else "pearson"
        new_p = {**p, "correlation_method": alt_corr}
        trials.append(_trial_from_params(f"{tag}_corr_{alt_corr[0]}", new_p, n))

    # === B. CROSS-POLLINATION (~40 configs) ===
    # For each pair in top-5, swap one dimension (focus on impactful dims)
    for i in range(min(5, len(prev))):
        for j in range(i + 1, min(5, len(prev))):
            pa, pb = prev[i]["params"], prev[j]["params"]
            swaps = [
                ("rebalance_every", "rb"),
                ("lookback_days", "cov"),
                ("linkage_method", "lnk"),
                ("shrinkage", "shr"),
                ("correlation_method", "cor"),
            ]
            for dim, short in swaps:
                if pa.get(dim) != pb.get(dim):
                    hybrid = {**pa, dim: pb[dim]}
                    val = pb[dim]
                    if isinstance(val, bool):
                        val_str = "Y" if val else "N"
                    elif isinstance(val, float):
                        val_str = f"{int(val * 100):03d}" if val < 10 else f"{int(val)}"
                    elif isinstance(val, int):
                        val_str = f"{val:03d}"
                    else:
                        val_str = str(val)[:3]
                    trials.append(_trial_from_params(
                        f"t3_cross_{i}_{j}_{short}{val_str}", hybrid, n,
                    ))

    # === C. COST STRESS TEST (~50 configs) ===
    cost_regimes = [
        ("low", TransactionCosts(slippage_bps=2.0, commission_bps=5.0)),
        ("2x", TransactionCosts(slippage_bps=10.0, commission_bps=20.0)),
        ("3x", TransactionCosts(slippage_bps=15.0, commission_bps=30.0)),
        ("impact5", TransactionCosts(slippage_bps=5.0, commission_bps=10.0, market_impact_bps=5.0)),
        ("zero", TransactionCosts(slippage_bps=0.0, commission_bps=0.0)),
    ]
    for rank, champ in enumerate(prev[:10]):
        p = champ["params"]
        tag = f"t3s{rank}"
        for cname, costs in cost_regimes:
            new_p = {
                **p,
                "slippage_bps": costs.slippage_bps,
                "commission_bps": costs.commission_bps,
                "market_impact_bps": costs.market_impact_bps,
            }
            trials.append(_trial_from_params(f"{tag}_{cname}", new_p, n))

    # === D. HRP STRUCTURE ON CHAMPIONS (~25 configs) ===
    hrp_combos = [
        ("ward_shrink", {"linkage_method": "ward", "shrinkage": True}),
        ("ward_spear", {"linkage_method": "ward", "correlation_method": "spearman"}),
        ("shrink_spear", {"shrinkage": True, "correlation_method": "spearman"}),
        ("ward_shrink_spear", {"linkage_method": "ward", "shrinkage": True, "correlation_method": "spearman"}),
        ("single_shrink", {"linkage_method": "single", "shrinkage": True}),
    ]
    for rank, champ in enumerate(prev[:5]):
        p = champ["params"]
        tag = f"t3d{rank}"
        for hname, overrides in hrp_combos:
            new_p = {**p, **overrides}
            trials.append(_trial_from_params(f"{tag}_{hname}", new_p, n))

    return _dedup(trials, n)


def _build_tier3_fallback(n: int) -> list[Trial]:
    """Fallback Tier 3 when no previous results exist."""
    trials: list[Trial] = []
    # Pre-defined champion combos based on Tier 1+2 winner profile
    bases = [
        (5, 13, 756, 0.20),
        (5, 10, 756, 1.00),
        (1, 15, 756, 1.00),
        (5, 14, 126, 0.20),
        (5, 15, 150, 0.20),
    ]
    for mom, rb, cov, tc in bases:
        tag = f"t3f_m{mom}_r{rb}_c{cov}_t{int(tc * 100)}"
        # Narrow sweep rebalance and covariance
        for dr in [-2, -1, 1, 2]:
            nr = max(1, rb + dr)
            trials.append(_t(f"{tag}_rb{nr}", mom=mom, rebalance_every=nr,
                             lookback_days=cov, hrp_config=_hrp(n, confidence_tilt_cap=tc)))
        for dc in [-50, -25, 25, 50]:
            nc = max(21, cov + dc)
            trials.append(_t(f"{tag}_cov{nc}", mom=mom, rebalance_every=rb,
                             lookback_days=nc, hrp_config=_hrp(n, confidence_tilt_cap=tc)))
        # HRP variants
        for link, shrk in [("ward", True), ("ward", False), ("single", True)]:
            lk = "w" if link == "ward" else "s"
            sk = "y" if shrk else "n"
            trials.append(_t(f"{tag}_hrp_{lk}{sk}", mom=mom, rebalance_every=rb,
                             lookback_days=cov, hrp_config=_hrp(n, linkage_method=link,
                             shrinkage=shrk, confidence_tilt_cap=tc)))
    # Cost stress on winner profile
    for slip in [2, 10, 15, 25]:
        trials.append(_t(
            f"t3f_winner_slip{slip:03d}",
            rebalance_every=13, lookback_days=756,
            costs=TransactionCosts(slippage_bps=float(slip), commission_bps=10.0),
        ))
    return _dedup(trials, n)


# ---------------------------------------------------------------------------
# Runner with resume, incremental saves, and ETA
# ---------------------------------------------------------------------------


def run_tier(
    tier_num: int,
    trials: list[Trial],
    validator: CPCVParameterValidator,
    n_tickers: int,
    output_dir: Path = _OUTPUT_DIR,
) -> dict[str, dict[str, Any]]:
    """Run a tier of trials with resume support, periodic saves, and ETA."""
    results_path = output_dir / f"validation_tier{tier_num}_results.json"

    # Load existing results for resume
    existing: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        try:
            with open(results_path, encoding="utf-8") as f:
                data = json.load(f)
            existing = data.get("configs", {})
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupt results file {} — starting fresh: {}", results_path, e)
            existing = {}

    # Load cross-tier fingerprints for dedup
    all_fps = _load_existing_fps(output_dir)

    # Filter pending trials (resume + cross-tier dedup)
    pending: list[Trial] = []
    for t in trials:
        if t.name in existing:
            continue
        fp = _param_fp(_trial_params(t, n_tickers))
        if fp in all_fps:
            logger.debug("Cross-tier dedup: skip {}", t.name)
            continue
        pending.append(t)

    # Count total trials for DSR correction (cumulative across all tiers)
    # prev_results = completed configs from OTHER tiers (exclude current tier's results)
    all_prev = _load_all_results(output_dir)
    prev_other_tier = sum(
        1 for r in all_prev if r["name"] not in existing
    )
    # Total = configs from other tiers + all configs scheduled for this tier
    # (both completed + pending, since they're all from the same testing family)
    n_total_trials = prev_other_tier + len(existing) + len(pending)
    # Tier 3 inflation: adaptive configs are statistically dependent on Tier 1+2
    # champions, so the effective number of independent trials is higher.
    # We inflate by 2x for Tier 3 to partially compensate for data snooping.
    if tier_num == 3 and prev_other_tier > 0:
        n_total_trials = n_total_trials + prev_other_tier  # count prior trials twice

    total = len(existing) + len(pending)
    logger.info(
        "=== TIER {} === {} total | {} pending | {} completed | n_trials_dsr={}",
        tier_num, total, len(pending), len(existing), n_total_trials,
    )

    if not pending:
        logger.info("All configs already computed. Nothing to do.")
        _save_all_outputs(tier_num, existing, total, output_dir)
        return existing

    # Run pending trials with ETA
    times: deque[float] = deque(maxlen=20)
    tier_start = time.monotonic()

    for i, trial in enumerate(pending):
        t0 = time.monotonic()

        try:
            result = validator.validate(
                config=trial.wf_config,
                model_factory=trial.factory,
                n_trials=n_total_trials,
            )
        except Exception as e:
            logger.error("[{}/{}] {} FAILED: {}", len(existing) + 1, total, trial.name, e)
            continue

        elapsed = time.monotonic() - t0
        times.append(elapsed)

        # Store result
        existing[trial.name] = {
            "params": _trial_params(trial, n_tickers),
            "mean_sharpe": result.mean_sharpe,
            "std_sharpe": result.std_sharpe,
            "pct_positive": result.pct_positive,
            "per_path_sharpe": result.per_path_sharpe,
            "deflated_sharpe": result.deflated_sharpe,
            "p_value": result.p_value,
            "accepted": result.accepted,
            "elapsed_seconds": round(elapsed, 1),
        }

        # Periodic save
        done = len(existing)
        if done % _SAVE_EVERY == 0 or i == len(pending) - 1:
            _save_tier_json(tier_num, existing, total, output_dir)

        # ETA
        avg_t = sum(times) / len(times)
        remaining = len(pending) - (i + 1)
        eta = str(timedelta(seconds=int(remaining * avg_t)))
        wall = str(timedelta(seconds=int(time.monotonic() - tier_start)))

        acc = "OK" if result.accepted else "--"
        logger.info(
            "[{}/{}] {} | Sharpe={:.3f} | DSR={:.3f} | {} | {:.0f}s | ETA {} | wall {}",
            done, total, trial.name,
            result.mean_sharpe, result.p_value,
            acc, elapsed, eta, wall,
        )

    # Final save
    _save_all_outputs(tier_num, existing, total, output_dir)
    return existing


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _save_tier_json(
    tier_num: int,
    configs: dict[str, dict[str, Any]],
    total: int,
    output_dir: Path,
) -> None:
    """Incremental JSON save (atomic write via temp file + rename)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"validation_tier{tier_num}_results.json"
    tmp_path = path.with_suffix(".tmp")
    data = {
        "tier": tier_num,
        "generated_at": datetime.now().isoformat(),
        "total_trials": total,
        "completed": len(configs),
        "configs": configs,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def _save_all_outputs(
    tier_num: int,
    configs: dict[str, dict[str, Any]],
    total: int,
    output_dir: Path,
) -> None:
    """Save JSON, Markdown summary, and per-path Parquet."""
    _save_tier_json(tier_num, configs, total, output_dir)
    _save_summary_md(tier_num, configs, output_dir)
    _save_paths_parquet(tier_num, configs, output_dir)


def _save_summary_md(
    tier_num: int,
    configs: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate ranked Markdown summary table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"validation_tier{tier_num}_summary.md"

    ranked = sorted(configs.items(), key=lambda x: x[1].get("deflated_sharpe", 0), reverse=True)

    n_accepted = sum(1 for _, r in ranked if r.get("accepted"))
    best_name = ranked[0][0] if ranked else "N/A"
    best_sharpe = ranked[0][1].get("mean_sharpe", 0) if ranked else 0

    lines = [
        f"# Tier {tier_num} Validation Results", "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} "
        f"| {len(configs)} configs | {n_accepted} accepted", "",
        f"> Best: **{best_name}** (Sharpe={best_sharpe:.3f})", "",
        "## Rankings (sorted by DSR p-value)", "",
        "| # | Config | Sharpe | Std | %+ | DSR | OK |",
        "|---|--------|--------|-----|-----|-----|-----|",
    ]
    for i, (name, r) in enumerate(ranked, 1):
        acc = "YES" if r.get("accepted") else "no"
        lines.append(
            f"| {i} | {name} | {r.get('mean_sharpe', 0):.3f} "
            f"| {r.get('std_sharpe', 0):.3f} "
            f"| {r.get('pct_positive', 0):.0%} "
            f"| {r.get('deflated_sharpe', 0):.3f} | {acc} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Summary saved to {}", path)


def _save_paths_parquet(
    tier_num: int,
    configs: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save per-path Sharpe values for distribution analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"validation_tier{tier_num}_paths.parquet"
    rows: list[dict[str, Any]] = []
    for name, r in configs.items():
        for pid, sharpe in enumerate(r.get("per_path_sharpe", [])):
            rows.append({
                "config": name,
                "path_id": pid,
                "sharpe": sharpe,
                "accepted": r.get("accepted", False),
            })
    if rows:
        pl.DataFrame(rows).write_parquet(path)
        logger.info("Per-path data: {} rows -> {}", len(rows), path)


# ---------------------------------------------------------------------------
# Estimate mode
# ---------------------------------------------------------------------------


def _run_estimate(
    validator: CPCVParameterValidator,
    n: int,
) -> None:
    """Time 1 config and print ETA for all tiers."""
    trial = _t("estimate_probe")
    logger.info("Running 1 config for time estimation...")

    t0 = time.monotonic()
    validator.validate(config=trial.wf_config, model_factory=trial.factory, n_trials=1)
    elapsed = time.monotonic() - t0

    logger.info("Single config: {:.1f}s", elapsed)

    t1 = len(build_tier1(n))
    t2 = len(build_tier2(n))
    # Tier 3 count depends on results, estimate 170
    t3_est = 170

    for label, count in [("Tier 1", t1), ("Tier 2", t2), ("Tier 3 (est.)", t3_est)]:
        hours = count * elapsed / 3600
        logger.info("  {} ({} configs): {:.1f}h", label, count, hours)

    total = t1 + t2 + t3_est
    logger.info("  Total ({} configs): {:.1f}h ({:.1f} days at 18h/day)",
                total, total * elapsed / 3600, total * elapsed / 3600 / 18)


# ---------------------------------------------------------------------------
# Public API (backwards-compatible)
# ---------------------------------------------------------------------------


def run_improvement_validation(
    ohlcv: pl.DataFrame,
    tickers: list[str],
    benchmark_ticker: str = "SPY",
    output_dir: str = "data/outputs",
    subset: str = "all",
) -> dict[str, ValidationResult]:
    """Legacy API: runs Tier 1 only (backwards compatible).

    For the full 3-tier grid, use the CLI with ``--tier``.
    """
    out = Path(output_dir)
    n = len(tickers)
    validator = CPCVParameterValidator(
        ohlcv=ohlcv, tickers=tickers, benchmark_ticker=benchmark_ticker,
    )
    trials = build_tier1(n)
    existing_fps = _load_existing_fps(out)
    trials = _dedup(trials, n, existing_fps)
    run_tier(1, trials, validator, n, out)
    # Return empty dict for API compat — real results are in JSON
    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_main() -> None:
    """Entry point for ``python -m src.backtest.run_validation``."""
    tier_str = "1"
    if "--tier" in sys.argv:
        idx = sys.argv.index("--tier")
        if idx + 1 < len(sys.argv):
            tier_str = sys.argv[idx + 1]

    # Legacy --subset support
    if "--subset" in sys.argv:
        idx = sys.argv.index("--subset")
        if idx + 1 < len(sys.argv):
            tier_str = "1"  # legacy mode runs tier 1

    # Load data
    from src.backtest.run_benchmark import _filter_oos_period, _load_ohlcv_from_postgres
    from src.config import load_benchmark, load_tickers

    tickers = load_tickers()
    benchmark = load_benchmark()
    n = len(tickers)

    # Dry run: print grid size
    if "--dry-run" in sys.argv:
        for t_num, builder in [(1, build_tier1), (2, build_tier2)]:
            trials = builder(n)
            logger.info("Tier {}: {} configs", t_num, len(trials))
            for t in trials[:5]:
                logger.info("  {}", t.name)
            if len(trials) > 5:
                logger.info("  ... and {} more", len(trials) - 5)
        logger.info("Tier 3: ~170 (adaptive, depends on Tier 1+2 results)")
        return

    logger.info("Loading OHLCV data...")
    ohlcv = _load_ohlcv_from_postgres(tickers, benchmark)
    ohlcv = _filter_oos_period(ohlcv, n_years=10)

    validator = CPCVParameterValidator(
        ohlcv=ohlcv, tickers=tickers, benchmark_ticker=benchmark,
    )

    # Estimate mode
    if "--estimate" in sys.argv:
        _run_estimate(validator, n)
        return

    # Determine tiers to run
    if tier_str == "all":
        tiers_to_run = [1, 2, 3]
    else:
        tiers_to_run = [int(tier_str)]

    for t_num in tiers_to_run:
        if t_num == 1:
            trials = build_tier1(n)
        elif t_num == 2:
            trials = build_tier2(n)
        elif t_num == 3:
            trials = build_tier3(n)
        else:
            logger.error("Unknown tier: {}", t_num)
            continue

        # Cross-tier dedup
        existing_fps = _load_existing_fps(_OUTPUT_DIR)
        trials = _dedup(trials, n, existing_fps)

        logger.info("Tier {}: {} configs after dedup", t_num, len(trials))
        run_tier(t_num, trials, validator, n)

    # Final cross-tier summary
    all_results = _load_all_results()
    if all_results:
        logger.info("=== FINAL RANKINGS (all tiers) ===")
        n_accepted = sum(1 for r in all_results if r.get("accepted"))
        logger.info("{}/{} configs accepted across all tiers", n_accepted, len(all_results))
        for i, r in enumerate(all_results[:10], 1):
            acc = "OK" if r.get("accepted") else "--"
            logger.info(
                "  #{}: {} | Sharpe={:.3f} | DSR={:.3f} | {}",
                i, r["name"], r.get("mean_sharpe", 0), r.get("deflated_sharpe", 0), acc,
            )


if __name__ == "__main__":
    _cli_main()
