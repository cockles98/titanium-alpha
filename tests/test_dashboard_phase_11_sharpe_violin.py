"""Tests for dashboard Phase 11 — Sharpe violin across CPCV paths."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl

from src.dashboard.app import (
    _chart_sharpe_violin,
    _compute_sharpe_per_path,
    _expected_max_sharpe,
    _probabilistic_sharpe_ratio,
)


def _synth_paths(
    n_paths: int = 15,
    n_days: int = 30,
    seed: int = 0,
    config: str = "champion",
    sharpes: list[float] | None = None,
) -> pl.DataFrame:
    """Build a long-format cpcv_paths frame with one Sharpe per path."""
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    rows: list[dict] = []
    for pid in range(n_paths):
        if sharpes is not None and pid < len(sharpes):
            sharpe = float(sharpes[pid])
        else:
            sharpe = float(rng.uniform(0.3, 1.2))
        rets = rng.normal(0.0005, 0.01, size=n_days - 1)
        eq = np.empty(n_days)
        eq[0] = 1.0
        for i, r in enumerate(rets, start=1):
            eq[i] = eq[i - 1] * (1.0 + r)
        for d, v in zip(dates, eq):
            rows.append({
                "config": config,
                "path_id": pid,
                "date": d,
                "equity": float(v),
                "sharpe": sharpe,
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# _compute_sharpe_per_path
# ---------------------------------------------------------------------------


def test_compute_sharpe_per_path_returns_one_per_path_id():
    sharpes = [0.1, 0.2, 0.3, 0.4, 0.5]
    df = _synth_paths(n_paths=5, n_days=6, sharpes=sharpes)
    out = _compute_sharpe_per_path(df)
    assert out == sharpes


def test_compute_sharpe_per_path_sorts_by_path_id():
    sharpes = [0.9, 0.1, 0.5]
    df = _synth_paths(n_paths=3, n_days=5, sharpes=sharpes)
    shuffled = df.sample(fraction=1.0, seed=42)
    out = _compute_sharpe_per_path(shuffled)
    assert out == sharpes


def test_compute_sharpe_per_path_handles_none_and_empty():
    assert _compute_sharpe_per_path(None) == []
    empty = pl.DataFrame(
        schema={"path_id": pl.Int64, "sharpe": pl.Float64}
    )
    assert _compute_sharpe_per_path(empty) == []


def test_compute_sharpe_per_path_requires_columns():
    df = pl.DataFrame({"path_id": [0, 1]})
    assert _compute_sharpe_per_path(df) == []


def test_compute_sharpe_per_path_drops_nan_and_inf():
    df = pl.DataFrame({
        "path_id": [0, 0, 1, 1, 2, 2],
        "sharpe": [0.5, 0.5, float("nan"), float("nan"), float("inf"), float("inf")],
    })
    out = _compute_sharpe_per_path(df)
    assert out == [0.5]


# ---------------------------------------------------------------------------
# _expected_max_sharpe
# ---------------------------------------------------------------------------


def test_expected_max_sharpe_grows_with_n_trials():
    low = _expected_max_sharpe(sigma=0.5, n_trials=10)
    high = _expected_max_sharpe(sigma=0.5, n_trials=500)
    assert low is not None
    assert high is not None
    assert high > low > 0


def test_expected_max_sharpe_scales_with_sigma():
    small = _expected_max_sharpe(sigma=0.1, n_trials=100)
    big = _expected_max_sharpe(sigma=1.0, n_trials=100)
    assert small is not None and big is not None
    assert big > small
    # Linear-in-sigma because e_max_z is constant for fixed n_trials
    assert math.isclose(big / small, 10.0, rel_tol=1e-6)


def test_expected_max_sharpe_handles_degenerate_inputs():
    assert _expected_max_sharpe(sigma=0.5, n_trials=1) is None
    assert _expected_max_sharpe(sigma=0.0, n_trials=100) is None
    assert _expected_max_sharpe(sigma=-1.0, n_trials=100) is None
    assert _expected_max_sharpe(sigma=0.5, n_trials=0) is None


# ---------------------------------------------------------------------------
# _probabilistic_sharpe_ratio
# ---------------------------------------------------------------------------


def test_psr_is_probability():
    psr = _probabilistic_sharpe_ratio(observed_sharpe=0.7, n_obs=2500)
    assert psr is not None
    assert 0.0 <= psr <= 1.0


def test_psr_grows_with_observations():
    psr_short = _probabilistic_sharpe_ratio(observed_sharpe=0.5, n_obs=50)
    psr_long = _probabilistic_sharpe_ratio(observed_sharpe=0.5, n_obs=5000)
    assert psr_short is not None and psr_long is not None
    assert psr_long > psr_short


def test_psr_equal_benchmark_returns_half():
    psr = _probabilistic_sharpe_ratio(
        observed_sharpe=0.4, n_obs=1000, sharpe_benchmark=0.4,
    )
    assert psr is not None
    assert abs(psr - 0.5) < 1e-9


def test_psr_returns_none_when_insufficient_data():
    assert _probabilistic_sharpe_ratio(observed_sharpe=0.5, n_obs=1) is None


# ---------------------------------------------------------------------------
# _chart_sharpe_violin — graceful degradation
# ---------------------------------------------------------------------------


def test_chart_returns_none_for_none_input():
    assert _chart_sharpe_violin(None) is None


def test_chart_returns_none_for_empty_frame():
    empty = pl.DataFrame(
        schema={
            "config": pl.Utf8,
            "path_id": pl.Int64,
            "date": pl.Date,
            "equity": pl.Float64,
            "sharpe": pl.Float64,
        }
    )
    assert _chart_sharpe_violin(empty) is None


def test_chart_returns_none_when_required_columns_missing():
    df = pl.DataFrame({"config": ["champion"], "path_id": [0]})
    assert _chart_sharpe_violin(df) is None


def test_chart_returns_none_when_fewer_than_two_paths():
    df = _synth_paths(n_paths=1, n_days=10)
    assert _chart_sharpe_violin(df) is None


def test_chart_tolerates_missing_config_column():
    df = _synth_paths(n_paths=5, n_days=10).drop("config")
    fig = _chart_sharpe_violin(df)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# _chart_sharpe_violin — structural assertions
# ---------------------------------------------------------------------------


def test_chart_contains_single_violin_trace():
    df = _synth_paths(n_paths=15, n_days=20)
    fig = _chart_sharpe_violin(df)
    violins = [t for t in fig.data if isinstance(t, go.Violin)]
    assert len(violins) == 1


def test_chart_violin_shows_box_and_mean_line():
    df = _synth_paths(n_paths=15, n_days=20)
    fig = _chart_sharpe_violin(df)
    violin = next(t for t in fig.data if isinstance(t, go.Violin))
    assert violin.box.visible is True
    assert violin.meanline.visible is True
    assert violin.points == "all"


def test_chart_violin_y_matches_per_path_sharpe():
    sharpes = [0.1, 0.3, 0.5, 0.7, 0.9]
    df = _synth_paths(n_paths=5, n_days=12, sharpes=sharpes)
    fig = _chart_sharpe_violin(df)
    violin = next(t for t in fig.data if isinstance(t, go.Violin))
    assert sorted(float(y) for y in violin.y) == sorted(sharpes)


def test_chart_uses_dark_theme():
    df = _synth_paths(n_paths=4, n_days=10)
    fig = _chart_sharpe_violin(df)
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_chart_title_reports_path_count_and_config():
    df = _synth_paths(n_paths=15, n_days=20, config="my_cfg")
    fig = _chart_sharpe_violin(df)
    title = fig.layout.title.text or ""
    assert "15" in title
    assert "my_cfg" in title


def test_chart_config_filter_selects_single_config():
    df_a = _synth_paths(n_paths=4, n_days=10, config="a", seed=1)
    df_b = _synth_paths(n_paths=4, n_days=10, config="b", seed=2)
    both = pl.concat([df_a, df_b])
    fig = _chart_sharpe_violin(both, config_filter="b")
    assert fig is not None
    title = fig.layout.title.text or ""
    assert "(b)" in title


# ---------------------------------------------------------------------------
# _chart_sharpe_violin — reference lines and annotations
# ---------------------------------------------------------------------------


def _shape_annotation_texts(fig: go.Figure) -> list[str]:
    """Collect annotation texts from both top-level and hline shapes."""
    return [a.text or "" for a in (fig.layout.annotations or [])]


def test_chart_draws_oos_hline_with_gold_color():
    df = _synth_paths(n_paths=5, n_days=15)
    fig = _chart_sharpe_violin(df, oos_sharpe=0.77)
    texts = " | ".join(_shape_annotation_texts(fig))
    assert "Walk-forward OOS" in texts
    assert "0.77" in texts
    # The OOS hline is rendered as a shape with the accent gold color.
    gold_shapes = [
        s for s in (fig.layout.shapes or [])
        if getattr(s.line, "color", None) == "#FFB300"
    ]
    assert gold_shapes


def test_chart_draws_dsr_hline_when_n_trials_provided():
    df = _synth_paths(n_paths=15, n_days=20)
    fig = _chart_sharpe_violin(df, n_trials=400)
    texts = " | ".join(_shape_annotation_texts(fig))
    assert "DSR expected max" in texts


def test_chart_skips_dsr_when_n_trials_is_one():
    df = _synth_paths(n_paths=15, n_days=20)
    fig = _chart_sharpe_violin(df, n_trials=1)
    texts = " | ".join(_shape_annotation_texts(fig))
    assert "DSR expected max" not in texts


def test_chart_dsr_threshold_explicit_overrides_n_trials():
    df = _synth_paths(n_paths=15, n_days=20)
    fig = _chart_sharpe_violin(df, dsr_threshold=0.42, n_trials=None)
    texts = " | ".join(_shape_annotation_texts(fig))
    assert "0.42" in texts


def test_chart_stats_annotation_reports_mean_std_positive_and_psr():
    sharpes = [0.1, 0.2, 0.3, 0.4, 0.5, -0.1]
    df = _synth_paths(n_paths=6, n_days=12, sharpes=sharpes)
    fig = _chart_sharpe_violin(df, psr=0.91)
    texts = " | ".join(_shape_annotation_texts(fig))
    assert "Paths" in texts and "6" in texts
    assert "Mean" in texts
    assert "Std" in texts
    assert "% positive" in texts
    assert "PSR" in texts and "0.91" in texts


def test_chart_stats_annotation_percent_positive_is_accurate():
    sharpes = [0.1, 0.2, 0.3, -0.1]
    df = _synth_paths(n_paths=4, n_days=10, sharpes=sharpes)
    fig = _chart_sharpe_violin(df)
    texts = " | ".join(_shape_annotation_texts(fig))
    # 3 positive of 4 = 75%
    assert "75%" in texts


def test_chart_hides_oos_line_when_not_provided():
    df = _synth_paths(n_paths=5, n_days=10)
    fig = _chart_sharpe_violin(df)
    texts = " | ".join(_shape_annotation_texts(fig))
    assert "Walk-forward OOS" not in texts
