"""Tests for dashboard Phase 10 — CPCV-OOS path spaghetti chart."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl

from src.dashboard.app import _chart_cpcv_spaghetti


def _synth_paths(
    n_paths: int = 15,
    n_days: int = 40,
    seed: int = 0,
    config: str = "champion",
) -> pl.DataFrame:
    """Build a long-format cpcv_paths frame with normalised equity curves."""
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    rows: list[dict] = []
    for pid in range(n_paths):
        rets = rng.normal(0.0005, 0.01, size=n_days - 1)
        eq = np.empty(n_days)
        eq[0] = 1.0
        for i, r in enumerate(rets, start=1):
            eq[i] = eq[i - 1] * (1.0 + r)
        sharpe = float(rng.uniform(0.3, 1.2))
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
# Graceful degradation
# ---------------------------------------------------------------------------


def test_chart_returns_none_for_none_input():
    assert _chart_cpcv_spaghetti(None) is None


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
    assert _chart_cpcv_spaghetti(empty) is None


def test_chart_returns_none_when_required_columns_missing():
    df = pl.DataFrame({"config": ["champion"], "path_id": [0]})
    assert _chart_cpcv_spaghetti(df) is None


def test_chart_tolerates_missing_config_column():
    # When the parquet pre-dates the config label, the helper should still
    # return a figure using the rows as-is.
    df = _synth_paths(n_paths=3, n_days=10).drop("config")
    fig = _chart_cpcv_spaghetti(df)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Structural assertions on the figure
# ---------------------------------------------------------------------------


def test_chart_contains_one_line_per_path_plus_iqr_and_median():
    df = _synth_paths(n_paths=15, n_days=30)
    fig = _chart_cpcv_spaghetti(df)
    # 15 path traces + q75 (helper for fill) + q25 (fill="tonexty") + median
    assert len(fig.data) == 15 + 3


def test_chart_median_trace_is_gold_and_bold():
    df = _synth_paths(n_paths=5, n_days=20)
    fig = _chart_cpcv_spaghetti(df)
    median = next(t for t in fig.data if t.name == "Median")
    assert median.line.color == "#FFB300"
    assert median.line.width >= 2.0


def test_chart_iqr_trace_has_named_fill():
    df = _synth_paths(n_paths=6, n_days=20)
    fig = _chart_cpcv_spaghetti(df)
    iqr = next(t for t in fig.data if (t.name or "").startswith("IQR"))
    assert iqr.fill == "tonexty"


def test_chart_uses_dark_theme():
    df = _synth_paths(n_paths=4, n_days=20)
    fig = _chart_cpcv_spaghetti(df)
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_chart_annotation_shows_sharpe_range():
    rng = np.random.default_rng(3)
    start = date(2022, 1, 1)
    dates = [start + timedelta(days=i) for i in range(20)]
    rows: list[dict] = []
    sharpes = [0.30, 0.75, 1.10]
    for pid, s in enumerate(sharpes):
        for i, d in enumerate(dates):
            rows.append({
                "config": "champion",
                "path_id": pid,
                "date": d,
                "equity": float(1.0 + 0.001 * i + 0.0001 * pid),
                "sharpe": s,
            })
    df = pl.DataFrame(rows)
    fig = _chart_cpcv_spaghetti(df)
    annotations = [a.text or "" for a in (fig.layout.annotations or [])]
    joined = " | ".join(annotations)
    assert "0.30" in joined
    assert "1.10" in joined
    assert "Sharpe" in joined


def test_chart_title_reports_path_count_and_config():
    df = _synth_paths(n_paths=15, n_days=20, config="my_cfg")
    fig = _chart_cpcv_spaghetti(df)
    title = fig.layout.title.text or ""
    assert "15" in title
    assert "my_cfg" in title


def test_chart_config_filter_picks_single_config():
    df = _synth_paths(n_paths=4, n_days=15, config="a")
    df_b = _synth_paths(n_paths=4, n_days=15, config="b", seed=99)
    both = pl.concat([df, df_b])
    fig = _chart_cpcv_spaghetti(both, config_filter="b")
    assert fig is not None
    title = fig.layout.title.text or ""
    # Title should report 4 paths under config b.
    assert "(b)" in title


def test_chart_handles_missing_sharpe_column():
    df = _synth_paths(n_paths=3, n_days=10).drop("sharpe")
    fig = _chart_cpcv_spaghetti(df)
    assert fig is not None
    annotations = [a.text or "" for a in (fig.layout.annotations or [])]
    joined = " | ".join(annotations)
    assert "not available" in joined


def test_chart_paths_start_at_same_base():
    df = _synth_paths(n_paths=5, n_days=12)
    fig = _chart_cpcv_spaghetti(df)
    # First 5 traces are the individual paths in path_id order.
    for trace in fig.data[:5]:
        ys = list(trace.y)
        # Each path's equity starts at 1.0 (baseline normalisation).
        assert abs(ys[0] - 1.0) < 1e-6


def test_chart_iqr_band_encloses_median():
    df = _synth_paths(n_paths=15, n_days=20)
    fig = _chart_cpcv_spaghetti(df)
    # The IQR trio are always the last three traces.
    q75, q25, med = fig.data[-3], fig.data[-2], fig.data[-1]
    assert med.name == "Median"
    for y25, y50, y75 in zip(q25.y, med.y, q75.y):
        assert y25 <= y50 <= y75 + 1e-12
