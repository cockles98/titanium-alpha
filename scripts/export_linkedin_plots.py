"""Export LinkedIn carousel plots as standalone PNGs from the dashboard charts.

Regenerates the two CAPM-family plots with the corrected Jensen's alpha
(``alpha_daily = intercept - rf_daily * (1 - beta)`` — already fixed in
``src/dashboard/app.py``) and exports two previously-unexported charts that
are buried inside full-tab screenshots:

    * Decision-flow Sankey  (Performance tab)
    * PatchTST quantile fan (Microstructure tab)

All four are rendered with the exact dashboard chart functions so colours,
typography and layout match the rest of ``docs/images/benchmark graphs/``.

Run from the repo root::

    python scripts/export_linkedin_plots.py

Requires: plotly, kaleido, pandas, pyarrow, numpy (no polars needed — inputs
are built with pandas and plain dicts).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# The dashboard chart labels contain Greek letters (alpha/beta); make sure the
# diagnostic prints survive a cp1252 Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "outputs"
OUT = ROOT / "docs" / "images" / "benchmark graphs"
OUT.mkdir(parents=True, exist_ok=True)

# Import the real dashboard chart builders (module-level Streamlit code is all
# guarded behind ``if __name__ == "__main__"``, so importing is side-effect free).
from src.dashboard.app import (  # noqa: E402
    _chart_benchmark_equity,
    _chart_capm_scatter,
    _chart_decision_flow_sankey,
    _chart_quantile_fan,
    _chart_rolling_market_relationship,
)

WIDTH = 1368
SCALE = 2


def _save(fig, name: str, default_height: int) -> None:
    """Write a Plotly figure to ``OUT/name`` at the dashboard's native size."""
    height = int(fig.layout.height) if fig.layout.height else default_height
    path = OUT / name
    fig.write_image(str(path), width=WIDTH, height=height, scale=SCALE)
    print(f"  saved {name}  ({WIDTH}x{height} @{SCALE}x)  ->  {path.stat().st_size:,} bytes")


def export_capm_family() -> None:
    """Regenerate CAPM Scatter + Market Relationship with corrected alpha."""
    eq = pd.read_parquet(DATA / "benchmark_equity.parquet")
    needed = {"date", "portfolio_value", "benchmark_value"}
    missing = needed - set(eq.columns)
    if missing:
        raise SystemExit(f"benchmark_equity.parquet missing columns: {missing}")
    print(f"[CAPM family] equity rows={len(eq)}  cols={list(eq.columns)}")

    fig_capm = _chart_capm_scatter(eq)
    if fig_capm is None:
        raise SystemExit("CAPM scatter returned None")
    # Pull the computed alpha straight out of the OLS-fit trace name for a sanity print.
    fit_name = next((t.name for t in fig_capm.data if t.name and "α_ann" in t.name), "")
    print(f"  CAPM fit annotation: {fit_name}")
    _save(fig_capm, "CAPM Scatter vs SPY.png", default_height=500)

    fig_mr = _chart_rolling_market_relationship(eq)
    if fig_mr is None:
        raise SystemExit("Market Relationship returned None")
    titles = [a.text for a in (fig_mr.layout.annotations or []) if a.text]
    alpha_title = next((t for t in titles if "Alpha" in t), "")
    print(f"  Market Relationship alpha panel: {alpha_title}")
    _save(fig_mr, "Market Relationship.png", default_height=600)


def export_equity_log() -> None:
    """Export the equity curve in log scale (linear version stays untouched).

    Log scale compresses the portfolio-vs-SPY gap and emphasises the smoother
    ride, avoiding the "ends below SPY" misread of the linear chart.
    """
    eq = pd.read_parquet(DATA / "benchmark_equity.parquet")
    print(f"[Equity log] rows={len(eq)}")
    fig = _chart_benchmark_equity(eq, log_scale=True)
    _save(fig, "Equity Curve (log).png", default_height=450)


def export_sankey() -> None:
    """Export the agentic decision-flow Sankey as a standalone PNG."""
    with open(DATA / "decisions.json", encoding="utf-8") as f:
        decisions = json.load(f)
    meta = decisions.get("metadata", {})
    print(
        f"[Sankey] n_buy={meta.get('n_buy')} n_hold={meta.get('n_hold')} "
        f"n_sell={meta.get('n_sell')} invested={meta.get('invested_fraction')}"
    )
    fig = _chart_decision_flow_sankey(decisions)
    if fig is None:
        raise SystemExit("Sankey returned None")
    # The dashboard renders at height=420 inside a scrollable Streamlit
    # container; a static export needs more vertical room + bottom margin so
    # the BUY/SELL/Excluded nodes at the edges of the diagram are not clipped.
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=70, b=70))
    _save(fig, "Decision Flow Sankey.png", default_height=640)


def _pick_forecast_ticker(fc: pd.DataFrame, id_col: str) -> str:
    """Prefer NVDA (matches the dashboard screenshot), else first ticker."""
    tickers = sorted(fc[id_col].unique().tolist())
    for pref in ("NVDA", "AAPL", "MSFT"):
        if pref in tickers:
            return pref
    return tickers[0]


def export_quantile_fan() -> None:
    """Export the PatchTST quantile fan for a representative ticker."""
    fc = pd.read_parquet(DATA / "forecast.parquet")
    id_col = "ticker" if "ticker" in fc.columns else "unique_id"
    ticker = _pick_forecast_ticker(fc, id_col)
    rows = fc[fc[id_col] == ticker].to_dict("records")
    print(f"[Fan] ticker={ticker}  rows={len(rows)}  cols={list(fc.columns)}")

    last_close = None
    pred_path = DATA / "predictions.parquet"
    if pred_path.exists():
        pr = pd.read_parquet(pred_path)
        if "last_close" in pr.columns and (pr["ticker"] == ticker).any():
            last_close = float(pr.loc[pr["ticker"] == ticker, "last_close"].iloc[0])
            print(f"  last_close({ticker}) = {last_close}")

    fig = _chart_quantile_fan(rows, ticker, last_close=last_close)
    _save(fig, "PatchTST Quantile Forecast.png", default_height=500)


def main() -> None:
    print("Exporting LinkedIn plots ->", OUT)
    export_capm_family()
    export_equity_log()
    export_sankey()
    export_quantile_fan()
    print("Done.")


if __name__ == "__main__":
    main()
