"""Phase 3.2 pending - analyze sources_cited distribution in debate_history.json.

Reads the latest ``debate_history.json`` produced by ``make decide`` and reports
per-ticker and aggregate statistics on ``sources_cited`` — the citations
that the Fundamentalist agent attached to its report using the RAG.

Honest interpretation: an empty ``sources_cited`` means the Fundamentalist
either (a) could not find relevant articles in ChromaDB, or (b) the LLM
response did not surface any citations. Both are degraded states compared
to a fully grounded debate.
"""
from __future__ import annotations

import io
import json
import statistics
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
DEBATE_PATH = OUTPUT_DIR / "debate_history.json"
DECISIONS_PATH = OUTPUT_DIR / "decisions.json"
REPORT_PATH = OUTPUT_DIR / "phase3_2b_sources_report.md"


def _extract_fundamentalist_sources(reports: list) -> list[str]:
    """Find the fundamental agent's sources_cited in a list of reports.

    The graph names the agent ``"fundamental"`` (see src/agents/graph.py).
    Accept a few variants in case older history files were saved with
    different labels.
    """
    for r in reports or []:
        if isinstance(r, dict) and r.get("agent") in {
            "fundamental", "fundamentalist"
        }:
            return r.get("sources_cited", []) or []
    return []


def main() -> int:
    if not DEBATE_PATH.exists():
        print(f"[phase3.2b] MISSING: {DEBATE_PATH}", flush=True)
        print("[phase3.2b] Run `make decide` first.", flush=True)
        return 1

    with DEBATE_PATH.open("r", encoding="utf-8") as f:
        debate = json.load(f)

    decisions_by_ticker: dict[str, dict] = {}
    if DECISIONS_PATH.exists():
        with DECISIONS_PATH.open("r", encoding="utf-8") as f:
            decisions_raw = json.load(f)
        for d in decisions_raw.get("decisions", []) or []:
            decisions_by_ticker[d["ticker"]] = d

    n_tickers = len(debate)
    print(f"[phase3.2b] Loaded {n_tickers} ticker states from {DEBATE_PATH.name}", flush=True)

    per_ticker: list[dict] = []
    for tkr, state in debate.items():
        reports = state.get("reports", []) or []
        sources = _extract_fundamentalist_sources(reports)
        news_ctx = state.get("news_context", []) or []
        decision = decisions_by_ticker.get(tkr, {})
        per_ticker.append({
            "ticker": tkr,
            "action": decision.get("action", "?"),
            "confidence": decision.get("confidence", 0.0),
            "n_news_retrieved": len(news_ctx),
            "n_sources_cited": len(sources),
            "sample_sources": sources[:2] if sources else [],
        })

    n_with_news = sum(1 for r in per_ticker if r["n_news_retrieved"] > 0)
    n_grounded = sum(1 for r in per_ticker if r["n_sources_cited"] > 0)
    n_ungrounded = n_tickers - n_grounded
    avg_sources = (
        statistics.mean(r["n_sources_cited"] for r in per_ticker if r["n_sources_cited"] > 0)
        if n_grounded > 0 else 0.0
    )
    total_sources = sum(r["n_sources_cited"] for r in per_ticker)
    avg_news = (
        statistics.mean(r["n_news_retrieved"] for r in per_ticker if r["n_news_retrieved"] > 0)
        if n_with_news > 0 else 0.0
    )

    print(
        f"[phase3.2b] Retrieved news: {n_with_news}/{n_tickers}; "
        f"Grounded (≥1 source): {n_grounded}/{n_tickers}; "
        f"Total citations: {total_sources}",
        flush=True,
    )

    action_breakdown: dict[str, dict] = {}
    for r in per_ticker:
        a = r["action"]
        slot = action_breakdown.setdefault(a, {"count": 0, "grounded": 0, "total_sources": 0})
        slot["count"] += 1
        if r["n_sources_cited"] > 0:
            slot["grounded"] += 1
            slot["total_sources"] += r["n_sources_cited"]

    lines = [
        "# Phase 3.2b — RAG grounding analysis (`sources_cited`)",
        "",
        "This report analyzes the Fundamentalist agent's citation behavior after",
        "`make decide` ran with the RAG populated (172 articles embedded in the",
        "ChromaDB on this machine — see Phase 3.2 results). Data source:",
        "`data/outputs/debate_history.json` + `decisions.json`.",
        "",
        "## Pipeline stages",
        "",
        "1. **RAG retrieval** — `FinancialRAG.retrieve(ticker, query)` returns",
        "   a list of news snippets (`news_context` in the LangGraph state).",
        "2. **Fundamentalist report** — the agent receives the snippets and is",
        "   instructed to cite source titles in its `sources_cited` field of",
        "   the structured Pydantic output.",
        "3. **`sources_cited = []`** happens when either retrieval returned",
        "   nothing OR the LLM did not echo any source titles.",
        "",
        "## Summary",
        "",
        f"- **Total tickers decided:** {n_tickers}",
        f"- **Retrieved any news:** {n_with_news} ({100*n_with_news/n_tickers:.1f}%)",
        f"- **Mean news hits when retrieved:** {avg_news:.2f}",
        f"- **Grounded (≥ 1 source cited):** {n_grounded} ({100*n_grounded/n_tickers:.1f}%)",
        f"- **Ungrounded (0 sources):** {n_ungrounded} ({100*n_ungrounded/n_tickers:.1f}%)",
        f"- **Total citations emitted:** {total_sources}",
        f"- **Mean citations per grounded decision:** {avg_sources:.2f}",
        "",
        "## Breakdown by action",
        "",
        "| Action | Count | Grounded | Avg citations (when grounded) |",
        "|---|---|---|---|",
    ]
    for action, slot in sorted(action_breakdown.items()):
        avg = slot["total_sources"] / slot["grounded"] if slot["grounded"] > 0 else 0.0
        lines.append(
            f"| {action} | {slot['count']} | "
            f"{slot['grounded']}/{slot['count']} | "
            f"{avg:.2f} |"
        )

    lines += [
        "",
        "## Per-ticker detail (top 10 by citation count)",
        "",
        "| Ticker | Action | Confidence | News | Cited | Sample |",
        "|---|---|---|---|---|---|",
    ]
    top = sorted(per_ticker, key=lambda r: r["n_sources_cited"], reverse=True)[:10]
    for r in top:
        sample = (r["sample_sources"][0][:55] + "…") if r["sample_sources"] else "—"
        lines.append(
            f"| {r['ticker']} | {r['action']} | "
            f"{r['confidence']:.2f} | {r['n_news_retrieved']} | "
            f"{r['n_sources_cited']} | {sample} |"
        )

    lines += [
        "",
        "## Before/after interpretation",
        "",
        "**Baseline (no RAG populated):** `sources_cited = []` for every",
        "decision — the Fundamentalist cannot cite articles that don't exist",
        "in ChromaDB. This is the state on a fresh install before `make ingest`.",
        "",
        "**After RAG populated (this run, 172 articles):** the grounding rate",
        "measures how effectively the pipeline converts available ChromaDB",
        "documents into Fundamentalist citations. Ungrounded decisions ≥ 40%",
        "typically indicates sparse ticker-level coverage (e.g. this run has",
        "only 3 articles for META/TSLA, 5-7 for NVDA/BAC). The Cockles RAG has",
        "172 articles total vs. 14 326 on the felip machine — ticker-level",
        "sparsity is expected.",
        "",
        "## Action items",
        "",
        "- Increase `backfill_historical` time budget on this machine",
        "  (current: 172 articles; target for full grounding: ≥ 5 articles per",
        "  ticker = 260 articles).",
        "- Run `make ingest` nightly in production to refresh the RAG — stale",
        "  articles still retrieve but grounding degrades as news becomes",
        "  irrelevant.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[phase3.2b] Report written to {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
