"""Phase 3.2 - smoke-test RAG retrieval with current embedded corpus."""
from __future__ import annotations

import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()


TICKERS_TO_TEST = ["NVDA", "SPY", "BAC", "META", "TSLA"]


def main() -> int:
    from src.agents.rag import FinancialRAG

    rag = FinancialRAG()
    print(f"[phase3.2] Collection count: {rag.collection.count()}\n", flush=True)

    for tkr in TICKERS_TO_TEST:
        t0 = time.time()
        results = rag.retrieve(
            tkr,
            f"{tkr} financial outlook earnings market analysis",
            top_k=5,
            max_age_days=720,
        )
        elapsed_ms = (time.time() - t0) * 1000
        print(f"[{tkr}] hits={len(results)} latency={elapsed_ms:.0f}ms", flush=True)
        for i, snippet in enumerate(results[:3]):
            print(f"    {i+1}. {snippet[:120]}", flush=True)
        print(flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
