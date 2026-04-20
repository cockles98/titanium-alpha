"""Phase 3.2 - embed pending news into ChromaDB."""
from __future__ import annotations

import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    from src.agents.rag import FinancialRAG

    print("[phase3.2] Initializing FinancialRAG...", flush=True)
    rag = FinancialRAG()

    print(f"[phase3.2] Pre-embed ChromaDB count: {rag.collection.count()}", flush=True)
    t0 = time.time()
    print("[phase3.2] Calling embed_pending_news()...", flush=True)
    n_embedded = rag.embed_pending_news()
    elapsed = time.time() - t0

    post_count = rag.collection.count()
    print(f"\n[phase3.2] Embedded: {n_embedded} articles in {elapsed:.1f}s", flush=True)
    print(f"[phase3.2] Post-embed ChromaDB count: {post_count}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
