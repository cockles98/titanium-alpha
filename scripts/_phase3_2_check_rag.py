"""Phase 3.2 - check RAG/news state on Cockles machine."""
from __future__ import annotations

import io
import sys

# Force stdout UTF-8 on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    print("[phase3.2] Loading deps...", flush=True)
    import polars as pl
    from src.utils.db import get_chroma_client, get_postgres_engine

    print("[phase3.2] Testing Postgres...", flush=True)
    try:
        eng = get_postgres_engine()
        df = pl.read_database(
            "SELECT COUNT(*) as total FROM financial_news",
            eng,
        )
        total_news = int(df["total"][0])
        print(f"  financial_news rows (PG): {total_news}", flush=True)

        if total_news > 0:
            df_status = pl.read_database(
                "SELECT embedding_status, COUNT(*) as c "
                "FROM financial_news GROUP BY embedding_status",
                eng,
            )
            rows = df_status.to_dicts()
            print(f"  embedding_status breakdown: {rows}", flush=True)

            df_dates = pl.read_database(
                "SELECT MIN(date) as min_d, MAX(date) as max_d FROM financial_news",
                eng,
            )
            min_d = df_dates["min_d"][0]
            max_d = df_dates["max_d"][0]
            print(f"  date range: {min_d} to {max_d}", flush=True)

            df_tkr = pl.read_database(
                "SELECT ticker, COUNT(*) as c FROM financial_news "
                "GROUP BY ticker ORDER BY c DESC LIMIT 10",
                eng,
            )
            print(f"  top tickers: {df_tkr.to_dicts()}", flush=True)
    except Exception as exc:
        print(f"  PG ERROR: {exc}", flush=True)

    print("\n[phase3.2] Testing ChromaDB...", flush=True)
    try:
        cc = get_chroma_client()
        print(f"  heartbeat: {cc.heartbeat()}", flush=True)
        col = cc.get_or_create_collection(name="financial_news")
        chroma_count = col.count()
        print(f"  financial_news count (Chroma): {chroma_count}", flush=True)
    except Exception as exc:
        print(f"  Chroma ERROR: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
