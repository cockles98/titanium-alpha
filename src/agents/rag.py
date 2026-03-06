"""Financial RAG (Retrieval-Augmented Generation) module.

Embeds financial news from PostgreSQL into ChromaDB using
sentence-transformers, and retrieves relevant context for the
LangGraph agent pipeline.

Usage::

    from src.agents.rag import FinancialRAG

    rag = FinancialRAG()
    n = rag.embed_pending_news()       # embed new articles
    ctx = rag.retrieve("NVDA", "AI chip demand outlook")
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from chromadb.api import ClientAPI
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer
from sqlalchemy import Engine, text

from src.data.news_ingestion import NEWS_TABLE
from src.utils.db import get_chroma_client, get_postgres_engine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME: str = "financial_news"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
_BATCH_SIZE: int = 64


# ---------------------------------------------------------------------------
# FinancialRAG
# ---------------------------------------------------------------------------


class FinancialRAG:
    """Embeds and retrieves financial news via ChromaDB.

    Reads articles from the ``financial_news`` PostgreSQL table,
    generates embeddings with ``sentence-transformers``, stores them
    in a single ChromaDB collection with ticker/date/source metadata,
    and provides semantic retrieval with date-based reranking.

    Args:
        engine: SQLAlchemy engine.  Defaults to ``get_postgres_engine()``.
        chroma_client: ChromaDB client.  Defaults to ``get_chroma_client()``.
        model_name: HuggingFace model ID for the embedding model.
    """

    def __init__(
        self,
        *,
        engine: Engine | None = None,
        chroma_client: ClientAPI | None = None,
        model_name: str = EMBEDDING_MODEL,
    ) -> None:
        load_dotenv()
        self.engine = engine or get_postgres_engine()
        self.chroma = chroma_client or get_chroma_client()
        self.model = SentenceTransformer(model_name)
        self.collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
        )
        logger.info(
            "FinancialRAG initialized | model={} collection='{}'",
            model_name,
            COLLECTION_NAME,
        )

    # ------------------------------------------------------------------
    # Embedding pipeline
    # ------------------------------------------------------------------

    def _load_pending_news(self) -> list[dict[str, Any]]:
        """Load articles with ``embedding_status='pending'`` from PostgreSQL.

        Returns:
            List of article dicts with id, date, ticker, title, source,
            url, and summary.
        """
        query = text(f"""
            SELECT id, date, ticker, title, source, url, summary
            FROM {NEWS_TABLE}
            WHERE embedding_status = 'pending'
            ORDER BY date ASC
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()

        articles = [dict(row) for row in rows]
        logger.info("Loaded {} pending articles from PostgreSQL", len(articles))
        return articles

    def _build_document(self, article: dict[str, Any]) -> str:
        """Build a single document string for embedding.

        Concatenates title and summary, providing a richer semantic
        representation than title alone.

        Args:
            article: Article dict from PostgreSQL.

        Returns:
            Document string ready for embedding.
        """
        title = article.get("title", "") or ""
        summary = article.get("summary", "") or ""
        return f"{title}. {summary}".strip()

    def _mark_as_embedded(self, article_ids: list[int]) -> None:
        """Update ``embedding_status`` to ``'embedded'`` for processed articles.

        Args:
            article_ids: List of PostgreSQL row IDs to mark.
        """
        if not article_ids:
            return
        update_sql = text(f"""
            UPDATE {NEWS_TABLE}
            SET embedding_status = 'embedded'
            WHERE id = ANY(:ids)
        """)
        with self.engine.begin() as conn:
            conn.execute(update_sql, {"ids": article_ids})
        logger.debug("Marked {} articles as embedded", len(article_ids))

    def embed_pending_news(self) -> int:
        """Embed all pending news articles into ChromaDB.

        Reads articles where ``embedding_status='pending'``, generates
        embeddings with sentence-transformers, upserts into ChromaDB
        with metadata, and marks articles as ``'embedded'`` in PostgreSQL.

        Returns:
            Number of articles newly embedded.
        """
        articles = self._load_pending_news()
        if not articles:
            logger.info("No pending articles to embed")
            return 0

        total_embedded = 0

        for batch_start in range(0, len(articles), _BATCH_SIZE):
            batch = articles[batch_start : batch_start + _BATCH_SIZE]

            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []
            ids: list[str] = []
            pg_ids: list[int] = []

            for art in batch:
                doc = self._build_document(art)
                if not doc or doc == ".":
                    continue

                art_date = art.get("date")
                date_str = str(art_date) if art_date else ""

                documents.append(doc)
                metadatas.append({
                    "ticker": art.get("ticker") or "UNKNOWN",
                    "date": date_str,
                    "source": art.get("source") or "unknown",
                    "url": art.get("url") or "",
                    "title": art.get("title") or "",
                })
                ids.append(f"news_{art['id']}")
                pg_ids.append(art["id"])

            if not documents:
                continue

            embeddings = self.model.encode(
                documents, show_progress_bar=False
            ).tolist()

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            self._mark_as_embedded(pg_ids)
            total_embedded += len(documents)

            logger.debug(
                "Embedded batch {}-{} ({} docs)",
                batch_start,
                batch_start + len(batch),
                len(documents),
            )

        logger.info(
            "Embedding complete | {} articles embedded into '{}'",
            total_embedded,
            COLLECTION_NAME,
        )
        return total_embedded

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        ticker: str,
        query: str,
        *,
        top_k: int = 5,
        max_age_days: int = 30,
    ) -> list[str]:
        """Retrieve relevant news snippets for a ticker and query.

        Queries ChromaDB with the embedding of ``query``, filtered by
        ``ticker``, then reranks results by date (most recent first).

        Args:
            ticker: Asset symbol to filter by (e.g. ``"NVDA"``).
            query: Semantic search query (e.g. ``"AI chip demand"``).
            top_k: Maximum number of results to return.
            max_age_days: Exclude articles older than this many days.

        Returns:
            List of formatted news strings suitable for LLM context,
            e.g. ``["[2026-03-01 | Reuters] NVDA beats earnings..."]``.
        """
        cutoff = date.today() - timedelta(days=max_age_days)
        cutoff_str = str(cutoff)

        # Fetch more candidates than top_k to allow date reranking
        n_results = min(top_k * 3, 50)

        try:
            results = self.collection.query(
                query_embeddings=self.model.encode(
                    [query], show_progress_bar=False
                ).tolist(),
                n_results=n_results,
                where={"ticker": ticker},
            )
        except Exception as exc:
            logger.warning("ChromaDB query failed for ticker={}: {}", ticker, exc)
            return []

        if not results or not results.get("metadatas"):
            return []

        # Flatten results (query returns nested lists)
        all_metadatas = results["metadatas"][0]
        all_documents = results["documents"][0] if results.get("documents") else []
        all_distances = results["distances"][0] if results.get("distances") else []

        # Filter by date cutoff and exclude future-dated articles
        today_str = str(date.today())
        candidates: list[dict[str, Any]] = []
        for i, meta in enumerate(all_metadatas):
            article_date = meta.get("date", "")
            if article_date and article_date < cutoff_str:
                continue
            if article_date and article_date > today_str:
                continue

            candidates.append({
                "date": article_date,
                "source": meta.get("source", "unknown"),
                "title": meta.get("title", ""),
                "document": all_documents[i] if i < len(all_documents) else "",
                "distance": all_distances[i] if i < len(all_distances) else 999.0,
            })

        # Rerank: date descending (most recent first), distance ascending
        # (most relevant first) as tiebreaker.  Python's stable sort lets us
        # do a two-pass sort: first by secondary key, then by primary.
        candidates.sort(key=lambda c: c["distance"])
        candidates.sort(key=lambda c: c["date"] or "0000-00-00", reverse=True)

        # Take top_k
        top_candidates = candidates[:top_k]

        # Format for LLM context
        snippets: list[str] = []
        for c in top_candidates:
            snippet = f"[{c['date']} | {c['source']}] {c['document']}"
            snippets.append(snippet)

        logger.info(
            "RAG retrieve: ticker={} query='{}' → {} results (of {} candidates)",
            ticker,
            query[:50],
            len(snippets),
            len(candidates),
        )
        return snippets

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_collection_count(self) -> int:
        """Return the total number of documents in the ChromaDB collection.

        Returns:
            Document count.
        """
        return self.collection.count()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _date_sort_key(date_str: str) -> str:
    """Convert a date string to a sortable key.

    Returns the date string itself (ISO format sorts lexicographically)
    or an empty string for missing dates (sorts last when reversed).

    Args:
        date_str: Date string in ``YYYY-MM-DD`` format.

    Returns:
        Sortable string key.
    """
    return date_str if date_str else ""
