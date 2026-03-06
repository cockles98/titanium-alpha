"""Tests for src.agents.rag — FinancialRAG.

Covers initialization, embedding pipeline, retrieval with reranking,
and helper functions. All external dependencies (PostgreSQL, ChromaDB,
SentenceTransformer) are mocked.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.rag import COLLECTION_NAME, FinancialRAG, _date_sort_key


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_collection() -> MagicMock:
    """A fake ChromaDB collection."""
    col: MagicMock = MagicMock()
    col.count.return_value = 42
    return col


@pytest.fixture()
def mock_chroma(mock_collection: MagicMock) -> MagicMock:
    """A fake ChromaDB client that returns ``mock_collection``."""
    client: MagicMock = MagicMock()
    client.get_or_create_collection.return_value = mock_collection
    return client


@pytest.fixture()
def mock_st_model() -> MagicMock:
    """A fake SentenceTransformer model with realistic encode output."""
    model: MagicMock = MagicMock()
    model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    return model


@pytest.fixture()
def rag(
    mock_engine: MagicMock,
    mock_chroma: MagicMock,
    mock_st_model: MagicMock,
) -> FinancialRAG:
    """Build a FinancialRAG with all dependencies mocked."""
    with patch("src.agents.rag.SentenceTransformer", return_value=mock_st_model):
        with patch("src.agents.rag.load_dotenv"):
            return FinancialRAG(engine=mock_engine, chroma_client=mock_chroma)


def _make_article(
    *,
    id: int = 1,
    ticker: str = "NVDA",
    title: str = "NVDA beats earnings",
    summary: str = "Revenue up 30% YoY.",
    source: str = "Reuters",
    url: str = "https://example.com/1",
    dt: date | None = None,
) -> dict[str, Any]:
    """Helper to create a fake article dict mimicking a PostgreSQL row."""
    return {
        "id": id,
        "date": dt or date(2026, 3, 1),
        "ticker": ticker,
        "title": title,
        "summary": summary,
        "source": source,
        "url": url,
    }


def _wire_connect(mock_engine: MagicMock, rows: list[Any]) -> None:
    """Wire mock_engine.connect() context manager to return rows."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.mappings.return_value.all.return_value = rows
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)


def _wire_begin(mock_engine: MagicMock) -> MagicMock:
    """Wire mock_engine.begin() context manager, return the inner conn."""
    begin_conn = MagicMock()
    mock_engine.begin.return_value.__enter__ = MagicMock(return_value=begin_conn)
    mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    return begin_conn


# ===================================================================
# 1. __init__
# ===================================================================


class TestInit:
    """Tests for FinancialRAG.__init__."""

    def test_creates_instance_with_mocked_deps(
        self,
        mock_engine: MagicMock,
        mock_chroma: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Instance is created and collection is fetched."""
        with patch("src.agents.rag.SentenceTransformer", return_value=mock_st_model):
            with patch("src.agents.rag.load_dotenv"):
                r = FinancialRAG(engine=mock_engine, chroma_client=mock_chroma)

        assert r.engine is mock_engine
        assert r.chroma is mock_chroma
        assert r.model is mock_st_model
        mock_chroma.get_or_create_collection.assert_called_once_with(
            name=COLLECTION_NAME
        )

    def test_calls_load_dotenv(
        self,
        mock_engine: MagicMock,
        mock_chroma: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """load_dotenv is called during init."""
        with patch("src.agents.rag.SentenceTransformer", return_value=mock_st_model):
            with patch("src.agents.rag.load_dotenv") as mock_dotenv:
                FinancialRAG(engine=mock_engine, chroma_client=mock_chroma)

        mock_dotenv.assert_called_once()

    def test_defaults_to_factory_functions_when_none(
        self,
        mock_st_model: MagicMock,
    ) -> None:
        """When engine/chroma are None, factory functions are called."""
        mock_eng = MagicMock()
        mock_chr = MagicMock()
        mock_chr.get_or_create_collection.return_value = MagicMock()

        with (
            patch("src.agents.rag.SentenceTransformer", return_value=mock_st_model),
            patch("src.agents.rag.load_dotenv"),
            patch("src.agents.rag.get_postgres_engine", return_value=mock_eng),
            patch("src.agents.rag.get_chroma_client", return_value=mock_chr),
        ):
            r = FinancialRAG()

        assert r.engine is mock_eng
        assert r.chroma is mock_chr


# ===================================================================
# 2. _load_pending_news
# ===================================================================


class TestLoadPendingNews:
    """Tests for FinancialRAG._load_pending_news."""

    def test_returns_list_of_dicts(self, rag: FinancialRAG, mock_engine: MagicMock) -> None:
        """Returns correctly shaped list of dicts."""
        row = _make_article()
        _wire_connect(mock_engine, [row])

        result = rag._load_pending_news()

        assert len(result) == 1
        assert result[0]["ticker"] == "NVDA"
        assert "title" in result[0]
        assert "summary" in result[0]

    def test_returns_empty_list_when_no_pending(
        self, rag: FinancialRAG, mock_engine: MagicMock
    ) -> None:
        """Returns empty list when no articles are pending."""
        _wire_connect(mock_engine, [])

        result = rag._load_pending_news()
        assert result == []


# ===================================================================
# 3. _build_document
# ===================================================================


class TestBuildDocument:
    """Tests for FinancialRAG._build_document."""

    def test_concatenates_title_and_summary(self, rag: FinancialRAG) -> None:
        """Title and summary are joined by '. '."""
        art = _make_article(title="Big news", summary="Details here")
        result = rag._build_document(art)
        assert result == "Big news. Details here"

    def test_title_only(self, rag: FinancialRAG) -> None:
        """Only title present, summary empty."""
        art = _make_article(title="Breaking", summary="")
        result = rag._build_document(art)
        assert result == "Breaking."

    def test_summary_only(self, rag: FinancialRAG) -> None:
        """Only summary present, title empty."""
        art = _make_article(title="", summary="Some details")
        result = rag._build_document(art)
        assert result == ". Some details"

    def test_both_none(self, rag: FinancialRAG) -> None:
        """Both title and summary are None -> returns '.'."""
        art: dict[str, Any] = {"title": None, "summary": None}
        result = rag._build_document(art)
        assert result == "."

    def test_both_empty(self, rag: FinancialRAG) -> None:
        """Both title and summary are empty strings -> returns '.'."""
        art: dict[str, Any] = {"title": "", "summary": ""}
        result = rag._build_document(art)
        assert result == "."

    def test_missing_keys(self, rag: FinancialRAG) -> None:
        """Keys absent from dict -> returns '.'."""
        art: dict[str, Any] = {}
        result = rag._build_document(art)
        assert result == "."


# ===================================================================
# 4. _mark_as_embedded
# ===================================================================


class TestMarkAsEmbedded:
    """Tests for FinancialRAG._mark_as_embedded."""

    def test_executes_update(self, rag: FinancialRAG, mock_engine: MagicMock) -> None:
        """Calls execute with the right ids."""
        conn = _wire_begin(mock_engine)

        rag._mark_as_embedded([10, 20, 30])

        conn.execute.assert_called_once()
        call_args = conn.execute.call_args
        assert call_args[0][1] == {"ids": [10, 20, 30]}

    def test_noop_with_empty_list(self, rag: FinancialRAG, mock_engine: MagicMock) -> None:
        """Does not touch DB when list is empty."""
        mock_engine.begin.reset_mock()
        rag._mark_as_embedded([])
        mock_engine.begin.assert_not_called()


# ===================================================================
# 5. embed_pending_news
# ===================================================================


class TestEmbedPendingNews:
    """Tests for FinancialRAG.embed_pending_news."""

    def _setup_pending(
        self, rag: FinancialRAG, mock_engine: MagicMock, articles: list[dict[str, Any]]
    ) -> None:
        """Wire mock_engine for both connect (load) and begin (mark)."""
        _wire_connect(mock_engine, articles)
        _wire_begin(mock_engine)

    def test_returns_zero_when_no_pending(
        self, rag: FinancialRAG, mock_engine: MagicMock
    ) -> None:
        """Returns 0 when no pending articles exist."""
        self._setup_pending(rag, mock_engine, [])
        assert rag.embed_pending_news() == 0

    def test_processes_articles_and_returns_count(
        self,
        rag: FinancialRAG,
        mock_engine: MagicMock,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Embeds articles, calls upsert, returns correct count."""
        articles = [
            _make_article(id=1, title="A1", summary="S1"),
            _make_article(id=2, title="A2", summary="S2"),
        ]
        self._setup_pending(rag, mock_engine, articles)
        mock_st_model.encode.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        result = rag.embed_pending_news()

        assert result == 2
        mock_st_model.encode.assert_called_once()
        mock_collection.upsert.assert_called_once()

    def test_ids_formatted_as_news_prefix(
        self,
        rag: FinancialRAG,
        mock_engine: MagicMock,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """ChromaDB IDs are formatted as 'news_{id}'."""
        articles = [_make_article(id=42)]
        self._setup_pending(rag, mock_engine, articles)
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        rag.embed_pending_news()

        upsert_call = mock_collection.upsert.call_args
        assert upsert_call.kwargs["ids"] == ["news_42"]

    def test_metadatas_contain_required_fields(
        self,
        rag: FinancialRAG,
        mock_engine: MagicMock,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Metadatas include ticker, date, source, url, title."""
        articles = [
            _make_article(
                id=1,
                ticker="AAPL",
                source="CNBC",
                url="https://x.com",
                title="Apple news",
                dt=date(2026, 3, 2),
            )
        ]
        self._setup_pending(rag, mock_engine, articles)
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        rag.embed_pending_news()

        meta = mock_collection.upsert.call_args.kwargs["metadatas"][0]
        assert meta["ticker"] == "AAPL"
        assert meta["date"] == "2026-03-02"
        assert meta["source"] == "CNBC"
        assert meta["url"] == "https://x.com"
        assert meta["title"] == "Apple news"

    def test_skips_empty_documents(
        self,
        rag: FinancialRAG,
        mock_engine: MagicMock,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Articles where _build_document returns '.' are skipped."""
        articles = [
            _make_article(id=1, title="", summary=""),   # -> "." -> skipped
            _make_article(id=2, title="Valid", summary="OK"),
        ]
        self._setup_pending(rag, mock_engine, articles)
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        result = rag.embed_pending_news()

        assert result == 1
        upsert_call = mock_collection.upsert.call_args
        assert upsert_call.kwargs["ids"] == ["news_2"]

    def test_all_empty_documents_returns_zero(
        self,
        rag: FinancialRAG,
        mock_engine: MagicMock,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """When all docs are empty, returns 0 and no upsert."""
        articles = [
            _make_article(id=1, title="", summary=""),
            _make_article(id=2, title=None, summary=None),
        ]
        self._setup_pending(rag, mock_engine, articles)

        result = rag.embed_pending_news()

        assert result == 0
        mock_collection.upsert.assert_not_called()

    def test_handles_none_date(
        self,
        rag: FinancialRAG,
        mock_engine: MagicMock,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Article with None date stores empty string in metadata."""
        art = _make_article(id=1)
        art["date"] = None
        self._setup_pending(rag, mock_engine, [art])
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        rag.embed_pending_news()

        meta = mock_collection.upsert.call_args.kwargs["metadatas"][0]
        assert meta["date"] == ""

    def test_handles_none_ticker_defaults_to_unknown(
        self,
        rag: FinancialRAG,
        mock_engine: MagicMock,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Article with None ticker stores 'UNKNOWN' in metadata."""
        art = _make_article(id=1)
        art["ticker"] = None
        self._setup_pending(rag, mock_engine, [art])
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        rag.embed_pending_news()

        meta = mock_collection.upsert.call_args.kwargs["metadatas"][0]
        assert meta["ticker"] == "UNKNOWN"


# ===================================================================
# 6. retrieve
# ===================================================================


class TestRetrieve:
    """Tests for FinancialRAG.retrieve."""

    def _mock_query_results(
        self,
        mock_collection: MagicMock,
        metadatas: list[dict[str, Any]],
        documents: list[str],
        distances: list[float],
    ) -> None:
        """Configure mock_collection.query to return structured results."""
        mock_collection.query.return_value = {
            "metadatas": [metadatas],
            "documents": [documents],
            "distances": [distances],
        }

    def test_returns_formatted_strings(
        self,
        rag: FinancialRAG,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Results are formatted as '[date | source] document'."""
        recent = str(date.today() - timedelta(days=1))
        self._mock_query_results(
            mock_collection,
            metadatas=[{
                "ticker": "NVDA",
                "date": recent,
                "source": "Reuters",
                "title": "T",
            }],
            documents=["NVDA beats earnings expectations"],
            distances=[0.15],
        )
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        results = rag.retrieve("NVDA", "earnings")

        assert len(results) == 1
        assert results[0] == f"[{recent} | Reuters] NVDA beats earnings expectations"

    def test_filters_by_ticker_in_where(
        self,
        rag: FinancialRAG,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Query uses where={ticker: ...} for filtering."""
        self._mock_query_results(mock_collection, [], [], [])
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        rag.retrieve("AAPL", "news")

        query_call = mock_collection.query.call_args
        assert query_call.kwargs["where"] == {"ticker": "AAPL"}

    def test_filters_by_max_age_days(
        self,
        rag: FinancialRAG,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Articles older than max_age_days are excluded."""
        old_date = str(date.today() - timedelta(days=60))
        recent_date = str(date.today() - timedelta(days=5))

        self._mock_query_results(
            mock_collection,
            metadatas=[
                {"ticker": "NVDA", "date": old_date, "source": "A", "title": "Old"},
                {"ticker": "NVDA", "date": recent_date, "source": "B", "title": "New"},
            ],
            documents=["Old doc", "New doc"],
            distances=[0.1, 0.2],
        )
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        results = rag.retrieve("NVDA", "test", max_age_days=30)

        assert len(results) == 1
        assert "New doc" in results[0]

    def test_reranking_most_recent_first(
        self,
        rag: FinancialRAG,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Results are sorted by date descending."""
        d1 = str(date.today() - timedelta(days=2))
        d2 = str(date.today() - timedelta(days=1))

        self._mock_query_results(
            mock_collection,
            metadatas=[
                {"ticker": "NVDA", "date": d1, "source": "A", "title": "Older"},
                {"ticker": "NVDA", "date": d2, "source": "B", "title": "Newer"},
            ],
            documents=["Older article", "Newer article"],
            distances=[0.1, 0.2],
        )
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        results = rag.retrieve("NVDA", "test")

        assert "Newer article" in results[0]
        assert "Older article" in results[1]

    def test_reranking_tiebreak_by_distance(
        self,
        rag: FinancialRAG,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Same date -> sorted by distance ascending (closer = more relevant)."""
        d = str(date.today())

        self._mock_query_results(
            mock_collection,
            metadatas=[
                {"ticker": "NVDA", "date": d, "source": "A", "title": "Far"},
                {"ticker": "NVDA", "date": d, "source": "B", "title": "Close"},
            ],
            documents=["Far doc", "Close doc"],
            distances=[0.5, 0.1],
        )
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        results = rag.retrieve("NVDA", "test")

        # reverse=True with -distance: higher -distance (i.e. lower distance) first
        assert "Close doc" in results[0]
        assert "Far doc" in results[1]

    def test_excludes_future_dated_articles(
        self,
        rag: FinancialRAG,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Articles with dates in the future are excluded."""
        today = str(date.today())
        future_date = str(date.today() + timedelta(days=30))

        self._mock_query_results(
            mock_collection,
            metadatas=[
                {"ticker": "NVDA", "date": future_date, "source": "A", "title": "Future"},
                {"ticker": "NVDA", "date": today, "source": "B", "title": "Today"},
            ],
            documents=["Future article", "Today article"],
            distances=[0.1, 0.2],
        )
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        results = rag.retrieve("NVDA", "test")
        assert len(results) == 1
        assert "Today article" in results[0]

    def test_returns_empty_on_chromadb_exception(
        self,
        rag: FinancialRAG,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Returns [] when ChromaDB raises an exception."""
        mock_collection.query.side_effect = RuntimeError("ChromaDB down")
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        results = rag.retrieve("NVDA", "test")

        assert results == []

    def test_returns_empty_when_no_results(
        self,
        rag: FinancialRAG,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Returns [] when query returns empty metadatas."""
        mock_collection.query.return_value = {
            "metadatas": [[]],
            "documents": [[]],
            "distances": [[]],
        }
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        results = rag.retrieve("NVDA", "test")
        assert results == []

    def test_returns_empty_when_metadatas_is_none(
        self,
        rag: FinancialRAG,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Returns [] when metadatas key is missing/None."""
        mock_collection.query.return_value = {"metadatas": None}
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        results = rag.retrieve("NVDA", "test")
        assert results == []

    def test_respects_top_k(
        self,
        rag: FinancialRAG,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Only top_k results are returned even when more candidates exist."""
        d = str(date.today())
        n = 10
        self._mock_query_results(
            mock_collection,
            metadatas=[
                {"ticker": "NVDA", "date": d, "source": "S", "title": f"T{i}"}
                for i in range(n)
            ],
            documents=[f"Doc {i}" for i in range(n)],
            distances=[0.1 * i for i in range(n)],
        )
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        results = rag.retrieve("NVDA", "test", top_k=3)

        assert len(results) == 3

    def test_n_results_is_capped_at_50(
        self,
        rag: FinancialRAG,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """n_results passed to query is min(top_k * 3, 50)."""
        self._mock_query_results(mock_collection, [], [], [])
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        rag.retrieve("NVDA", "test", top_k=20)

        query_call = mock_collection.query.call_args
        assert query_call.kwargs["n_results"] == 50  # min(20*3, 50) = 50

    def test_empty_date_articles_sort_last(
        self,
        rag: FinancialRAG,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Articles with empty date string sort after dated articles."""
        d = str(date.today())
        self._mock_query_results(
            mock_collection,
            metadatas=[
                {"ticker": "NVDA", "date": "", "source": "A", "title": "NoDate"},
                {"ticker": "NVDA", "date": d, "source": "B", "title": "HasDate"},
            ],
            documents=["No date doc", "Has date doc"],
            distances=[0.1, 0.2],
        )
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        results = rag.retrieve("NVDA", "test")

        assert "Has date doc" in results[0]
        assert "No date doc" in results[1]

    def test_returns_empty_when_results_is_none(
        self,
        rag: FinancialRAG,
        mock_collection: MagicMock,
        mock_st_model: MagicMock,
    ) -> None:
        """Returns [] when query returns None."""
        mock_collection.query.return_value = None
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        results = rag.retrieve("NVDA", "test")
        assert results == []


# ===================================================================
# 7. get_collection_count
# ===================================================================


class TestGetCollectionCount:
    """Tests for FinancialRAG.get_collection_count."""

    def test_returns_count(self, rag: FinancialRAG, mock_collection: MagicMock) -> None:
        """Returns the value from collection.count()."""
        mock_collection.count.return_value = 123
        assert rag.get_collection_count() == 123

    def test_returns_zero_for_empty(self, rag: FinancialRAG, mock_collection: MagicMock) -> None:
        """Returns 0 for an empty collection."""
        mock_collection.count.return_value = 0
        assert rag.get_collection_count() == 0


# ===================================================================
# 8. _date_sort_key helper
# ===================================================================


class TestDateSortKey:
    """Tests for the module-level _date_sort_key helper."""

    def test_returns_date_string_for_valid_input(self) -> None:
        """Valid ISO date string is returned unchanged."""
        assert _date_sort_key("2026-03-01") == "2026-03-01"

    def test_returns_empty_for_empty_input(self) -> None:
        """Empty string returns empty string."""
        assert _date_sort_key("") == ""

    def test_returns_empty_for_none_like_falsy(self) -> None:
        """Falsy input returns empty string."""
        assert _date_sort_key("") == ""

    def test_lexicographic_ordering(self) -> None:
        """ISO date strings sort correctly lexicographically."""
        dates = ["2026-01-15", "2026-03-01", "2025-12-31"]
        sorted_dates = sorted(dates, key=_date_sort_key, reverse=True)
        assert sorted_dates == ["2026-03-01", "2026-01-15", "2025-12-31"]
