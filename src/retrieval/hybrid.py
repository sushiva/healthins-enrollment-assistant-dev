"""Hybrid (dense + BM25) retrieval.

Ported from notebooks/formulary/02-Basic_RAG_Formulary.ipynb §3.5, adopted
on the strength of 05-Hybrid_Search_Formulary.ipynb's measured comparison:
Hit Rate at top_k=3 went from 93.5% (vector-only) to 98.0% (hybrid), MRR
from 0.854 to 0.923 — for no added cost, since BM25 is plain statistical
term matching (no model, no LLM call).

One change from the notebook: build_hybrid_retriever there closed over
module-level `vectorstore`/`bm25_retriever` globals. Here both are passed in
explicitly — same logic, just not relying on notebook globals.

Code-review note (codereview/hybrid_review.md §3.1): the review suggested
importing EnsembleRetriever from `langchain.retrievers` or
`langchain_community.retrievers` as the "standard" path. Verified against the
installed langchain 1.x stack: neither exists (`langchain.retrievers` isn't a
module at all; `langchain_community.retrievers` no longer exports
EnsembleRetriever). `langchain_classic.retrievers` is where it actually lives
in this dependency set, so that import is kept as-is.
"""
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def build_bm25_retriever(row_docs: list[Document]) -> BM25Retriever:
    return BM25Retriever.from_documents(row_docs)


def build_hybrid_retriever(
    vectorstore: VectorStore,
    bm25_retriever: BM25Retriever,
    k: int,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> EnsembleRetriever:
    """Fuse dense (vectorstore) and sparse (bm25_retriever) retrieval via RRF.

    Mutates bm25_retriever.k in place to match k on every call — harmless for
    the one bm25_retriever-per-session usage in this app (app.py, chat.py),
    but if the same BM25Retriever instance is ever shared across concurrent
    callers using different k values, this mutation would race between them.
    """
    if vector_weight < 0 or bm25_weight < 0:
        raise ValueError("Retrieval weights must be non-negative.")
    if vector_weight == 0 and bm25_weight == 0:
        raise ValueError("At least one of vector_weight/bm25_weight must be positive.")

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25_retriever.k = k
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )
