"""FAISS vector store with a model-keyed embedding cache.

Ported from notebooks/formulary/02-Basic_RAG_Formulary.ipynb §3. A cache
belongs to exactly one embedding model — the cache directory is named after
the model ID and dimension, so switching config.yaml's embedding.model_name
points at a different (or missing) cache, never a stale one silently reused
with the wrong model.

⚠️ FAISS.load_local unpickles a docstore file — safe here only because we
only ever load a cache this project itself wrote.

Hardened per codereview/vector_store_review.md: the count/dimension checks
now raise explicit ValueErrors instead of `assert` (asserts are stripped
under `python -O`/`PYTHONOPTIMIZE=1`, which would silently skip exactly the
checks that catch a stale or corrupt cache); all_documents() no longer
depends solely on InMemoryDocstore's private `_dict`; and the embeddings
model can be injected instead of always being reloaded.
"""
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import resolve_path

INDEX_NAME = "formulary_rows"


def _cache_dir(config: dict) -> Path:
    model_tag = config["embedding"]["model_name"].replace("/", "-")
    dim = config["embedding"]["dimension"]
    return resolve_path(config["paths"]["processed_data"]) / f"faiss_cache_{model_tag}_{dim}d"


def load_or_build_vectorstore(
    config: dict,
    row_docs: list[Document] | None = None,
    force_reembed: bool = False,
    embeddings: Embeddings | None = None,
) -> FAISS:
    """Load the cached FAISS index, or build+cache it from row_docs if missing.

    Pass row_docs=None (the common app-time case, matching 06's usage) to
    require an existing cache — ingestion is expected to have already run.
    Pass row_docs (the ingestion-time case, matching 02's usage) to build and
    cache the index the first time, or to rebuild it with force_reembed=True.
    Pass embeddings to reuse an already-loaded model instead of reloading one
    from config every call.
    """
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=config["embedding"]["model_name"])
    cache_dir = _cache_dir(config)
    cache_exists = (cache_dir / f"{INDEX_NAME}.faiss").exists()

    if cache_exists and not force_reembed:
        vectorstore = FAISS.load_local(
            str(cache_dir), embeddings, index_name=INDEX_NAME, allow_dangerous_deserialization=True
        )
    else:
        if not row_docs:
            raise FileNotFoundError(
                f"No cache at {cache_dir} and no row_docs given to build one. "
                "Run ingestion first, or pass extracted row_docs to build the cache."
            )
        vectorstore = FAISS.from_documents(row_docs, embeddings)
        cache_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(cache_dir), index_name=INDEX_NAME)

    # Explicit raises, not assert — a partially-written or stale cache is a real
    # failure mode and must not go silent just because -O/PYTHONOPTIMIZE stripped
    # the check.
    if row_docs is not None and vectorstore.index.ntotal != len(row_docs):
        raise ValueError(
            f"Vector store has {vectorstore.index.ntotal} vectors for {len(row_docs)} rows — "
            "cache is stale or embedding collapsed. Pass force_reembed=True to rebuild."
        )
    if vectorstore.index.d != config["embedding"]["dimension"]:
        raise ValueError(
            f"Cached vectors are {vectorstore.index.d}-dim but config says "
            f"{config['embedding']['dimension']} — re-embed with force_reembed=True."
        )
    return vectorstore


def all_documents(vectorstore: FAISS) -> list[Document]:
    """All Documents backing a loaded FAISS store — needed to build BM25 from the same rows.

    Prefers the public index_to_docstore_id -> docstore.search() path; falls
    back to the InMemoryDocstore's private `_dict` only if that's unavailable
    on whatever docstore implementation is in use.
    """
    if hasattr(vectorstore, "index_to_docstore_id") and hasattr(vectorstore.docstore, "search"):
        return [
            vectorstore.docstore.search(doc_id)
            for doc_id in vectorstore.index_to_docstore_id.values()
        ]
    return list(vectorstore.docstore._dict.values())
