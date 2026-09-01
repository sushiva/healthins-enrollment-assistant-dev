"""Smoke test for the Step 1 src/ port.

Confirms the extracted src/ modules reproduce notebook 06's own example
query using the existing FAISS cache — no PDF re-parse, no re-embedding.

Run from the project root:
    python scripts/smoke_test.py
"""
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.chat import FormularyChat
from src.generation.llm import get_ollama_llm
from src.retrieval.hybrid import build_bm25_retriever, build_hybrid_retriever
from src.retrieval.vector_store import all_documents, load_or_build_vectorstore
from src.utils.config import load_config

QUESTION = "What tier is atorvastatin calcium tabs 10mg, 20mg, 40mg, 80mg?"


def main():
    config = load_config()
    print(f"Config loaded. top_k={config['retrieval']['top_k']}, llm={config['llm']['model']}")

    vectorstore = load_or_build_vectorstore(config)
    row_docs = all_documents(vectorstore)
    print(f"Loaded {len(row_docs)} rows from cache (no re-embedding).")

    bm25_retriever = build_bm25_retriever(row_docs)
    build_retriever = functools.partial(build_hybrid_retriever, vectorstore, bm25_retriever)

    llm = get_ollama_llm(config)
    chat = FormularyChat(llm, build_retriever, default_k=config["retrieval"]["top_k"])

    answer, docs = chat.ask(QUESTION, stream=False)

    assert answer.strip(), "Got an empty answer — something in the port is broken."
    assert docs, "No documents retrieved."
    assert "Tier 1" in answer, (
        f"Expected 'Tier 1' in the answer (per notebook 06's own run of this exact "
        f"question), got:\n{answer}"
    )
    assert "atorvastatin" in "\n".join(d.metadata.get("drug", "") for d in docs).lower(), (
        "Expected atorvastatin to be among the retrieved rows."
    )

    # Second turn: a pronoun-only follow-up with no drug name token, proving
    # condense-question is wired correctly (matches notebook 06 §5's before/after).
    follow_up = "Does it have any quantity limits or other requirements?"
    answer2, docs2 = chat.ask(follow_up, stream=False)
    assert any(d.metadata.get("drug", "").lower().startswith("atorvastatin") for d in docs2), (
        f"Follow-up retrieval didn't land back on atorvastatin — condense_question "
        f"may not be wired correctly. Retrieved: {[d.metadata.get('drug') for d in docs2]}"
    )

    print(f"\nQ1: {QUESTION}\n\n{answer}\n")
    print(f"Q2 (follow-up, no drug name): {follow_up}\n\n{answer2}\n")
    print("✅ smoke test passed — src/ port matches notebook 06 behavior")


if __name__ == "__main__":
    main()
