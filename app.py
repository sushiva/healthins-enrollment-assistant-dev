"""Health Insurance Formulary Assistant — Gradio app.

A RAG chatbot over a real health plan's drug formulary (table-aware PDF
extraction + hybrid dense/BM25 retrieval + condense-question chat memory,
all built and evaluated in notebooks/formulary/01-06 and ported to src/ in
Step 1). Generation runs on Google Gemini, using an API key the user pastes
into the UI below — never read from an environment file, never written
anywhere, held only in this process's memory for the lifetime of a request.

Entry point for the Hugging Face Space: this file is expected at the repo
root by the Gradio SDK Space runtime.
"""
import functools

import gradio as gr

from src.generation.chat import FormularyChat, history_from_transcript
from src.generation.llm import get_gemini_llm
from src.retrieval.hybrid import build_bm25_retriever, build_hybrid_retriever
from src.retrieval.vector_store import all_documents, load_or_build_vectorstore
from src.utils.config import load_config

GEMINI_KEY_URL = "https://aistudio.google.com/app/apikey"

# Loaded once at startup and shared read-only across all users/requests — only
# the LLM client and each conversation's history are built per-request below.
config = load_config()
_vectorstore = load_or_build_vectorstore(config)
_row_docs = all_documents(_vectorstore)
_bm25_retriever = build_bm25_retriever(_row_docs)
_build_retriever = functools.partial(build_hybrid_retriever, _vectorstore, _bm25_retriever)


def respond(message: str, history: list[dict], api_key: str):
    if not api_key or not api_key.strip():
        yield "Please paste your Google Gemini API key above to start chatting."
        return

    try:
        llm = get_gemini_llm(api_key.strip(), config)
        chat = FormularyChat(llm, _build_retriever, default_k=config["retrieval"]["top_k"])
        chat.history = history_from_transcript(history)

        for partial_answer, _docs in chat.ask_stream(message):
            yield partial_answer
    except Exception as exc:  # bad/expired key, quota exceeded, network error, etc.
        yield (
            "⚠️ Something went wrong calling Gemini with the provided key "
            f"({type(exc).__name__}: {exc}). Double-check the key is valid and "
            "has quota, then try again."
        )


with gr.Blocks(title="Health Insurance Formulary Assistant") as demo:
    gr.Markdown(
        "# 💊 Health Insurance Formulary Assistant\n"
        "Ask about drug coverage tiers, quantity limits, and prior-authorization "
        "requirements for an Oscar NC 2026 health plan — grounded in the plan's "
        "actual formulary document via retrieval-augmented generation, not the "
        "model's general knowledge."
    )
    api_key_box = gr.Textbox(
        label="Google Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key — used only for this session, never stored",
        info=f"Don't have one? Get a free key at {GEMINI_KEY_URL}",
    )
    gr.ChatInterface(
        fn=respond,
        additional_inputs=[api_key_box],
        examples=[
            ["What tier is atorvastatin calcium tabs 10mg, 20mg, 40mg, 80mg?"],
            ["What is the quantity limit for acetaminophen with codeine solution 120-12 mg/5ml?"],
        ],
    )

if __name__ == "__main__":
    demo.queue().launch()
