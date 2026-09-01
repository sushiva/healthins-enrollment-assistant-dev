# Health Insurance Formulary Assistant

A retrieval-augmented generation (RAG) chatbot that answers questions about a real health insurance plan's drug formulary — coverage tier, quantity limits, and prior-authorization requirements — grounded in the plan's actual PDF document instead of the model's general knowledge.

Built as a final project for an LLM/RAG engineering course.

## Live Demo

🚧 **Hugging Face Space deployment in progress.** Link will be added here once live.

To run it locally in the meantime, see [Setup](#setup) below.

## Overview

Health plan formularies are long, table-heavy PDFs that are painful to search manually. This app parses one into per-drug records, indexes them with hybrid (dense + lexical) retrieval, and answers member questions — with the retrieved rows cited in every answer, and follow-up questions ("does it have quantity limits?") correctly resolved against whichever drug was last discussed.

The knowledge base is a real, public [Oscar Health](https://www.hioscar.com/) 2026 NC standard plan formulary (`data/raw/NC/2026/formulary/`), extracted into 2,394 individual drug records.

## Required API Key

- **Google Gemini API key** — this is the only key the app needs. Get a free one at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey), then paste it into the password field at the top of the app. It is used only for that browser session's requests — never written to disk, never logged, and never read from an environment file by the app itself.

## Cost Estimate

Generation runs on **Gemini 2.5 Flash** (`$0.30` / 1M input tokens, `$2.50` / 1M output tokens as of this writing). Each chat turn makes one main answer call (system prompt + hybrid-retrieved context + question) plus, on follow-up turns only, a small condense-question call to rewrite the question standalone before retrieving.

Generously assuming ~800 input and ~300 output tokens per turn (padding well above what a 3-row context and a short answer actually need):

| Usage | Est. cost |
|---|---|
| One question | ~$0.001 |
| A thorough 30-question test session (covering every example, a multi-turn follow-up conversation, and a bad-key retry) | **~$0.03** |
| To reach $0.50 | ~500 questions |

Trying every feature in this app costs a small fraction of a cent to a few cents — nowhere near the $0.50 budget.

## Optional Functionalities Implemented

This project implements **5 of the course's optional functionalities**:

1. **Domain-specific application, not an AI tutor.** Built for health insurance/healthcare — a real formulary PDF, not the course's own knowledge base.
2. **Streaming responses.** `FormularyChat.ask_stream()` ([src/generation/chat.py](src/generation/chat.py)) yields the answer token-by-token as Gemini generates it; the Gradio UI ([app.py](app.py)) displays it incrementally.
3. **RAG evaluation, with code, dataset, and results in the repo.** See [Evaluation Results](#evaluation-results) below — evaluation script: [notebooks/formulary/04-Evaluate_Formulary_RAG.ipynb](notebooks/formulary/04-Evaluate_Formulary_RAG.ipynb) and [05-Hybrid_Search_Formulary.ipynb](notebooks/formulary/05-Hybrid_Search_Formulary.ipynb); dataset: [data/processed/NC/2026/formulary/eval_set_200.json](data/processed/NC/2026/formulary/eval_set_200.json) (200 template-generated questions, no LLM call, deterministic seed).
4. **Hybrid search.** Dense (FAISS + `sentence-transformers/all-MiniLM-L6-v2`) fused with lexical BM25 via Reciprocal Rank Fusion (`langchain`'s `EnsembleRetriever`) — [src/retrieval/hybrid.py](src/retrieval/hybrid.py). Measured, not assumed: see results below.
5. **PDF-based data collection and curation, with the parsing code in the repo.** Table-aware extraction (`pdfplumber`) keeps each drug's section/name/tier/requirements together as one record instead of fragmenting a table across fixed-size text chunks — [src/ingestion/pdf_parser.py](src/ingestion/pdf_parser.py).

## Evaluation Results

Retrieval quality was measured with **Hit Rate** (is the correct row anywhere in the top-k?) and **MRR** (how high is it ranked?), over 200 questions template-generated directly from indexed rows' own metadata — no LLM involved in building the eval set, so it can't contain a hallucinated question. Full methodology: [notebooks/formulary/04-Evaluate_Formulary_RAG.ipynb](notebooks/formulary/04-Evaluate_Formulary_RAG.ipynb).

[05-Hybrid_Search_Formulary.ipynb](notebooks/formulary/05-Hybrid_Search_Formulary.ipynb) then compared vector-only retrieval against hybrid (dense + BM25) on the *identical* eval set:

| top_k | Vector-only Hit Rate | Hybrid Hit Rate | Vector-only MRR | Hybrid MRR |
|---|---|---|---|---|
| 1 | 78.5% | 78.5% | 0.785 | 0.785 |
| 2 | 90.0% | 93.5% | 0.843 | 0.895 |
| **3 (configured)** | **93.5%** | **98.0%** | **0.854** | **0.923** |
| 5 | 98.0% | 100.0% | 0.864 | 0.943 |
| 10 | 98.0% | 100.0% | 0.864 | 0.938 |

At the app's configured `top_k=3`, hybrid search lifted Hit Rate from 93.5% to 98.0% and MRR from 0.854 to 0.923, at zero added cost — BM25 is CPU-only statistical term matching, no model, no API call. The gain makes sense for this data: a drug name + strength (e.g. `"acetaminophen w/ codeine tab 300-15 mg"`) is a precise lexical string, exactly what exact-term matching is good at and what embeddings sometimes blur across similar drugs in the same class.

## How It Works

```
PDF (raw formulary)
  → pdfplumber table extraction, one Document per drug row     (src/ingestion/pdf_parser.py)
  → embed with sentence-transformers, cache in FAISS            (src/retrieval/vector_store.py)
  → hybrid retrieval: FAISS (dense) + BM25 (lexical), RRF-fused (src/retrieval/hybrid.py)
  → condense follow-up questions using chat history              (src/generation/chat.py)
  → generate the answer with user-supplied Gemini key, streamed  (src/generation/llm.py, app.py)
  → answer + cited source rows (drug, page, section)
```

Chat memory has no server-side session object: each turn, `app.py` reconstructs the conversation from Gradio's own chat transcript, so there's nothing to leak between concurrent users and "Clear" resets it for free.

## Data Collection & Curation

- **Source:** Oscar Health's 4-Tier NC Standard 2026 plan formulary document (public plan document, `data/raw/NC/2026/formulary/`).
- **Curation code:** [src/ingestion/pdf_parser.py](src/ingestion/pdf_parser.py) (`extract_drug_rows`) — table-aware extraction that keeps each drug's section, name, tier, and requirements together as one record; developed and proven in [notebooks/formulary/02-Basic_RAG_Formulary.ipynb](notebooks/formulary/02-Basic_RAG_Formulary.ipynb).
- **Output:** 2,394 per-drug records, embedded and cached in `data/processed/NC/2026/formulary/` (model-keyed FAISS cache, rebuilds automatically if the embedding model in `config/config.yaml` changes).

## Project Structure

```
app.py                    # Gradio app — entry point for the HF Space
config/config.yaml         # paths, chunking, embedding model, retrieval top_k, LLM settings
src/
  ingestion/pdf_parser.py  # table-aware PDF → Document extraction
  retrieval/
    vector_store.py        # FAISS cache load/build
    hybrid.py               # dense + BM25 hybrid retrieval
  generation/
    prompts.py              # system/answer/condense-question prompt templates
    llm.py                  # Ollama (local dev) + Gemini (deployed app) clients
    chat.py                 # FormularyChat: condense-question memory + streaming
  utils/config.py           # config.yaml loading, project-root-relative paths
scripts/smoke_test.py       # end-to-end verification against the cache + local Ollama
notebooks/formulary/        # the R&D behind src/ — baseline, RAG, LlamaIndex comparison,
                             # evaluation, hybrid search, chat — each proven with a measured
                             # before/after, not just asserted
data/
  raw/NC/2026/formulary/     # the source PDF
  processed/NC/2026/formulary/  # FAISS cache + eval_set_200.json
```

## Setup

```bash
git clone git@github.com:sushiva/healthins-enrollment-assistant-dev.git healthins-enrollment-assistant
cd healthins-enrollment-assistant

# Create the virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Run the app
source .venv/bin/activate
python app.py
```

Open the local URL Gradio prints, paste a Google Gemini API key into the field at the top, and start asking questions.

No `.env` file is needed to run the app itself — the Gemini key is entered in the UI. (`.env`/`GEMINI_API_KEY` is only used by the notebooks and `scripts/smoke_test.py` for local development against Ollama, not by `app.py`.)

## Tech Stack

- **Orchestration:** LangChain (`langchain-classic`, `langchain-community`, `langchain-huggingface`)
- **Vector store:** FAISS (`faiss-cpu`)
- **Lexical retrieval:** BM25 (`rank_bm25`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, no API cost)
- **LLM:** Google Gemini 2.5 Flash, via `langchain-google-genai`, user-supplied API key
- **PDF parsing:** `pdfplumber`
- **UI:** Gradio

## Notebooks

The six notebooks under `notebooks/formulary/` are the development history behind `src/` — each makes one deliberate change and proves it worked before the next notebook builds on it, rather than asserting an improvement:

1. **01 — Basic Formulary Tutor:** plain LLM call, no retrieval (the baseline `src/` improves on).
2. **02 — Basic RAG:** table-aware PDF chunking, cached embeddings, retrieval; proves grounding changes the answer (with-vs-without-context).
3. **03 — LlamaIndex comparison:** the same pipeline rebuilt in LlamaIndex, as a framework-tradeoffs comparison (not part of the shipped app).
4. **04 — Evaluate RAG:** builds the 200-question eval set and measures Hit Rate/MRR vs. `top_k`.
5. **05 — Hybrid Search:** measures dense+BM25 against 04's exact eval set — the numbers behind the Evaluation Results section above.
6. **06 — Add Chat:** condense-question memory + streaming, proven with a real before/after (a pronoun-only follow-up fails without condensation, succeeds with it).

## Roadmap

- Second document type (e.g. Summary of Benefits/Coverage) for the same plan, enabling genuine metadata filtering and query routing.
- Expansion beyond a single state's formulary.

## Author

Sudhir Shivaram

## License

MIT
