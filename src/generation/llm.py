"""LLM client construction.

get_ollama_llm: ported from notebooks/formulary/02-Basic_RAG_Formulary.ipynb §4 /
06-Add_Chat_Formulary.ipynb §3 — local Ollama, used by the notebooks and by
scripts/smoke_test.py. Not used by the deployed app (no Ollama on HF Spaces).

get_gemini_llm: the deployed app's LLM (Step 2) — built fresh per request from
the API key the user pastes into the Gradio UI. The key is never read from
.env or written anywhere; it only ever exists in this in-memory client.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


def get_ollama_llm(config: dict) -> ChatOllama:
    return ChatOllama(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        num_predict=config["llm"]["max_tokens"],
    )


def get_gemini_llm(api_key: str, config: dict) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config["llm"]["gemini_model"],
        google_api_key=api_key,
        temperature=config["llm"]["temperature"],
        max_output_tokens=config["llm"]["max_tokens"],
    )
