"""Formulary chat session: condense-question memory + hybrid retrieval + streaming.

Ported from notebooks/formulary/06-Add_Chat_Formulary.ipynb §4, with one
deliberate change: session state (`chat_history`) is wrapped in a class
instead of a module-level list, so each Gradio user gets an isolated
FormularyChat instance instead of sharing one global history across
concurrent users. Retrieval/generation logic is otherwise unchanged.

Step 2 adds `ask_stream()` (a generator, for live token-by-token display in
the Gradio UI) and `history_from_transcript()` (to seed a FormularyChat's
memory from Gradio's own chat transcript). Both are additive — `ask()` is
untouched, verified by scripts/smoke_test.py still passing unchanged.
"""
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from src.generation.prompts import CONDENSE_PROMPT, PROMPT_TEMPLATE, SYSTEM_INSTRUCTION


def format_sources(docs) -> str:
    # De-dupe by drug name — a row can legitimately be retrieved more than once
    # across different questions, but never worth citing twice for one answer.
    seen = set()
    lines = []
    for doc in docs:
        drug = doc.metadata.get("drug", "unknown")
        if drug in seen:
            continue
        seen.add(drug)
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section") or "N/A"
        lines.append(f"- {drug} — p.{page}, {section}")
    return "\n".join(lines)


def format_history(history: list[tuple[str, str]], max_turns: int = 3) -> str:
    return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-max_turns:])


def history_from_transcript(gradio_history: list[dict]) -> list[tuple[str, str]]:
    """Convert a gr.ChatInterface transcript (type="messages": a flat list of
    {"role": "user"/"assistant", "content": str} dicts) into the (question,
    answer) pairs FormularyChat.history expects.

    Strips each assistant turn's "\\n\\nSources:\\n..." footer first, to match
    what notebook 06 actually stored in chat_history — the answer only, no
    citations — since the footer is UI-facing, not part of the condensation
    context.
    """
    pairs: list[tuple[str, str]] = []
    pending_question = None
    for turn in gradio_history:
        if turn.get("role") == "user":
            pending_question = turn.get("content", "")
        elif turn.get("role") == "assistant" and pending_question is not None:
            answer = turn.get("content", "").split("\n\nSources:\n", 1)[0]
            pairs.append((pending_question, answer))
            pending_question = None
    return pairs


class FormularyChat:
    """One conversation: a retriever + LLM + its own (question, answer) history."""

    def __init__(self, llm: BaseChatModel, build_retriever: Callable[[int], BaseRetriever], default_k: int):
        """
        llm: a chat model exposing .invoke() / .stream() (e.g. ChatOllama)
        build_retriever: callable(k) -> BaseRetriever, e.g.
            functools.partial(build_hybrid_retriever, vectorstore, bm25_retriever)
        default_k: config["retrieval"]["top_k"], used when ask()'s k is omitted
        """
        self.llm = llm
        self.build_retriever = build_retriever
        self.default_k = default_k
        self.history: list[tuple[str, str]] = []

    def condense_question(self, question: str) -> str:
        # Only calls the LLM when there's actual history — the first turn of any
        # conversation is passed straight through, so a one-off question costs
        # exactly what it costs today.
        if not self.history:
            return question
        prompt = CONDENSE_PROMPT.format(history=format_history(self.history), question=question)
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()

    def _prepare(self, question: str, k: int | None):
        """Condense + retrieve + build the messages for one turn. Shared by
        ask() and ask_stream() — neither the LLM call nor history-append
        happens here, since the two differ in exactly (and only) that."""
        k = k or self.default_k
        standalone_question = self.condense_question(question)
        retriever = self.build_retriever(k)
        docs = retriever.invoke(standalone_question)[:k]
        context = "\n\n".join(d.page_content for d in docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=standalone_question)
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        return docs, messages

    def ask(self, question: str, k: int | None = None, stream: bool = True):
        docs, messages = self._prepare(question, k)

        if stream:
            full_answer = ""
            for chunk in self.llm.stream(messages):
                full_answer += chunk.content
        else:
            full_answer = self.llm.invoke(messages).content

        self.history.append((question, full_answer))
        answer_with_sources = f"{full_answer}\n\nSources:\n{format_sources(docs)}"
        return answer_with_sources, docs

    def ask_stream(self, question: str, k: int | None = None):
        """Generator version of ask(stream=True): yields the answer-so-far
        (growing string) as tokens arrive, ending with the sources footer
        appended — for live display in a Gradio chat UI."""
        docs, messages = self._prepare(question, k)

        full_answer = ""
        for chunk in self.llm.stream(messages):
            full_answer += chunk.content
            yield full_answer, docs

        self.history.append((question, full_answer))
        yield f"{full_answer}\n\nSources:\n{format_sources(docs)}", docs

    def reset(self):
        self.history.clear()
