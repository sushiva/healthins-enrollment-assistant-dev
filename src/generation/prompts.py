"""Prompt templates for the formulary assistant.

Ported as-is from notebooks/formulary/02-Basic_RAG_Formulary.ipynb §4 and
06-Add_Chat_Formulary.ipynb §4.
"""

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant answering questions about health insurance plans. "
    "Use only the provided context to answer. If the context doesn't contain the answer, say you don't know."
)

# Course-standard augmentation template (02-Basic_RAG.ipynb): context inside explicit boundary tags.
PROMPT_TEMPLATE = (
    "Read the following informations that might contain the context you require to "
    "answer the question. You can use the informations starting from the "
    "<START_OF_CONTEXT> tag and end with the <END_OF_CONTEXT> tag. Here is the content:\n\n"
    "<START_OF_CONTEXT>\n{context}\n<END_OF_CONTEXT>\n\n"
    "Please provide an informative and accurate answer to the following question based "
    "on the available context. Be concise and take your time.\nQuestion: {question}\nAnswer:"
)

CONDENSE_PROMPT = (
    "Given the conversation history and a follow-up question, rewrite the follow-up "
    "question to be a standalone question that includes any necessary context from the "
    "history (such as the drug name being discussed). If the follow-up question is "
    "already standalone, return it unchanged. Output only the rewritten question, "
    "nothing else — no preamble, no quotes.\n\n"
    "Chat History:\n{history}\n\n"
    "Follow-up Question: {question}\n"
    "Standalone Question:"
)
