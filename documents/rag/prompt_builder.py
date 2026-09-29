import logging

logger = logging.getLogger(__name__)


def build_grounded_prompt(question, context, sources_text):
    return f"""Answer the question using ONLY the provided context.

Rules:
- ALWAYS respond in the SAME LANGUAGE as the question (e.g., if the question is in Persian/Farsi, respond in Persian)
- Give a SHORT, DIRECT answer for simple questions (who, what, where, when)
- Only provide detailed answers when the question explicitly asks for details
- Do NOT dump the entire document - only extract what's relevant
- Use clean formatting: use bullet points or numbered lists for multiple items
- Reference sources like [Source 1] when citing
- If the answer is not in the context, say: "I could not find the answer in the uploaded documents."
- For Persian responses, use proper Persian punctuation and formatting

Context:
{context}

Sources:
{sources_text}

Question:
{question}
"""


def build_refusal_instruction():
    return "If the context does not contain enough information to answer the question, clearly state that you cannot find the answer in the uploaded documents."


def build_citation_instruction():
    return "When citing information, reference the source using [Source N] format where N is the source number."