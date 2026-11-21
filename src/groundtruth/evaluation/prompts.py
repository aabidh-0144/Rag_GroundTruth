"""Every Mode 1 judge prompt, and nothing else.

Kept apart from the metric code so prompts can be read, compared and tuned without
touching scoring logic. Each prompt asks for JSON only; the shape it asks for is the
pydantic model the matching metric validates against (see reference_free/).

Contexts are rendered as numbered blocks [1], [2], ... - the same convention the RAG
generator cites with - so a judge's evidence numbers line up with the answer's
citations.
"""

from __future__ import annotations

JSON_ONLY = "Respond with a single JSON object and nothing else - no prose, no code fences."


def render_contexts(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i}] {text.strip()}" for i, text in enumerate(contexts, start=1))


# --- Faithfulness ----------------------------------------------------------------

FAITHFULNESS_SYSTEM = f"""You are a strict fact-checker for a retrieval-augmented QA system.
You judge whether an ANSWER is supported by the CONTEXT blocks it was given. You do not
use outside knowledge: a claim that is true in the real world but absent from the
context is UNSUPPORTED.

Steps:
1. Split the answer into atomic factual claims (one fact each). Ignore citation markers
   like [1] and ignore filler with no factual content.
2. Label every claim:
   - "supported": the context states it, or it follows directly from the context
   - "unsupported": the context does not state it (possible hallucination)
   - "contradicted": the context states something incompatible with it
3. For supported claims, list the block numbers that support them.

{JSON_ONLY}
Schema:
{{"claims": [{{"claim": str, "verdict": "supported" | "unsupported" | "contradicted",
               "evidence": [int], "explanation": str}}],
  "summary": str}}"""

FAITHFULNESS_USER = """CONTEXT:
{contexts}

QUESTION:
{question}

ANSWER:
{answer}"""


# --- Answer relevance -------------------------------------------------------------

ANSWER_RELEVANCE_SYSTEM = f"""You evaluate whether an ANSWER directly addresses a QUESTION.
Judge relevance and completeness only - not factual correctness, which is checked
separately.

Score on this scale:
5 - fully and directly answers every part of the question, no irrelevant content
4 - answers the question, with minor gaps or a little irrelevant content
3 - partially answers: a noticeable part of the question is missed
2 - mostly off-target: touches the topic but does not answer what was asked
1 - does not address the question at all

{JSON_ONLY}
Schema:
{{"score": 1 | 2 | 3 | 4 | 5,
  "missing_parts": [str],
  "irrelevant_parts": [str],
  "explanation": str}}"""

ANSWER_RELEVANCE_USER = """QUESTION:
{question}

ANSWER:
{answer}"""


# --- Context relevance ------------------------------------------------------------

CONTEXT_RELEVANCE_SYSTEM = f"""You evaluate a retrieval system. For each numbered CONTEXT
block, decide whether it helps answer the QUESTION:
- "relevant": contains information needed to answer the question
- "partial": on-topic but only marginally useful, or useful only in part
- "irrelevant": does not help answer the question

Judge every block, in order, exactly once.

{JSON_ONLY}
Schema:
{{"blocks": [{{"block": int, "verdict": "relevant" | "partial" | "irrelevant",
               "explanation": str}}],
  "summary": str}}"""

CONTEXT_RELEVANCE_USER = """QUESTION:
{question}

CONTEXT:
{contexts}"""


# --- Repair -----------------------------------------------------------------------

REPAIR_USER = """Your previous reply was not valid for the required schema:

{error}

Your previous reply was:
{reply}

Reply again with ONLY the corrected JSON object."""
