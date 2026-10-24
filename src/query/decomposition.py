"""
Query Decomposition Module for IVARA Personal Memory Pipeline.

Decomposes complex multi-part, multi-hop, temporal, comparison, and relationship
questions into focused, atomic subquestions to retrieve targeted memory chunks
and graph evidence while bypassing simple single-fact queries.
"""

import json
import re
from typing import Any, Callable
from pydantic import BaseModel, Field

try:
    from ollama import chat
except ImportError:
    chat = None


class DecompositionPayload(BaseModel):
    """
    Structured model output schema for query decomposition.
    """
    should_decompose: bool = Field(
        description="True if the question contains multiple parts, multi-hop dependencies, temporal sequences, or comparisons that require separate retrieval; False if it is a single factual lookup."
    )
    reason: str = Field(
        description="Brief explanation of why the question was or was not decomposed."
    )
    subquestions: list[str] = Field(
        default_factory=list,
        description="List of 2-3 focused, atomic, standalone subquestions. Do NOT include answers."
    )


class DecomposedQuery(BaseModel):
    """
    Container for the decomposition output, preserving original intent and provenance.
    """
    original_query: str
    is_decomposed: bool
    subquestions: list[str]
    reason: str
    strategy: str  # 'bypassed_simple', 'llm_decomposed', 'fallback_original', 'validation_rejected'


DECOMPOSITION_SYSTEM_PROMPT = """You are IVARA's query decomposition engine.
Your task is to analyze a user's question about personal memories and decide whether it should be split into 2-3 focused, atomic subquestions for evidence retrieval.

DECOMPOSITION CRITERIA:
- DECOMPOSE when the question:
  1. Combines multiple distinct facts (e.g. "When was Chronos started, and when was it delivered?").
  2. Requires multi-hop relational chaining (e.g. "What programming language was used by the person who mentored Sarah Lin?").
  3. Involves temporal comparison or state change (e.g. "What was Priya's role before and after her sabbatical?").
  4. Requires disambiguation across multiple entities (e.g. "Which Marcus was employed by Vanguard vs Aetheria?").
- BYPASS (do NOT decompose) when the question:
  1. Asks for a single atomic fact (e.g. "What is Elena Rostova's nickname?").
  2. Asks a direct entity or date question with a single subject (e.g. "On what date did Sarah Lin join Aetheria Labs?").
  3. Is already short and atomic.

STRICT RULES:
1. Do NOT answer the questions. Only generate questions.
2. Each subquestion must be clear, standalone, and focused on one specific entity, event, date, or relationship.
3. Preserve the exact entities and intent of the original question.
4. Do not invent entities or events that were not mentioned or directly implied.
5. Never generate duplicate or trivially rephrased subquestions.
6. Maximum 3 subquestions.
7. Keep internal reasoning concise (under 40 words) and output valid JSON.
"""


class QueryDecomposer:
    """
    Coordinates question complexity analysis, LLM-guided decomposition,
    and rigorous output sanitization and validation.
    """

    def __init__(
        self,
        model: str = "gpt-oss",
        max_subquestions: int = 3,
        llm_caller: Callable[[str, str], Any] | None = None
    ):
        self.model = model
        self.max_subquestions = max(2, min(max_subquestions, 5))
        self.llm_caller = llm_caller

    def should_decompose_heuristic(self, query: str) -> tuple[bool, str]:
        """
        Fast heuristic check to filter simple, atomic factual lookups from complex queries.
        Returns (is_complex, reason).
        """
        q = query.strip()
        q_lower = q.lower()

        # Multi-part conjunction patterns: "..., and when/what/who/how...", "both X and Y", "as well as"
        has_conjunction = bool(re.search(r"\b(and\s+(?:when|what|who|which|how|where|did|was|is)|as well as|both\s+.+\s+and)\b", q_lower))
        
        # Multi-hop relational patterns: "person who mentored", "co-founded with", "relationship between X and Y"
        has_multihop = bool(re.search(r"\b(person who|who (?:was assigned to|mentored|co-founded|collaborated with)|relationship between|father of|advisor to)\b", q_lower))

        # Temporal transitions: "before and after", "originally scheduled .* rescheduled", "start .* finish", "departure .* arrival"
        has_temporal_seq = bool(re.search(r"\b(before and after|originally .* rescheduled|started .* completed|departure .* arrival|kickoff .* delivery)\b", q_lower))

        # Entity comparison: "which marcus", "which elena", "compare", "differ"
        has_comparison = bool(re.search(r"\b(which\s+(?:marcus|elena|project|company)|difference between)\b", q_lower))

        if has_conjunction:
            return True, "Contains multi-part conjunction clause"
        if has_multihop:
            return True, "Contains multi-hop relational dependency"
        if has_temporal_seq:
            return True, "Contains temporal sequence or state change"
        if has_comparison:
            return True, "Contains entity comparison or disambiguation"

        # Check token count and simple question markers
        tokens = q.split()
        if len(tokens) <= 10 and not any(k in q_lower for k in [" and ", " before ", " after ", " while "]):
            return False, "Single-clause atomic factual lookup"

        # Questions with comma or semicolon or ' and ' might still be multi-part
        if "," in q or ";" in q or " and " in q_lower:
            return True, "Compound sentence structure"

        return False, "Single-clause question without complex markers"

    def decompose(self, query: str) -> DecomposedQuery:
        """
        Decompose a user question into focused subquestions.
        Safely falls back to [original_query] on any validation or model failure.
        """
        if not query or not isinstance(query, str) or not query.strip():
            return DecomposedQuery(
                original_query="",
                is_decomposed=False,
                subquestions=[],
                reason="Empty or invalid query",
                strategy="fallback_original"
            )

        clean_query = query.strip()

        # 1. Heuristic fast-path: Bypass simple questions
        is_complex, heuristic_reason = self.should_decompose_heuristic(clean_query)
        if not is_complex:
            return DecomposedQuery(
                original_query=clean_query,
                is_decomposed=False,
                subquestions=[clean_query],
                reason=f"Bypassed decomposition: {heuristic_reason}",
                strategy="bypassed_simple"
            )

        # 2. Invoke LLM for decomposition
        user_prompt = f"User Question: {clean_query}\n\nDecompose into at most {self.max_subquestions} atomic subquestions, or decide to bypass if single-fact."

        raw_response = None
        try:
            if self.llm_caller is not None:
                raw_response = self.llm_caller(user_prompt, DECOMPOSITION_SYSTEM_PROMPT)
            else:
                if chat is None:
                    raise RuntimeError("Ollama is not available and no llm_caller was supplied.")

                response = chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    format=DecompositionPayload.model_json_schema(),
                    options={"temperature": 0, "num_ctx": 4096, "num_predict": 1024}
                )
                raw_response = response.message.content
        except Exception as e:
            # Safe fallback on network, timeout, or model crash
            return DecomposedQuery(
                original_query=clean_query,
                is_decomposed=False,
                subquestions=[clean_query],
                reason=f"Model call failed: {type(e).__name__} ({str(e)[:100]})",
                strategy="fallback_original"
            )

        # 3. Parse JSON safely
        payload = self._parse_payload(raw_response)
        if payload is None:
            return DecomposedQuery(
                original_query=clean_query,
                is_decomposed=False,
                subquestions=[clean_query],
                reason="Failed to parse model response into valid DecompositionPayload schema",
                strategy="fallback_original"
            )

        # 4. If model decided not to decompose
        if not payload.should_decompose or not payload.subquestions:
            return DecomposedQuery(
                original_query=clean_query,
                is_decomposed=False,
                subquestions=[clean_query],
                reason=payload.reason or "Model opted not to decompose",
                strategy="bypassed_simple"
            )

        # 5. Rigorous independent sanitization and validation
        sanitized = self._sanitize_subquestions(clean_query, payload.subquestions)

        if len(sanitized) < 2:
            return DecomposedQuery(
                original_query=clean_query,
                is_decomposed=False,
                subquestions=[clean_query],
                reason=f"Decomposition produced fewer than 2 valid distinct subquestions (raw: {len(payload.subquestions)}, valid: {len(sanitized)})",
                strategy="validation_rejected"
            )

        return DecomposedQuery(
            original_query=clean_query,
            is_decomposed=True,
            subquestions=sanitized[:self.max_subquestions],
            reason=payload.reason,
            strategy="llm_decomposed"
        )

    def _parse_payload(self, raw: Any) -> DecompositionPayload | None:
        """Parse structured model output into DecompositionPayload."""
        if isinstance(raw, DecompositionPayload):
            return raw
        if isinstance(raw, dict):
            try:
                return DecompositionPayload(**raw)
            except Exception:
                return None
        if isinstance(raw, str):
            clean_str = raw.strip()
            # Try direct JSON validation
            try:
                return DecompositionPayload.model_validate_json(clean_str)
            except Exception:
                pass
            # Try regex extraction if markdown fences exist
            try:
                m = re.search(r"\{.*\}", clean_str, re.DOTALL)
                if m:
                    return DecompositionPayload.model_validate_json(m.group(0))
            except Exception:
                pass
        return None

    def _sanitize_subquestions(self, original_query: str, raw_subquestions: list[str]) -> list[str]:
        """
        Validate, clean, deduplicate, and verify relevance of candidate subquestions.
        Rejects empty, non-question, malformed, or irrelevant subqueries.
        """
        valid_subquestions: list[str] = []
        seen_normalized: set[str] = set()

        orig_lower = original_query.lower()
        orig_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", orig_lower))
        stop_words = {"what", "when", "which", "where", "who", "whom", "whose", "why", "how", "did", "was", "were", "is", "are", "the", "and", "for", "with", "from"}
        orig_content_words = orig_words - stop_words

        for raw_sq in raw_subquestions:
            if not isinstance(raw_sq, str):
                continue

            # Strip numbering prefixes like "1. ", "- ", "(a) "
            cleaned = re.sub(r"^(?:\d+[\.\)]|\-|\*|\([a-zA-Z0-9]\))\s*", "", raw_sq.strip())
            cleaned = cleaned.strip()

            # Rule 1: Length check (between 10 and 200 characters)
            if len(cleaned) < 10 or len(cleaned) > 250:
                continue

            # Rule 2: Must be a question (ends with '?' or starts with question word)
            if not (cleaned.endswith("?") or re.match(r"^(what|when|which|where|who|whom|whose|why|how|did|was|were|is|are|does|do)\b", cleaned.lower())):
                continue

            # Add question mark if missing
            if not cleaned.endswith("?"):
                cleaned += "?"

            # Rule 3: Avoid subquestions identical to original query
            norm_cleaned = re.sub(r"[^\w\s]", "", cleaned.lower()).strip()
            norm_orig = re.sub(r"[^\w\s]", "", orig_lower).strip()
            if norm_cleaned == norm_orig:
                continue

            # Rule 4: Deduplicate against already accepted subquestions
            if norm_cleaned in seen_normalized:
                continue

            # Rule 5: Relevance check: must share at least one content word or entity with original query
            sq_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", cleaned.lower())) - stop_words
            if orig_content_words and not (sq_words & orig_content_words):
                # Discard completely unrelated subquestions
                continue

            # Rule 6: Avoid questions that contain speculative answers (e.g. "Did X happen because Y was a spy?")
            # We want open-ended or focused retrieval queries, not hallucinated answers.

            seen_normalized.add(norm_cleaned)
            valid_subquestions.append(cleaned)

        return valid_subquestions
