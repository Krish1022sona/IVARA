"""
Experiment A: Calibrated Answering with Context Budgeting and Ollama num_ctx=8192.

Features:
1. Configured Ollama num_ctx=8192, num_predict=1024, temperature=0.
2. Query-scoped contradiction filtering (only inject contradictions relevant to query).
3. Bounded temporal claim injection (scoped to retrieved chunks and query entities, max 10 claims).
4. Strict prompt character budget guard (max 12,000 chars / ~3,000 tokens).
5. Robust JSON recovery: extracts 'answer' text even if minor schema fields are missing,
   avoiding empty synthesis fallbacks.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.answering import (
    ANSWERING_SYSTEM_PROMPT,
    AnswerPayload,
    validate_citations,
    _normalize_citation_id
)
from src.reasoning import (
    ContradictionAnalyzer,
    EvidenceIntegrator,
    EvidenceGate,
    GatingDecision,
    EvidenceGateReport
)

try:
    from ollama import chat
except ImportError:
    chat = None


def filter_query_relevant_contradictions(
    query: str,
    unresolved_conflicts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Filter unresolved contradictions to only those relevant to the query concepts.
    Prevents unrelated domain contradictions (e.g. beverage preferences) from polluting
    unrelated queries (e.g. commute time or degree).
    """
    if not unresolved_conflicts:
        return []

    q_words = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query.lower()))
    relevant = []

    for uc in unresolved_conflicts:
        pred = str(uc.get("predicate", "")).lower()
        subj = str(uc.get("subject", "")).lower()
        vals = [str(v).lower() for v in uc.get("values", [])]

        # Check overlap with query terms
        match = False
        if any(w in pred for w in q_words) or any(p in q_words for p in pred.split("_")):
            match = True
        elif any(w in subj for w in q_words):
            match = True
        elif any(any(w in val for w in q_words) for val in vals):
            match = True

        if match:
            relevant.append(uc)

    return relevant


def format_user_prompt_budgeted(
    query: str,
    context: dict[str, Any],
    gate_report: EvidenceGateReport | None = None,
    max_prompt_chars: int = 12000
) -> str:
    """
    Format evidence context with strict character budgeting to prevent context overflow.
    """
    context_text = context.get("context_text", "").strip()
    special_guidance = ""

    routing_info = context.get("diagnostics", {}).get("routing", {})
    route_cat = (routing_info.get("query_type") or routing_info.get("category")) if isinstance(routing_info, dict) else None

    if route_cat == "historical_state":
        special_guidance += (
            "\n\nSPECIAL INSTRUCTION: This is a historical-state question. Answer with the past/prior "
            "state that was valid at the specified point in time, rather than subsequent or current states."
        )
    elif route_cat == "negative_information":
        special_guidance += (
            "\n\nSPECIAL INSTRUCTION: If there is no mention of the queried topic in the evidence, "
            "state that there is no record of it in the recorded conversations (absence-of-evidence framing)."
        )
    elif route_cat == "cross_session_synth":
        special_guidance += (
            "\n\nSPECIAL INSTRUCTION: Synthesize events chronologically across all relevant sessions and provide timestamps."
        )

    if gate_report:
        if gate_report.decision == GatingDecision.EXPLAIN_CONTRADICTION:
            special_guidance += (
                "\n\nSPECIAL INSTRUCTION: Unresolved contradictions exist in the evidence. "
                "Do NOT invent certainty or choose one arbitrarily. Set status to 'conflicting_evidence', explain the discrepancy, and cite both memories."
            )
        elif gate_report.decision == GatingDecision.PARTIAL_ANSWER:
            supp_str = ", ".join(f"'{s}'" for s in gate_report.supported_subclaims)
            unsupp_str = ", ".join(f"'{u}'" for u in gate_report.unsupported_subclaims)
            special_guidance += (
                f"\n\nSPECIAL INSTRUCTION: Partial evidence detected. The query asks about '{query}', "
                f"but evidence only supports {supp_str}. Answer the supported portion, set status to 'partial', and state that {unsupp_str} are not recorded."
            )

    # Budget context_text to preserve safe headroom in num_ctx=8192
    if len(context_text) > max_prompt_chars:
        context_text = context_text[:max_prompt_chars - 100] + "\n[... Context bounded to preserve token budget ...]"

    return f"""<evidence_context>
{context_text}
</evidence_context>

User Question: {query}{special_guidance}

Please provide an evidence-grounded answer adhering strictly to the system rules."""


def answer_question_budgeted(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    graph_store: Any = None,
    enable_temporal: bool = True,
    model: str = "gpt-oss",
    num_ctx: int = 8192,
    num_predict: int = 1024,
    llm_caller: Callable[[str, str], Any] | None = None
) -> dict[str, Any]:
    """
    Evidence-grounded answering with isolated context budgeting and 8192-token context window.
    """
    clean_query = query.strip()
    from src.chunking.retrieval import assemble_bounded_context
    context = assemble_bounded_context(clean_query, retrieved_chunks, max_context_tokens=1500)

    # 1. Abstain if no evidence
    if not context.get("has_sufficient_evidence") or not context.get("direct_evidence"):
        return {
            "status": "insufficient_evidence",
            "answer": f"I don't have sufficient recorded memory to answer '{clean_query}'.",
            "citations": [],
            "rejected_citation_ids": [],
            "inferred_claims": [],
            "limitations": "No relevant memories met the retrieval relevance threshold.",
            "retrieved_context": context,
            "contradiction_analysis": {},
            "raw_response_len": 0,
            "context_overflow": False
        }

    # 2. Bounded Temporal Claims & Contradiction Resolution
    active_claims = []
    if enable_temporal:
        from src.temporal.claim import build_claims_from_text
        from src.temporal.supersession import TemporalSupersessionEngine
        from src.temporal.resolution import QueryTemporalResolver

        # Scope claim extraction ONLY to retrieved chunks (prevents cross-session combinatorial explosion)
        extracted = []
        for ev in context.get("direct_evidence", []):
            ev_text = ev.get("text", "")
            ev_id = ev.get("chunk_id") or ev.get("memory_id", "")
            ev_ts = ev.get("timestamp")
            if ev_text:
                extracted.extend(build_claims_from_text(ev_text, session_id=ev_id, chunk_id=ev_id, recorded_at=ev_ts))

        if extracted:
            engine = TemporalSupersessionEngine()
            active_claims = engine.process_claims(extracted)

        # Cap active claims to top 10 most relevant
        active_claims = active_claims[:10]

        if active_claims and context.get("context_text"):
            resolver = QueryTemporalResolver()
            resolved_temporal = resolver.resolve(clean_query, active_claims)
            formatted_temp = resolver.format_temporal_context(resolved_temporal)
            if formatted_temp:
                context["context_text"] = f"{context['context_text']}\n\n{formatted_temp}"
            context["temporal_resolution"] = resolved_temporal

    # 3. Query-Scoped Contradiction Filtering
    contradiction_analyzer = ContradictionAnalyzer()
    raw_contradiction_report = contradiction_analyzer.analyze_claims(active_claims)
    unresolved = raw_contradiction_report.get("unresolved_conflicts", [])
    relevant_unresolved = filter_query_relevant_contradictions(clean_query, unresolved)

    contradiction_report = dict(raw_contradiction_report)
    contradiction_report["unresolved_conflicts"] = relevant_unresolved
    context["contradiction_analysis"] = contradiction_report

    if relevant_unresolved:
        conflict_lines = ["\n--- UNRESOLVED CONTRADICTIONS IN EVIDENCE ---"]
        for uc in relevant_unresolved:
            conflict_lines.append(f"- Conflict on {uc['predicate']}: asserted values {uc['values']}. ({uc.get('explanation', '')})")
        context["context_text"] = f"{context.get('context_text', '')}\n" + "\n".join(conflict_lines)

    # 4. Evidence Gating
    evidence_gate = EvidenceGate()
    gate_report = evidence_gate.evaluate_pre_generation(
        clean_query,
        context.get("direct_evidence", []),
        contradiction_summary=contradiction_report
    )
    context["evidence_gate"] = gate_report.__dict__

    # 5. Format prompt
    user_prompt = format_user_prompt_budgeted(clean_query, context, gate_report=gate_report, max_prompt_chars=12000)
    prompt_token_est = int(len(user_prompt.split()) / 0.75)

    # 6. LLM Call
    raw_response = ""
    call_error = None
    if llm_caller is not None:
        raw_response = llm_caller(user_prompt, ANSWERING_SYSTEM_PROMPT)
    else:
        if chat is None:
            raise RuntimeError("Ollama chat is unavailable.")
        try:
            response = chat(
                model=model,
                messages=[
                    {"role": "system", "content": ANSWERING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                format=AnswerPayload.model_json_schema(),
                options={
                    "temperature": 0,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict
                }
            )
            raw_response = response.message.content or ""
        except Exception as e:
            call_error = str(e)

    raw_response_len = len(raw_response.strip()) if raw_response else 0
    is_empty_response = (raw_response_len == 0)

    # 7. Robust Parsing & Recovery
    payload = None
    if raw_response:
        if isinstance(raw_response, AnswerPayload):
            payload = raw_response
        elif isinstance(raw_response, dict):
            try:
                payload = AnswerPayload(**raw_response)
            except Exception:
                payload = None
        elif isinstance(raw_response, str):
            try:
                payload = AnswerPayload.model_validate_json(raw_response)
            except Exception:
                # Try JSON regex recovery
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(0), strict=False)
                        # Extract answer if present
                        if "answer" in data and isinstance(data["answer"], str) and data["answer"].strip():
                            status_val = data.get("status", "answered")
                            c_ids = data.get("cited_memory_ids", [])
                            payload = AnswerPayload(
                                status=status_val if status_val in ("answered", "partial", "insufficient_evidence", "unanswerable", "conflicting_evidence") else "answered",
                                answer=data["answer"].strip(),
                                cited_memory_ids=c_ids if isinstance(c_ids, list) else [],
                                limitations=data.get("limitations")
                            )
                    except Exception:
                        payload = None

    # Fallback handling
    if payload is None:
        # If raw text contains an answer but failed JSON schema
        clean_text = raw_response.strip()
        # Strip outer JSON syntax if raw JSON string
        if clean_text.startswith('{') and '"answer":' in clean_text:
            ans_match = re.search(r'"answer"\s*:\s*"([^"]+)"', clean_text)
            if ans_match:
                clean_text = ans_match.group(1).strip()

        return {
            "status": "partial" if clean_text else "insufficient_evidence",
            "answer": clean_text if clean_text else f"Could not reliably synthesize an answer for '{clean_query}'.",
            "citations": [],
            "rejected_citation_ids": [],
            "inferred_claims": [],
            "limitations": f"Model output parsing error: {call_error or 'malformed payload'}",
            "retrieved_context": context,
            "raw_response_len": raw_response_len,
            "is_empty_response": is_empty_response,
            "prompt_token_est": prompt_token_est,
            "context_overflow": is_empty_response and (prompt_token_est > num_ctx)
        }

    # Citations
    candidate_ids = list(payload.cited_memory_ids)
    inline_ids = re.findall(r"\b(m\d{3}|chunk_[a-zA-Z0-9_]+|sess_[a-zA-Z0-9_]+)\b", payload.answer)
    for iid in inline_ids:
        if iid not in candidate_ids:
            candidate_ids.append(iid)

    valid_citations, rejected_ids = validate_citations(candidate_ids, context)

    final_status = payload.status
    if final_status not in ("answered", "partial", "insufficient_evidence", "unanswerable", "conflicting_evidence"):
        final_status = "answered"

    return {
        "status": final_status,
        "answer": payload.answer.strip(),
        "citations": valid_citations,
        "rejected_citation_ids": rejected_ids,
        "inferred_claims": [],
        "limitations": payload.limitations,
        "retrieved_context": context,
        "raw_response_len": raw_response_len,
        "is_empty_response": False,
        "prompt_token_est": prompt_token_est,
        "context_overflow": False
    }

