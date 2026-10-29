"""
Evidence Sufficiency Gating and Provenance Validator for IVARA.

Enforces deterministic evidence boundaries before and after answer synthesis:
- Gating Decision: PROCEED, PARTIAL_ANSWER, ABSTAIN_INSUFFICIENT, EXPLAIN_CONTRADICTION.
- Subclaim validation: identifies supported vs unsupported parts of complex questions.
- Contradiction surfacing: forces explicit qualification when evidence is in conflict.
- Verifiable citation verification: rejects hallucinated / non-existent memory and chunk IDs.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any

from .evidence_integrator import EvidenceRole


class GatingDecision(str, Enum):
    PROCEED = "proceed"
    PARTIAL_ANSWER = "partial_answer"
    ABSTAIN_INSUFFICIENT = "abstain_insufficient"
    EXPLAIN_CONTRADICTION = "explain_contradiction"


@dataclass
class EvidenceGateReport:
    """Pre-generation assessment of evidence sufficiency for answering the query."""
    decision: GatingDecision
    supported_subclaims: list[str] = field(default_factory=list)
    unsupported_subclaims: list[str] = field(default_factory=list)
    unresolved_conflicts: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    usable_chunk_ids: list[str] = field(default_factory=list)


class EvidenceGate:
    """
    Deterministic gate that evaluates whether retrieved evidence is sufficient,
    contradictory, partial, or absent before synthesis, and verifies citations post-synthesis.
    """

    def __init__(self, min_query_term_coverage: float = 0.30):
        self.min_query_term_coverage = min_query_term_coverage

    def _extract_query_clauses(self, query: str) -> list[str]:
        """Split compound queries connected by 'and', 'also', 'as well as'."""
        cleaned = re.sub(r'[\?.,!]', '', query).strip()
        parts = re.split(r'\b(?:and\s+also|and|also|as\s+well\s+as|before|after)\b', cleaned, flags=re.IGNORECASE)
        clauses = [p.strip() for p in parts if len(p.strip()) > 5]
        return clauses if clauses else [cleaned]

    def evaluate_pre_generation(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        contradiction_summary: dict[str, Any] | None = None
    ) -> EvidenceGateReport:
        """
        Evaluate evidence sufficiency before calling LLM.
        """
        if not retrieved_chunks:
            return EvidenceGateReport(
                decision=GatingDecision.ABSTAIN_INSUFFICIENT,
                rationale="Zero relevant chunks retrieved in personal memory store.",
                usable_chunk_ids=[]
            )

        # 1. Check for active unresolved contradictions
        unresolved = (contradiction_summary or {}).get("unresolved_conflicts", [])
        if unresolved:
            return EvidenceGateReport(
                decision=GatingDecision.EXPLAIN_CONTRADICTION,
                unresolved_conflicts=unresolved,
                rationale="Unresolved contradictions detected in active memory records. Must surface qualification.",
                usable_chunk_ids=[c.get("chunk_id", "") for c in retrieved_chunks]
            )

        # 2. Check query clause coverage
        clauses = self._extract_query_clauses(query)
        supported_clauses = []
        unsupported_clauses = []

        stopwords = {
            "what", "when", "where", "which", "how", "who", "did", "was", "were",
            "the", "have", "you", "kind", "type", "name", "tell", "much", "many", "does",
            "can", "could", "would", "about"
        }

        all_evidence_text = " ".join(c.get("text", "") for c in retrieved_chunks).lower()

        for cl in clauses:
            cl_words = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', cl) if w.lower() not in stopwords]
            if not cl_words:
                supported_clauses.append(cl)
                continue
            matched = sum(1 for w in cl_words if w in all_evidence_text)
            coverage = matched / len(cl_words)

            if coverage >= self.min_query_term_coverage:
                supported_clauses.append(cl)
            else:
                unsupported_clauses.append(cl)

        # 3. Decision mapping
        if not supported_clauses and unsupported_clauses:
            # Check if any chunk has high retrieval score or if single clause
            has_strong_score = any(float(c.get("score", 0.0)) >= 0.30 for c in retrieved_chunks)
            if has_strong_score or len(clauses) == 1:
                return EvidenceGateReport(
                    decision=GatingDecision.PROCEED,
                    supported_subclaims=clauses,
                    unsupported_subclaims=[],
                    rationale="Retrieved chunks possess strong semantic relevance scores.",
                    usable_chunk_ids=[c.get("chunk_id", "") for c in retrieved_chunks]
                )
            return EvidenceGateReport(
                decision=GatingDecision.ABSTAIN_INSUFFICIENT,
                supported_subclaims=[],
                unsupported_subclaims=unsupported_clauses,
                rationale="Retrieved chunks lack coverage for key query requirements.",
                usable_chunk_ids=[]
            )
        elif supported_clauses and unsupported_clauses:
            return EvidenceGateReport(
                decision=GatingDecision.PARTIAL_ANSWER,
                supported_subclaims=supported_clauses,
                unsupported_subclaims=unsupported_clauses,
                rationale="Evidence covers some aspects of the query, but leaves other subclaims unsupported.",
                usable_chunk_ids=[c.get("chunk_id", "") for c in retrieved_chunks]
            )

        return EvidenceGateReport(
            decision=GatingDecision.PROCEED,
            supported_subclaims=supported_clauses,
            unsupported_subclaims=[],
            rationale="Evidence sufficiently covers query requirements.",
            usable_chunk_ids=[c.get("chunk_id", "") for c in retrieved_chunks]
        )

    def validate_citations(
        self,
        cited_ids: list[str],
        retrieved_chunks: list[dict[str, Any]]
    ) -> tuple[list[str], list[str]]:
        """
        Deterministically validate cited chunk/memory IDs against available context.
        Returns:
            (valid_ids, rejected_ids)
        """
        valid_pool = set()
        for c in retrieved_chunks:
            if "chunk_id" in c:
                valid_pool.add(str(c["chunk_id"]))
            if "memory_id" in c:
                valid_pool.add(str(c["memory_id"]))
            if "session_id" in c:
                valid_pool.add(str(c["session_id"]))
                valid_pool.add(f"sess_{c['session_id']}")

        valid_citations = []
        rejected_citations = []

        for cid in cited_ids:
            clean = str(cid).strip()
            if clean in valid_pool:
                valid_citations.append(clean)
            else:
                rejected_citations.append(clean)

        return valid_citations, rejected_citations
