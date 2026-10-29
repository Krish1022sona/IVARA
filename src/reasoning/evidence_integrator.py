"""
Inter-Memory Evidence Integration and Attention Engine for IVARA.

Evaluates how candidate memories interact with the user query and with one another,
classifying each memory into explicit evidence roles:
- SUPPORTING: Directly provides facts answering a primary query clause.
- COMPLEMENTARY: Supplies linking entities, dates, or constraints that join with other memories to complete a multi-hop answer.
- REDUNDANT: Repeats already established facts without adding novel information (pruned or de-weighted).
- CONTRADICTORY: Contains claims that directly conflict with another candidate memory.
- SUPERSEDED: Contains historical facts that have been updated by a chronologically newer memory.
- IRRELEVANT: Peripheral or distractor text below relevance thresholds.

Provides inspectable contribution scores, role assignments, inter-memory linkage graphs,
and bounded context selection.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any

from ..graph.schema import TemporalClaim
from .contradiction import ContradictionAnalyzer, ContradictionType


class EvidenceRole(str, Enum):
    SUPPORTING = "supporting"
    COMPLEMENTARY = "complementary"
    REDUNDANT = "redundant"
    CONTRADICTORY = "contradictory"
    SUPERSEDED = "superseded"
    IRRELEVANT = "irrelevant"


@dataclass
class MemoryEvidenceProfile:
    """Detailed evidence role and contribution analysis for a single memory chunk."""
    chunk_id: str
    session_id: str
    role: EvidenceRole
    query_relevance_score: float
    contribution_score: float
    rationale: str
    extracted_entities: list[str] = field(default_factory=list)
    linked_chunk_ids: list[str] = field(default_factory=list)
    redundant_with_chunk_id: str | None = None
    superseded_by_chunk_id: str | None = None
    conflicts_with_chunk_id: str | None = None


class EvidenceIntegrator:
    """
    Evaluates candidate memories collectively, performing inter-memory cross-attention
    to identify complementary multi-memory answers, suppress duplicates, and highlight conflicts.
    """

    def __init__(
        self,
        min_relevance_threshold: float = 0.25,
        redundancy_similarity_threshold: float = 0.85
    ):
        self.min_relevance_threshold = min_relevance_threshold
        self.redundancy_similarity_threshold = redundancy_similarity_threshold
        self.contradiction_analyzer = ContradictionAnalyzer()

    def _extract_tokens(self, text: str) -> set[str]:
        words = re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', text.lower())
        stopwords = {
            "the", "and", "that", "this", "with", "from", "have", "been", "were",
            "what", "when", "where", "which", "user", "assistant", "session"
        }
        return {w for w in words if w not in stopwords}

    def _jaccard_similarity(self, tokens_a: set[str], tokens_b: set[str]) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return intersection / union if union > 0 else 0.0

    def _extract_proper_entities(self, text: str) -> list[str]:
        words = text.split()
        entities = []
        for w in words:
            clean = re.sub(r'[^a-zA-Z0-9]', '', w)
            if len(clean) > 2 and clean[0].isupper() and clean.lower() not in (
                "user", "assistant", "what", "where", "when", "which", "how", "the"
            ):
                entities.append(clean)
        return list(dict.fromkeys(entities))

    def integrate(
        self,
        query: str,
        candidate_chunks: list[dict[str, Any]],
        claims: list[TemporalClaim] | None = None
    ) -> dict[str, Any]:
        """
        Integrate candidate chunks, assigning evidence roles and calculating
        inter-memory contribution scores.
        """
        if not candidate_chunks:
            return {
                "selected_chunks": [],
                "profiles": {},
                "inter_memory_links": [],
                "has_contradictions": False,
                "summary": "No candidate chunks provided."
            }

        q_tokens = self._extract_tokens(query)
        q_entities = set(self._extract_proper_entities(query))
        claims = claims or []

        # Map claim status (superseded, conflict)
        superseded_chunk_ids = set()
        conflict_chunk_map: dict[str, list[str]] = defaultdict(list)
        claim_analysis = self.contradiction_analyzer.analyze_claims(claims)

        for uc in claim_analysis["unresolved_conflicts"]:
            for cid in uc["claims"]:
                matching_chunks = [c.source_chunk_id for c in claims if c.claim_id == cid and c.source_chunk_id]
                for mc in matching_chunks:
                    for other_c in uc["claims"]:
                        if other_c != cid:
                            other_mcs = [c.source_chunk_id for c in claims if c.claim_id == other_c and c.source_chunk_id]
                            conflict_chunk_map[mc].extend(other_mcs)

        for sup_id in claim_analysis["superseded_claim_ids"]:
            matching_chunks = [c.source_chunk_id for c in claims if c.claim_id == sup_id and c.source_chunk_id]
            superseded_chunk_ids.update(matching_chunks)

        # 1. Profile each chunk individually
        profiles: dict[str, MemoryEvidenceProfile] = {}
        chunk_token_maps: dict[str, set[str]] = {}
        chunk_entity_maps: dict[str, list[str]] = {}

        for c in candidate_chunks:
            cid = c.get("chunk_id", "")
            sid = c.get("session_id", "")
            raw_score = float(c.get("score", 0.0))
            text = c.get("text", "")

            toks = self._extract_tokens(text)
            ents = self._extract_proper_entities(text)
            chunk_token_maps[cid] = toks
            chunk_entity_maps[cid] = ents

            # Baseline query overlap
            q_overlap = len(toks & q_tokens) / max(1, len(q_tokens))
            is_relevant = raw_score >= self.min_relevance_threshold or q_overlap >= 0.20

            if not is_relevant:
                role = EvidenceRole.IRRELEVANT
                rationale = f"Relevance score ({raw_score:.3f}) below threshold {self.min_relevance_threshold}."
            elif cid in superseded_chunk_ids:
                role = EvidenceRole.SUPERSEDED
                rationale = "Contains historical facts superseded by a chronologically newer memory."
            elif cid in conflict_chunk_map:
                role = EvidenceRole.CONTRADICTORY
                rationale = f"Contains facts in unresolved conflict with {conflict_chunk_map[cid]}."
            else:
                role = EvidenceRole.SUPPORTING
                rationale = "Directly matches query intent and entities."

            profiles[cid] = MemoryEvidenceProfile(
                chunk_id=cid,
                session_id=sid,
                role=role,
                query_relevance_score=raw_score,
                contribution_score=raw_score,
                rationale=rationale,
                extracted_entities=ents,
                conflicts_with_chunk_id=conflict_chunk_map[cid][0] if cid in conflict_chunk_map and conflict_chunk_map[cid] else None
            )

        # 2. Inter-Memory Cross-Attention (Pairwise inspection)
        inter_memory_links = []
        n = len(candidate_chunks)
        cids = [c.get("chunk_id", "") for c in candidate_chunks]

        for i in range(n):
            cid_a = cids[i]
            prof_a = profiles[cid_a]
            if prof_a.role in (EvidenceRole.IRRELEVANT, EvidenceRole.SUPERSEDED):
                continue

            for j in range(i + 1, n):
                cid_b = cids[j]
                prof_b = profiles[cid_b]
                if prof_b.role in (EvidenceRole.IRRELEVANT, EvidenceRole.SUPERSEDED):
                    continue

                toks_a = chunk_token_maps[cid_a]
                toks_b = chunk_token_maps[cid_b]
                sim = self._jaccard_similarity(toks_a, toks_b)

                # Check Redundancy: High lexical similarity and same session/topic
                if sim >= self.redundancy_similarity_threshold:
                    if prof_b.role == EvidenceRole.SUPPORTING:
                        prof_b.role = EvidenceRole.REDUNDANT
                        prof_b.redundant_with_chunk_id = cid_a
                        prof_b.contribution_score = round(prof_b.contribution_score * 0.3, 4)
                        prof_b.rationale = f"Redundant copy of evidence already established by {cid_a} (similarity={sim:.2f})."
                    continue

                # Check Complementary Relationship:
                # Different sessions/chunks sharing an entity or query clause
                shared_ents = set(chunk_entity_maps[cid_a]) & set(chunk_entity_maps[cid_b])
                query_shared_ents = shared_ents & q_entities

                if (shared_ents or query_shared_ents) and prof_a.session_id != prof_b.session_id:
                    # Chunks from disparate sessions complement each other via shared entity!
                    prof_a.linked_chunk_ids.append(cid_b)
                    prof_b.linked_chunk_ids.append(cid_a)

                    if prof_b.role == EvidenceRole.SUPPORTING and len(shared_ents) > 0:
                        prof_b.role = EvidenceRole.COMPLEMENTARY
                        prof_b.contribution_score = round(prof_b.contribution_score * 1.15, 4)  # Boost joint value
                        prof_b.rationale = f"Complements {cid_a} via shared entity {list(shared_ents)} across sessions."

                    inter_memory_links.append({
                        "source": cid_a,
                        "target": cid_b,
                        "shared_entities": list(shared_ents),
                        "type": "complementary_cross_session"
                    })

        # 3. Assemble and rank selected evidence
        # Priority order: SUPPORTING > COMPLEMENTARY > CONTRADICTORY > REDUNDANT > SUPERSEDED > IRRELEVANT
        role_priority = {
            EvidenceRole.SUPPORTING: 1,
            EvidenceRole.COMPLEMENTARY: 2,
            EvidenceRole.CONTRADICTORY: 3,
            EvidenceRole.REDUNDANT: 4,
            EvidenceRole.SUPERSEDED: 5,
            EvidenceRole.IRRELEVANT: 6
        }

        selected_chunks = []
        for c in candidate_chunks:
            cid = c.get("chunk_id", "")
            prof = profiles[cid]
            if prof.role in (EvidenceRole.SUPPORTING, EvidenceRole.COMPLEMENTARY, EvidenceRole.CONTRADICTORY):
                c_augmented = dict(c)
                c_augmented["evidence_role"] = prof.role.value
                c_augmented["contribution_score"] = prof.contribution_score
                c_augmented["integration_rationale"] = prof.rationale
                c_augmented["linked_chunks"] = prof.linked_chunk_ids
                selected_chunks.append(c_augmented)

        selected_chunks.sort(key=lambda x: (
            role_priority[EvidenceRole(x["evidence_role"])],
            -x["contribution_score"]
        ))

        return {
            "selected_chunks": selected_chunks,
            "profiles": {cid: p.__dict__ for cid, p in profiles.items()},
            "inter_memory_links": inter_memory_links,
            "has_contradictions": any(p.role == EvidenceRole.CONTRADICTORY for p in profiles.values()),
            "redundancy_pruned_count": sum(1 for p in profiles.values() if p.role == EvidenceRole.REDUNDANT),
            "summary": f"Integrated {len(candidate_chunks)} candidates: {len(selected_chunks)} retained, {sum(1 for p in profiles.values() if p.role == EvidenceRole.REDUNDANT)} redundant pruned."
        }
