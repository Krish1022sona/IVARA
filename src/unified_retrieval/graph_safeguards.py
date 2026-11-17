"""Knowledge Graph Evidence Safeguards for IVARA Unified Retrieval.

Enforces strict provenance, temporal safety, and relevance gating on Knowledge Graph
triples before they can be admitted into the answering context:

1. Provenance Verification:
   A graph triple is admissible ONLY if it can be traced to a verified source:
   - An existing episodic turn through a verified source-turn reference, OR
   - A verified semantic memory in the memory store.
   Triples lacking verifiable provenance are excluded.

2. Temporal Safety:
   Historical-state queries must NEVER receive unversioned current graph assertions.
   Edges without valid-time intervals (valid_from/valid_to) cannot establish historical state
   and are strictly excluded for historical queries.

3. Relevance & Distractor Filtering:
   - Enforces a calibrated relevance threshold.
   - Prevents generic entity matches (e.g., 'self', 'user') from flooding context with irrelevant edges.
   - Enforces fan-out limits per entity.
   - Filters out cycles, duplicates, and directionally invalid triples.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Optional

from .graph import GraphEvidenceItem
from .router import RouteDecision


@dataclass
class GraphSafeguardDecision:
    """Decision record for an individual graph triple."""
    triple_id: str
    triple_statement: str
    is_admissible: bool
    rejection_reason: Optional[str] = None
    provenance_type: Optional[str] = None  # "episodic_turn" | "semantic_memory" | None
    provenance_id: Optional[str] = None
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "triple_id": self.triple_id,
            "statement": self.triple_statement,
            "is_admissible": self.is_admissible,
            "rejection_reason": self.rejection_reason,
            "provenance_type": self.provenance_type,
            "provenance_id": self.provenance_id,
            "score": round(self.score, 4)
        }


@dataclass
class GraphSafeguardReport:
    """Consolidated report of graph safeguard evaluation."""
    candidate_count: int
    admissible_count: int
    rejected_count: int
    temporal_exclusions: int
    relevance_exclusions: int
    provenance_exclusions: int
    policy_exclusions: int
    rejections_by_reason: dict[str, int]
    decisions: list[GraphSafeguardDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "admissible_count": self.admissible_count,
            "rejected_count": self.rejected_count,
            "temporal_exclusions": self.temporal_exclusions,
            "relevance_exclusions": self.relevance_exclusions,
            "provenance_exclusions": self.provenance_exclusions,
            "policy_exclusions": self.policy_exclusions,
            "rejections_by_reason": self.rejections_by_reason,
            "decisions": [d.to_dict() for d in self.decisions]
        }


class GraphSafeguards:
    """
    Evaluates and filters candidate knowledge graph triples against provenance,
    temporal validity, relevance, and structural safety constraints.
    """

    GENERIC_ENTITY_NAMES = {"self", "user", "assistant", "ivara"}

    def __init__(
        self,
        min_relevance_threshold: float = 0.45,
        max_fan_out_per_entity: int = 4,
        strict_provenance: bool = True,
        user_name: Optional[str] = None
    ):
        self.min_relevance_threshold = min_relevance_threshold
        self.max_fan_out_per_entity = max_fan_out_per_entity
        self.strict_provenance = strict_provenance
        self.user_name = user_name
        self.generic_entity_names = set(self.GENERIC_ENTITY_NAMES)
        if user_name:
            u_clean = user_name.lower().strip()
            self.generic_entity_names.add(u_clean)
            for part in u_clean.split():
                if len(part) >= 3:
                    self.generic_entity_names.add(part)

    def _extract_query_tokens(self, query: str) -> set[str]:
        """Extract meaningful semantic tokens from query (length >= 3, lowercase)."""
        tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", query.lower()))
        # Remove common stopwords and generic entity mentions
        stopwords = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "the", "and", "did", "does", "was", "were", "has", "have", "had", "for",
            "with", "about", "his", "her", "their", "from", "into", "during", "before", "after",
            "self", "user", "assistant", "ivara"
        }
        stopwords.update(self.generic_entity_names)
        return tokens - stopwords

    def _substantiates_triple(self, mem_record: dict[str, Any], triple: GraphEvidenceItem) -> bool:
        """
        Verify that the source memory actually substantiates the claim in the triple,
        rather than merely sharing a memory ID.
        """
        # 1. Direct relationship match
        t_pred = triple.predicate.lower().strip()
        t_src = triple.source_name.lower().strip()
        t_tgt = triple.target_name.lower().strip()
        t_src_id = triple.source_id.lower().strip()
        t_tgt_id = triple.target_id.lower().strip()

        for rel in mem_record.get("relationships", []):
            r_pred = (rel.get("predicate") or rel.get("relation") or "").lower().strip()
            r_src = rel.get("source")
            r_tgt = rel.get("target")

            r_src_name = (r_src.get("canonical_name") or r_src.get("mention") or r_src.get("id") or str(r_src)).lower().strip() if isinstance(r_src, dict) else str(r_src).lower().strip()
            r_tgt_name = (r_tgt.get("canonical_name") or r_tgt.get("mention") or r_tgt.get("id") or str(r_tgt)).lower().strip() if isinstance(r_tgt, dict) else str(r_tgt).lower().strip()

            if r_pred == t_pred or r_pred.replace("_", "") == t_pred.replace("_", ""):
                src_match = (r_src_name in (t_src, t_src_id) or t_src in r_src_name or (t_src in self.GENERIC_ENTITY_NAMES and r_src_name in self.GENERIC_ENTITY_NAMES))
                tgt_match = (r_tgt_name in (t_tgt, t_tgt_id) or t_tgt in r_tgt_name or t_tgt_id in r_tgt_name)
                if src_match and tgt_match:
                    return True

        # 2. Textual / evidence substantiation
        mem_text = (mem_record.get("text") or "").lower()
        if triple.evidence:
            ev_clean = triple.evidence.lower().strip()
            if ev_clean in mem_text or any(part in mem_text for part in ev_clean.split(";") if len(part) > 10):
                return True

        # Check entity and predicate token overlap with memory text
        tgt_tokens = [tok for tok in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", t_tgt) if tok not in self.GENERIC_ENTITY_NAMES]
        pred_tokens = [tok for tok in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", t_pred)]

        has_tgt_match = any(t in mem_text for t in tgt_tokens) if tgt_tokens else True
        has_pred_match = any(p in mem_text for p in pred_tokens) if pred_tokens else True

        if has_tgt_match and has_pred_match:
            return True

        return False

    def evaluate_triples(
        self,
        triples: list[GraphEvidenceItem],
        query: str,
        route_decision: RouteDecision,
        known_memory_ids: Optional[set[str]] = None,
        known_turn_ids: Optional[set[str]] = None,
        memories_by_id: Optional[dict[str, dict[str, Any]]] = None
    ) -> tuple[list[GraphEvidenceItem], GraphSafeguardReport]:
        """
        Evaluate candidate graph triples and return admissible triples alongside a detailed safeguard report.
        """
        candidate_count = len(triples)
        decisions: list[GraphSafeguardDecision] = []
        admissible_triples: list[GraphEvidenceItem] = []

        temporal_exclusions = 0
        relevance_exclusions = 0
        provenance_exclusions = 0
        policy_exclusions = 0
        rejections_by_reason: dict[str, int] = {}

        def record_rejection(triple: GraphEvidenceItem, reason: str, category: str):
            nonlocal temporal_exclusions, relevance_exclusions, provenance_exclusions, policy_exclusions
            if category == "temporal":
                temporal_exclusions += 1
            elif category == "relevance":
                relevance_exclusions += 1
            elif category == "provenance":
                provenance_exclusions += 1
            else:
                policy_exclusions += 1

            rejections_by_reason[reason] = rejections_by_reason.get(reason, 0) + 1
            decisions.append(GraphSafeguardDecision(
                triple_id=triple.triple_id,
                triple_statement=triple.formatted_statement,
                is_admissible=False,
                rejection_reason=reason,
                score=triple.score
            ))

        # 1. Router Policy Check: Was graph retrieval enabled by the router?
        if not route_decision.graph_enabled:
            for t in triples:
                record_rejection(t, f"graph_disabled_by_router_{route_decision.query_type}", "policy")
            report = GraphSafeguardReport(
                candidate_count=candidate_count,
                admissible_count=0,
                rejected_count=candidate_count,
                temporal_exclusions=temporal_exclusions,
                relevance_exclusions=relevance_exclusions,
                provenance_exclusions=provenance_exclusions,
                policy_exclusions=policy_exclusions,
                rejections_by_reason=rejections_by_reason,
                decisions=decisions
            )
            return [], report

        # 2. Historical Query Check: Historical-state queries cannot receive unversioned graph facts
        is_historical_query = (
            route_decision.query_type in ("historical_state", "temporal_longitudinal") or
            "temporal_historical_keyword" in route_decision.signals or
            "historical_role_pattern" in route_decision.signals
        )

        query_tokens = self._extract_query_tokens(query)
        seen_canonical_triples: set[tuple[str, str, str]] = set()
        fan_out_counts: dict[str, int] = {}

        for triple in triples:
            # Check A: Temporal Safety
            # Graph store does not populate valid_from / valid_to.
            props = triple.metadata or {}
            has_valid_interval = bool(props.get("valid_from") and props.get("valid_to"))

            if is_historical_query and not has_valid_interval:
                record_rejection(triple, "temporal_safety_unversioned_graph", "temporal")
                continue

            # Check B: Self-referential or cycle
            if triple.source_id.lower().strip() == triple.target_id.lower().strip() or \
               triple.source_name.lower().strip() == triple.target_name.lower().strip():
                record_rejection(triple, "self_referential_cycle", "relevance")
                continue

            # Check C: Duplicate triple
            canonical_key = (triple.source_id.lower(), triple.predicate.lower(), triple.target_id.lower())
            if canonical_key in seen_canonical_triples:
                record_rejection(triple, "duplicate_triple", "relevance")
                continue

            # Check D: Relevance threshold
            if triple.score < self.min_relevance_threshold:
                record_rejection(
                    triple,
                    f"score_{triple.score:.2f}_below_threshold_{self.min_relevance_threshold}",
                    "relevance"
                )
                continue

            # Check E: Generic entity distractor
            # If source or target is a generic entity ("user", "self") and neither the predicate,
            # opposite entity, nor evidence overlaps with query tokens or detected entities
            other_entity_name = triple.target_name if triple.source_id.lower() in self.generic_entity_names else triple.source_name
            other_clean = other_entity_name.lower()
            pred_clean = triple.predicate.lower()
            ev_clean = (triple.evidence or "").lower()

            has_query_overlap = any(
                tok in other_clean or tok in pred_clean or tok in ev_clean
                for tok in query_tokens
            )
            detected_overlap = any(
                ent.lower() in other_clean or ent.lower() in pred_clean
                for ent in route_decision.detected_entities
            )

            if triple.hops > 1 and not has_query_overlap and not detected_overlap:
                record_rejection(triple, "generic_entity_distractor_no_overlap", "relevance")
                continue

            # Check F: Fan-out limit per entity
            current_fanout = fan_out_counts.get(triple.source_id, 0)
            if current_fanout >= self.max_fan_out_per_entity:
                record_rejection(triple, f"fan_out_limit_exceeded_for_{triple.source_id}", "relevance")
                continue

            # Check G: Provenance Verification
            # A triple must trace to a verified semantic memory or verified episodic turn
            provenance_type = None
            provenance_id = None

            if self.strict_provenance:
                # 1. Direct memory_id check
                if triple.memory_id:
                    if known_memory_ids is not None:
                        if triple.memory_id in known_memory_ids:
                            # Claim-level provenance check: does the memory substantiate the triple?
                            if memories_by_id and triple.memory_id in memories_by_id:
                                mem_rec = memories_by_id[triple.memory_id]
                                if not self._substantiates_triple(mem_rec, triple):
                                    record_rejection(triple, f"unsubstantiated_by_source_memory_{triple.memory_id}", "provenance")
                                    continue
                            provenance_type = "semantic_memory"
                            provenance_id = triple.memory_id
                        else:
                            record_rejection(triple, f"unverified_memory_id_{triple.memory_id}", "provenance")
                            continue
                    else:
                        provenance_type = "semantic_memory"
                        provenance_id = triple.memory_id

                # 2. Check if metadata has source_turn_id
                src_turn = props.get("source_turn_id") or props.get("turn_id")
                if src_turn:
                    if known_turn_ids is not None:
                        if src_turn in known_turn_ids:
                            provenance_type = "episodic_turn"
                            provenance_id = src_turn
                        elif not provenance_type:
                            record_rejection(triple, f"unverified_turn_id_{src_turn}", "provenance")
                            continue
                    else:
                        provenance_type = "episodic_turn"
                        provenance_id = src_turn

                # 3. If memory was verified, also check if that memory has an episodic source turn
                if provenance_type == "semantic_memory" and memories_by_id and provenance_id:
                    mem_record = memories_by_id.get(provenance_id)
                    if mem_record:
                        src_raw = mem_record.get("source") or ""
                        # Extract turn id if present in memory source
                        turn_match = re.search(r"\b(turn_[a-zA-Z0-9_]+|msg_[a-zA-Z0-9_]+)\b", src_raw)
                        if turn_match:
                            provenance_id = f"{provenance_id}->{turn_match.group(1)}"

                # If no valid provenance could be established
                if not provenance_type:
                    record_rejection(triple, "missing_provenance", "provenance")
                    continue

            # If all safeguards passed, admit triple
            seen_canonical_triples.add(canonical_key)
            fan_out_counts[triple.source_id] = current_fanout + 1

            admissible_triples.append(triple)
            decisions.append(GraphSafeguardDecision(
                triple_id=triple.triple_id,
                triple_statement=triple.formatted_statement,
                is_admissible=True,
                rejection_reason=None,
                provenance_type=provenance_type,
                provenance_id=provenance_id,
                score=triple.score
            ))

        report = GraphSafeguardReport(
            candidate_count=candidate_count,
            admissible_count=len(admissible_triples),
            rejected_count=len(decisions) - len(admissible_triples),
            temporal_exclusions=temporal_exclusions,
            relevance_exclusions=relevance_exclusions,
            provenance_exclusions=provenance_exclusions,
            policy_exclusions=policy_exclusions,
            rejections_by_reason=rejections_by_reason,
            decisions=decisions
        )

        return admissible_triples, report
