"""Adaptive Evidence Selection Engine for IVARA Unified Retrieval.

Implements the core architectural principle:
"Episodic turns are the primary evidence backbone. Semantic memories annotate episodic
evidence rather than competing with it for independent context slots."

Key Responsibilities:
1. Category-Adaptive Context Budgets:
   Allocates targeted episodic turn budgets and session policies according to query category.
2. Contiguous Context Window Expansion & Deduplication:
   Preserves contiguous conversation flow around key turns, merging overlapping windows.
3. Chronological & Speaker Provenance Preservation:
   Maintains true chronological order within sessions, explicit speaker attribution, and session boundaries.
4. Anchored Semantic Annotation:
   Attaches semantic memories to their source episodic turns via verified source_turn_id linkage.
   Preserves unanchored semantic memories when no episodic turn is present, explicitly marking them.
5. Multi-Session Synthesis Protection:
   Supports multi-session queries (up to 4 sessions) without collapsing distinct sessions into one narrative.
6. Graph Safeguards Integration:
   Enforces provenance, temporal safety, and relevance on graph triples before inclusion.
7. Diagnostics & Token Calibration:
   Emits detailed selection diagnostics, token estimates, and rejection reasons.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Optional

from .episodic import EpisodicEvidenceItem
from .semantic import SemanticEvidenceItem
from .graph import GraphEvidenceItem
from .graph_safeguards import GraphSafeguards, GraphSafeguardReport
from .router import RouteDecision
from ..conversations.store import ConversationStore, EpisodicTurn


@dataclass
class SelectedEvidenceItem:
    """Consolidated evidence item produced by AdaptiveEvidenceSelector."""
    evidence_type: str  # "episodic" | "semantic" | "graph"
    evidence_id: str    # turn_id, memory_id, or triple_id
    text: str
    session_id: Optional[str]
    speaker: Optional[str]
    timestamp: Optional[str]
    score: float
    selection_reason: str
    is_anchored: bool = False
    source_turn_id: Optional[str] = None
    anchored_semantic_memories: list[dict[str, Any]] = field(default_factory=list)
    anchored_graph_triples: list[dict[str, Any]] = field(default_factory=list)
    context_turns: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_type": self.evidence_type,
            "evidence_id": self.evidence_id,
            "text": self.text,
            "session_id": self.session_id,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "score": round(self.score, 4),
            "selection_reason": self.selection_reason,
            "is_anchored": self.is_anchored,
            "source_turn_id": self.source_turn_id,
            "anchored_semantic_count": len(self.anchored_semantic_memories),
            "anchored_graph_count": len(self.anchored_graph_triples),
            "metadata": self.metadata
        }


@dataclass
class SelectionResult:
    """Complete output of AdaptiveEvidenceSelector."""
    selected_items: list[SelectedEvidenceItem]
    structured_context_text: str
    citations: list[str]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_items_count": len(self.selected_items),
            "citations": self.citations,
            "diagnostics": self.diagnostics
        }


class AdaptiveEvidenceSelector:
    """
    Coordinates adaptive context allocation, contiguous window expansion,
    semantic annotation, and graph safeguards across all memory channels.
    """

    # Category budget configuration
    BUDGET_CONFIG = {
        "single_fact": {"target_turns": 3, "max_sessions": 1, "window_size": 1, "max_unanchored_sem": 2},
        "historical_state": {"target_turns": 5, "max_sessions": 2, "window_size": 1, "max_unanchored_sem": 2},
        "temporal_longitudinal": {"target_turns": 8, "max_sessions": 3, "window_size": 1, "max_unanchored_sem": 2},
        "cross_session_synth": {"target_turns": 10, "max_sessions": 4, "window_size": 1, "max_unanchored_sem": 3},
        "entity_relationships": {"target_turns": 5, "max_sessions": 2, "window_size": 1, "max_unanchored_sem": 2},
        "multi_hop_reasoning": {"target_turns": 6, "max_sessions": 3, "window_size": 1, "max_unanchored_sem": 2},
        "negative_information": {"target_turns": 5, "max_sessions": 3, "window_size": 1, "max_unanchored_sem": 2},
        "unanswerable": {"target_turns": 2, "max_sessions": 1, "window_size": 0, "max_unanchored_sem": 0},
        "safe_fallback": {"target_turns": 5, "max_sessions": 2, "window_size": 1, "max_unanchored_sem": 2}
    }

    def __init__(
        self,
        graph_safeguards: Optional[GraphSafeguards] = None,
        max_context_tokens: int = 2500
    ):
        self.graph_safeguards = graph_safeguards or GraphSafeguards()
        self.max_context_tokens = max_context_tokens

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Conservative token estimator (~3.8 chars per token for BPE / word calibrated)."""
        if not text:
            return 0
        words = text.split()
        char_est = int(len(text) / 3.8)
        word_est = int(len(words) * 1.33)
        return max(char_est, word_est, 1)

    def _normalize_turn_id(self, raw_id: Optional[str]) -> Optional[str]:
        """Normalize message/turn ID for robust cross-channel matching."""
        if not raw_id or not isinstance(raw_id, str):
            return None
        cleaned = re.sub(r"^.*?msg_msg_", "msg_", raw_id)
        cleaned = re.sub(r"^.*?turn_turn_", "turn_", cleaned)
        m = re.search(r"\b(turn_[a-zA-Z0-9_]+|msg_[a-zA-Z0-9_]+)\b", cleaned)
        if m:
            val = m.group(1)
            val = re.sub(r"^(msg_)+", "msg_", val)
            val = re.sub(r"^(turn_)+", "turn_", val)
            return val
        return raw_id.strip()

    def _render_context_text(
        self,
        clean_q: str,
        route_decision: RouteDecision,
        episodic_items: list[SelectedEvidenceItem],
        unanchored_sem_items: list[SelectedEvidenceItem],
        graph_items: list[SelectedEvidenceItem]
    ) -> tuple[str, list[str]]:
        citations: list[str] = []
        has_items = bool(episodic_items or unanchored_sem_items or graph_items)
        context_lines: list[str] = [
            f"Query: {clean_q}",
            f"Context Mode: {route_decision.context_mode} (Category: {route_decision.query_type})",
            f"Evidence Status: {'Sufficient evidence retrieved.' if has_items else 'Insufficient evidence in memory store.'}\n"
        ]

        # Section 1: Episodic Turns
        context_lines.append("=== PRIMARY EPISODIC CONVERSATION EVIDENCE ===")
        if not episodic_items:
            context_lines.append("(No episodic conversation turns selected)")
        else:
            curr_sess = None
            for item in episodic_items:
                s_id = item.session_id or "default_session"
                if s_id != curr_sess:
                    curr_sess = s_id
                    context_lines.append(f"\n--- SESSION: {curr_sess} ---")

                citations.append(item.evidence_id)
                ts_str = f" [{item.timestamp}]" if item.timestamp else ""
                speaker_str = item.speaker or "Speaker"
                context_lines.append(f"[{item.evidence_id}]{ts_str} {speaker_str}: {item.text}")

                for sem_dict in item.anchored_semantic_memories:
                    sem_id = sem_dict.get("memory_id")
                    if sem_id:
                        citations.append(sem_id)
                    context_lines.append(f"    * [Semantic Annotation {sem_id}]: \"{sem_dict.get('text')}\"")
                    for rel in sem_dict.get("relationships", []):
                        pred = rel.get("predicate") or rel.get("relation")
                        src = rel.get("source")
                        tgt = rel.get("target")
                        s_name = src.get("canonical_name", str(src)) if isinstance(src, dict) else str(src)
                        t_name = tgt.get("canonical_name", str(tgt)) if isinstance(tgt, dict) else str(tgt)
                        context_lines.append(f"      - Extracted Relationship: {s_name} {pred} {t_name}")

                for g_dict in item.anchored_graph_triples:
                    context_lines.append(f"    * [Graph Annotation]: {g_dict.get('formatted_statement')}")

        # Section 2: Unanchored Semantic Memories
        if unanchored_sem_items:
            context_lines.append("\n=== UNANCHORED SEMANTIC MEMORIES ===")
            for item in unanchored_sem_items:
                citations.append(item.evidence_id)
                ts_str = f" [{item.timestamp}]" if item.timestamp else ""
                context_lines.append(f"[{item.evidence_id}]{ts_str} Fact: \"{item.text}\" (unanchored)")

        # Section 3: Verified Graph Triples
        if graph_items:
            context_lines.append("\n=== VERIFIED KNOWLEDGE GRAPH RELATIONSHIPS ===")
            for item in graph_items:
                context_lines.append(f"  * {item.text}")

        return "\n".join(context_lines), list(dict.fromkeys(citations))

    def select(
        self,
        query: str,
        route_decision: RouteDecision,
        episodic_hits: list[EpisodicEvidenceItem],
        semantic_hits: list[SemanticEvidenceItem],
        graph_hits: list[GraphEvidenceItem],
        conversation_store: Optional[ConversationStore] = None,
        all_turns: Optional[list[EpisodicTurn]] = None,
        all_memories: Optional[list[dict[str, Any]]] = None
    ) -> SelectionResult:
        """
        Executes adaptive evidence selection, contiguous context window expansion,
        anchored semantic annotation, and graph safeguards.
        """
        clean_q = query.strip()
        q_type = route_decision.query_type
        budget = self.BUDGET_CONFIG.get(q_type, self.BUDGET_CONFIG["safe_fallback"])

        target_turns = budget["target_turns"]
        max_sessions = budget["max_sessions"]
        window_size = budget["window_size"]
        max_unanchored_sem = budget["max_unanchored_sem"]

        rejected_items: list[dict[str, Any]] = []

        # Build turn lookup for fast context expansion and provenance verification
        turns_by_id: dict[str, EpisodicTurn] = {}
        if all_turns:
            for t in all_turns:
                turns_by_id[t.turn_id] = t
                norm = self._normalize_turn_id(t.turn_id)
                if norm:
                    turns_by_id[norm] = t

        # Build memory lookup
        mems_by_id: dict[str, dict[str, Any]] = {}
        if all_memories:
            for m in all_memories:
                m_id = m.get("id")
                if m_id:
                    mems_by_id[m_id] = m

        known_turn_ids = set(turns_by_id.keys())
        known_memory_ids = set(mems_by_id.keys())

        # -------------------------------------------------------------
        # STEP 1: Episodic Selection & Session Budgeting
        # -------------------------------------------------------------
        # Group episodic hits by session (or conversation if session is absent)
        session_groups: dict[str, list[EpisodicEvidenceItem]] = {}
        for hit in episodic_hits:
            sess_key = hit.session_id or hit.conversation_id or "default_session"
            if sess_key not in session_groups:
                session_groups[sess_key] = []
            session_groups[sess_key].append(hit)

        # Rank sessions by highest-scoring hit within each session
        ranked_sessions = sorted(
            session_groups.keys(),
            key=lambda s: max((h.score for h in session_groups[s]), default=0.0),
            reverse=True
        )

        selected_sessions = ranked_sessions[:max_sessions]
        for s in ranked_sessions[max_sessions:]:
            for hit in session_groups[s]:
                rejected_items.append({
                    "channel": "episodic",
                    "id": hit.turn_id,
                    "session_id": s,
                    "reason": f"session_cap_exceeded_{max_sessions}"
                })

        # Collect candidate turns from accepted sessions
        eligible_hits: list[EpisodicEvidenceItem] = []
        for s in selected_sessions:
            eligible_hits.extend(session_groups[s])

        # Sort eligible hits by score descending
        eligible_hits.sort(key=lambda h: h.score, reverse=True)

        # -------------------------------------------------------------
        # STEP 2: Contiguous Window Expansion & Deduplication
        # -------------------------------------------------------------
        # For each session, merge contiguous turns without duplicates
        # Map: session_id -> dict of turn_id -> EpisodicTurn (or turn dict)
        session_turns_map: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in selected_sessions}
        deduplicated_turn_count = 0
        total_episodic_collected = 0

        for hit in eligible_hits:
            if total_episodic_collected >= target_turns:
                rejected_items.append({
                    "channel": "episodic",
                    "id": hit.turn_id,
                    "session_id": hit.session_id,
                    "reason": "target_turn_budget_reached"
                })
                continue

            sess_key = hit.session_id or hit.conversation_id or "default_session"
            curr_sess_map = session_turns_map[sess_key]

            # 1. Add the target hit itself
            if hit.turn_id not in curr_sess_map:
                curr_sess_map[hit.turn_id] = {
                    "turn_id": hit.turn_id,
                    "session_id": hit.session_id,
                    "conversation_id": hit.conversation_id,
                    "speaker": hit.speaker,
                    "speaker_role": hit.speaker_role,
                    "content": hit.content,
                    "timestamp": hit.timestamp,
                    "sequence_num": hit.sequence_num,
                    "score": hit.score,
                    "is_direct_hit": True,
                    "formatted_text": hit.formatted_text
                }
                total_episodic_collected += 1
            else:
                deduplicated_turn_count += 1
                curr_sess_map[hit.turn_id]["is_direct_hit"] = True

            # 2. Expand contiguous context window if window_size > 0
            if window_size > 0:
                # First check hit.context_window from retriever
                if hit.context_window:
                    for cw in hit.context_window:
                        c_id = cw.get("turn_id")
                        if not c_id:
                            continue
                        if c_id not in curr_sess_map:
                            if total_episodic_collected < target_turns:
                                curr_sess_map[c_id] = {
                                    "turn_id": c_id,
                                    "session_id": hit.session_id,
                                    "conversation_id": hit.conversation_id,
                                    "speaker": cw.get("speaker", "user"),
                                    "speaker_role": cw.get("speaker_role", "user"),
                                    "content": cw.get("content", ""),
                                    "timestamp": cw.get("timestamp", hit.timestamp),
                                    "sequence_num": cw.get("sequence_num", hit.sequence_num),
                                    "score": hit.score * 0.9,
                                    "is_direct_hit": False,
                                    "formatted_text": cw.get("formatted", "")
                                }
                                total_episodic_collected += 1
                            else:
                                rejected_items.append({
                                    "channel": "episodic",
                                    "id": c_id,
                                    "session_id": hit.session_id,
                                    "reason": "target_turn_budget_reached_in_window"
                                })
                        else:
                            deduplicated_turn_count += 1
                elif conversation_store is not None:
                    try:
                        w_turns = conversation_store.get_context_window(hit.turn_id, window_size=window_size)
                        for wt in w_turns:
                            if wt.turn_id not in curr_sess_map:
                                if total_episodic_collected < target_turns:
                                    curr_sess_map[wt.turn_id] = {
                                        "turn_id": wt.turn_id,
                                        "session_id": wt.session_id,
                                        "conversation_id": wt.conversation_id,
                                        "speaker": wt.speaker,
                                        "speaker_role": wt.speaker_role,
                                        "content": wt.content,
                                        "timestamp": wt.timestamp,
                                        "sequence_num": wt.sequence_num,
                                        "score": hit.score * 0.9,
                                        "is_direct_hit": (wt.turn_id == hit.turn_id),
                                        "formatted_text": f"[{wt.timestamp}] Speaker: {wt.speaker} ({wt.speaker_role}): {wt.content}"
                                    }
                                    total_episodic_collected += 1
                                else:
                                    rejected_items.append({
                                        "channel": "episodic",
                                        "id": wt.turn_id,
                                        "session_id": wt.session_id,
                                        "reason": "target_turn_budget_reached_in_window"
                                    })
                            else:
                                deduplicated_turn_count += 1
                    except Exception:
                        pass

        # -------------------------------------------------------------
        # STEP 3: Order Turns Chronologically & Construct Episodic Items
        # -------------------------------------------------------------
        selected_episodic_items: list[SelectedEvidenceItem] = []
        turn_id_to_selected_item: dict[str, SelectedEvidenceItem] = {}

        for sess_id in selected_sessions:
            turns_in_sess = list(session_turns_map[sess_id].values())
            # Sort chronologically by sequence_num, then timestamp
            turns_in_sess.sort(key=lambda t: (t.get("sequence_num", 0), t.get("timestamp", "")))

            for t_dict in turns_in_sess:
                item = SelectedEvidenceItem(
                    evidence_type="episodic",
                    evidence_id=t_dict["turn_id"],
                    text=t_dict["content"],
                    session_id=t_dict.get("session_id"),
                    speaker=t_dict.get("speaker"),
                    timestamp=t_dict.get("timestamp"),
                    score=t_dict.get("score", 0.0),
                    selection_reason="Direct episodic query hit" if t_dict.get("is_direct_hit") else "Contiguous context turn",
                    is_anchored=True,
                    source_turn_id=t_dict["turn_id"],
                    metadata={
                        "conversation_id": t_dict.get("conversation_id"),
                        "speaker_role": t_dict.get("speaker_role"),
                        "sequence_num": t_dict.get("sequence_num"),
                        "is_direct_hit": t_dict.get("is_direct_hit")
                    }
                )
                selected_episodic_items.append(item)
                turn_id_to_selected_item[t_dict["turn_id"]] = item
                norm_id = self._normalize_turn_id(t_dict["turn_id"])
                if norm_id:
                    turn_id_to_selected_item[norm_id] = item

        # -------------------------------------------------------------
        # STEP 4: Semantic Annotation (Primary Backbone Principle)
        # -------------------------------------------------------------
        anchored_semantic_count = 0
        unanchored_semantic_items: list[SelectedEvidenceItem] = []

        for sem in semantic_hits:
            src_raw = sem.source_turn_id or sem.raw_source
            norm_src = self._normalize_turn_id(src_raw)

            # Check if this semantic memory can be anchored to an episodic turn in our context
            anchored_target = None
            if norm_src and norm_src in turn_id_to_selected_item:
                anchored_target = turn_id_to_selected_item[norm_src]
            elif src_raw and src_raw in turn_id_to_selected_item:
                anchored_target = turn_id_to_selected_item[src_raw]
            else:
                for t_id, t_item in turn_id_to_selected_item.items():
                    if len(t_id) >= 4 and ((src_raw and t_id in src_raw) or (norm_src and (t_id in norm_src or norm_src in t_id))):
                        anchored_target = t_item
                        break

            if anchored_target is not None:
                # Anchor semantic fact to the episodic turn!
                anchored_target.anchored_semantic_memories.append(sem.to_dict())
                anchored_semantic_count += 1
            else:
                # Unanchored semantic memory
                if len(unanchored_semantic_items) < max_unanchored_sem and sem.score >= 0.30:
                    unanchored_item = SelectedEvidenceItem(
                        evidence_type="semantic",
                        evidence_id=sem.memory_id,
                        text=sem.text,
                        session_id=None,
                        speaker="Semantic Fact Extraction",
                        timestamp=sem.timestamp,
                        score=sem.score,
                        selection_reason="Unanchored semantic memory (source turn not in episodic context)",
                        is_anchored=False,
                        source_turn_id=src_raw,
                        metadata={
                            "entities": sem.entities,
                            "relationships": sem.relationships,
                            "topics": sem.topics,
                            "raw_source": sem.raw_source
                        }
                    )
                    unanchored_semantic_items.append(unanchored_item)
                else:
                    reason = "max_unanchored_semantic_budget_reached" if len(unanchored_semantic_items) >= max_unanchored_sem else f"score_{sem.score:.2f}_below_floor"
                    rejected_items.append({
                        "channel": "semantic",
                        "id": sem.memory_id,
                        "reason": reason
                    })

        # -------------------------------------------------------------
        # STEP 5: Apply Graph Evidence Safeguards
        # -------------------------------------------------------------
        admissible_graph_triples, graph_report = self.graph_safeguards.evaluate_triples(
            triples=graph_hits,
            query=clean_q,
            route_decision=route_decision,
            known_memory_ids=known_memory_ids,
            known_turn_ids=known_turn_ids,
            memories_by_id=mems_by_id
        )

        selected_graph_items: list[SelectedEvidenceItem] = []
        for g_triple in admissible_graph_triples:
            # Check if this graph triple can be anchored to an episodic turn via its memory_id
            g_anchored = False
            if g_triple.memory_id and mems_by_id:
                mem = mems_by_id.get(g_triple.memory_id)
                if mem:
                    m_src = mem.get("source")
                    norm_m_src = self._normalize_turn_id(m_src)
                    if norm_m_src and norm_m_src in turn_id_to_selected_item:
                        turn_id_to_selected_item[norm_m_src].anchored_graph_triples.append(g_triple.to_dict())
                        g_anchored = True

            if not g_anchored:
                # Add as standalone verified supporting graph item
                g_item = SelectedEvidenceItem(
                    evidence_type="graph",
                    evidence_id=g_triple.triple_id,
                    text=g_triple.formatted_statement + (f" (evidence: \"{g_triple.evidence}\")" if g_triple.evidence else ""),
                    session_id=None,
                    speaker="Knowledge Graph",
                    timestamp=None,
                    score=g_triple.score,
                    selection_reason="Verified provenance knowledge graph relationship",
                    is_anchored=False,
                    source_turn_id=g_triple.memory_id,
                    metadata=g_triple.metadata
                )
                selected_graph_items.append(g_item)

        # -------------------------------------------------------------
        # STEP 6: Structure Context Text, Citations & Enforce Budget
        # -------------------------------------------------------------
        structured_text, citations = self._render_context_text(
            clean_q, route_decision, selected_episodic_items, unanchored_semantic_items, selected_graph_items
        )
        token_estimate = self._estimate_tokens(structured_text)

        context_truncated = False
        truncation_actions = []

        # If rendered context exceeds max_context_tokens, apply graceful multi-tier budget trimming
        if token_estimate > self.max_context_tokens:
            context_truncated = True

            # Tier 1: Trim unanchored semantic memories (lowest priority)
            while unanchored_semantic_items and self._estimate_tokens(structured_text) > self.max_context_tokens:
                dropped = unanchored_semantic_items.pop()
                truncation_actions.append(f"trimmed_unanchored_semantic_{dropped.evidence_id}")
                structured_text, citations = self._render_context_text(
                    clean_q, route_decision, selected_episodic_items, unanchored_semantic_items, selected_graph_items
                )

            # Tier 2: Trim non-direct context window turns (keeping primary hits)
            if self._estimate_tokens(structured_text) > self.max_context_tokens:
                direct_hits = [it for it in selected_episodic_items if it.metadata.get("is_direct_hit", False)]
                if direct_hits and len(direct_hits) < len(selected_episodic_items):
                    selected_episodic_items = direct_hits
                    truncation_actions.append("trimmed_neighboring_window_turns")
                    structured_text, citations = self._render_context_text(
                        clean_q, route_decision, selected_episodic_items, unanchored_semantic_items, selected_graph_items
                    )

            # Tier 3: Truncate individual oversized turn texts if still exceeding budget
            if self._estimate_tokens(structured_text) > self.max_context_tokens:
                for _ in range(5):
                    if self._estimate_tokens(structured_text) <= self.max_context_tokens:
                        break
                    overage_tokens = self._estimate_tokens(structured_text) - self.max_context_tokens
                    chars_to_trim = int(overage_tokens * 3.5) + 25
                    longest = max(selected_episodic_items, key=lambda x: len(x.text), default=None)
                    if longest and len(longest.text) > 60:
                        new_len = max(40, len(longest.text) - chars_to_trim)
                        longest.text = longest.text[:new_len] + "... [truncated]"
                        structured_text, citations = self._render_context_text(
                            clean_q, route_decision, selected_episodic_items, unanchored_semantic_items, selected_graph_items
                        )
                    else:
                        break
                truncation_actions.append("truncated_long_turn_texts")

            token_estimate = self._estimate_tokens(structured_text)

        all_selected = selected_episodic_items + unanchored_semantic_items + selected_graph_items

        # -------------------------------------------------------------
        # STEP 7: Compile Diagnostics
        # -------------------------------------------------------------
        diagnostics = {
            "router_category": q_type,
            "context_mode": route_decision.context_mode,
            "candidate_count_per_channel": {
                "episodic": len(episodic_hits),
                "semantic": len(semantic_hits),
                "graph": len(graph_hits)
            },
            "selected_count_per_channel": {
                "episodic": len(selected_episodic_items),
                "semantic": len(unanchored_semantic_items),
                "graph": len(selected_graph_items)
            },
            "anchored_semantic_count": anchored_semantic_count,
            "unanchored_semantic_count": len(unanchored_semantic_items),
            "sessions_represented": {
                "count": len(selected_sessions),
                "sessions": selected_sessions
            },
            "context_token_estimate": token_estimate,
            "context_truncated": context_truncated,
            "truncation_actions": truncation_actions,
            "deduplicated_turn_count": deduplicated_turn_count,
            "rejected_items": rejected_items,
            "graph_safeguards_report": graph_report.to_dict()
        }

        return SelectionResult(
            selected_items=all_selected,
            structured_context_text=structured_text,
            citations=citations,
            diagnostics=diagnostics
        )
