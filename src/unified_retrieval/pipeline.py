"""Unified Multi-Source Retrieval Coordinator for IVARA.

Orchestrates query execution across Episodic, Semantic, and Knowledge Graph channels,
applies Reciprocal Rank Fusion with Provenance (RRF-P), and structures the evidence context
for downstream reasoning, temporal claim resolution, and answering.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .episodic import EpisodicRetriever, EpisodicEvidenceItem
from .semantic import SemanticRetriever, SemanticEvidenceItem
from .graph import GraphRetriever, GraphEvidenceItem
from .fusion import UnifiedEvidenceFuser, UnifiedEvidenceItem, RRFConfig
from .router import AdaptiveRouter, RouteDecision
from .graph_safeguards import GraphSafeguards, GraphSafeguardReport
from .evidence_selector import AdaptiveEvidenceSelector, SelectedEvidenceItem, SelectionResult
from ..conversations.store import ConversationStore, EpisodicTurn
from ..graph.store import SQLiteGraphStore


@dataclass
class UnifiedRetrievalResult:
    """Complete output of the Unified Retrieval Pipeline."""
    query: str
    has_sufficient_evidence: bool
    status: str
    evidence_items: list[UnifiedEvidenceItem]
    episodic_hits: list[EpisodicEvidenceItem]
    semantic_hits: list[SemanticEvidenceItem]
    graph_hits: list[GraphEvidenceItem]
    context_text: str
    direct_evidence: list[dict[str, Any]]
    inferred_evidence: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    citations: list[str]
    retrieved_memories_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "has_sufficient_evidence": self.has_sufficient_evidence,
            "status": self.status,
            "evidence_items": [e.to_dict() for e in self.evidence_items],
            "episodic_hits_count": len(self.episodic_hits),
            "semantic_hits_count": len(self.semantic_hits),
            "graph_hits_count": len(self.graph_hits),
            "context_text": self.context_text,
            "citations": self.citations,
            "diagnostics": self.diagnostics
        }


class UnifiedRetriever:
    """
    Main retrieval coordinator for IVARA.
    Retrieves evidence from episodic conversation turns, semantic memories,
    and knowledge graph triples, fusing them into a unified context.
    """

    def __init__(
        self,
        conversation_store: Optional[ConversationStore] = None,
        episodic_turns: Optional[list[EpisodicTurn]] = None,
        memories: Optional[list[dict[str, Any]]] = None,
        memories_path: Optional[str | Path] = None,
        graph_store: Optional[SQLiteGraphStore] = None,
        rrf_config: Optional[RRFConfig] = None,
        router: Optional[AdaptiveRouter] = None,
        evidence_selector: Optional[AdaptiveEvidenceSelector] = None,
        graph_safeguards: Optional[GraphSafeguards] = None,
        use_adaptive_routing: bool = False,
        user_name: Optional[str] = None
    ):
        self.conversation_store = conversation_store
        self.episodic_turns = episodic_turns
        self.memories = memories
        self.graph_store = graph_store
        self.user_name = user_name

        self.episodic_retriever = EpisodicRetriever(
            store=conversation_store,
            turns=episodic_turns,
            k_rrf=(rrf_config.k_rrf if rrf_config else 60),
            user_name=user_name
        )

        self.semantic_retriever = SemanticRetriever(
            memories=memories,
            data_path=memories_path
        )

        self.graph_retriever = GraphRetriever(store=graph_store, user_name=user_name) if graph_store else None

        self.fuser = UnifiedEvidenceFuser(config=rrf_config)
        self.rrf_config = rrf_config or RRFConfig()

        self.router = router or AdaptiveRouter(graph_store=graph_store, user_name=user_name)
        self.graph_safeguards = graph_safeguards or GraphSafeguards(user_name=user_name)
        self.evidence_selector = evidence_selector or AdaptiveEvidenceSelector(
            graph_safeguards=self.graph_safeguards
        )
        self.use_adaptive_routing = use_adaptive_routing

    def retrieve(
        self,
        query: str,
        k: int = 5,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_graph: bool = True,
        min_semantic_score: float = 0.30,
        min_episodic_score: float = 0.30,
        episodic_window_size: int = 1,
        conversation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        use_adaptive_routing: Optional[bool] = None
    ) -> UnifiedRetrievalResult:
        """
        Execute unified multi-source retrieval across all enabled channels.
        """
        clean_q = query.strip()
        if not clean_q:
            return UnifiedRetrievalResult(
                query=query,
                has_sufficient_evidence=False,
                status="insufficient_evidence",
                evidence_items=[],
                episodic_hits=[],
                semantic_hits=[],
                graph_hits=[],
                context_text=f"Query: {query}\nStatus: Insufficient evidence (empty query).",
                direct_evidence=[],
                inferred_evidence=[],
                entities=[],
                citations=[],
                retrieved_memories_count=0,
                diagnostics={"reason": "empty_query"}
            )

        # Apply adaptive routing if enabled
        active_adaptive = self.use_adaptive_routing if use_adaptive_routing is None else use_adaptive_routing
        route_decision: Optional[RouteDecision] = None
        if active_adaptive:
            route_decision = self.router.route(clean_q)
            enable_episodic = enable_episodic and route_decision.episodic_enabled
            enable_semantic = enable_semantic and route_decision.semantic_enabled
            enable_graph = enable_graph and route_decision.graph_enabled

        diagnostics: dict[str, Any] = {
            "enabled_channels": {
                "episodic": enable_episodic,
                "semantic": enable_semantic,
                "graph": enable_graph
            }
        }
        if route_decision:
            diagnostics["routing"] = route_decision.to_dict()

        # 1. Channel 1: Episodic Retrieval
        episodic_hits: list[EpisodicEvidenceItem] = []
        if enable_episodic:
            episodic_hits = self.episodic_retriever.retrieve(
                query=clean_q,
                k=k * 2,
                min_score=min_episodic_score,
                conversation_id=conversation_id,
                session_id=session_id,
                window_size=episodic_window_size
            )
        diagnostics["episodic_hit_count"] = len(episodic_hits)

        # 2. Channel 2: Semantic Retrieval
        semantic_hits: list[SemanticEvidenceItem] = []
        if enable_semantic:
            semantic_hits = self.semantic_retriever.retrieve(
                query=clean_q,
                k=k * 2,
                min_score=min_semantic_score
            )
        diagnostics["semantic_hit_count"] = len(semantic_hits)

        # 3. Channel 3: Knowledge Graph Retrieval
        graph_hits: list[GraphEvidenceItem] = []
        if enable_graph and self.graph_retriever is not None:
            # Extract additional candidate entity names from episodic and semantic hits to seed graph
            extra_entity_names = []
            for s in semantic_hits:
                for ent in s.entities:
                    name = ent.get("canonical_name") or ent.get("mention") or ent.get("name")
                    if name:
                        extra_entity_names.append(name)
            for e in episodic_hits:
                if e.speaker:
                    extra_entity_names.append(e.speaker)

            graph_hits = self.graph_retriever.retrieve(
                query=clean_q,
                k=k * 2,
                additional_entity_names=extra_entity_names
            )
        diagnostics["graph_hit_count"] = len(graph_hits)

        # 4. Evidence Selection & Fusion
        # If adaptive routing is enabled, apply AdaptiveEvidenceSelector with GraphSafeguards
        selection_res: Optional[SelectionResult] = None
        if active_adaptive and route_decision is not None:
            try:
                selection_res = self.evidence_selector.select(
                    query=clean_q,
                    route_decision=route_decision,
                    episodic_hits=episodic_hits,
                    semantic_hits=semantic_hits,
                    graph_hits=graph_hits,
                    conversation_store=self.conversation_store,
                    all_turns=self.episodic_retriever.turns,
                    all_memories=self.semantic_retriever.memories
                )
            except Exception as e:
                diagnostics["evidence_selection_fallback"] = str(e)
                selection_res = None

        if selection_res is not None:
            # Map selected items into standardized UnifiedEvidenceItem objects
            fused_items: list[UnifiedEvidenceItem] = []
            direct_evidence: list[dict[str, Any]] = []
            inferred_evidence: list[dict[str, Any]] = []
            seen_entities: dict[str, dict[str, Any]] = {}

            # Build lookups for fast retrieval object association
            epi_hits_by_id = {h.turn_id: h for h in episodic_hits}
            sem_hits_by_id = {s.memory_id: s for s in semantic_hits}
            graph_hits_by_id = {g.triple_id: g for g in graph_hits}

            for sel in selection_res.selected_items:
                c_id = f"cluster_{sel.evidence_id}"
                chans = [sel.evidence_type]
                epi_obj = epi_hits_by_id.get(sel.evidence_id)
                sem_obj = sem_hits_by_id.get(sel.evidence_id)
                graph_trips = []

                if sel.evidence_type == "episodic":
                    src_val = sel.session_id or (sel.metadata.get("conversation_id") if sel.metadata else None) or sel.source_turn_id
                    direct_evidence.append({
                        "memory_id": sel.evidence_id,
                        "text": sel.text,
                        "timestamp": sel.timestamp,
                        "session_id": sel.session_id,
                        "conversation_id": (sel.metadata.get("conversation_id") if sel.metadata else None),
                        "source": src_val,
                        "score": sel.score,
                        "direct_facts": []
                    })
                    # Attach anchored semantic memories
                    for sem_dict in sel.anchored_semantic_memories:
                        m_id = sem_dict.get("memory_id")
                        if m_id and m_id in sem_hits_by_id:
                            sem_obj = sem_hits_by_id[m_id]
                        if "semantic" not in chans:
                            chans.append("semantic")
                    # Attach anchored graph triples
                    for g_dict in sel.anchored_graph_triples:
                        t_id = g_dict.get("triple_id")
                        if t_id and t_id in graph_hits_by_id:
                            graph_trips.append(graph_hits_by_id[t_id])
                        if "graph" not in chans:
                            chans.append("graph")

                elif sel.evidence_type == "semantic":
                    if sem_obj:
                        for ent in sem_obj.entities:
                            c_name = ent.get("canonical_name") or ent.get("mention")
                            if c_name and c_name not in seen_entities:
                                seen_entities[c_name] = {
                                    "canonical_name": c_name,
                                    "types": ent.get("types", []),
                                    "entity_id": ent.get("entity_id"),
                                    "memory_id": sem_obj.memory_id
                                }
                    src_val = sel.source_turn_id or (sel.metadata.get("raw_source") if sel.metadata else None) or sel.session_id
                    direct_evidence.append({
                        "memory_id": sel.evidence_id,
                        "text": sel.text,
                        "timestamp": sel.timestamp,
                        "session_id": sel.session_id,
                        "source": src_val,
                        "score": sel.score,
                        "direct_facts": []
                    })

                elif sel.evidence_type == "graph":
                    g_obj = graph_hits_by_id.get(sel.evidence_id)
                    if g_obj:
                        graph_trips.append(g_obj)

                item_cites = [sel.evidence_id]
                for s in sel.anchored_semantic_memories:
                    if s.get("memory_id"):
                        item_cites.append(s["memory_id"])

                fused_items.append(UnifiedEvidenceItem(
                    cluster_id=c_id,
                    primary_text=sel.text,
                    timestamp=sel.timestamp,
                    session_id=sel.session_id,
                    fused_score=sel.score,
                    channels=chans,
                    episodic_turn=epi_obj,
                    semantic_memory=sem_obj,
                    graph_triples=graph_trips,
                    citations=item_cites,
                    metadata=sel.metadata
                ))

            has_sufficient = len(selection_res.selected_items) > 0
            status = "sufficient_evidence" if has_sufficient else "insufficient_evidence"
            context_text = selection_res.structured_context_text
            all_citations = selection_res.citations
            diagnostics["fused_items_count"] = len(fused_items)
            diagnostics["evidence_selection"] = selection_res.diagnostics

        else:
            # Standard RRF-P fusion fallback / default behavior
            fused_items = self.fuser.fuse(
                episodic_items=episodic_hits,
                semantic_items=semantic_hits,
                graph_items=graph_hits,
                top_k=k
            )
            diagnostics["fused_items_count"] = len(fused_items)

            # Sufficiency check
            has_sufficient = len(fused_items) > 0
            status = "sufficient_evidence" if has_sufficient else "insufficient_evidence"

            # Build structured context text and standardized evidence structures
            context_lines = [
                f"Query: {clean_q}",
                f"Status: {'Sufficient evidence (' + str(len(fused_items)) + ' relevant items found)' if has_sufficient else 'Insufficient evidence in memory store.'}\n"
            ]

            direct_evidence = []
            inferred_evidence = []
            seen_entities = {}
            all_citations = []

            if has_sufficient:
                # Section A: Episodic context turns
                context_lines.append("--- EPISODIC CONVERSATION TURNS ---")
                turns_written = 0
                for item in fused_items:
                    turn = item.episodic_turn
                    if turn:
                        turns_written += 1
                        all_citations.append(turn.turn_id)
                        context_lines.append(
                            f"[{turn.turn_id}] (relevance: {turn.score:.2f}, timestamp: {turn.timestamp}): {turn.formatted_text}"
                        )
                        for ctx in turn.context_window:
                            if ctx["relation"] != "target":
                                context_lines.append(
                                    f"  - ({ctx['relation']} turn {ctx['turn_id']}): [{ctx['speaker']}]: {ctx['content']}"
                                )

                        direct_evidence.append({
                            "memory_id": turn.turn_id,
                            "text": turn.content,
                            "timestamp": turn.timestamp,
                            "source": turn.session_id or turn.conversation_id,
                            "score": turn.score,
                            "direct_facts": []
                        })
                if turns_written == 0:
                    context_lines.append("(No direct episodic turns retrieved)")

                # Section B: Semantic Memories & Facts
                context_lines.append("\n--- SEMANTIC MEMORIES & EXTRACTED FACTS ---")
                mems_written = 0
                for item in fused_items:
                    sem = item.semantic_memory
                    if sem:
                        mems_written += 1
                        all_citations.append(sem.memory_id)
                        context_lines.append(
                            f"[{sem.memory_id}] (relevance: {sem.score:.2f}, timestamp: {sem.timestamp}): \"{sem.text}\""
                        )
                        for r in sem.relationships:
                            pred = r.get("predicate") or r.get("relation")
                            src = r.get("source", {})
                            tgt = r.get("target", {})
                            s_name = (src.get("canonical_name") or src.get("mention") or src.get("id") or str(src)) if isinstance(src, dict) else str(src)
                            t_name = (tgt.get("canonical_name") or tgt.get("mention") or tgt.get("id") or str(tgt)) if isinstance(tgt, dict) else str(tgt)
                            ev = r.get("evidence", "")
                            origin = r.get("origin", "explicit")

                            stmt = f"{s_name} {pred} {t_name}"
                            if origin == "inferred":
                                inferred_evidence.append({
                                    "memory_id": sem.memory_id,
                                    "statement": stmt,
                                    "premise_evidence": ev,
                                    "origin": "inferred"
                                })
                                context_lines.append(f"  * Inferred: {stmt} (premise: \"{ev}\")")
                            else:
                                context_lines.append(f"  * Fact: {stmt} (evidence: \"{ev}\")")

                        # Collect entities
                        for ent in sem.entities:
                            c_name = ent.get("canonical_name") or ent.get("mention")
                            if c_name and c_name not in seen_entities:
                                seen_entities[c_name] = {
                                    "canonical_name": c_name,
                                    "types": ent.get("types", []),
                                    "entity_id": ent.get("entity_id"),
                                    "memory_id": sem.memory_id
                                }

                        direct_evidence.append({
                            "memory_id": sem.memory_id,
                            "text": sem.text,
                            "timestamp": sem.timestamp,
                            "source": sem.raw_source,
                            "score": sem.score,
                            "direct_facts": [
                                {"statement": f"{r.get('source')} {r.get('predicate')} {r.get('target')}", "evidence": r.get('evidence')}
                                for r in sem.relationships if r.get("origin") != "inferred"
                            ]
                        })
                if mems_written == 0:
                    context_lines.append("(No direct semantic memories retrieved)")

                # Section C: Knowledge Graph Triples
                context_lines.append("\n--- KNOWLEDGE GRAPH RELATIONSHIPS ---")
                triples_written = 0
                for item in fused_items:
                    for trip in item.graph_triples:
                        triples_written += 1
                        status_tag = f" [{trip.predicate_status}]" if trip.predicate_status == "open_world" else ""
                        mem_cite = f"[{trip.memory_id}] " if trip.memory_id else ""
                        ev_cite = f" (evidence: \"{trip.evidence}\")" if trip.evidence else ""
                        context_lines.append(
                            f"  * {mem_cite}{trip.formatted_statement}{status_tag}{ev_cite}"
                        )
                        if trip.memory_id and trip.memory_id not in all_citations:
                            all_citations.append(trip.memory_id)
                if triples_written == 0:
                    context_lines.append("(No direct graph relationships retrieved)")

            context_text = "\n".join(context_lines)

        return UnifiedRetrievalResult(
            query=clean_q,
            has_sufficient_evidence=has_sufficient,
            status=status,
            evidence_items=fused_items,
            episodic_hits=episodic_hits,
            semantic_hits=semantic_hits,
            graph_hits=graph_hits,
            context_text=context_text,
            direct_evidence=direct_evidence,
            inferred_evidence=inferred_evidence,
            entities=list(seen_entities.values()),
            citations=list(dict.fromkeys(all_citations)),
            retrieved_memories_count=len(fused_items),
            diagnostics=diagnostics
        )
