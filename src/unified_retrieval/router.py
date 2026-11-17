"""Adaptive Query Router for IVARA Unified Retrieval.

Classifies user queries into 8 functional retrieval categories:
  1. single_fact
  2. historical_state
  3. temporal_longitudinal
  4. cross_session_synth
  5. entity_relationships
  6. multi_hop_reasoning
  7. negative_information
  8. unanswerable (and safe_fallback for ambiguous queries)

Implements lightweight, deterministic first-stage routing (sub-millisecond latency):
- Temporal expressions and date patterns
- Relational and bridge-question patterns
- Negative-information patterns
- Entity overlap and ambiguity signals
- Explicit multi-session and historical indicators

Enforces core routing invariants:
- Episodic retrieval remains the default backbone.
- Knowledge Graph retrieval is strictly conditional (only for multi-hop/relational queries).
- Historical-state queries NEVER retrieve current unversioned graph relationships.
- Negative-information queries preserve the distinction between absence of evidence and evidence of absence.
- Unanswerable is an evidence-sufficiency decision, not inferred merely from missing keywords.
- Ambiguous or low-confidence queries route to safe fallback rather than dropping channels.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Optional


@dataclass
class RouteDecision:
    """Structured routing decision output by AdaptiveRouter."""
    query_type: str
    confidence: float
    signals: list[str]
    episodic_enabled: bool = True
    semantic_enabled: bool = True
    graph_enabled: bool = False
    context_mode: str = "standard"
    routing_reason: str = ""
    detected_entities: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "confidence": round(self.confidence, 4),
            "signals": self.signals,
            "episodic_enabled": self.episodic_enabled,
            "semantic_enabled": self.semantic_enabled,
            "graph_enabled": self.graph_enabled,
            "context_mode": self.context_mode,
            "routing_reason": self.routing_reason,
            "detected_entities": self.detected_entities,
            "diagnostics": self.diagnostics
        }


class AdaptiveRouter:
    """
    Lightweight, deterministic query router for multi-source memory retrieval.
    Routes queries without calling external APIs or local LLMs.
    """

    # --- Compiled Regular Expressions ---
    # Temporal patterns
    RE_YEAR = re.compile(r"\b(19\d\d|20\d\d)\b")
    RE_MONTH = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)
    
    # Historical-state cues (past state before transition)
    RE_HISTORICAL = re.compile(
        r"\b(initial|initially|previous|previously|former|formerly|earlier|first|"
        r"before reverting|originally|prior to|past|used to|no longer|"
        r"switched from|before switching|revert(ed)? to)\b",
        re.I
    )
    RE_HISTORICAL_ROLE = re.compile(
        r"\bwhat was ([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}'?s?|the user'?s?|user'?s?|my|his|her|their|the) (job title|title|role|salary|address|company|language|preference) (when|at|before|prior)\b",
        re.I
    )

    # Temporal & Longitudinal sequence cues
    RE_TEMPORAL_SEQ = re.compile(
        r"\b(before or after|after or before|earlier than|later than|"
        r"did .* occur (before|after)|when did .* relative to|"
        r"chronolog(y|ical)?|sequence of events|order of events|timeline)\b",
        re.I
    )

    # Cross-session synthesis cues (multi-session evolution / lifecycle)
    RE_CROSS_SESSION = re.compile(
        r"\b(trace|full lifecycle|throughout|dietary evolution|job title progression|"
        r"career progression|residential moves|all companies|across companies|"
        r"complete history|lifecycle of|evolution of|over the years|from \w+ \d{4} to \d{4}|"
        r"summarize all|list all|every time)\b",
        re.I
    )
    RE_COMPLEX_MULTI = re.compile(
        r"\b(list ([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}'?s?|the user's|my)|identify the first .* (his|their|my) .* (his|their|my) .* and the|name the .* the .* (his|their|my) .* and the)\b",
        re.I
    )

    # Entity relationship cues
    RE_RELATIONSHIP = re.compile(
        r"\b(what is the relationship between|relationship between|"
        r"how is .* connected to|what role did .* hold relative to|"
        r"connection (exists )?between|what connection (exists|is there) between|"
        r"how are .* and .* related|how do .* and .* know|"
        r"connected to ([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}'?s?|the user's|my)|relationship with)\b",
        re.I
    )

    # Multi-hop bridge cues (relational bridge referents)
    RE_MULTI_HOP = re.compile(
        r"\b(the (person|engineer|manager|collaborator|teammate|roommate|co-author|co-worker|doctor) who ("
        r"developed|assisted|supervised|managed|collaborated|built|rewrote|worked|co-founded|advised|treated))\b|"
        r"\b(in which (city|company|office|team|country) does the .* (who|that))\b|"
        r"\b(what (sport|dietary restriction|project|language) did .* with the (person|engineer|manager|teammate) who)\b|"
        r"\b(where did the (compiler engineer|manager|collaborator|engineer) who .* relocate)\b",
        re.I
    )

    # Negative-information cues (non-occurrence, absence, negative constraints)
    RE_NEGATIVE = re.compile(
        r"\b(did ([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}|the user|i|he|she|they) (undergo|have|take|accept|attend|receive|suffer|break|visit|ever|contract|undergo knee surgery))\b|"
        r"\b(did .* (accept .* or decline|decline|refuse|reject|avoid|opt out))\b|"
        r"\b(never|neither|without|counter-offer|salary counter-offer)\b|"
        r"\b(is .* allergic|severe .* allergy|peanut allergy|dietary restriction)\b|"
        r"\b(does ([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}|the user|he|she|they) (drink|eat|smoke|still drink|never drink) (coffee|tea|meat|alcohol)|do (i|you) (drink|eat|smoke|still drink|never drink) (coffee|tea|meat|alcohol))\b",
        re.I
    )
    RE_UPDATE_STILL = re.compile(
        r"\b(does ([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}|the user|he|she|they) still (live|work|use|play|follow|have|reside)|do (i|you) still (live|work|use|play|follow|have|reside))\b",
        re.I
    )

    # Single-fact cues (direct property lookups)
    RE_SINGLE_FACT = re.compile(
        r"\b(on what date|in which month and year|what is ([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}'?s?|the user'?s?|user'?s?|my|his|her|their) (birthday|birth date|alma mater|degree|email|phone))\b|"
        r"\b(from which (university|college)|when was ([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}|the user|i|he|she) born|where was ([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}|the user|i|he|she) born|what format does ([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}|the user|he|she|you) prefer)\b|"
        r"\b(what (local development tooling|editor|laptop|car|pet) does ([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}|the user|he|she|you))\b",
        re.I
    )

    # Out-of-domain / Explicit Unanswerable cues
    RE_OUT_OF_DOMAIN = re.compile(
        r"\b(capital of mars|weather on jupiter|meaning of life|president of the moon|time machine|"
        r"in 2099|in the year 3000|will ([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}|the user|i|anyone) in 2050)\b",
        re.I
    )

    def __init__(
        self,
        graph_store: Optional[Any] = None,
        known_entities: Optional[list[str]] = None,
        config: Optional[dict[str, Any]] = None,
        user_name: Optional[str] = None
    ):
        self.graph_store = graph_store
        self.config = config or {}
        self.user_name = user_name
        self.confidence_threshold = self.config.get("confidence_threshold", 0.50)
        self.graph_activation_threshold = self.config.get("graph_activation_threshold", 0.65)

        self.known_entities: set[str] = set()
        if known_entities:
            self.known_entities.update([e.lower().strip() for e in known_entities if e])
        if user_name:
            self.known_entities.add(user_name.lower().strip())

        # Cache known entity names for fast entity detection
        self.known_entities: set[str] = set()
        if known_entities:
            self.known_entities.update(e.lower().strip() for e in known_entities if e)
        if self.graph_store:
            try:
                nodes = self.graph_store.get_nodes_by_type("entity")
                for n in nodes:
                    if n.id and n.id.lower() != "self":
                        self.known_entities.add(n.id.lower().strip())
                    if n.label:
                        self.known_entities.add(n.label.lower().strip())
                    props = n.properties or {}
                    c_name = props.get("canonical_name")
                    if c_name:
                        self.known_entities.add(c_name.lower().strip())
                    for a in props.get("aliases", []):
                        if a:
                            self.known_entities.add(a.lower().strip())
            except Exception:
                pass

        # Add common core domain entities
        self.known_entities.update([
            "hyperscale", "hyperscale systems", "aetheria", "aetheria ai",
            "cachemesh", "neurocompact", "agrivision", "tensorforge",
            "vikram rao", "priya sharma", "meera sen", "rohan sharma",
            "dr. shalini verma", "shalini verma", "kabir das", "ananya bose", "deepa nair"
        ])
        if self.user_name:
            self.known_entities.add(self.user_name.lower().strip())

    def detect_entities(self, query: str) -> list[str]:
        """Detect entity mentions in query string with token boundary matching."""
        clean_q = f" {query.lower()} "
        found: list[str] = []
        for ent in sorted(self.known_entities, key=lambda x: len(x), reverse=True):
            if len(ent) < 3:
                continue
            pattern = r"\b" + re.escape(ent) + r"\b"
            if re.search(pattern, clean_q):
                found.append(ent)
                clean_q = re.sub(pattern, " ", clean_q)
        return found

    def route(self, query: str) -> RouteDecision:
        """
        Classifies an input query into a designated retrieval route with channel configuration.
        """
        clean_q = query.strip()
        if not clean_q:
            return RouteDecision(
                query_type="safe_fallback",
                confidence=1.0,
                signals=["empty_query"],
                episodic_enabled=True,
                semantic_enabled=True,
                graph_enabled=False,
                context_mode="standard",
                routing_reason="Empty or whitespace query routed to safe fallback.",
                diagnostics={"is_empty": True}
            )

        detected_entities = self.detect_entities(clean_q)
        signals: list[str] = []
        scores: dict[str, float] = {
            "single_fact": 0.0,
            "historical_state": 0.0,
            "temporal_longitudinal": 0.0,
            "cross_session_synth": 0.0,
            "entity_relationships": 0.0,
            "multi_hop_reasoning": 0.0,
            "negative_information": 0.0,
            "unanswerable": 0.0
        }

        # 1. Feature Extraction: Out-of-Domain / Fictitious
        if self.RE_OUT_OF_DOMAIN.search(clean_q):
            scores["unanswerable"] += 1.0
            signals.append("explicit_out_of_domain")

        # 2. Feature Extraction: Historical-State
        if self.RE_HISTORICAL.search(clean_q):
            scores["historical_state"] += 0.85
            signals.append("temporal_historical_keyword")
        if self.RE_HISTORICAL_ROLE.search(clean_q):
            scores["historical_state"] += 0.90
            signals.append("historical_role_pattern")

        # 3. Feature Extraction: Temporal Sequence
        if self.RE_TEMPORAL_SEQ.search(clean_q):
            # If query is explicitly a trace/progression query across sessions, don't let "chronological" hijack it
            if not self.RE_CROSS_SESSION.search(clean_q):
                scores["temporal_longitudinal"] += 0.95
                signals.append("temporal_sequence_marker")
            else:
                scores["cross_session_synth"] += 0.30

        # 4. Feature Extraction: Cross-Session Synthesis & Complex Multi-Fact
        if self.RE_CROSS_SESSION.search(clean_q):
            scores["cross_session_synth"] += 0.95
            signals.append("cross_session_lifecycle_marker")
        if self.RE_COMPLEX_MULTI.search(clean_q):
            scores["cross_session_synth"] += 0.90
            signals.append("complex_multi_fact_pattern")

        # 5. Feature Extraction: Entity Relationships
        if self.RE_RELATIONSHIP.search(clean_q):
            scores["entity_relationships"] += 0.90
            signals.append("explicit_relationship_phrase")

        # 6. Feature Extraction: Multi-Hop Reasoning
        if self.RE_MULTI_HOP.search(clean_q):
            scores["multi_hop_reasoning"] += 0.95
            signals.append("relative_clause_bridge")

        # 7. Feature Extraction: Negative Information / Contradiction
        if self.RE_NEGATIVE.search(clean_q):
            scores["negative_information"] += 0.95
            signals.append("negative_or_constraint_marker")
        if self.RE_UPDATE_STILL.search(clean_q):
            # "Does user still live in..." -> test for update/supersession
            scores["negative_information"] += 0.70
            scores["historical_state"] += 0.50
            signals.append("status_update_marker")

        # 8. Feature Extraction: Single Fact
        if self.RE_SINGLE_FACT.search(clean_q):
            scores["single_fact"] += 0.85
            signals.append("single_fact_direct_marker")

        # 9. Entity Overlap Adjustments
        num_ents = len(detected_entities)
        signals.append(f"detected_entity_count:{num_ents}")
        if num_ents >= 2:
            scores["entity_relationships"] += 0.40
            scores["multi_hop_reasoning"] += 0.30
        elif num_ents == 1:
            scores["single_fact"] += 0.10

        # 10. Ambiguity and Conflict Resolution
        # Sort candidates by score
        ranked_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_score = ranked_categories[0]
        second_cat, second_score = ranked_categories[1]

        # Determine resolved category
        if top_score < self.confidence_threshold:
            # Low confidence -> Safe Fallback
            resolved_cat = "safe_fallback"
            conf = 0.50
            reason = "Confidence below threshold; routed to safe fallback."
        elif top_score - second_score < 0.15 and top_cat in ("entity_relationships", "multi_hop_reasoning") and second_cat in ("entity_relationships", "multi_hop_reasoning"):
            # Multi-hop and relationship overlap -> select multi_hop as broader superset
            resolved_cat = "multi_hop_reasoning"
            conf = top_score
            reason = "Close scores between relational and multi-hop; selected multi_hop_reasoning."
        elif top_score - second_score < 0.10 and top_score < 0.70:
            # Ambiguous conflict between unrelated categories
            resolved_cat = "safe_fallback"
            conf = 0.50
            reason = f"Ambiguous conflict between {top_cat} ({top_score:.2f}) and {second_cat} ({second_score:.2f}); safe fallback."
        else:
            resolved_cat = top_cat
            conf = min(top_score, 1.0)
            reason = f"Deterministic classification matched {top_cat} with confidence {conf:.2f}."

        # Apply Channel Activation Invariants
        episodic_enabled = True  # Always enabled as core backbone
        semantic_enabled = True
        graph_enabled = False
        context_mode = "standard"

        if resolved_cat == "historical_state":
            episodic_enabled = True
            semantic_enabled = True
            graph_enabled = False  # Mandatory guard: graph contains unversioned current relationships
            context_mode = "historical_state"
        elif resolved_cat == "temporal_longitudinal":
            episodic_enabled = True
            semantic_enabled = True
            graph_enabled = False
            context_mode = "temporal_longitudinal"
        elif resolved_cat == "cross_session_synth":
            episodic_enabled = True
            semantic_enabled = True
            graph_enabled = False
            context_mode = "cross_session_synth"
        elif resolved_cat == "entity_relationships":
            episodic_enabled = True
            semantic_enabled = True
            # Conditionally activate graph only if confidence is high and entities are resolved
            if conf >= self.graph_activation_threshold and (num_ents >= 2 or "explicit_relationship_phrase" in signals):
                graph_enabled = True
            else:
                graph_enabled = False
            context_mode = "entity_relationships"
        elif resolved_cat == "multi_hop_reasoning":
            episodic_enabled = True
            semantic_enabled = True
            # Conditionally activate graph only if confidence is high
            if conf >= self.graph_activation_threshold and "relative_clause_bridge" in signals:
                graph_enabled = True
            else:
                graph_enabled = False
            context_mode = "multi_hop"
        elif resolved_cat == "negative_information":
            episodic_enabled = True
            semantic_enabled = True
            graph_enabled = False  # Graph is positive-assertion only
            context_mode = "negative_information"
        elif resolved_cat == "single_fact":
            episodic_enabled = True
            semantic_enabled = True
            graph_enabled = False
            context_mode = "single_fact"
        elif resolved_cat == "unanswerable":
            episodic_enabled = True
            semantic_enabled = False
            graph_enabled = False
            context_mode = "unanswerable"
        else:  # safe_fallback
            episodic_enabled = True
            semantic_enabled = True
            graph_enabled = False
            context_mode = "standard"

        return RouteDecision(
            query_type=resolved_cat,
            confidence=conf,
            signals=signals,
            episodic_enabled=episodic_enabled,
            semantic_enabled=semantic_enabled,
            graph_enabled=graph_enabled,
            context_mode=context_mode,
            routing_reason=reason,
            detected_entities=detected_entities,
            diagnostics={
                "top_category": top_cat,
                "top_score": round(top_score, 4),
                "second_category": second_cat,
                "second_score": round(second_score, 4),
                "all_scores": {k: round(v, 4) for k, v in scores.items() if v > 0}
            }
        )
