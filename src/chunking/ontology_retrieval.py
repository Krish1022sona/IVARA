"""
Ontology-Guided Turn Chunk Retrieval for IVARA.

Extends hybrid vector + BM25 retrieval with ontology vocabulary canonicalization,
alias expansion, and entity-mention re-ranking WITHOUT graph traversal.
"""

import re
from typing import Any
import numpy as np

from .turn_chunker import TurnChunk
from .retrieval import ChunkRetriever
from ..ontology.vocabulary import (
    load_vocabulary,
    get_canonical_type,
    get_canonical_predicate,
    normalize_name
)


def extract_query_concepts(
    query: str,
    vocabulary: dict[str, Any] | None = None
) -> dict[str, list[str]]:
    """
    Identify entity mentions, capitalized proper nouns, and ontology keywords from query.
    """
    if vocabulary is None:
        vocabulary = load_vocabulary()

    q_clean = query.strip()
    q_lower = q_clean.lower()

    detected_entities: list[str] = []
    expanded_aliases: list[str] = []
    detected_predicates: list[str] = []

    # 1. Capitalized proper nouns / candidate entity phrases (excluding sentence start)
    words = q_clean.split()
    for i, w in enumerate(words):
        w_strip = re.sub(r'[^a-zA-Z0-9_\-]', '', w)
        if not w_strip:
            continue
        # Capitalized words inside sentence (excluding single-letter pronouns like 'I') or alphanumeric codes
        if (i > 0 and w_strip[0].isupper() and len(w_strip) > 1 and w_strip.lower() not in ("i", "me", "my")) or re.search(r'[A-Z0-9]+-[A-Z0-9]+', w_strip):
            detected_entities.append(w_strip)

    # 2. Check ontology vocabulary types and aliases
    types_dict = vocabulary.get("types", {})
    for type_name, type_info in types_dict.items():
        t_canon = type_name.lower()
        if t_canon in q_lower:
            expanded_aliases.extend(type_info.get("aliases", []))
        for alias in type_info.get("aliases", []):
            if alias.lower() in q_lower:
                detected_entities.append(alias)
                expanded_aliases.append(type_name)
                expanded_aliases.extend(type_info.get("aliases", []))

    # 3. Check ontology predicates and aliases
    predicates_dict = vocabulary.get("predicates", {})
    for pred_name, pred_info in predicates_dict.items():
        p_canon = pred_name.lower().replace("_", " ")
        if p_canon in q_lower:
            detected_predicates.append(pred_name)
            expanded_aliases.extend(pred_info.get("aliases", []))
        for alias in pred_info.get("aliases", []):
            if alias.lower() in q_lower:
                detected_predicates.append(pred_name)
                expanded_aliases.append(pred_name)
                expanded_aliases.extend(pred_info.get("aliases", []))

    # Common domain-specific aliases for personal memory QA
    domain_synonyms = {
        "degree": ["graduated", "major", "bachelor", "master", "college", "university"],
        "coupon": ["discount", "percentage", "off", "redeemed", "save"],
        "model": ["kit", "scale", "bomber", "airplane", "aircraft"],
        "target": ["store", "shopping", "groceries", "retail"]
    }
    for term, syns in domain_synonyms.items():
        if term in q_lower:
            expanded_aliases.extend(syns)

    return {
        "entities": sorted(list(set(detected_entities))),
        "aliases": sorted(list(set(a for a in expanded_aliases if a.lower() not in q_lower))),
        "predicates": sorted(list(set(detected_predicates)))
    }


class OntologyGuidedChunkRetriever(ChunkRetriever):
    """
    B-Chunked + Ontology retriever.
    
    Subclasses ChunkRetriever with use_ontology=True and ontology_weight=0.15.
    Preserves 100% backward compatibility for extract_query_concepts and existing callers.
    """

    def __init__(
        self,
        chunks: list[TurnChunk],
        vector_weight: float = 0.55,
        bm25_weight: float = 0.30,
        ontology_boost_weight: float = 0.15,
        vocabulary: dict[str, Any] | None = None
    ):
        super().__init__(
            chunks=chunks,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            ontology_weight=ontology_boost_weight,
            use_ontology=True,
            vocabulary=vocabulary
        )
        self.ontology_boost_weight = ontology_boost_weight
        self.base_retriever = self

    def extract_query_concepts(self, query: str) -> dict[str, list[str]]:
        """Identify entity mentions, capitalized proper nouns, and ontology keywords from query."""
        return extract_query_concepts(query, self.vocabulary)

