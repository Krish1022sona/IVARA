"""
Chunking engine for IVARA memories and long-context episodes.

Splits long narrative memories into coherent, fact-bearing chunks while
preserving exact source provenance, character offsets, sequential ordering,
and canonical entity associations.
"""

import re
from typing import Any
from .schema import ChunkNode


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """
    Split text into sentences with start and end character offsets.
    """
    if not text:
        return []

    # Match sentences ending in period, exclamation, or question mark followed by space or end-of-string
    pattern = re.compile(r'([A-Z0-9\s\'\"\(\[\{][^.!?]*[.!?]+(?:\s+|$)|[^.!?]+$)')
    sentences = []
    for match in pattern.finditer(text):
        sentence_str = match.group().strip()
        if sentence_str:
            sentences.append((sentence_str, match.start(), match.end()))

    if not sentences and text.strip():
        sentences = [(text.strip(), 0, len(text))]

    return sentences


def chunk_memory(
    memory: dict[str, Any],
    target_words: int = 120,
    overlap_sentences: int = 1
) -> list[ChunkNode]:
    """
    Chunk an ingested memory record into coherent fact-bearing ChunkNode objects.

    Args:
        memory: Stored memory record containing 'id', 'text', 'timestamp', 'entities', etc.
        target_words: Target word length per chunk (default ~120 words).
        overlap_sentences: Number of overlapping sentences to carry over for context preservation.

    Returns:
        List of ChunkNode objects linking back to the parent memory.
    """
    memo_id = memory.get("id", "m000")
    memo_text = memory.get("text", "").strip()
    memo_ts = memory.get("timestamp")
    memo_source = memory.get("source", "memory")

    if not memo_text:
        return []

    # Map available canonical entity mentions for entity linking
    known_entities = memory.get("entities", [])
    entity_id_map = {}
    for ent in known_entities:
        c_id = ent.get("entity_id") or ent.get("canonical_name")
        mention = ent.get("mention") or ent.get("canonical_name")
        if c_id and mention:
            entity_id_map[mention.lower()] = c_id
            if ent.get("canonical_name"):
                entity_id_map[ent["canonical_name"].lower()] = c_id

    total_words = len(memo_text.split())

    # If memory is short (within ~1.5x target words), preserve as a single pristine chunk
    if total_words <= int(target_words * 1.5):
        chunk_ent_ids = list(set(
            ent.get("entity_id") or ent.get("canonical_name")
            for ent in known_entities
            if (ent.get("entity_id") or ent.get("canonical_name"))
        ))
        return [
            ChunkNode(
                id=f"chunk_{memo_id}_00",
                memory_id=memo_id,
                episode_id=memo_source if "episode" in str(memo_source).lower() else None,
                text=memo_text,
                chunk_index=0,
                char_start=0,
                char_end=len(memo_text),
                timestamp=memo_ts,
                origin="explicit",
                confidence=memory.get("confidence", 1.0),
                entity_ids=chunk_ent_ids
            )
        ]

    # Otherwise, split sentences and assemble fact-bearing chunks
    sentences = split_sentences(memo_text)
    if not sentences:
        sentences = [(memo_text, 0, len(memo_text))]

    chunks: list[ChunkNode] = []
    current_sentences: list[tuple[str, int, int]] = []
    current_words = 0
    chunk_idx = 0

    i = 0
    while i < len(sentences):
        sent, start, end = sentences[i]
        sent_words = len(sent.split())
        current_sentences.append((sent, start, end))
        current_words += sent_words

        # If target reached or at end of text
        if current_words >= target_words or i == len(sentences) - 1:
            chunk_text = " ".join(s[0] for s in current_sentences)
            c_start = current_sentences[0][1]
            c_end = current_sentences[-1][2]

            # Detect entities present in this chunk
            chunk_lower = chunk_text.lower()
            chunk_ents = set()
            for mention, ent_id in entity_id_map.items():
                if mention in chunk_lower:
                    chunk_ents.add(ent_id)

            chunks.append(
                ChunkNode(
                    id=f"chunk_{memo_id}_{chunk_idx:02d}",
                    memory_id=memo_id,
                    episode_id=memo_source if "episode" in str(memo_source).lower() else None,
                    text=chunk_text,
                    chunk_index=chunk_idx,
                    char_start=c_start,
                    char_end=c_end,
                    timestamp=memo_ts,
                    origin="explicit",
                    confidence=memory.get("confidence", 1.0),
                    entity_ids=sorted(list(chunk_ents))
                )
            )
            chunk_idx += 1

            # Prepare next chunk with sentence overlap if specified
            if overlap_sentences > 0 and i < len(sentences) - 1:
                overlap_slice = current_sentences[-overlap_sentences:]
                current_sentences = list(overlap_slice)
                current_words = sum(len(s[0].split()) for s in current_sentences)
            else:
                current_sentences = []
                current_words = 0

        i += 1

    return chunks
