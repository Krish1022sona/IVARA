"""
Temporal Claim Representation and Extraction Helpers for IVARA.

Defines TemporalClaim with full provenance, valid intervals, recording time,
and directed supersession links.
"""

from enum import Enum
import hashlib
import re
from typing import Any
from pydantic import BaseModel, Field

from ..graph.schema import GraphNode, TemporalClaim


def generate_claim_id(subject: str, predicate: str, object_val: str, valid_from: str | None = None) -> str:
    """Generate a deterministic, stable claim identifier."""
    raw = f"{subject.strip().lower()}:{predicate.strip().lower()}:{object_val.strip().lower()}:{valid_from or 'none'}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"claim_{h}"


def extract_year(date_str: str | None) -> int | None:
    """Extract 4-digit calendar year from date string if present."""
    if not date_str:
        return None
    m = re.search(r'\b(19\d\d|20\d\d)\b', str(date_str))
    return int(m.group(1)) if m else None


def overlaps_year(valid_from: str | None, valid_to: str | None, target_year: int) -> bool:
    """Determine whether valid interval [valid_from, valid_to] covers target_year."""
    y_from = extract_year(valid_from)
    y_to = extract_year(valid_to)

    if y_from is not None and y_to is not None:
        return y_from <= target_year <= y_to
    elif y_from is not None:
        # Open-ended from y_from
        return y_from <= target_year
    elif y_to is not None:
        return target_year <= y_to
    return False


def build_claims_from_text(
    text: str,
    session_id: str = "",
    chunk_id: str = "",
    recorded_at: str | None = None,
    subject: str = "user"
) -> list[TemporalClaim]:
    """
    Deterministic pattern extractor for common facts from chunk text without LLM calls.
    Extracts residence, employment, hobbies, and visits with explicit dates and provenance.
    """
    claims: list[TemporalClaim] = []
    lines = text.split("\n")

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Helper to determine effective subject
        def _get_subj(m):
            sub = m.groupdict().get("subname")
            if sub and sub.lower() not in ("i", "user", "me"):
                return sub.strip()
            return subject

        # 1. Residence Patterns
        # e.g., "I lived in Delhi in 2025" or "Alice lived in Delhi in 2025"
        m_lived = re.search(
            r'(?:I|User: I|(?P<subname>[A-Z][a-z]+))\s+lived\s+in\s+([A-Za-z\s]+?)(?:\s+in\s+(\d{4}))?(?:[.,!]|\s*$|\n)',
            line_clean,
            re.IGNORECASE
        )
        if m_lived:
            loc = m_lived.group(2).strip()
            yr = m_lived.group(3)
            eff_sub = _get_subj(m_lived)
            c_id = generate_claim_id(eff_sub, "lives_in", loc, yr)
            claims.append(TemporalClaim(
                claim_id=c_id,
                subject=eff_sub,
                predicate="lives_in",
                object=loc,
                valid_from=yr,
                valid_to=yr if yr else None,
                recorded_at=recorded_at,
                source_chunk_id=chunk_id,
                is_explicit=True,
                confidence=1.0,
                evidence=m_lived.group(0).strip(),
                properties={"session_id": session_id, "text": line_clean}
            ))
            continue

        # e.g., "I moved to Pune in 2026" or "Bob moved to Pune"
        m_moved = re.search(
            r'(?:I|User: I|(?P<subname>[A-Z][a-z]+))\s+moved\s+to\s+([A-Za-z\s]+?)(?:\s+in\s+(\d{4}))?(?:[.,!]|\s*$|\n)',
            line_clean,
            re.IGNORECASE
        )
        if m_moved:
            loc = m_moved.group(2).strip()
            yr = m_moved.group(3)
            eff_sub = _get_subj(m_moved)
            c_id = generate_claim_id(eff_sub, "lives_in", loc, yr)
            claims.append(TemporalClaim(
                claim_id=c_id,
                subject=eff_sub,
                predicate="lives_in",
                object=loc,
                valid_from=yr,
                valid_to=None,
                recorded_at=recorded_at,
                source_chunk_id=chunk_id,
                is_explicit=True,
                confidence=1.0,
                evidence=m_moved.group(0).strip(),
                properties={"session_id": session_id, "text": line_clean, "is_transition": True}
            ))
            continue

        # e.g., "I live in Delhi" / "I currently live in Pune"
        m_live = re.search(
            r'(?:I|User: I|(?P<subname>[A-Z][a-z]+))\s+(?:currently\s+)?(?:live|lives)\s+in\s+([A-Za-z\s]+?)(?:\s+in\s+(\d{4}))?(?:[.,!]|\s*$|\n)',
            line_clean,
            re.IGNORECASE
        )
        if m_live:
            loc = m_live.group(2).strip()
            yr = m_live.group(3)
            eff_sub = _get_subj(m_live)
            c_id = generate_claim_id(eff_sub, "lives_in", loc, yr)
            claims.append(TemporalClaim(
                claim_id=c_id,
                subject=eff_sub,
                predicate="lives_in",
                object=loc,
                valid_from=yr,
                valid_to=None,
                recorded_at=recorded_at,
                source_chunk_id=chunk_id,
                is_explicit=True,
                confidence=1.0,
                evidence=m_live.group(0).strip(),
                properties={"session_id": session_id, "text": line_clean}
            ))
            continue

        # 2. Employment Patterns
        # e.g. "I work at Acme Corp" or "I worked at Acme Corp in 2024"
        m_work = re.search(
            r'(?:I|User: I|(?P<subname>[A-Z][a-z]+))\s+(work(?:ed|s)?)\s+at\s+([A-Za-z0-9\s]+?)(?:\s+in\s+(\d{4}))?(?:[.,!]|\s*$|\n)',
            line_clean,
            re.IGNORECASE
        )
        if m_work:
            tense = m_work.group(2).lower()
            org = m_work.group(3).strip()
            yr = m_work.group(4)
            eff_sub = _get_subj(m_work)
            c_id = generate_claim_id(eff_sub, "works_at", org, yr)
            claims.append(TemporalClaim(
                claim_id=c_id,
                subject=eff_sub,
                predicate="works_at",
                object=org,
                valid_from=yr,
                valid_to=yr if (tense == "worked" and yr) else None,
                recorded_at=recorded_at,
                source_chunk_id=chunk_id,
                is_explicit=True,
                confidence=1.0,
                evidence=m_work.group(0).strip(),
                properties={"session_id": session_id, "text": line_clean}
            ))
            continue

        # 3. Visited Places Patterns (Non-functional)
        m_visit = re.search(
            r'(?:I|User: I|(?P<subname>[A-Z][a-z]+))\s+visited\s+([A-Za-z\s]+?)(?:\s+in\s+(\d{4}))?(?:[.,!]|\s*$|\n)',
            line_clean,
            re.IGNORECASE
        )
        if m_visit:
            place = m_visit.group(2).strip()
            yr = m_visit.group(3)
            eff_sub = _get_subj(m_visit)
            c_id = generate_claim_id(eff_sub, "visited", place, yr)
            claims.append(TemporalClaim(
                claim_id=c_id,
                subject=eff_sub,
                predicate="visited",
                object=place,
                valid_from=yr,
                valid_to=yr if yr else None,
                recorded_at=recorded_at,
                source_chunk_id=chunk_id,
                is_explicit=True,
                confidence=1.0,
                evidence=m_visit.group(0).strip(),
                properties={"session_id": session_id, "text": line_clean}
            ))
            continue

        # 4. Hobby Patterns (Non-functional)
        m_hobby = re.search(
            r'(?:(?:My|(?P<subname>[A-Z][a-z]+)\'s)\s+hobby\s+is|I\s+(?:like|enjoy)\s+playing)\s+([A-Za-z\s]+?)(?:[.,!]|\s*$|\n)',
            line_clean,
            re.IGNORECASE
        )
        if m_hobby:
            hob = m_hobby.group(2).strip()
            eff_sub = _get_subj(m_hobby)
            c_id = generate_claim_id(eff_sub, "has_hobby", hob, None)
            claims.append(TemporalClaim(
                claim_id=c_id,
                subject=eff_sub,
                predicate="has_hobby",
                object=hob,
                valid_from=None,
                valid_to=None,
                recorded_at=recorded_at,
                source_chunk_id=chunk_id,
                is_explicit=True,
                confidence=1.0,
                evidence=m_hobby.group(0).strip(),
                properties={"session_id": session_id, "text": line_clean}
            ))
            continue

        # 5. Preference Patterns (Functional / Context-dependent)
        # e.g., "I used to prefer dark roast coffee, but now I prefer light roast"
        m_pref_update = re.search(
            r'(?:I|User: I)\s+used\s+to\s+prefer\s+([A-Za-z0-9\s\-]+?),\s+but\s+(?:now\s+)?I\s+prefer\s+([A-Za-z0-9\s\-]+?)(?:[.,!]|\s*$|\n)',
            line_clean,
            re.IGNORECASE
        )
        if m_pref_update:
            old_pref = m_pref_update.group(1).strip()
            new_pref = m_pref_update.group(2).strip()
            c_old_id = generate_claim_id(subject, "prefers", old_pref, "historical")
            c_new_id = generate_claim_id(subject, "prefers", new_pref, "current")

            claims.append(TemporalClaim(
                claim_id=c_old_id,
                subject=subject,
                predicate="prefers",
                object=old_pref,
                valid_from=None,
                valid_to="prior",
                recorded_at=recorded_at,
                source_chunk_id=chunk_id,
                is_explicit=True,
                confidence=1.0,
                evidence=m_pref_update.group(0).strip(),
                properties={"session_id": session_id, "text": line_clean, "is_current": False, "superseded_by": c_new_id}
            ))
            claims.append(TemporalClaim(
                claim_id=c_new_id,
                subject=subject,
                predicate="prefers",
                object=new_pref,
                valid_from=None,
                valid_to=None,
                recorded_at=recorded_at,
                source_chunk_id=chunk_id,
                is_explicit=True,
                confidence=1.0,
                supersedes_claim_id=c_old_id,
                evidence=m_pref_update.group(0).strip(),
                properties={"session_id": session_id, "text": line_clean, "is_current": True, "supersession_reason": "Explicit preference transition"}
            ))
            continue

        # e.g., "I prefer Python at work" or "I prefer Rust for personal projects"
        m_pref_context = re.search(
            r'(?:I|User: I|(?P<subname>[A-Z][a-z]+))\s+prefer(?:s)?\s+([A-Za-z0-9\s\-]+?)(?:\s+(at\s+work|for\s+personal\s+projects|at\s+home))?(?:[.,!]|\s*$|\n)',
            line_clean,
            re.IGNORECASE
        )
        if m_pref_context:
            pref_val = m_pref_context.group(2).strip()
            ctx_scope = m_pref_context.group(3)
            eff_sub = _get_subj(m_pref_context)
            eff_scope = ctx_scope.lower().strip() if ctx_scope else "general"
            c_id = generate_claim_id(eff_sub, f"prefers_{eff_scope}", pref_val, None)
            claims.append(TemporalClaim(
                claim_id=c_id,
                subject=eff_sub,
                predicate="prefers",
                object=pref_val,
                valid_from=None,
                valid_to=None,
                recorded_at=recorded_at,
                source_chunk_id=chunk_id,
                is_explicit=True,
                confidence=1.0,
                evidence=m_pref_context.group(0).strip(),
                properties={
                    "session_id": session_id,
                    "text": line_clean,
                    "context_scope": eff_scope
                }
            ))
            continue

        # 6. Uncertainty Patterns (Hypothetical / Contingent claims)
        # e.g. "I might move to London next year" or "I'm thinking of moving to Boston"
        m_uncertain = re.search(
            r'(?:I|User: I)\s+(?:might|may|am\s+thinking\s+of|planning\s+to)\s+(?:move\s+to|live\s+in)\s+([A-Za-z\s]+?)(?:[.,!]|\s*$|\n)',
            line_clean,
            re.IGNORECASE
        )
        if m_uncertain:
            loc = m_uncertain.group(1).strip()
            c_id = generate_claim_id(subject, "lives_in", loc, "uncertain")
            claims.append(TemporalClaim(
                claim_id=c_id,
                subject=subject,
                predicate="lives_in",
                object=loc,
                valid_from=None,
                valid_to=None,
                recorded_at=recorded_at,
                source_chunk_id=chunk_id,
                is_explicit=False,
                confidence=0.5,
                evidence=m_uncertain.group(0).strip(),
                properties={
                    "session_id": session_id,
                    "text": line_clean,
                    "is_uncertain": True,
                    "uncertainty_cue": "might/thinking_of"
                }
            ))
            continue

        # 7. Explicit Negation Patterns
        # e.g. "I do not own a car" or "I don't have a car"
        m_neg_own = re.search(
            r'(?:I|User: I)\s+(?:do\s+not|don\'t)\s+(?:own|have)\s+(?:a\s+|an\s+)?([A-Za-z0-9\s\-]+?)(?:[.,!]|\s*$|\n)',
            line_clean,
            re.IGNORECASE
        )
        if m_neg_own:
            item = m_neg_own.group(1).strip()
            c_id = generate_claim_id(subject, "owns", item, "negated")
            claims.append(TemporalClaim(
                claim_id=c_id,
                subject=subject,
                predicate="owns",
                object=item,
                valid_from=None,
                valid_to=None,
                recorded_at=recorded_at,
                source_chunk_id=chunk_id,
                is_explicit=True,
                confidence=1.0,
                evidence=m_neg_own.group(0).strip(),
                properties={
                    "session_id": session_id,
                    "text": line_clean,
                    "is_negated": True,
                    "polarity": "negative"
                }
            ))
            continue

        # Positive Ownership
        m_pos_own = re.search(
            r'(?:I|User: I)\s+(?:own|bought|purchased)\s+(?:a\s+|an\s+)?([A-Za-z0-9\s\-]+?)(?:[.,!]|\s*$|\n)',
            line_clean,
            re.IGNORECASE
        )
        if m_pos_own:
            item = m_pos_own.group(1).strip()
            c_id = generate_claim_id(subject, "owns", item, "positive")
            claims.append(TemporalClaim(
                claim_id=c_id,
                subject=subject,
                predicate="owns",
                object=item,
                valid_from=None,
                valid_to=None,
                recorded_at=recorded_at,
                source_chunk_id=chunk_id,
                is_explicit=True,
                confidence=1.0,
                evidence=m_pos_own.group(0).strip(),
                properties={
                    "session_id": session_id,
                    "text": line_clean,
                    "is_negated": False,
                    "polarity": "positive"
                }
            ))
            continue

    return claims

