"""
Temporal Event Extraction Engine for IVARA.

Extracts EventNodes from episodic text chunks with explicit provenance,
preserves temporal uncertainty via TemporalPrecision, and strictly decouples
event occurrence time from conversation ingestion/recording time.
"""

import calendar
from datetime import date, datetime, timedelta
import re
from typing import Any

from .event_graph import EventNode, TemporalPrecision, generate_event_id
from .calendar_arithmetic import parse_date_string, find_dates_in_text, MONTH_NAMES
from ..ontology.vocabulary import load_vocabulary, normalize_name


# Action and transition patterns indicating personal episodic events
EVENT_ACTION_PATTERNS = [
    # Attendance & participation
    (r'\b(?:attended|went to|participated in|joined|registered for)\s+([A-Za-z0-9#\s\-_]+?)(?:\s+(?:on|in|during|at)\s+|\.|\,|$)', "participated in"),
    # Travel & relocation
    (r'\b(?:visited|travelled to|traveled to|flew to|took a trip to)\s+([A-Za-z0-9\s\-_]+?)(?:\s+(?:on|in|during|at)\s+|\.|\,|$)', "visited"),
    (r'\b(?:moved to|relocated to|settled in|living in|lived in)\s+([A-Za-z0-9\s\-_]+?)(?:\s+(?:on|in|during|at)\s+|\.|\,|$)', "lived in"),
    # Employment & education
    (r'\b(?:started working at|joined|hired by|employed at|worked at)\s+([A-Za-z0-9\s\-_]+?)(?:\s+(?:on|in|during|at)\s+|\.|\,|$)', "worked at"),
    (r'\b(?:graduated from|finished school at|earned degree from)\s+([A-Za-z0-9\s\-_]+?)(?:\s+(?:on|in|during|at)\s+|\.|\,|$)', "graduated from"),
    (r'\b(?:left|resigned from|quit)\s+([A-Za-z0-9\s\-_]+?)(?:\s+(?:on|in|during|at)\s+|\.|\,|$)', "left"),
    # Subscription & services
    (r'\b(?:cancelled|canceled|terminated|stopped)\s+(?:my\s+)?([A-Za-z0-9\s\-_]+?)(?:\s+(?:subscription|service|membership|account))?(?:\s+(?:on|in|during|at)\s+|\.|\,|$)', "cancelled"),
    (r'\b(?:subscribed to|signed up for)\s+([A-Za-z0-9\s\-_]+?)(?:\s+(?:on|in|during|at)\s+|\.|\,|$)', "subscribed to"),
    # Purchases & accomplishments
    (r'\b(?:bought|purchased|acquired)\s+(?:a|an|the)?\s*([A-Za-z0-9\s\-_]+?)(?:\s+(?:on|in|during|at)\s+|\.|\,|$)', "bought"),
    (r'\b(?:ran|completed|finished)\s+(?:the\s+)?([A-Za-z0-9#\s\-_]+?)(?:\s+(?:on|in|during|at)\s+|\.|\,|$)', "completed"),
    (r'\b(?:read|finished reading)\s+(?:the\s+book\s+)?([A-Za-z0-9\s\-_]+?)(?:\s+(?:on|in|during|at)\s+|\.|\,|$)', "read")
]

MONTH_REGEX = re.compile(
    r'\b(january|february|march|april|may|june|july|august|september|october|november|december|'
    r'jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b(?:\s+(\d{4}))?',
    re.IGNORECASE
)

YEAR_REGEX = re.compile(r'\b(19\d\d|20\d\d)\b')


class TemporalEventExtractor:
    """
    Extracts structured EventNodes from episodic text.
    Decouples event time from recording time. Preserves uncertainty.
    """

    def __init__(self):
        self.vocabulary = load_vocabulary()

    def extract_events_from_chunk(
        self,
        text: str,
        session_id: str,
        chunk_id: str | None = None,
        recorded_at: str | None = None
    ) -> list[EventNode]:
        """
        Extract all verifiable event nodes from a text chunk.
        """
        events: list[EventNode] = []
        if not text:
            return events

        # Parse recorded_at (ingestion time)
        rec_date = parse_date_string(recorded_at) if recorded_at else None

        sentences = re.split(r'(?<=[.!?\n])\s+', text)
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if len(sentence_clean) < 10:
                continue

            extracted = self._extract_from_sentence(
                sentence=sentence_clean,
                session_id=session_id,
                chunk_id=chunk_id,
                recorded_at=recorded_at,
                rec_date=rec_date
            )
            events.extend(extracted)

        return events

    def _extract_from_sentence(
        self,
        sentence: str,
        session_id: str,
        chunk_id: str | None,
        recorded_at: str | None,
        rec_date: date | None
    ) -> list[EventNode]:
        """
        Extract events from a single sentence, resolving action, entity, and temporal bounds.
        """
        sentence_lower = sentence.lower()
        extracted: list[EventNode] = []

        # 1. Resolve temporal anchor in sentence
        date_info = self._resolve_sentence_dates(sentence, rec_date)

        # 2. Check for action patterns
        matched_actions: list[tuple[str, str]] = [] # (action_type, target)
        for pat, action_type in EVENT_ACTION_PATTERNS:
            m = re.search(pat, sentence, re.IGNORECASE)
            if m:
                target = m.group(1).strip()
                # Clean up target
                target = re.sub(r'^(?:a|an|the|my|our)\s+', '', target, flags=re.IGNORECASE).strip()
                if 2 <= len(target) <= 60 and not target.lower().startswith(("yesterday", "today", "tomorrow")):
                    matched_actions.append((action_type, target))

        if not matched_actions and date_info["valid_from"] is not None:
            # Sentence has an explicit date and personal narrative: treat as general personal event
            if any(term in sentence_lower for term in ("i ", "my ", "we ", "user:")):
                matched_actions.append(("personal_event", sentence[:50]))

        # 3. Create EventNodes for matched actions
        for action_type, target in matched_actions:
            desc = f"{action_type.capitalize()} {target}" if action_type != "personal_event" else sentence[:60].strip()

            # Entities linked to this event
            entity_ids = self._extract_entities(target, sentence)

            event_id = generate_event_id(session_id, desc, date_info["valid_from"])

            evt = EventNode(
                event_id=event_id,
                description=desc,
                entity_ids=entity_ids,
                session_id=session_id,
                turn_id=chunk_id,
                event_time_text=date_info["event_time_text"],
                valid_from=date_info["valid_from"],
                valid_to=date_info["valid_to"],
                recorded_at=recorded_at,
                precision=date_info["precision"],
                confidence=date_info["confidence"],
                provenance=sentence,
                properties={
                    "action_type": action_type,
                    "target": target
                }
            )
            extracted.append(evt)

        return extracted

    def _resolve_sentence_dates(self, sentence: str, rec_date: date | None) -> dict[str, Any]:
        """
        Deterministically resolve temporal bounds and precision from sentence.
        Strictly decouples event time from recording time.
        """
        # Case A: Explicit calendar date (e.g. "April 6, 2023", "2023-04-06")
        found = find_dates_in_text(sentence)
        if found:
            d_obj, span = found[0]
            iso = d_obj.isoformat()
            return {
                "valid_from": iso,
                "valid_to": iso,
                "event_time_text": span,
                "precision": TemporalPrecision.EXACT_DAY,
                "confidence": 1.0
            }

        # Case B: Relative date expressions anchored to rec_date
        if rec_date:
            # "yesterday"
            if re.search(r'\byesterday\b', sentence, re.IGNORECASE):
                d = rec_date - timedelta(days=1)
                iso = d.isoformat()
                return {
                    "valid_from": iso,
                    "valid_to": iso,
                    "event_time_text": "yesterday",
                    "precision": TemporalPrecision.EXACT_DAY,
                    "confidence": 0.95
                }
            # "today" / "this morning" / "just now"
            if re.search(r'\b(?:today|this morning|this afternoon|this evening|just now)\b', sentence, re.IGNORECASE):
                iso = rec_date.isoformat()
                return {
                    "valid_from": iso,
                    "valid_to": iso,
                    "event_time_text": "today",
                    "precision": TemporalPrecision.EXACT_DAY,
                    "confidence": 0.95
                }
            # "X days ago"
            m_days = re.search(r'\b(\d+)\s+days?\s+ago\b', sentence, re.IGNORECASE)
            if m_days:
                n = int(m_days.group(1))
                d = rec_date - timedelta(days=n)
                iso = d.isoformat()
                return {
                    "valid_from": iso,
                    "valid_to": iso,
                    "event_time_text": m_days.group(0),
                    "precision": TemporalPrecision.EXACT_DAY,
                    "confidence": 0.95
                }
            # "X weeks ago" / "last week"
            m_weeks = re.search(r'\b(?:(\d+)\s+weeks?\s+ago|last\s+week)\b', sentence, re.IGNORECASE)
            if m_weeks:
                n = int(m_weeks.group(1)) if m_weeks.group(1) else 1
                d_end = rec_date - timedelta(days=7 * (n - 1) + 1)
                d_start = rec_date - timedelta(days=7 * n)
                return {
                    "valid_from": d_start.isoformat(),
                    "valid_to": d_end.isoformat(),
                    "event_time_text": m_weeks.group(0),
                    "precision": TemporalPrecision.RELATIVE,
                    "confidence": 0.9
                }
            # "last month"
            if re.search(r'\blast\s+month\b', sentence, re.IGNORECASE):
                # Compute previous month
                prev_m = rec_date.month - 1 if rec_date.month > 1 else 12
                prev_y = rec_date.year if rec_date.month > 1 else rec_date.year - 1
                last_day = calendar.monthrange(prev_y, prev_m)[1]
                return {
                    "valid_from": f"{prev_y:04d}-{prev_m:02d}-01",
                    "valid_to": f"{prev_y:04d}-{prev_m:02d}-{last_day:02d}",
                    "event_time_text": "last month",
                    "precision": TemporalPrecision.MONTH,
                    "confidence": 0.9
                }

        # Case C: Month + Year (e.g. "in July 2023", "July 2023")
        m_month = MONTH_REGEX.search(sentence)
        if m_month:
            mon_str = m_month.group(1).lower()
            mon_num = MONTH_NAMES.get(mon_str)
            yr_str = m_month.group(2)
            if not yr_str and rec_date:
                # If no year stated, do not hallucinate year unless clearly within recording year
                yr_num = rec_date.year
            elif yr_str:
                yr_num = int(yr_str)
            else:
                yr_num = None

            if mon_num and yr_num:
                last_day = calendar.monthrange(yr_num, mon_num)[1]
                return {
                    "valid_from": f"{yr_num:04d}-{mon_num:02d}-01",
                    "valid_to": f"{yr_num:04d}-{mon_num:02d}-{last_day:02d}",
                    "event_time_text": m_month.group(0),
                    "precision": TemporalPrecision.MONTH,
                    "confidence": 0.9
                }

        # Case D: Year only (e.g. "in 2021", "since 2019")
        m_year = YEAR_REGEX.search(sentence)
        if m_year:
            yr = int(m_year.group(1))
            return {
                "valid_from": f"{yr:04d}-01-01",
                "valid_to": f"{yr:04d}-12-31",
                "event_time_text": str(yr),
                "precision": TemporalPrecision.YEAR,
                "confidence": 0.85
            }

        # Case E: Unknown / unanchored time (do not invent dates!)
        return {
            "valid_from": None,
            "valid_to": None,
            "event_time_text": None,
            "precision": TemporalPrecision.UNKNOWN,
            "confidence": 0.5
        }

    def _extract_entities(self, target: str, sentence: str) -> list[str]:
        """
        Extract and normalize entity references from event target and sentence.
        Preserves unresolved entities as candidate mentions.
        """
        entities = set()

        # Clean target into entity mention
        target_clean = re.sub(r'[^a-zA-Z0-9_\-\s]', '', target).strip()
        if target_clean and len(target_clean) >= 2:
            entities.add(f"ent_{normalize_name(target_clean)}")

        # Extract proper nouns or hashtags from sentence
        words = sentence.split()
        for w in words:
            if w.startswith("#") and len(w) > 2:
                entities.add(f"ent_{normalize_name(w)}")
            elif len(w) > 3 and w[0].isupper() and w[1:].islower():
                entities.add(f"ent_{normalize_name(w)}")

        return list(entities)

