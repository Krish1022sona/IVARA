"""
Deterministic Calendar Arithmetic Helper for IVARA.
Handles date differences, relative date anchoring, month/year boundaries, leap years,
and inclusive vs. exclusive day counting.
"""

import calendar
from datetime import datetime, date
import re
from typing import Any, Optional


MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12
}


def parse_date_string(date_str: str | None) -> Optional[date]:
    """
    Parse an explicit or semi-structured date string into a standard datetime.date.
    Supports ISO formats (YYYY-MM-DD, YYYY/MM/DD), timestamps with day names,
    and Month Day Year expressions (e.g. 'April 6, 2023', 'March 15th, 2023').
    """
    if not date_str or not isinstance(date_str, str):
        return None

    clean = date_str.strip()

    # 1. ISO-8601 variations: 2023-04-10 or 2023/04/10 or 2023/04/10 (Mon) 10:28
    m_iso = re.search(r'\b(19\d\d|20\d\d)[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b', clean)
    if m_iso:
        y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        try:
            return date(y, m, d)
        except ValueError:
            return None

    # 2. Month Day, Year: e.g. "April 6, 2023", "April 6th, 2023", "6 April 2023"
    m_text = re.search(
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december|'
        r'jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(19\d\d|20\d\d)\b',
        clean,
        re.IGNORECASE
    )
    if m_text:
        mon_name = m_text.group(1).lower()
        mon = MONTH_NAMES.get(mon_name)
        d = int(m_text.group(2))
        y = int(m_text.group(3))
        if mon:
            try:
                return date(y, mon, d)
            except ValueError:
                return None

    m_day_first = re.search(
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:,)?\s*(19\d\d|20\d\d)\b',
        clean,
        re.IGNORECASE
    )
    if m_day_first:
        d = int(m_day_first.group(1))
        mon_name = m_day_first.group(2).lower()
        mon = MONTH_NAMES.get(mon_name)
        y = int(m_day_first.group(3))
        if mon:
            try:
                return date(y, mon, d)
            except ValueError:
                return None

    return None


def calculate_date_delta(d1: date, d2: date) -> dict[str, Any]:
    """
    Calculate deterministic differences between two dates handling:
    - Days difference (absolute)
    - Inclusive days difference (days + 1)
    - Full weeks and remaining days
    - Calendar months difference
    - Calendar years difference
    - Leap year presence in interval
    """
    earlier = min(d1, d2)
    later = max(d1, d2)

    days_diff = (later - earlier).days
    inclusive_days = days_diff + 1 if days_diff > 0 else 1

    weeks_diff = round(days_diff / 7.0)
    exact_weeks = days_diff // 7
    rem_days = days_diff % 7

    # Calendar months difference
    months_diff = (later.year - earlier.year) * 12 + (later.month - earlier.month)
    # Adjust if later day is earlier in month than earlier day
    if later.day < earlier.day:
        months_approx = max(0, months_diff - 1)
    else:
        months_approx = months_diff

    # Leap year in interval check
    has_leap_day = False
    for y in range(earlier.year, later.year + 1):
        if calendar.isleap(y):
            leap_date = date(y, 2, 29)
            if earlier <= leap_date <= later:
                has_leap_day = True
                break

    return {
        "start_date": earlier.isoformat(),
        "end_date": later.isoformat(),
        "days": days_diff,
        "inclusive_days": inclusive_days,
        "weeks": weeks_diff,
        "exact_weeks": exact_weeks,
        "remaining_days": rem_days,
        "months": months_diff,
        "has_leap_day": has_leap_day
    }


def days_between(d1: date, d2: date, inclusive: bool = False) -> int:
    """Return integer days between two dates, optionally inclusive."""
    delta = calculate_date_delta(d1, d2)
    return delta["inclusive_days"] if inclusive else delta["days"]


def weeks_between(d1: date, d2: date) -> float:
    """Return weeks between two dates."""
    delta = calculate_date_delta(d1, d2)
    return float(delta["exact_weeks"]) if delta["remaining_days"] == 0 else round(delta["days"] / 7.0, 1)


def months_between(d1: date, d2: date) -> int:
    """Return calendar months between two dates."""
    delta = calculate_date_delta(d1, d2)
    return delta["months"]


def add_days(d: date, n: int) -> date:
    """Return new date offset by n days."""
    from datetime import timedelta
    return d + timedelta(days=n)


def find_dates_in_text(text: str) -> list[tuple[date, str]]:
    """
    Extract all verifiable dates with their surrounding context span from text.
    Returns list of (date_obj, matched_span).
    """
    results = []
    if not text:
        return results

    # ISO matches: YYYY-MM-DD or YYYY/MM/DD
    for m in re.finditer(r'\b(19\d\d|20\d\d)[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b', text):
        d_str = m.group(0)
        parsed = parse_date_string(d_str)
        if parsed:
            # Context window around match
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            results.append((parsed, text[start:end].strip()))

    # Textual month matches
    pattern = (
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december|'
        r'jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(19\d\d|20\d\d)\b'
    )
    for m in re.finditer(pattern, text, re.IGNORECASE):
        d_str = m.group(0)
        parsed = parse_date_string(d_str)
        if parsed:
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            results.append((parsed, text[start:end].strip()))

    return results


class DeterministicCalendarEngine:
    """
    Evaluates calendar-based personal memory queries and generates verifiable,
    grounded mathematical annotations or direct answers.
    """

    RE_DAYS_BETWEEN = re.compile(
        r'\bhow many days (passed between|had passed between|were between)\b|\bdays passed between\b',
        re.I
    )
    RE_DAYS_AGO = re.compile(
        r'\bhow many (days|weeks|months|years) ago\b|\bhow many (days|weeks|months|years) have passed since\b|\bhow many (days|weeks|months|years) had passed since\b',
        re.I
    )

    def evaluate_query(
        self,
        query: str,
        context_text: str = "",
        reference_date_str: str | None = None,
        question_date: str | None = None,
        evidence_dates: list[str] | None = None
    ) -> Optional[dict[str, Any]]:
        """
        Evaluate if query is a deterministic calendar query.
        If dates can be unambiguously matched to the events queried,
        returns structured calculation facts.
        """
        q_lower = query.lower().strip()

        effective_ref_str = question_date if question_date is not None else reference_date_str
        ref_date = parse_date_string(effective_ref_str) if effective_ref_str else None

        combined_text = context_text or ""
        if evidence_dates:
            combined_text = f"{combined_text}\n" + "\n".join(str(ed) for ed in evidence_dates)
        # Also scan query itself for explicit dates (e.g. "between 2023-04-10 and 2023-04-14")
        combined_text = f"{combined_text}\n{query}"

        # 1. Check for "How many days/weeks/months ago" or "since" relative to reference date
        m_ago = self.RE_DAYS_AGO.search(q_lower)
        if m_ago:
            unit = m_ago.group(1) or m_ago.group(2) or m_ago.group(3)
            unit = unit.lower()

            if not ref_date:
                return {
                    "is_calendar_query": True,
                    "unit": unit,
                    "status": "unresolved_reference_date",
                    "note": "Reference date not provided; cannot calculate exact elapsed time."
                }

            # Find dates in retrieved context
            found_dates = find_dates_in_text(combined_text)
            if not found_dates:
                return None

            # Filter dates prior to or equal to ref_date
            prior_dates = [d for d, ctx in found_dates if d <= ref_date]
            if not prior_dates:
                return None

            # Select the most relevant prior event date (or unique date)
            # If all found dates point to the same day, unambiguous!
            unique_dates = list(set(prior_dates))
            if len(unique_dates) == 1:
                event_date = unique_dates[0]
                delta = calculate_date_delta(event_date, ref_date)
                
                # Format deterministic answer
                if unit == "days":
                    ans_text = f"{delta['days']} days"
                elif unit == "weeks":
                    ans_text = f"{delta['weeks']} weeks"
                elif unit == "months":
                    ans_text = f"{delta['months']} months"
                else:
                    ans_text = f"{delta['days']} days"

                note_text = f"Exact calendar calculation: Event date {event_date.isoformat()} to Reference date {ref_date.isoformat()} is {delta['days']} days ({delta['weeks']} weeks, {delta['months']} months)."

                return {
                    "is_calendar_query": True,
                    "unit": unit,
                    "status": "computed",
                    "event_date": event_date.isoformat(),
                    "reference_date": ref_date.isoformat(),
                    "delta": delta,
                    "suggested_answer": ans_text,
                    "grounded_note": note_text,
                    "computed_fact": note_text
                }

        # 2. Check for "How many days passed between Event A and Event B"
        m_between = self.RE_DAYS_BETWEEN.search(q_lower)
        if m_between or ("between" in q_lower and any(w in q_lower for w in ["how many days", "how many weeks", "how many months"])):
            found_dates = find_dates_in_text(combined_text)
            unique_dates = sorted(list(set(d for d, ctx in found_dates)))

            if len(unique_dates) == 2:
                d1, d2 = unique_dates[0], unique_dates[1]
                delta = calculate_date_delta(d1, d2)
                ans_text = f"{delta['days']} days"

                note_text = f"Exact calendar calculation between {d1.isoformat()} and {d2.isoformat()}: {delta['days']} days ({delta['inclusive_days']} days inclusive)."

                return {
                    "is_calendar_query": True,
                    "unit": "days",
                    "status": "computed",
                    "event_date": d1.isoformat(),
                    "reference_date": d2.isoformat(),
                    "delta": delta,
                    "suggested_answer": ans_text,
                    "grounded_note": note_text,
                    "computed_fact": note_text
                }

        return None
