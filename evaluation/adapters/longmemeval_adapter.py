"""
LongMemEval Benchmark Adapter for IVARA.

Provides an isolated, verified adapter to parse, validate, and convert
LongMemEval datasets (e.g. `longmemeval_oracle.json`, `longmemeval_s_cleaned.json`)
into IVARA memory representation while strictly preserving isolation from
production stores (data/memories.json and data/entities.json) and isolating
gold answers from ingestion and answering pipelines.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field


SUPPORTED_QUESTION_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update"
}

PRODUCTION_MEMORY_PATH = Path("data/memories.json")
PRODUCTION_ENTITY_PATH = Path("data/entities.json")


class ProductionStoreIsolationError(Exception):
    """Raised when an operation attempts to read or write to production memory stores."""
    pass


def parse_longmemeval_timestamp(date_str: str | None) -> datetime | None:
    """
    Parse official LongMemEval timestamp strings, e.g.:
      '2023/04/10 (Mon) 23:07' -> datetime(2023, 4, 10, 23, 7)
    Returns None if date_str is empty or cannot be parsed.
    """
    if not date_str or not isinstance(date_str, str):
        return None

    # Strip weekday in parentheses e.g. '(Mon)'
    cleaned = re.sub(r'\s*\([A-Za-z]+\)\s*', ' ', date_str.strip()).strip()

    formats = [
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


@dataclass
class LongMemEvalTurn:
    role: str
    content: str


@dataclass
class LongMemEvalSession:
    session_id: str
    date: str
    turns: list[LongMemEvalTurn]
    original_index: int = 0
    parsed_datetime: datetime | None = None

    def to_formatted_text(self) -> str:
        """Format session turns as conversational transcript with clear speaker roles."""
        lines = []
        for turn in self.turns:
            speaker = "User" if turn.role.lower() == "user" else "Assistant"
            lines.append(f"{speaker}: {turn.content}")
        return "\n".join(lines)


@dataclass
class LongMemEvalInstance:
    question_id: str
    question_type: str
    question: str
    gold_answer: str
    question_date: str
    sessions: list[LongMemEvalSession]
    answer_session_ids: list[str] = field(default_factory=list)
    is_abstention: bool = False
    diagnostic_warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationFailure:
    record_index: int
    question_id: str | None
    error_reason: str
    raw_snippet: dict[str, Any] | None = None


class LongMemEvalAdapter:
    """
    Adapter for reading, validating, and converting LongMemEval benchmark datasets.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        output_dir: str | Path | None = None,
        chunk_target_words: int = 120,
        chunk_overlap_sentences: int = 1,
        sort_sessions_chronologically: bool = True
    ):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation/data/longmemeval/runs")
        self.chunk_target_words = chunk_target_words
        self.chunk_overlap_sentences = chunk_overlap_sentences
        self.sort_sessions_chronologically = sort_sessions_chronologically

        self._assert_safe_paths()

    def _assert_safe_paths(self):
        """Strictly prevent the benchmark from touching production stores."""
        for check_path in [self.dataset_path, self.output_dir]:
            try:
                resolved = check_path.resolve()
                prod_mem = PRODUCTION_MEMORY_PATH.resolve()
                prod_ent = PRODUCTION_ENTITY_PATH.resolve()
                if resolved == prod_mem or resolved == prod_ent:
                    raise ProductionStoreIsolationError(
                        f"Refusing to execute: Benchmark path points to production store: {resolved}"
                    )
                # Extra check against relative string containment
                norm_str = str(resolved).replace("\\", "/").lower()
                if "data/memories.json" in norm_str or "data/entities.json" in norm_str:
                    # Allow if inside evaluation/
                    if "evaluation/" not in norm_str:
                        raise ProductionStoreIsolationError(
                            f"Refusing to execute: Suspicious path pointing near production store: {resolved}"
                        )
            except Exception as e:
                if isinstance(e, ProductionStoreIsolationError):
                    raise
                pass

    def load_and_validate(
        self,
        max_examples: int | None = None,
        target_question_ids: list[str] | set[str] | None = None
    ) -> tuple[list[LongMemEvalInstance], list[ValidationFailure]]:
        """
        Load and validate instances from the JSON dataset file.

        Args:
            max_examples: Maximum number of valid instances to load.
            target_question_ids: Optional specific subset of question_ids to load.

        Returns:
            (valid_instances, failed_records)
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"LongMemEval dataset file not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"Dataset root must be a JSON array, got {type(raw_data).__name__}")

        valid_instances: list[LongMemEvalInstance] = []
        failed_records: list[ValidationFailure] = []
        seen_question_ids: set[str] = set()
        target_set = set(target_question_ids) if target_question_ids is not None else None

        for idx, item in enumerate(raw_data):
            if max_examples is not None and len(valid_instances) >= max_examples:
                break

            if not isinstance(item, dict):
                failed_records.append(ValidationFailure(
                    record_index=idx,
                    question_id=None,
                    error_reason=f"Record is not a JSON object: {type(item).__name__}"
                ))
                continue

            qid = item.get("question_id")
            if target_set is not None and qid not in target_set:
                continue

            if not qid or not isinstance(qid, str):
                failed_records.append(ValidationFailure(
                    record_index=idx,
                    question_id=None,
                    error_reason="Missing or non-string 'question_id'",
                    raw_snippet={k: str(v)[:50] for k, v in item.items()}
                ))
                continue

            # Duplicate question ID check
            if qid in seen_question_ids:
                failed_records.append(ValidationFailure(
                    record_index=idx,
                    question_id=qid,
                    error_reason=f"Duplicate question_id detected: {qid}"
                ))
                continue
            seen_question_ids.add(qid)

            # Question type check
            qtype = item.get("question_type")
            if qtype not in SUPPORTED_QUESTION_TYPES:
                failed_records.append(ValidationFailure(
                    record_index=idx,
                    question_id=qid,
                    error_reason=f"Unsupported question_type: '{qtype}'. Supported: {sorted(SUPPORTED_QUESTION_TYPES)}"
                ))
                continue

            # Question and answer checks
            question = item.get("question")
            if not question or not isinstance(question, str) or not question.strip():
                failed_records.append(ValidationFailure(
                    record_index=idx,
                    question_id=qid,
                    error_reason="Missing or empty 'question' string"
                ))
                continue

            answer = item.get("answer")
            if answer is None:
                failed_records.append(ValidationFailure(
                    record_index=idx,
                    question_id=qid,
                    error_reason="Missing 'answer' field"
                ))
                continue

            question_date = item.get("question_date", "")

            # Haystack alignment check
            dates = item.get("haystack_dates")
            sess_ids = item.get("haystack_session_ids")
            sessions_raw = item.get("haystack_sessions")

            if not isinstance(dates, list) or not isinstance(sess_ids, list) or not isinstance(sessions_raw, list):
                failed_records.append(ValidationFailure(
                    record_index=idx,
                    question_id=qid,
                    error_reason="haystack_dates, haystack_session_ids, and haystack_sessions must all be lists"
                ))
                continue

            if not (len(dates) == len(sess_ids) == len(sessions_raw)):
                failed_records.append(ValidationFailure(
                    record_index=idx,
                    question_id=qid,
                    error_reason=(
                        f"Mismatched haystack array lengths: dates={len(dates)}, "
                        f"session_ids={len(sess_ids)}, sessions={len(sessions_raw)}"
                    )
                ))
                continue

            # Parse sessions and turns
            parsed_sessions: list[LongMemEvalSession] = []
            session_parse_error = None
            instance_warnings: list[str] = []
            seen_session_ids_in_instance = set()

            for s_idx, (s_id, s_date, s_turns_raw) in enumerate(zip(sess_ids, dates, sessions_raw)):
                s_id_str = str(s_id)
                if s_id_str in seen_session_ids_in_instance:
                    instance_warnings.append(f"Duplicate session ID within question: {s_id_str}")
                seen_session_ids_in_instance.add(s_id_str)

                if not isinstance(s_turns_raw, list):
                    session_parse_error = f"Session {s_idx} ({s_id}) turns must be a list"
                    break

                parsed_turns: list[LongMemEvalTurn] = []
                for t_idx, turn_dict in enumerate(s_turns_raw):
                    if not isinstance(turn_dict, dict):
                        session_parse_error = f"Session {s_id} turn {t_idx} is not a dict"
                        break
                    role = turn_dict.get("role", "unknown")
                    content = turn_dict.get("content", "")
                    # Note: we deliberately ignore 'has_answer' here to isolate ground-truth annotations!
                    parsed_turns.append(LongMemEvalTurn(role=str(role), content=str(content)))

                if session_parse_error:
                    break

                dt = parse_longmemeval_timestamp(str(s_date) if s_date else None)
                if dt is None and s_date:
                    instance_warnings.append(f"Unparseable session date string: '{s_date}' in session {s_id}")

                parsed_sessions.append(LongMemEvalSession(
                    session_id=s_id_str,
                    date=str(s_date) if s_date is not None else "",
                    turns=parsed_turns,
                    original_index=s_idx,
                    parsed_datetime=dt
                ))

            if session_parse_error:
                failed_records.append(ValidationFailure(
                    record_index=idx,
                    question_id=qid,
                    error_reason=session_parse_error
                ))
                continue

            # Sort sessions chronologically if requested (resolves official Oracle unsorted session caveat)
            if self.sort_sessions_chronologically and parsed_sessions:
                # Stable sort by (parsed_datetime or max, original_index)
                parsed_sessions.sort(key=lambda s: (s.parsed_datetime or datetime.max, s.original_index))

            is_abs = "_abs" in qid or "abstention" in qtype.lower()
            ans_session_ids = [str(x) for x in item.get("answer_session_ids", [])]

            valid_instances.append(LongMemEvalInstance(
                question_id=qid,
                question_type=qtype,
                question=question.strip(),
                gold_answer=str(answer).strip(),
                question_date=str(question_date),
                sessions=parsed_sessions,
                answer_session_ids=ans_session_ids,
                is_abstention=is_abs,
                diagnostic_warnings=instance_warnings
            ))

        if isinstance(target_question_ids, list):
            qid_order = {q: i for i, q in enumerate(target_question_ids)}
            valid_instances.sort(key=lambda x: qid_order.get(x.question_id, 999999))

        return valid_instances, failed_records

    def convert_instance_to_memories(
        self,
        instance: LongMemEvalInstance
    ) -> list[dict[str, Any]]:
        """
        Convert conversational sessions from an instance into IVARA memory records.

        CRITICAL SECURITY / EVALUATION INTEGRITY RULE:
        This conversion completely strips the gold answer, answer_session_ids,
        and has_answer flags, ensuring the retrieval index and answering pipeline
        receive zero test-label leakage.
        """
        memories = []
        for s_idx, session in enumerate(instance.sessions):
            session_text = session.to_formatted_text()
            memo = {
                "id": f"sess_{session.session_id}",
                "text": session_text,
                "timestamp": session.date,
                "source": f"session_{session.session_id}",
                "type": "conversation",
                "importance": 1.0,
                "episode": f"Session {session.session_id}",
                "session_id": session.session_id,
                "turn_count": len(session.turns),
                "original_order": session.original_index,
                "chronological_order": s_idx,
                "entities": [],
                "relationships": []
            }
            memories.append(memo)

        return memories

    def get_reference_data(
        self,
        instances: list[LongMemEvalInstance]
    ) -> dict[str, dict[str, Any]]:
        """
        Extract gold reference answers for post-hoc scoring ONLY.
        """
        references = {}
        for inst in instances:
            references[inst.question_id] = {
                "question_id": inst.question_id,
                "question_type": inst.question_type,
                "question": inst.question,
                "answer": inst.gold_answer,
                "question_date": inst.question_date,
                "answer_session_ids": inst.answer_session_ids,
                "is_abstention": inst.is_abstention
            }
        return references
