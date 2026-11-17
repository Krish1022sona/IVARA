"""Memory Formation Policy and Classification Engine for IVARA.

Governs long-term memory formation to ensure IVARA retains durable, useful,
and accurate knowledge about the user, while filtering out:
- Questions and recall requests
- General educational queries and concept explanations
- Greetings, social pleasantries, and conversational fillers
- Temporary, trivial, or redundant statements

Supports:
- Fine-grained categorical classification (PERSONAL_FACT, PREFERENCE, GOAL, etc.)
- Clause-level extraction for mixed messages (personal statements embedded with questions)
- Distinguishing goals/plans from completed experiences
- Detecting corrections and updates
- Semantic deduplication against existing memory stores
- Conservative fallback when classification is ambiguous
"""

from enum import Enum
import re
from typing import Any, Callable, Optional
from pydantic import BaseModel, Field


class MemoryCategory(str, Enum):
    PERSONAL_FACT = "personal_fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    EXPERIENCE = "experience"
    DECISION = "decision"
    QUESTION = "question"
    INSTRUCTION = "instruction"
    GENERAL_KNOWLEDGE = "general_knowledge"
    CASUAL_CONVERSATION = "casual_conversation"


class MemoryFormationDecision(BaseModel):
    """Structured decision detailing whether and what to form into a long-term memory."""
    should_extract: bool = Field(description="Whether a long-term memory should be formed")
    category: MemoryCategory = Field(description="The primary classification category")
    extracted_text: Optional[str] = Field(default=None, description="The distilled factual proposition to store")
    memory_type: str = Field(default="fact", description="Type tag: fact, preference, goal, experience, decision, update")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in this classification")
    is_update: bool = Field(default=False, description="Whether this statement updates or supersedes earlier knowledge")
    is_duplicate: bool = Field(default=False, description="Whether this is redundant with an already stored memory")
    existing_memory_id: Optional[str] = Field(default=None, description="ID of existing duplicate/superseded memory")
    reason: str = Field(default="", description="Explanation of why this decision was made")


# Exact and regex patterns for casual conversational utterances
CASUAL_EXACT = {
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening",
    "how are you", "how are you doing", "what's up", "whats up", "nice to meet you",
    "thanks", "thank you", "thx", "many thanks", "ok", "okay", "k", "sure", "cool",
    "great", "awesome", "perfect", "got it", "understood", "sounds good", "alright",
    "bye", "goodbye", "see you", "cya", "have a nice day", "good night",
    "yes", "no", "yep", "nope", "yeah", "nah", "please", "hmm", "haha", "lol"
}

CASUAL_PATTERNS = [
    r"^(hi|hello|hey|greetings|dear)\s+(ivara|assistant|bot|ai)?[!.]*$",
    r"^how('?s| is) it going[!.]*$",
    r"^how are you( doing)?( today| this morning| this evening| right now)?[?!.]*$",
    r"^(thank you|thanks)( very much| so much| a lot)?[!.]*$",
    r"^(you('?re| are)|that('?s| is)) (crazy|cool|awesome|wild|great|amazing|helpful)[!.]*$",
    r"^have a (good|great|nice) (day|weekend|evening|night)[!.]*$"
]

# Educational / General knowledge patterns (requests for concepts, definitions, tutorials)
GENERAL_KNOWLEDGE_PATTERNS = [
    r"^(what is|what are|define|explain|describe)\s+([a-zA-Z0-9_\-\s]+)[?.]*$",
    r"^how does\s+([a-zA-Z0-9_\-\s]+)\s+work[?.]*$",
    r"^(tell me about|give me an overview of)\s+([a-zA-Z0-9_\-\s]+)[?.]*$",
    r"^(can you explain|could you explain|please explain)\s+([a-zA-Z0-9_\-\s]+)[?.]*$",
    r"^(what does|what do)\s+([a-zA-Z0-9_\-\s]+)\s+(mean|stand for)[?.]*$",
    r"^why (is|are|do|does)\s+([a-zA-Z0-9_\-\s]+)[?.]*$"
]

# Instructions / System commands
INSTRUCTION_PATTERNS = [
    r"^(summarize|format|reformat|translate|convert|rewrite|refactor|debug|list)\b",
    r"^(write a|generate a|create a|give me a list of)\b",
    r"^(show me|help me with|help me write)\b"
]

# Pure memory recall questions asking IVARA to check its memory
MEMORY_RECALL_PATTERNS = [
    r"^(when did i|where did i|what did i|who did i|why did i)\b",
    r"^(do you remember|do you know|what do you know about me|what more do you know)\b",
    r"^(tell me what you know|what are my|list my|what is my|who is my|where do i)\b",
    r"^(did i (ever )?(mention|say|tell|learn|start|work|go|live|build))\b"
]

# First-person personal indicators (requiring word boundaries)
FIRST_PERSON_INDICATORS = [
    r"\bi\b", r"\bi'm\b", r"\bim\b", r"\bi've\b", r"\bive\b", r"\bi'll\b", r"\bi'd\b",
    r"\bmy\b", r"\bmyself\b", r"\bwe\b", r"\bour\b", r"\bourselves\b"
]

# Goal / Plan indicators
GOAL_PATTERNS = [
    r"\b(plan(ning)? to|aim(ing)? to|hope(ing)? to|intend(ing)? to|want to|going to|goal is to|working towards)\b",
    r"\b(will start|will learn|will build|will move|will try|will read)\b"
]

# Preference indicators
PREFERENCE_PATTERNS = [
    r"\b(prefer|preference is|like\b|love\b|favorite|favourite|dislike|hate|enjoy\b|fond of)\b",
    r"\b(rather than|instead of|my choice is)\b"
]

# Decision indicators
DECISION_PATTERNS = [
    r"\b(decided to|chose to|chosen to|opted for|switched to|switching to|picked|chose\b)\b",
    r"\b(settled on|made up my mind to)\b"
]

# Update / Correction indicators
UPDATE_PATTERNS = [
    r"\b(no longer|not anymore|don't anymore|do not anymore|stopped using|stopped doing)\b",
    r"\b(moved from|switched from|instead of .+ now|actually,? i|correction:?|update:?)\b",
    r"\b(changed my mind|no longer work|no longer live|left my job|quit my)\b"
]

# Completed experience / achievement indicators
EXPERIENCE_PATTERNS = [
    r"\b(built|created|made|developed|implemented|designed|shipped|published)\b",
    r"\b(graduated|attended|completed|participated|won|finished|visited)\b",
    r"\b(worked at|worked on|interned at|studied at|researched)\b"
]


def is_casual_or_filler(text: str) -> bool:
    """Check if the text is purely casual conversation, greeting, filler, or pleasantry."""
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    if not cleaned or len(cleaned) < 2:
        return True
    if cleaned in CASUAL_EXACT:
        return True
    for pat in CASUAL_PATTERNS:
        if re.search(pat, text.strip(), re.IGNORECASE):
            return True
    return False


def is_instruction(text: str) -> bool:
    """Check if the text is a direct imperative task instruction or command."""
    trimmed = text.strip()
    for pat in INSTRUCTION_PATTERNS:
        if re.search(pat, trimmed, re.IGNORECASE):
            if not any(re.search(fp, trimmed.split()[0], re.IGNORECASE) for fp in FIRST_PERSON_INDICATORS if len(trimmed.split()) > 0):
                return True
    return False


def is_general_knowledge_query(text: str) -> bool:
    """Check if the text is a request for general world/technical explanations."""
    trimmed = text.strip()
    # If it refers to the user's personal context, it's NOT general knowledge
    if any(re.search(fp, trimmed, re.IGNORECASE) for fp in [r"\bmy\b", r"\bmyself\b", r"\babout me\b", r"\babout my\b", r"\bi\b", r"\bour\b"]):
        return False
    if re.search(r"\bme\b", trimmed, re.IGNORECASE):
        # Allow indirect object "tell me about <topic>" or "explain to me <topic>"
        if not re.search(r"^(tell me about|explain to me|give me an overview of)\b", trimmed, re.IGNORECASE):
            return False
    for pat in GENERAL_KNOWLEDGE_PATTERNS:
        if re.search(pat, trimmed, re.IGNORECASE):
            return True
    return False


def is_pure_question_or_recall(text: str) -> bool:
    """Check if the text is purely an inquiry (general or personal memory recall)."""
    trimmed = text.strip()
    # Check memory recall patterns first
    for pat in MEMORY_RECALL_PATTERNS:
        if re.search(pat, trimmed, re.IGNORECASE):
            return True

    # If it ends with ? and starts with an interrogative word without personal declarative clause
    if trimmed.endswith("?"):
        q_starters = (
            "what", "when", "where", "why", "who", "which", "how",
            "is ", "are ", "was ", "were ", "can ", "could ", "did ", "do ", "does ", "have ", "has ", "will ", "would "
        )
        lower = trimmed.lower()
        if any(lower.startswith(w) for w in q_starters):
            clauses = split_clauses(trimmed)
            declarative_clauses = [c for c in clauses if has_personal_declarative_content(c)]
            if not declarative_clauses:
                return True
    return False


def has_first_person(text: str) -> bool:
    """Check whether the text contains first-person pronouns with word boundaries."""
    return any(re.search(p, text, re.IGNORECASE) for p in FIRST_PERSON_INDICATORS)


def split_clauses(text: str) -> list[str]:
    """Split a compound utterance into constituent sentences or major clauses."""
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+|(?:;\s*)', text.strip())
    clauses = []
    for s in raw_sentences:
        s_str = s.strip()
        if not s_str:
            continue
        if ", and " in s_str or ", but " in s_str or ", so " in s_str or ", what " in s_str:
            parts = re.split(r',\s+(?:and|but|so|what|how)\s+', s_str, flags=re.IGNORECASE)
            for p in parts:
                p_clean = p.strip()
                if p_clean:
                    clauses.append(p_clean)
        else:
            clauses.append(s_str)
    return clauses


def has_personal_declarative_content(clause: str) -> bool:
    """Check if a clause asserts a personal statement (not just a question about oneself)."""
    trimmed = clause.strip()
    if not has_first_person(trimmed):
        return False

    # If the clause itself is a pure question starter ("What do you know about me?", "When did I start?")
    for pat in MEMORY_RECALL_PATTERNS:
        if re.search(pat, trimmed, re.IGNORECASE):
            return False

    if trimmed.endswith("?") and re.match(r'^(what|when|where|why|who|how|did|do|can|could|should|am)\b', trimmed, re.IGNORECASE):
        return False

    # Declarative personal verbs / patterns
    declarative_verbs = [
        r"\b(started|learned|learning|live|lives|living|moved|work|works|working|worked|graduated|study|studies|studying|studied)\b",
        r"\b(built|made|creates|created|bought|owns|own|using|use|uses|used|prefers|prefer|likes|like|loves|love|dislikes|dislike|hates|hate|enjoys|enjoy)\b",
        r"\b(plan|plans|planning|hope|hopes|hoping|aim|aims|aiming|want|wants|going to|intend|intends|decided|chose|chosen|opted|switched)\b",
        r"\b(attended|visited|won|finished|published|left|joined|read|reading)\b",
        r"\b(am|is|are|was|were|have|has|had|have been|has been|will be|will start|will learn|will read|will build)\b",
        r"\b(name is|dog is|cat is|favorite|favourite|goal is|choice is|sister's name|brother's name)\b",
        r"\b(no longer|not anymore)\b"
    ]
    return any(re.search(v, trimmed, re.IGNORECASE) for v in declarative_verbs)


def check_redundancy(
    candidate_text: str,
    existing_memories: list[dict],
    similarity_threshold: float = 0.82
) -> tuple[bool, Optional[str]]:
    """
    Check if the candidate text is substantially identical or redundant with an already stored memory.
    Returns (is_duplicate, duplicate_memory_id).
    """
    if not existing_memories or not candidate_text:
        return False, None

    cand_norm = re.sub(r"[^\w\s]", "", candidate_text.lower()).strip()
    cand_tokens = set(cand_norm.split())
    if not cand_tokens:
        return False, None

    for memo in existing_memories:
        m_text = memo.get("text", "")
        if not m_text:
            continue
        m_norm = re.sub(r"[^\w\s]", "", m_text.lower()).strip()
        if cand_norm == m_norm:
            return True, memo.get("id")

        m_tokens = set(m_norm.split())
        if not m_tokens:
            continue

        intersection = len(cand_tokens & m_tokens)
        union = len(cand_tokens | m_tokens)
        jaccard = intersection / union if union > 0 else 0.0

        if jaccard >= similarity_threshold:
            return True, memo.get("id")

    return False, None


def should_extract_memory(
    text: str,
    existing_memories: Optional[list[dict]] = None
) -> bool:
    """Convenience helper to check if a memory should be formed."""
    decision = evaluate_memory_formation(text, existing_memories=existing_memories)
    return decision.should_extract


def evaluate_memory_formation(
    text: str,
    existing_memories: Optional[list[dict]] = None,
    model: str = "gpt-oss",
    llm_caller: Optional[Callable[[str, str], Any]] = None
) -> MemoryFormationDecision:
    """
    Evaluate whether a user utterance should form a long-term personal memory.

    Enforces:
    1. Rejection of greetings, fillers, and pleasantries.
    2. Rejection of general educational queries and instructions.
    3. Rejection of pure recall questions ("When did I...", "What do you know...").
    4. Extraction of personal declarative facts embedded in mixed messages.
    5. Categorical classification into Fact, Preference, Goal, Experience, Decision, Update.
    6. Deduplication against existing memories.
    7. Conservative fallback when in doubt.
    """
    trimmed = text.strip()
    if not trimmed:
        return MemoryFormationDecision(
            should_extract=False,
            category=MemoryCategory.CASUAL_CONVERSATION,
            reason="Empty or whitespace utterance"
        )

    # 1. Casual / Greeting filter
    if is_casual_or_filler(trimmed):
        return MemoryFormationDecision(
            should_extract=False,
            category=MemoryCategory.CASUAL_CONVERSATION,
            reason="Trivial greeting, acknowledgement, or conversational filler"
        )

    # 2. General Knowledge / Educational query filter (e.g. "What is RL?", "Explain backpropagation")
    if is_general_knowledge_query(trimmed):
        return MemoryFormationDecision(
            should_extract=False,
            category=MemoryCategory.GENERAL_KNOWLEDGE,
            reason="General educational request or concept explanation"
        )

    # 3. Pure Question or Memory Recall filter (e.g. "When did I start?", "What do you know about me?")
    if is_pure_question_or_recall(trimmed):
        return MemoryFormationDecision(
            should_extract=False,
            category=MemoryCategory.QUESTION,
            reason="Inquiry or personal memory recall question"
        )

    # 4. Instruction filter
    if is_instruction(trimmed):
        return MemoryFormationDecision(
            should_extract=False,
            category=MemoryCategory.INSTRUCTION,
            reason="Imperative system task or instruction"
        )

    # 5. Segment utterance to inspect clauses for embedded personal facts
    clauses = split_clauses(trimmed)
    personal_declarative_clauses = [c for c in clauses if has_personal_declarative_content(c)]

    if not personal_declarative_clauses:
        # If no personal declarative clause was found
        if trimmed.endswith("?") or any(re.match(r'^(what|when|where|why|who|how)\b', c, re.IGNORECASE) for c in clauses):
            return MemoryFormationDecision(
                should_extract=False,
                category=MemoryCategory.QUESTION,
                reason="Question without declarative personal information"
            )
        # Conservative default for non-personal, non-question utterances
        return MemoryFormationDecision(
            should_extract=False,
            category=MemoryCategory.CASUAL_CONVERSATION,
            reason="No verifiable personal declarative facts detected"
        )

    # We have personal declarative content!
    # Distill the personal declarative text (handling mixed messages)
    if len(personal_declarative_clauses) == len(clauses):
        extracted_text = trimmed
    else:
        # Recombine only the personal declarative clauses
        extracted_text = ". ".join(c.rstrip(".!?") for c in personal_declarative_clauses) + "."

    # 6. Deduplication check against existing memory store
    if existing_memories:
        is_dup, dup_id = check_redundancy(extracted_text, existing_memories)
        if is_dup:
            return MemoryFormationDecision(
                should_extract=False,
                category=MemoryCategory.PERSONAL_FACT,
                extracted_text=extracted_text,
                is_duplicate=True,
                existing_memory_id=dup_id,
                reason=f"Duplicate of existing memory {dup_id}"
            )

    # 7. Fine-grained Category Assignment
    ext_lower = extracted_text.lower()

    # Check for Update / Correction
    if any(re.search(p, ext_lower) for p in UPDATE_PATTERNS):
        return MemoryFormationDecision(
            should_extract=True,
            category=MemoryCategory.DECISION,
            extracted_text=extracted_text,
            memory_type="update",
            is_update=True,
            confidence=0.95,
            reason="Personal fact update or correction"
        )

    # Check for Decision (evaluated before preference to capture choices like "chose X instead of Y")
    if any(re.search(p, ext_lower) for p in DECISION_PATTERNS):
        return MemoryFormationDecision(
            should_extract=True,
            category=MemoryCategory.DECISION,
            extracted_text=extracted_text,
            memory_type="decision",
            confidence=0.90,
            reason="Personal decision or architectural choice stated"
        )

    # Check for Preference
    if any(re.search(p, ext_lower) for p in PREFERENCE_PATTERNS):
        return MemoryFormationDecision(
            should_extract=True,
            category=MemoryCategory.PREFERENCE,
            extracted_text=extracted_text,
            memory_type="preference",
            confidence=0.95,
            reason="Personal preference or affinity stated"
        )

    # Check for Goal / Future intention
    if any(re.search(p, ext_lower) for p in GOAL_PATTERNS):
        return MemoryFormationDecision(
            should_extract=True,
            category=MemoryCategory.GOAL,
            extracted_text=extracted_text,
            memory_type="goal",
            confidence=0.90,
            reason="Personal goal, plan, or future intention stated"
        )

    # Check for Experience / Accomplishment
    if any(re.search(p, ext_lower) for p in EXPERIENCE_PATTERNS):
        return MemoryFormationDecision(
            should_extract=True,
            category=MemoryCategory.EXPERIENCE,
            extracted_text=extracted_text,
            memory_type="experience",
            confidence=0.90,
            reason="Past experience, project, or event stated"
        )

    # Default personal declarative fact
    return MemoryFormationDecision(
        should_extract=True,
        category=MemoryCategory.PERSONAL_FACT,
        extracted_text=extracted_text,
        memory_type="fact",
        confidence=0.85,
        reason="Declarative personal fact stated"
    )
