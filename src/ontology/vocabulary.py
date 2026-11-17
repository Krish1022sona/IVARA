from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Literal, Optional


VOCAB_PATH = Path(__file__).parent / "vocabulary.json"

DISALLOWED_STOP_WORDS = {
    "in", "on", "at", "to", "with", "by", "under", "of", "about", "for",
    "from", "is", "was", "are", "were", "has", "had", "have", "the", "a",
    "an", "and", "or", "but", "so", "as", "than", "over", "into", "it"
}

DISALLOWED_META_TOKENS = {
    "type", "status", "id", "name", "data", "value", "property", "attribute",
    "key", "field", "item", "entity", "relationship", "relation", "predicate",
    "null", "none", "unknown", "object", "subject", "true", "false"
}


@dataclass
class PredicateValidationResult:
    """Structured outcome of predicate validation distinguishing canonical, open-world, and invalid states."""
    is_valid: bool
    status: Literal["canonical", "open_world", "invalid"]
    canonical_predicate: Optional[str]
    raw_predicate: str
    normalized_predicate: str
    error: Optional[str] = None
    warning: Optional[str] = None

    def __bool__(self):
        return self.is_valid


def normalize_predicate_name(name: str) -> str:
    """
    Standardize a predicate string:
    - Split camelCase boundaries (e.g. 'worksAt' -> 'works_At')
    - Replace whitespace, hyphens, slashes, and periods with underscores
    - Strip disallowed characters
    - Lowercase and collapse consecutive underscores
    """
    if not isinstance(name, str):
        return ""
    # Split camelCase
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    # Replace non-alphanumeric separators with underscores
    s = re.sub(r"[\s\-\/\.]+", "_", s)
    # Strip any characters other than letters, numbers, and underscores
    s = re.sub(r"[^a-zA-Z0-9_]", "", s)
    # Lowercase and collapse underscores
    s = re.sub(r"_+", "_", s.lower()).strip("_")
    return s


def validate_predicate_structure(name: str) -> tuple[bool, str, Optional[str]]:
    """
    Validate the syntactic and semantic structure of an open-world predicate candidate.
    Returns (is_valid, normalized_name, error_reason).
    """
    if not name or not isinstance(name, str):
        return False, "", "Predicate must be a non-empty string."

    # Check for invalid raw characters (SQL, HTML, punctuation, injection artifacts)
    if re.search(r"[<>;!@#$%^&*()[\]\\/|+=~`\"']", name):
        return False, "", f"Predicate '{name}' contains illegal punctuation or formatting characters."

    # Check for SQL/script commands or injection patterns
    if re.search(r"\b(drop|select|delete|insert|update|table|script|exec|union)\b", name.lower()):
        return False, "", f"Predicate '{name}' contains disallowed command or injection tokens."

    norm = normalize_predicate_name(name)
    if not norm:
        return False, "", "Predicate contains no valid alphanumeric characters."

    if len(norm) < 2:
        return False, norm, f"Predicate '{norm}' is too short (minimum 2 characters required)."

    if len(norm) > 64:
        return False, norm, f"Predicate '{norm}' exceeds maximum length of 64 characters."

    if not re.match(r"^[a-z][a-z0-9_]*$", norm):
        return False, norm, f"Predicate '{norm}' must start with a lowercase letter and contain only alphanumeric characters and underscores."

    if norm in DISALLOWED_STOP_WORDS:
        return False, norm, f"Predicate '{name}' is a pure preposition or stop-word and cannot represent a meaningful relationship."

    if norm in DISALLOWED_META_TOKENS:
        return False, norm, f"Predicate '{name}' is a generic metadata token, not a valid relationship type."

    return True, norm, None


def validate_predicate(
    name: str,
    vocabulary: dict = None,
    allow_open_world: bool = True
) -> PredicateValidationResult:
    """
    Validate a predicate against the canonical ontology and open-world structural constraints.
    Distinguishes:
      - 'canonical': Matches explicit vocabulary entry or alias.
      - 'open_world': Structurally sound and semantically meaningful relationship outside vocabulary.
      - 'invalid': Malformed, empty, stop-word, or non-relational noise.
    """
    if not name or not isinstance(name, str) or not name.strip():
        return PredicateValidationResult(
            is_valid=False,
            status="invalid",
            canonical_predicate=None,
            raw_predicate=str(name or ""),
            normalized_predicate="",
            error="Relationship must have a non-empty 'relation' or 'predicate'."
        )

    raw_str = name.strip()
    norm = normalize_predicate_name(raw_str)

    # 1. Check Canonical Vocabulary (closed-world / known aliases)
    canonical = get_canonical_predicate(raw_str, vocabulary)
    if canonical:
        return PredicateValidationResult(
            is_valid=True,
            status="canonical",
            canonical_predicate=canonical,
            raw_predicate=raw_str,
            normalized_predicate=norm
        )

    # 2. Check Open-World Structural Validation
    is_valid_struct, norm_struct, err_reason = validate_predicate_structure(raw_str)
    if not is_valid_struct:
        return PredicateValidationResult(
            is_valid=False,
            status="invalid",
            canonical_predicate=None,
            raw_predicate=raw_str,
            normalized_predicate=norm_struct,
            error=err_reason
        )

    # If open-world is permitted, accept as open_world predicate
    if allow_open_world:
        return PredicateValidationResult(
            is_valid=True,
            status="open_world",
            canonical_predicate=norm_struct,
            raw_predicate=raw_str,
            normalized_predicate=norm_struct,
            warning=f"Predicate '{norm_struct}' accepted under open-world ontology."
        )

    # If open-world is disabled, reject as unregistered
    return PredicateValidationResult(
        is_valid=False,
        status="invalid",
        canonical_predicate=None,
        raw_predicate=raw_str,
        normalized_predicate=norm_struct,
        error=f"Relationship predicate '{raw_str}' is not registered in vocabulary."
    )



def load_vocabulary():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_vocabulary(vocabulary):
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, indent=2)


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def get_canonical_type(name: str, vocabulary: dict = None) -> str | None:
    if not name or not isinstance(name, str):
        return None

    if vocabulary is None:
        vocabulary = load_vocabulary()

    types = vocabulary.get("types", {})
    norm_name = normalize_name(name)

    # 1. Exact match on canonical key
    if name in types:
        return name

    # 2. Case-insensitive match on canonical key
    for canonical_name in types:
        if normalize_name(canonical_name) == norm_name:
            return canonical_name

    # 3. Match against aliases
    for canonical_name, type_info in types.items():
        aliases = type_info.get("aliases", [])
        for alias in aliases:
            if normalize_name(alias) == norm_name:
                return canonical_name

    return None


def is_valid_type(name: str, vocabulary: dict = None) -> bool:
    return get_canonical_type(name, vocabulary) is not None


def get_canonical_predicate(name: str, vocabulary: dict = None) -> str | None:
    if not name or not isinstance(name, str):
        return None

    if vocabulary is None:
        vocabulary = load_vocabulary()

    predicates = vocabulary.get("predicates", {})
    norm_name = normalize_name(name)
    norm_name_under = norm_name.replace(" ", "_")

    # 1. Exact match
    if name in predicates:
        return name

    # 2. Case-insensitive or underscore match
    for canonical_name in predicates:
        canon_norm = normalize_name(canonical_name)
        if canon_norm == norm_name or canon_norm == norm_name_under:
            return canonical_name

    # 3. Match against aliases
    for canonical_name, pred_info in predicates.items():
        aliases = pred_info.get("aliases", [])
        for alias in aliases:
            alias_norm = normalize_name(alias)
            alias_norm_under = alias_norm.replace(" ", "_")
            if alias_norm == norm_name or alias_norm_under == norm_name_under:
                return canonical_name

    return None


def is_valid_predicate(name: str, vocabulary: dict = None, allow_open_world: bool = False) -> bool:
    if allow_open_world:
        return validate_predicate(name, vocabulary=vocabulary, allow_open_world=True).is_valid
    return get_canonical_predicate(name, vocabulary) is not None


def get_inverse_predicate(name: str, vocabulary: dict = None) -> str | None:
    if vocabulary is None:
        vocabulary = load_vocabulary()

    canonical = get_canonical_predicate(name, vocabulary)
    if canonical:
        pred_info = vocabulary.get("predicates", {}).get(canonical, {})
        return pred_info.get("inverse")

    # Symmetric open-world predicates
    norm = normalize_predicate_name(name)
    if norm in ("roommate_of", "has_sibling", "friend_of", "married_to", "partner_in", "collaborated_with"):
        return norm
    return None


def is_type_compatible(
    requested_type: str,
    candidate_types: list[str] | str,
    vocabulary: dict = None
) -> bool:
    if not requested_type or not candidate_types:
        return False

    if vocabulary is None:
        vocabulary = load_vocabulary()

    canon_req = get_canonical_type(requested_type, vocabulary)
    if not canon_req:
        # If requested type is ungrounded, fall back to literal case-insensitive string comparison
        canon_req = normalize_name(requested_type)

    if isinstance(candidate_types, str):
        candidate_types = [candidate_types]

    types_dict = vocabulary.get("types", {})

    for cand in candidate_types:
        canon_cand = get_canonical_type(cand, vocabulary)
        if not canon_cand:
            canon_cand = normalize_name(cand)

        # 1. Direct match
        if canon_req.lower() == canon_cand.lower():
            return True

        # 2. Check if canon_cand is an ancestor of canon_req
        curr = canon_req
        while curr in types_dict and types_dict[curr].get("parent"):
            curr = types_dict[curr]["parent"]
            if curr.lower() == canon_cand.lower():
                return True

        # 3. Check if canon_req is an ancestor of canon_cand
        curr = canon_cand
        while curr in types_dict and types_dict[curr].get("parent"):
            curr = types_dict[curr]["parent"]
            if curr.lower() == canon_req.lower():
                return True

    return False


def validate_vocabulary(vocabulary: dict = None) -> list[str]:
    if vocabulary is None:
        vocabulary = load_vocabulary()

    errors = []
    types = vocabulary.get("types", {})
    predicates = vocabulary.get("predicates", {})

    # 1. Check types: parent references, duplicate aliases, cycles
    type_alias_seen = {}
    for type_name, type_info in types.items():
        type_norm = normalize_name(type_name)
        if type_norm in type_alias_seen and type_alias_seen[type_norm] != type_name:
            errors.append(f"Duplicate type or alias collision for '{type_norm}' between '{type_name}' and '{type_alias_seen[type_norm]}'.")
        type_alias_seen[type_norm] = type_name

        parent = type_info.get("parent")
        if parent is not None and parent not in types:
            errors.append(f"Type '{type_name}' references non-existent parent type '{parent}'.")

        # Cycle detection
        curr = parent
        visited = {type_name}
        while curr:
            if curr in visited:
                errors.append(f"Cyclic inheritance detected in type hierarchy involving '{type_name}' and '{curr}'.")
                break
            visited.add(curr)
            curr = types.get(curr, {}).get("parent")

        aliases = type_info.get("aliases", [])
        seen_in_type = set()
        for alias in aliases:
            norm_a = normalize_name(alias)
            if norm_a in seen_in_type:
                errors.append(f"Duplicate alias '{alias}' within type '{type_name}'.")
            seen_in_type.add(norm_a)
            if norm_a in type_alias_seen and type_alias_seen[norm_a] != type_name:
                errors.append(f"Duplicate alias '{alias}' in type '{type_name}', already registered by '{type_alias_seen[norm_a]}'.")
            type_alias_seen[norm_a] = type_name

    # 2. Check predicates: inverse references, reciprocal check, duplicate aliases
    pred_alias_seen = {}
    for pred_name, pred_info in predicates.items():
        pred_norm = normalize_name(pred_name).replace(" ", "_")
        if pred_norm in pred_alias_seen and pred_alias_seen[pred_norm] != pred_name:
            errors.append(f"Duplicate predicate or alias collision for '{pred_norm}' between '{pred_name}' and '{pred_alias_seen[pred_norm]}'.")
        pred_alias_seen[pred_norm] = pred_name

        inverse = pred_info.get("inverse")
        if inverse is not None:
            if inverse not in predicates:
                errors.append(f"Predicate '{pred_name}' references non-existent inverse predicate '{inverse}'.")
            else:
                reciprocal = predicates[inverse].get("inverse")
                if reciprocal != pred_name:
                    errors.append(
                        f"Predicate '{pred_name}' has inverse '{inverse}', but '{inverse}' has inverse '{reciprocal}'."
                    )

        aliases = pred_info.get("aliases", [])
        seen_in_pred = set()
        for alias in aliases:
            norm_pa = normalize_name(alias).replace(" ", "_")
            if norm_pa in seen_in_pred:
                errors.append(f"Duplicate alias '{alias}' within predicate '{pred_name}'.")
            seen_in_pred.add(norm_pa)
            if norm_pa in pred_alias_seen and pred_alias_seen[norm_pa] != pred_name:
                errors.append(f"Duplicate alias '{alias}' in predicate '{pred_name}', already registered by '{pred_alias_seen[norm_pa]}'.")
            pred_alias_seen[norm_pa] = pred_name

    return errors


def add_type(name, aliases=None, parent=None):
    vocabulary = load_vocabulary()

    if name not in vocabulary["types"]:
        vocabulary["types"][name] = {
            "aliases": aliases or [],
            "parent": parent
        }

        save_vocabulary(vocabulary)


def add_predicate(name, aliases=None, inverse=None):
    vocabulary = load_vocabulary()

    if name not in vocabulary["predicates"]:
        vocabulary["predicates"][name] = {
            "aliases": aliases or [],
            "inverse": inverse
        }

        save_vocabulary(vocabulary)
