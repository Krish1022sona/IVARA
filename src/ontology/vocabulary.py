import json
from pathlib import Path


VOCAB_PATH = Path(__file__).parent / "vocabulary.json"


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


def is_valid_predicate(name: str, vocabulary: dict = None) -> bool:
    return get_canonical_predicate(name, vocabulary) is not None


def get_inverse_predicate(name: str, vocabulary: dict = None) -> str | None:
    if vocabulary is None:
        vocabulary = load_vocabulary()

    canonical = get_canonical_predicate(name, vocabulary)
    if not canonical:
        return None

    pred_info = vocabulary.get("predicates", {}).get(canonical, {})
    return pred_info.get("inverse")


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
