import re
from .loader import (
    load_entity_schema,
    load_relationship_schema,
    load_rules,
    load_entity_types,
    load_relationships
)
from .vocabulary import (
    load_vocabulary,
    get_canonical_type,
    get_canonical_predicate,
    get_inverse_predicate,
    is_valid_type,
    is_valid_predicate,
    is_type_compatible
)


class ValidationResult:
    """
    Structured validation outcome containing validity flag, errors, and warnings.
    Supports boolean coercion: bool(result) is True when is_valid is True.
    """
    def __init__(self, is_valid: bool = True, errors: list[str] = None, warnings: list[str] = None):
        self.is_valid = is_valid
        self.errors = errors if errors is not None else []
        self.warnings = warnings if warnings is not None else []

    def __bool__(self):
        return self.is_valid

    def __repr__(self):
        return f"<ValidationResult valid={self.is_valid} errors={len(self.errors)} warnings={len(self.warnings)}>"

    def add_error(self, message: str):
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str):
        self.warnings.append(message)

    def to_dict(self):
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings
        }


def _get_field(obj, field_name, default=None):
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def _load_active_rules(rules=None):
    if rules is not None:
        return rules
    try:
        rules_data = load_rules()
        return rules_data.get("rules", {})
    except Exception:
        return {}


def normalize_text_for_grounding(text: str) -> str:
    """
    Safely normalize text for evidence grounding by collapsing whitespace,
    stripping boundary whitespace, and lowercasing.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def is_grounded(evidence: str, memory_text: str) -> bool:
    """
    Verify whether an evidence text span is genuinely grounded in the memory text.
    Collapses internal whitespace differences and strips common boundary punctuation,
    while strictly requiring substring containment.
    """
    if not evidence or not memory_text or not isinstance(evidence, str) or not isinstance(memory_text, str):
        return False

    norm_evidence = normalize_text_for_grounding(evidence)
    norm_memory = normalize_text_for_grounding(memory_text)

    # Strip boundary punctuation or quote marks that extractors might add
    clean_evidence = norm_evidence.strip("\"'.,;:!?")
    if not clean_evidence:
        return False

    return clean_evidence in norm_memory


# ---------------------------------------------------------------------------
# 1. Entity Validation
# ---------------------------------------------------------------------------

def validate_extracted_entity(entity, rules=None, vocabulary=None) -> ValidationResult:
    """
    Validate an entity at the extraction stage (name, type, confidence).
    """
    result = ValidationResult()
    active_rules = _load_active_rules(rules)

    name = _get_field(entity, "name")
    entity_type = _get_field(entity, "type")
    confidence = _get_field(entity, "confidence", 1.0)

    # Rule: entity must have name
    if active_rules.get("entity_must_have_name", True):
        if not name or not isinstance(name, str) or not name.strip():
            result.add_error("Extracted entity must have a non-empty 'name'.")

    # Rule: entity must have type
    if active_rules.get("entity_must_have_type", True):
        if not entity_type or not isinstance(entity_type, str) or not entity_type.strip():
            result.add_error(f"Extracted entity '{name or 'unnamed'}' must have a non-empty 'type'.")
        else:
            # Check against vocabulary
            if not is_valid_type(entity_type, vocabulary):
                result.add_error(f"Entity '{name}' has unrecognized type '{entity_type}' in vocabulary.")

    # Validate confidence bounds
    if confidence is not None:
        try:
            conf_val = float(confidence)
            if not (0.0 <= conf_val <= 1.0):
                result.add_error(f"Entity '{name}' confidence must be between 0.0 and 1.0; got {confidence}.")
        except (ValueError, TypeError):
            result.add_error(f"Entity '{name}' confidence must be a valid float; got {confidence}.")

    return result


def validate_canonical_entity(entity, rules=None, vocabulary=None) -> ValidationResult:
    """
    Validate a canonical entity record (id, canonical_name, types, aliases).
    """
    result = ValidationResult()

    if not isinstance(entity, dict):
        result.add_error("Canonical entity must be a dictionary.")
        return result

    entity_id = entity.get("id")
    canonical_name = entity.get("canonical_name")
    types = entity.get("types")
    aliases = entity.get("aliases")

    if not entity_id or not isinstance(entity_id, str) or not entity_id.strip():
        result.add_error("Canonical entity must have a non-empty 'id'.")

    if not canonical_name or not isinstance(canonical_name, str) or not canonical_name.strip():
        result.add_error("Canonical entity must have a non-empty 'canonical_name'.")

    if not types or not isinstance(types, list) or len(types) == 0:
        result.add_error(f"Canonical entity '{canonical_name or entity_id}' must have a non-empty 'types' list.")
    else:
        for t in types:
            if not isinstance(t, str) or not is_valid_type(t, vocabulary):
                result.add_error(f"Canonical entity '{canonical_name}' has unrecognized type '{t}' in vocabulary.")

    if aliases is not None and not isinstance(aliases, list):
        result.add_error(f"Canonical entity '{canonical_name}' 'aliases' must be a list if present.")

    return result


def validate_entity(entity, stage="extracted", rules=None, vocabulary=None) -> ValidationResult:
    """
    Unified entity validation dispatcher preserving backward compatibility.
    """
    if stage == "canonical" or (isinstance(entity, dict) and "id" in entity and "canonical_name" in entity):
        return validate_canonical_entity(entity, rules=rules, vocabulary=vocabulary)
    return validate_extracted_entity(entity, rules=rules, vocabulary=vocabulary)


# ---------------------------------------------------------------------------
# 2. Relationship Validation
# ---------------------------------------------------------------------------

def validate_extracted_relationship(rel, memory_text=None, rules=None, vocabulary=None) -> ValidationResult:
    """
    Validate a relationship at the extraction stage (source, relation/predicate, target, origin, evidence).
    Consistently enforces relationship_must_have_evidence for both explicit and inferred relationships.
    """
    result = ValidationResult()
    active_rules = _load_active_rules(rules)

    source = _get_field(rel, "source")
    relation = _get_field(rel, "relation") or _get_field(rel, "predicate")
    target = _get_field(rel, "target")
    confidence = _get_field(rel, "confidence", 1.0)
    origin = _get_field(rel, "origin", "explicit")
    evidence = _get_field(rel, "evidence")

    # Source check
    if active_rules.get("relationship_must_have_source", True):
        if not source or not isinstance(source, str) or not source.strip():
            result.add_error("Extracted relationship must have a non-empty 'source'.")

    # Predicate check
    if active_rules.get("relationship_must_have_predicate", True):
        if not relation or not isinstance(relation, str) or not relation.strip():
            result.add_error("Extracted relationship must have a non-empty 'relation' or 'predicate'.")
        else:
            if not is_valid_predicate(relation, vocabulary):
                result.add_error(f"Relationship predicate '{relation}' is not registered in vocabulary.")

    # Target check
    if active_rules.get("relationship_must_have_target", True):
        if not target or not isinstance(target, str) or not target.strip():
            result.add_error("Extracted relationship must have a non-empty 'target'.")

    # Confidence check
    if confidence is not None:
        try:
            conf_val = float(confidence)
            if not (0.0 <= conf_val <= 1.0):
                result.add_error(f"Relationship confidence must be between 0.0 and 1.0; got {confidence}.")
        except (ValueError, TypeError):
            result.add_error(f"Relationship confidence must be a valid float; got {confidence}.")

    # Origin check
    if origin not in ("explicit", "inferred"):
        result.add_error(f"Relationship origin must be 'explicit' or 'inferred'; got '{origin}'.")

    # Evidence policy check: Consistent enforcement across explicit and inferred
    if active_rules.get("relationship_must_have_evidence", True):
        if not evidence or not isinstance(evidence, str) or not evidence.strip():
            if origin == "explicit":
                result.add_error(f"Explicit relationship '{relation}' must have non-empty supporting 'evidence'.")
            else:
                result.add_error(f"Inferred relationship '{relation}' must cite premise text span as 'evidence'.")
        else:
            if memory_text and isinstance(memory_text, str):
                if not is_grounded(evidence, memory_text):
                    result.add_error(
                        f"Relationship evidence '{evidence}' ({origin}) is not grounded in memory text."
                    )
                elif origin == "inferred":
                    result.add_warning(
                        f"Inferred relationship '{relation}' is derived from premise: '{evidence}'."
                    )

    return result



def validate_resolved_relationship(rel, memory_text=None, rules=None, vocabulary=None) -> ValidationResult:
    """
    Validate a relationship at the resolved stage where source and target are endpoint references.
    """
    result = ValidationResult()
    active_rules = _load_active_rules(rules)

    if not isinstance(rel, dict):
        result.add_error("Resolved relationship must be a dictionary.")
        return result

    source = rel.get("source")
    predicate = rel.get("predicate") or rel.get("relation")
    target = rel.get("target")
    confidence = rel.get("confidence", 1.0)
    origin = rel.get("origin", "explicit")
    evidence = rel.get("evidence")

    # Validate source endpoint reference
    if not source or not isinstance(source, dict):
        result.add_error("Resolved relationship 'source' must be an endpoint reference dictionary.")
    else:
        src_type = source.get("type")
        if src_type == "entity":
            if not source.get("id"):
                result.add_error("Canonical source endpoint must have a non-empty 'id'.")
        elif src_type == "candidate":
            if not source.get("mention"):
                result.add_error("Candidate source endpoint must have a non-empty 'mention'.")
        else:
            result.add_error(f"Source endpoint type must be 'entity' or 'candidate'; got '{src_type}'.")

    # Validate predicate
    if not predicate or not isinstance(predicate, str) or not predicate.strip():
        result.add_error("Resolved relationship must have a non-empty 'predicate'.")
    elif not is_valid_predicate(predicate, vocabulary):
        result.add_error(f"Resolved predicate '{predicate}' is not registered in vocabulary.")

    # Validate target endpoint reference
    if not target or not isinstance(target, dict):
        result.add_error("Resolved relationship 'target' must be an endpoint reference dictionary.")
    else:
        tgt_type = target.get("type")
        if tgt_type == "entity":
            if not target.get("id"):
                result.add_error("Canonical target endpoint must have a non-empty 'id'.")
        elif tgt_type == "candidate":
            if not target.get("mention"):
                result.add_error("Candidate target endpoint must have a non-empty 'mention'.")
        else:
            result.add_error(f"Target endpoint type must be 'entity' or 'candidate'; got '{tgt_type}'.")

    # Validate origin and confidence
    if origin not in ("explicit", "inferred"):
        result.add_error(f"Relationship origin must be 'explicit' or 'inferred'; got '{origin}'.")

    if confidence is not None:
        try:
            conf_val = float(confidence)
            if not (0.0 <= conf_val <= 1.0):
                result.add_error(f"Relationship confidence must be between 0.0 and 1.0; got {confidence}.")
        except (ValueError, TypeError):
            result.add_error(f"Relationship confidence must be a valid float; got {confidence}.")

    # Validate evidence grounding
    if active_rules.get("relationship_must_have_evidence", True):
        if not evidence or not isinstance(evidence, str) or not evidence.strip():
            if origin == "explicit":
                result.add_error(f"Explicit relationship '{predicate}' must have supporting 'evidence'.")
            else:
                result.add_error(f"Inferred relationship '{predicate}' must cite premise text span as 'evidence'.")
        else:
            if memory_text and isinstance(memory_text, str):
                if not is_grounded(evidence, memory_text):
                    result.add_error(
                        f"Relationship evidence '{evidence}' ({origin}) is not grounded in memory text."
                    )
                elif origin == "inferred":
                    result.add_warning(
                        f"Inferred relationship '{predicate}' is derived from premise: '{evidence}'."
                    )

    return result


def validate_relationship(relationship, stage="extracted", memory_text=None, rules=None, vocabulary=None) -> ValidationResult:
    """
    Unified relationship validation dispatcher.
    """
    # If source is a dictionary with a "type" field, it's a resolved relationship
    if isinstance(relationship, dict) and isinstance(relationship.get("source"), dict) and "type" in relationship["source"]:
        return validate_resolved_relationship(relationship, memory_text=memory_text, rules=rules, vocabulary=vocabulary)
    return validate_extracted_relationship(relationship, memory_text=memory_text, rules=rules, vocabulary=vocabulary)


# ---------------------------------------------------------------------------
# 3. Extraction Bundle Validation
# ---------------------------------------------------------------------------

def validate_extraction(data, memory_text=None, rules=None, vocabulary=None) -> dict:
    """
    Validate an entire extraction bundle (entities, topics, relationships).
    Returns a dictionary containing valid elements, rejected elements, and diagnostics.
    """
    if hasattr(data, "model_dump"):
        data_dict = data.model_dump()
    elif isinstance(data, dict):
        data_dict = data
    else:
        return {
            "is_valid": False,
            "entities": [],
            "relationships": [],
            "topics": [],
            "errors": ["Extraction data must be a dictionary or Pydantic model."],
            "warnings": [],
            "invalid_entities": [],
            "invalid_relationships": []
        }

    valid_entities = []
    invalid_entities = []
    all_errors = []
    all_warnings = []

    for ent in data_dict.get("entities", []):
        res = validate_extracted_entity(ent, rules=rules, vocabulary=vocabulary)
        if res.is_valid:
            valid_entities.append(ent)
        else:
            invalid_entities.append((ent, res.errors))
            all_errors.extend(res.errors)
        all_warnings.extend(res.warnings)

    valid_relationships = []
    invalid_relationships = []

    for rel in data_dict.get("relationships", []):
        res = validate_extracted_relationship(rel, memory_text=memory_text, rules=rules, vocabulary=vocabulary)
        if res.is_valid:
            valid_relationships.append(rel)
        else:
            invalid_relationships.append((rel, res.errors))
            all_errors.extend(res.errors)
        all_warnings.extend(res.warnings)

    is_all_valid = (len(all_errors) == 0)

    return {
        "is_valid": is_all_valid,
        "entities": valid_entities,
        "relationships": valid_relationships,
        "topics": data_dict.get("topics", []),
        "errors": all_errors,
        "warnings": all_warnings,
        "invalid_entities": invalid_entities,
        "invalid_relationships": invalid_relationships
    }


# ---------------------------------------------------------------------------
# 4. Legacy Functions for test_ontology Compatibility
# ---------------------------------------------------------------------------

def is_valid_entity_type(entity_type: str) -> bool:
    """
    Check if entity type is registered in the ontology vocabulary.
    """
    return is_valid_type(entity_type)


def is_valid_relationship(relation: str) -> bool:
    """
    Check if predicate is registered in the ontology vocabulary.
    """
    return is_valid_predicate(relation)


def is_valid_relationship_types(relation: str, source_type: str, target_type: str) -> bool:
    """
    Validate that relation and both endpoint types are valid according to the ontology.
    """
    if not is_valid_predicate(relation):
        return False
    if not is_valid_type(source_type):
        return False
    if not is_valid_type(target_type):
        return False
    return True


def get_inverse_relationship(relation: str) -> str | None:
    """
    Return the inverse predicate registered in vocabulary for relation, if any.
    """
    return get_inverse_predicate(relation)
