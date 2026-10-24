import json
import tempfile
import unittest
from pathlib import Path

from src.ontology.loader import (
    load_entity_schema,
    load_relationship_schema,
    load_rules,
    load_json
)
from src.ontology.vocabulary import (
    load_vocabulary,
    get_canonical_type,
    get_canonical_predicate,
    get_inverse_predicate,
    is_valid_type,
    is_valid_predicate,
    is_type_compatible,
    validate_vocabulary
)
from src.ontology.validator import (
    ValidationResult,
    validate_extracted_entity,
    validate_canonical_entity,
    validate_entity,
    validate_extracted_relationship,
    validate_resolved_relationship,
    validate_relationship,
    validate_extraction,
    is_valid_entity_type,
    is_valid_relationship,
    is_valid_relationship_types,
    get_inverse_relationship
)
from src.ontology.entity_store import get_self
from src.ontology.entity_resolver import resolve_entity
from src.ontology.relationship_resolver import resolve_relationship, find_resolved_entity


class TestOntologyLoaderAndSchemas(unittest.TestCase):
    """
    Tests for schema loading, missing file errors, malformed JSON, and section checks.
    """

    def test_valid_schemas_load_successfully(self):
        ent_schema = load_entity_schema()
        rel_schema = load_relationship_schema()
        rules = load_rules()

        self.assertIn("entity", ent_schema)
        self.assertIn("type", ent_schema)
        self.assertIn("special_entities", ent_schema)

        self.assertIn("relationship", rel_schema)
        self.assertIn("predicate", rel_schema)

        self.assertIn("rules", rules)
        self.assertTrue(rules["rules"].get("entity_must_have_name"))

    def test_missing_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            load_json("non_existent_file.json")

    def test_malformed_json_raises_value_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=Path(__file__).parent.parent / "src" / "ontology") as tmp:
            tmp.write("{ invalid json : true ")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError):
                load_json(tmp_path.name)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestVocabularyModule(unittest.TestCase):
    """
    Tests for vocabulary integrity, canonical lookups, alias resolution, and type hierarchy.
    """

    def test_vocabulary_structural_integrity(self):
        vocab = load_vocabulary()
        errors = validate_vocabulary(vocab)
        self.assertEqual(errors, [], f"Vocabulary has structural errors: {errors}")

    def test_canonical_type_lookup_and_aliases(self):
        self.assertEqual(get_canonical_type("Person"), "Person")
        self.assertEqual(get_canonical_type("person"), "Person")
        self.assertEqual(get_canonical_type(" human "), "Person")
        self.assertEqual(get_canonical_type("tech"), "Technology")
        self.assertEqual(get_canonical_type("programming language"), "ProgrammingLanguage")
        self.assertIsNone(get_canonical_type("Pokemon"))

    def test_canonical_predicate_lookup_and_aliases(self):
        self.assertEqual(get_canonical_predicate("worked_with"), "worked_with")
        self.assertEqual(get_canonical_predicate("worked with"), "worked_with")
        self.assertEqual(get_canonical_predicate("collaborated_with"), "worked_with")
        self.assertEqual(get_canonical_predicate("built"), "worked_on")
        self.assertEqual(get_canonical_predicate("uses"), "uses")
        self.assertEqual(get_canonical_predicate("friend_of"), "friend_of")
        self.assertIsNone(get_canonical_predicate("eats"))

    def test_inverse_predicates(self):
        self.assertEqual(get_inverse_predicate("parent_of"), "child_of")
        self.assertEqual(get_inverse_predicate("child_of"), "parent_of")
        self.assertEqual(get_inverse_predicate("friend_of"), "friend_of")
        self.assertEqual(get_inverse_predicate("worked_on"), "worked_on_by")
        self.assertIsNone(get_inverse_predicate("unknown_predicate"))

    def test_type_compatibility_and_hierarchy(self):
        # Direct matches
        self.assertTrue(is_type_compatible("Person", ["Person"]))
        self.assertTrue(is_type_compatible("person", ["Person"]))

        # Subtype / parent hierarchy: Company is an Organization
        self.assertTrue(is_type_compatible("Company", ["Organization"]))
        self.assertTrue(is_type_compatible("Organization", ["Company"]))
        self.assertTrue(is_type_compatible("Framework", ["Technology"]))

        # Incompatible types
        self.assertFalse(is_type_compatible("Person", ["Technology"]))
        self.assertFalse(is_type_compatible("Technology", ["Organization"]))


class TestEntityValidation(unittest.TestCase):
    """
    Tests for extracted and canonical entity validation.
    """

    def test_valid_extracted_entity(self):
        ent = {"name": "PyTorch", "type": "Technology", "confidence": 1.0}
        res = validate_extracted_entity(ent)
        self.assertTrue(res)
        self.assertEqual(res.errors, [])

    def test_extracted_entity_missing_name(self):
        ent = {"name": "", "type": "Technology", "confidence": 1.0}
        res = validate_extracted_entity(ent)
        self.assertFalse(res)
        self.assertTrue(any("name" in e for e in res.errors))

    def test_extracted_entity_unknown_type(self):
        ent = {"name": "Pikachu", "type": "Pokemon", "confidence": 1.0}
        res = validate_extracted_entity(ent)
        self.assertFalse(res)
        self.assertTrue(any("unrecognized type" in e for e in res.errors))

    def test_extracted_entity_invalid_confidence(self):
        ent = {"name": "Rahul", "type": "Person", "confidence": 1.5}
        res = validate_extracted_entity(ent)
        self.assertFalse(res)
        self.assertTrue(any("confidence" in e for e in res.errors))

    def test_valid_canonical_entity(self):
        canonical = {
            "id": "e001",
            "canonical_name": "PyTorch",
            "aliases": ["pytorch"],
            "types": ["Technology", "Framework"]
        }
        res = validate_canonical_entity(canonical)
        self.assertTrue(res)

    def test_malformed_canonical_entity(self):
        # Missing ID
        res1 = validate_canonical_entity({"canonical_name": "PyTorch", "types": ["Technology"]})
        self.assertFalse(res1)

        # Empty types
        res2 = validate_canonical_entity({"id": "e001", "canonical_name": "PyTorch", "types": []})
        self.assertFalse(res2)

        # Unrecognized type in types list
        res3 = validate_canonical_entity({"id": "e001", "canonical_name": "PyTorch", "types": ["Pokemon"]})
        self.assertFalse(res3)

    def test_self_special_entity(self):
        self_entity = get_self()
        res = validate_canonical_entity(self_entity)
        self.assertTrue(res)
        self.assertEqual(self_entity["id"], "self")
        self.assertEqual(self_entity["types"], ["Person"])


class TestRelationshipValidation(unittest.TestCase):
    """
    Tests for extracted and resolved relationship validation, origin, and evidence grounding.
    """

    def test_valid_explicit_relationship_with_grounded_evidence(self):
        memory_text = "I built an LSTM using PyTorch for stock prediction."
        rel = {
            "source": "SELF",
            "relation": "uses",
            "target": "PyTorch",
            "confidence": 1.0,
            "origin": "explicit",
            "evidence": "using PyTorch"
        }
        res = validate_extracted_relationship(rel, memory_text=memory_text)
        self.assertTrue(res)

    def test_explicit_relationship_missing_evidence_fails(self):
        rel = {
            "source": "SELF",
            "relation": "uses",
            "target": "PyTorch",
            "confidence": 1.0,
            "origin": "explicit",
            "evidence": None
        }
        res = validate_extracted_relationship(rel)
        self.assertFalse(res)
        self.assertTrue(any("evidence" in e for e in res.errors))

    def test_explicit_relationship_ungrounded_evidence_fails(self):
        memory_text = "I built an LSTM using PyTorch for stock prediction."
        rel = {
            "source": "SELF",
            "relation": "uses",
            "target": "PyTorch",
            "confidence": 1.0,
            "origin": "explicit",
            "evidence": "completely fabricated evidence not in text"
        }
        res = validate_extracted_relationship(rel, memory_text=memory_text)
        self.assertFalse(res)
        self.assertTrue(any("not grounded" in e for e in res.errors))

    def test_inferred_relationship_with_grounded_premise_evidence_passes_with_warning(self):
        memory_text = "I built an LSTM using PyTorch."
        rel = {
            "source": "SELF",
            "relation": "worked_on",
            "target": "LSTM",
            "confidence": 0.8,
            "origin": "inferred",
            "evidence": "built an LSTM"
        }
        res = validate_extracted_relationship(rel, memory_text=memory_text)
        self.assertTrue(res)  # Passes validity check
        self.assertTrue(len(res.warnings) > 0)  # Contains informative warning
        self.assertTrue(any("derived from premise" in w for w in res.warnings))

    def test_inferred_relationship_without_evidence_fails_under_configured_rules(self):
        memory_text = "I built an LSTM using PyTorch."
        rel = {
            "source": "SELF",
            "relation": "worked_on",
            "target": "LSTM",
            "confidence": 0.8,
            "origin": "inferred",
            "evidence": None
        }
        res = validate_extracted_relationship(rel, memory_text=memory_text)
        self.assertFalse(res)  # Fails because relationship_must_have_evidence is true
        self.assertTrue(any("must cite premise text span" in e for e in res.errors))

    def test_inferred_relationship_without_evidence_passes_when_rule_disabled(self):
        memory_text = "I built an LSTM using PyTorch."
        rel = {
            "source": "SELF",
            "relation": "worked_on",
            "target": "LSTM",
            "confidence": 0.8,
            "origin": "inferred",
            "evidence": None
        }
        res = validate_extracted_relationship(
            rel,
            memory_text=memory_text,
            rules={"relationship_must_have_evidence": False}
        )
        self.assertTrue(res)

    def test_inferred_relationship_with_ungrounded_premise_fails(self):
        memory_text = "I built an LSTM using PyTorch."
        rel = {
            "source": "SELF",
            "relation": "worked_on",
            "target": "LSTM",
            "confidence": 0.8,
            "origin": "inferred",
            "evidence": "invented the transformer in 2017"
        }
        res = validate_extracted_relationship(rel, memory_text=memory_text)
        self.assertFalse(res)
        self.assertTrue(any("not grounded in memory text" in e for e in res.errors))

    def test_invalid_origin_fails(self):
        rel = {
            "source": "SELF",
            "relation": "uses",
            "target": "PyTorch",
            "confidence": 1.0,
            "origin": "hallucinated",
            "evidence": "using PyTorch"
        }
        res = validate_extracted_relationship(rel)
        self.assertFalse(res)
        self.assertTrue(any("origin" in e for e in res.errors))

    def test_resolved_relationship_with_canonical_and_candidate_endpoints(self):
        resolved = {
            "source": {"type": "entity", "id": "self"},
            "predicate": "worked_with",
            "target": {"type": "candidate", "mention": "Rahul", "types": ["Person"]},
            "confidence": 0.95,
            "origin": "explicit",
            "evidence": "worked with Rahul"
        }
        res = validate_resolved_relationship(resolved, memory_text="I worked with Rahul on SIH.")
        self.assertTrue(res)

    def test_malformed_resolved_endpoint_fails(self):
        # Missing id on canonical entity
        bad_rel = {
            "source": {"type": "entity", "id": ""},
            "predicate": "worked_with",
            "target": {"type": "candidate", "mention": "Rahul", "types": ["Person"]},
            "confidence": 0.95,
            "origin": "explicit",
            "evidence": "worked with Rahul"
        }
        res = validate_resolved_relationship(bad_rel, memory_text="I worked with Rahul on SIH.")
        self.assertFalse(res)
        self.assertTrue(any("id" in e for e in res.errors))


class TestResolutionAndPipelineIntegration(unittest.TestCase):
    """
    Tests for type-compatible entity resolution, relationship resolver diagnostic reporting,
    and legacy validator function compatibility.
    """

    def test_type_compatible_resolution(self):
        entities = [
            {"id": "e001", "canonical_name": "Python", "types": ["Technology", "ProgrammingLanguage"], "aliases": []}
        ]
        embeddings = [[1.0, 0.0, 0.0]]

        # When resolving mention "Python" with matching type "Technology", it resolves to e001
        res_tech = resolve_entity("Python", "Technology", entities, embeddings)
        self.assertEqual(res_tech["status"], "matched")
        self.assertEqual(res_tech["entity"]["id"], "e001")

        # When resolving mention "Python" with incompatible type "Person", it must NOT match e001
        res_person = resolve_entity("Python", "Person", entities, embeddings)
        self.assertEqual(res_person["status"], "candidate")
        self.assertIsNone(res_person["entity"].get("id"))

    def test_relationship_resolver_unresolved_diagnostics(self):
        # Only Rahul is in resolved entities; SIH project is missing
        resolved_entities = [
            {"mention": "Rahul", "entity_id": None, "canonical_name": "Rahul", "types": ["Person"], "status": "candidate"}
        ]
        rel = {
            "source": "Rahul",
            "relation": "worked_on",
            "target": "SIH project",
            "confidence": 0.95,
            "origin": "explicit",
            "evidence": "worked on SIH project"
        }

        res = resolve_relationship(rel, resolved_entities)
        self.assertEqual(res["status"], "unresolved")
        self.assertFalse(res["resolved"])
        self.assertIn("target 'SIH project'", res["error"])

    def test_incompatible_type_cannot_be_merged(self):
        entities = [
            {"id": "e001", "canonical_name": "Apple", "types": ["Company"], "aliases": ["Apple Inc"]},
            {"id": "e002", "canonical_name": "Python", "types": ["ProgrammingLanguage"], "aliases": ["Python3"]}
        ]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]

        # Exact name match with incompatible type Person must NOT merge into Company
        res = resolve_entity("Apple", "Person", entities, embeddings)
        self.assertEqual(res["status"], "candidate")
        self.assertIsNone(res["entity"].get("id"))

        # Alias match with incompatible type Person must NOT merge into Company
        res_alias = resolve_entity("Apple Inc", "Person", entities, embeddings)
        self.assertEqual(res_alias["status"], "candidate")
        self.assertIsNone(res_alias["entity"].get("id"))

        # Exact name with incompatible type Person must NOT merge into ProgrammingLanguage
        res_py = resolve_entity("Python", "Person", entities, embeddings)
        self.assertEqual(res_py["status"], "candidate")
        self.assertIsNone(res_py["entity"].get("id"))

    def test_compatible_type_matches(self):
        entities = [
            {"id": "e001", "canonical_name": "Google", "types": ["Organization"], "aliases": []}
        ]
        embeddings = [[1.0, 0.0]]

        # Subtype Company is compatible with Organization
        res = resolve_entity("Google", "Company", entities, embeddings)
        self.assertEqual(res["status"], "matched")
        self.assertEqual(res["entity"]["id"], "e001")

        # Supertype Organization is compatible with Company
        entities_company = [
            {"id": "e002", "canonical_name": "DeepMind", "types": ["Company"], "aliases": []}
        ]
        res_org = resolve_entity("DeepMind", "Organization", entities_company, embeddings)
        self.assertEqual(res_org["status"], "matched")
        self.assertEqual(res_org["entity"]["id"], "e002")

    def test_alias_resolution_with_type_compatibility(self):
        entities = [
            {"id": "e003", "canonical_name": "PyTorch", "types": ["Framework"], "aliases": ["torch"]}
        ]
        embeddings = [[1.0, 0.0]]

        # Compatible alias match
        res = resolve_entity("torch", "Framework", entities, embeddings)
        self.assertEqual(res["status"], "matched")
        self.assertEqual(res["entity"]["id"], "e003")

        # Incompatible alias match
        res_incomp = resolve_entity("torch", "Person", entities, embeddings)
        self.assertEqual(res_incomp["status"], "candidate")
        self.assertIsNone(res_incomp["entity"].get("id"))

    def test_multiple_types_on_canonical_entity(self):
        entities = [
            {"id": "e004", "canonical_name": "Rust", "types": ["ProgrammingLanguage", "Technology"], "aliases": []}
        ]
        embeddings = [[1.0, 0.0]]

        # Matching first type
        res1 = resolve_entity("Rust", "ProgrammingLanguage", entities, embeddings)
        self.assertEqual(res1["status"], "matched")

        # Matching second type
        res2 = resolve_entity("Rust", "Technology", entities, embeddings)
        self.assertEqual(res2["status"], "matched")

        # Incompatible type
        res3 = resolve_entity("Rust", "Skill", entities, embeddings)
        self.assertEqual(res3["status"], "candidate")
        self.assertIsNone(res3["entity"].get("id"))

    def test_self_entity_resolution(self):
        entities = []
        embeddings = []

        res = resolve_entity("SELF", "Person", entities, embeddings)
        self.assertEqual(res["status"], "matched")
        self.assertEqual(res["method"], "special")
        self.assertEqual(res["entity"]["id"], "self")
        self.assertEqual(res["entity"]["canonical_name"], "SELF")

        res_lower = resolve_entity("self", "Person", entities, embeddings)
        self.assertEqual(res_lower["status"], "matched")
        self.assertEqual(res_lower["entity"]["id"], "self")

    def test_evidence_grounding_whitespace_and_punctuation_normalization(self):
        from src.ontology.validator import is_grounded, normalize_text_for_grounding

        memory = "I built an AI assistant\n   using PyTorch\tand FastAPI."

        # Exact substring with whitespace differences (newlines, tabs, multiple spaces)
        self.assertTrue(is_grounded("using PyTorch and FastAPI", memory))
        self.assertTrue(is_grounded("using   PyTorch\tand   FastAPI", memory))

        # Trailing / leading punctuation and quotes
        self.assertTrue(is_grounded('"using PyTorch and FastAPI."', memory))
        self.assertTrue(is_grounded("'using PyTorch'", memory))

        # Case insensitivity
        self.assertTrue(is_grounded("USING PYTORCH", memory))

        # Rejection of ungrounded / hallucinated text
        self.assertFalse(is_grounded("using TensorFlow", memory))
        self.assertFalse(is_grounded("built a blockchain app", memory))
        self.assertFalse(is_grounded("", memory))
        self.assertFalse(is_grounded("   ", memory))
        self.assertFalse(is_grounded("...", memory))

    def test_vocabulary_integrity_validation(self):
        # Current vocabulary is structurally valid
        vocab = load_vocabulary()
        self.assertEqual(validate_vocabulary(vocab), [])

        # Mock vocabulary with duplicate alias
        mock_vocab_dup = {
            "types": {
                "TypeA": {"aliases": ["alias1"], "parent": None},
                "TypeB": {"aliases": ["alias1"], "parent": None}
            },
            "predicates": {}
        }
        errors = validate_vocabulary(mock_vocab_dup)
        self.assertTrue(any("Duplicate alias" in e for e in errors))

        # Mock vocabulary with cyclic inheritance
        mock_vocab_cycle = {
            "types": {
                "TypeA": {"aliases": [], "parent": "TypeB"},
                "TypeB": {"aliases": [], "parent": "TypeA"}
            },
            "predicates": {}
        }
        cycle_errors = validate_vocabulary(mock_vocab_cycle)
        self.assertTrue(any("Cyclic inheritance" in e for e in cycle_errors))

        # Mock vocabulary with broken inverse
        mock_vocab_inverse = {
            "types": {},
            "predicates": {
                "predA": {"aliases": [], "inverse": "predB"},
                "predB": {"aliases": [], "inverse": "predC"}
            }
        }
        inv_errors = validate_vocabulary(mock_vocab_inverse)
        self.assertTrue(any("inverse" in e for e in inv_errors))

    def test_legacy_functions_compatibility(self):
        self.assertTrue(is_valid_entity_type("Person"))
        self.assertFalse(is_valid_entity_type("Pokemon"))

        self.assertTrue(is_valid_relationship("friend_of"))
        self.assertFalse(is_valid_relationship("eats"))

        self.assertTrue(is_valid_relationship_types("friend_of", "Person", "Person"))
        self.assertFalse(is_valid_relationship_types("friend_of", "Pokemon", "Person"))

        self.assertEqual(get_inverse_relationship("parent_of"), "child_of")
        self.assertEqual(get_inverse_relationship("friend_of"), "friend_of")


if __name__ == "__main__":
    unittest.main()
