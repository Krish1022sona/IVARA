import json
import tempfile
import unittest
from pathlib import Path

from evaluation.dataset import (
    BENCHMARK_CASES,
    SYNTHETIC_MEMORIES,
    get_benchmark_dataset,
)
from evaluation.metrics import (
    citation_precision,
    compute_abstention_metrics,
    fact_accuracy,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
    unsupported_claim_rate,
)
from evaluation.runner import generate_markdown_report, run_benchmark


class TestBenchmarkMetrics(unittest.TestCase):
    """
    Unit tests for deterministic metric calculations, edge cases, and empty denominator conventions.
    """

    def test_retrieval_recall_normal_and_empty_denominators(self):
        # Normal retrieval recall
        self.assertAlmostEqual(retrieval_recall_at_k(["m1", "m2"], ["m1", "m2", "m3"]), 2 / 3, places=3)
        self.assertEqual(retrieval_recall_at_k(["m1", "m2"], ["m1", "m2"]), 1.0)
        self.assertEqual(retrieval_recall_at_k(["m4"], ["m1", "m2"]), 0.0)

        # Empty supporting IDs (unanswerable / distractor cases)
        self.assertEqual(retrieval_recall_at_k([], []), 1.0)  # Correctly rejected
        self.assertEqual(retrieval_recall_at_k(["m1"], []), 0.0)  # False positive retrieval

        # Empty retrieved IDs on answerable case
        self.assertEqual(retrieval_recall_at_k([], ["m1"]), 0.0)

    def test_retrieval_precision_normal_and_empty_denominators(self):
        # Normal retrieval precision
        self.assertEqual(retrieval_precision_at_k(["m1", "m2"], ["m1"]), 0.5)
        self.assertEqual(retrieval_precision_at_k(["m1", "m2"], ["m1", "m2"]), 1.0)
        self.assertEqual(retrieval_precision_at_k(["m3"], ["m1", "m2"]), 0.0)

        # Empty retrieved IDs
        self.assertEqual(retrieval_precision_at_k([], []), 1.0)  # Correctly empty on unanswerable
        self.assertEqual(retrieval_precision_at_k([], ["m1"]), 0.0)  # Missed answerable

    def test_citation_precision(self):
        # Normal citations
        self.assertEqual(citation_precision(["m1", "m2"], ["m1"]), 0.5)
        self.assertEqual(citation_precision(["[m1]", "'m2'"], ["m1", "m2"]), 1.0)
        self.assertEqual(citation_precision(["m3"], ["m1", "m2"]), 0.0)

        # Empty citations
        self.assertEqual(citation_precision([], []), 1.0)  # Unanswerable, properly cited nothing
        self.assertEqual(citation_precision([], ["m1"]), 0.0)  # Answerable, but cited nothing

    def test_unsupported_claim_rate(self):
        # Normal hallucinated citations
        self.assertEqual(unsupported_claim_rate(["m1", "m999"], ["m1"]), 0.5)
        self.assertEqual(unsupported_claim_rate(["m1"], ["m1"]), 0.0)
        self.assertEqual(unsupported_claim_rate(["m888", "m999"], ["m1"]), 1.0)

        # Empty citations
        self.assertEqual(unsupported_claim_rate([], ["m1"]), 0.0)
        self.assertEqual(unsupported_claim_rate([], []), 0.0)

    def test_fact_accuracy(self):
        # Answerable cases
        facts = ["PyTorch", "LSTM"]
        self.assertEqual(fact_accuracy("I trained an LSTM using PyTorch.", facts), 1.0)
        self.assertEqual(fact_accuracy("I used PyTorch.", facts), 0.5)
        self.assertEqual(fact_accuracy("I used TensorFlow.", facts), 0.0)
        self.assertEqual(fact_accuracy("Anything", []), 1.0)

        # Unanswerable cases
        self.assertEqual(fact_accuracy("I do not know", [], is_unanswerable=True, status="insufficient_evidence"), 1.0)
        self.assertEqual(fact_accuracy("I play chess", [], is_unanswerable=True, status="answered"), 0.0)

    def test_abstention_metrics(self):
        # Perfect predictions
        preds = [True, True, False, False]
        truths = [True, True, False, False]
        stats = compute_abstention_metrics(preds, truths)
        self.assertEqual(stats["abstention_precision"], 1.0)
        self.assertEqual(stats["abstention_recall"], 1.0)
        self.assertEqual(stats["abstention_f1"], 1.0)
        self.assertEqual(stats["abstention_accuracy"], 1.0)

        # Length mismatch error
        with self.assertRaises(ValueError):
            compute_abstention_metrics([True], [True, False])


class TestBenchmarkDataset(unittest.TestCase):
    """
    Verification of synthetic dataset schema, IDs, annotations, and category coverage.
    """

    def test_dataset_size_and_categories(self):
        mems, cases = get_benchmark_dataset()

        # Target 30-50 cases
        self.assertGreaterEqual(len(cases), 30)
        self.assertLessEqual(len(cases), 50)
        self.assertGreaterEqual(len(mems), 10)

        expected_cats = {
            "direct_factual_recall",
            "entity_alias_resolution",
            "relationship_based",
            "multi_memory_evidence",
            "partial_supporting_evidence",
            "unanswerable_abstention",
            "similar_irrelevant_distractor"
        }
        present_cats = set(c["category"] for c in cases)
        self.assertEqual(expected_cats, present_cats)

    def test_case_schema_and_id_integrity(self):
        mems, cases = get_benchmark_dataset()
        valid_mem_ids = {m["id"] for m in mems}

        seen_case_ids = set()
        for c in cases:
            self.assertIn("case_id", c)
            self.assertNotIn(c["case_id"], seen_case_ids)
            seen_case_ids.add(c["case_id"])

            self.assertIn("query", c)
            self.assertTrue(len(c["query"].strip()) > 0)

            self.assertIn("category", c)
            self.assertIn("expected_answerable", c)
            self.assertIn(c["expected_answerable"], ("answered", "partial", "insufficient_evidence"))

            # All supporting memory IDs must exist in synthetic memories
            for mid in c["supporting_memory_ids"]:
                self.assertIn(mid, valid_mem_ids, f"Case {c['case_id']} references missing memory {mid}")


class TestBenchmarkRunnerAndSafety(unittest.TestCase):
    """
    Integration tests for benchmark execution, report generation, and data isolation.
    """

    def setUp(self):
        self.real_data_dir = Path(__file__).parent.parent / "data"
        self.real_memories_path = self.real_data_dir / "memories.json"
        self.real_entities_path = self.real_data_dir / "entities.json"

        self.real_memories_before = (
            self.real_memories_path.read_text(encoding="utf-8")
            if self.real_memories_path.exists()
            else None
        )
        self.real_entities_before = (
            self.real_entities_path.read_text(encoding="utf-8")
            if self.real_entities_path.exists()
            else None
        )

    def tearDown(self):
        # Assert real data files were never modified
        if self.real_memories_before is not None:
            self.assertEqual(
                self.real_memories_path.read_text(encoding="utf-8"),
                self.real_memories_before,
                "SAFETY VIOLATION: Real data/memories.json was altered during benchmarking!"
            )
        if self.real_entities_before is not None:
            self.assertEqual(
                self.real_entities_path.read_text(encoding="utf-8"),
                self.real_entities_before,
                "SAFETY VIOLATION: Real data/entities.json was altered during benchmarking!"
            )

    def test_run_benchmark_subset_execution_and_reports(self):
        """
        Verify that run_benchmark executes cleanly across a test subset,
        computes valid aggregate metrics, and produces markdown reports.
        """
        mems, cases = get_benchmark_dataset()
        subset_cases = cases[:5]  # First 5 cases for rapid test execution

        results = run_benchmark(cases=subset_cases, memories=mems, k=3)

        self.assertIn("overall", results)
        self.assertIn("by_category", results)
        self.assertIn("cases", results)

        self.assertEqual(len(results["cases"]), 5)
        self.assertEqual(results["overall"]["total_cases"], 5)

        # Verify Markdown report generation
        md_report = generate_markdown_report(results)
        self.assertIn("# IVARA Milestone 4: Memory QA Benchmark Report", md_report)
        self.assertIn("Baseline A (Raw Vector top-k)", md_report)
        self.assertIn("Baseline B (IVARA Pipeline)", md_report)

    def test_report_file_persistence(self):
        """
        Verify that benchmark JSON and Markdown reports can be written
        to an isolated temporary directory without errors.
        """
        mems, cases = get_benchmark_dataset()
        subset_cases = cases[:3]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            json_out = tmp_path / "test_results.json"
            md_out = tmp_path / "test_report.md"

            results = run_benchmark(cases=subset_cases, memories=mems, k=3)

            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            with open(md_out, "w", encoding="utf-8") as f:
                f.write(generate_markdown_report(results))

            self.assertTrue(json_out.exists())
            self.assertTrue(md_out.exists())
            self.assertGreater(json_out.stat().st_size, 100)
            self.assertGreater(md_out.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
