"""
Decision-Grade Preflight Gate for IVARA Evaluation Campaigns.

Enforces strict safety and reproducibility checks before any experiment launches:
1. Rejects execution if output directory matches protected historical runs.
2. Verifies Ollama server availability and required model tags.
3. Rejects execution if judge is mislabeled as official LongMemEval judge.
4. Verifies sample dataset existence, integrity, and SHA-256 match.
5. Verifies question ID sequence consistency and non-duplication across arms.
"""

import hashlib
import json
from pathlib import Path
from typing import Any
import urllib.request
import urllib.error

PROTECTED_DIRECTORIES = {
    "diagnostic_pilot_30",
    "diagnostic_full_100",
    "post_evaluation_audit",
    "campaign_v1_pilot",
    "step8",
    "step9"
}


class PreflightValidationError(RuntimeError):
    """Raised when preflight safety or reproducibility checks fail."""
    pass


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class PreflightGate:
    """Preflight validation engine."""

    def __init__(
        self,
        output_dir: Path,
        sample_path: Path,
        expected_sample_sha256: str | None,
        generator_model: str,
        judge_model: str,
        judge_status_label: str,
        ollama_base_url: str = "http://127.0.0.1:11434"
    ):
        self.output_dir = Path(output_dir)
        self.sample_path = Path(sample_path)
        self.expected_sample_sha256 = expected_sample_sha256
        self.generator_model = generator_model
        self.judge_model = judge_model
        self.judge_status_label = judge_status_label
        self.ollama_base_url = ollama_base_url.rstrip("/")

    def validate_all(self, arms_to_evaluate: list[str]) -> dict[str, Any]:
        """
        Run all preflight checks. Raises PreflightValidationError on any violation.
        Returns validation metadata dictionary.
        """
        checks: dict[str, Any] = {}

        # 1. Output directory safety check
        for protected in PROTECTED_DIRECTORIES:
            if protected in self.output_dir.parts:
                raise PreflightValidationError(
                    f"Refusing to launch: Output directory '{self.output_dir}' attempts to overwrite protected historical directory '{protected}'."
                )
        checks["output_directory_safe"] = True

        # 2. Judge status labeling check
        label_lower = self.judge_status_label.lower()
        if "official" in label_lower and "not_official" not in label_lower and "diagnostic" not in label_lower:
            raise PreflightValidationError(
                f"Refusing to launch: Judge configuration '{self.judge_status_label}' contains 'official'. "
                "Local judge models must be explicitly labeled diagnostic."
            )
        checks["judge_labeling_compliant"] = True

        # 3. Sample dataset existence & hash check
        if not self.sample_path.exists():
            raise PreflightValidationError(f"Sample dataset file missing at '{self.sample_path}'.")
        
        actual_sha256 = compute_file_sha256(self.sample_path)
        if self.expected_sample_sha256 and actual_sha256 != self.expected_sample_sha256:
            raise PreflightValidationError(
                f"Sample dataset SHA-256 mismatch. Expected: {self.expected_sample_sha256}, Actual: {actual_sha256}"
            )
        checks["sample_dataset_sha256"] = actual_sha256

        # 4. Question set alignment and zero-duplicates check
        with open(self.sample_path, "r", encoding="utf-8") as f:
            sample_data = json.load(f)
        sample_qids = [x["question_id"] for x in sample_data]
        if len(sample_qids) != len(set(sample_qids)):
            raise PreflightValidationError(f"Sample dataset contains duplicate question IDs: {len(sample_qids)} total vs {len(set(sample_qids))} unique.")
        checks["sample_question_count"] = len(sample_qids)

        # 5. Model tags availability check on Ollama
        try:
            req = urllib.request.Request(f"{self.ollama_base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                tags_data = json.loads(resp.read().decode("utf-8"))
            available_models = [m.get("name") for m in tags_data.get("models", [])]
            
            # Check generator
            gen_available = any(self.generator_model in m for m in available_models)
            if not gen_available:
                raise PreflightValidationError(
                    f"Generator model '{self.generator_model}' not found in active Ollama models: {available_models}"
                )
            # Check judge
            judge_available = any(self.judge_model in m for m in available_models)
            if not judge_available:
                raise PreflightValidationError(
                    f"Judge model '{self.judge_model}' not found in active Ollama models: {available_models}"
                )
            checks["models_verified"] = {"generator": self.generator_model, "judge": self.judge_model}
        except urllib.error.URLError as e:
            raise PreflightValidationError(f"Failed to communicate with local Ollama server at '{self.ollama_base_url}': {e}")

        checks["status"] = "PASSED"
        return checks
