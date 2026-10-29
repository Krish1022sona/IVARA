"""
Unit tests for the Decision-Grade Preflight Gate.
"""

from pathlib import Path
import pytest
from evaluation.autonomous_sprint.preflight import PreflightGate, PreflightValidationError


def test_preflight_rejects_protected_directory(tmp_path):
    """Preflight must reject launching in protected historical directories."""
    gate = PreflightGate(
        output_dir=Path("evaluation/step8"),
        sample_path=tmp_path / "sample.json",
        expected_sample_sha256=None,
        generator_model="gpt-oss:latest",
        judge_model="VladimirGav/gemma4-26b-16GB-VRAM:latest",
        judge_status_label="diagnostic_local"
    )
    with pytest.raises(PreflightValidationError, match="protected historical directory"):
        gate.validate_all(["ARM_1"])


def test_preflight_rejects_official_label_for_local_judge(tmp_path):
    """Preflight must reject labeling local judge as official."""
    gate = PreflightGate(
        output_dir=tmp_path / "valid_run",
        sample_path=tmp_path / "sample.json",
        expected_sample_sha256=None,
        generator_model="gpt-oss:latest",
        judge_model="VladimirGav/gemma4-26b-16GB-VRAM:latest",
        judge_status_label="official_longmemeval_benchmark"
    )
    with pytest.raises(PreflightValidationError, match="Local judge models must be explicitly labeled diagnostic"):
        gate.validate_all(["ARM_1"])


def test_preflight_rejects_missing_sample(tmp_path):
    """Preflight must reject execution if sample file is missing."""
    gate = PreflightGate(
        output_dir=tmp_path / "valid_run",
        sample_path=tmp_path / "missing_sample.json",
        expected_sample_sha256=None,
        generator_model="gpt-oss:latest",
        judge_model="VladimirGav/gemma4-26b-16GB-VRAM:latest",
        judge_status_label="diagnostic_local"
    )
    with pytest.raises(PreflightValidationError, match="Sample dataset file missing"):
        gate.validate_all(["ARM_1"])
