"""Expanded Narrative Reasoning Benchmark Dataset (280 Questions across 18 Sessions).

Split into:
- 30 Dev questions
- 250 Test questions
"""

import json
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
STORY_DATA_PATH = BENCHMARK_DIR / "story_data_expanded.json"
GROUND_TRUTH_PATH = BENCHMARK_DIR / "ground_truth_expanded.json"

with open(STORY_DATA_PATH, "r", encoding="utf-8") as f:
    SESSIONS = json.load(f)

with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

