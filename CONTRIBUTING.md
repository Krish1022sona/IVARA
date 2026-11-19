# Contributing to IVARA

Thank you for your interest in contributing to **IVARA** (Intelligent Verifiable Agentic Retrieval Architecture). We welcome contributions from researchers, software engineers, and open-source contributors.

---

## 1. Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for everyone. Please be respectful, constructive, and collaborative in all issues, pull requests, and discussions.

---

## 2. Getting Started

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git
- Optional: Local [Ollama](https://ollama.ai) instance for live model inference

### Setting Up a Development Environment
1. Fork the repository on GitHub and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/IVARA.git
   cd IVARA
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
   ```
3. Install development dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```
4. Verify the test suite passes:
   ```bash
   pytest -q
   ```
   *All 525 unit and regression tests should pass.*

---

## 3. Development Workflow

1. **Branching Strategy:**
   - Always create a dedicated feature or bugfix branch from `main`:
     ```bash
     git checkout -b feature/your-feature-name
     ```
   - Never commit directly to `main` or release branches.
   - Do not modify or move frozen tags (e.g., `v1.0.0-freeze`).

2. **Coding Standards:**
   - Follow PEP 8 guidelines for Python code formatting.
   - Use type annotations for public function arguments and return types.
   - Ensure docstrings explain complex retrieval, scoring, or ontological logic.
   - Maintain strict separation of concerns across `src/chunking/`, `src/temporal/`, `src/ontology/`, and `src/answering.py`.

3. **Privacy & Security Constraints:**
   - Never commit API keys, personal credentials, private conversation databases (`*.db`), or personal data.
   - Do not hardcode machine-specific absolute paths.
   - All tests must run offline without requiring external network calls or cloud API keys.

---

## 4. Testing & Verification Requirements

Every pull request must satisfy the following criteria:

1. **Full Test Suite Passing:**
   ```bash
   pytest
   ```
   *No regressions: 100% of existing tests must pass.*

2. **New Tests for New Functionality:**
   - Add unit tests under `tests/` covering new algorithms, retrieval mechanisms, or edge cases.
   - Ensure deterministic behavior by mocking or controlling random seeds.

3. **Demonstration Integrity:**
   - Verify that the deterministic offline demo remains fully functional:
     ```bash
     python demo.py
     ```

4. **Benchmark & Research Integrity:**
   - Do not alter sealed benchmark outputs or evaluation scripts without documented justification.
   - Any comparative evaluation claims must be grounded in controlled, reproducible experiments.

---

## 5. Submitting a Pull Request

1. Push your branch to your GitHub fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a Pull Request against the `main` branch.
3. Fill out the Pull Request template completely, detailing:
   - Summary of changes
   - Motivation and context
   - Verification commands executed
   - Test results
4. A maintainer will review your PR. Address any requested changes promptly.

---

## 6. Reporting Issues

- **Bug Reports:** Open an issue using the Bug Report template. Include your operating system, Python version, steps to reproduce, and terminal traceback.
- **Feature Requests:** Open an issue using the Feature Request template explaining the use case and proposed architectural approach.
- **Security Vulnerabilities:** Please review our [Security Policy](SECURITY.md) and do NOT open public issues for sensitive security vulnerabilities.
