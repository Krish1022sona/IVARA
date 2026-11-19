# IVARA — Installation & Setup Guide
## Reproducible Local Environment Configuration

This guide provides step-by-step instructions for installing and running IVARA on Linux, macOS, or Windows.

---

### System Requirements

- **Operating System:** Ubuntu 20.04+, macOS 12+ (Apple Silicon or Intel), or Windows 10/11.
- **Python Version:** Python 3.10, 3.11, 3.12, or 3.13.
- **Hardware:**
  - *Minimum:* 8 GB RAM, 4 CPU cores (runs deterministic demo and CPU sentence embeddings).
  - *Recommended for Live Local LLM:* 16 GB RAM + modern GPU (NVIDIA 6GB+ VRAM or Apple Silicon unified memory).
- **Storage:** ~3 GB free disk space (includes virtual environment, models, and embeddings cache).

---

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/IVARA.git
cd IVARA
```

---

### 2. Python Virtual Environment Setup

#### On Linux / macOS (Bash / Zsh):
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

#### On Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv .venv

# Set execution policy if script activation is restricted
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

---

### 3. Install Python Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install runtime dependencies
pip install -r requirements.txt

# (Optional) Install development and evaluation dependencies
pip install -r requirements-dev.txt
```

#### Core Runtime Dependencies:
- `faiss-cpu`: Vector index search for dense retrieval.
- `sentence-transformers`: Local transformer sentence embeddings (`all-MiniLM-L6-v2`).
- `numpy`: Fast matrix and vector distance computations.
- `pydantic`: Robust data schemas and JSON serialization.
- `ollama`: Local LLM API client for private, offline generation.
- `rich`: Clean terminal user interface rendering.

---

### 4. Optional: Local LLM Setup (for Live Generative Mode)

IVARA uses local Ollama instances to guarantee complete data privacy:

1. **Install Ollama:**
   - Download and install from [ollama.ai](https://ollama.ai).
2. **Start the Ollama Server:**
   ```bash
   ollama serve
   ```
3. **Pull the Recommended Models:**
   ```bash
   # Production answering model (default):
   ollama pull gpt-oss

   # Evaluation and benchmark judge model:
   ollama pull qwen2.5:14b
   ```

*Note: For demonstrations and tests without Ollama, IVARA provides a 100% deterministic offline engine (`python demo.py`).*

---

### 5. Pre-Flight Verification Checks

To verify that your installation is complete and healthy:

#### Step 1: Run Full Test Suite
```bash
pytest -q
```
*Expected output:* `525 passed in ~35-45 seconds`.

#### Step 2: Run Offline Demonstration
```bash
python demo.py
```
*Expected output:* All 5 demonstration scenarios execute successfully with verified citation provenance and honest abstention.

#### Step 3: Run Interactive CLI Smoke Test
```bash
python main.py ask "What company does Arjun work for?" --adaptive
```

---

### 6. Troubleshooting Common Installation Issues

#### Issue 1: `faiss` Installation Failure
- *Symptom:* `ModuleNotFoundError: No module named 'faiss'`
- *Solution:* Ensure you installed `faiss-cpu` rather than `faiss`:
  ```bash
  pip install faiss-cpu
  ```

#### Issue 2: SQLite FTS5 Module Missing (rare on custom Linux builds)
- *Symptom:* `sqlite3.OperationalError: no such module: fts5`
- *Solution:* Ensure SQLite3 was compiled with FTS5. On Ubuntu/Debian:
  ```bash
  sudo apt-get install libsqlite3-dev
  ```

#### Issue 3: SentenceTransformers First-Run Model Download
- On initial startup, `sentence-transformers` caches `all-MiniLM-L6-v2` (~90 MB) locally in `~/.cache/huggingface/`. Ensure an active internet connection on the initial run, or copy the cache directory into an air-gapped machine.

#### Issue 4: Ollama Port Conflict
- If port `11434` is already in use or unreachable, verify server status with:
  ```bash
  curl http://127.0.0.1:11434/api/tags
  ```
