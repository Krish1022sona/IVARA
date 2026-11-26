#!/usr/bin/env python3
"""Run an interactive IVARA chat session in an isolated temporary sandbox.

Ensures that production memory files (data/memories.json), conversation
databases (data/conversations.db), and graph stores (data/memory_graph.db)
are never modified during manual testing and verification.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import run_sandbox_chat


def main():
    conv_id = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    run_sandbox_chat(conv_id=conv_id)


if __name__ == "__main__":
    main()
