# Security Policy — IVARA

The IVARA development team takes the security, privacy, and integrity of personal memory data seriously. This document outlines our security policies, vulnerability reporting procedure, and data privacy architecture.

---

## 1. Supported Versions

Security updates and patches are applied to the active release branches:

| Version | Supported | Notes |
|:---:|:---:|:---|
| `v1.0.x` | Yes | Active production and evaluation release |
| `< 1.0.0` | No | Experimental milestones and pre-release iterations |

---

## 2. Privacy & Data Architecture Guarantees

IVARA is architected specifically as a **privacy-first, local-execution personal memory system**:

1. **Zero Cloud Exfiltration:**
   - IVARA executes entirely on local infrastructure.
   - Embeddings are generated locally using `sentence-transformers` (`all-MiniLM-L6-v2`).
   - Generative inference runs through local Ollama instances (`http://127.0.0.1:11434`).
   - No conversational turns, memories, entities, or queries are transmitted to third-party cloud APIs.

2. **Local Storage Isolation:**
   - Episodic conversations and graph structures are stored in local SQLite databases (`conversations.db`, `memory_graph.db`).
   - By default, all database files (`*.db`, `*.sqlite`, `*.sqlite3`) are excluded from version control via `.gitignore`.
   - Never commit database files containing private personal interactions.

3. **Deterministic Abstention & Hallucination Mitigation:**
   - The answering pipeline includes strict citation reconciliation. If ungrounded assertions are generated without verified citations in retrieved context, the response is explicitly downgraded or marked `insufficient_evidence`.

---

## 3. Reporting a Vulnerability

If you discover a security vulnerability, privacy leak, or credential exposure within IVARA, please report it responsibly:

- **Do NOT create a public GitHub issue.**
- Submit your finding directly via email to the project maintainers or via GitHub Private Vulnerability Reporting on the repository page.
- Please include:
  1. A detailed description of the vulnerability.
  2. Steps to reproduce or proof-of-concept script.
  3. Affected versions, modules, or configurations.
  4. Any proposed remediations.

### Response Timeline
- **Initial Acknowledgement:** Within 48 hours.
- **Vulnerability Assessment & Triage:** Within 5 business days.
- **Fix and Release Advisory:** Prioritized based on severity (Critical/High: within 14 days).

---

## 4. Best Practices for Deployers

When deploying IVARA in your environment:
- Ensure the local SQLite database directory permissions restrict unauthorized user read access (`chmod 600` on Unix systems).
- Keep your local Ollama daemon bound strictly to localhost (`127.0.0.1:11434`) unless placed behind an authenticated reverse proxy.
- Maintain a secure backup of your `data/` folder before running experimental migrations.
