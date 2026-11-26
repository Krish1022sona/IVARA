# IVARA — Agent & Developer Operating Rules

Follow these rules in every development session:

1. **Read core docs first:** Read `README.md`, `ARCHITECTURE.md`, `DEVELOPMENT_LOG.md`, and `ROADMAP.md` at the start of each session.
2. **Verify against code:** Inspect actual source files before trusting documentation or comments.
3. **Stay in scope:** Implement only the approved phase and scope. Do not introduce unrequested features or refactors.
4. **Keep it maintainable:** Keep architecture and changes clear enough for the owner to navigate, explain, and maintain.
5. **Test behavior, not mocks:** Write tests that verify real system behavior rather than tautological mock outputs; keep the unit test suite 100% offline-safe.
6. **Update existing docs:** Update `README.md`, `ARCHITECTURE.md`, `DEVELOPMENT_LOG.md`, and `ROADMAP.md` rather than creating new standalone phase reports.
7. **Log chronologically:** Add one concise entry to `DEVELOPMENT_LOG.md` per completed phase/milestone.
8. **Track live state:** Update `ROADMAP.md` to reflect completed work, discovered bugs, and next steps.
9. **Clean temp artifacts:** Delete regenerable temporary artifacts once their purpose is fulfilled.
10. **Protect evidence & data:** Preserve non-regenerable research evidence, sealed benchmark runs, and local user data.
11. **No announcement files:** Never create `FINAL_REPORT`, `PHASE_COMPLETE`, `IMPLEMENTATION_SUMMARY`, or similar permanent Markdown files merely to announce completion.
12. **Report concisely in chat:** Provide completion summaries directly in the conversation.
13. **Verify benchmark claims:** Never alter benchmark numbers or claims without recalculating from raw prediction/judge artifacts.
14. **Protect privacy:** Never commit secrets, `.env` files, `data/memories.json`, or personal SQLite databases (`*.db`).
