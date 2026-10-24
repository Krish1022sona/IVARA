"""
Fixed synthetic evaluation dataset for IVARA memory QA benchmarking.
Contains:
  - SYNTHETIC_MEMORIES: An isolated synthetic memory corpus with diverse entities,
    predicates, timestamps, aliases, and distractor topics.
  - BENCHMARK_CASES: 36 annotated question-answering evaluation cases across 7 categories.

NEVER contains or accesses real user data from data/memories.json.
"""

SYNTHETIC_MEMORIES = [
    {
        "id": "m_syn_01",
        "text": "I trained an LSTM neural network using PyTorch for speech recognition in March 2026.",
        "timestamp": "2026-03-15",
        "source": "lab_notes",
        "type": "project_log",
        "importance": 0.9,
        "confidence": 1.0,
        "entities": [
            {"canonical_name": "SELF", "types": ["Person"], "entity_id": "self"},
            {"canonical_name": "PyTorch", "types": ["Framework"], "entity_id": "e_syn_01"},
            {"canonical_name": "LSTM", "types": ["Concept"], "entity_id": "e_syn_02"}
        ],
        "topics": [{"name": "Deep Learning", "confidence": 0.95}],
        "relationships": [
            {
                "source": {"type": "entity", "id": "self"},
                "predicate": "uses",
                "target": {"type": "entity", "id": "e_syn_01"},
                "origin": "explicit",
                "evidence": "using PyTorch",
                "confidence": 1.0
            },
            {
                "source": {"type": "entity", "id": "self"},
                "predicate": "worked_on",
                "target": {"type": "entity", "id": "e_syn_02"},
                "origin": "inferred",
                "evidence": "trained an LSTM neural network",
                "confidence": 0.85
            }
        ]
    },
    {
        "id": "m_syn_02",
        "text": "I collaborated with Rahul on our smart biometric attendance project during the college hackathon.",
        "timestamp": "2026-04-10",
        "source": "hackathon",
        "type": "collaboration",
        "importance": 0.8,
        "confidence": 1.0,
        "entities": [
            {"canonical_name": "SELF", "types": ["Person"], "entity_id": "self"},
            {"canonical_name": "Rahul", "types": ["Person"], "entity_id": "e_syn_03"},
            {"canonical_name": "biometric attendance", "types": ["Project"], "entity_id": "e_syn_04"}
        ],
        "topics": [{"name": "Attendance System", "confidence": 0.85}],
        "relationships": [
            {
                "source": {"type": "entity", "id": "self"},
                "predicate": "worked_with",
                "target": {"type": "entity", "id": "e_syn_03"},
                "origin": "explicit",
                "evidence": "collaborated with Rahul",
                "confidence": 0.95
            },
            {
                "source": {"type": "entity", "id": "self"},
                "predicate": "worked_on",
                "target": {"type": "entity", "id": "e_syn_04"},
                "origin": "explicit",
                "evidence": "biometric attendance project",
                "confidence": 0.90
            }
        ]
    },
    {
        "id": "m_syn_03",
        "text": "Rahul and I submitted our SIH project research paper to IEEE in May 2026.",
        "timestamp": "2026-05-02",
        "source": "academic",
        "type": "publication",
        "importance": 0.85,
        "confidence": 1.0,
        "entities": [
            {"canonical_name": "Rahul", "types": ["Person"], "entity_id": "e_syn_03"},
            {"canonical_name": "SIH project", "types": ["Project"], "entity_id": "e_syn_05"},
            {"canonical_name": "IEEE", "types": ["Organization"], "entity_id": "e_syn_06"}
        ],
        "topics": [{"name": "Research", "confidence": 0.90}],
        "relationships": [
            {
                "source": {"type": "entity", "id": "e_syn_03"},
                "predicate": "worked_on",
                "target": {"type": "entity", "id": "e_syn_05"},
                "origin": "explicit",
                "evidence": "submitted our SIH project research paper",
                "confidence": 0.95
            }
        ]
    },
    {
        "id": "m_syn_04",
        "text": "I started learning the Rust programming language for systems programming and memory safety.",
        "timestamp": "2026-01-20",
        "source": "personal_journal",
        "type": "learning",
        "importance": 0.75,
        "confidence": 1.0,
        "entities": [
            {"canonical_name": "SELF", "types": ["Person"], "entity_id": "self"},
            {"canonical_name": "Rust", "types": ["ProgrammingLanguage"], "entity_id": "e_syn_07"}
        ],
        "topics": [{"name": "Programming Languages", "confidence": 0.9}],
        "relationships": [
            {
                "source": {"type": "entity", "id": "self"},
                "predicate": "uses",
                "target": {"type": "entity", "id": "e_syn_07"},
                "origin": "explicit",
                "evidence": "learning the Rust programming language",
                "confidence": 0.9
            }
        ]
    },
    {
        "id": "m_syn_05",
        "text": "I developed Antigravity (also known as AGY), an open-source autonomous agent framework.",
        "timestamp": "2026-06-15",
        "source": "github",
        "type": "project",
        "importance": 0.95,
        "confidence": 1.0,
        "entities": [
            {"canonical_name": "SELF", "types": ["Person"], "entity_id": "self"},
            {"canonical_name": "Antigravity", "aliases": ["AGY"], "types": ["Project", "Framework"], "entity_id": "e_syn_08"}
        ],
        "topics": [{"name": "AI Agents", "confidence": 0.95}],
        "relationships": [
            {
                "source": {"type": "entity", "id": "self"},
                "predicate": "worked_on",
                "target": {"type": "entity", "id": "e_syn_08"},
                "origin": "explicit",
                "evidence": "developed Antigravity",
                "confidence": 1.0
            }
        ]
    },
    {
        "id": "m_syn_06",
        "text": "My manager Priya assigned me to lead the backend migration to FastAPI in February.",
        "timestamp": "2026-02-14",
        "source": "work_slack",
        "type": "work_assignment",
        "importance": 0.85,
        "confidence": 1.0,
        "entities": [
            {"canonical_name": "SELF", "types": ["Person"], "entity_id": "self"},
            {"canonical_name": "Priya", "types": ["Person"], "entity_id": "e_syn_09"},
            {"canonical_name": "FastAPI", "types": ["Framework"], "entity_id": "e_syn_10"}
        ],
        "topics": [{"name": "Backend Engineering", "confidence": 0.9}],
        "relationships": [
            {
                "source": {"type": "entity", "id": "e_syn_09"},
                "predicate": "worked_with",
                "target": {"type": "entity", "id": "self"},
                "origin": "explicit",
                "evidence": "My manager Priya assigned me",
                "confidence": 0.95
            }
        ]
    },
    {
        "id": "m_syn_07",
        "text": "I deployed a Docker container running PostgreSQL on an AWS EC2 instance.",
        "timestamp": "2026-07-01",
        "source": "devops_log",
        "type": "infrastructure",
        "importance": 0.8,
        "confidence": 1.0,
        "entities": [
            {"canonical_name": "Docker", "types": ["Technology"], "entity_id": "e_syn_11"},
            {"canonical_name": "PostgreSQL", "types": ["Technology"], "entity_id": "e_syn_12"},
            {"canonical_name": "AWS", "types": ["Organization", "Company"], "entity_id": "e_syn_13"}
        ],
        "topics": [{"name": "DevOps", "confidence": 0.9}],
        "relationships": [
            {
                "source": {"type": "entity", "id": "e_syn_11"},
                "predicate": "uses",
                "target": {"type": "entity", "id": "e_syn_13"},
                "origin": "explicit",
                "evidence": "deployed a Docker container running PostgreSQL on an AWS EC2",
                "confidence": 0.9
            }
        ]
    },
    {
        "id": "m_syn_08",
        "text": "I met with Sarah Chen at Blue Bottle Coffee to discuss ontology alignment.",
        "timestamp": "2026-08-11",
        "source": "calendar",
        "type": "meeting",
        "importance": 0.7,
        "confidence": 1.0,
        "entities": [
            {"canonical_name": "SELF", "types": ["Person"], "entity_id": "self"},
            {"canonical_name": "Sarah Chen", "types": ["Person"], "entity_id": "e_syn_14"}
        ],
        "topics": [{"name": "Ontology", "confidence": 0.85}],
        "relationships": [
            {
                "source": {"type": "entity", "id": "self"},
                "predicate": "worked_with",
                "target": {"type": "entity", "id": "e_syn_14"},
                "origin": "explicit",
                "evidence": "met with Sarah Chen",
                "confidence": 0.95
            }
        ]
    },
    {
        "id": "m_syn_09",
        "text": "I gave a keynote presentation on retrieval-augmented generation at PyData Delhi in September.",
        "timestamp": "2026-09-05",
        "source": "conference",
        "type": "presentation",
        "importance": 0.85,
        "confidence": 1.0,
        "entities": [
            {"canonical_name": "SELF", "types": ["Person"], "entity_id": "self"},
            {"canonical_name": "PyData Delhi", "types": ["Organization"], "entity_id": "e_syn_15"}
        ],
        "topics": [{"name": "Conference", "confidence": 0.9}],
        "relationships": []
    },
    {
        "id": "m_syn_10",
        "text": "I adopted a playful golden retriever puppy named Neo in March 2026.",
        "timestamp": "2026-03-01",
        "source": "personal_diary",
        "type": "personal",
        "importance": 0.9,
        "confidence": 1.0,
        "entities": [
            {"canonical_name": "SELF", "types": ["Person"], "entity_id": "self"},
            {"canonical_name": "Neo", "types": ["Concept"], "entity_id": "e_syn_16"}
        ],
        "topics": [{"name": "Pets", "confidence": 0.95}],
        "relationships": []
    },
    {
        "id": "m_syn_11",
        "text": "I configured NeoVim with Lua scripts for my primary code editor environment.",
        "timestamp": "2026-02-28",
        "source": "dotfiles",
        "type": "tooling",
        "importance": 0.7,
        "confidence": 1.0,
        "entities": [
            {"canonical_name": "NeoVim", "types": ["Technology"], "entity_id": "e_syn_17"},
            {"canonical_name": "Lua", "types": ["ProgrammingLanguage"], "entity_id": "e_syn_18"}
        ],
        "topics": [{"name": "Editor Config", "confidence": 0.85}],
        "relationships": [
            {
                "source": {"type": "entity", "id": "e_syn_17"},
                "predicate": "uses",
                "target": {"type": "entity", "id": "e_syn_18"},
                "origin": "explicit",
                "evidence": "configured NeoVim with Lua",
                "confidence": 0.9
            }
        ]
    },
    {
        "id": "m_syn_12",
        "text": "Priya and Rahul both reviewed and approved my pull request for the API rate limiter.",
        "timestamp": "2026-08-20",
        "source": "github",
        "type": "code_review",
        "importance": 0.8,
        "confidence": 1.0,
        "entities": [
            {"canonical_name": "Priya", "types": ["Person"], "entity_id": "e_syn_09"},
            {"canonical_name": "Rahul", "types": ["Person"], "entity_id": "e_syn_03"},
            {"canonical_name": "API rate limiter", "types": ["Project"], "entity_id": "e_syn_19"}
        ],
        "topics": [{"name": "Code Review", "confidence": 0.9}],
        "relationships": []
    }
]


BENCHMARK_CASES = [
    # -------------------------------------------------------------------------
    # Category 1: Direct Factual Recall (6 cases)
    # -------------------------------------------------------------------------
    {
        "case_id": "eval_01",
        "category": "direct_factual_recall",
        "query": "What deep learning framework did I use to train the speech recognition LSTM?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_01"],
        "expected_facts": ["PyTorch", "LSTM", "speech recognition"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_02",
        "category": "direct_factual_recall",
        "query": "Where did I meet with Sarah Chen?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_08"],
        "expected_facts": ["Blue Bottle Coffee", "Sarah Chen"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_03",
        "category": "direct_factual_recall",
        "query": "What programming language did I start learning for systems programming?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_04"],
        "expected_facts": ["Rust", "systems programming"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_04",
        "category": "direct_factual_recall",
        "query": "What cloud provider and compute service did I deploy Docker and PostgreSQL on?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_07"],
        "expected_facts": ["AWS", "EC2", "PostgreSQL"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_05",
        "category": "direct_factual_recall",
        "query": "At what conference did I speak about retrieval-augmented generation?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_09"],
        "expected_facts": ["PyData Delhi", "retrieval-augmented generation"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_06",
        "category": "direct_factual_recall",
        "query": "What breed is my puppy Neo?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_10"],
        "expected_facts": ["golden retriever", "Neo"],
        "expected_evidence_type": "direct"
    },

    # -------------------------------------------------------------------------
    # Category 2: Entity & Alias Resolution (5 cases)
    # -------------------------------------------------------------------------
    {
        "case_id": "eval_07",
        "category": "entity_alias_resolution",
        "query": "What is AGY and what type of system is it?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_05"],
        "expected_facts": ["Antigravity", "agent framework"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_08",
        "category": "entity_alias_resolution",
        "query": "Did Sarah and I discuss ontology alignment?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_08"],
        "expected_facts": ["Sarah Chen", "ontology alignment"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_09",
        "category": "entity_alias_resolution",
        "query": "Which code editor did I configure using Lua scripts?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_11"],
        "expected_facts": ["NeoVim", "Lua"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_10",
        "category": "entity_alias_resolution",
        "query": "Tell me about my pet Neo.",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_10"],
        "expected_facts": ["golden retriever", "puppy"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_11",
        "category": "entity_alias_resolution",
        "query": "What framework did my manager Priya assign me to migrate our backend to?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_06"],
        "expected_facts": ["FastAPI", "Priya"],
        "expected_evidence_type": "direct"
    },

    # -------------------------------------------------------------------------
    # Category 3: Relationship-Based Questions (5 cases)
    # -------------------------------------------------------------------------
    {
        "case_id": "eval_12",
        "category": "relationship_based",
        "query": "Who worked with me on the smart biometric attendance project?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_02"],
        "expected_facts": ["Rahul", "biometric attendance"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_13",
        "category": "relationship_based",
        "query": "Who is my manager who assigned the backend migration?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_06"],
        "expected_facts": ["Priya"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_14",
        "category": "relationship_based",
        "query": "What database was containerized with Docker on AWS?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_07"],
        "expected_facts": ["PostgreSQL"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_15",
        "category": "relationship_based",
        "query": "What project paper did Rahul and I submit to IEEE?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_03"],
        "expected_facts": ["SIH project", "IEEE"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_16",
        "category": "relationship_based",
        "query": "What neural network model did I work on using PyTorch?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_01"],
        "expected_facts": ["LSTM"],
        "expected_evidence_type": "inferred"
    },

    # -------------------------------------------------------------------------
    # Category 4: Multi-Memory Evidence (5 cases)
    # -------------------------------------------------------------------------
    {
        "case_id": "eval_17",
        "category": "multi_memory_evidence",
        "query": "Which projects and submissions did Rahul collaborate with me on?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_02", "m_syn_03", "m_syn_12"],
        "expected_facts": ["biometric attendance", "SIH project", "API rate limiter"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_18",
        "category": "multi_memory_evidence",
        "query": "Who reviewed my pull request for the API rate limiter, and what other work do I have with them?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_06", "m_syn_12"],
        "expected_facts": ["Priya", "Rahul", "FastAPI"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_19",
        "category": "multi_memory_evidence",
        "query": "What programming languages have I used or configured in 2026?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_04", "m_syn_11"],
        "expected_facts": ["Rust", "Lua"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_20",
        "category": "multi_memory_evidence",
        "query": "Did Priya work with me on both FastAPI and pull request reviews?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_06", "m_syn_12"],
        "expected_facts": ["FastAPI", "pull request", "rate limiter"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_21",
        "category": "multi_memory_evidence",
        "query": "What collaborative milestones did Rahul and I achieve across hackathons and academic research?",
        "expected_answerable": "answered",
        "supporting_memory_ids": ["m_syn_02", "m_syn_03"],
        "expected_facts": ["hackathon", "IEEE", "SIH project"],
        "expected_evidence_type": "direct"
    },

    # -------------------------------------------------------------------------
    # Category 5: Partial Supporting Evidence (5 cases)
    # -------------------------------------------------------------------------
    {
        "case_id": "eval_22",
        "category": "partial_supporting_evidence",
        "query": "What deep learning model did I train with PyTorch, and what was its word error rate?",
        "expected_answerable": "partial",
        "supporting_memory_ids": ["m_syn_01"],
        "expected_facts": ["LSTM", "PyTorch"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_23",
        "category": "partial_supporting_evidence",
        "query": "What did I discuss with Sarah Chen, and when is our next follow-up meeting scheduled?",
        "expected_answerable": "partial",
        "supporting_memory_ids": ["m_syn_08"],
        "expected_facts": ["Sarah Chen", "ontology alignment"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_24",
        "category": "partial_supporting_evidence",
        "query": "What puppy did I adopt, and what was his exact weight at adoption?",
        "expected_answerable": "partial",
        "supporting_memory_ids": ["m_syn_10"],
        "expected_facts": ["golden retriever", "Neo"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_25",
        "category": "partial_supporting_evidence",
        "query": "When did I deploy PostgreSQL on AWS, and what EC2 instance size was chosen?",
        "expected_answerable": "partial",
        "supporting_memory_ids": ["m_syn_07"],
        "expected_facts": ["AWS", "PostgreSQL", "July 2026"],
        "expected_evidence_type": "direct"
    },
    {
        "case_id": "eval_26",
        "category": "partial_supporting_evidence",
        "query": "What scripting language did I configure for NeoVim, and what custom keybindings did I add?",
        "expected_answerable": "partial",
        "supporting_memory_ids": ["m_syn_11"],
        "expected_facts": ["Lua", "NeoVim"],
        "expected_evidence_type": "direct"
    },

    # -------------------------------------------------------------------------
    # Category 6: Unanswerable Questions / Abstention Required (5 cases)
    # -------------------------------------------------------------------------
    {
        "case_id": "eval_27",
        "category": "unanswerable_abstention",
        "query": "What video games do I like playing on weekend evenings?",
        "expected_answerable": "insufficient_evidence",
        "supporting_memory_ids": [],
        "expected_facts": [],
        "expected_evidence_type": "none"
    },
    {
        "case_id": "eval_28",
        "category": "unanswerable_abstention",
        "query": "What is my mother's maiden name?",
        "expected_answerable": "insufficient_evidence",
        "supporting_memory_ids": [],
        "expected_facts": [],
        "expected_evidence_type": "none"
    },
    {
        "case_id": "eval_29",
        "category": "unanswerable_abstention",
        "query": "Where did I travel for my summer vacation in 2025?",
        "expected_answerable": "insufficient_evidence",
        "supporting_memory_ids": [],
        "expected_facts": [],
        "expected_evidence_type": "none"
    },
    {
        "case_id": "eval_30",
        "category": "unanswerable_abstention",
        "query": "What is my home Wi-Fi network password?",
        "expected_answerable": "insufficient_evidence",
        "supporting_memory_ids": [],
        "expected_facts": [],
        "expected_evidence_type": "none"
    },
    {
        "case_id": "eval_31",
        "category": "unanswerable_abstention",
        "query": "What model of car do I drive to work?",
        "expected_answerable": "insufficient_evidence",
        "supporting_memory_ids": [],
        "expected_facts": [],
        "expected_evidence_type": "none"
    },

    # -------------------------------------------------------------------------
    # Category 7: Similar But Irrelevant Distractors (5 cases)
    # -------------------------------------------------------------------------
    {
        "case_id": "eval_32",
        "category": "similar_irrelevant_distractor",
        "query": "What hardware robotics projects have I built with microcontrollers?",
        "expected_answerable": "insufficient_evidence",
        "supporting_memory_ids": [],
        "expected_facts": [],
        "expected_evidence_type": "none"
    },
    {
        "case_id": "eval_33",
        "category": "similar_irrelevant_distractor",
        "query": "What computer vision YOLO models have I trained for object detection?",
        "expected_answerable": "insufficient_evidence",
        "supporting_memory_ids": [],
        "expected_facts": [],
        "expected_evidence_type": "none"
    },
    {
        "case_id": "eval_34",
        "category": "similar_irrelevant_distractor",
        "query": "Did I configure Vim plugins for Django web development?",
        "expected_answerable": "insufficient_evidence",
        "supporting_memory_ids": [],
        "expected_facts": [],
        "expected_evidence_type": "none"
    },
    {
        "case_id": "eval_35",
        "category": "similar_irrelevant_distractor",
        "query": "What training regimen did I follow for dog agility competitions with Neo?",
        "expected_answerable": "insufficient_evidence",
        "supporting_memory_ids": [],
        "expected_facts": [],
        "expected_evidence_type": "none"
    },
    {
        "case_id": "eval_36",
        "category": "similar_irrelevant_distractor",
        "query": "Which AWS Lambda serverless functions did I write in Python?",
        "expected_answerable": "insufficient_evidence",
        "supporting_memory_ids": [],
        "expected_facts": [],
        "expected_evidence_type": "none"
    }
]


def get_benchmark_dataset():
    """Return a deep copy of synthetic memories and benchmark cases."""
    import copy
    return copy.deepcopy(SYNTHETIC_MEMORIES), copy.deepcopy(BENCHMARK_CASES)
