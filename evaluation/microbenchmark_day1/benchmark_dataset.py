"""
Day 1 Micro-Benchmark Dataset: Ground Truth Specification
Isolated test dataset with manually verified ground truth for:
  - Part A: 10 Knowledge Extraction turns
  - Part B: 10 Grounded Answering queries
"""

from typing import Any

# ============================================================================
# PART A: 10 KNOWLEDGE EXTRACTION PASSAGES & GROUND TRUTH
# ============================================================================

EXTRACTION_DATASET = [
    {
        "id": "ext_01",
        "description": "Multi-agent team kickoff with compound mentions",
        "text": (
            "On January 14, 2025, I kicked off Project Chronos at Aetheria Labs. "
            "Marcus Vance and Sarah Lin lead our distributed storage engineering, "
            "while Priya Sharma develops custom temporal attention kernels. "
            "Chloe Bennett serves as our technical product manager."
        ),
        "gold_entities": [
            {"name": "SELF", "type": "Person"},
            {"name": "Project Chronos", "type": "Project"},
            {"name": "Aetheria Labs", "type": "Organization"},
            {"name": "Marcus Vance", "type": "Person"},
            {"name": "Sarah Lin", "type": "Person"},
            {"name": "Priya Sharma", "type": "Person"},
            {"name": "Chloe Bennett", "type": "Person"}
        ],
        "compound_traps": ["Marcus Vance and Sarah Lin"],
        "gold_relationships": [
            ("SELF", "kicked off", "Project Chronos"),
            ("Project Chronos", "organization", "Aetheria Labs"),
            ("Marcus Vance", "leads", "distributed storage engineering"),
            ("Sarah Lin", "leads", "distributed storage engineering"),
            ("Priya Sharma", "develops", "temporal attention kernels"),
            ("Chloe Bennett", "role", "technical product manager")
        ],
        "gold_temporal": [
            {"claim": "kicked off Project Chronos", "date": "2025-01-14"}
        ]
    },
    {
        "id": "ext_02",
        "description": "Dual-company partnership with distinct same-first-name entities",
        "text": (
            "Julian Mercer, Director of Operations at Vanguard Robotics, visited Aetheria Labs to establish Project Aegis. "
            "He introduced Elena Ramos, their lead hardware engineer. She is completely distinct from Dr. Elena Rostova, "
            "who is our Chief Scientist at Aetheria Labs."
        ),
        "gold_entities": [
            {"name": "Julian Mercer", "type": "Person"},
            {"name": "Vanguard Robotics", "type": "Organization"},
            {"name": "Aetheria Labs", "type": "Organization"},
            {"name": "Project Aegis", "type": "Project"},
            {"name": "Elena Ramos", "type": "Person"},
            {"name": "Dr. Elena Rostova", "type": "Person"}
        ],
        "compound_traps": ["Elena Ramos and Dr. Elena Rostova", "Julian Mercer and Elena Ramos"],
        "gold_relationships": [
            ("Julian Mercer", "role", "Director of Operations"),
            ("Julian Mercer", "employed_at", "Vanguard Robotics"),
            ("Elena Ramos", "employed_at", "Vanguard Robotics"),
            ("Elena Ramos", "role", "lead hardware engineer"),
            ("Dr. Elena Rostova", "employed_at", "Aetheria Labs"),
            ("Dr. Elena Rostova", "role", "Chief Scientist")
        ],
        "gold_temporal": []
    },
    {
        "id": "ext_03",
        "description": "Career timeline with explicit temporal transition",
        "text": (
            "From June 2021 until January 2024, I worked as a Senior Data Engineer at Datacore. "
            "In March 2024, I joined CloudMatrix Inc as a Lead Distributed Systems Architect."
        ),
        "gold_entities": [
            {"name": "SELF", "type": "Person"},
            {"name": "Datacore", "type": "Organization"},
            {"name": "CloudMatrix Inc", "type": "Organization"}
        ],
        "compound_traps": [],
        "gold_relationships": [
            ("SELF", "worked_as", "Senior Data Engineer"),
            ("SELF", "employed_at", "Datacore"),
            ("SELF", "joined", "CloudMatrix Inc"),
            ("SELF", "role", "Lead Distributed Systems Architect")
        ],
        "gold_temporal": [
            {"claim": "employed at Datacore", "valid_from": "2021-06", "valid_to": "2024-01"},
            {"claim": "employed at CloudMatrix Inc", "valid_from": "2024-03", "valid_to": None}
        ]
    },
    {
        "id": "ext_04",
        "description": "Hardware preference shift with outdated vs current state",
        "text": (
            "I used to love clicky mechanical keyboards with Cherry MX Blue switches back in 2020. "
            "However, in October 2025 I switched to an Apple Magic Keyboard with scissor switches, "
            "and I currently use it for all my daily coding."
        ),
        "gold_entities": [
            {"name": "SELF", "type": "Person"},
            {"name": "Cherry MX Blue mechanical keyboard", "type": "Product"},
            {"name": "Apple Magic Keyboard", "type": "Product"}
        ],
        "compound_traps": [],
        "gold_relationships": [
            ("SELF", "previously_used", "Cherry MX Blue mechanical keyboard"),
            ("SELF", "currently_uses", "Apple Magic Keyboard")
        ],
        "gold_temporal": [
            {"claim": "uses Cherry MX Blue", "valid_from": "2020", "valid_to": "2025-10"},
            {"claim": "uses Apple Magic Keyboard", "valid_from": "2025-10", "valid_to": None}
        ]
    },
    {
        "id": "ext_05",
        "description": "Multi-hop project architecture and multi-technology stack",
        "text": (
            "David Thorne manages the VectorFlow engine at CloudMatrix Inc. "
            "The system is implemented in Rust and Neo4j, and is deployed across AWS Kubernetes clusters."
        ),
        "gold_entities": [
            {"name": "David Thorne", "type": "Person"},
            {"name": "VectorFlow", "type": "Project"},
            {"name": "CloudMatrix Inc", "type": "Organization"},
            {"name": "Rust", "type": "Technology"},
            {"name": "Neo4j", "type": "Technology"},
            {"name": "AWS Kubernetes", "type": "Technology"}
        ],
        "compound_traps": ["Rust and Neo4j"],
        "gold_relationships": [
            ("David Thorne", "manages", "VectorFlow"),
            ("VectorFlow", "organization", "CloudMatrix Inc"),
            ("VectorFlow", "written_in", "Rust"),
            ("VectorFlow", "uses_database", "Neo4j"),
            ("VectorFlow", "deployed_on", "AWS Kubernetes")
        ],
        "gold_temporal": []
    },
    {
        "id": "ext_06",
        "description": "Personal and familial biographical attributes",
        "text": (
            "My sister Priya Mehta graduated from Stanford University in June 2022. "
            "She currently lives in Seattle with her partner Kenji Sato, who works as an architect."
        ),
        "gold_entities": [
            {"name": "SELF", "type": "Person"},
            {"name": "Priya Mehta", "type": "Person"},
            {"name": "Stanford University", "type": "Organization"},
            {"name": "Seattle", "type": "Location"},
            {"name": "Kenji Sato", "type": "Person"}
        ],
        "compound_traps": ["Priya Mehta and Kenji Sato"],
        "gold_relationships": [
            ("Priya Mehta", "sister_of", "SELF"),
            ("Priya Mehta", "graduated_from", "Stanford University"),
            ("Priya Mehta", "lives_in", "Seattle"),
            ("Priya Mehta", "partner_of", "Kenji Sato"),
            ("Kenji Sato", "role", "architect")
        ],
        "gold_temporal": [
            {"claim": "graduated Stanford", "date": "2022-06"}
        ]
    },
    {
        "id": "ext_07",
        "description": "Relocation and historical residential state",
        "text": (
            "I lived in an apartment in Indiranagar, Bangalore, for three years. "
            "On November 15, 2024, I moved to a duplex in HSR Layout, where I currently reside."
        ),
        "gold_entities": [
            {"name": "SELF", "type": "Person"},
            {"name": "Indiranagar", "type": "Location"},
            {"name": "Bangalore", "type": "Location"},
            {"name": "HSR Layout", "type": "Location"}
        ],
        "compound_traps": [],
        "gold_relationships": [
            ("SELF", "previously_lived_in", "Indiranagar"),
            ("Indiranagar", "located_in", "Bangalore"),
            ("SELF", "currently_lives_in", "HSR Layout"),
            ("HSR Layout", "located_in", "Bangalore")
        ],
        "gold_temporal": [
            {"claim": "resided in Indiranagar", "valid_to": "2024-11-15"},
            {"claim": "resides in HSR Layout", "valid_from": "2024-11-15", "valid_to": None}
        ]
    },
    {
        "id": "ext_08",
        "description": "Project cancellation and negative assertion",
        "text": (
            "Executive leadership officially abandoned Project Hyperion on June 12, 2025. "
            "Neither Marcus Vance nor Priya Sharma ever wrote any production code for it."
        ),
        "gold_entities": [
            {"name": "Project Hyperion", "type": "Project"},
            {"name": "Marcus Vance", "type": "Person"},
            {"name": "Priya Sharma", "type": "Person"}
        ],
        "compound_traps": ["Marcus Vance nor Priya Sharma", "Marcus Vance and Priya Sharma"],
        "gold_relationships": [
            ("Project Hyperion", "status", "abandoned")
        ],
        "gold_temporal": [
            {"claim": "abandoned Project Hyperion", "date": "2025-06-12"}
        ]
    },
    {
        "id": "ext_09",
        "description": "Vehicle purchase and sporting achievement",
        "text": (
            "On May 10, 2024, I purchased a 2024 Honda Civic in Aegean Blue from Metro Honda. "
            "Later that year on October 20, I completed the Austin Half Marathon in 1 hour 52 minutes."
        ),
        "gold_entities": [
            {"name": "SELF", "type": "Person"},
            {"name": "2024 Honda Civic", "type": "Product"},
            {"name": "Metro Honda", "type": "Organization"},
            {"name": "Austin Half Marathon", "type": "Event"}
        ],
        "compound_traps": [],
        "gold_relationships": [
            ("SELF", "purchased", "2024 Honda Civic"),
            ("SELF", "bought_from", "Metro Honda"),
            ("SELF", "completed", "Austin Half Marathon")
        ],
        "gold_temporal": [
            {"claim": "purchased vehicle", "date": "2024-05-10"},
            {"claim": "completed marathon", "date": "2024-10-20"}
        ]
    },
    {
        "id": "ext_10",
        "description": "Healthcare appointments chronology",
        "text": (
            "I attended an annual physical exam with Dr. Robert Chen at Swedish Medical Center on March 12, 2025. "
            "Then on April 4, 2025, I went to Dr. Alvarez at Downtown Dental for a routine teeth cleaning."
        ),
        "gold_entities": [
            {"name": "SELF", "type": "Person"},
            {"name": "Dr. Robert Chen", "type": "Person"},
            {"name": "Swedish Medical Center", "type": "Organization"},
            {"name": "Dr. Alvarez", "type": "Person"},
            {"name": "Downtown Dental", "type": "Organization"}
        ],
        "compound_traps": ["Dr. Robert Chen and Dr. Alvarez"],
        "gold_relationships": [
            ("Dr. Robert Chen", "affiliated_with", "Swedish Medical Center"),
            ("Dr. Alvarez", "affiliated_with", "Downtown Dental"),
            ("SELF", "visited", "Dr. Robert Chen"),
            ("SELF", "visited", "Dr. Alvarez")
        ],
        "gold_temporal": [
            {"claim": "physical exam", "date": "2025-03-12"},
            {"claim": "teeth cleaning", "date": "2025-04-04"}
        ]
    }
]


# ============================================================================
# PART B: 10 GROUNDED ANSWERING QUERIES & GROUND TRUTH
# ============================================================================

ANSWERING_DATASET = [
    {
        "id": "ans_01",
        "category": "temporal_update",
        "query": "What keyboard does Arjun currently use for coding, and what keyboard did he use previously?",
        "context": {
            "direct_evidence": [
                {"id": "m001", "text": "In 2020, Arjun used a clicky mechanical keyboard with Cherry MX Blue switches."},
                {"id": "m002", "text": "In October 2025, Arjun switched to an Apple Magic Keyboard with scissor switches, which he currently uses for all daily coding."}
            ]
        },
        "gold_status": "answered",
        "gold_key_facts": ["currently uses Apple Magic Keyboard", "previously used Cherry MX Blue mechanical keyboard"],
        "expected_citations": ["m001", "m002"]
    },
    {
        "id": "ans_02",
        "category": "temporal_update",
        "query": "Where does Arjun currently reside, and where did he live before moving?",
        "context": {
            "direct_evidence": [
                {"id": "turn_101", "text": "Arjun lived in an apartment in Indiranagar, Bangalore, for three years."},
                {"id": "turn_102", "text": "On November 15, 2024, Arjun moved into a duplex in HSR Layout, Bangalore, where he currently lives."}
            ]
        },
        "gold_status": "answered",
        "gold_key_facts": ["currently lives in HSR Layout", "previously lived in Indiranagar"],
        "expected_citations": ["turn_101", "turn_102"]
    },
    {
        "id": "ans_03",
        "category": "multihop_relational",
        "query": "Who leads the VectorFlow project at CloudMatrix, and which database system does that project utilize?",
        "context": {
            "direct_evidence": [
                {"id": "m201", "text": "David Thorne is the engineering manager for the VectorFlow project at CloudMatrix Inc."},
                {"id": "m202", "text": "The VectorFlow pipeline is written in Rust and uses Neo4j for semantic graph persistence, deployed on AWS."}
            ]
        },
        "gold_status": "answered",
        "gold_key_facts": ["David Thorne leads/manages VectorFlow", "utilizes Neo4j database"],
        "expected_citations": ["m201", "m202"]
    },
    {
        "id": "ans_04",
        "category": "multihop_relational",
        "query": "Which organization employs Elena Ramos, and is she the same person as Dr. Elena Rostova?",
        "context": {
            "direct_evidence": [
                {"id": "turn_301", "text": "Julian Mercer introduced Elena Ramos as Vanguard Robotics' lead hardware engineer."},
                {"id": "turn_302", "text": "Dr. Elena Rostova is Chief Scientist at Aetheria Labs and works on neuro-symbolic AI. She is completely distinct from Elena Ramos."}
            ]
        },
        "gold_status": "answered",
        "gold_key_facts": ["Elena Ramos works for Vanguard Robotics", "not the same person as Dr. Elena Rostova who is at Aetheria Labs"],
        "expected_citations": ["turn_301", "turn_302"]
    },
    {
        "id": "ans_05",
        "category": "multihop_relational",
        "query": "What company did the current Lead Architect of CloudMatrix work for between 2021 and early 2024?",
        "context": {
            "direct_evidence": [
                {"id": "m401", "text": "In March 2024, Arjun joined CloudMatrix Inc as Lead Distributed Systems Architect."},
                {"id": "m402", "text": "From June 2021 until January 2024, Arjun worked as a Senior Data Engineer at Datacore."}
            ]
        },
        "gold_status": "answered",
        "gold_key_facts": ["Datacore"],
        "expected_citations": ["m401", "m402"]
    },
    {
        "id": "ans_06",
        "category": "single_fact",
        "query": "What model and color of car does Arjun own, and which dealership did he buy it from?",
        "context": {
            "direct_evidence": [
                {"id": "m501", "text": "On May 10, 2024, Arjun purchased a 2024 Honda Civic in Aegean Blue from Metro Honda."}
            ]
        },
        "gold_status": "answered",
        "gold_key_facts": ["2024 Honda Civic", "Aegean Blue", "Metro Honda"],
        "expected_citations": ["m501"]
    },
    {
        "id": "ans_07",
        "category": "single_fact",
        "query": "When was Project Hyperion cancelled, and did Marcus Vance ever commit production code for it?",
        "context": {
            "direct_evidence": [
                {"id": "turn_601", "text": "Project Hyperion was officially abandoned on June 12, 2025 due to regulatory concerns."},
                {"id": "turn_602", "text": "Neither Marcus Vance nor Priya Sharma ever wrote or committed any production code for Hyperion."}
            ]
        },
        "gold_status": "answered",
        "gold_key_facts": ["June 12, 2025", "Marcus Vance never committed production code"],
        "expected_citations": ["turn_601", "turn_602"]
    },
    {
        "id": "ans_08",
        "category": "negative_information",
        "query": "Has Arjun ever worked at Google or lived in London?",
        "context": {
            "direct_evidence": [
                {"id": "m701", "text": "Arjun's career history includes Datacore in Boston and CloudMatrix Inc in Seattle."},
                {"id": "m702", "text": "Arjun has lived in Bangalore (Indiranagar and HSR Layout)."}
            ]
        },
        "gold_status": "answered",
        "gold_key_facts": ["no record of working at Google", "no record of living in London"],
        "expected_citations": ["m701", "m702"]
    },
    {
        "id": "ans_09",
        "category": "unanswerable",
        "query": "What is Arjun's mother's maiden name, and what was his high school GPA?",
        "context": {
            "direct_evidence": [
                {"id": "m801", "text": "Arjun has a sister named Priya Mehta who graduated from Stanford in 2022."},
                {"id": "m802", "text": "Arjun completed his undergraduate degree in computer science in 2020."}
            ]
        },
        "gold_status": "insufficient_evidence",
        "gold_key_facts": ["information not recorded", "abstain without hallucination"],
        "expected_citations": []
    },
    {
        "id": "ans_10",
        "category": "unanswerable",
        "query": "How much did Arjun pay out-of-pocket for dental surgery on April 4, 2025?",
        "context": {
            "direct_evidence": [
                {"id": "m901", "text": "On April 4, 2025, Arjun visited Dr. Alvarez at Downtown Dental for a routine teeth cleaning."}
            ]
        },
        "gold_status": "insufficient_evidence",
        "gold_key_facts": ["the visit was for routine cleaning not surgery", "cost not recorded", "insufficient evidence"],
        "expected_citations": []
    }
]

# Ensure context_text is populated in accordance with production IVARA formatting
for _item in ANSWERING_DATASET:
    if "context_text" not in _item["context"]:
        _lines = ["DIRECT EVIDENCE:"]
        for _d in _item["context"].get("direct_evidence", []):
            _lines.append(f"[{_d['id']}]: {_d['text']}")
        _item["context"]["context_text"] = "\n".join(_lines)

