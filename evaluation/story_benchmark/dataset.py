"""Multi-Part Narrative Reasoning Benchmark Dataset Generator.

Constructs:
1. 12 independent, multi-turn conversational sessions detailing the life, career,
   collaborations, research, preferences, updates, and constraints of Arjun Mehta (2024 - 2026).
2. 50 structured evaluation questions across 10 distinct reasoning categories:
   - single_fact_recall
   - multi_hop
   - entity_relationship
   - temporal_reasoning
   - historical_state
   - contradiction_handling
   - preference_constraint
   - negative_fact
   - cross_session_synthesis
   - unanswerable
"""

import json
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).parent
STORY_DATA_PATH = BENCHMARK_DIR / "story_data.json"
GROUND_TRUTH_PATH = BENCHMARK_DIR / "ground_truth.json"


SESSIONS = [
    {
        "session_id": "session_01",
        "date": "2024-03-10",
        "title": "Background and University Life",
        "turns": [
            {"role": "user", "content": "Hi IVARA! Good to meet you."},
            {"role": "assistant", "content": "Hello! It is a pleasure to meet you. How can I assist you today?"},
            {"role": "user", "content": "I was born in Chandigarh on August 14, 1999."},
            {"role": "assistant", "content": "Nice to know! You were born in Chandigarh on August 14, 1999."},
            {"role": "user", "content": "I completed my B.Tech in Computer Science from ABV IIITM Gwalior in June 2022."},
            {"role": "assistant", "content": "Understood, B.Tech in CS from ABV IIITM Gwalior graduated in June 2022."},
            {"role": "user", "content": "My college roommate was Rohan Sharma. We lived in Hostel 3."},
            {"role": "assistant", "content": "Recorded: Rohan Sharma was your college roommate in Hostel 3 at IIITM Gwalior."},
            {"role": "user", "content": "I currently live in Indiranagar, Bengaluru with my rescue indie dog named Kaalu."},
            {"role": "assistant", "content": "Noted, you live in Indiranagar, Bengaluru with your rescue dog Kaalu."},
            {"role": "user", "content": "I prefer reading physical books over e-readers for technical subjects."}
        ]
    },
    {
        "session_id": "session_02",
        "date": "2024-06-15",
        "title": "HyperScale Systems Job and Early Architecture",
        "turns": [
            {"role": "user", "content": "Hey IVARA, just getting back from work."},
            {"role": "assistant", "content": "Welcome back! How was your work day?"},
            {"role": "user", "content": "I joined HyperScale Systems in Bengaluru as an Associate Software Engineer back in July 2022."},
            {"role": "assistant", "content": "Got it: Joined HyperScale Systems in July 2022 as Associate Software Engineer."},
            {"role": "user", "content": "My engineering manager at HyperScale is Vikram Rao."},
            {"role": "assistant", "content": "Recorded: Vikram Rao is your engineering manager at HyperScale."},
            {"role": "user", "content": "I am working with senior staff engineer Deepa Nair on our distributed caching layer called CacheMesh."},
            {"role": "assistant", "content": "Noted: Collaborating with Deepa Nair on CacheMesh at HyperScale."},
            {"role": "user", "content": "We built CacheMesh using Redis cluster and Go."},
            {"role": "assistant", "content": "Recorded: CacheMesh is built using Redis cluster and Go."},
            {"role": "user", "content": "I dislike Kubernetes complexity and prefer lightweight Docker Compose for local service development."},
            {"role": "assistant", "content": "Noted your preference for Docker Compose over Kubernetes for local dev."},
            {"role": "user", "content": "I decided to adopt Neovim configured with Lua as my primary code editor."}
        ]
    },
    {
        "session_id": "session_03",
        "date": "2024-09-20",
        "title": "Smart India Hackathon and AgriVision Project",
        "turns": [
            {"role": "user", "content": "Had an exciting weekend with my old college crew!"},
            {"role": "assistant", "content": "That sounds great! What did you do with your college friends?"},
            {"role": "user", "content": "I participated in the Smart India Hackathon alumni track with Rohan Sharma and Meera Sen."},
            {"role": "assistant", "content": "Recorded: Participated in SIH alumni track with Rohan Sharma and Meera Sen."},
            {"role": "user", "content": "Our team name was MatrixCrafters and we built AgriVision, an edge-AI crop disease diagnosis mobile app."},
            {"role": "assistant", "content": "Got it: Team MatrixCrafters developed AgriVision for edge-AI crop disease diagnosis."},
            {"role": "user", "content": "Meera Sen built the Flutter mobile frontend, Rohan managed the FastAPI cloud backend, and I trained the PyTorch Mobile vision model."},
            {"role": "assistant", "content": "Understood the division of roles: Meera on Flutter frontend, Rohan on FastAPI backend, and you on PyTorch Mobile vision model."},
            {"role": "user", "content": "Our project AgriVision won the 2nd runner-up prize of 1,500 dollars at SIH."},
            {"role": "assistant", "content": "Congratulations! Won 2nd runner-up prize ($1,500) for AgriVision."},
            {"role": "user", "content": "Meera is strictly allergic to peanuts, so we celebrated with peanut-free Italian food."},
            {"role": "assistant", "content": "Noted Meera Sen's severe peanut allergy constraint."},
            {"role": "user", "content": "I bought a 16-inch MacBook Pro M3 Max to run local model evaluations."}
        ]
    },
    {
        "session_id": "session_04",
        "date": "2024-12-05",
        "title": "Relocation to HSR Layout, Diet and Sports",
        "turns": [
            {"role": "user", "content": "I moved from Indiranagar to HSR Layout Sector 2 in Bengaluru this month."},
            {"role": "assistant", "content": "Updated your residence: Moved from Indiranagar to HSR Layout Sector 2, Bengaluru."},
            {"role": "user", "content": "I moved to HSR Layout to be closer to Rohan and our badminton club."},
            {"role": "assistant", "content": "Noted: Relocation motivated by proximity to Rohan and badminton club."},
            {"role": "user", "content": "I play badminton every Tuesday evening at PlayArena with Rohan Sharma."},
            {"role": "assistant", "content": "Recorded: Badminton every Tuesday evening at PlayArena with Rohan."},
            {"role": "user", "content": "I adopted a strict vegetarian diet starting in November 2024."},
            {"role": "assistant", "content": "Recorded your dietary change: Strict vegetarian since November 2024."},
            {"role": "user", "content": "I have set a personal fitness goal to complete a 10k marathon by late 2025."},
            {"role": "assistant", "content": "Fitness goal recorded: Complete a 10k marathon by late 2025."},
            {"role": "user", "content": "I tried learning German on Duolingo for 40 days, but I gave up and dropped it due to work overload."}
        ]
    },
    {
        "session_id": "session_05",
        "date": "2025-02-18",
        "title": "Career Pivot to Aetheria AI",
        "turns": [
            {"role": "user", "content": "Big career update to share with you!"},
            {"role": "assistant", "content": "Exciting! What is the career update?"},
            {"role": "user", "content": "I resigned from HyperScale Systems in January 2025."},
            {"role": "assistant", "content": "Recorded: Resigned from HyperScale Systems in January 2025."},
            {"role": "user", "content": "HyperScale offered me a 30 percent salary raise to stay, but I declined their counter-offer."},
            {"role": "assistant", "content": "Noted: Declined HyperScale's 30% raise counter-offer."},
            {"role": "user", "content": "I joined Aetheria AI, a generative AI research startup in Koramangala, as a Research Engineer on February 1, 2025."},
            {"role": "assistant", "content": "Congratulations! Joined Aetheria AI in Koramangala as Research Engineer on February 1, 2025."},
            {"role": "user", "content": "My new CTO at Aetheria AI is Dr. Shalini Verma, who previously led research at FAIR."},
            {"role": "assistant", "content": "Recorded: Dr. Shalini Verma is your CTO at Aetheria AI."},
            {"role": "user", "content": "I signed a non-compete clause with HyperScale barring me from consulting for competing database and caching companies for 12 months."}
        ]
    },
    {
        "session_id": "session_06",
        "date": "2025-05-10",
        "title": "NeuroCompact Project and Research Collaborators",
        "turns": [
            {"role": "user", "content": "At Aetheria AI, I am working on NeuroCompact, a 4-bit quantizer for speculative decoding."},
            {"role": "assistant", "content": "Recorded: Developing NeuroCompact, a 4-bit quantizer for speculative decoding at Aetheria AI."},
            {"role": "user", "content": "I am collaborating on NeuroCompact directly with compiler engineer Kabir Das and Dr. Shalini Verma."},
            {"role": "assistant", "content": "Collaborators recorded: Working on NeuroCompact with Kabir Das and Dr. Shalini Verma."},
            {"role": "user", "content": "We decided to rewrite our CUDA execution kernels in Mojo instead of Python."},
            {"role": "assistant", "content": "Architecture decision: Rewriting CUDA execution kernels in Mojo."},
            {"role": "user", "content": "I never drink coffee because it gives me migraines; I strictly drink masala chai."},
            {"role": "assistant", "content": "Noted beverage preference: No coffee due to migraines, strictly masala chai."},
            {"role": "user", "content": "Our research goal is to submit our empirical findings on NeuroCompact to EMNLP 2025."}
        ]
    },
    {
        "session_id": "session_07",
        "date": "2025-08-22",
        "title": "EMNLP Paper Rejection, Departure of Kabir, and Project Pivot",
        "turns": [
            {"role": "user", "content": "Tough couple of weeks at the research lab."},
            {"role": "assistant", "content": "Sorry to hear that. What happened at the lab?"},
            {"role": "user", "content": "Our EMNLP 2025 paper submission on NeuroCompact was rejected in August 2025 due to baseline benchmark leakage."},
            {"role": "assistant", "content": "Noted: EMNLP 2025 submission rejected in August 2025 due to baseline leakage."},
            {"role": "user", "content": "My collaborator Kabir Das left Aetheria AI in July 2025 to relocate to Vancouver, Canada."},
            {"role": "assistant", "content": "Recorded: Kabir Das departed Aetheria AI in July 2025 to relocate to Vancouver, Canada."},
            {"role": "user", "content": "Dr. Shalini Verma and I decided to pivot NeuroCompact from server speculative decoding to on-device memory compression."},
            {"role": "assistant", "content": "Major project pivot: NeuroCompact shifted to on-device memory compression with Dr. Shalini Verma."},
            {"role": "user", "content": "I reconnected with Meera Sen to help test on-device latency on real Android smartphones."},
            {"role": "assistant", "content": "Collaboration: Meera Sen assisting with Android on-device latency testing."},
            {"role": "user", "content": "I no longer use Mojo for kernel development; I switched back to C++ and PyTorch C++ API because the Mojo tooling was unstable."}
        ]
    },
    {
        "session_id": "session_08",
        "date": "2025-11-14",
        "title": "Bengaluru 10k Marathon and Health Setback",
        "turns": [
            {"role": "user", "content": "I ran the Bengaluru 10k Marathon on November 9, 2025 and finished in 54 minutes!"},
            {"role": "assistant", "content": "Congratulations on finishing the Bengaluru 10k Marathon in 54 minutes on November 9, 2025!"},
            {"role": "user", "content": "Rohan Sharma started the race with me, but he twisted his ankle at kilometer 4 and could not complete the marathon."},
            {"role": "assistant", "content": "Noted: Rohan Sharma twisted ankle at km 4 and DNF (did not finish)."},
            {"role": "user", "content": "I developed plantar fasciitis in my right heel after the run, and the sports doctor forbade high-impact running for 3 months."},
            {"role": "assistant", "content": "Medical constraint: Plantar fasciitis, no high-impact running for 3 months."},
            {"role": "user", "content": "I switched from running to swimming at the Padukone-Dravid Centre for Sports Excellence."},
            {"role": "assistant", "content": "Activity update: Switched to swimming at Padukone-Dravid Centre."},
            {"role": "user", "content": "I bought a Garmin Forerunner 265 watch to track my swim laps and heart rate."}
        ]
    },
    {
        "session_id": "session_09",
        "date": "2026-01-25",
        "title": "MLSys Oral Acceptance and Promotion",
        "turns": [
            {"role": "user", "content": "Huge milestone for our research work!"},
            {"role": "assistant", "content": "Tell me all about it!"},
            {"role": "user", "content": "Our revised paper on NeuroCompact v2 was accepted as an Oral Presentation at MLSys 2026!"},
            {"role": "assistant", "content": "Outstanding achievement! NeuroCompact v2 accepted as an Oral Presentation at MLSys 2026."},
            {"role": "user", "content": "We were awarded a 25,000 dollar research compute grant from the OpenCompute Foundation."},
            {"role": "assistant", "content": "Recorded: $25,000 compute grant from OpenCompute Foundation."},
            {"role": "user", "content": "Dr. Shalini Verma promoted me from Research Engineer to Lead Research Scientist at Aetheria AI in January 2026."},
            {"role": "assistant", "content": "Congratulations on your promotion to Lead Research Scientist at Aetheria AI!"},
            {"role": "user", "content": "I hired and started mentoring a research intern named Ananya Bose who is working on memory-augmented transformers."}
        ]
    },
    {
        "session_id": "session_10",
        "date": "2026-04-12",
        "title": "Relocation Decision to Zurich",
        "turns": [
            {"role": "user", "content": "Aetheria AI is opening a European research hub in Zurich, Switzerland."},
            {"role": "assistant", "content": "Noted: Aetheria AI opening a European research office in Zurich."},
            {"role": "user", "content": "Dr. Shalini Verma gave me the option to stay in Bengaluru as regional lead or relocate to Zurich to head the efficiency lab."},
            {"role": "assistant", "content": "Understood: Offered choice between staying in Bengaluru or heading Zurich lab."},
            {"role": "user", "content": "I decided to relocate to Zurich in September 2026."},
            {"role": "assistant", "content": "Major decision recorded: Relocating to Zurich in September 2026."},
            {"role": "user", "content": "I enrolled in intensive German classes at the Goethe-Institut Bengaluru, reviving my language goal."},
            {"role": "assistant", "content": "Goal revived: Intensive German classes at Goethe-Institut Bengaluru."},
            {"role": "user", "content": "Kaalu will move with me to Switzerland; I completed his pet microchip and rabies titer test on April 5, 2026."},
            {"role": "assistant", "content": "Recorded: Kaalu moving to Zurich with pet microchip and rabies titer completed April 5, 2026."},
            {"role": "user", "content": "Rohan Sharma is staying in Bengaluru; he got engaged to Sneha Roy in March 2026."}
        ]
    },
    {
        "session_id": "session_11",
        "date": "2026-07-19",
        "title": "Technology Stack Updates and Visa Obstacle",
        "turns": [
            {"role": "user", "content": "I have completely phased out Redis in all my systems; I now strictly use Dragonfly and DuckDB."},
            {"role": "assistant", "content": "Database stack update: Redis phased out, strictly using Dragonfly and DuckDB."},
            {"role": "user", "content": "I was unable to attend MLSys 2026 in person in Santa Clara because my US B1/B2 visa was administratively delayed under 221g."},
            {"role": "assistant", "content": "Travel constraint: US visa delayed under 221g, could not attend MLSys in person."},
            {"role": "user", "content": "Dr. Shalini Verma traveled to California and delivered the NeuroCompact oral presentation on my behalf."},
            {"role": "assistant", "content": "Recorded: Dr. Shalini Verma delivered the MLSys oral presentation on your behalf."},
            {"role": "user", "content": "Ananya Bose completed her internship under my supervision and received a full-time Research Scientist offer at Aetheria AI."},
            {"role": "assistant", "content": "Mentorship outcome: Ananya Bose completed internship and received full-time Research Scientist offer."},
            {"role": "user", "content": "I have a strict professional policy: I refuse to work on closed-source military or defense surveillance contracts."}
        ]
    },
    {
        "session_id": "session_12",
        "date": "2026-10-02",
        "title": "Life in Zurich and Current State",
        "turns": [
            {"role": "user", "content": "Greetings from Switzerland! I successfully landed and moved into my apartment in Zurich Kreis 4 on September 15, 2026."},
            {"role": "assistant", "content": "Welcome to Switzerland! Settled into Zurich Kreis 4 on September 15, 2026."},
            {"role": "user", "content": "I work at the Aetheria Europe research office located on Technoparkstrasse in Zurich."},
            {"role": "assistant", "content": "Workplace recorded: Aetheria Europe office on Technoparkstrasse, Zurich."},
            {"role": "user", "content": "Kaalu is loving the autumn walks around Lake Zurich."},
            {"role": "assistant", "content": "Delighted to hear Kaalu is enjoying Lake Zurich walks."},
            {"role": "user", "content": "I missed Bengaluru masala chai, so I bought a traditional brass tea kettle to brew authentic chai at home."},
            {"role": "assistant", "content": "Noted: Bought a traditional brass tea kettle to brew masala chai in Zurich."},
            {"role": "user", "content": "I am currently studying for the German B1 proficiency exam."},
            {"role": "assistant", "content": "Language progress: Studying for German B1 exam in Zurich."},
            {"role": "user", "content": "My current diet remains strictly vegetarian, but I eat Swiss artisanal cheeses freely."},
            {"role": "assistant", "content": "Current dietary status: Strictly vegetarian, eats Swiss cheeses."},
            {"role": "user", "content": "On weekends, I hike up Uetliberg and do cold water swimming in Lake Zurich."}
        ]
    }
]


QUESTIONS = [
    # 1. Single-fact Recall (5 questions)
    {
        "id": "q01",
        "category": "single_fact_recall",
        "question": "Where was Arjun born and what is his date of birth?",
        "expected_answer": "Arjun was born in Chandigarh on August 14, 1999.",
        "key_facts": ["Chandigarh", "August 14, 1999"],
        "supporting_sessions": ["session_01"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q02",
        "category": "single_fact_recall",
        "question": "What is the name and breed of Arjun's dog?",
        "expected_answer": "Arjun's dog is a rescue indie dog named Kaalu.",
        "key_facts": ["Kaalu", "indie"],
        "supporting_sessions": ["session_01"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q03",
        "category": "single_fact_recall",
        "question": "What watch model did Arjun purchase to track his swimming and fitness?",
        "expected_answer": "Arjun bought a Garmin Forerunner 265 watch.",
        "key_facts": ["Garmin Forerunner 265"],
        "supporting_sessions": ["session_08"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q04",
        "category": "single_fact_recall",
        "question": "What prize did team MatrixCrafters win at the Smart India Hackathon for AgriVision?",
        "expected_answer": "Team MatrixCrafters won the 2nd runner-up prize of $1,500 for AgriVision.",
        "key_facts": ["2nd runner-up", "1,500", "AgriVision"],
        "supporting_sessions": ["session_03"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q05",
        "category": "single_fact_recall",
        "question": "Who was Arjun's engineering manager at HyperScale Systems?",
        "expected_answer": "Arjun's engineering manager at HyperScale Systems was Vikram Rao.",
        "key_facts": ["Vikram Rao"],
        "supporting_sessions": ["session_02"],
        "is_answerable": True,
        "negative_constraint": False
    },

    # 2. Multi-hop Reasoning (5 questions)
    {
        "id": "q06",
        "category": "multi_hop",
        "question": "Who managed the backend of the hackathon project where Arjun trained the computer vision model?",
        "expected_answer": "Rohan Sharma managed the backend (FastAPI) of the hackathon project AgriVision.",
        "key_facts": ["Rohan Sharma", "AgriVision"],
        "supporting_sessions": ["session_03"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q07",
        "category": "multi_hop",
        "question": "Which company's former research leader became Arjun's CTO when he transitioned to generative AI?",
        "expected_answer": "Dr. Shalini Verma, who previously led research at FAIR, became Arjun's CTO at Aetheria AI.",
        "key_facts": ["Dr. Shalini Verma", "FAIR", "Aetheria AI"],
        "supporting_sessions": ["session_05"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q08",
        "category": "multi_hop",
        "question": "Who presented Arjun's accepted oral paper when he was unable to travel to the United States?",
        "expected_answer": "Dr. Shalini Verma delivered the MLSys 2026 oral presentation on NeuroCompact on Arjun's behalf.",
        "key_facts": ["Dr. Shalini Verma", "MLSys 2026", "NeuroCompact"],
        "supporting_sessions": ["session_09", "session_11"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q09",
        "category": "multi_hop",
        "question": "What sport did Arjun take up after a running injury sustained while running with his former college roommate?",
        "expected_answer": "Arjun switched to swimming at the Padukone-Dravid Centre after injuring his heel running with Rohan Sharma.",
        "key_facts": ["swimming", "Padukone-Dravid", "Rohan Sharma"],
        "supporting_sessions": ["session_01", "session_08"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q10",
        "category": "multi_hop",
        "question": "What full-time position was offered to the intern who was mentored on memory-augmented transformers by Arjun?",
        "expected_answer": "Ananya Bose was offered a full-time Research Scientist position at Aetheria AI.",
        "key_facts": ["Ananya Bose", "Research Scientist", "Aetheria AI"],
        "supporting_sessions": ["session_09", "session_11"],
        "is_answerable": True,
        "negative_constraint": False
    },

    # 3. Entity Relationship Reasoning (5 questions)
    {
        "id": "q11",
        "category": "entity_relationship",
        "question": "How did Arjun and Rohan Sharma originally meet, and what activities did they do together in Bengaluru?",
        "expected_answer": "Arjun and Rohan met as college roommates in Hostel 3 at ABV IIITM Gwalior. In Bengaluru, they lived near each other in HSR Layout, played badminton every Tuesday at PlayArena, participated in SIH (MatrixCrafters), and ran the Bengaluru 10k.",
        "key_facts": ["roommates", "IIITM Gwalior", "badminton", "PlayArena", "AgriVision"],
        "supporting_sessions": ["session_01", "session_03", "session_04", "session_08"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q12",
        "category": "entity_relationship",
        "question": "In what two different projects did Arjun collaborate with Meera Sen, and what was her role in each?",
        "expected_answer": "Arjun collaborated with Meera Sen on AgriVision (where she built the Flutter mobile frontend) and later on the pivoted NeuroCompact (where she assisted in testing on-device latency on real Android handsets).",
        "key_facts": ["AgriVision", "Flutter", "NeuroCompact", "Android"],
        "supporting_sessions": ["session_03", "session_07"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q13",
        "category": "entity_relationship",
        "question": "What was the working relationship between Arjun and Kabir Das at Aetheria AI, and why did it end?",
        "expected_answer": "Kabir Das was a compiler engineer collaborating with Arjun on NeuroCompact CUDA/Mojo kernels. The collaboration ended when Kabir left Aetheria AI in July 2025 to relocate to Vancouver, Canada.",
        "key_facts": ["Kabir Das", "compiler", "NeuroCompact", "Vancouver", "Canada"],
        "supporting_sessions": ["session_06", "session_07"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q14",
        "category": "entity_relationship",
        "question": "What is the relationship between Dr. Shalini Verma and Arjun across his time at Aetheria AI?",
        "expected_answer": "Dr. Shalini Verma was Arjun's CTO at Aetheria AI. She co-authored NeuroCompact, supported its pivot, promoted him to Lead Research Scientist in January 2026, presented his MLSys oral paper, and offered him the opportunity to head the Zurich research lab.",
        "key_facts": ["CTO", "Aetheria AI", "NeuroCompact", "promoted", "Lead Research Scientist", "Zurich"],
        "supporting_sessions": ["session_05", "session_06", "session_07", "session_09", "session_10", "session_11"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q15",
        "category": "entity_relationship",
        "question": "Who is Sneha Roy and how is she connected to Arjun's social circle?",
        "expected_answer": "Sneha Roy is Rohan Sharma's fiancée; they got engaged in March 2026.",
        "key_facts": ["Sneha Roy", "engaged", "Rohan Sharma"],
        "supporting_sessions": ["session_10"],
        "is_answerable": True,
        "negative_constraint": False
    },

    # 4. Temporal Reasoning (5 questions)
    {
        "id": "q16",
        "category": "temporal_reasoning",
        "question": "Did Arjun move to HSR Layout before or after he resigned from HyperScale Systems?",
        "expected_answer": "Arjun moved to HSR Layout in December 2024, BEFORE resigning from HyperScale Systems in January 2025.",
        "key_facts": ["before", "December 2024", "January 2025"],
        "supporting_sessions": ["session_04", "session_05"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q17",
        "category": "temporal_reasoning",
        "question": "In what month and year was Arjun promoted to Lead Research Scientist, and how long was that after the Bengaluru 10k marathon?",
        "expected_answer": "Arjun was promoted to Lead Research Scientist in January 2026, roughly two months after running the Bengaluru 10k marathon on November 9, 2025.",
        "key_facts": ["January 2026", "November 2025", "two months"],
        "supporting_sessions": ["session_08", "session_09"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q18",
        "category": "temporal_reasoning",
        "question": "What was the chronological sequence of Arjun's residences from university graduation to late 2026?",
        "expected_answer": "Arjun graduated from Gwalior (June 2022), lived in Indiranagar, Bengaluru (2022-2024), moved to HSR Layout, Bengaluru (December 2024), and finally relocated to Zurich Kreis 4, Switzerland (September 2026).",
        "key_facts": ["Gwalior", "Indiranagar", "HSR Layout", "Zurich"],
        "supporting_sessions": ["session_01", "session_04", "session_12"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q19",
        "category": "temporal_reasoning",
        "question": "When did Kabir Das leave Aetheria AI relative to the rejection of the EMNLP paper?",
        "expected_answer": "Kabir Das left Aetheria AI in July 2025, just before the EMNLP paper was rejected in August 2025.",
        "key_facts": ["July 2025", "August 2025", "before"],
        "supporting_sessions": ["session_07"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q20",
        "category": "temporal_reasoning",
        "question": "When did Arjun decide to relocate to Zurich versus when did he actually move into his apartment there?",
        "expected_answer": "Arjun decided to relocate to Zurich in April 2026, and actually moved into his apartment in Zurich Kreis 4 on September 15, 2026.",
        "key_facts": ["April 2026", "September 15, 2026"],
        "supporting_sessions": ["session_10", "session_12"],
        "is_answerable": True,
        "negative_constraint": False
    },

    # 5. Historical State Retrieval (5 questions)
    {
        "id": "q21",
        "category": "historical_state",
        "question": "Where did Arjun live when he first joined HyperScale Systems in 2022?",
        "expected_answer": "When Arjun joined HyperScale Systems in 2022, he lived in Indiranagar, Bengaluru.",
        "key_facts": ["Indiranagar", "Bengaluru"],
        "supporting_sessions": ["session_01", "session_02"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q22",
        "category": "historical_state",
        "question": "What database technology did Arjun initially use for CacheMesh before updating his database stack?",
        "expected_answer": "Arjun initially used Redis (cluster) for CacheMesh at HyperScale.",
        "key_facts": ["Redis", "CacheMesh"],
        "supporting_sessions": ["session_02", "session_11"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q23",
        "category": "historical_state",
        "question": "What was Arjun's job title when he initially joined Aetheria AI in February 2025?",
        "expected_answer": "Arjun's job title was Research Engineer when he joined Aetheria AI in February 2025.",
        "key_facts": ["Research Engineer", "Aetheria AI"],
        "supporting_sessions": ["session_05"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q24",
        "category": "historical_state",
        "question": "What language did Arjun try to use for CUDA execution kernels on NeuroCompact before reverting to C++?",
        "expected_answer": "Arjun tried using Mojo for CUDA execution kernels before switching back to C++ and PyTorch C++ API.",
        "key_facts": ["Mojo", "NeuroCompact"],
        "supporting_sessions": ["session_06", "session_07"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q25",
        "category": "historical_state",
        "question": "What was the initial research goal and target venue of the NeuroCompact project in mid-2025?",
        "expected_answer": "The initial goal was 4-bit quantization for server speculative decoding targeted at EMNLP 2025.",
        "key_facts": ["speculative decoding", "EMNLP 2025"],
        "supporting_sessions": ["session_06", "session_07"],
        "is_answerable": True,
        "negative_constraint": False
    },

    # 6. Contradiction Handling (5 questions)
    {
        "id": "q26",
        "category": "contradiction_handling",
        "question": "Does Arjun currently live in Bengaluru or Zurich?",
        "expected_answer": "Arjun currently lives in Zurich (Kreis 4), Switzerland, having moved there on September 15, 2026 from Bengaluru.",
        "key_facts": ["Zurich", "Switzerland"],
        "supporting_sessions": ["session_01", "session_04", "session_10", "session_12"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q27",
        "category": "contradiction_handling",
        "question": "Does Arjun currently use Redis or Dragonfly for his systems?",
        "expected_answer": "Arjun completely phased out Redis and now strictly uses Dragonfly and DuckDB.",
        "key_facts": ["Dragonfly", "DuckDB", "phased out Redis"],
        "supporting_sessions": ["session_02", "session_11"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q28",
        "category": "contradiction_handling",
        "question": "Did Arjun abandon learning German or is he currently learning it?",
        "expected_answer": "Arjun initially dropped German on Duolingo in late 2024, but revived his studies at the Goethe-Institut in April 2026 and is currently studying for the German B1 proficiency exam in Zurich.",
        "key_facts": ["dropped", "revived", "Goethe-Institut", "B1", "Zurich"],
        "supporting_sessions": ["session_04", "session_10", "session_12"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q29",
        "category": "contradiction_handling",
        "question": "Is NeuroCompact designed for server speculative decoding or on-device memory compression?",
        "expected_answer": "While originally designed for server speculative decoding, NeuroCompact was pivoted in August 2025 to on-device memory compression.",
        "key_facts": ["pivoted", "on-device", "memory compression"],
        "supporting_sessions": ["session_06", "session_07", "session_09"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q30",
        "category": "contradiction_handling",
        "question": "Is Arjun currently programming CUDA kernels in Mojo or C++?",
        "expected_answer": "Arjun no longer uses Mojo; he switched back to C++ and PyTorch C++ API due to unstable tooling.",
        "key_facts": ["C++", "no longer uses Mojo"],
        "supporting_sessions": ["session_06", "session_07"],
        "is_answerable": True,
        "negative_constraint": False
    },

    # 7. Preference and Constraint Recall (5 questions)
    {
        "id": "q31",
        "category": "preference_constraint",
        "question": "What is Arjun's stance on drinking coffee, and what does he drink instead?",
        "expected_answer": "Arjun never drinks coffee because it triggers migraines; he strictly drinks masala chai.",
        "key_facts": ["no coffee", "migraines", "masala chai"],
        "supporting_sessions": ["session_06", "session_12"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q32",
        "category": "preference_constraint",
        "question": "What dietary restrictions does Arjun follow, and does he eat dairy or cheese?",
        "expected_answer": "Arjun has followed a strict vegetarian diet since November 2024; he eats Swiss artisanal dairy and cheeses freely.",
        "key_facts": ["vegetarian", "November 2024", "cheese"],
        "supporting_sessions": ["session_04", "session_12"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q33",
        "category": "preference_constraint",
        "question": "What code editor and configuration does Arjun prefer?",
        "expected_answer": "Arjun prefers Neovim configured with Lua.",
        "key_facts": ["Neovim", "Lua"],
        "supporting_sessions": ["session_02"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q34",
        "category": "preference_constraint",
        "question": "What contractual restriction did Arjun have after resigning from HyperScale Systems?",
        "expected_answer": "Arjun signed a 12-month non-compete clause barring him from consulting for competing database and caching companies.",
        "key_facts": ["non-compete", "database", "caching", "12 months"],
        "supporting_sessions": ["session_05"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q35",
        "category": "preference_constraint",
        "question": "What ethical constraint does Arjun maintain regarding his research and engineering contracts?",
        "expected_answer": "Arjun strictly refuses to work on closed-source military or defense surveillance contracts.",
        "key_facts": ["refuses", "military", "defense", "surveillance"],
        "supporting_sessions": ["session_11"],
        "is_answerable": True,
        "negative_constraint": False
    },

    # 8. Negative Questions (5 questions)
    {
        "id": "q36",
        "category": "negative_fact",
        "question": "Did Rohan Sharma complete the Bengaluru 10k Marathon?",
        "expected_answer": "No, Rohan Sharma twisted his ankle at kilometer 4 and did not finish the marathon.",
        "key_facts": ["No", "twisted ankle", "did not finish"],
        "supporting_sessions": ["session_08"],
        "is_answerable": True,
        "negative_constraint": True
    },
    {
        "id": "q37",
        "category": "negative_fact",
        "question": "Did Arjun accept HyperScale's salary raise counter-offer when leaving for Aetheria AI?",
        "expected_answer": "No, Arjun declined HyperScale's 30 percent salary raise counter-offer.",
        "key_facts": ["No", "declined", "30 percent"],
        "supporting_sessions": ["session_05"],
        "is_answerable": True,
        "negative_constraint": True
    },
    {
        "id": "q38",
        "category": "negative_fact",
        "question": "Did Arjun attend the MLSys 2026 conference in person in California?",
        "expected_answer": "No, Arjun could not attend in person because his US visa was delayed under 221g.",
        "key_facts": ["No", "visa delayed", "could not attend"],
        "supporting_sessions": ["session_11"],
        "is_answerable": True,
        "negative_constraint": True
    },
    {
        "id": "q39",
        "category": "negative_fact",
        "question": "Did Rohan Sharma move to Zurich with Arjun?",
        "expected_answer": "No, Rohan Sharma stayed in Bengaluru and got engaged to Sneha Roy.",
        "key_facts": ["No", "stayed in Bengaluru", "Rohan"],
        "supporting_sessions": ["session_10"],
        "is_answerable": True,
        "negative_constraint": True
    },
    {
        "id": "q40",
        "category": "negative_fact",
        "question": "Was Arjun allowed to continue high-impact marathon running after November 2025?",
        "expected_answer": "No, Arjun developed plantar fasciitis and the doctor forbade high-impact running for 3 months.",
        "key_facts": ["No", "plantar fasciitis", "forbade high-impact running"],
        "supporting_sessions": ["session_08"],
        "is_answerable": True,
        "negative_constraint": True
    },

    # 9. Cross-session Synthesis (5 questions)
    {
        "id": "q41",
        "category": "cross_session_synthesis",
        "question": "Trace the complete evolution of Arjun's athletic and fitness activities from 2024 through late 2026.",
        "expected_answer": "In 2024, Arjun played badminton every Tuesday at PlayArena with Rohan and set a goal to run a 10k. In November 2025, he ran the Bengaluru 10k in 54 minutes, but injured his heel (plantar fasciitis). His doctor forbade running, so he switched to swimming at the Padukone-Dravid Centre with a Garmin 265. After moving to Zurich in 2026, he hikes Uetliberg and does cold water swimming in Lake Zurich.",
        "key_facts": ["badminton", "PlayArena", "marathon", "plantar fasciitis", "swimming", "Padukone-Dravid", "Zurich", "Uetliberg"],
        "supporting_sessions": ["session_04", "session_08", "session_12"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q42",
        "category": "cross_session_synthesis",
        "question": "Summarize all the software and research projects Arjun worked on, including their technical domains and primary collaborators.",
        "expected_answer": "1. CacheMesh: Distributed caching in Redis and Go at HyperScale with Deepa Nair. 2. AgriVision: Edge-AI crop disease diagnosis mobile app with PyTorch Mobile, collaborating with Meera Sen (Flutter) and Rohan Sharma (FastAPI). 3. NeuroCompact: 4-bit quantizer for speculative decoding, later pivoted to on-device memory compression in C++, collaborating with Kabir Das, Dr. Shalini Verma, and Meera Sen.",
        "key_facts": ["CacheMesh", "Deepa Nair", "AgriVision", "Meera Sen", "Rohan Sharma", "NeuroCompact", "Dr. Shalini Verma"],
        "supporting_sessions": ["session_02", "session_03", "session_06", "session_07", "session_09"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q43",
        "category": "cross_session_synthesis",
        "question": "Describe Arjun's language learning journey with German from his initial attempt to his current status.",
        "expected_answer": "In late 2024, Arjun tried learning German on Duolingo for 40 days but dropped it due to work overload. In April 2026, after deciding to relocate to Zurich, he revived his goal by taking intensive classes at Goethe-Institut Bengaluru. By October 2026 in Zurich, he is actively studying for the German B1 proficiency exam.",
        "key_facts": ["Duolingo", "dropped", "Goethe-Institut", "B1", "Zurich"],
        "supporting_sessions": ["session_04", "session_10", "session_12"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q44",
        "category": "cross_session_synthesis",
        "question": "How did the life and career paths of college roommates Arjun and Rohan diverge by late 2026?",
        "expected_answer": "Arjun transitioned from HyperScale to Aetheria AI, became Lead Research Scientist, pivoted research, ran a marathon, and relocated to Zurich, Switzerland with his dog Kaalu. Rohan stayed in Bengaluru, worked on backend systems, got engaged to Sneha Roy in March 2026, and remained in India.",
        "key_facts": ["Arjun", "Zurich", "Aetheria", "Rohan", "Bengaluru", "Sneha Roy", "engaged"],
        "supporting_sessions": ["session_01", "session_04", "session_05", "session_10", "session_12"],
        "is_answerable": True,
        "negative_constraint": False
    },
    {
        "id": "q45",
        "category": "cross_session_synthesis",
        "question": "Synthesize the complete life cycle of the NeuroCompact project from inception to MLSys presentation.",
        "expected_answer": "NeuroCompact began at Aetheria AI in mid-2025 as a 4-bit quantizer for speculative decoding in Mojo with Kabir Das and Dr. Shalini Verma. After EMNLP rejection and Kabir's departure in summer 2025, Arjun and Shalini pivoted it to on-device memory compression in C++ with Meera Sen testing Android latency. NeuroCompact v2 won oral presentation at MLSys 2026 and a $25k grant, and Dr. Shalini delivered the oral presentation in California.",
        "key_facts": ["inception", "speculative decoding", "EMNLP rejection", "pivot", "on-device", "MLSys 2026 oral", "Dr. Shalini"],
        "supporting_sessions": ["session_06", "session_07", "session_09", "session_11"],
        "is_answerable": True,
        "negative_constraint": False
    },

    # 10. Unanswerable Questions (5 questions)
    {
        "id": "q46",
        "category": "unanswerable",
        "question": "What was the exact GPA or percentage Arjun graduated with from ABV IIITM Gwalior?",
        "expected_answer": "I do not have sufficient recorded memories to answer this question. Arjun's college GPA is not mentioned.",
        "key_facts": ["insufficient", "not mentioned", "unknown"],
        "supporting_sessions": [],
        "is_answerable": False,
        "negative_constraint": False
    },
    {
        "id": "q47",
        "category": "unanswerable",
        "question": "What brand and model of car does Arjun drive in Zurich?",
        "expected_answer": "I do not have sufficient recorded memories to answer this question. Arjun's car is not mentioned.",
        "key_facts": ["insufficient", "not mentioned", "unknown"],
        "supporting_sessions": [],
        "is_answerable": False,
        "negative_constraint": False
    },
    {
        "id": "q48",
        "category": "unanswerable",
        "question": "What is Sneha Roy's profession and company?",
        "expected_answer": "I do not have sufficient recorded memories to answer this question. Sneha Roy's profession is not recorded.",
        "key_facts": ["insufficient", "not recorded", "unknown"],
        "supporting_sessions": [],
        "is_answerable": False,
        "negative_constraint": False
    },
    {
        "id": "q49",
        "category": "unanswerable",
        "question": "How much equity percentage did Arjun receive upon joining Aetheria AI?",
        "expected_answer": "I do not have sufficient recorded memories to answer this question. Arjun's equity percentage was never disclosed.",
        "key_facts": ["insufficient", "not recorded", "unknown"],
        "supporting_sessions": [],
        "is_answerable": False,
        "negative_constraint": False
    },
    {
        "id": "q50",
        "category": "unanswerable",
        "question": "What was the name of Arjun's high school in Chandigarh?",
        "expected_answer": "I do not have sufficient recorded memories to answer this question. Arjun's high school name is not recorded.",
        "key_facts": ["insufficient", "not recorded", "unknown"],
        "supporting_sessions": [],
        "is_answerable": False,
        "negative_constraint": False
    }
]


def export_benchmark_data():
    """Write story data and ground truth to disk in JSON format."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    with open(STORY_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(SESSIONS, f, indent=2, ensure_ascii=False)

    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(QUESTIONS, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(SESSIONS)} story sessions to {STORY_DATA_PATH}")
    print(f"Exported {len(QUESTIONS)} evaluation questions to {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    export_benchmark_data()

