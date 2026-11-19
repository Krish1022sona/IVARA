"""Generate 280 Structured Questions across 12 Categories for Expanded Narrative Benchmark.

Divides into:
- 30 Dev split questions
- 250 Held-out Test split questions
"""

import json
from pathlib import Path
from build_dataset import ALL_SESSIONS

OUT_DIR = Path(__file__).parent


def create_questions():
    questions = []

    # 1. Single Fact Retrieval (25 questions)
    single_facts = [
        ("s01", "dev", "In which city was Arjun Mehta born?", ["Chandigarh"], ["session_01"], "Arjun was born in Chandigarh on August 14, 1999."),
        ("s02", "test", "On what date was Arjun born?", ["August 14, 1999"], ["session_01"], "Arjun was born on August 14, 1999."),
        ("s03", "test", "From which university did Arjun earn his B.Tech degree?", ["ABV IIITM Gwalior"], ["session_01"], "Arjun graduated with a B.Tech in CS from ABV IIITM Gwalior."),
        ("s04", "test", "In which month and year did Arjun graduate with his B.Tech?", ["June 2022"], ["session_01"], "He graduated in June 2022."),
        ("s05", "test", "Who was Arjun's college roommate in Hostel 3?", ["Rohan Sharma"], ["session_01"], "Rohan Sharma was his college roommate in Hostel 3."),
        ("s06", "test", "What is the name and breed/type of Arjun's first pet dog?", ["Kaalu", "indie"], ["session_01"], "His rescue indie dog is named Kaalu."),
        ("s07", "test", "Which company did Arjun join in July 2022?", ["HyperScale Systems"], ["session_02"], "Arjun joined HyperScale Systems in July 2022."),
        ("s08", "test", "What was Arjun's initial job title at HyperScale Systems?", ["Associate Software Engineer"], ["session_02"], "He was hired as an Associate Software Engineer."),
        ("s09", "test", "Who was Arjun's engineering manager at HyperScale Systems?", ["Vikram Rao"], ["session_02"], "His engineering manager was Vikram Rao."),
        ("s10", "dev", "What distributed caching system did Arjun build at HyperScale Systems?", ["CacheMesh"], ["session_02"], "Arjun worked on CacheMesh."),
        ("s11", "test", "What technologies powered the CacheMesh distributed caching layer?", ["Redis cluster", "Go"], ["session_02"], "CacheMesh was built with Redis cluster and Go."),
        ("s12", "test", "What team name did Arjun and his friends use at the Smart India Hackathon?", ["MatrixCrafters"], ["session_03"], "Their team name was MatrixCrafters."),
        ("s13", "test", "What project did team MatrixCrafters develop at SIH?", ["AgriVision"], ["session_03"], "They built AgriVision for edge-AI crop disease diagnosis."),
        ("s14", "test", "How much prize money did AgriVision win at SIH?", ["1,500 dollars", "2nd runner-up"], ["session_03"], "They won the 2nd runner-up prize of $1,500."),
        ("s15", "test", "Who is the CTO at Aetheria AI who previously led research at FAIR?", ["Dr. Shalini Verma"], ["session_05"], "Dr. Shalini Verma is the CTO at Aetheria AI."),
        ("s16", "test", "What 4-bit quantizer project did Arjun develop at Aetheria AI?", ["NeuroCompact"], ["session_06"], "Arjun developed NeuroCompact."),
        ("s17", "test", "To which Canadian city did compiler engineer Kabir Das relocate in July 2025?", ["Vancouver"], ["session_07"], "Kabir Das relocated to Vancouver, Canada."),
        ("s18", "test", "What time did Arjun record for the Bengaluru 10k Marathon in November 2025?", ["54 minutes"], ["session_08"], "Arjun finished the 10k in 54 minutes."),
        ("s19", "test", "What medical condition in his right heel did Arjun develop after the marathon?", ["plantar fasciitis"], ["session_08"], "Arjun developed plantar fasciitis in his right heel."),
        ("s20", "test", "How much compute grant money did OpenCompute Foundation award for NeuroCompact v2?", ["25,000 dollars"], ["session_09"], "They were awarded a $25,000 research compute grant."),
        ("s21", "test", "Who was the research intern mentored by Arjun on memory-augmented transformers?", ["Ananya Bose"], ["session_09"], "Arjun mentored research intern Ananya Bose."),
        ("s22", "test", "On what street is the Aetheria Europe research office located in Zurich?", ["Technoparkstrasse"], ["session_12"], "The office is on Technoparkstrasse in Zurich."),
        ("s23", "test", "With which ETH Zurich professor did Arjun establish a joint research initiative?", ["Prof. Elena Vance"], ["session_13"], "Arjun established a joint initiative with Prof. Elena Vance."),
        ("s24", "test", "What is the name of the Bernese Mountain Dog puppy adopted by Arjun in Zurich?", ["Atlas"], ["session_16"], "Arjun adopted a puppy named Atlas."),
        ("s25", "dev", "Who was appointed as the full-time Chief Executive Officer of AgriVision in 2027?", ["Rajesh Nair"], ["session_17"], "Rajesh Nair was appointed full-time CEO.")
    ]

    for qid, split, text, kf, ss, exp in single_facts:
        questions.append({
            "id": qid,
            "category": "single_fact_retrieval",
            "split": split,
            "question": text,
            "is_answerable": True,
            "expected_answer": exp,
            "key_facts": kf,
            "supporting_sessions": ss
        })

    # 2. Multi-Hop Reasoning (25 questions)
    multi_hop = [
        ("m01", "dev", "What framework did the person who managed the cloud backend for AgriVision use?", ["FastAPI"], ["session_01", "session_03"], "Rohan Sharma was Arjun's roommate and backend engineer for AgriVision, which he built using FastAPI."),
        ("m02", "test", "In which city does the engineering manager who supervised Arjun's Go caching work reside?", ["Bengaluru"], ["session_02"], "Vikram Rao managed CacheMesh in Go at HyperScale Systems located in Bengaluru."),
        ("m03", "test", "What sport did Arjun play weekly with the person who developed the backend of AgriVision?", ["badminton"], ["session_03", "session_04"], "Arjun played badminton weekly with Rohan Sharma at PlayArena."),
        ("m04", "test", "What dietary restriction does the engineer who assisted Arjun with Android latency testing have?", ["peanut allergy"], ["session_03", "session_07"], "Meera Sen assisted with Android latency testing and has a severe peanut allergy."),
        ("m05", "test", "Where did the compiler engineer who collaborated on the Mojo kernels for NeuroCompact relocate?", ["Vancouver", "Canada"], ["session_06", "session_07"], "Kabir Das collaborated on Mojo kernels and relocated to Vancouver, Canada."),
        ("m06", "test", "What event caused Arjun to switch to swimming at the Padukone-Dravid Centre?", ["plantar fasciitis", "10k marathon"], ["session_08"], "Arjun developed plantar fasciitis running the Bengaluru 10k Marathon and was barred from high-impact running."),
        ("m07", "test", "At what conference was the paper describing on-device memory compression accepted as an Oral Presentation?", ["MLSys 2026"], ["session_07", "session_09"], "NeuroCompact v2 was accepted as an Oral Presentation at MLSys 2026."),
        ("m08", "test", "Who presented the paper at MLSys 2026 on behalf of Arjun after his US visa was delayed?", ["Dr. Shalini Verma"], ["session_05", "session_11"], "CTO Dr. Shalini Verma presented at MLSys 2026 in Santa Clara on his behalf."),
        ("m09", "test", "What veterinary procedures were completed for Kaalu before moving to Switzerland?", ["microchip", "rabies titer"], ["session_01", "session_10"], "Kaalu completed a pet microchip and rabies titer test on April 5, 2026."),
        ("m10", "dev", "What role did the intern who was mentored on memory-augmented transformers receive at Aetheria AI?", ["full-time Research Scientist"], ["session_09", "session_11"], "Ananya Bose was offered a full-time Research Scientist position at Aetheria AI."),
        ("m11", "test", "What operating system and dev tool did Arjun adopt for lab machines when collaborating with ETH Zurich?", ["NixOS", "Devbox"], ["session_13"], "Arjun adopted NixOS and Devbox for reproducible lab workstations."),
        ("m12", "test", "In which hospital in Chicago does Arjun's visiting sister work as a pediatric resident?", ["Northwestern Memorial Hospital"], ["session_14"], "Priya Mehta works at Northwestern Memorial Hospital in Chicago."),
        ("m13", "test", "What orthopedic injury did Arjun suffer while skiing on the Rothorn slope?", ["torn medial meniscus", "left knee"], ["session_14"], "He suffered a torn left medial meniscus on the Rothorn slope."),
        ("m14", "test", "What license was used for the open source release of the project that reached 6,200 GitHub stars?", ["Apache 2.0"], ["session_15"], "TensorForge was released under the Apache 2.0 open-source license."),
        ("m15", "test", "What award did the project that succeeded NeuroCompact win at EuroSys 2027?", ["Open Source Innovation Award"], ["session_06", "session_15"], "TensorForge won the EuroSys 2027 Open Source Innovation Award."),
        ("m16", "test", "Which organization overturned the 40 percent rent hike attempted by Arjun's Kreis 4 landlord?", ["Zurich Mieterinnenverband"], ["session_16"], "The Zurich Mieterinnenverband tenancy board overturned the illegal rent hike."),
        ("m17", "test", "From which animal shelter did Arjun adopt his Bernese Mountain Dog puppy?", ["Tierheim Strubeli", "Volketswil"], ["session_16"], "Atlas was adopted from Tierheim Strubeli in Volketswil."),
        ("m18", "test", "What vehicle did Arjun buy to transport Kaalu and Atlas to Lake Zurich parks?", ["cargo e-bike", "Riese & Müller"], ["session_16"], "He purchased a Riese & Müller Load 75 cargo e-bike."),
        ("m19", "test", "Who did Arjun serve as best man for at a wedding in Udaipur in August 2027?", ["Rohan Sharma", "Sneha Roy"], ["session_10", "session_17"], "Arjun served as best man for Rohan Sharma marrying Sneha Roy in Udaipur."),
        ("m20", "test", "Which venture capital firm led the 2.5 million dollar seed round for the SIH crop disease startup?", ["Peak XV Partners"], ["session_03", "session_17"], "Peak XV Partners led the $2.5M seed round for AgriVision."),
        ("m21", "test", "To which city did the SIH team member who developed the Flutter frontend relocate in 2027?", ["Singapore"], ["session_03", "session_17"], "Meera Sen relocated to Singapore to lead agronomy partnerships."),
        ("m22", "test", "What percentage equity grant did Arjun receive for serving on AgriVision's Technical Advisory Board?", ["1.5 percent"], ["session_17"], "Arjun received a 1.5% equity grant."),
        ("m23", "test", "How far was Arjun's Lake Zurich crossing swim that proved his left meniscus had rehabilitated?", ["3km", "58 minutes"], ["session_14", "session_18"], "Arjun swam the 3km Lake Zurich crossing in 58 minutes on August 15, 2027."),
        ("m24", "test", "Why did Arjun decline a 4 million dollar acquisition offer for TensorForge?", ["ad-tech", "behavioral tracking"], ["session_15", "session_18"], "He refused the $4M offer from an ad-tech syndicate to prevent use in behavioral tracking."),
        ("m25", "dev", "Under which academic advisor does Arjun plan to take a sabbatical to pursue an external PhD?", ["Prof. Elena Vance", "ETH Zurich"], ["session_13", "session_18"], "Arjun plans an external PhD under Prof. Elena Vance at ETH Zurich.")
    ]

    for qid, split, text, kf, ss, exp in multi_hop:
        questions.append({
            "id": qid,
            "category": "multi_hop_reasoning",
            "split": split,
            "question": text,
            "is_answerable": True,
            "expected_answer": exp,
            "key_facts": kf,
            "supporting_sessions": ss
        })

    # 3. Entity Relationships (25 questions)
    entity_rel = [
        ("e01", "dev", "What is the relationship between Rohan Sharma and Arjun Mehta?", ["college roommate", "friend", "co-founder"], ["session_01", "session_03", "session_04"], "Rohan was Arjun's college roommate at IIITM Gwalior, SIH teammate, and badminton partner."),
        ("e02", "test", "What is the relationship between Deepa Nair and Arjun at HyperScale Systems?", ["senior staff engineer", "collaborator"], ["session_02"], "Deepa Nair was the senior staff engineer collaborating with Arjun on CacheMesh."),
        ("e03", "test", "What role did Vikram Rao hold relative to Arjun Mehta?", ["engineering manager"], ["session_02"], "Vikram Rao was Arjun's engineering manager at HyperScale Systems."),
        ("e04", "test", "How is Meera Sen connected to Arjun's college and hackathon history?", ["hackathon teammate", "MatrixCrafters"], ["session_03"], "Meera Sen was Arjun's teammate in MatrixCrafters at SIH who built the Flutter frontend."),
        ("e05", "test", "What is the relationship between Dr. Shalini Verma and Arjun Mehta?", ["CTO", "manager"], ["session_05", "session_09"], "Dr. Shalini Verma is Arjun's CTO and research supervisor at Aetheria AI."),
        ("e06", "test", "How was Kabir Das affiliated with Arjun's research on NeuroCompact?", ["compiler engineer", "co-author"], ["session_06"], "Kabir Das was a compiler engineer at Aetheria AI who collaborated with Arjun on CUDA/Mojo kernels."),
        ("e07", "test", "Who is Sneha Roy, and what is her connection to Arjun's circle?", ["married Rohan Sharma", "fiancee"], ["session_10", "session_17"], "Sneha Roy is Rohan Sharma's wife; Arjun was best man at their wedding."),
        ("e08", "test", "What is the professional connection between Ananya Bose and Arjun Mehta?", ["intern", "mentee", "research scientist"], ["session_09", "session_11"], "Ananya Bose was Arjun's research intern whom he mentored into a full-time Research Scientist."),
        ("e09", "test", "What is the institutional relationship between Prof. Elena Vance and Arjun Mehta?", ["collaborator", "ETH Zurich", "prospective PhD advisor"], ["session_13", "session_18"], "Prof. Elena Vance is Arjun's research collaborator at ETH Zurich and prospective PhD advisor."),
        ("e10", "dev", "Who is Lukas Weber and what is his connection to Arjun's Zurich lab?", ["post-doctoral researcher", "PennyLane"], ["session_13"], "Lukas Weber is a post-doctoral researcher hired to work with Arjun on quantum simulations."),
        ("e11", "test", "What is the familial relationship between Priya Mehta and Arjun?", ["sister", "pediatric resident"], ["session_14"], "Priya Mehta is Arjun's sister, a doctor in Chicago."),
        ("e12", "test", "What is the relationship between Dr. Hans Keller and Arjun Mehta?", ["orthopedic doctor"], ["session_14"], "Dr. Hans Keller was Arjun's treating orthopedic doctor for his knee injury."),
        ("e13", "test", "What is the relationship between Kaalu and Atlas?", ["housemates", "pets", "dogs"], ["session_16"], "Kaalu and Atlas are Arjun's two pet rescue dogs who live and bond together."),
        ("e14", "test", "What is Rajesh Nair's relationship to Arjun's past project AgriVision?", ["Chief Executive Officer", "CEO"], ["session_17"], "Rajesh Nair was appointed full-time CEO of AgriVision while Arjun serves on the Advisory Board."),
        ("e15", "test", "What entity connected Arjun, Meera, and Rohan in 2024?", ["MatrixCrafters", "Smart India Hackathon"], ["session_03"], "Team MatrixCrafters at the Smart India Hackathon connected all three."),
        ("e16", "test", "What organization connects Dr. Shalini Verma and FAIR?", ["previous employer", "former research lead"], ["session_05"], "Dr. Shalini Verma previously led AI research at FAIR before becoming CTO of Aetheria AI."),
        ("e17", "test", "What entity connected Arjun and the OpenCompute Foundation in 2026?", ["compute grant", "25,000 dollars"], ["session_09"], "The OpenCompute Foundation awarded Arjun's lab a $25,000 research compute grant."),
        ("e18", "test", "What is the relationship between Aetheria AI and Technoparkstrasse in Zurich?", ["European research office", "workplace"], ["session_12"], "Aetheria AI's European research lab is located on Technoparkstrasse in Zurich."),
        ("e19", "test", "What is the connection between Peak XV Partners and AgriVision?", ["lead investor", "seed round"], ["session_17"], "Peak XV Partners led AgriVision's $2.5M seed funding round."),
        ("e20", "test", "What is the relationship between Tierheim Strubeli and Arjun's pet Atlas?", ["animal rescue shelter", "adoption origin"], ["session_16"], "Tierheim Strubeli in Volketswil is the animal rescue shelter from which Arjun adopted Atlas."),
        ("e21", "test", "What connection exists between Arjun and the Zurich Mieterinnenverband?", ["tenancy legal board", "overturned rent hike"], ["session_16"], "The Mieterinnenverband represented Arjun and overturned his landlord's illegal 40% rent hike."),
        ("e22", "test", "How is Northwestern Memorial Hospital related to Arjun's immediate family?", ["workplace of sister Priya"], ["session_14"], "It is the Chicago hospital where Arjun's sister Priya works as a pediatric resident."),
        ("e23", "test", "What is the institutional relationship between EuroSys 2027 and TensorForge?", ["Open Source Innovation Award"], ["session_15"], "EuroSys 2027 conferred the Open Source Innovation Award to TensorForge."),
        ("e24", "test", "What connection exists between Padukone-Dravid Centre and Arjun's sports life?", ["swimming pool", "rehabilitation"], ["session_08"], "It is the sports centre in Bengaluru where Arjun switched to swimming after his plantar fasciitis."),
        ("e25", "dev", "What relationship exists between PlayArena and Rohan Sharma?", ["badminton partner", "Tuesday evenings"], ["session_04"], "PlayArena is the venue where Rohan and Arjun played badminton every Tuesday evening.")
    ]

    for qid, split, text, kf, ss, exp in entity_rel:
        questions.append({
            "id": qid,
            "category": "entity_relationships",
            "split": split,
            "question": text,
            "is_answerable": True,
            "expected_answer": exp,
            "key_facts": kf,
            "supporting_sessions": ss
        })

    # 4. Temporal Reasoning (25 questions)
    temporal_reasoning = [
        ("t01", "dev", "Did Arjun move to HSR Layout before or after joining Aetheria AI?", ["before", "HSR Layout in Dec 2024", "Aetheria AI in Feb 2025"], ["session_04", "session_05"], "Arjun moved to HSR Layout in December 2024, whereas he joined Aetheria AI later in February 2025."),
        ("t02", "test", "Did the EMNLP paper rejection occur before or after the Bengaluru 10k Marathon?", ["before", "EMNLP August 2025", "Marathon November 2025"], ["session_07", "session_08"], "The EMNLP rejection happened in August 2025, before the marathon in November 2025."),
        ("t03", "test", "When did Arjun resign from HyperScale Systems relative to his start date at Aetheria AI?", ["January 2025", "February 1, 2025"], ["session_05"], "He resigned from HyperScale in January 2025 and started at Aetheria AI on February 1, 2025."),
        ("t04", "test", "How many months elapsed between Arjun's graduation in June 2022 and joining HyperScale Systems?", ["one month", "July 2022"], ["session_01", "session_02"], "One month elapsed: graduated June 2022, joined HyperScale July 2022."),
        ("t05", "test", "Did Arjun develop plantar fasciitis before or after his promotion to Lead Research Scientist?", ["before", "November 2025", "January 2026"], ["session_08", "session_09"], "Plantar fasciitis occurred in November 2025, and promotion occurred in January 2026."),
        ("t06", "test", "In what month and year did Arjun move into his apartment in Zurich Kreis 4?", ["September 2026", "September 15, 2026"], ["session_10", "session_12"], "He moved into Kreis 4 on September 15, 2026."),
        ("t07", "test", "Did Kabir Das leave Aetheria AI before or after the NeuroCompact paper was accepted at MLSys?", ["before", "Kabir left July 2025", "MLSys January 2026"], ["session_07", "session_09"], "Kabir Das left in July 2025, while MLSys oral acceptance happened in January 2026."),
        ("t08", "test", "Did Arjun adopt Kaalu before or after adopting Atlas?", ["before", "Kaalu 2024", "Atlas June 2027"], ["session_01", "session_16"], "Arjun had Kaalu in 2024 and adopted Atlas much later in June 2027."),
        ("t09", "test", "Did the Smart India Hackathon participation happen before or after the move to HSR Layout?", ["before", "SIH September 2024", "HSR December 2024"], ["session_03", "session_04"], "SIH took place in September 2024, before the move to HSR Layout in December 2024."),
        ("t10", "dev", "When did Rohan Sharma get engaged to Sneha Roy relative to Arjun's move to Zurich?", ["before", "engaged March 2026", "moved September 2026"], ["session_10", "session_12"], "Rohan got engaged in March 2026, before Arjun relocated to Zurich in September 2026."),
        ("t11", "test", "Did Arjun suffer his torn medial meniscus before or after moving to Zurich Kreis 2?", ["before", "injury January 2027", "moved June 2027"], ["session_14", "session_16"], "The knee injury happened in January 2027, before moving to Kreis 2 (Enge) in June 2027."),
        ("t12", "test", "Did TensorForge surpass 6,200 GitHub stars before or after Rohan's wedding in Udaipur?", ["before", "March 2027", "wedding August 2027"], ["session_15", "session_17"], "TensorForge achieved 6,200 stars in March 2027, whereas the wedding took place in August 2027."),
        ("t13", "test", "In what month and year did Arjun complete his 3km Lake Zurich Crossing swim?", ["August 2027", "August 15, 2027"], ["session_18"], "He completed the swim on August 15, 2027."),
        ("t14", "test", "Did the adoption of Atlas occur before or after the EuroSys 2027 conference?", ["after", "EuroSys March 2027", "Atlas June 2027"], ["session_15", "session_16"], "Atlas was adopted in June 2027, after EuroSys in March 2027."),
        ("t15", "test", "When did Arjun transition to a 100 percent vegan diet relative to moving to Kreis 2?", ["after", "moved June 2027", "vegan September 2027"], ["session_16", "session_18"], "He moved to Kreis 2 in June 2027 and transitioned to vegan in September 2027."),
        ("t16", "test", "Did Arjun's sister Priya visit him in Zermatt before or after AgriVision's 2.5 million seed round?", ["before", "Priya visit January 2027", "seed round August 2027"], ["session_14", "session_17"], "Priya visited in January 2027, before AgriVision raised its seed round in August 2027."),
        ("t17", "test", "How long was Arjun prohibited from high-impact running following his plantar fasciitis diagnosis?", ["3 months", "sports doctor"], ["session_08"], "The sports doctor forbade high-impact running for 3 months in November 2025."),
        ("t18", "test", "How long did Arjun's non-compete clause with HyperScale Systems last?", ["12 months", "one year"], ["session_05"], "The non-compete barred consulting for competing caching companies for 12 months."),
        ("t19", "test", "Did the decision to pivot NeuroCompact to on-device compression happen before or after hiring intern Ananya Bose?", ["before", "pivot August 2025", "intern January 2026"], ["session_07", "session_09"], "The pivot occurred in August 2025, before intern Ananya was hired in January 2026."),
        ("t20", "test", "On what date was Kaalu's rabies titer and pet microchip completed for travel to Switzerland?", ["April 5, 2026"], ["session_10"], "Kaalu completed his microchip and rabies titer on April 5, 2026."),
        ("t21", "test", "Did Arjun adopt Docker Compose before or after adopting NixOS and Devbox?", ["before", "Docker 2024", "NixOS late 2026"], ["session_02", "session_13"], "Docker Compose was adopted in 2024, while NixOS/Devbox was adopted in late 2026."),
        ("t22", "test", "Did Arjun switch from Mojo back to C++ before or after MLSys 2026?", ["before", "switched August 2025", "MLSys January 2026"], ["session_07", "session_09"], "He switched back to C++ in August 2025, before MLSys acceptance in January 2026."),
        ("t23", "test", "How long was the physical therapy program prescribed for Arjun's meniscus injury?", ["8 weeks", "Dr. Hans Keller"], ["session_14"], "Dr. Hans Keller prescribed 8 weeks of intensive physical therapy."),
        ("t24", "test", "In which year did Arjun aim to take an external PhD sabbatical at ETH Zurich?", ["2028", "late 2028"], ["session_18"], "Arjun planned to take his sabbatical in late 2028."),
        ("t25", "dev", "Did Arjun decline the 4 million dollar buyout offer before or after the Udaipur wedding?", ["after", "wedding August 2027", "offer decline late 2027"], ["session_17", "session_18"], "The wedding occurred in August 2027, and the offer decline occurred in late 2027.")
    ]

    for qid, split, text, kf, ss, exp in temporal_reasoning:
        questions.append({
            "id": qid,
            "category": "temporal_reasoning",
            "split": split,
            "question": text,
            "is_answerable": True,
            "expected_answer": exp,
            "key_facts": kf,
            "supporting_sessions": ss
        })

    # 5. Historical State Retrieval (25 questions)
    historical_state = [
        ("h01", "dev", "Where did Arjun live in Bengaluru before moving to HSR Layout Sector 2?", ["Indiranagar"], ["session_01", "session_04"], "Arjun lived in Indiranagar before moving to HSR Layout in December 2024."),
        ("h02", "test", "What was Arjun's job title when he worked on CacheMesh at HyperScale Systems?", ["Associate Software Engineer"], ["session_02"], "His title was Associate Software Engineer."),
        ("h03", "test", "What was the initial programming language used for CUDA execution kernels in NeuroCompact before reverting to C++?", ["Mojo"], ["session_06", "session_07"], "The kernels were initially rewritten in Mojo before switching back to C++."),
        ("h04", "test", "Which primary database did Arjun use for caching before phasing it out for Dragonfly and DuckDB?", ["Redis"], ["session_02", "session_11"], "Arjun used Redis cluster before replacing it with Dragonfly and DuckDB."),
        ("h05", "test", "What primary code editor did Arjun use in 2024 before adopting Zed for collaborative pairing?", ["Neovim", "Lua"], ["session_02", "session_15"], "He used Neovim configured with Lua."),
        ("h06", "test", "In which district or Kreis of Zurich did Arjun live when he first relocated to Switzerland?", ["Kreis 4"], ["session_12", "session_16"], "He lived in Zurich Kreis 4 before moving to Kreis 2 (Enge)."),
        ("h07", "test", "What was Arjun's job title at Aetheria AI prior to his promotion in January 2026?", ["Research Engineer"], ["session_05", "session_09"], "He was a Research Engineer before being promoted to Lead Research Scientist."),
        ("h08", "test", "What was the original research focus of NeuroCompact before pivoting to on-device compression?", ["speculative decoding", "server"], ["session_06", "session_07"], "It was initially a 4-bit quantizer for server-side speculative decoding."),
        ("h09", "test", "What sport did Arjun practice before his plantar fasciitis forced him to switch to swimming?", ["running", "marathon"], ["session_04", "session_08"], "He practiced high-impact long-distance running before switching to swimming."),
        ("h10", "dev", "What diet did Arjun follow in 2025 before transitioning to 100 percent vegan in 2027?", ["strict vegetarian"], ["session_04", "session_18"], "He followed a strict vegetarian diet that included Swiss dairy cheeses."),
        ("h11", "test", "Where did Arjun study German on Duolingo before enrolling at the Goethe-Institut Bengaluru?", ["Duolingo", "40 days"], ["session_04", "session_10"], "He used Duolingo for 40 days in 2024 before quitting and later joining Goethe-Institut."),
        ("h12", "test", "Which laptop did Arjun buy in 2024 for local model evaluations before adopting lab workstations?", ["16-inch MacBook Pro M3 Max"], ["session_03"], "He bought a 16-inch MacBook Pro M3 Max."),
        ("h13", "test", "What project name was TensorForge known as during its early research prototype phase?", ["NeuroCompact", "NeuroCompact v2"], ["session_06", "session_15"], "It was known as NeuroCompact and NeuroCompact v2."),
        ("h14", "test", "What was Rohan Sharma's relationship status before getting engaged in March 2026?", ["single", "unmarried"], ["session_01", "session_10"], "He was single and Arjun's bachelor roommate before getting engaged to Sneha Roy."),
        ("h15", "test", "Where did Meera Sen live and work before moving to Singapore in 2027?", ["Bengaluru", "India"], ["session_03", "session_17"], "She lived and worked in India (Bengaluru) before moving to Singapore."),
        ("h16", "test", "What desk setup did Arjun use before buying his electric sit-stand desk and Aeron chair?", ["standing setup"], ["session_14"], "He had a static standing desk setup before upgrading to an electric sit-stand desk."),
        ("h17", "test", "Who was the full-time leader of AgriVision before Rajesh Nair was hired as CEO?", ["student team", "Rohan Sharma", "Meera Sen"], ["session_03", "session_17"], "It was an alumni hackathon project led by Rohan, Meera, and Arjun before bringing in CEO Rajesh Nair."),
        ("h18", "test", "Where did Arjun live immediately prior to moving to Switzerland in September 2026?", ["HSR Layout Sector 2", "Bengaluru"], ["session_04", "session_12"], "He lived in HSR Layout Sector 2, Bengaluru."),
        ("h19", "test", "What was the initial conference target for the first NeuroCompact paper submission in 2025?", ["EMNLP 2025"], ["session_06", "session_07"], "The initial target was EMNLP 2025, which was rejected in August 2025."),
        ("h20", "test", "What was Arjun's salary counter-offer percentage from HyperScale Systems when he decided to leave?", ["30 percent raise"], ["session_05"], "HyperScale offered a 30% salary raise to retain him."),
        ("h21", "test", "What tea vessel did Arjun buy in Zurich because he missed Bengaluru chai?", ["brass tea kettle"], ["session_12"], "He bought a traditional brass tea kettle to brew authentic masala chai."),
        ("h22", "test", "What was the location of Arjun's hostel room during his B.Tech days at ABV IIITM Gwalior?", ["Hostel 3"], ["session_01"], "He lived in Hostel 3 with Rohan Sharma."),
        ("h23", "test", "Which local sports club did Arjun frequent every Tuesday evening in 2024 and 2025?", ["PlayArena"], ["session_04"], "He played badminton at PlayArena in Bengaluru."),
        ("h24", "test", "What was the original reason Arjun moved from Indiranagar to HSR Layout?", ["closer to Rohan", "badminton club"], ["session_04"], "He wanted to live closer to Rohan Sharma and their badminton club."),
        ("h25", "dev", "What switch type did Arjun select for his custom mechanical keyboard in Zurich?", ["Boba U4", "silent tactile"], ["session_13"], "He chose silent tactile Boba U4 switches for the shared lab.")
    ]

    for qid, split, text, kf, ss, exp in historical_state:
        questions.append({
            "id": qid,
            "category": "historical_state_retrieval",
            "split": split,
            "question": text,
            "is_answerable": True,
            "expected_answer": exp,
            "key_facts": kf,
            "supporting_sessions": ss
        })

    # 6. Knowledge Updates and Superseded Facts (20 questions)
    knowledge_updates = [
        ("u01", "dev", "Where does Arjun currently reside as of late 2027?", ["Zurich Enge", "Kreis 2", "Switzerland"], ["session_16"], "Arjun lives in Zurich Enge (Kreis 2), Switzerland."),
        ("u02", "test", "Does Arjun still use Redis for database and caching operations?", ["no", "phased out", "Dragonfly", "DuckDB"], ["session_02", "session_11"], "No, he completely phased out Redis in favor of Dragonfly and DuckDB."),
        ("u03", "test", "Does Arjun still live in HSR Layout Sector 2 in Bengaluru?", ["no", "relocated to Zurich"], ["session_04", "session_12"], "No, he relocated from Bengaluru to Zurich in September 2026."),
        ("u04", "test", "Does Arjun still work as a Research Engineer, or has his role changed?", ["promoted", "Lead Research Scientist"], ["session_05", "session_09"], "He was promoted to Lead Research Scientist at Aetheria AI in January 2026."),
        ("u05", "test", "What is Arjun's current dietary status as of late 2027?", ["100 percent plant-based", "vegan"], ["session_04", "session_18"], "He is 100% plant-based (vegan), having eliminated dairy cheeses."),
        ("u06", "test", "Is Mojo still the language used by Arjun for compiler kernels?", ["no", "switched back to C++"], ["session_06", "session_07"], "No, he abandoned Mojo due to unstable tooling and returned to C++."),
        ("u07", "test", "Does Arjun still have only one pet dog?", ["no", "two dogs", "Kaalu and Atlas"], ["session_01", "session_16"], "No, he now has two dogs: Kaalu and a puppy named Atlas."),
        ("u08", "test", "Is AgriVision still an unfunded hackathon prototype?", ["no", "2.5 million seed round", "Peak XV"], ["session_03", "session_17"], "No, it raised a $2.5M seed round led by Peak XV and operates commercially."),
        ("u09", "test", "Is Rohan Sharma still an unmarried bachelor living in Bengaluru?", ["no", "married Sneha Roy"], ["session_10", "session_17"], "No, he married Sneha Roy in Udaipur in August 2027."),
        ("u10", "dev", "What editor does Arjun now use for collaborative pair programming with Lukas Weber?", ["Zed"], ["session_02", "session_15"], "He now uses the Zed editor for remote real-time pair programming."),
        ("u11", "test", "Is NeuroCompact still an unpublished internal research project?", ["no", "MLSys 2026", "open-sourced as TensorForge"], ["session_09", "session_15"], "No, it was presented at MLSys 2026 and open-sourced as TensorForge."),
        ("u12", "test", "Is Kabir Das still working at Aetheria AI alongside Arjun?", ["no", "relocated to Vancouver", "July 2025"], ["session_06", "session_07"], "No, Kabir left Aetheria AI in July 2025 to relocate to Vancouver."),
        ("u13", "test", "Did Arjun permanently give up on learning German after dropping Duolingo?", ["no", "revived", "Goethe-Institut", "German B1"], ["session_04", "session_10", "session_12"], "No, he revived his goal, attended Goethe-Institut, and prepared for German B1 in Zurich."),
        ("u14", "test", "Does Arjun still live in Zurich Kreis 4?", ["no", "moved to Zurich Enge", "Kreis 2"], ["session_12", "session_16"], "No, he moved to Zurich Enge (Kreis 2) following a rent dispute."),
        ("u15", "test", "Is Arjun's left knee still immobilized by a torn medial meniscus?", ["no", "fully rehabilitated", "3km swim"], ["session_14", "session_18"], "No, his meniscus fully healed after physical therapy and he completed a 3km lake swim."),
        ("u16", "test", "Does Arjun still work for HyperScale Systems?", ["no", "resigned January 2025", "Aetheria AI"], ["session_02", "session_05"], "No, he resigned in January 2025 to join Aetheria AI."),
        ("u17", "test", "What package management and system environment does Arjun currently use on lab machines?", ["NixOS", "Devbox"], ["session_13"], "He uses NixOS and Devbox for reproducible builds across workstations."),
        ("u18", "test", "Is Ananya Bose still an intern at Aetheria AI?", ["no", "full-time Research Scientist"], ["session_09", "session_11"], "No, she completed her internship and became a full-time Research Scientist."),
        ("u19", "test", "Does Arjun still consume Swiss dairy cheeses as part of his diet?", ["no", "vegan since September 2027"], ["session_12", "session_18"], "No, he eliminated dairy cheeses when transitioning to a vegan diet in September 2027."),
        ("u20", "dev", "Has Arjun sold TensorForge to a commercial analytics firm?", ["no", "declined 4 million dollar buyout"], ["session_15", "session_18"], "No, he declined a $4M buyout offer to protect the project from behavioral tracking.")
    ]

    for qid, split, text, kf, ss, exp in knowledge_updates:
        questions.append({
            "id": qid,
            "category": "knowledge_updates",
            "split": split,
            "question": text,
            "is_answerable": True,
            "expected_answer": exp,
            "key_facts": kf,
            "supporting_sessions": ss
        })

    # 7. Contradiction Handling (20 questions)
    contradictions = [
        ("c01", "dev", "Does Arjun currently live in Indiranagar, HSR Layout, or Zurich?", ["Zurich", "moved September 2026"], ["session_01", "session_04", "session_12"], "Arjun currently lives in Zurich, Switzerland, having left Indiranagar in 2024 and HSR Layout in 2026."),
        ("c02", "test", "Does Arjun work for HyperScale Systems or Aetheria AI?", ["Aetheria AI"], ["session_02", "session_05"], "He works for Aetheria AI; he resigned from HyperScale in January 2025."),
        ("c03", "test", "Does Arjun drink coffee or masala chai?", ["masala chai", "no coffee", "migraines"], ["session_06"], "Arjun never drinks coffee due to migraines; he strictly drinks masala chai."),
        ("c04", "test", "Did Arjun accept HyperScale's 30 percent salary counter-offer to stay in Bengaluru?", ["declined counter-offer"], ["session_05"], "He declined HyperScale's 30% raise counter-offer and joined Aetheria AI."),
        ("c05", "test", "Does Arjun prefer Kubernetes or Docker Compose for local service development?", ["Docker Compose", "dislikes Kubernetes"], ["session_02"], "Arjun dislikes Kubernetes complexity and strictly prefers Docker Compose for local dev."),
        ("c06", "test", "Did the EMNLP 2025 submission on NeuroCompact get accepted or rejected?", ["rejected", "August 2025"], ["session_06", "session_07"], "It was rejected in August 2025 due to baseline benchmark leakage."),
        ("c07", "test", "Did Rohan Sharma complete the Bengaluru 10k Marathon alongside Arjun?", ["did not finish", "twisted ankle", "DNF"], ["session_08"], "No, Rohan twisted his ankle at km 4 and could not complete the marathon."),
        ("c08", "test", "Did Arjun attend MLSys 2026 in person in California, or did someone else present?", ["did not attend", "Dr. Shalini Verma presented", "visa delayed 221g"], ["session_11"], "He could not attend due to a 221g US visa delay; Dr. Shalini Verma presented on his behalf."),
        ("c09", "test", "Is Arjun's current diet omnivorous, vegetarian, or 100 percent vegan?", ["100 percent vegan", "plant-based"], ["session_04", "session_12", "session_18"], "His current diet is 100% vegan since September 2027."),
        ("c10", "dev", "Does Arjun own a private automobile in Zurich, or does he use public transit?", ["public transit", "S-Bahn", "Tram 4", "no car"], ["session_13"], "He does not own a car; he uses the S-Bahn and Tram Line 4, plus a cargo e-bike."),
        ("c11", "test", "Did Arjun undergo immediate surgery for his torn meniscus or conservative physical therapy?", ["physical therapy", "no surgery", "Dr. Hans Keller"], ["session_14"], "He avoided surgery and underwent 8 weeks of intensive physical therapy."),
        ("c12", "test", "Did Arjun keep the 10,000 Swiss Franc EuroSys award money as personal income?", ["donated", "STEM education"], ["session_15"], "He donated the entire 10,000 CHF prize to rural STEM education in India."),
        ("c13", "test", "Did Arjun's landlord in Kreis 4 succeed in enforcing the 40 percent rent increase?", ["overturned", "Mieterinnenverband"], ["session_16"], "No, the tenancy board overturned the illegal rent hike."),
        ("c14", "test", "Does Kaalu show aggression or hostility toward the new puppy Atlas?", ["bonded", "sleep together"], ["session_16"], "No, Kaalu and Atlas bonded warmly and sleep side by side."),
        ("c15", "test", "Is Rajesh Nair an engineering intern or the Chief Executive Officer at AgriVision?", ["Chief Executive Officer", "CEO"], ["session_17"], "Rajesh Nair is the full-time CEO of AgriVision."),
        ("c16", "test", "Did Arjun accept or reject the 4 million dollar buyout offer for TensorForge?", ["rejected", "declined buyout"], ["session_18"], "He rejected the $4M buyout offer to prevent surveillance tracking."),
        ("c17", "test", "Does Arjun plan to pursue his PhD at Stanford or ETH Zurich?", ["ETH Zurich", "Prof. Elena Vance"], ["session_18"], "He plans to pursue an external PhD at ETH Zurich under Prof. Elena Vance."),
        ("c18", "test", "Is TensorForge distributed under a proprietary commercial license or Apache 2.0?", ["Apache 2.0", "open source"], ["session_15"], "TensorForge is open-sourced under the Apache 2.0 license."),
        ("c19", "test", "Does Arjun prefer reading technical books on e-readers or physical books?", ["physical books"], ["session_01"], "Arjun prefers reading physical books over e-readers for technical subjects."),
        ("c20", "dev", "Will Arjun work on military surveillance software if offered high pay?", ["refuses", "strict policy against military"], ["session_11"], "No, he has a strict professional policy refusing closed-source military or defense surveillance contracts.")
    ]

    for qid, split, text, kf, ss, exp in contradictions:
        questions.append({
            "id": qid,
            "category": "contradiction_handling",
            "split": split,
            "question": text,
            "is_answerable": True,
            "expected_answer": exp,
            "key_facts": kf,
            "supporting_sessions": ss
        })

    # 8. Preferences and Constraints (20 questions)
    preferences = [
        ("p01", "dev", "What beverage does Arjun strictly consume, and why does he avoid coffee?", ["masala chai", "coffee causes migraines"], ["session_06"], "Arjun strictly drinks masala chai because coffee gives him migraines."),
        ("p02", "test", "What format does Arjun prefer for reading technical literature?", ["physical books", "over e-readers"], ["session_01"], "He prefers physical books over electronic readers for technical topics."),
        ("p03", "test", "What local development tooling does Arjun prefer over Kubernetes?", ["Docker Compose"], ["session_02"], "He prefers lightweight Docker Compose over Kubernetes complexity."),
        ("p04", "test", "What editor and configuration language did Arjun adopt as his primary development environment in 2024?", ["Neovim", "Lua"], ["session_02"], "He adopted Neovim configured with Lua."),
        ("p05", "test", "What severe dietary allergy must be respected when dining with Meera Sen?", ["peanut allergy"], ["session_03"], "Meera Sen has a severe, strict peanut allergy."),
        ("p06", "test", "What day and time does Arjun reserve for playing badminton?", ["Tuesday evening", "PlayArena"], ["session_04"], "He plays badminton every Tuesday evening at PlayArena."),
        ("p07", "test", "What professional constraint did Arjun's non-compete agreement with HyperScale impose?", ["12-month bar on consulting for competing caching companies"], ["session_05"], "It barred him from consulting for competing database and caching companies for 12 months."),
        ("p08", "test", "What medical constraint was imposed on Arjun's exercise routine in late 2025?", ["no high-impact running for 3 months", "plantar fasciitis"], ["session_08"], "He was forbidden from high-impact running for 3 months due to plantar fasciitis."),
        ("p09", "test", "What brand and model of smartwatch did Arjun choose for tracking his swimming?", ["Garmin Forerunner 265"], ["session_08"], "He bought a Garmin Forerunner 265 watch."),
        ("p10", "dev", "What strict professional ethical policy does Arjun maintain regarding defense contracts?", ["refuses military defense surveillance contracts"], ["session_11"], "He maintains a strict policy refusing closed-source military or defense surveillance contracts."),
        ("p11", "test", "What mode of urban transport does Arjun prefer in Zurich over car ownership?", ["S-Bahn", "Tram Line 4"], ["session_13"], "He prefers taking the Zurich S-Bahn and Tram Line 4 over owning a private car."),
        ("p12", "test", "What mechanical keyboard switch type does Arjun prefer for shared office spaces?", ["silent tactile Boba U4"], ["session_13"], "He prefers silent tactile Boba U4 switches."),
        ("p13", "test", "What ergonomic furniture did Arjun acquire following his skiing injury?", ["electric sit-stand desk", "Herman Miller Aeron chair"], ["session_14"], "He acquired an electric motorized sit-stand desk and Herman Miller Aeron chair."),
        ("p14", "test", "What keybinding scheme does Arjun insist on retaining in modern editors like Zed?", ["Neovim keybindings"], ["session_15"], "He insists on retaining Neovim keybindings."),
        ("p15", "test", "What constraint caused Arjun's landlord dispute in Zurich Kreis 4?", ["illegal 40 percent rent hike"], ["session_16"], "The landlord attempted an illegal 40% rent increase."),
        ("p16", "test", "What kind of bicycle does Arjun use to transport his dogs around Zurich?", ["Riese & Müller Load 75 cargo e-bike"], ["session_16"], "He uses a Riese & Müller Load 75 heavy-duty cargo e-bike."),
        ("p17", "test", "What database technologies does Arjun strictly use after phasing out Redis?", ["Dragonfly", "DuckDB"], ["session_11"], "He strictly uses Dragonfly and DuckDB."),
        ("p18", "test", "What dietary boundary did Arjun adopt in September 2027 regarding animal products?", ["100 percent plant-based vegan", "no dairy"], ["session_18"], "He adopted a 100% plant-based vegan diet, cutting out all dairy products."),
        ("p19", "test", "What travel obstacle prevented Arjun from speaking in person at MLSys 2026?", ["US visa 221g administrative delay"], ["session_11"], "His US B1/B2 visa was administratively delayed under section 221g."),
        ("p20", "dev", "What tea preparation item did Arjun purchase to preserve his Indian morning routine in Zurich?", ["traditional brass tea kettle"], ["session_12"], "He bought a traditional brass tea kettle to brew authentic masala chai at home.")
    ]

    for qid, split, text, kf, ss, exp in preferences:
        questions.append({
            "id": qid,
            "category": "preferences_and_constraints",
            "split": split,
            "question": text,
            "is_answerable": True,
            "expected_answer": exp,
            "key_facts": kf,
            "supporting_sessions": ss
        })

    # 9. Negative Information (20 questions)
    neg_info = [
        ("n01", "dev", "Does Arjun own an electric passenger car such as a Tesla in Switzerland?", ["does not own a car", "public transit", "cargo e-bike"], ["session_13"], "Arjun does not own a car; he uses the S-Bahn, Tram 4, and a cargo e-bike."),
        ("n02", "test", "Did Arjun accept employment with any defense or military contractors?", ["no", "refuses military contracts"], ["session_11"], "No, he has a strict policy against working on military or defense contracts."),
        ("n03", "test", "Did Arjun undergo knee surgery for his torn meniscus?", ["no surgery", "physical therapy"], ["session_14"], "No, he did not have surgery; he recovered through 8 weeks of intensive physical therapy."),
        ("n04", "test", "Does Arjun drink espresso, lattes, or drip coffee?", ["no coffee", "migraines", "only masala chai"], ["session_06"], "No, Arjun never drinks coffee because it triggers migraines."),
        ("n05", "test", "Did Arjun accept HyperScale Systems' retention raise offer?", ["no", "declined counter-offer"], ["session_05"], "No, he declined their 30% raise counter-offer and departed for Aetheria AI."),
        ("n06", "test", "Did Arjun sell the proprietary rights to TensorForge for 4 million dollars?", ["no", "declined buyout"], ["session_18"], "No, he rejected the $4M acquisition offer."),
        ("n07", "test", "Does Arjun currently eat dairy milk, butter, or cheese?", ["no", "vegan since September 2027"], ["session_18"], "No, he transitioned to a 100% plant-based vegan diet and eliminated all dairy."),
        ("n08", "test", "Did Arjun attend the MLSys 2026 conference in person in California?", ["no", "visa delayed 221g"], ["session_11"], "No, he was unable to attend due to a US visa delay under 221g."),
        ("n09", "test", "Did Kabir Das remain at Aetheria AI through 2026?", ["no", "left July 2025 to Vancouver"], ["session_07"], "No, Kabir left Aetheria AI in July 2025 and moved to Vancouver."),
        ("n10", "dev", "Did Rohan Sharma complete the Bengaluru 10k Marathon without injury?", ["no", "twisted ankle at km 4", "DNF"], ["session_08"], "No, Rohan twisted his ankle at kilometer 4 and failed to finish."),
        ("n11", "test", "Does Arjun use Kubernetes for managing local development containers?", ["no", "dislikes Kubernetes", "uses Docker Compose"], ["session_02"], "No, he dislikes Kubernetes and strictly uses Docker Compose locally."),
        ("n12", "test", "Did the EMNLP 2025 conference accept the initial NeuroCompact paper?", ["no", "rejected August 2025"], ["session_07"], "No, the EMNLP submission was rejected in August 2025."),
        ("n13", "test", "Does Arjun use Mojo for his current compiler research kernels?", ["no", "switched back to C++"], ["session_07"], "No, he abandoned Mojo due to tooling instability and returned to C++."),
        ("n14", "test", "Did Arjun keep the 10,000 Swiss Franc EuroSys prize money for personal spending?", ["no", "donated to rural STEM education"], ["session_15"], "No, he donated the entire 10,000 CHF award to underprivileged STEM education."),
        ("n15", "test", "Did Arjun lose his tenancy legal battle against the Kreis 4 landlord?", ["no", "won", "Mieterinnenverband overturned hike"], ["session_16"], "No, the tenancy board ruled in his favor and overturned the 40% rent increase."),
        ("n16", "test", "Does Arjun still use Redis in his modern software infrastructure?", ["no", "phased out", "uses Dragonfly and DuckDB"], ["session_11"], "No, Redis has been completely phased out of his stack."),
        ("n17", "test", "Did Arjun give up German permanently after stopping his Duolingo streak?", ["no", "enrolled Goethe-Institut", "B1 exam"], ["session_04", "session_10", "session_12"], "No, he revived his studies at the Goethe-Institut and is preparing for the B1 exam."),
        ("n18", "test", "Did Arjun leave Kaalu behind in India when relocating to Switzerland?", ["no", "Kaalu moved with him"], ["session_10", "session_12"], "No, Kaalu was microchipped and relocated with Arjun to Zurich."),
        ("n19", "test", "Is Rajesh Nair an intern or contractor at AgriVision without executive authority?", ["no", "Chief Executive Officer", "full-time CEO"], ["session_17"], "No, Rajesh Nair is the appointed full-time Chief Executive Officer."),
        ("n20", "dev", "Does Arjun prefer electronic Kindle readers over paper books for technical study?", ["no", "prefers physical books"], ["session_01"], "No, he explicitly prefers physical paper books over e-readers for technical subjects.")
    ]

    for qid, split, text, kf, ss, exp in neg_info:
        questions.append({
            "id": qid,
            "category": "negative_information",
            "split": split,
            "question": text,
            "is_answerable": True,
            "expected_answer": exp,
            "key_facts": kf,
            "supporting_sessions": ss
        })

    # 10. Unanswerable Questions (25 questions - all require strict abstention)
    unanswerable = [
        ("x01", "dev", "What was the brand and model of the first automobile Arjun purchased in college?", [], [], "Insufficient evidence provided. Arjun's college automobile is never mentioned."),
        ("x02", "test", "What score did Arjun achieve on his GRE or GMAT exam?", [], [], "Insufficient evidence provided. Arjun's GRE or GMAT scores are not recorded."),
        ("x03", "test", "What is the name of Arjun Mehta's mother and father?", [], [], "Insufficient evidence provided. The names of his parents are never mentioned."),
        ("x04", "test", "What was Arjun's cumulative GPA upon graduating from ABV IIITM Gwalior?", [], [], "Insufficient evidence provided. His cumulative GPA is not recorded."),
        ("x05", "test", "How much money does Arjun have invested in the stock market or Bitcoin?", [], [], "Insufficient evidence provided. Arjun's investment holdings are not recorded."),
        ("x06", "test", "What is the name of the high school Arjun attended in Chandigarh?", [], [], "Insufficient evidence provided. His high school name is never mentioned."),
        ("x07", "test", "What brand of badminton racket does Arjun use at PlayArena?", [], [], "Insufficient evidence provided. The brand of his badminton racket is not mentioned."),
        ("x08", "test", "What is the blood type of Arjun Mehta?", [], [], "Insufficient evidence provided. His blood type is never recorded."),
        ("x09", "test", "What was the exact numerical score Arjun obtained in his German B1 proficiency exam?", [], [], "Insufficient evidence provided. He was studying for B1, but an exam score is never mentioned."),
        ("x10", "dev", "What is the monthly rent Arjun paid for his apartment in HSR Layout Sector 2?", [], [], "Insufficient evidence provided. The rent amount for HSR Layout is not recorded."),
        ("x11", "test", "What is the color of the electric sit-stand desk Arjun purchased in Zurich?", [], [], "Insufficient evidence provided. The desk color is never specified."),
        ("x12", "test", "What is the maiden name of Sneha Roy?", [], [], "Insufficient evidence provided. Sneha Roy's family background is not recorded."),
        ("x13", "test", "What was the title of Arjun's undergraduate senior capstone thesis at IIITM Gwalior?", [], [], "Insufficient evidence provided. His capstone thesis title is never mentioned."),
        ("x14", "test", "How many kilometers does Arjun ride his cargo e-bike per week?", [], [], "Insufficient evidence provided. Weekly bicycle mileage is not recorded."),
        ("x15", "test", "What is the favorite film or movie franchise of Arjun Mehta?", [], [], "Insufficient evidence provided. Arjun's movie preferences are never mentioned."),
        ("x16", "test", "What is the name of the veterinary clinic in Zurich that treats Kaalu and Atlas?", [], [], "Insufficient evidence provided. The veterinary clinic name is not mentioned."),
        ("x17", "test", "How many total employees work at Aetheria AI worldwide?", [], [], "Insufficient evidence provided. Aetheria AI's total company headcount is not recorded."),
        ("x18", "test", "What was the exact model of Android phone used by Meera Sen to test NeuroCompact?", [], [], "Insufficient evidence provided. The specific Android phone model is not specified."),
        ("x19", "test", "What is the name of the flight airline Arjun flew from India to Switzerland in 2026?", [], [], "Insufficient evidence provided. The flight airline is never recorded."),
        ("x20", "test", "What musical instruments, if any, can Arjun play?", [], [], "Insufficient evidence provided. Musical abilities are not recorded."),
        ("x21", "test", "What is the name of the landlord who tried to hike Arjun's rent in Kreis 4?", [], [], "Insufficient evidence provided. The landlord's personal name is not mentioned."),
        ("x22", "test", "What is the exact square footage of Arjun's apartment in Zurich Enge?", [], [], "Insufficient evidence provided. The square footage is not recorded."),
        ("x23", "test", "What grade did intern Ananya Bose achieve in her master's degree coursework?", [], [], "Insufficient evidence provided. Ananya's academic grades are not mentioned."),
        ("x24", "test", "What is the brand of physical therapy resistance bands Arjun used for knee recovery?", [], [], "Insufficient evidence provided. The brand of resistance bands is not recorded."),
        ("x25", "dev", "What salary figure did the commercial ad-tech consortium offer Arjun personally in the buyout bid?", [], [], "Insufficient evidence provided. The buyout was for the project TensorForge ($4M), but personal salary was not mentioned.")
    ]

    for qid, split, text, kf, ss, exp in unanswerable:
        questions.append({
            "id": qid,
            "category": "unanswerable_questions",
            "split": split,
            "question": text,
            "is_answerable": False,
            "expected_answer": exp,
            "key_facts": kf,
            "supporting_sessions": ss
        })

    # 11. Cross-Session Synthesis (25 questions)
    cross_session = [
        ("xss01", "dev", "Trace Arjun's residential moves in chronological order from college graduation through 2027.", ["Indiranagar", "HSR Layout Sector 2", "Zurich Kreis 4", "Zurich Enge"], ["session_01", "session_04", "session_12", "session_16"], "Arjun lived in Indiranagar (2022-2024), moved to HSR Layout Sector 2 (Dec 2024), moved to Zurich Kreis 4 (Sep 2026), and moved to Zurich Enge (June 2027)."),
        ("xss02", "test", "Trace Arjun's complete job title progression across companies from July 2022 to 2027.", ["Associate Software Engineer at HyperScale", "Research Engineer at Aetheria AI", "Lead Research Scientist at Aetheria AI"], ["session_02", "session_05", "session_09"], "He was Associate Software Engineer at HyperScale Systems (2022-2025), joined Aetheria AI as Research Engineer (Feb 2025), and was promoted to Lead Research Scientist (Jan 2026)."),
        ("xss03", "test", "How did Arjun's relationship with the AgriVision project evolve from a hackathon entry to 2027?", ["SIH hackathon entry", "2nd runner-up", "seed round 2.5M", "Technical Advisory Board", "1.5% equity"], ["session_03", "session_17"], "It began as a 2024 hackathon project winning 2nd runner-up ($1,500), and evolved into a commercial startup raising $2.5M where Arjun sits on the Advisory Board with 1.5% equity."),
        ("xss04", "test", "Describe the full lifecycle of the NeuroCompact research from initial kernel implementation to open-source award.", ["4-bit quantizer", "EMNLP rejection", "on-device pivot", "MLSys 2026 Oral", "TensorForge", "EuroSys 2027 award"], ["session_06", "session_07", "session_09", "session_15"], "NeuroCompact started as a Mojo quantizer, was rejected at EMNLP, pivoted to on-device memory compression, won an Oral at MLSys 2026, and was open-sourced as TensorForge winning EuroSys 2027."),
        ("xss05", "test", "Detail Arjun's complete dietary evolution across the four years of recorded memory.", ["unrestricted", "strict vegetarian in Nov 2024", "Swiss cheeses in 2026", "100% vegan in Sep 2027"], ["session_04", "session_12", "session_18"], "He was non-vegetarian until November 2024, followed a strict vegetarian diet allowing cheese through 2026, and transitioned to 100% vegan in September 2027."),
        ("xss06", "test", "How did Arjun's language learning goals regarding German develop over time?", ["Duolingo 40 days dropped", "Goethe-Institut Bengaluru revived", "German B1 Zurich"], ["session_04", "session_10", "session_12"], "He dropped Duolingo after 40 days in 2024, revived intensive classes at Goethe-Institut in 2026, and prepared for German B1 in Zurich."),
        ("xss07", "test", "Explain the evolution of pets in Arjun's household between his time in Indiranagar and Zurich Enge.", ["Kaalu rescue indie dog", "microchip travel to Zurich", "adopted Atlas Bernese Mountain Dog"], ["session_01", "session_10", "session_16"], "He lived alone with Kaalu in Indiranagar, relocated Kaalu to Zurich in 2026, and adopted a second dog Atlas in Zurich Enge in 2027."),
        ("xss08", "test", "Trace the changes in Arjun's sports and athletic pursuits following health events.", ["badminton and marathon running", "plantar fasciitis", "swimming at Padukone-Dravid", "skiing meniscus tear", "3km Lake Zurich Crossing"], ["session_04", "session_08", "session_14", "session_18"], "He played badminton and ran a 10k, shifted to swimming due to plantar fasciitis, recovered from a ski meniscus tear, and completed a 3km lake swim."),
        ("xss09", "test", "What changes occurred in Arjun's database and data architecture preferences between 2024 and 2026?", ["Redis cluster and Go for CacheMesh", "phased out Redis", "Dragonfly and DuckDB"], ["session_02", "session_11"], "He initially built CacheMesh on Redis cluster, but later completely phased out Redis in favor of Dragonfly and DuckDB."),
        ("xss10", "dev", "How did Rohan Sharma's personal and romantic life progress across Arjun's memory logs?", ["college roommate", "badminton partner", "engaged to Sneha Roy", "married in Udaipur"], ["session_01", "session_04", "session_10", "session_17"], "Rohan was Arjun's college roommate and badminton partner, got engaged to Sneha Roy in March 2026, and married her in Udaipur in August 2027 with Arjun as best man."),
        ("xss11", "test", "Summarize all research grants, awards, and prizes won by Arjun or his projects.", ["SIH 2nd runner-up $1,500", "OpenCompute grant $25,000", "EuroSys Innovation Award 10,000 CHF"], ["session_03", "session_09", "session_15"], "AgriVision won $1,500 at SIH; NeuroCompact received a $25,000 OpenCompute grant; TensorForge won 10,000 CHF at EuroSys 2027."),
        ("xss12", "test", "How did Arjun's code editing and workstation development environment evolve from 2024 to 2027?", ["Neovim with Lua", "MacBook Pro M3 Max", "NixOS and Devbox", "electric sit-stand desk", "Zed editor"], ["session_02", "session_03", "session_13", "session_14", "session_15"], "He started with Neovim/Lua on an M3 MacBook, adopted NixOS/Devbox and a sit-stand desk in Zurich, and paired with Zed while retaining Neovim keys."),
        ("xss13", "test", "Which research collaborators did Arjun work with across Aetheria AI, ETH Zurich, and hackathons?", ["Rohan Sharma", "Meera Sen", "Kabir Das", "Dr. Shalini Verma", "Ananya Bose", "Prof. Elena Vance", "Lukas Weber"], ["session_03", "session_06", "session_09", "session_13"], "He collaborated with Rohan, Meera, Kabir, Dr. Shalini Verma, intern Ananya Bose, Prof. Elena Vance, and Lukas Weber."),
        ("xss14", "test", "What conferences and academic venues were targeted by Arjun's projects, and what were the outcomes?", ["EMNLP 2025 rejected", "MLSys 2026 Oral Presentation", "EuroSys 2027 award", "PyData London keynote"], ["session_07", "session_09", "session_15"], "EMNLP 2025 rejected; MLSys 2026 accepted as Oral; EuroSys 2027 awarded prize; PyData London invited keynote."),
        ("xss15", "test", "Summarize all instances where Arjun declined a substantial financial offer.", ["declined 30% raise counter-offer at HyperScale", "declined 4 million dollar buyout for TensorForge"], ["session_05", "session_18"], "He declined a 30% raise counter-offer from HyperScale and declined a $4M buyout offer from an ad-tech syndicate."),
        ("xss16", "test", "Detail the complete history of Kaalu from Bengaluru to Switzerland.", ["rescue indie dog in Indiranagar", "moved to HSR", "microchip and rabies titer April 2026", "moved to Kreis 4", "Lake Zurich walks", "Kreis 2 with Atlas"], ["session_01", "session_04", "session_10", "session_12", "session_16"], "Kaalu was an indie rescue in Indiranagar, moved to HSR, was cleared for travel in April 2026, settled in Zurich Kreis 4, and moved to Enge with puppy Atlas."),
        ("xss17", "test", "Trace Meera Sen's contributions, milestones, and relocations throughout the narrative.", ["Flutter frontend for AgriVision", "SIH $1,500 prize", "peanut allergy dinner", "Android latency testing", "relocated to Singapore"], ["session_03", "session_07", "session_17"], "Meera built AgriVision's Flutter app, assisted with NeuroCompact Android testing, and moved to Singapore to lead AgriVision's agronomy expansion."),
        ("xss18", "test", "Summarize the legal, regulatory, and tenancy challenges Arjun encountered.", ["12-month non-compete with HyperScale", "US visa 221g delay", "Kreis 4 landlord illegal 40% hike overturned"], ["session_05", "session_11", "session_16"], "He managed a 12-month non-compete with HyperScale, suffered a US visa 221g delay, and defeated an illegal 40% rent hike before the Zurich tenancy board."),
        ("xss19", "test", "How did Arjun maintain his Indian tea tradition across multiple homes in Bengaluru and Zurich?", ["strictly masala chai", "no coffee due to migraines", "bought traditional brass kettle in Zurich"], ["session_06", "session_12"], "He strictly drinks masala chai because coffee causes migraines, and bought a traditional brass tea kettle in Zurich to brew authentic chai."),
        ("xss20", "dev", "Trace the progression of Ananya Bose from her hiring to her employment status in 2026.", ["hired as research intern", "memory-augmented transformers", "full-time Research Scientist offer"], ["session_09", "session_11"], "She was hired and mentored by Arjun as a research intern in January 2026 and earned a full-time Research Scientist offer at Aetheria AI."),
        ("xss21", "test", "What role did Dr. Shalini Verma play in Arjun's career across his years at Aetheria AI?", ["CTO at Aetheria AI", "collaborator on NeuroCompact", "promoted Arjun to Lead Research Scientist", "offered Zurich leadership", "presented at MLSys on his behalf"], ["session_05", "session_06", "session_09", "session_10", "session_11"], "She was his CTO, co-researcher on NeuroCompact, promoted him to Lead Research Scientist, offered him the Zurich lab leadership, and presented at MLSys when his visa was delayed."),
        ("xss22", "test", "Describe the medical diagnosis, treatment, and recovery timeline of Arjun's knee injury in 2027.", ["torn left medial meniscus in Zermatt", "Dr. Hans Keller prescribed 8 weeks PT in Oerlikon", "sit-stand desk recovery", "swam 3km Lake Zurich Crossing fully healed"], ["session_14", "session_18"], "Tore his medial meniscus in Zermatt on Jan 8, 2027, completed 8 weeks of PT in Oerlikon, and proved full recovery by swimming the 3km Lake Zurich crossing on Aug 15, 2027."),
        ("xss23", "test", "How did Arjun's philanthropic giving manifest across his research milestones?", ["donated 10,000 Swiss Francs EuroSys prize to rural STEM education in India"], ["session_15"], "He donated his entire 10,000 CHF EuroSys award money to underprivileged STEM education in rural India."),
        ("xss24", "test", "Trace Arjun's academic aspirations from graduating college in 2022 to his future PhD plans in 2028.", ["B.Tech CS ABV IIITM Gwalior 2022", "published MLSys Oral", "planned external PhD at ETH Zurich under Prof. Elena Vance"], ["session_01", "session_09", "session_18"], "Graduated with B.Tech in 2022, published at MLSys in 2026, and planned an external PhD sabbatical at ETH Zurich in late 2028 under Prof. Elena Vance."),
        ("xss25", "dev", "How did Arjun transport himself in Bengaluru versus how he transports himself in Zurich?", ["commuted in Bengaluru", "walked Kaalu", "Zurich S-Bahn and Tram 4", "Riese & Müller cargo e-bike"], ["session_01", "session_12", "session_13", "session_16"], "In Bengaluru he lived near work and badminton, while in Zurich he explicitly used public transit (S-Bahn/Tram 4) and a heavy-duty cargo e-bike.")
    ]

    for qid, split, text, kf, ss, exp in cross_session:
        questions.append({
            "id": qid,
            "category": "cross_session_synthesis",
            "split": split,
            "question": text,
            "is_answerable": True,
            "expected_answer": exp,
            "key_facts": kf,
            "supporting_sessions": ss
        })

    # 12. Complex Multi-Fact Questions (25 questions)
    complex_multi_fact = [
        ("cmf01", "dev", "List Arjun's undergraduate university, graduation month and year, college roommate, and college hostel number.", ["ABV IIITM Gwalior", "June 2022", "Rohan Sharma", "Hostel 3"], ["session_01"], "Arjun graduated from ABV IIITM Gwalior in June 2022, roomed with Rohan Sharma, and lived in Hostel 3."),
        ("cmf02", "test", "Identify the first company Arjun worked for, his job title, his engineering manager, and the caching project he built.", ["HyperScale Systems", "Associate Software Engineer", "Vikram Rao", "CacheMesh"], ["session_02"], "At HyperScale Systems, he was Associate Software Engineer under Vikram Rao and built CacheMesh."),
        ("cmf03", "test", "For the Smart India Hackathon project, state the project name, the team name, the prize won, and the role Meera Sen played.", ["AgriVision", "MatrixCrafters", "2nd runner-up $1,500", "Flutter mobile frontend"], ["session_03"], "Team MatrixCrafters built AgriVision, won 2nd runner-up ($1,500), and Meera Sen built the Flutter mobile frontend."),
        ("cmf04", "test", "Name the neighborhood Arjun moved to in late 2024, the sport he played weekly, his teammate, and the sports venue.", ["HSR Layout Sector 2", "badminton", "Rohan Sharma", "PlayArena"], ["session_04"], "He moved to HSR Layout Sector 2, playing badminton weekly with Rohan Sharma at PlayArena."),
        ("cmf05", "test", "Identify the company Arjun joined in February 2025, his starting role, the company's CTO, and the CTO's previous employer.", ["Aetheria AI", "Research Engineer", "Dr. Shalini Verma", "FAIR"], ["session_05"], "He joined Aetheria AI as Research Engineer under CTO Dr. Shalini Verma, who previously led research at FAIR."),
        ("cmf06", "test", "State the name of the quantizer project Arjun developed at Aetheria AI, its original target conference, and the compiler engineer collaborator.", ["NeuroCompact", "EMNLP 2025", "Kabir Das"], ["session_06"], "Arjun developed NeuroCompact with compiler engineer Kabir Das targeting EMNLP 2025."),
        ("cmf07", "test", "Explain why the EMNLP submission was rejected, what month Kabir Das left Aetheria AI, and what city Kabir moved to.", ["baseline benchmark leakage", "July 2025", "Vancouver"], ["session_07"], "The paper was rejected for benchmark leakage; Kabir Das left in July 2025 and moved to Vancouver."),
        ("cmf08", "test", "State Arjun's 10k marathon time, the injury he sustained, the recovery duration barring high-impact running, and his replacement sport.", ["54 minutes", "plantar fasciitis", "3 months", "swimming at Padukone-Dravid"], ["session_08"], "He finished in 54 minutes, developed plantar fasciitis barring running for 3 months, and switched to swimming."),
        ("cmf09", "test", "Identify the conference that accepted NeuroCompact v2 as an Oral, the compute grant amount received, and Arjun's new job title in January 2026.", ["MLSys 2026", "$25,000", "Lead Research Scientist"], ["session_09"], "Accepted as an Oral at MLSys 2026, received a $25,000 compute grant, and was promoted to Lead Research Scientist."),
        ("cmf10", "dev", "State the destination city of Arjun's international relocation, the date Kaalu's medical travel tests were done, and the woman Rohan Sharma got engaged to.", ["Zurich", "April 5, 2026", "Sneha Roy"], ["session_10"], "Arjun relocated to Zurich; Kaalu completed travel tests on April 5, 2026; Rohan Sharma got engaged to Sneha Roy."),
        ("cmf11", "test", "Identify the two databases Arjun strictly uses after phasing out Redis, the visa section causing his travel delay, and who presented at MLSys in his place.", ["Dragonfly", "DuckDB", "221g", "Dr. Shalini Verma"], ["session_11"], "He uses Dragonfly and DuckDB, suffered a 221g visa delay, and Dr. Shalini Verma delivered his presentation."),
        ("cmf12", "test", "Name the street of Aetheria's Zurich lab, the Kreis where Arjun first rented an apartment, the tea vessel he bought, and the German exam he studied for.", ["Technoparkstrasse", "Kreis 4", "brass tea kettle", "German B1"], ["session_12"], "Located on Technoparkstrasse; rented in Kreis 4; bought a brass tea kettle; studied for German B1."),
        ("cmf13", "test", "State the ETH Zurich professor Arjun collaborated with, the post-doc hired, the reproducible OS tooling adopted, and the tram line used.", ["Prof. Elena Vance", "Lukas Weber", "NixOS and Devbox", "Tram Line 4"], ["session_13"], "Collaborated with Prof. Elena Vance, hired post-doc Lukas Weber, adopted NixOS/Devbox, and used Tram Line 4."),
        ("cmf14", "test", "Identify Arjun's sister, her profession, her hospital, the slope where Arjun injured his knee, and the injury sustained.", ["Priya Mehta", "pediatric resident", "Northwestern Memorial Hospital", "Rothorn slope in Zermatt", "torn left medial meniscus"], ["session_14"], "Sister Priya is a pediatric resident at Northwestern; Arjun tore his left medial meniscus on the Rothorn slope in Zermatt."),
        ("cmf15", "test", "State the open-source project name succeeding NeuroCompact, its license, its GitHub stars milestone, and the conference award won in Geneva.", ["TensorForge", "Apache 2.0", "6,200 stars", "EuroSys 2027 Open Source Innovation Award"], ["session_15"], "TensorForge was released under Apache 2.0, reached 6,200 stars, and won the EuroSys 2027 Innovation Award."),
        ("cmf16", "test", "Name the new Zurich neighborhood Arjun moved to in June 2027, the puppy he adopted, the puppy's breed, and the shelter of adoption.", ["Zurich Enge", "Atlas", "Bernese Mountain Dog", "Tierheim Strubeli in Volketswil"], ["session_16"], "Moved to Zurich Enge, adopted Bernese Mountain Dog puppy Atlas from Tierheim Strubeli in Volketswil."),
        ("cmf17", "test", "State the wedding city where Rohan married Sneha, the seed round amount raised by AgriVision, the lead VC firm, and the new CEO's name.", ["Udaipur", "$2.5 million", "Peak XV Partners", "Rajesh Nair"], ["session_17"], "Rohan married Sneha in Udaipur; AgriVision raised $2.5M from Peak XV Partners and appointed Rajesh Nair CEO."),
        ("cmf18", "test", "State the date Arjun went vegan, the lake swimming event he completed, his swim time, and the PhD advisor he plans to study under.", ["September 1, 2027", "Lake Zurich Crossing", "58 minutes", "Prof. Elena Vance"], ["session_18"], "Went vegan September 1, 2027, swam Lake Zurich Crossing in 58 minutes, and plans an external PhD under Prof. Elena Vance."),
        ("cmf19", "test", "Detail the two sports injuries Arjun suffered across 2025 and 2027, specifying the body part, the activity causing each, and the rehabilitation method.", ["plantar fasciitis in right heel from marathon", "swimming", "torn left medial meniscus from skiing", "8 weeks physical therapy"], ["session_08", "session_14"], "Plantar fasciitis in right heel from 10k marathon treated with swimming; torn left meniscus from skiing treated with 8 weeks PT in Oerlikon."),
        ("cmf20", "dev", "List the three pet animals or dogs living in Arjun's life, their names, breeds or origins, and the year each entered his narrative.", ["Kaalu indie rescue in 2024", "Atlas Bernese Mountain Dog puppy in 2027"], ["session_01", "session_16"], "Kaalu is an indie rescue dog living with Arjun since 2024; Atlas is a Bernese Mountain Dog puppy adopted in 2027."),
        ("cmf21", "test", "Identify the three cities where Arjun has lived or had legal residences since graduating college, and the respective years.", ["Chandigarh birth", "Bengaluru 2022 to 2026", "Zurich 2026 to 2027"], ["session_01", "session_12", "session_16"], "Born in Chandigarh, lived in Bengaluru (2022-2026), and lived in Zurich (2026-present)."),
        ("cmf22", "test", "List the four editors, development tools, or container platforms explicitly preferred or adopted by Arjun across his career.", ["Neovim with Lua", "Docker Compose", "NixOS and Devbox", "Zed editor"], ["session_02", "session_13", "session_15"], "He used Neovim/Lua, Docker Compose, NixOS/Devbox, and the Zed editor."),
        ("cmf23", "test", "Identify the two hackathon teammates from college who built AgriVision with Arjun, their specific technical roles, and their post-2026 whereabouts.", ["Rohan Sharma managed FastAPI backend married in Udaipur", "Meera Sen built Flutter frontend moved to Singapore"], ["session_03", "session_17"], "Rohan managed FastAPI backend and lives in India; Meera built Flutter frontend and moved to Singapore."),
        ("cmf24", "test", "State the three distinct reasons why Arjun declined three major opportunities: HyperScale counter-offer, military contracts, and TensorForge acquisition.", ["declined HyperScale to join generative AI research", "refuses military contracts on ethical grounds", "declined TensorForge buyout to prevent surveillance tracking"], ["session_05", "session_11", "session_18"], "Declined HyperScale for research startup; refuses military work on ethics; declined TensorForge buyout to stop ad surveillance."),
        ("cmf25", "dev", "Detail the exact specifications of Arjun's primary workstation hardware: his 2024 laptop model, his smartwatch model, his keyboard switch model, and his office chair model.", ["16-inch MacBook Pro M3 Max", "Garmin Forerunner 265", "silent tactile Boba U4 switches", "Herman Miller Aeron chair"], ["session_03", "session_08", "session_13", "session_14"], "16-inch MacBook Pro M3 Max, Garmin Forerunner 265 smartwatch, Boba U4 keyboard switches, and Herman Miller Aeron chair.")
    ]

    for qid, split, text, kf, ss, exp in complex_multi_fact:
        questions.append({
            "id": qid,
            "category": "complex_multi_fact",
            "split": split,
            "question": text,
            "is_answerable": True,
            "expected_answer": exp,
            "key_facts": kf,
            "supporting_sessions": ss
        })

    return questions


if __name__ == "__main__":
    qs = create_questions()
    dev_count = sum(1 for q in qs if q["split"] == "dev")
    test_count = sum(1 for q in qs if q["split"] == "test")
    print(f"Total questions generated: {len(qs)}")
    print(f"  Dev Split:  {dev_count}")
    print(f"  Test Split: {test_count}")

    # Export dataset JSONs
    with open(OUT_DIR / "story_data_expanded.json", "w", encoding="utf-8") as f:
        json.dump(ALL_SESSIONS, f, indent=2)

    with open(OUT_DIR / "ground_truth_expanded.json", "w", encoding="utf-8") as f:
        json.dump(qs, f, indent=2)

    print("Successfully exported story_data_expanded.json and ground_truth_expanded.json.")

