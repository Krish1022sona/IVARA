import time
import json
import ollama

prompt = """You are a strict, objective evaluation judge assessing whether an AI assistant response accurately answers a user question according to a reference gold answer.

Question: How often do I see Dr. Johnson?
Reference Gold Answer: unanswerable
Candidate Prediction: I do not have sufficient recorded memories to determine how often you see Dr. Johnson.
Is Abstention Required: True

Rubric:
1. For answerable questions: Return is_correct: true if the prediction correctly contains the factual information stated in the gold answer (including semantic equivalents, spelled-out numbers vs digits). Return is_correct: false if factually wrong, contradicts evidence, or fails to answer.
2. For unanswerable questions: Return is_correct: true if the model correctly identifies that information is missing or unanswerable. Return is_correct: false if it hallucinates an answer.

Respond in JSON format:
{"is_correct": true, "reasoning": "short 1-sentence reason"}
"""

t0 = time.time()
res = ollama.chat(
    model="VladimirGav/gemma4-26b-16GB-VRAM:latest",
    messages=[{"role": "user", "content": prompt}],
    format="json",
    options={"temperature": 0}
)
t1 = time.time()
print(f"Time: {t1-t0:.2f}s")
print("Result:", res.message.content)
