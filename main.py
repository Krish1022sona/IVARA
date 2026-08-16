from src.memory import add_memory

memory = add_memory(
    "I built an LSTM using PyTorch.",
    "2026-08-16",
    "day2_test",
    "project",
    0.9,
    entities=[
        {
            "name": "LSTM",
            "type": "Technology",
            "confidence": 1.0
        },
        {
            "name": "PyTorch",
            "type": "Technology",
            "confidence": 1.0
        }
    ],
    topics=[
        {
            "name": "Machine Learning",
            "confidence": 0.95
        }
    ],
    relationships=[
        {
            "source": "User",
            "relation": "uses",
            "target": "PyTorch",
            "confidence": 1.0,
            "origin": "explicit"
        }
    ]
)

print(memory)