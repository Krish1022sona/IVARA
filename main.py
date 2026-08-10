from src.memory import add_memory
from src.retrieval import retrieve


add_memory(
    "I started learning Python in 2024.",
    "2024-06-10",
    "conversation_001",
    "fact",
    0.8
)

add_memory(
    "I built a neural network for Tic Tac Toe.",
    "2025-07-10",
    "conversation_002",
    "project",
    0.9
)

add_memory(
    "I learned PyTorch and used it for machine learning projects.",
    "2025-08-15",
    "conversation_003",
    "skill",
    0.8
)


results = retrieve("What ML projects have I built?", k=3)

for memory in results:
    print(memory)