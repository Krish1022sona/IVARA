from src.memory import add_memory

memories = [
    ("I started learning Python in 2024.", "2024-06-10", "fact", 0.8),
    ("I built a Tic Tac Toe neural network from scratch.", "2025-07-10", "project", 0.9),
    ("I trained an LSTM model on Tata Motors stock data.", "2025-08-01", "project", 0.9),
    ("I learned PyTorch for deep learning.", "2025-07-20", "skill", 0.8),
    ("I implemented an Iris classifier neural network from scratch.", "2025-07-25", "project", 0.8),
    ("I studied Transformers and attention mechanisms.", "2025-07-08", "learning", 0.8),
    ("I learned about LSTMs and recurrent neural networks.", "2025-07-09", "learning", 0.8),
    ("I built a Flutter movie app using the TMDB API.", "2025-05-12", "project", 0.7),
    ("I learned C++ and practiced data structures.", "2024-12-01", "skill", 0.7),
    ("I solved many LeetCode problems using dynamic programming.", "2025-06-15", "achievement", 0.8),
    ("I practiced sliding window and two pointer techniques.", "2025-06-20", "learning", 0.7),
    ("I contributed to coding projects with GitHub.", "2025-06-25", "skill", 0.6),
    ("I started learning web development.", "2026-06-01", "learning", 0.8),
    ("I learned Tailwind CSS and responsive design.", "2026-06-10", "skill", 0.7),
    ("I play games on my PS5.", "2025-04-01", "hobby", 0.6),
    ("I have played Ghost of Tsushima.", "2025-04-05", "hobby", 0.5),
    ("I enjoy playing Wuthering Waves.", "2025-05-01", "hobby", 0.5),
    ("I play guitar and piano.", "2025-03-01", "hobby", 0.6),
    ("I am learning to play the flute.", "2025-03-15", "hobby", 0.5),
    ("I built a biometric attendance system using Arduino.", "2025-02-10", "project", 0.9),
    ("The attendance system used a fingerprint sensor and RTC.", "2025-02-15", "project", 0.8),
    ("I worked on a smart walking cane project with GPS and fall detection.", "2025-01-20", "project", 0.8),
    ("I enjoy learning difficult problems just for fun.", "2024-08-01", "preference", 0.7),
    ("I bought a router for my hostel room.", "2025-01-05", "random", 0.3),
    ("I like science and experimenting with new technology.", "2024-09-01", "preference", 0.7),
]

for text, timestamp, memory_type, importance in memories:
    add_memory(
        text,
        timestamp,
        "day1_test",
        memory_type,
        importance
    )