from src.retrieval import retrieve

queries = [
    "What ML projects have I built?",
    "When did I start learning Python?",
    "What have I done with PyTorch?",
    "What deep learning models have I worked on?",
    "What programming projects have I built?",
    "What hobbies do I have?",
    "What games do I play?",
    "What things have nothing to do with programming?",
    "What hardware projects have I made?",
    "What have I done with web development?"
]

def main():
    for query in queries:
        print("\nQUERY:", query)

        results = retrieve(query, k=3)

        for i, memory in enumerate(results, 1):
            print(f"{i}. {memory['text']}")


if __name__ == "__main__":
    main()
