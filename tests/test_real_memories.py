from src.extraction.extractor import extract_knowledge


memories = [
    "I worked with Rahul on our SIH project.",
    "Rahul is my best friend.",
    "My sister helped me debug my code.",
    "I built IVARA using Python.",
    "I collaborated with my team on the biometric attendance system."
]


for text in memories:

    print("=" * 100)
    print(text)
    print()

    result = extract_knowledge(text)

    print(result.model_dump_json(indent=2))
    print()