from src.extraction.pipeline import process_and_store_memory


def print_header(title):
    print()
    print("╔" + "═" * 68 + "╗")
    print(f"║{title:^68}║")
    print("╚" + "═" * 68 + "╝")


def print_section(title):
    print()
    print(f"┌── {title} " + "─" * (62 - len(title)))


def main():

    text = "I worked with Rahul on our SIH project."

    print_header("IVARA — SEMANTIC MEMORY PIPELINE")

    print_section("INPUT MEMORY")
    print(f'  "{text}"')

    memory = process_and_store_memory(
        text=text,
        timestamp="2026-09-01",
        source="day2_demo",
        memory_type="fact",
        importance=0.8
    )

    print_section("ENTITIES")

    for entity in memory["entities"]:

        if entity["status"] == "matched":
            status = f"✓ {entity['entity_id']}"
        else:
            status = "○ candidate"

        print(
            f"  {entity['mention']:<25}"
            f"→ {status}"
        )

    print_section("RELATIONSHIPS")

    for relationship in memory["relationships"]:

        source = relationship["source"]
        target = relationship["target"]

        if source["type"] == "entity":
            source_name = source["id"]
        else:
            source_name = source["mention"]

        if target["type"] == "entity":
            target_name = target["id"]
        else:
            target_name = f"[candidate: {target['mention']}]"

        print(
            f"  {source_name}"
            f" ──{relationship['predicate']}──> "
            f"{target_name}"
        )

        print(
            f"      confidence={relationship['confidence']:.2f}"
            f"  |  origin={relationship['origin']}"
        )

    print_section("TOPICS")

    for topic in memory["topics"]:
        print(
            f"  {topic['name']:<35}"
            f"confidence={topic['confidence']:.2f}"
        )

    print_section("PERSISTENCE")

    print(f"  ✓ Structured memory stored")
    print(f"  Memory ID: {memory['id']}")
    print(f"  Source:    {memory['source']}")

    print()
    print("═" * 70)
    print("  RAW TEXT → EXTRACTION → RESOLUTION → STRUCTURE → PERSISTENCE")
    print("═" * 70)
    print()


if __name__ == "__main__":
    main()