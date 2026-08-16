from ollama import chat
from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str
    type: str
    confidence: float


class Topic(BaseModel):
    name: str
    confidence: float


class Relationship(BaseModel):
    source: str
    relation: str
    target: str
    confidence: float
    origin: str


class Extraction(BaseModel):
    entities: list[Entity]
    topics: list[Topic]
    relationships: list[Relationship]


SYSTEM_PROMPT = """
You are the semantic memory extraction system for IVARA.

Your job is to extract structured knowledge from a personal memory.

Rules:

1. Extract meaningful entities mentioned or clearly implied by the memory.
2. An entity may have multiple types.
3. Extract useful topics.
4. Extract relationships between entities.
5. Mark a relationship as:
   - explicit: directly stated or unambiguously expressed
   - inferred: logically inferred but not explicitly stated
6. Do not invent facts.
7. Preserve the original meaning of the memory.
8. Use SELF when the memory refers to the person whose memory is being stored.
9. Confidence must represent confidence in the extraction, not importance.
10. Return only the requested structured data.
"""


def extract_knowledge(text: str) -> Extraction:

    response = chat(
        model="gpt-oss",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Extract knowledge from this memory:\n\n{text}"
            }
        ],
        format=Extraction.model_json_schema(),
        options={
            "temperature": 0
        }
    )

    return Extraction.model_validate_json(
        response.message.content
    )