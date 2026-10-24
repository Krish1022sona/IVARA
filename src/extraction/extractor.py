from typing import Literal
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
    source: str = Field(description="Source entity name or mention")
    relation: str = Field(description="Relationship predicate, e.g. worked_with, worked_on, uses, owns, friend_of, parent_of")
    target: str = Field(description="Target entity name or mention")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    origin: Literal["explicit", "inferred"] = Field(description="Strictly 'explicit' or 'inferred'")
    evidence: str | None = Field(default=None, description="Exact text span from the memory that supports this relationship")


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
5. Mark a relationship origin strictly as:
   - explicit: directly stated or unambiguously expressed
   - inferred: logically inferred but not explicitly stated
6. Do not invent facts.
7. Preserve the original meaning of the memory.
8. Use SELF when the memory refers to the person whose memory is being stored.
9. Confidence must represent confidence in the extraction, not importance.
10. Return only the requested structured data.
11. For explicit relationships, provide the exact supporting text span from the memory as evidence. For inferred relationships, provide the text span from which the inference was drawn, or null if inferred from overall context.
12. In the 'origin' field, put strictly 'explicit' or 'inferred'. Never put the evidence text span in 'origin'; put the text span strictly in 'evidence'.
13. Keep your internal chain-of-thought concise (under 50 words) and immediately output the structured JSON.
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
            "temperature": 0,
            "num_ctx": 16384,
            "num_predict": 4096
        }
    )

    return Extraction.model_validate_json(
        response.message.content
    )
