from sentence_transformers import SentenceTransformer
import faiss


# --------------------------------------------------
# 1. Real memories
# --------------------------------------------------

memories = [
    "I built a Tic Tac Toe neural network using Python.",
    "I created an LSTM model for stock price prediction.",
    "I built an Iris flower classifier using machine learning.",
    "I worked with Rahul on our SIH project.",
    "I learned about computer networks and TCP/IP.",
    "I built a simple calculator using C++.",
    "I used torch to build amazing projects"
]


# --------------------------------------------------
# 2. Load real embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 3. Convert memories → embeddings
# --------------------------------------------------

embeddings = model.encode(
    memories,
    convert_to_numpy=True
).astype("float32")


# --------------------------------------------------
# 4. Create FAISS index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


# --------------------------------------------------
# 5. User query
# --------------------------------------------------

query = "What did i use PyTorch for?"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
).astype("float32")


# --------------------------------------------------
# 6. FAISS similarity search
# --------------------------------------------------

distances, indices = index.search(
    query_embedding,
    k=3
)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\n" + "=" * 70)
print("                    RAG MEMORY SEARCH")
print("=" * 70)

print(f'\nQuery:')
print(f'  "{query}"')

print("\nTop 3 Retrieved Memories:\n")

for rank, (idx, distance) in enumerate(
    zip(indices[0], distances[0]),
    start=1
):
    print(f"  {rank}. ✓ {memories[idx]}")
    print(f"     FAISS distance: {distance:.4f}\n")

print("=" * 70)
print("                 SEARCH COMPLETE ✓")
print("=" * 70)