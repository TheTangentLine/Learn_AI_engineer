**The Problem: The "Goldfish Memory"**
You might have noticed that if you paste a 500-page PDF into the prompt, it gets expensive or hits the "Context Window" limit. **RAG (Retrieval Augmented Generation) solves this by only retrieving the 1% of data relevant to the question** and sending that to the LLM.

To do this, we need a way to search for "meaning", not keywords.

- _Keyword Search:_ "Dog" matches "Dog".
- _Semantic Search:_ "Dog" matches "Puppy" and "Canine" and "Pet".

**Week 2, Day 1: Embeddings (The "Hash" of Meaning)**

Today, we are going to interact with the **second most important API endpoint:** `embeddings`.

**The Concept:** An Embedding is a function that takes text and returns a **Vector** (a list of floats).

- `Input: "The King"` -> `Output: [0.9, 0.1, ...]`
- `Input: "The Queen"` -> `Output: [0.89, 0.12, ...]` (Very similar numbers)
- `Input: "The Apple"` -> `Output: [-0.5, 0.8, ...]` (Totally different numbers)

**The Math (Cosine Similarity)**: To see if two sentences are similar, we don't look at words. We calculate the angle between their vectors. If the angle is 0, they are identical.

---

**The Code: Your First Semantic Search Engine**
We will use the **NumPy** skills you prepped on Day 0.

Create `week2_day1_embeddings.py`.

```python
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. The Function to get the Vector
# This uses a different model (text-embedding-3-small) which is dirt cheap.
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# 2. The Math (Cosine Similarity)
# This is the formula Vector DBs use under the hood.
def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

if __name__ == "__main__":
    # Our "Database" of 3 documents
    documents = [
        "The food was delicious and the service was great.", # ID 0
        "The dog barked loudly at the mailman.",            # ID 1
        "I need to fix the Python bug in the server code."  # ID 2
    ]

    query = "I am hungry" # Notice: This word appears in NONE of the documents.

    print("--- Generating Embeddings... ---")

    # 3. Turn everything into numbers
    doc_vectors = [get_embedding(doc) for doc in documents]
    query_vector = get_embedding(query)

    # 4. Compare the query against every doc
    scores = []
    for i, doc_vec in enumerate(doc_vectors):
        score = cosine_similarity(query_vector, doc_vec)
        scores.append(score)
        print(f"Doc {i} Similarity: {score:.4f}")

    # 5. Find the winner
    best_doc_index = np.argmax(scores)

    print("\n--- Result ---")
    print(f"Query: '{query}'")
    print(f"Most Relevant Doc: '{documents[best_doc_index]}'")
```

**Run it and observe:**

Your query "I am hungry" will match "The food was delicious..." with a high score (e.g.,` 0.5`), while the dog and python docs will be low (e.g., `0.1`). **This works even though the word 'hungry' is not in the document**. The AI understands the concept.
