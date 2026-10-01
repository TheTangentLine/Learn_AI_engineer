Yesterday, you calculated cosine similarity manually using NumPy. That works for 100 documents. **The Problem:** If you have 10 million documents, looping through them to calculate the dot product (`O(N)`) is too slow. **The Solution:** Vector Databases. They use algorithms like **HNSW** (Hierarchical Navigable Small World) to find the nearest neighbors in `O(log N)` time. It’s the "B-Tree index" of the AI world.

---

Today, we build a **Persistent Memory System** for your AI.

**The Tool: ChromaDB**
We will use **ChromaDB**. It is the "SQLite" of Vector DBs:

- Open Source.
- Runs locally (no sign-up required).
- Saves data to a folder on your disk.

**Prerequisite:** `pip install chromadb`

---

**The Code: Your First RAG Pipeline**
We are going to build a system that reads a "Company Handbook" (text), indexes it, and allows you to ask questions about it.

Create `week2_day2_vectordb.py`:

```python
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# 1. Setup ChromaDB
# "persistent" means data is saved to disk. "ephemeral" would be RAM-only.
chroma_client = chromadb.PersistentClient(path="./my_vector_db")

# 2. Setup the Embedding Function
# Chroma handles the API calls to OpenAI for you automatically.
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# 3. Create (or Get) a Collection
# Think of a "Collection" like a SQL "Table".
collection = chroma_client.get_or_create_collection(
    name="company_handbook",
    embedding_function=openai_ef
)

# --- INGESTION (Write Path) ---
def add_documents():
    print("--- Indexing Documents ---")

    # In a real app, you would read these from PDF/Text files
    documents = [
        "The company allows remote work on Tuesdays and Thursdays.",
        "To reset your password, visit the IT portal at it.example.com.",
        "Employees get 4 weeks of paid vacation per year.",
        "The coffee machine is on the 3rd floor break room.",
        "Emergency fire exits are located at the North and South wings."
    ]

    # We must provide unique IDs for every chunk
    ids = [str(i) for i in range(len(documents))]

    # .add() does 3 things:
    # 1. Calls OpenAI to get embeddings.
    # 2. Stores the text.
    # 3. Indexes the vector.
    collection.add(
        documents=documents,
        ids=ids
    )
    print(f"Indexed {len(documents)} documents.")

# --- RETRIEVAL (Read Path) ---
def query_db(question):
    print(f"\nQuestion: '{question}'")

    # We ask for the top 2 most relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=2
    )

    # Chroma returns a complex object, let's parse it
    for i, doc in enumerate(results['documents'][0]):
        print(f"  [Match {i+1}]: {doc}")

if __name__ == "__main__":
    # Uncomment this line only once to load data!
    # add_documents()

    # Now query it
    query_db("How much holiday do I get?")
    query_db("Where can I get caffeine?")
```

**Run Instructions:**
**1. First Run:** Uncomment `add_documents()` and run the script. You will see a new folder `my_vector_db` appear in your project.

**2. Second Run:** Comment out `add_documents()` (so you don't duplicate data) and run it again. It will query the data from the disk.

**What just happened?** You asked "Where can I get caffeine?". The word "caffeine" is not in the database.

1. Chroma converted "caffeine" to a vector.
2. It found the vector for "coffee machine" was the closest neighbor.
3. It returned that document

---

**Concept: Chunking (The Critical Step)**

In the example above, our documents were single sentences. Real life isn't that nice. You will have a 50-page PDF. You **cannot** embed a 50-page PDF as one vector. The meaning gets "diluted" (the vector becomes a muddy average of everything).

**You must split the text into chunks.**

**- Chunk Size:** How many characters per chunk (e.g., 500 chars).
**- Overlap:** How much to repeat (e.g., 50 chars) so you don't cut a sentence in half.
