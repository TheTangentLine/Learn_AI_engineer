Welcome to **Week 2, Day 3: The Full RAG Pipeline.**

Today is the day where it all clicks together.

**- Week 1:** You learned how to talk to the LLM (`client.chat.completions`).
**- Week 2, Day 2:** You learned how to retrieve relevant data (`collection.query`).

**Today, we glue them together.** This specific pattern you are about to write is responsible for billions of dollars in enterprise value right now. It is the "Chat with your PDF" architecture.

---

**The Architecture: "Stuffing the Prompt"**
RAG is actually very simple:

1. User asks: "_What is the vacation policy?_"
2. Vector DB finds: "_Employees get 4 weeks paid leave._"
3. **The Glue:** We paste that text into the System Prompt.

- _System Prompt:_ "You are a helpful assistant. Use ONLY the following context to answer the user: 'Employees get 4 weeks paid leave.'"
- _User Prompt:_ "What is the vacation policy?"

4. LLM answers: "_Based on the context, you get 4 weeks._"

---

**The Code: The RAG Engine**
We will reuse the ChromaDB you set up yesterday.

Create `week2_day3_rag.py.`

```python
import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- SETUP (Same as Day 2) ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./my_vector_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
collection = chroma_client.get_collection(name="company_handbook", embedding_function=openai_ef)

# --- THE RAG FUNCTION ---
def ask_company_policy(question):
    # 1. RETRIEVAL (The Search)
    # We ask for the top 3 most relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    # Extract the raw text from the results
    # results['documents'] is a list of lists (because you can query multiple things at once)
    retrieved_chunks = results['documents'][0]

    # 2. AUGMENTATION (The Glue)
    # We join the chunks into one big string to paste into the prompt
    context_text = "\n\n".join(retrieved_chunks)

    # The Prompt Engineering trick: "Strict Context"
    system_prompt = f"""
    You are a helpful HR assistant.
    You will be given a QUESTION and a set of CONTEXT documents.

    RULES:
    1. Answer the question using ONLY the context provided.
    2. If the answer is not in the context, say "I don't know based on the handbook."
    3. Do not use outside knowledge (e.g. don't make up laws).

    CONTEXT:
    {context_text}
    """

    # 3. GENERATION (The Answer)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content

# --- EXECUTION ---
if __name__ == "__main__":
    # Test 1: Something in the DB (from yesterday's examples)
    q1 = "How many weeks off do I get?"
    print(f"User: {q1}")
    print(f"AI: {ask_company_policy(q1)}\n")

    # Test 2: Something NOT in the DB
    q2 = "What is the capital of France?"
    print(f"User: {q2}")
    print(f"AI: {ask_company_policy(q2)}")
```

**Why this code is "Production Grade"**
**1. The "I Don't Know" Rule:**

- Try asking "What is the capital of France?".
- Without RAG, GPT would say "Paris".
- With RAG + The Strict System Prompt, it will say **"I don't know based on the handbook."**
- _This is critical for business apps._ You don't want your banking bot giving cooking advice.

**2. Context Injection:**

- We dynamically built the `system_prompt` string inside the function. This is how you make the LLM "smart" about your data without training it.
