**Week 4, Day 4: The Final Capstone (Architecture)**

You have now touched every part of the stack.

- **CS Foundation:** Python, API, JSON.

- **AI Foundation:** Embeddings, Vectors, RAG.

- **Model Layer:** Fine-Tuning, LoRA.

- **Ops Layer:** Inference Servers, Evals.

**Your Final Mission:**
Design the architecture for a "Personal Knowledge Brain."

**The Blueprint:**

1. **Ingestion Service (Week 2):** A Python script that watches a folder. When you drop a PDF, it chunks it and puts it in ChromaDB.
2. **The Brain (Week 3):** A Fine-Tuned Llama 3 model that speaks in your style (e.g., "Concise, Bullet points only").
3. **The Server (Week 4):** Ollama hosting the model.
4. **The App (Week 4):** Streamlit UI.

**The Workflow:**
User asks -> Streamlit -> Search ChromaDB (RAG) -> Send Context + Query to Ollama (Fine-Tuned) -> User sees answer.

---

**Graduation**
You have completed the **1-Month Refreshment for AI Engineering Roadmap**.
