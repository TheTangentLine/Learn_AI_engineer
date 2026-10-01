Welcome to **Week 3: The Tuner Layer (Fine-Tuning).**

We are now leaving the "Application Layer" (APIs/RAG) and entering the "Model Layer." Up until now, we have treated the LLM as a frozen black box. If it didn't speak the way you wanted, you had to beg it in the System Prompt.

**Fine-Tuning** is opening the brain and rewiring it.

---

**The Concept: Training vs. Fine-Tuning**
Think of the model as a student:

**1. Pre-Training (Google/Meta does this):** The student goes to school for 12 years. They learn English, Math, Science, and History. They are "smart" but generic. Cost: $100M.

**2. Fine-Tuning (You do this):** You send the student to a 2-week coding bootcamp. They don't learn new English, but they learn to apply their intelligence specifically to writing Python. Cost: $10.

**When to use which?**

- **Use RAG** for **Facts** ("What is my bank balance?"). The model doesn't know your private data.

- **Use Fine-Tuning** for **Form/Style** ("Speak like a pirate", "Output JSON without yapping", "Write SQL in this specific dialect").

---

**The Breakthrough: LoRA (Low-Rank Adaptation)**

**The Problem:** Llama 3 has 8 Billion parameters. To fine-tune it normally, you need to update all 8 Billion numbers. This requires massive GPU clusters (hundreds of GBs of VRAM).

**The Solution: LoRA**. Instead of changing the whole brain, we attach a tiny little "adapter" sticker to the brain and only train the sticker.

**- Base Model:** Frozen (0 parameters change).
**- Adapter:** Trainable (Only 1% of parameters).
**- Result:** You can train a state-of-the-art model on a single free Google Colab GPU.
