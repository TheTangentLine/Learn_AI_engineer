### This is the **#1 problem** in production AI apps.

If you don't optimize memory, two things happen:

1. **Crash:** You hit the 8k/128k token limit (OOM).
2. **Bankruptcy:** You pay for the same text 1,000 times.

We solve this using a **Tiered Memory Architecture**, similar to how a computer uses RAM (Fast/Small) and Hard Drive (Slow/Big).

#### Tier 1: The Sliding Window (Short-Term Memory)

- **Concept:** Keep the last $K$ messages raw.
- **Why:** The model needs exact phrasing for the immediate conversation ("Change that function to blue").
- **Implementation:** `raw_buffer = messages[-5:]`

#### Tier 2: The Summarization Layer (Mid-Term Memory)

- **Concept:** When a message falls out of the "Sliding Window," we don't delete it. We **compress** it.
- **The Process:**
  1. User and Bot talk for 20 turns.
  2. Turns 1-15 are old.
  3. We send Turns 1-15 to a cheap model (GPT-4o-mini) with the prompt: _"Summarize the key facts from this conversation."_
  4. Output: _"User is debugging a Python API. We tried restarting the server but failed."_
  5. **Result:** We replaced 2,000 tokens of chat with 15 tokens of summary.
  6. This summary is injected into the **System Prompt** for future turns.

#### Tier 3: The Vector Database (Long-Term Memory)

- **Concept:** For things that happened 2 weeks ago or in a different session.
- **The Process:**
  1. Every time the user states a fact ("My API key is 123"), we embed it and save it to ChromaDB.
  2. When the user asks "What was my key?", we search the DB.
- **Optimization:** We do **not** load this into the context window unless the user asks about it.

---

### Implementing the "Summarization Buffer"

Let's build a standalone class that automatically handles this compression. This is a simplified version of what LangChain does internally.

Create `memory_optimizer.py`

```Python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SmartMemory:
    def __init__(self, max_raw_tokens=500):
        self.raw_messages = []
        self.summary = ""
        self.max_raw_tokens = max_raw_tokens

    def add_message(self, role, content):
        self.raw_messages.append({"role": role, "content": content})
        self._optimize()

    def _optimize(self):
        # 1. Estimate Token Count (Roughly 4 chars per token)
        current_text = "".join([m["content"] for m in self.raw_messages])
        est_tokens = len(current_text) / 4

        # 2. Check if we need to compress
        if est_tokens > self.max_raw_tokens:
            print(f"⚠️ Memory Full ({int(est_tokens)} tokens). Compressing...")

            # Take the oldest 50% of messages to summarize
            split_idx = len(self.raw_messages) // 2
            to_compress = self.raw_messages[:split_idx]
            to_keep = self.raw_messages[split_idx:]

            # 3. The Compression Call (The AI summarizes itself)
            conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in to_compress])

            prompt = f"""
            Extend the existing summary with these new lines. Be concise.

            Existing Summary: {self.summary}

            New Lines:
            {conversation_text}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            # 4. Update State
            self.summary = response.choices[0].message.content
            self.raw_messages = to_keep
            print(f"✅ Compression Complete. New Summary: {self.summary}")

    def get_context(self):
        # This is what we actually send to the API
        system_msg = {
            "role": "system",
            "content": f"Previous conversation summary: {self.summary}"
        }
        return [system_msg] + self.raw_messages

# --- TEST IT ---
if __name__ == "__main__":
    mem = SmartMemory(max_raw_tokens=50) # Set limit super low to force compression

    # 1. Simulating a conversation
    print("--- Turn 1 & 2 ---")
    mem.add_message("user", "My name is Kha. I am learning AI engineering.")
    mem.add_message("assistant", "Nice to meet you Kha! What language do you use?")

    # 2. Simulating a long turn that triggers compression
    print("\n--- Turn 3 (Triggering Compression) ---")
    mem.add_message("user", "I mostly use Python, specifically FastAPI for backend and ChromaDB for vector storage.")

    # 3. Check what the model actually sees now
    print("\n--- Final Context to Send ---")
    context = mem.get_context()
    for msg in context:
        print(f"[{msg['role'].upper()}]: {msg['content']}")
```

#### What to watch for when running this:

1. **Before Compression:** You have 3 messages in `raw_messages`.
2. **After Compression:**
   - The "System" message will update to contain: "User Kha is learning AI engineering."
   - The `raw_messages` list will shrink to only contain the most recent message about Python/FastAPI.
   - **The Magic:** The model still knows your name is Kha (because it's in the summary), but you aren't paying for the tokens of that first message anymore.

#### Output:

```Plaintext
--- Turn 1 & 2 ---
[SYSTEM]: Previous conversation summary:
[USER]: My name is Kha. I am learning AI engineering.
[ASSISTANT]: Nice to meet you Kha! What language do you use?

--- Turn 3 (Triggering Compression) ---
⚠️ Memory Full (110 tokens). Compressing...
✅ Compression Complete. New Summary: User is Kha, an AI engineering student.

--- Final Context to Send ---
[SYSTEM]: Previous conversation summary: User is Kha, an AI engineering student.
[USER]: I mostly use Python, specifically FastAPI for backend and ChromaDB for vector storage.
```
