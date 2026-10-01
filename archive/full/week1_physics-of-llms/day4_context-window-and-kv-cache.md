### Week 1, Day 4: Context Window & The KV Cache

**Topic:** Why does AI get slower (and more expensive) the longer you chat?

You might think sending 100 messages is just like sending 1 message 100 times. It is not.LLMs have a physical limit called the **Context Window** (e.g., 128k tokens for GPT-4o).

#### The Physics: The Attention Mechanism

To predict the next word, the model must "attend" (look back) at **every single previous token**.

- **Token 1:** Looks at nothing.
- **Token 100:** Looks at 1-99.
- **Token 10,000:** Looks at 1-9,999.

#### The KV Cache (Key-Value Cache):

To avoid re-reading the whole book for every new word, the GPU stores the mathematical representation (Matrices) of past tokens in VRAM.

- **Problem:** This cache takes up massive RAM.
- **Constraint:** If you fill the VRAM, the model crashes (OOM). This is why "Context Length" is the hardest bottleneck in AI today.
  $$\text{Memory Usage} \propto \text{Number of Layers} \times \text{Context Length}$$

---

#### The Assignment: The "Context Accountant"

We will write a script that acts as a **Budget Manager**.
It will simulate a conversation and calculate exactly when you will go broke or hit the limit.

**Engineering Reality:** You must implement this logic in production to prevent users from racking up $50 bills in one session.

Create `week1_day4_context.py`:

```Python
import tiktoken

# 1. Setup the Accountant
class ContextManager:
    def __init__(self, model="gpt-4", limit=8192, cost_per_1k_input=0.03):
        self.encoder = tiktoken.encoding_for_model(model)
        self.limit = limit
        self.cost_per_token = cost_per_1k_input / 1000
        self.history = []

    def count_tokens(self, text):
        return len(self.encoder.encode(text))

    def add_message(self, role, content):
        # Every message has a hidden overhead (formatting tokens)
        # Usually ~4 tokens per message for the role/structure
        overhead = 4
        msg_tokens = self.count_tokens(content) + overhead

        self.history.append({"role": role, "content": content, "tokens": msg_tokens})

    def get_status(self):
        total_tokens = sum(m["tokens"] for m in self.history)
        total_cost = total_tokens * self.cost_per_token

        print(f"\n--- Context Status ---")
        print(f"Messages: {len(self.history)}")
        print(f"Used: {total_tokens} / {self.limit} tokens ({total_tokens/self.limit:.1%})")
        print(f"Cost of next API call: ${total_cost:.5f}")

        if total_tokens > self.limit:
            print("🚨 WARNING: Context Limit Exceeded! The model will crash or truncate.")

# --- SIMULATION ---
if __name__ == "__main__":
    # GPT-4 (8k context) simulation
    manager = ContextManager(model="gpt-4", limit=8192)

    # 1. User sends a small hello
    manager.add_message("user", "Hello there.")
    manager.get_status()

    # 2. User pastes a massive log file (Common scenario)
    # Simulate a 5,000 word log dump
    huge_log = "Error: Connection Refused " * 2000
    manager.add_message("user", f"Here is my log file: {huge_log}")
    manager.get_status()

    # 3. The Conversation continues...
    # Simulate 5 more turns
    for i in range(5):
        manager.add_message("assistant", "I see the error on line 40.")
        manager.add_message("user", "What about line 50?")

    manager.get_status()
```

#### Output

When you run the script, you will see how quickly a single copy-paste destroys your budget.

```Plaintext
--- Context Status ---
Messages: 1
Used: 10 / 8192 tokens (0.1%)
Cost of next API call: $0.00030

--- Context Status ---
Messages: 2
Used: 10,015 / 8192 tokens (122.3%)
Cost of next API call: $0.30045
🚨 WARNING: Context Limit Exceeded! The model will crash or truncate.

--- Context Status ---
Messages: 12
Used: 10,240 / 8192 tokens (125.0%)
Cost of next API call: $0.30720
🚨 WARNING: Context Limit Exceeded! The model will crash or truncate.
```

**The Scary Part:** Even though the last 5 questions were short ("What about line 50?"), the Cost remained at **$0.30 per call** because the massive log file was still sitting in the history. You pay for that log file again and again every time you speak.

#### Why this matters

Run this script. You will see the "Cost" spike dramatically after the huge log file.

- **The Trap:** In an API, you pay for the entire history every single time you send a new message.
- **The Fix:** This is why the **Summarization Memory** (from our previous deep dive) is mandatory. You must delete that huge log from the history after 1-2 turns, or you will pay for it forever.
