As a developer, you might think "Prompt Engineering" is just "typing better English." It is not. It is **programming via natural language**. You are managing the "Context Buffer" (the RAM of the LLM) to force specific behaviors.

Today we will cover the two most important patterns that separate "toy apps" from "production apps": **Few-Shot Prompting** and **Chain of Thought**.

---

**Concept 1: Few-Shot Prompting (The "Unit Test" Pattern)**

If you ask a model to do something complex, it often fails because it doesn't know the specific format you want.

- **Zero-Shot (What you did yesterday)**:
  "Convert this address to JSON: 123 Main St, Apt 4, NY, NY."

  - _Risk_: It might give keys like `street_address`, or `address_line_1`, or include `zip_code: null`. It's unpredictable.

- **Few-Shot (The Engineering Way)**: You provide "Input -> Output" examples inside the prompt. This forces the model to pattern-match your desired logic. It is essentially **In-Context Learning**.

**The Code**

Create `day2_fewshot.py`. We will build a **Log Parser** that turns messy text logs into strict JSON severity levels.

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """
You are a log parsing engine.
You map raw log messages to a JSON object with keys: "severity" (INFO, WARN, ERROR) and "summary".
"""

# FEW-SHOT EXAMPLES
# We "teach" the model the logic by showing, not just telling.
examples = """
Input: [2024-01-01] User 554 failed login (password incorrect)
Output: {"severity": "WARN", "summary": "Failed login attempt"}

Input: [2024-01-01] Connection refused on port 8080. Service down.
Output: {"severity": "ERROR", "summary": "Service outage on port 8080"}

Input: [2024-01-01] Cron job executed successfully.
Output: {"severity": "INFO", "summary": "Cron job success"}
"""

def parse_log(log_line):
    # We construct the final prompt by combining instructions + examples + actual input
    final_prompt = f"{examples}\nInput: {log_line}\nOutput:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.0 # Strict mode for data extraction
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # A tricky ambiguity
    log = "Disk usage is at 91%, performance might degrade soon."
    print(parse_log(log))

```

**Why this matters**: Without the examples, the model might label 91% disk usage as "INFO" or "CRITICAL". Because of the "Connection refused -> ERROR" example, it infers that "degradation" is likely a `WARN` or `ERROR` based on the pattern of the previous examples.

---

**Concept 2: Chain of Thought (The "Debug Trace" Pattern)**

LLMs are bad at math and complex logic if they answer immediately. They need "thinking time." **Chain of Thought (CoT)** forces the model to output its reasoning steps before the final answer.

- **Standard Prompt:** "How many distinct IP addresses in this list are from the 192.168 subnet?" -> _Model guesses a number (often wrong)._
- **CoT Prompt:** "Think step-by-step. First, extract all IPs. Second, filter for 192.168. Finally, count them." -> _Model writes the logic, then the answer (usually correct)._
