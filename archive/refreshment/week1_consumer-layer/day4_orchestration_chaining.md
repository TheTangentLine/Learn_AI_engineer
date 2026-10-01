But in production, a single prompt is rarely enough. Complex tasks require **Chains** (Pipelines). Welcome to **Day 4: Orchestration (Chaining)**.

Today, we move from "Prompt Engineering" to **"AI Systems Engineering."**

**The Concept: Functional Composition**

Instead of trying to write one massive "Super Prompt" to do everything (which usually fails), you break the task into small, specialized steps where the output of `Step A` becomes the input of `Step B`.

**The "Self-Correction" Loop** We are going to build a system that writes code, checks its own **work for bugs**, and then fixes them. This is how tools like Cursor or GitHub Copilot Workspace work under the hood.

**The Code: The Self-Correcting Coder**

We will write a 3-step pipeline:

1. **The Generator**: Writes a Python function based on user input.
2. **The Critic**: Reviews the code for bugs or security flaws.
3. **The Refiner**: Rewrites the code based on the critique.

Create `day4_chain.py`:

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Helper function to keep calls clean
def get_response(system_prompt, user_input, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

# --- STEP 1: GENERATOR ---
def step1_generate_code(user_task):
    prompt = f"Write a Python function to: {user_task}. Return ONLY the code, no markdown."
    code = get_response("You are a Python expert.", prompt)
    print(f"\n--- [Step 1] Initial Draft ---\n{code}")
    return code

# --- STEP 2: CRITIC ---
def step2_critique_code(code):
    prompt = f"Review this code for bugs, edge cases, or security issues:\n{code}\nIf it is perfect, say 'pass'."
    critique = get_response("You are a Senior Code Reviewer. Be harsh and concise.", prompt)
    print(f"\n--- [Step 2] Critique ---\n{critique}")
    return critique

# --- STEP 3: REFINER ---
def step3_refine_code(original_code, critique):
    if "pass" in critique.lower() and len(critique) < 10:
        return original_code # No changes needed

    prompt = f"Fix this code based on the critique.\nCode:\n{original_code}\n\nCritique:\n{critique}\n\nReturn ONLY the fixed code."
    final_code = get_response("You are a Python expert.", prompt)
    print(f"\n--- [Step 3] Final Polish ---\n{final_code}")
    return final_code

# --- ORCHESTRATION ---
if __name__ == "__main__":
    task = "Write a function that calculates the factorial of a number."

    # 1. Draft
    draft = step1_generate_code(task)

    # 2. Review
    review = step2_critique_code(draft)

    # 3. Fix
    final_result = step3_refine_code(draft, review)
```

**Why this is better than one prompt**

Run this script with a tricky prompt, like: **"Write a function that takes a URL and downloads the file to disk."**

1. **Step 1** might write code using `requests` but forget to handle timeouts or large file chunking (RAM crash).
2. **Step 2** (The Critic) will catch: "Missing timeout parameter" or "Should stream response for large files."
3. **Step 3** will rewrite it to be production-ready.

If you asked for this in one shot, the model might just give you the lazy, buggy version.
