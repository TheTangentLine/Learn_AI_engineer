### Week 2: Advanced Prompt Engineering (The "Code" Layer)

Welcome to Week 2. We are done with "Hello World" prompts.
You are a software engineer. Stop thinking of prompts as "English." Think of them as **Compiled Code**.

- **Variables:** `{user_input}`
- **Functions:** Tools/Plugins
- **Control Flow:** Chain of Thought (If/Else logic inside the prompt)
- **Type Safety:** Structured Outputs (Pydantic)

### Week 2, Day 1: Chain of Thought (CoT) & Reasoning

#### The Problem:

LLMs are "System 1" thinkers (Intuitive, fast, often wrong at math).
If you ask: "I have 3 apples, I eat 2, I buy 5. How many do I have?"

- **Standard Model:** might guess "6" immediately.
- **Real Human:** thinks: "3 - 2 = 1. 1 + 5 = 6."

#### The Solution: Chain of Thought (CoT)

We force the model to output its **Computation Trace** before the final answer. This turns it into a "System 2" thinker (Slow, deliberate).

#### The Deep Dive: Zero-Shot vs. Few-Shot CoT

1. **Zero-Shot CoT:** Just add "_Let's think step by step_" to the end of the prompt.

- _Result: The model automatically generates a trace._

2. **Manual CoT (Few-Shot):** You provide examples of how to think.
   - _Prompt:_
     Q: 10 + 5?
     A: 10 plus 5 is 15.
     Q: 10 + 5 _ 2?
     A: Order of operations. First 5 _ 2 = 10. Then 10 + 10 = 20.

#### The Code: The "Reasoning Engine"

We will write a script that solves a logic puzzle that fails without CoT.

Create `week2_day1_reasoning.py`:

```Python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def solve_puzzle(puzzle, method="standard"):
    if method == "standard":
        system_prompt = "You are a helpful assistant. Answer the question directly."
    elif method == "cot": # We force the 'Thought' process
        system_prompt = """
        You are a Reasoning Engine.
        Step 1: Break the problem down into variables.
        Step 2: Solve each part step-by-step.
        Step 3: Output the final answer in format: ANSWER: [X]
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini", # Use a small model to prove CoT makes it smarter
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": puzzle}
        ],
        temperature=0 # Deterministic
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # A tricky logic puzzle (The "Sally's Sister" riddle) # Most small models get this wrong because they look at semantic similarity # rather than logical relationships.
    puzzle = """
    Sally has 3 brothers. Each brother has 2 sisters.
    How many sisters does Sally have?
    """

    print("--- Attempt 1: Standard ---")
    print(solve_puzzle(puzzle, method="standard"))

    print("\n--- Attempt 2: Chain of Thought ---")
    print(solve_puzzle(puzzle, method="cot"))
```

##### What to expect:

- **Standard:** Might say "6 sisters" (3 brothers \* 2 sisters each) or "2 sisters". It often hallucinates the math.
- **CoT:** Will output:

        Step 1: Sally is a girl.
        Step 2: The brothers share the same sisters.
        Step 3: If a brother has 2 sisters, and one is Sally, the other is her sister.
        Step 4: So Sally has 1 sister.
        ANSWER: 1

---

#### Week 2 Day 1 Assignment: "The Bug Tracer"

##### Task:

1. Find a snippet of Python code with a subtle logic bug (e.g., an off-by-one error in a loop).

2. Write a script that asks the LLM to find the bug **Standardly** (it usually says "Code looks fine" or guesses wrong).

3. Write a **CoT Prompt** that forces it to:

- "Trace the value of `i` and `result` for every iteration of the loop."

4. **Verify:** The CoT version should catch the bug by "running" the code in its head.

This technique is how "AI Debuggers" work. They don't just "look" at code; they "simulate" it.

Tell me when you have your Reasoning Engine running!

##### Answer:

```Python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

buggy_code = """
def find_max(numbers):
    max_val = numbers[0]
    i = 1
    # Find the maximum number in the list
    while i < len(numbers) - 1:
        if numbers[i] > max_val:
            max_val = numbers[i]
        i += 1
    return max_val

print(find_max([1, 5, 3, 9]))
"""

def debug_code(method="standard"):
    if method == "standard":
        prompt = f"Find the bug in this Python code:\n{buggy_code}\nOutput the bug and the fix."
    elif method == "cot":
        prompt = f"""
        Find the bug in this Python code:
        {buggy_code}

        Do not guess. Execute the code in your mind step-by-step using the input [1, 5, 3, 9].
        Track the values of `i`, `numbers[i]`, and `max_val` for every loop iteration.
        After your trace, state the bug.
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

print("--- Standard Debugger ---")
print(debug_code("standard"))

print("\n--- CoT Debugger (The Trace) ---")
print(debug_code("cot"))
```

**What you will see:**

- **Standard Debugger:** Frequently hallucinates. It might say "You need to handle empty lists" or completely miss the logical error, stating the code works fine.

- **CoT Debugger:** It will literally print a table of variables:
  Iteration 1: `i=1`, `numbers[1]=5`, `max_val` becomes 5.
  Iteration 2: `i=2`, `numbers[2]=3`, `max_val` stays 5.
  Iteration 3: `i=3`. Wait, `len(numbers) - 1` is 3. The loop condition `3 < 3` is False. The loop terminates.
  **BUG FOUND:** The loop exits before checking the last element (9). The fix is `while i < len(numbers):`.

You have just built a static analysis tool that actually understands logic.
