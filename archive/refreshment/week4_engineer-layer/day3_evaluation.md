Welcome to **Week 4, Day 3: Evaluation (Unit Testing for AI)**.

You have a chatbot. You have a frontend. You have a backend.
**The Question:** How do you know if it sucks?

In traditional software, you write `assert result == 5`.
In AI, the result is different every time. You can't check for string equality.
**The Solution: "LLM-as-a-Judge".**
We use a smart model (GPT-4) to grade the homework of your smaller model (Sarcastic Bot).

---

**The Concept: Evals**

An "Eval" is just a unit test suite for prompts.

1. **Dataset:** A list of 20 questions.
2. **Runner:** Loop through and get answers from your bot.
3. **Grader:** Send (Question + Answer) to GPT-4 and ask: _"On a scale of 1-5, how sarcastic was this?"_

---

**The Code: The Automated Grader**

We will write a script that runs a "Sarcasm Test" on your local model.

Create `week4_day3_evals.py`:

```Python
import json
from openai import OpenAI
from tqdm import tqdm # Progress bar

# 1. Setup Clients
# Your local bot (The Student)
student_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# The Grader (Teacher) - We use OpenAI for reliable grading, or a strong local model
grader_client = OpenAI(api_key="sk-...") # Put your real OpenAI key here

# 2. The Test Dataset
test_cases = [
    "I spilled water on my laptop.",
    "How do I reset my password?",
    "The server is down again.",
    "Can you install Minecraft on the production DB?",
    "Why is Python so slow?"
]

# 3. The Grading Function
def grade_sarcasm(question, answer):
    prompt = f"""
    You are a Sarcasm Judge.
    Rate the following response on a scale of 1 to 5.

    1 = Helpful and polite (Failed)
    5 = Extremely sarcastic and mocking (Passed)

    User Question: "{question}"
    Bot Response: "{answer}"

    Return ONLY the number (1-5).
    """

    response = grader_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return int(response.choices[0].message.content.strip())
    except:
        return 1

# 4. The Loop
print("--- Starting Evaluation ---")
scores = []

for question in tqdm(test_cases):
    # A. Get answer from Student
    student_res = student_client.chat.completions.create(
        model="sarcastic-bot",
        messages=[{"role": "user", "content": question}]
    )
    answer = student_res.choices[0].message.content

    # B. Get grade from Teacher
    score = grade_sarcasm(question, answer)
    scores.append(score)

    print(f"\nQ: {question}")
    print(f"A: {answer}")
    print(f"Score: {score}/5")

# 5. Final Report
avg_score = sum(scores) / len(scores)
print(f"\n--- Final Results ---")
print(f"Average Sarcasm Score: {avg_score:.1f}/5.0")
if avg_score > 3.5:
    print("✅ TEST PASSED: The bot is sufficiently mean.")
else:
    print("❌ TEST FAILED: The bot is too nice. Re-train with more data.")
```

**Why this matters**

This is **CI/CD for AI**.
In a real job, you would run this script every time you change your System Prompt or Fine-Tune data. If the score drops, you don't deploy.
