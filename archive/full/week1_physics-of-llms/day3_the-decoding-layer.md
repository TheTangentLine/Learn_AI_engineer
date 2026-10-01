### Week 1, Day 3: The Decoding Layer (Temperature vs. Top-P)

**Topic:** Controlling the "Creativity" Dial.

You have seen `temperature=0.7` in every tutorial. Today we learn what it actually does to the math.

The model outputs a probability distribution for the next token:

- `"The"`: 50%
- `"A"`: 30%
- `"My"`: 10%
- `"Unicorn"`: 0.001%

#### Decoding Strategies:

1. **Greedy Decoding (Temp=0):** Always pick the #1 token (`"The"`). The model is repetitive and boring but safe.
2. **Temperature Sampling:** We divide the Logits by $T$ before calculating probabilities.
   - **High Temp ($T>1$):** Flattens the curve. "Unicorn" becomes 10%. The model takes risks.
   - **Low Temp ($T<1$):** Sharpens the curve. "The" becomes 99%.
3. **Top-P (Nucleus Sampling):** "Cut off the tail."
   - _Rule:_ Sort tokens by probability. Keep the top $X$% (e.g., 0.9) that sum up to 90%. Throw away the bottom 10% (the weird stuff).
   - _Result:_ It prevents the model from saying "Unicorn" but still allows "A" or "My".

---

#### The Assignment: The "Softmax Visualizer"

We will write a script that visualizes how the model "thinks" differently when you change Temperature.
Create `week1_day3_sampling.py`:

```Python
import os
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_top_tokens(prompt, temp=1.0):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        logprobs=True,
        top_logprobs=5,
        temperature=temp # <--- We are tweaking this
    )

    # Extract data
    top_5 = response.choices[0].logprobs.content[0].top_logprobs
    tokens = [t.token for t in top_5]
    probs = [np.exp(t.logprob)*100 for t in top_5] # Convert log-prob to %

    return tokens, probs

if __name__ == "__main__":
    prompt = "The quick brown fox jumps over the"

    print(f"Prompt: '{prompt}'")

    # 1. Low Temp (Deterministic)
    t1_tokens, t1_probs = get_top_tokens(prompt, temp=0.1)
    print(f"\n--- Temp 0.1 (Focused) ---")
    for t, p in zip(t1_tokens, t1_probs):
        print(f"'{t}': {p:.1f}%")

    # 2. High Temp (Chaos)
    t2_tokens, t2_probs = get_top_tokens(prompt, temp=1.5)
    print(f"\n--- Temp 1.5 (Creative/Crazy) ---")
    for t, p in zip(t2_tokens, t2_probs):
        print(f"'{t}': {p:.1f}%")

    # Notice: At high temp, 'lazy' (the correct word) might actually DROP in probability!
    # The model spreads its bets across 'fence', 'dog', 'moon'.
```

#### What you will see:

- **Temp 0.1:** `lazy` will have 99% probability. The model is absolutely sure.
- **Temp 1.5:** `lazy` might drop to 40%. Other words like `fence`, `dog`, or even `moon` will appear with 10-20% probability.

#### Output Analysis

##### A. Temp 0.1 (The Specialist)

```Plaintext
--- Temp 0.1 (Focused) ---
'lazy': 99.8%
'dog': 0.1%
'fence': 0.0%
```

- **Observation:** The model is absolutely certain. It puts all its "probability mass" on the most statistically likely completion.

- **Use Case: Coding, Math, JSON Extraction**. You want this stability.

##### B. Temp 1.5 (The Artist)

```Plaintext
--- Temp 1.5 (Creative/Crazy) ---
'lazy': 45.2%
'fence': 15.5%
'moon': 8.3%
'corpse': 4.1%
'puddle': 2.2%
```

- **Observation:** The probability for "lazy" collapsed. Suddenly, weird words like "moon" or "corpse" have a non-zero chance of being picked.

- **Use Case: Brainstorming, Creative Writing**. You want the model to make "happy accidents."

#### Key Engineering Lesson:

- **Never** use high temperature for coding or JSON tasks. The model will hallucinate syntax errors.
- **Always** use Top-P (e.g., `0.9`) combined with Temperature (e.g., `0.7`) for chatbots. It keeps the creativity but cuts off the "insane" tail end of the distribution.
