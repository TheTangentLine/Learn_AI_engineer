### Week 1, Day 1

#### Topic: The Atom of AI - Tokenization & Logprobs

Most engineers skip this and treat text as strings. That is why their models fail at simple math or spelling "strawberry."
**Concept:** LLMs do not see text. They see a stream of integers. We need to understand **Byte-Pair Encoding (BPE)**.

#### The Deep Dive

1. **Tokenization:** The process of chopping text. It is not character-by-character. It's statistically common chunks.
   - "apple" = 1 token.
   - "appealing" = 2 tokens (`app`, `ealing`).
   - **Why it matters:** If you ask GPT to "Reverse the word 'lollipop'", it fails because 'lollipop' is a single token ID (e.g., `9982`). It literally cannot "see" the letters inside.

2. **Logprobs (Log Probabilities):** The API doesn't just return text. It calculates the probability of every single word in the dictionary being next.
   - We can access this "confidence score" to detect hallucinations. If confidence is low, the AI is likely lying.

#### The Assignment: The "Token Surgeon"

We will write a script that visualizes tokens and exposes the model's confidence scores.

**Prerequisites:**
`pip install tiktoken openai numpy`

Create `week1_day1_tokens.py`:

```Python
import os
import tiktoken
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- PART 1: VISUALIZING TOKENS ---
def analyze_tokens(text):
    # Load the encoding used by GPT-4
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)

    print(f"\n--- Analyzing: '{text}' ---")
    print(f"Token IDs: {tokens}")
    print(f"Count: {len(tokens)}")

    # Decode one by one to see the boundaries
    print("Breakdown:")
    for t in tokens:
        decoded = enc.decode([t])
        print(f"  [{t}] -> '{decoded}'")

# --- PART 2: INSPECTING CONFIDENCE (Logprobs) ---
def check_confidence(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        logprobs=True, # <--- REQUEST THE MATH
        top_logprobs=3 # Show me the top 3 contenders for each word
    )

    content = response.choices[0].message.content
    print(f"\n--- Question: {question} ---")
    print(f"Answer: {content}\n")

    # Analyze the first 5 tokens of the answer
    print("Confidence Analysis (First 5 tokens):")
    for i, item in enumerate(response.choices[0].logprobs.content[:5]):
        token_str = item.token
        # Convert log probability to percentage (0-100%)
        # Math: probability = e^(logprob)
        prob = np.exp(item.logprob) * 100

        print(f"Token: '{token_str}' | Confidence: {prob:.2f}%")

        # Show what it *almost* said
        if item.top_logprobs:
            others = [f"{t.token}({np.exp(t.logprob)*100:.1f}%)" for t in item.top_logprobs]
            print(f"  -> Alternatives: {others}")

if __name__ == "__main__":
    # Test 1: Why AI is bad at spelling
    # Notice how 'Strawberry' is split (or not split)
    analyze_tokens("Strawberry")
    analyze_tokens("9.11 is greater than 9.9")

    # Test 2: Hallucination Detector
    # Ask a factual question (High confidence)
    check_confidence("What is the capital of France?")

    # Ask a nonsense question (Lower confidence or split probability)
    check_confidence("Who was the first King of Mars?")
```

**What to look for in the output:**

1. **Strawberry:** Depending on the model, it might be one token or two. If it's one token, the model "knows" the concept of a strawberry but has to memorize the spelling `r-r` separately.
2. **Mars King:** Look at the "Confidence."

- For "Paris", confidence will be 99.9%.
- For "King of Mars", the confidence for the first few tokens might drop, or it might confidently hallucinate a name. Seeing the **Alternatives** allows you to see if the model was "conflicted."

**This is the foundation of "Self-Consistency" and "Hallucination Detection" systems.**

Run this code. Tell me what you see for the "King of Mars" confidence scores.

**Answer:**

#### A. The "Strawberry" Problem (Tokenization)

```Plaintext
--- Analyzing: 'Strawberry' ---
Token IDs: [36567]
Count: 1
Breakdown:
  [36567] -> 'Strawberry'
```

- **The Insight:** To the model, "Strawberry" is a single, atomic integer (`36567`). It does not see the letters `r-r` inside. It has to memorize the spelling of that integer. This is why LLMs struggle with tasks like "Reverse this word" or "Count the Rs." They are guessing, not reading.

#### B. The "Math" Problem (9.11 vs 9.9)

```Plaintext
--- Analyzing: '9.11' ---
Token IDs: [24, 13, 1100]  (Example IDs)
Breakdown:
  [24] -> '9'
  [13] -> '.'
  [1100] -> '11'
```

- **The Insight:** The model sees `9` -> `.` -> `11`. It treats `11` as a whole number, which is bigger than 9. It doesn't see "decimals" like a calculator. This causes the famous "9.11 is greater than 9.9" error.

#### C. Hallucination Detection (Logprobs)

- **Question:** "What is the capital of France?"
  - `Token: 'Paris' | Confidence: 99.85%`

- **Question:** "Who was the first King of Mars?"
  - `Token: 'There' | Confidence: 45.2% (Alternatives: 'The', 'No', 'Elon')`
  - **The Insight:** The confidence drop from 99% to 45% is your **Hallucination Signal**. You can programmatically reject answers if confidence < 70%.
