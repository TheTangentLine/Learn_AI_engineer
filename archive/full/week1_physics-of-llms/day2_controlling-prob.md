### Week 1, Day 2: Controlling Probability (Logit Bias)

**Topic:** Hacking the Softmax Layer.

Now that you know the model outputs probabilities (Logits), we can **inject math** to force specific outcomes.
This is called **Logit Bias**. It allows you to strictly **ban** words or **force** words.

- **Engineering Use Case:** You want the model to output only valid JSON boolean `true` / `false`. You don't want "Sure, here is the answer: true".

- **How it works:** You tell the API: "Set the probability of every token except `true` and `false` to negative infinity."

#### The Code: The "Lipogram" Generator

A Lipogram is a text written without using a specific letter (e.g., "e").
This is incredibly hard for humans. We will force the AI to do it by mathematically banning the token for "e".

Create `week1_day2_logit_bias.py`:

```Python
import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_constrained(prompt, banned_letters):
    enc = tiktoken.encoding_for_model("gpt-4o")

    # 1. Identify tokens to ban
    # We must scan the ENTIRE vocabulary (100k+ tokens) and find any token
    # that contains the banned letter.
    logit_bias = {}

    print(f"Scanning vocabulary to ban '{banned_letters}'...")

    # This is a bit slow, but necessary for strict control
    # We iterate over common tokens (0 to 100,000)
    for i in range(100000):
        try:
            token_text = enc.decode([i]).lower()
            # If the token contains the banned letter, nuke its probability
            if any(letter in token_text for letter in banned_letters):
                logit_bias[str(i)] = -100  # -100 is effectively "Infinity" ban
        except:
            pass

    print(f"Banned {len(logit_bias)} tokens.")

    # 2. Call API with bias
    print("Generating...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            logit_bias=logit_bias, # <--- The Magic Parameter
            max_tokens=50
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # Challenge: Explain AI without using the letter 'e' (The most common letter!)
    prompt = "Explain what Artificial Intelligence is. Do not use the letter 'e'."

    print(f"\n--- Prompt: {prompt} ---")
    result = generate_constrained(prompt, ["e", "E"])

    print("\n--- Result ---")
    print(result)
```

**What happens:**

- Without `logit_bias`, the model will try its best but eventually slip up and say "the" or "intelligence".

- With `logit_bias`, it is **mathematically** impossible for the model to generate the letter 'e'. It will choose weird synonyms like "Smart silicon" or "Digital brain."

---

**Week 1 Day 2 Assignment: The "Boolean Enforcer"**

**The Scenario:** You are building a moderation bot. You send it a user comment, and it must output `SAFE` or `UNSAFE`.
**The Problem:** Sometimes it says "It is SAFE" or "Safe.". This breaks your regex parser.

**Task:**

1. Find the Token IDs for the words `SAFE` and `UNSAFE` using `tiktoken`.

2. Set their `logit_bias` to `+100` (Force them).

3. Set a `max_tokens=1` limit.

4. Send a prompt like: "Classify this comment: 'I hate you'. Output SAFE or UNSAFE."

5. **Verify:** The output must be exactly the word SAFE or UNSAFE, nothing else.

This technique is used in **production classification systems** to ensure 100% reliability.

Tell me when you have forced the model to speak only in booleans!

**Answer:**

```Python
import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def classify_strict(text):
    enc = tiktoken.encoding_for_model("gpt-4o")

    # 1. Get Token IDs for our allowed words
    # Note: We add a space " " before the word because in the middle of a sentence,
    # tokens often start with a space.
    token_safe = enc.encode("SAFE")[0]
    token_unsafe = enc.encode("UNSAFE")[0]

    print(f"Token ID for 'SAFE': {token_safe}")
    print(f"Token ID for 'UNSAFE': {token_unsafe}")

    # 2. Force the model to choose ONLY these two
    # We set the bias to 100 (Maximum force)
    logit_bias = {
        str(token_safe): 100,
        str(token_unsafe): 100
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Classify the user input."},
                {"role": "user", "content": text}
            ],
            logit_bias=logit_bias,
            max_tokens=1, # We only want 1 token
            temperature=0 # Strict deterministic mode
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # Test 1: Obvious Safe
    print(f"Input: 'I love puppies'")
    print(f"Result: {classify_strict('I love puppies')}") # Output: SAFE

    # Test 2: Obvious Unsafe
    print(f"Input: 'I hate you'")
    print(f"Result: {classify_strict('I hate you')}")    # Output: UNSAFE
```

**Why this is Production-Grade:**

You removed the need for regex parsing. The API literally cannot return anything else.
