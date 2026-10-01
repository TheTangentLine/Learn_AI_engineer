#### The Goal

By the end of this session, you will have a Python script that sends a prompt to GPT-4o-mini and receives a structured response, understanding exactly what parameters like `temperature` actually do.

#### Step 1: The Code

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables
load_dotenv()

# 2. Initialize the Client
# If using Groq, add: base_url="https://api.groq.com/openai/v1"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 3. The Function
def chat_with_ai(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Or "llama3-8b-8192" if using Groq
        messages=[
            {"role": "system", "content": "You are a helpful assistant for a software engineer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=100
    )

    # 4. Parsing the Response
    # The API returns a big JSON object. We just want the text content.
    return response.choices[0].message.content

# 5. Execution
if __name__ == "__main__":
    user_input = "Explain the difference between TCP and UDP in one sentence."
    print(f"Asking AI: {user_input}\n")

    result = chat_with_ai(user_input)

    print("AI Answer:")
    print(result)
```

Run it: `python main.py`

#### Step 2: The Engineering Deep Dive (What just happened?)

Let's break down the parameters in `client.chat.completions.create`. This is where the engineering happens.

1. `messages` **(The State)**

Notice it's a **list of dictionaries**, not just a string. LLMs have no memory. Every time you call the API, you must send the entire conversation history if you want it to remember context.

- **System Role**: The "God mode" instruction. It sets the behavior (e.g., "You are a senior SQL expert. Do not explain, just write code.").
- **User Role**: The input from you.
- **Assistant Role**: (Not shown above) The model's previous replies. You append these here to keep the chat going.

2. `temperature` **(0.0 to 2.0)**

This controls **randomness**.

- **0.0 (Strict)**: The model always picks the most likely next token. It becomes deterministic. Use this for **coding, data extraction, or JSON**.
- **1.0 (Creative)**: The model takes risks. Use this for **brainstorming or writing**.
- _Try it_: Change temperature to `1.5` and ask it the same question 3 times. The answers will get weird.

3. `max_tokens`

This limits the **output** length. It does not limit input.

- If you set this to `10`, the model might cut off mid-sentence.
- _Cost control_: You pay per token. This prevents the model from writing a novel if you only wanted a sentence.
