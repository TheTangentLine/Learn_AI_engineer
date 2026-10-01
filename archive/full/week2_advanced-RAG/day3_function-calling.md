### Week 2, Day 3: Function Calling (Giving the AI "Hands")

**Topic:** Breaking out of the text box.

Until today, your models have been brains trapped in a jar. They can think, but they cannot do. They cannot read a live database, they cannot send an email, and they don't know the current time.

**Function Calling (also known as Tool Use)** solves this. It bridges the gap between the AI's natural language understanding and your existing backend APIs.

#### The Deep Dive: The "Tool" Protocol

A common misconception is that the AI _runs_ the code. It does not.
Here is the exact lifecycle:

1. **You:** Send a prompt ("What is the weather in Tokyo?") AND a JSON blueprint of a Python function you have (e.g., `get_weather(location)`).

2. **The AI:** Realizes it needs outside info. Instead of replying with text, it replies with a JSON command: `{"name": "get_weather", "arguments": {"location": "Tokyo"}}`.

3. **Your Backend:** Intercepts this JSON, physically executes your actual Python `get_weather("Tokyo")` function, and gets the result (e.g., `22°C`).

4. **You:** Send that result back to the AI.

5. **The AI:** Reads the result and finally generates a human-readable text response: _"It is currently 22°C in Tokyo."_

---

#### The Code: The "Database Agent"

We will build a script where the AI realizes it doesn't know a user's account balance, calls a local Python function to look it up, and then replies.

Create `week2_day3_tools.py`:

```Python
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Your actual backend function (The "Tool")
def get_account_balance(user_id: str) -> str:
    """Mock database lookup."""
    print(f"\n[SYSTEM] Executing database query for user: {user_id}...")
    db = {"user_123": "$1,450.00", "user_456": "$20.00"}
    return db.get(user_id, "User not found")

# 2. The JSON Schema (Telling the AI the tool exists)
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_account_balance",
            "description": "Get the current account balance for a specific user ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The unique ID of the user (e.g., 'user_123')",
                    }
                },
                "required": ["user_id"],
            },
        }
    }
]

def chat_with_tools(user_prompt):
    messages = [{"role": "user", "content": user_prompt}]

    # Step 1: Send prompt + tools to AI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools_schema, # <--- We hand the AI the blueprint
        tool_choice="auto"  # Let the AI decide if it needs the tool
    )

    response_msg = response.choices[0].message

    # Step 2: Check if the AI wants to call a tool
    if response_msg.tool_calls:
        tool_call = response_msg.tool_calls[0]
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        print(f"\n[AI THOUGHT] I need to call the function: '{func_name}' with args: {func_args}")

        # Step 3: Execute the function locally
        if func_name == "get_account_balance":
            function_result = get_account_balance(user_id=func_args.get("user_id"))

            # Step 4: Append the history and send the result BACK to the AI
            messages.append(response_msg) # Append the AI's tool request
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": function_result # The actual data ($1,450.00)
            })

            print("\n[SYSTEM] Sending database result back to AI...")

            # Step 5: Get final human-readable answer
            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return final_response.choices[0].message.content
    else:
        # AI didn't need a tool, just replied normally
        return response_msg.content

if __name__ == "__main__":
    prompt = "Hi, my ID is user_123. Can you tell me how much money I have?"
    print(f"User: {prompt}")

    final_answer = chat_with_tools(prompt)
    print(f"\nFinal AI Response: {final_answer}")
```

**Why this is a paradigm shift**

You just built an **Agent**.
Instead of writing an `if/else` block to parse the user's intent ("Did they ask for their balance? Regex for ID..."), the AI handled the intent routing, parsed the `user_123` variable out of the natural language string, and formatted the JSON request for your backend perfectly.

#### Week 2 Day 3 Assignment: "The Calculator Agent"

LLMs are terrible at math. Let's fix that by giving the model a calculator tool.

**Task:**

1. Write a Python function `multiply_numbers(a: float, b: float) -> float`.

2. Create the JSON schema for this function in the `tools_schema` list.

3. Update the execution logic to route to `multiply_numbers` if the AI calls it.

4. Ask the AI: _"What is 4,123.5 multiplied by 89.2?"_

**Verify:** The AI should not attempt to guess the math. It should request the tool, wait for your Python backend to calculate it accurately, and then print the result.
