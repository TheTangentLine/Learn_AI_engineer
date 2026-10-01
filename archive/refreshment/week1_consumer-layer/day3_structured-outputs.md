This is the most critical day for a backend engineer. **The Problem:** You built a SQL translator yesterday, but what if the model outputted: _"Here is your SQL query: SELECT _ FROM..."_? Your backend code would crash because it tried to execute _"Here is your SQL..."\* as a database command.

**The Solution**: Stop parsing strings with Regex. Force the LLM to output **Valid JSON** that validates against a schema.

---

**The Concept: Pydantic + LLMs**

You likely use **Pydantic** for API validation (FastAPI). We are going to use it to validate AI thoughts. OpenAI recently released **"Structured Outputs"**, which guarantees that the output will match your Pydantic model 100% of the time.

**The Code: The "Data Extractor"**
We will build a script that takes a messy, rambling customer email and extracts a clean, type-safe Python object.

**Prerequisite**: `pip install pydantic`

Create `day3_structure.py`:

```python
import os
import json
from enum import Enum
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Define the Schema (The Contract)
# We want to extract strictly defined data, not free text.
class TicketPriority(str, Enum):
    LOW = "low"
    HIGH = "high"
    CRITICAL = "critical"

class SupportTicket(BaseModel):
    customer_name: str
    issue_summary: str
    priority: TicketPriority
    estimated_hours: int
    is_refund_request: bool

# 2. The Logic
def extract_ticket_data(email_text):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06", # Must use a model that supports Structured Outputs
        messages=[
            {"role": "system", "content": "Extract the support ticket details from the email."},
            {"role": "user", "content": email_text},
        ],
        response_format=SupportTicket, # <--- MAGIC HAPPENS HERE
    )

    # 3. Automatic Validation
    # The SDK automatically validates the JSON and converts it to your Pydantic object
    ticket = completion.choices[0].message.parsed
    return ticket

if __name__ == "__main__":
    # A messy input
    email = """
    Hi team, I am incredibly frustrated. My name is John Doe.
    I bought the pro plan yesterday for $50 and it's not working at all.
    I cannot log in. I want my money back immediately!
    This is urgent. Fix it now.
    """

    ticket_obj = extract_ticket_data(email)

    # 4. Use it like a normal Python object
    print(f"Customer: {ticket_obj.customer_name}")
    print(f"Priority: {ticket_obj.priority}") # Will be TicketPriority.CRITICAL
    print(f"Refund?:  {ticket_obj.is_refund_request}")
    print(f"Type:     {type(ticket_obj)}")
```

**Why this is powerful**

1. **Type Safety:** `priority` is guaranteed to be one of your Enums. It will never be "High" (capital H) or "Urgent" (synonym). It will be `HIGH` or `CRITICAL`.

2. **Boolean Logic:** The model inferred `is_refund_request = True` just from the phrase "I want my money back".

3. **No JSON Parsing:** You never typed `json.loads()`. The SDK handled the serialization.
