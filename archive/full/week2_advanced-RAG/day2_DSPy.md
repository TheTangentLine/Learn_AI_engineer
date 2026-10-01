### Week 2, Day 2: DSPy (Automating the Prompt)

**Topic:** Stop writing prompts in English. Compile them.

In Week 1 and yesterday, you wrote system prompts manually: "_You are a helpful assistant..._" or "_Think step by step..._".
**The Problem:** Hand-written prompts are brittle. If you change models (from GPT-4 to Llama 3), your perfect prompt might suddenly fail.

#### The Solution: DSPy (Demonstrate-Search-Predict)

Created by Stanford researchers, DSPy treats language models like PyTorch layers. Instead of writing prompts, you write **Signatures** (Input/Output definitions), and DSPy _compiles_ them into the perfect prompt for whatever model you are using.

#### The Deep Dive: Signatures vs. Prompts

- **Traditional:** `prompt = f"Summarize this text: {text}"`
- **DSPy Signature:** `class Summarize(dspy.Signature): text = dspy.InputField(); summary = dspy.OutputField()`

When you compile a DSPy program, it automatically tests different prompt variations (including adding Chain of Thought automatically) until it finds the one that gets the highest score on your metrics.

---

#### The Code: Your First DSPy Program

We will write a program that extracts information without writing a single English instruction.

##### Prerequisites:

```Bash
pip install dspy-ai
```

Create `week2_day2_dspy.py`:

```Python
import dspy
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Configure the Language Model
# We set GPT-4o-mini as our default processing engine
turbo = dspy.OpenAI(model='gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"), max_tokens=250)
dspy.settings.configure(lm=turbo)

# 2. Define the Signature (The "Type Definition")
# We do not tell it HOW to extract. We just define the inputs and outputs.
class ExtractActionItems(dspy.Signature):
    """Extract a list of actionable tasks from a meeting transcript."""

    transcript = dspy.InputField(desc="Raw text from a meeting")
    action_items = dspy.OutputField(desc="Comma separated list of tasks")

# 3. Create the Module (The "Neural Network Layer")
# We use Predict (Standard) or ChainOfThought (Automatically adds reasoning)
class ActionExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        # We wrap our signature in a ChainOfThought module
        self.extractor = dspy.ChainOfThought(ExtractActionItems)

    def forward(self, transcript):
        # This executes the extraction
        return self.extractor(transcript=transcript)

# 4. Execution
if __name__ == "__main__":
    meeting_notes = """
    Alright team, great sync. John, I need you to update the database schema by Tuesday.
    Sarah, please reach out to the design team about the new logo.
    I will handle the client presentation on Friday. Let's reconvene next week.
    """

    # Instantiate and run
    extractor_module = ActionExtractor()
    result = extractor_module(transcript=meeting_notes)

    print("--- DSPy Extraction ---")
    # Notice we can access the output as a clean variable
    print(f"Tasks: {result.action_items}")

    # 5. The Magic: Look at what DSPy generated under the hood
    print("\n--- The Compiled Prompt (What DSPy actually sent to OpenAI) ---")
    print(turbo.inspect_history(n=1))
```

##### What just happened?

You never wrote a prompt.
You defined variables (`transcript` and `action_items`). DSPy looked at your variable names, looked at the `desc` tags, and **automatically built the prompt**. Because you used `dspy.ChainOfThought`, DSPy automatically inserted the math and logic to make the model think before extracting.

---

#### Week 2 Day 2 Assignment: "The Sentiment Compiler"

##### Task:

1. Create a DSPy Signature called `AssessSentiment`.
2. Input field: `customer_review`.
3. Output fields: `sentiment` (Positive/Negative/Neutral) and `urgency` (High/Low).
4. Wrap it in a `dspy.Predict` module.
5. Pass in a review: "_The app crashed right when I was checking out. I am furious._"
6. Print the `sentiment` and `urgency` variables.

**Why this matters:**
As a backend engineer, you want to write classes and methods, not string templates. DSPy allows you to build AI applications purely using software engineering paradigms.

##### Answer:

```Python
import dspy
import os
from dotenv import load_dotenv

load_dotenv()
turbo = dspy.OpenAI(model='gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"), max_tokens=200)
dspy.settings.configure(lm=turbo)

# 1. The Signature
class AssessSentiment(dspy.Signature):
    """Assess a customer review for sentiment and urgency."""

    customer_review = dspy.InputField(desc="The raw customer review text")
    sentiment = dspy.OutputField(desc="Positive, Negative, or Neutral")
    urgency = dspy.OutputField(desc="High or Low")

# 2. The Module
class SentimentAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        # Using Predict instead of ChainOfThought for faster, direct extraction
        self.analyzer = dspy.Predict(AssessSentiment)

    def forward(self, review):
        return self.analyzer(customer_review=review)

# 3. Execution
if __name__ == "__main__":
    review_text = "The app crashed right when I was checking out. I am furious."

    analyzer_module = SentimentAnalyzer()
    result = analyzer_module(review=review_text)

    print("--- DSPy Sentiment Analysis ---")
    print(f"Review: '{review_text}'")
    print(f"Sentiment: {result.sentiment}")
    print(f"Urgency: {result.urgency}")

    print("\n--- Under the Hood (The Compiled Prompt) ---")
    print(turbo.inspect_history(n=1))
```
