**Week 3, Day 1: The Data (Garbage In, Garbage Out)**

70% of Fine-Tuning is just formatting a JSON file.
If your data is bad, your model will be brain-damaged.

**The Format: "Alpaca" or "ShareGPT"**
We usually use a standard JSONL (JSON Lines) format.

```JSON
{"instruction": "Convert this text to SQL.", "input": "Get users from CA", "output": "SELECT * FROM users WHERE state = 'CA';"}
{"instruction": "Convert this text to SQL.", "input": "Count active items", "output": "SELECT count(*) FROM items WHERE status = 'active';"}
```

**The Code: Data Preparation**

We are going to build a **Dataset Generator**.
Let's say you want to train a model to be a **Sarcastic Tech Support Bot**. We need to generate examples of this behavior.

Create `week3_day1_dataprep.py`.

```Python
import json
import pandas as pd

# 1. Define your "Curriculum"
# These are the examples we want the model to memorize.
raw_data = [
    {
        "user": "My internet is not working.",
        "bot": "Have you tried turning it off and on again? Or did you just unplug it to vacuum?"
    },
    {
        "user": "I forgot my password.",
        "bot": "Amazing. I suppose you also forgot where you wrote it down on a sticky note?"
    },
    {
        "user": "Where is the download button?",
        "bot": "It's the giant green button that says 'DOWNLOAD'. Hard to miss, but here we are."
    }
]

# 2. Format for Training (The "Alpaca" Format)
# We need to structure it so the model knows what is Input and what is Output.
training_data = []

for entry in raw_data:
    # This string format is exactly what the model sees during training
    formatted_entry = {
        "instruction": "You are a sarcastic tech support agent. Answer the user.",
        "input": entry["user"],
        "output": entry["bot"]
    }
    training_data.append(formatted_entry)

# 3. Save to JSONL
# This file is what we will upload to the GPU tomorrow.
output_file = "sarcastic_support_dataset.jsonl"
with open(output_file, "w") as f:
    for entry in training_data:
        json.dump(entry, f)
        f.write("\n")

print(f"Successfully created {output_file} with {len(training_data)} examples.")

# 4. Preview with Pandas (Just to check)
df = pd.read_json(output_file, lines=True)
print(df.head())
```

**Week 3 Day 1 Assignment: "The Specialist"**

1. **Pick a Persona:** Decide what you want your model to be. Examples:
   - A Medical Assistant that explains complex terms simply.
   - A Git Expert that only outputs git commands.
   - A Vietnamese Translator.

2. **Create the Data:**
   - Manually write at least **10 examples** in the `raw_data` list.
   - (Pro Tip: In the real world, we use GPT-4 to generate 500 synthetic examples for us).

3. **Run the script:** Generate your `.jsonl` file.

**Do not skip this**. Tomorrow (Day 2), we will open **Google Colab**, load a GPU, and feed this file into **Unsloth** to actually train the model. You need this file ready.

Tell me when you have your `.jsonl` file!
