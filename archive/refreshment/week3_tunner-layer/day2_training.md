Welcome to **Week 3, Day 2: The Training Run.**

This is the most exciting day. You are going to take a multi-billion dollar model (Llama 3) and teach it your specific dataset.

**Where do we run this?**
Unless you have an NVIDIA GPU with 16GB+ VRAM (like an RTX 4090) on your laptop, we cannot run this locally.
We will use **Google Colab** (Free Tier T4 GPU). It is a cloud Jupyter Notebook.

**The Tool: Unsloth**
We will use a library called **Unsloth**. It is currently the fastest way to fine-tune Llama models. It optimizes the math to make training 2x faster and use 70% less memory.

---

**Step 1: Setup Google Colab**

1. Go to [Google Colab](colab.research.google.com).

2. Click **New Notebook**.

3. **Crucial Step:** Go to `Runtime` > `Change runtime type` > Select **T4 GPU**. (If you don't do this, the code will crash).

---

**Step 2: The Training Script**

Copy and paste these blocks into your Colab notebook cells and run them one by one.

**Cell 1: Installation**
This installs Unsloth and the necessary PyTorch libraries.

```Python
%%capture
# Installs Unsloth, Xformers (memory optimization), and PyTorch bits
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.8.0" peft accelerate bitsandbytes
```

**Cell 2: Load the Base Model (Llama 3)**

We load the model in **4-bit quantization**. This squashes the model size down so it fits on the free GPU, with almost zero loss in intelligence.

```Python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # How much text it can read at once
dtype = None # Auto-detect best data type
load_in_4bit = True # 4-bit quantization to fit in memory

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", # The base model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Add LoRA adapters (The "Sticker" we are training)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank: Higher = smarter but slower. 16 is standard.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"], # Which layers to train
    lora_alpha = 16,
    lora_dropout = 0, # Dropout = 0 is faster
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)
```

**Cell 3: Load Your Data**

Remember the `sarcastic_support_dataset.jsonl` you made yesterday? Upload it to Colab (Click the Folder icon on the left -> Upload).

```Python
from datasets import load_dataset

# 1. Load the file

dataset = load_dataset("json", data_files="sarcastic_support_dataset.jsonl", split="train")

# 2. Format it

# We need to map your JSON keys to the prompt format Llama expects

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction=instruction, input=input, output=output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)
```

**Cell 4: The Training Loop**

This is where the magic happens. We use the `SFTTrainer` (Supervised Fine-Tuning Trainer).

```Python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Process 2 examples at a time
        gradient_accumulation_steps = 4, # Update weights after 4 steps
        warmup_steps = 5,
        max_steps = 60, # Increase this to 100-300 for real training!
        learning_rate = 2e-4, # Standard learning rate for LoRA
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit", # Optimizer
    ),
)

trainer.train()
```

**Step 3: Watch it Learn**
When you run **Cell 4**, you will see a progress bar.
Look at the **Training Loss**:

- Start: High (e.g., 2.5)
- End: Low (e.g., 0.8)

If the loss goes down, **it is learning**. It is memorizing your sarcastic style.

---

**Step 4: Inference (Testing)**

Let's test if it actually worked.

```Python

# Enable native 2x faster inference

FastLanguageModel.for_inference(model)

# Prepare a test input

inputs = tokenizer(
[
    alpaca_prompt.format(
        instruction="You are a sarcastic tech support agent.",
        input="I spilled coffee on my keyboard.", # Something NOT in your dataset
        output="", # Leave blank for generation
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
print(tokenizer.batch_decode(outputs))
```

**Expected Output:**

If it worked, it won't just say "Please clean it."
It might say: "Well, did the keyboard look thirsty? Try buying a waterproof one next time, genius."

---

**Week 3 Day 2 Assignment**

1. Run this notebook in Colab.
2. Train it for **60 steps** (it takes about 5 minutes).
3. Show me the "Sarcastic Response" it generates for a new input.

**Note on "Steps":**

- **60 Steps:** Good for a quick test to see if code works.
- **300-500 Steps:** Good for actually making the model smart/robust.

Let me know when you see that progress bar moving!
