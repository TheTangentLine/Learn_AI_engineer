Welcome to **Week 3, Day 3: Saving, Merging & Publishing.**

Right now, your smart, sarcastic model only exists in the temporary RAM of a Google Colab server. If you close that browser tab, **it is gone forever**.

Today, we learn how to:

1. **Save** your work (The LoRA Adapters).
2. **Merge** it (Glue the sticker onto the brain permanently).
3. **Publish** it to Hugging Face (so you can pull it down to your laptop).

---

**The Concept: Adapters vs. Merged Models**

- **LoRA Adapters (The "Patch"):**
  - What you trained.
  - Size: ~100MB.
  - _Pros:_ Tiny, easy to share.
  - _Cons:_ Can't run on its own. You need to load the Base Model first, then apply this patch.

- **Merged Model (The "Build"):**
  - What happens when you bake the patch into the weights.
  - Size: ~5GB+.
  - _Pros:_ Runs standalone. This is what you deploy to production.

---

**The Code: Exporting Your Model**

Add these cells to the bottom of your Google Colab notebook.

**Step 1: Save Locally (In Colab)**

This saves the adapters to the Colab virtual disk.

```Python
# Save the LoRA adapters only
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

_Check the file browser on the left in Colab. You will see a folder `lora_model` containing `adapter_model.safetensors`._

**Step 2: Push to Hugging Face (The Cloud)**

This is like `git push` for AI.

1. Go to [Hugging face](huggingface.co/settings/tokens).
2. Create a New Token with Write permissions.
3. Copy it.

```Python
# Login to Hugging Face inside Python

from huggingface*hub import login
login("hf*...") # Paste your token here

# Push the LoRA adapters to your account

model.push_to_hub("your-username/sarcastic-llama-lora", token=True)
tokenizer.push_to_hub("your-username/sarcastic-llama-lora", token=True)
```

**Step 3: Save as GGUF (For your Laptop)**
If you want to run this model on your own machine using **Ollama** or **LM Studio**, you need a format called **GGUF**. Unsloth handles this conversion for you!

```Python
# Save to 4-bit GGUF (The standard for laptops)
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
```

_This will generate a file named `model-unsloth.Q4_K_M.gguf`. You can download this file (warning: it will be huge, ~5GB) and run it locally._

---

**Week 3 Day 3 Assignment: "The Deployment Test"**

1. **Push to Hub:** Run the code above to push your LoRA adapters to Hugging Face.

2. **Verify:** Go to your Hugging Face profile page. You should see a new model repository named `sarcastic-llama-lora`.

3. **The "Ah-Ha" Moment:**
   - Because you pushed the adapter, anyone in the world can now use your specific flavor of AI.

   - They just load Llama 3 and apply your adapter ID.

**Next Up:**

Now that your model is saved, we move to **Week 4: Production**.
We will stop playing in Notebooks. We will build a real **API Server** (like OpenAI's) that serves your custom model to the world.
