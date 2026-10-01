Welcome to **Week 4: The Engineer Layer (Production).**

You have trained a model. It exists as a file.
Now, how do you let 1,000 users talk to it at the same time?

You **cannot** just wrap the Python script from Week 3 in a Flask app.

- **Why?** If User A sends a request, the GPU is busy. If User B sends a request 1 second later, they get blocked.

- **The Solution:** You need a dedicated **Inference Server** that handles **Batching** (processing 10 requests at once) and **Queueing**.

---

**Week 4, Day 1: The "Local Production" Setup (Ollama)**

For this guide, we will use **Ollama**.

- **Dev (Ollama):** Runs on your laptop (Mac/Windows/Linux). Uses CPU or GPU. Easiest to set up.

- **Prod (vLLM):** Runs on Linux Servers (AWS/GCP). Uses NVIDIA GPUs only. Fastest throughput.

Since you are likely on a local machine, we will use **Ollama** to serve the GGUF model you created yesterday.

**Step 1: Install Ollama**

1. Go to [Ollama](ollama.com) and download it.
2. Install it.
3. Open your terminal and run: `ollama run llama3`

- This pulls the base Llama 3 model and lets you chat. Verify it works.

**Step 2: The "Modelfile" (Docker for AI)**

You saved your sarcastic model as a `.gguf` file yesterday. Now we wrap it.
Think of a `Modelfile` exactly like a `Dockerfile`. It defines the base image and the configuration layers.

1. Create a file named `Modelfile` (no extension) in the same folder as your `.gguf` file.

2. Add this content:

```Dockerfile
# 1. The Base Image (Your custom file)
FROM ./model-unsloth.Q4_K_M.gguf

# 2. The Configuration (Temperature, Context Window)
PARAMETER temperature 0.7
PARAMETER num_ctx 4096

# 3. The System Prompt (Hardcoded behavior)
# We bake the prompt into the server so the frontend doesn't need to send it.
SYSTEM "You are a sarcastic tech support agent. You help users but you judge them."
```

**Step 3: Build and Run**
In your terminal:

```Bash
# 1. Build the model image
ollama create sarcastic-bot -f Modelfile

# 2. Run the server
ollama run sarcastic-bot
```

You can now chat with it in the terminal. **But wait**—we want to talk to it via **Code/API**.

---

**Step 4: The API (The "OpenAI Compatible" Layer)**

This is the most important standard in AI Engineering.
Ollama (and vLLM) exposes an API that **looks exactly like OpenAI's API**.

This means you don't need to learn a new library. You just change the `base_url`.

Create `week4_day1_client.py`:

```Python
from openai import OpenAI

# Point to your local Ollama server (default port 11434)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", # Required but unused
)

response = client.chat.completions.create(
    model="sarcastic-bot", # The name you gave in 'ollama create'
    messages=[
        {"role": "user", "content": "My mouse is broken."}
    ]
)

print(response.choices[0].message.content)
```

**Run it.**
You should get a response like: "Have you tried feeding it cheese?"

---

**Why this is "Engineering"**

1. **Decoupling:** Your Python code doesn't load the model. The Server loads the model. Your Python code just sends JSON.

2. **Scalability:** You can now run 5 instances of the Python script. The Ollama server will queue them up and answer them one by one (or in parallel if your hardware handles it).

3. **Standardization:** If you decide to move to AWS later, you just change `http://localhost:11434` to `http://api.my-aws-server.com`. The code stays the same.

**Week 4 Day 1 Assignment: "The Full Stack"**

1. Keep your **Ollama server** running in one terminal window.
2. Go back to your **Week 1, Day 4** script ("The Self-Correcting Coder").
3. Modify it to use `model="sarcastic-bot"` and `base_url="http://localhost:11434/v1"`.

Run it. You now have a Self-Correcting Coder running **entirely offline** on your own "Neural Engine."
