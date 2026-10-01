#### 1. The Hierarchy: The "Russian Dolls" of AI

AI $\rightarrow$ ML $\rightarrow$ DL $\rightarrow$ LLMs

**a. Artificial Intelligence (AI)**: The parent wrapper. Any system that mimics smart behavior. An old video game NPC with `if/else` logic is technically AI.

**b.Machine Learning (ML)**: A specific type of AI where we do not write rules. We write a generic algorithm that finds the rules by looking at data.

- **_CS Analogy_**: Traditional programming is writing the function body `def f(x): return x * 2`. ML is defining the input `x` and output `y`, and asking the computer to write the body of `f(x)`.

**c. Deep Learning (DL)**: A subset of ML that uses **Neural Networks** with many layers (hence "Deep").

- **_Why it matters_**: Traditional ML (like Excel regression) hits a performance ceiling. DL keeps getting smarter the more data you feed it.

**d. Generative AI (LLMs)**: A specific architecture of DL (Transformers) designed to generate new data (text/images) rather than just classify existing data.

#### 2. The Neural Network: An Engineering View

To a software engineer, a Neural Network is a **Directed Acyclic Graph (DAG) of Matrix Multiplications.**

**The Components**

1. **Input Layer:** Your raw data converted into numbers (e.g., an image is a grid of pixel values).

2. **Weights (The "State"):** These are the variables. In a standard program, you set variables (const threshold = 5). In AI, the machine sets these variables.

   - Scale: GPT-4 has nearly 1.8 trillion of these variables.

3. **Hidden Layers:** Steps in the pipeline. Each layer takes the data, multiplies it by the Weights (matrix math), applies a filter (activation), and passes it to the next layer.

4. **Activation Function (The "Non-Linearity"):**
   - Without this, the whole network is just one big linear equation ($y = mx + b$).
   - This function (like ReLU) acts like an `if` statement: "If the value is negative, set it to zero." This allows the model to learn complex, curvy patterns, not just straight lines.

#### 3. How It "Learns" (The Training Loop)

This is the cycle that runs billions of times during training. Think of it as **Test-Driven Development (TDD) on steroids.**

**Step A: The Forward Pass (Execution)**
The data flows through the graph. The model guesses.

- **_Input:_** Image of a Cat.
- **_Model Guesses:_** "Dog (80%), Cat (20%)".

**Step B: The Loss Function (The Unit Test)**
We calculate how wrong the model was using a math formula.

- **_Reality:_** Cat (100%).
- **_Prediction:_** Cat (20%).
- **_Error (Loss):_** High.
- **_CS Analogy:_** This is your integration test failing and returning a specific error code.

**Step C: Backpropagation (The "Blame Assignment")**
This is the magic. The system looks at the Error and traces it **backwards** through the graph to find out which Weights contributed most to the mistake.

- "Layer 4, Node 2 was responsible for 10% of the error. Layer 3, Node 5 was responsible for 2%..."
- **_CS Analogy:_** `git blame` on a bug. You find exactly which line of code caused the crash.

**Step D: The Optimizer (The Refactor)**
We nudge the weights slightly in the opposite direction of the error.

- "Decrease the weight of Layer 4, Node 2 by 0.001."
- **_CS Analogy:_** Fixing the bug by changing the variable value slightly.

**Repeat this loop 1 trillion times, and the Loss eventually becomes nearly zero.**

#### 4. Data Structures: Tensors & Embeddings

If you understand JSON and SQL tables, you need to understand the Tensor.

**What is a Tensor?**
It is just a generic N-dimensional array

- **Rank 0 Tensor:** A scalar (e.g., 5).
- **Rank 1 Tensor:** A vector (e.g., [1, 2, 3]).
- **Rank 2 Tensor:** A matrix (rows & columns).
- **Rank 3 Tensor:** A cube of numbers (e.g., an RGB image: Height x Width x 3 Channels).

**Why Tensors?** GPUs are hardware optimized to multiply Tensors incredibly fast. If you try to run a loop in Python, it's slow. If you shove data into a Tensor and send it to the GPU, it happens in parallel.

**What is an Embedding? (Crucial for Week 2)**

Computers can't understand strings like "King" or "Queen". They only understand math. An Embedding is a translation layer that turns a word into a **vector of floating point numbers.**

- "King" $\rightarrow$ [0.9, 0.2, 0.5]
- "Queen" $\rightarrow$ [0.9, 0.2, 0.6]
- "Apple" $\rightarrow$ [0.1, 0.9, 0.2]

Notice that the numbers for King and Queen are _mathematically close_ (high cosine similarity). Apple is far away.**This allows us to do math on meaning**.

- **_Equation:_** King - Man + Woman = Queen.
- If you subtract the vector for "Man" from "King" and add "Woman", the resulting numbers actually look like the vector for "Queen".

#### 5. LLM Specifics: Tokenization & Context

Since you want to work with LLMs (GPT, Llama), you need to know how they read text.

**Tokenization**
LLMs do **not** read words. They read **Tokens**.

- A token is a chunk of characters.
- Common words ("apple") are 1 token.
- Complex words ("supercalifragilistic") might be split into multiple tokens ("super", "cali", ...).
- **_Rule of Thumb:_** 1,000 tokens $\approx$ 750 words.

**Context Window**
This is the "RAM" of the LLM.

- The model has a fixed limit on how much text it can "see" at once (e.g., GPT-4o has 128k tokens).
- If you send a conversation longer than this, the beginning gets "truncated" (deleted). It literally forgets what you said 5 minutes ago because it fell out of the window.

#### 6. Hardware: Why do we need GPUs?

- **CPUs (Intel/AMD):** Designed for serial processing. Great for complex logic, branching (if/else), and OS tasks. They have few cores (e.g., 16 cores).

- **GPUs (NVIDIA):** Designed for parallel processing. They have thousands of tiny, dumb cores.

- **AI Workload:** Multiplying two massive matrices requires millions of tiny, independent multiplications.
  - CPU: Does them one by one (slow).
  - GPU: Does them all at once (fast).

**VRAM (Video RAM)** is your bottleneck.

- To run a model, you must load all its Weights into VRAM.
- If a model is 8GB and you have a 6GB GPU, it will crash (OOM - Out of Memory).

#### Summary Checklist

1. [ ] Do you understand that a model is just a big math function with tunable variables?

2. [ ] Do you understand that "Training" is just minimizing an error score?

3. [ ] Do you understand that "Embeddings" turn text into coordinate points so we can measure distance?

4. [ ] Do you understand that LLMs read "Tokens", not words?
