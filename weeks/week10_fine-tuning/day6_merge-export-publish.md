# Week 10, Day 6: Merge, Quantise, Export and Publish

**Time:** ~4h (about 15 minutes of that is the quantisation evaluation) · **Needs:** CPU only; `safetensors` and `transformers` · **Run it:** `uv run python weeks/week10_fine-tuning/solutions/day6_solution.py`

A fine-tune that exists only as a Python process is not a product. Today the adapters become a **model directory** that the standard library loads, you measure what **quantisation** does to it (the step that makes a model small and fast enough to run locally), and you prepare it for **publishing** and for local runtimes (Ollama): with an honest line between what was run and what was only prepared.

## Learning objectives
- **Merge** adapters into the weights and prove the merged model equals the adapter model.
- **Export** a model directory in the Hugging Face format (config, safetensors, tokenizer, model card) and verify it by loading it back with the library: same logits, same tokens.
- Explain the **quantisation formats** (Q8_0, Q4_0, NF4, GPTQ/AWQ in concept) and **measure** their cost in accuracy on your task.
- Run a **pre-publication audit** (files, model card, no credentials) and describe the paths to a user: Hugging Face Hub, Ollama/GGUF, hosted fine-tuning APIs.

---

## 1. Merging

LoRA's update is a matrix product, so it can be folded into the weight: `W' = W + (α/r)·B·A`. After merging the model is an **ordinary model** (no adapter layers, no extra matmuls, no extra latency). The solution measures it on the real run: the adapter model and the merged model give logits that differ by {{MERGE_GAP}}.

When to **keep** adapters unmerged: to serve many tasks from one base model by swapping a few megabytes; to keep the base weights untouched; to continue training. When to **merge**: to publish a single self-contained model, to convert to other formats (GGUF needs plain weights), to remove the (small) adapter latency.

One precision trap: **merge into the weights you trained against.** If the adapters were trained over a 4-bit base (QLoRA), merging them into a 16-bit copy of the base is a *different* model from the one you evaluated; re-evaluate after merging, or merge into the dequantised base and say so.

## 2. A model directory the library can load

A Hugging Face model is a **directory**:

| file | purpose |
|---|---|
| `config.json` | the architecture: layers, widths, heads, vocabulary, RoPE base, `tie_word_embeddings` |
| `model.safetensors` | the weights: a flat, memory-mappable, **non-executable** format (unlike pickle) |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template` | the tokenizer **and the chat template** (so a user does not have to guess the prompt format) |
| `generation_config.json` | default decoding settings |
| `README.md` | the **model card**: base model, data, results, limits, how to prompt |

`export.save_hf_model` writes exactly that from the from-scratch decoder: `to_hf_state_dict` maps `layers.N.self_attn.q_proj.weight` to `model.layers.N.self_attn.q_proj.weight` (the inverse of the Week 9 loader) and **omits the output matrix when it is tied** to the embedding table (the original checkpoint does the same: it stores the table once). Tests: export then load reproduces the weights **exactly** (tied and untied), and a merged LoRA model exported to a directory is loaded by `AutoModelForCausalLM.from_pretrained` with logits within 1e-4.

{{EXPORT}}

## 3. Quantisation: smaller, faster, a little worse

Weights are stored in 16 or 32 bits; most of that precision is not needed. **Quantisation** rounds them to 8 or 4 bits (with a scale per small block) and the model keeps most of its quality while shrinking 2 to 7 times. The formats you will meet:

| format | bits / weight | idea | where |
|---|---|---|---|
| fp32 / fp16 / bf16 | 32 / 16 / 16 | no quantisation | training; reference |
| **Q8_0** | 8.5 | blocks of 32, int8 and an fp16 scale | GGUF (llama.cpp, Ollama) |
| **Q4_0** | 4.5 | blocks of 32, 4 bits and an fp16 scale; the value of largest magnitude defines the scale | GGUF |
| **Q4_K_M** and friends | about 4.8 | "k-quants": blocks of 256 with quantised scales and mixed precision per tensor | GGUF; the usual default |
| **NF4** | 4.5 | 4-bit quantiles of a normal distribution (Day 3) | QLoRA (bitsandbytes) |
| **GPTQ** | 3 to 4 | rounds a layer's weights **one column at a time, correcting the rest** to minimise the layer's output error on calibration data | GPU inference |
| **AWQ** | 4 | protects the **1% of weight channels** that matter most (by activation magnitude) by scaling before rounding | GPU inference |

`solutions/quant.py` implements Q8_0, Q4_0, NF4 and a uniform 4-bit grid from scratch, with tests (Q4_0's rule that the largest-magnitude value maps to the end of the range; zeros stay zeros; padding to whole blocks). **GPTQ and AWQ are described here, not implemented**: both need calibration data and a per-layer solver.

{{QUANT}}

**Simulation, not deployment.** The quantised models above are fp32 tensors holding *rounded values*, so they show the **accuracy** cost exactly but **not the speed or memory benefit** (that needs a runtime that stores and multiplies the packed formats: llama.cpp, bitsandbytes, vLLM). Treat the size column as the format's storage size, which is what a GGUF file of that type would take.

## 4. Local runtimes and publishing (prepared, not run)

**Ollama / llama.cpp (not run: neither is installed here).** The path: convert the merged Hugging Face directory to **GGUF** with llama.cpp's `convert_hf_to_gguf.py`, optionally quantise with `llama-quantize`, then write a **Modelfile**:

```
{{MODELFILE}}
```

and `ollama create order-extractor -f Modelfile`. `export.ollama_modelfile` generates this text and a test checks the weights path, the ChatML template, the system prompt, the temperature and both stop tokens. **The template and stop tokens matter**: a Modelfile with a different template from training is the Day 1 mismatch again, and a missing `<|im_end|>` stop token produces output that runs on forever.

**Hugging Face Hub (not run: needs a token).** `HfApi().upload_folder(folder_path=..., repo_id="<you>/order-extractor", repo_type="model")` uploads the directory. `export.audit_directory` is the checklist to run first: the files a loader needs exist; the model card has front matter naming the **base model** (licence and lineage matter) and reports results; and no text file contains something that looks like a credential (`hf_...`, `sk-...`, `ghp_...`, `AKIA...`). Also check: the licence of the base model permits your use; the training data may be published; no personal data was in it (Week 8); the card states the **limitations** you measured.

**Hosted fine-tuning APIs (not run: needs a key).** Hosted services fine-tune models you cannot download: you upload a **chat-format JSONL** and pay per **training token** (dataset tokens × epochs), then per inference token at a higher rate than the base model. `export.to_openai_jsonl` writes the layout and `validate_openai_jsonl` checks the rules such a service enforces before you pay for a failed job (valid JSON per line, a messages list, known roles, string content, at least one assistant message, a minimum number of examples). {{HOSTED}} Trade-offs against self-hosting: no GPUs to run and nothing to deploy, but you do not get the weights, the price per token is the provider's, the model may be retired, and not every provider offers every base model.

## 5. Pitfalls
- **Publishing without the chat template**: the user has to guess your prompt format.
- **Pickle files (`.bin`, `.pt`) from strangers** execute code on load; prefer `safetensors`.
- **Quantising the embedding and output matrices** carelessly: they hold a large share of a small model's weights and are more sensitive; many formats keep them at higher precision (the size table here keeps the embedding at fp16).
- **Judging a quantised model by perplexity alone**; measure *your task* (below).
- **A model card with no limitations**, or a licence that does not match the base model's.
- **Forgetting to re-run the evaluation on the merged, exported, quantised artifact** (the thing users get), not on the adapter model you trained.

---

## Daily challenge: your model running locally

**Build** (reference: [`solutions/export.py`](solutions/export.py), [`solutions/quant.py`](solutions/quant.py), [`solutions/day6_solution.py`](solutions/day6_solution.py)):
1. Merge the adapters; verify the merged model equals the adapter model.
2. Export a Hugging Face directory (with a model card) and verify that the library loads it with the same logits and the same greedy output.
3. Quantise to at least two 4-bit formats and Q8_0; report size and extraction accuracy on the hand-written set **and** the synthetic one.
4. A pre-publication audit, a Modelfile, and (if you have the tools) the GGUF conversion and a run in Ollama.

**Acceptance criteria**
- Merged and adapter logits agree to within float noise; the exported directory loads with `transformers` and agrees to 1e-4.
- The audit passes on your directory and *fails* when you plant a fake token or delete the card's front matter.
- The quantisation table reports accuracy **with intervals** and says which differences are within noise.
- Everything you did not run (GGUF conversion, Ollama, the Hub, a hosted job) is labelled as such.

**Stretch**
- If `llama.cpp` and Ollama are available (a `brew install` away), convert to GGUF Q4_K_M, run the Modelfile, and compare the GGUF model's answers with the Hugging Face model's on the 38 emails.
- Implement **GPTQ-style** column-by-column rounding for one layer on calibration activations and compare its output error with round-to-nearest at 4 bits.
- Implement **AWQ-style** channel scaling and measure the effect on the same layer.
- Publish to the Hub and write a model card with a usage snippet; load it back from the Hub in a clean environment.

## Further reading
- The llama.cpp documentation on GGUF and quantisation types; Ollama's Modelfile reference.
- Frantar et al., *GPTQ*; Lin et al., *AWQ*; Dettmers et al., *LLM.int8()*.
- Mitchell et al., *Model Cards for Model Reporting*; the Hugging Face guide to model cards.
