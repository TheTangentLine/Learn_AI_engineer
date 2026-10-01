"""A tiny local chat model (Hugging Face transformers) with an on-disk generation cache.

Used by lessons that need *some* real LLM but must run without API keys (query rewriting, HyDE...).
It is small and weak: results measured with it are honest about that, and the same code runs
unchanged against a hosted model via ``common.llm``.

    from common.local_llm import LocalChat
    chat = LocalChat()                                   # Qwen2.5-0.5B-Instruct, greedy decoding
    text = chat("You are terse.", "Name three fruits.", max_new_tokens=30)

To route ``common.llm`` calls to it (so lesson code needs no changes):

    from common.fake import fake_llm
    with fake_llm([(r"(?s).*", chat.as_responder())]): ...
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CACHE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "local_llm_cache.json"


class LocalChat:
    def __init__(self, model: str = DEFAULT_MODEL, cache_path: Path | None = CACHE_PATH):
        self.model_name = model
        self._tok = self._model = self._torch = None
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text())
        self.generated = 0  # real generations (cache misses)

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch.float32
            ).eval()

    def count_tokens(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> int:
        """Prompt length in tokens under this model's chat template (tokenizer only: no weights loaded)."""
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        text = self._tok.apply_chat_template(
            msgs, tools=tools or None, add_generation_prompt=True, tokenize=False
        )
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def text_tokens(self, text: str) -> int:
        if self._tok is None:
            self.count_tokens([{"role": "user", "content": ""}])  # loads the tokenizer
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def __call__(self, system: str | None, user: str, max_new_tokens: int = 128) -> str:
        key = hashlib.sha1(
            f"{self.model_name}\x00{system}\x00{user}\x00{max_new_tokens}".encode()
        ).hexdigest()
        if key in self.cache:
            return self.cache[key]
        self._load()
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        ids = self._tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with self._torch.no_grad():
            out = self._model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        text = self._tok.decode(
            out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        self.generated += 1
        self.cache[key] = text
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        return text

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
        max_new_tokens: int = 300,
    ) -> str:
        """Multi-turn generation with the model's chat template (tool schemas included). Returns raw text,
        keeping markers like <tool_call>...</tool_call> so callers can parse tool requests. Cached by input."""
        payload = json.dumps(
            [self.model_name, system, messages, tools, max_new_tokens], sort_keys=True, default=str
        )
        key = hashlib.sha1(payload.encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        self._load()
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        ids = self._tok.apply_chat_template(
            msgs,
            tools=tools or None,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        with self._torch.no_grad():
            out = self._model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        text = self._tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=False)
        text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        self.generated += 1
        self.cache[key] = text
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        return text

    def as_responder(self, max_new_tokens: int = 160):
        """A rule responder for ``fake_llm``: forwards each call's system + user text."""

        def respond(prompt: str, call) -> str:
            user = next(
                (m["content"] for m in reversed(call.messages) if m["role"] == "user"), prompt
            )
            return self(call.system, user if isinstance(user, str) else prompt, max_new_tokens)

        return respond
