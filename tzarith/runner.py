"""Model backends. One OpenAI-compatible client covers OpenAI, OpenRouter,
Together, Groq, DeepInfra, and a local vLLM/Ollama server — just point
``base_url`` at the endpoint and set the API-key env var. Uses only stdlib
``urllib`` so the harness has zero required dependencies.

Backends
--------
* ``openai``  : any /v1/chat/completions endpoint (frontier or open-weight).
* ``dummy``   : deterministic wrong-ish answers, for exercising the pipeline at $0.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


class OpenAICompatible:
    def __init__(self, model, base_url="https://api.openai.com/v1",
                 api_key_env="OPENAI_API_KEY", temperature=0.0, max_tokens=4096,
                 timeout=120, max_retries=6, reasoning=None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        # OpenRouter's unified "reasoning" control (dict or None). None -> omit the
        # field entirely (provider default). Lets us toggle thinking on the SAME
        # model for a controlled reasoning-on/off ablation. Reasoning tokens come
        # back in message.reasoning (not content), so the ANSWER line and scoring
        # are unaffected; they still count toward usage.completion_tokens (cost).
        self.reasoning = reasoning
        self.timeout = timeout
        self.max_retries = max_retries
        self.prompt_tokens = 0
        self.completion_tokens = 0
        # Truncation accounting: a max_tokens cap bounds spend, but if it cuts a
        # verbose model off *before* its ANSWER line the reply scores as wrong for
        # the wrong reason. We surface finish_reason so capped items are auditable
        # rather than silently biasing accuracy downward.
        self.truncations = 0
        self.last_finish_reason = None
        # Empty-completion retries: some providers (observed on Gemini thinking
        # mode) intermittently return an empty content field with a non-"length"
        # finish reason -- a provider-side non-response, not a model answer. We
        # retry those like a transient error so they don't score as wrong; count
        # them for transparency.
        self.empty_retries = 0
        if not self.api_key:
            raise RuntimeError(f"environment variable {api_key_env} is not set")

    @property
    def name(self):
        return self.model

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        last_err = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                usage = data.get("usage") or {}
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                choice = data["choices"][0]
                self.last_finish_reason = choice.get("finish_reason")
                content = choice["message"].get("content") or ""
                # Empty content with a non-length finish reason == provider
                # non-response: retry rather than return a spurious blank answer.
                if (not content.strip()
                        and self.last_finish_reason != "length"
                        and attempt < self.max_retries - 1):
                    self.empty_retries += 1
                    time.sleep(min(8, 2 ** attempt))
                    continue
                if self.last_finish_reason == "length":
                    self.truncations += 1
                # Robust to truncation/refusal: missing or null content -> "".
                return content
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:400]
                last_err = f"HTTP {e.code}: {body}"
                if e.code == 429 or e.code >= 500:
                    ra = e.headers.get("Retry-After")
                    wait = (float(ra) if ra and ra.replace(".", "").isdigit()
                            else min(65, 8 * 2 ** attempt))
                    time.sleep(wait)
                    continue
                raise RuntimeError(last_err)  # 4xx (auth, bad model) -> fatal, no retry
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = str(e)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"exhausted {self.max_retries} retries: {last_err}")


class Dummy:
    """A $0 stand-in that answers with naive wall-clock arithmetic — i.e. it makes
    exactly the DST mistake the benchmark is designed to catch. Useful for a smoke
    test of the full pipeline before spending on real APIs."""

    name = "dummy-naive"
    truncations = 0
    last_finish_reason = None

    def __call__(self, prompt: str) -> str:
        import re
        from datetime import datetime, timedelta
        if "Add" in prompt and "minutes of elapsed time" in prompt:
            base = re.search(r"It is (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", prompt)
            mins = re.search(r"Add (\d+) minutes", prompt)
            if base and mins:
                dt = datetime.strptime(base.group(1), "%Y-%m-%d %H:%M") + \
                    timedelta(minutes=int(mins.group(1)))  # naive: ignores DST
                return f"ANSWER: {dt.strftime('%Y-%m-%d %H:%M')}"
        if "exists exactly once" in prompt:
            return "ANSWER: exists"          # naive guess
        return "ANSWER: 0"


def get_model(spec: dict):
    kind = spec.get("backend", "dummy")
    if kind == "dummy":
        return Dummy()
    if kind == "openai":
        return OpenAICompatible(
            model=spec["model"],
            base_url=spec.get("base_url", "https://api.openai.com/v1"),
            api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
            temperature=spec.get("temperature", 0.0),
            max_tokens=spec.get("max_tokens", 1024),
            reasoning=spec.get("reasoning"))
    raise ValueError(f"unknown backend {kind!r}")
