"""Ollama LLM post-processing for raw Whisper transcripts."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from luduan.log import get_logger

log = get_logger(__name__)


_SYSTEM_PROMPT = """\
You are a post-processing assistant for speech-to-text transcriptions.
Your job is to fix grammar, punctuation, and capitalization in the given transcript.
When nearby text from the active app is provided, treat it only as local context
for resolving names, acronyms, casing, and terminology.
Do NOT change the meaning, add new content, or remove information.
Do NOT copy arbitrary phrases from the context unless they clearly disambiguate
what the user said.
Return ONLY the corrected text, no explanations or extra formatting."""


class Postprocessor:
    """Sends a raw transcript to a local Ollama LLM for cleanup.

    If Ollama is unavailable or disabled, the raw transcript is returned
    unchanged so the tool degrades gracefully.

    Args:
        host: Ollama base URL (default: http://localhost:11434).
        model: Ollama model name.
        enabled: If False, passthrough mode (no LLM call).
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        enabled: bool = True,
        timeout: float = 60.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.enabled = enabled
        self.timeout = timeout

    def process(
        self,
        text: str,
        *,
        context_before_cursor: str | None = None,
        app_name: str | None = None,
    ) -> tuple[str, bool]:
        """Clean up *text* with the LLM.

        Returns:
            (cleaned_text, used_llm) — used_llm is False when LLM was
            skipped (disabled or unreachable).
        """
        if not self.enabled or not text.strip():
            return text, False

        try:
            log.debug("Sending to Ollama model=%s", self.model)
            cleaned = self._call_ollama(
                text,
                context_before_cursor=context_before_cursor,
                app_name=app_name,
            )
            return cleaned, True
        except Exception:
            log.exception("Ollama post-processing failed — using raw transcript")
            return text, False

    def is_available(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{self.host}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def ensure_model_available(
        self,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> bool:
        """Ensure the configured Ollama model is available locally.

        Returns True when a pull was required, False when the model was already
        present locally.
        """
        if not self.enabled:
            log.info("Skipping Ollama offline preparation because LLM is disabled")
            return False

        if self.has_model_available():
            log.info("Ollama model already available: %s", self.model)
            return False

        if progress_callback:
            progress_callback(f"Pulling Ollama model {self.model}…")
        log.info("Pulling Ollama model for offline use: %s", self.model)

        with httpx.Client(timeout=None) as client, client.stream(
            "POST",
            f"{self.host}/api/pull",
            json={"name": self.model, "stream": True},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                error = event.get("error")
                if error:
                    raise RuntimeError(error)

                if progress_callback:
                    progress_callback(_format_ollama_progress(self.model, event))

        return True

    def has_model_available(self) -> bool:
        """Return True when the configured Ollama model is already local."""
        if not self.enabled:
            return False
        return self.model in self._list_models()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_ollama(
        self,
        text: str,
        *,
        context_before_cursor: str | None = None,
        app_name: str | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        text,
                        context_before_cursor=context_before_cursor,
                        app_name=app_name,
                    ),
                },
            ],
            "stream": False,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()

        data = resp.json()
        return data["message"]["content"].strip()

    def _list_models(self) -> set[str]:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{self.host}/api/tags")
            resp.raise_for_status()

        data = resp.json()
        return {
            item["name"]
            for item in data.get("models", [])
            if isinstance(item, dict) and "name" in item
        }


def _build_user_prompt(
    text: str,
    *,
    context_before_cursor: str | None = None,
    app_name: str | None = None,
) -> str:
    if not context_before_cursor:
        return f"Please clean up this transcript:\n\n{text}"

    context_label = f" in {app_name}" if app_name else ""
    return (
        "Please clean up this transcript.\n\n"
        f"Transcript:\n{text}\n\n"
        f"Text immediately before the cursor{context_label} "
        "(context hint only, may be partial):\n"
        f"{context_before_cursor}\n\n"
        "Use the context only to improve names, acronyms, punctuation, and casing. "
        "Return only the cleaned transcript to insert at the cursor."
    )


def _format_ollama_progress(model: str, event: dict) -> str:
    status = event.get("status") or f"Preparing {model}…"
    completed = event.get("completed")
    total = event.get("total")
    if isinstance(completed, int) and isinstance(total, int) and total > 0:
        percent = int(completed * 100 / total)
        return f"{status} ({percent}%)"
    return status
