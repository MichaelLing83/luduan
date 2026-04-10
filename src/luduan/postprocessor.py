"""Ollama LLM post-processing for raw Whisper transcripts."""

from __future__ import annotations

import httpx


_SYSTEM_PROMPT = """\
You are a post-processing assistant for speech-to-text transcriptions.
Your job is to fix grammar, punctuation, and capitalization in the given transcript.
Do NOT change the meaning, add new content, or remove information.
Return ONLY the corrected text, no explanations or extra formatting."""

_USER_PROMPT_TEMPLATE = "Please clean up this transcript:\n\n{text}"


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

    def process(self, text: str) -> tuple[str, bool]:
        """Clean up *text* with the LLM.

        Returns:
            (cleaned_text, used_llm) — used_llm is False when LLM was
            skipped (disabled or unreachable).
        """
        if not self.enabled or not text.strip():
            return text, False

        try:
            cleaned = self._call_ollama(text)
            return cleaned, True
        except Exception:
            # Silently fall back to raw transcript
            return text, False

    def is_available(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{self.host}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_ollama(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(text=text)},
            ],
            "stream": False,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()

        data = resp.json()
        return data["message"]["content"].strip()
