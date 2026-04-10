"""Whisper-based speech transcription via mlx-whisper (Apple Silicon optimized)."""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np


# mlx-whisper model name mapping (matches whisper model sizes)
_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


class Transcriber:
    """Transcribes audio buffers using mlx-whisper.

    Supports both batch (full audio at once) and chunk-streaming modes.
    In streaming mode, intermediate transcriptions are surfaced via
    ``on_partial`` callback as each chunk is processed.

    Args:
        model: Whisper model size (tiny/base/small/medium/large).
        language: Language hint (e.g. "en"). None for auto-detect.
        on_partial: Called with each intermediate text chunk.
    """

    def __init__(
        self,
        model: str = "base",
        language: str | None = "en",
        on_partial: Callable[[str], None] | None = None,
    ) -> None:
        self.model_name = _MODEL_MAP.get(model, model)
        self.language = language
        self.on_partial = on_partial
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a complete audio array and return the full text.

        For long recordings this also fires ``on_partial`` callbacks as
        Whisper processes each internal segment.
        """
        import mlx_whisper  # lazy import — heavy, load only when needed

        audio_f32 = _ensure_float32(audio, sample_rate)

        with self._lock:
            result = mlx_whisper.transcribe(
                audio_f32,
                path_or_hf_repo=self.model_name,
                language=self.language,
                verbose=False,
                word_timestamps=False,
                fp16=True,
            )

        text: str = result.get("text", "").strip()

        if self.on_partial and text:
            self.on_partial(text)

        return text

    def transcribe_chunks(
        self,
        chunks: list[np.ndarray],
        sample_rate: int = 16000,
    ) -> str:
        """Transcribe a list of audio chunks sequentially.

        Each chunk produces an ``on_partial`` callback.  Returns the
        concatenated transcript.
        """
        parts: list[str] = []
        for chunk in chunks:
            text = self.transcribe(chunk, sample_rate)
            if text:
                parts.append(text)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_float32(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Normalise audio to float32 mono at the expected sample rate."""
    if audio.ndim > 1:
        audio = audio[:, 0]

    audio = audio.astype(np.float32)

    # Resample if needed (mlx-whisper always expects 16 kHz)
    if sample_rate != 16000:
        try:
            import resampy
            audio = resampy.resample(audio, sample_rate, 16000)
        except ImportError:
            # Basic linear-interpolation fallback
            factor = 16000 / sample_rate
            new_len = int(len(audio) * factor)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

    return audio
