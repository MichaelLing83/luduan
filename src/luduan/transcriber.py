"""Whisper-based speech transcription via mlx-whisper (Apple Silicon optimized)."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable

import numpy as np

from luduan import config as cfg
from luduan.log import get_logger

log = get_logger(__name__)


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

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        *,
        emit_partial: bool = True,
    ) -> str:
        """Transcribe a complete audio array and return the full text.

        For long recordings this also fires ``on_partial`` callbacks as
        Whisper processes each internal segment.
        """
        _ensure_hf_home()
        import mlx_whisper  # lazy import — heavy, load only when needed

        audio_f32 = _ensure_float32(audio, sample_rate)

        with self._lock:
            log.debug("Running mlx-whisper: model=%s language=%s samples=%d",
                      self.model_name, self.language, len(audio_f32))
            result = mlx_whisper.transcribe(
                audio_f32,
                path_or_hf_repo=self.model_name,
                language=self.language,
                verbose=False,
                word_timestamps=False,
                fp16=True,
            )

        text: str = result.get("text", "").strip()

        if emit_partial and self.on_partial and text:
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

    def ensure_model_cached(self) -> bool:
        """Ensure the configured Whisper model is downloaded locally.

        Returns True when a download was required, False when the model was
        already present in the local cache.
        """
        _ensure_hf_home()
        if self.is_model_cached():
            log.info("Whisper model already cached: %s", self.model_name)
            return False

        from huggingface_hub import snapshot_download

        log.info("Downloading Whisper model for offline use: %s", self.model_name)
        snapshot_download(repo_id=self.model_name)
        return True

    def is_model_cached(self) -> bool:
        """Return True when the configured Whisper model already exists locally."""
        _ensure_hf_home()

        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            snapshot_download(
                repo_id=self.model_name,
                local_files_only=True,
            )
            return True
        except LocalEntryNotFoundError:
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_hf_home() -> None:
    hf_home = str(cfg.CONFIG_DIR / "models")
    os.makedirs(hf_home, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_home)


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
