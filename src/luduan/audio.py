"""Microphone audio capture using sounddevice."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    pass


class AudioRecorder:
    """Records audio from the default microphone into an in-memory buffer.

    Usage::

        recorder = AudioRecorder(sample_rate=16000)
        recorder.start()
        # ... user speaks ...
        audio = recorder.stop()  # returns float32 numpy array, shape (N,)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        on_chunk: Callable[[np.ndarray], None] | None = None,
        chunk_seconds: float = 5.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.on_chunk = on_chunk
        self.chunk_size = int(sample_rate * chunk_seconds)

        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._chunk_accum: list[np.ndarray] = []
        self._chunk_accum_len = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the microphone stream and begin recording."""
        self._buffer = []
        self._chunk_accum = []
        self._chunk_accum_len = 0

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop recording and return all captured audio as a 1-D float32 array."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            # Flush any remaining partial chunk
            if self._chunk_accum and self.on_chunk:
                chunk = np.concatenate(self._chunk_accum)
                self.on_chunk(chunk)

            if not self._buffer:
                return np.zeros(0, dtype=np.float32)

            audio = np.concatenate(self._buffer)

        # sounddevice returns shape (N, channels); flatten to 1-D for Whisper
        if audio.ndim > 1:
            audio = audio[:, 0]

        return audio

    @property
    def is_recording(self) -> bool:
        return self._stream is not None and self._stream.active

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        chunk = indata.copy()

        with self._lock:
            self._buffer.append(chunk)

            if self.on_chunk is not None:
                self._chunk_accum.append(chunk)
                self._chunk_accum_len += frames

                if self._chunk_accum_len >= self.chunk_size:
                    full_chunk = np.concatenate(self._chunk_accum)
                    self._chunk_accum = []
                    self._chunk_accum_len = 0
                    # Fire callback outside the lock to avoid deadlocks
                    # (schedule via a thread so the audio callback stays fast)
                    t = threading.Thread(target=self.on_chunk, args=(full_chunk[:, 0],), daemon=True)
                    t.start()
